#include "../include/autotagger.h"
#include "../include/image_preprocesser.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <json.hpp>
#include <onnxruntime_cxx_api.h>
#include <thread>

using json = nlohmann::json;

std::vector<uint8_t> boolVecToBits(const std::vector<uint8_t>& boolVec) {
    if (boolVec.empty()) return {};
    size_t byteSize = (boolVec.size() + 7) / 8;
    std::vector<uint8_t> bitVec(byteSize, 0);
    for (size_t i = 0; i < boolVec.size(); ++i) {
        if (boolVec[i]) {
            bitVec[i / 8] |= (1 << (i % 8));
        }
    }
    return bitVec;
}

class DefaltAutoTagger : public AutoTagger {
public:
    DefaltAutoTagger(const std::filesystem::path& modelPath = "./model/defalt.onnx");
    ~DefaltAutoTagger() override;

    ImageTagResult analyzeImage(const std::filesystem::path& imagePath) override;

    std::vector<float> preprocess(const std::filesystem::path& imagePath) override { return preprocessImage(imagePath); }
    PredictResult predict(const std::vector<float>& inputTensorVec) override;
    ImageTagResult postprocess(PredictResult& predictResult) override;

    std::vector<std::pair<std::string, bool>> getTagSet() override;
    std::string getModelName() override;

    bool gpuAvailable() override { return isGpuAvailable; }
    std::string getLog() override {
        std::string log = logStream.str();
        logStream.str("");
        return log;
    }

private:
    std::pair<std::vector<int>, ModelRestrictType> getTagIndexesAndRestrictType(const std::vector<float>& outputTensor);
    bool loadModel();
    std::filesystem::path modelPath;

    // ONNX Runtime
    Ort::Env ortEnv;
    Ort::SessionOptions sessionOptions;
    Ort::Session* ortSession = nullptr;

    // logging and status
    std::ostringstream logStream;
    bool isGpuAvailable = false;

    // Model parameters
    std::string modelName = "deepdanbooru-v3-20211112-sgd-e28-ONNX";
    std::string inputName = "input_1";
    std::vector<int64_t> inputShape = {1, 512, 512, 3};
    std::string tagOutputName = "activation_172";
    size_t tagOutputDim = 9176;
    std::string hashOutputName = "hash_output";
    size_t hashOutputDim = 512;
    int outputCount = 9176;
    int tagEndIndex = 9175;
    int generaltagStartIndex = 0;
    int characterTagStartIndex = 6891;
    int systemTagStartIndex = 9173;
    float tagThreshold = 0.5f;
};
DefaltAutoTagger::DefaltAutoTagger(const std::filesystem::path& modelPath)
    : ortEnv(ORT_LOGGING_LEVEL_WARNING, "DefaltAutoTagger"), sessionOptions(), modelPath(modelPath) {
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions.SetIntraOpNumThreads(std::max(1u, std::thread::hardware_concurrency()));
    try {
        auto providers = Ort::GetAvailableProviders();
        bool dmlPresent = std::find(providers.begin(), providers.end(), "DmlExecutionProvider") != providers.end();
        if (dmlPresent) {
            sessionOptions.AppendExecutionProvider("DML");
            isGpuAvailable = true;
            logStream << "ONNXRuntime: DML Execution Provider registered." << std::endl;
        } else {
            logStream << "ONNXRuntime: DML not available, will use CPU." << std::endl;
        }
    } catch (const Ort::Exception& e) {
        logStream << "ONNXRuntime: DML not available, fallback to CPU. Reason: " << e.what() << std::endl;
    }
}
DefaltAutoTagger::~DefaltAutoTagger() {
    delete ortSession;
}
bool DefaltAutoTagger::loadModel() {
    try {
        ortSession = new Ort::Session(ortEnv, modelPath.wstring().c_str(), sessionOptions);
    } catch (const Ort::Exception& e) {
        logStream << "ONNXRuntime: Failed to create session. Reason: " << e.what() << std::endl;
        return false;
    }
    return true;
}
std::string DefaltAutoTagger::getModelName() {
    return modelName;
}
std::vector<std::pair<std::string, bool>> DefaltAutoTagger::getTagSet() {
    std::filesystem::path jsonPath = modelPath;
    jsonPath.replace_extension(".json");
    std::ifstream in(jsonPath);
    if (!in) {
        logStream << "Failed to open tag set json file: " << jsonPath << std::endl;
        return {};
    };
    json j;
    in >> j;

    // validate model name in json to check if tag set matches model
    if (!j.contains("name") || !j["name"].is_string() || j["name"].get<std::string>() != modelName) {
        logStream << "Tag set json model name does not match. Expected: " << modelName
                  << ", Found: " << (j.contains("name") && j["name"].is_string() ? j["name"].get<std::string>() : "N/A")
                  << std::endl;
        return {};
    }

    std::vector<std::pair<std::string, bool>> result;
    result.reserve(systemTagStartIndex);
    if (j.contains("tags") && j["tags"].is_array()) {
        for (size_t i = 0; i < systemTagStartIndex; ++i) {
            result.emplace_back(j["tags"][i].get<std::string>(), i >= characterTagStartIndex);
        }
    }
    return result;
}
PredictResult DefaltAutoTagger::predict(const std::vector<float>& inputTensorVec) {
    if (ortSession == nullptr) {
        if (!loadModel()) {
            logStream << "ONNXRuntime: Model not loaded and failed to load." << std::endl;
            return PredictResult{};
        }
    }

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); // maybe not the best choice, but works

    const char* inputNames[] = {inputName.c_str()};
    const char* outputNames[] = {tagOutputName.c_str(), hashOutputName.c_str()};

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputTensorVec.data()), inputTensorVec.size(), inputShape.data(), inputShape.size());

    std::vector<Ort::Value> outputTensors =
        ortSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 2);

    PredictResult result;
    float* tagProbData = outputTensors[0].GetTensorMutableData<float>();
    uint8_t* featureHashData = outputTensors[1].GetTensorMutableData<uint8_t>();
    result.tagProbabilities = std::vector<float>(tagProbData, tagProbData + tagOutputDim);
    result.featureHash = std::vector<uint8_t>(featureHashData, featureHashData + hashOutputDim);

    return result;
}

std::pair<std::vector<int>, ModelRestrictType>
DefaltAutoTagger::getTagIndexesAndRestrictType(const std::vector<float>& outputTensor) {
    // get tag indexes above threshold
    std::vector<int> tagIndexes;
    tagIndexes.reserve(30);
    for (int i = 0; i < systemTagStartIndex; i++) {
        if (outputTensor[i] >= tagThreshold) {
            tagIndexes.push_back(i);
        }
    }
    // last tags in tagSet are system tags, there are 3 system tags: rating:safe, rating:questionable, rating:explicit
    // find the max probability system tag for restrict type
    int maxIdx = systemTagStartIndex;
    float maxProb = 0.0f;
    for (int i = systemTagStartIndex; i < outputCount; i++) {
        if (outputTensor[i] > maxProb) {
            maxProb = outputTensor[i];
            maxIdx = i;
        }
    }
    ModelRestrictType restrictType = ModelRestrictType::Unknown;
    int offset = maxIdx - systemTagStartIndex;
    switch (offset) {
    case 0:
        restrictType = ModelRestrictType::General;
        break;
    case 1:
        restrictType = ModelRestrictType::Questionable;
        break;
    case 2:
        restrictType = ModelRestrictType::Explicit;
        break;
    default:
        restrictType = ModelRestrictType::Unknown;
        break;
    }

    return {tagIndexes, restrictType};
}
ImageTagResult DefaltAutoTagger::postprocess(PredictResult& predictResult) {
    auto [tagIndexes, restrictType] = getTagIndexesAndRestrictType(predictResult.tagProbabilities);
    ImageTagResult result;
    result.tagIndexes = std::move(tagIndexes);
    result.restrictType = restrictType;
    result.tagProbabilities.reserve(tagIndexes.size());
    result.featureHash = boolVecToBits(predictResult.featureHash);
    for (const auto& val : result.tagIndexes) {
        result.tagProbabilities.push_back(predictResult.tagProbabilities[val]);
    }
    return result;
}
ImageTagResult DefaltAutoTagger::analyzeImage(const std::filesystem::path& imagePath) {
    std::vector<float> inputTensor = preprocessImage(imagePath);
    PredictResult predictionOutput = predict(inputTensor);
    ImageTagResult result = postprocess(predictionOutput);
    return result;
}
#ifndef STATIC_TEST_BUILD
extern "C" {
AUTOTAGGER AutoTagger* createAutoTagger() {
    return new DefaltAutoTagger();
}
AUTOTAGGER void destroyAutoTagger(AutoTagger* ptr) {
    delete ptr;
}
}
#endif
