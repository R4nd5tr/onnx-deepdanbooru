#include "../include/autotagger_api.h"
#include "../include/image_preprocesser.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <json.hpp>
#include <onnxruntime_cxx_api.h>
#include <thread>

using json = nlohmann::json;

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

    bool gpuAvailable() override { return gpuIsAvailable; }
    std::string getLog() override {
        std::string log = logStream.str();
        logStream.str("");
        return log;
    }

private:
    std::pair<std::vector<int>, ModelRestrictType> getTagIndexesAndRestrictType(const std::vector<float>& outputTensor);
    void loadModel();
    std::string modelName = "deepdanbooru-v3-20211112-sgd-e28-ONNX";
    std::filesystem::path modelPath;
    Ort::Env ortEnv;
    Ort::SessionOptions sessionOptions;
    Ort::Session* ortSession = nullptr;
    std::ostringstream logStream;
    bool gpuIsAvailable = false;

    // Model parameters
    std::string inputName = "input_1";
    std::vector<int64_t> inputShape = {1, 512, 512, 3};
    std::string outputName = "activation_172";
    std::vector<int64_t> tagOutputShape = {1, 9176};
    std::vector<int64_t> outputShapeFeatureVec = {1, 4096};
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
            gpuIsAvailable = true;
            logStream << "ONNXRuntime: DML Execution Provider registered." << std::endl;
        } else {
            logStream << "ONNXRuntime: DML not available in GetAvailableProviders(), will use CPU." << std::endl;
        }
    } catch (const Ort::Exception& e) {
        logStream << "ONNXRuntime: DML not available, fallback to CPU. Reason: " << e.what() << std::endl;
    }
}
DefaltAutoTagger::~DefaltAutoTagger() {
    delete ortSession;
}
void DefaltAutoTagger::loadModel() {
    try {
        ortSession = new Ort::Session(ortEnv, modelPath.wstring().c_str(), sessionOptions);
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNXRuntime: Failed to create session. Reason: " << e.what() << std::endl;
    }
}
std::string DefaltAutoTagger::getModelName() {
    return modelName;
}
std::vector<std::pair<std::string, bool>> DefaltAutoTagger::getTagSet() {
    std::filesystem::path jsonPath = modelPath;
    jsonPath.replace_extension(".json");
    std::ifstream in(jsonPath);
    if (!in) throw std::runtime_error("Failed to open file");
    json j;
    in >> j;

    // validate model name in json to check if tag set matches model
    if (!j.contains("name") || !j["name"].is_string() || j["name"].get<std::string>() != modelName) {
        throw std::runtime_error("Invalid model name in json");
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
        loadModel();
        if (ortSession == nullptr) {
            throw std::runtime_error("ONNXRuntime: Session is not initialized.");
        }
    }
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Create the input Ort::Value (rename to avoid shadowing the parameter)
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(inputTensorVec.data()), inputTensorVec.size(), inputShape.data(), inputShape.size());

    // Prepare names
    const char* inputNames[] = {inputName.c_str()};
    const char* outputNames[] = {outputName.c_str(), "feature_vec_output"};

    // Run the model
    std::vector<Ort::Value> outputTensors =
        ortSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 2);

    // Extract output data
    float* tagProbData = outputTensors[0].GetTensorMutableData<float>();
    float* featureVecData = outputTensors[1].GetTensorMutableData<float>();

    // compute output size from tagOutputShape
    size_t tagProbOutSize = 1;
    for (auto d : tagOutputShape)
        tagProbOutSize *= static_cast<size_t>(d);

    size_t featureVecOutSize = 1;
    for (auto d : outputShapeFeatureVec)
        featureVecOutSize *= static_cast<size_t>(d);

    PredictResult result;
    result.tagProbabilities = std::vector<float>(tagProbData, tagProbData + tagProbOutSize);
    result.featureVector = std::vector<float>(featureVecData, featureVecData + featureVecOutSize);

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
    result.featureVector = std::move(predictResult.featureVector);
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
AUTOTAGGER_API AutoTagger* createAutoTagger() {
    return new DefaltAutoTagger();
}
AUTOTAGGER_API void destroyAutoTagger(AutoTagger* ptr) {
    delete ptr;
}
}
#endif
