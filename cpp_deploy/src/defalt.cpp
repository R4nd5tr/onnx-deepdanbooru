#include "../include/autotagger_api.h"
#include "../include/image_preprocesser.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <json.hpp>
#include <onnxruntime_cxx_api.h>
#include <thread>

using json = nlohmann::json;

class DefaltAutoTagger : public AutoTagger { // TODO: use multi-threaded preprocessing
public:
    DefaltAutoTagger(const std::filesystem::path& modelPath = "./model/defalt.onnx");
    ~DefaltAutoTagger() override;

    std::vector<std::pair<std::string, bool>> getTagSet() override;
    std::string getModelName() override;
    ImageTagResult analyzeImage(const std::filesystem::path& imagePath) override;

    bool gpuAvailable() override { return gpuIsAvailable; }
    std::string getLog() override { return logStream.str(); }

private:
    std::vector<float> predict(const std::vector<float>& inputTensorVec);
    std::pair<std::vector<int>, ModelRestrictType> getTagIndexesAndRestrictType(const std::vector<float>& outputTensor);
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
    std::vector<int64_t> outputShape = {1, 9176};
    int tagCount = 9176;
    int tagEndIndex = 9175;
    int generaltagStartIndex = 0;
    int characterTagStartIndex = 6891;
    int systemTagStartIndex = 9173;
    float tagThreshold = 0.5f;
};
DefaltAutoTagger::DefaltAutoTagger(const std::filesystem::path& modelPath)
    : ortEnv(ORT_LOGGING_LEVEL_WARNING, "DefaltAutoTagger"), sessionOptions(), modelPath(modelPath) {
    sessionOptions.SetIntraOpNumThreads(std::thread::hardware_concurrency());
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0; // 0表示GPU id
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        sessionOptions.SetIntraOpNumThreads(1);
        gpuIsAvailable = true;
        logStream << "ONNXRuntime: CUDA Execution Provider registered." << std::endl;
    } catch (const Ort::Exception& e) {
        logStream << "ONNXRuntime: CUDA not available, fallback to CPU. Reason: " << e.what() << std::endl;
    }
    ortSession = new Ort::Session(ortEnv, modelPath.wstring().c_str(), sessionOptions);
}
DefaltAutoTagger::~DefaltAutoTagger() {
    delete ortSession;
}
std::string DefaltAutoTagger::getModelName() {
    return modelName;
}
std::vector<std::pair<std::string, bool>> DefaltAutoTagger::getTagSet() {
    std::filesystem::path json_path = modelPath.replace_extension(".json");
    std::ifstream in(json_path);
    if (!in) throw std::runtime_error("Failed to open file");
    json j;
    in >> j;

    // validate model name in json to check if tag set matches model
    if (!j.contains("name") || !j["name"].is_string() || j["name"].get<std::string>() != modelName) {
        throw std::runtime_error("Invalid model name in json");
    }

    std::vector<std::pair<std::string, bool>> result;
    if (j.contains("tags") && j["tags"].is_array()) {
        for (size_t i = 0; i < systemTagStartIndex; ++i) {
            std::string tag_str = j["tags"][i].get<std::string>();
            if (i >= characterTagStartIndex) {
                result.emplace_back(tag_str, true); // character tag
            } else {
                result.emplace_back(tag_str, false); // general tag
            }
        }
    }
    return result;
}
std::vector<float> DefaltAutoTagger::predict(const std::vector<float>& inputTensorVec) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Create the input Ort::Value (rename to avoid shadowing the parameter)
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(inputTensorVec.data()), inputTensorVec.size(), inputShape.data(), inputShape.size());

    // Prepare names
    const char* input_names[] = {inputName.c_str()};
    const char* output_names[] = {outputName.c_str()};

    // Run the model
    std::vector<Ort::Value> output_tensors =
        ortSession->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Extract output data
    Ort::Value& out_value = output_tensors.front();
    float* out_data = out_value.GetTensorMutableData<float>();

    // compute output size from outputShape
    size_t out_size = 1;
    for (auto d : outputShape)
        out_size *= static_cast<size_t>(d);

    return std::vector<float>(out_data, out_data + out_size);
}

std::pair<std::vector<int>, ModelRestrictType>
DefaltAutoTagger::getTagIndexesAndRestrictType(const std::vector<float>& outputTensor) {
    // get tag indexes above threshold
    std::vector<int> tagIndexes;
    for (int i = 0; i < systemTagStartIndex; ++i) {
        if (outputTensor[i] >= tagThreshold) {
            tagIndexes.push_back(i);
        }
    }
    // last tags in tagSet are system tags, there are 3 system tags: rating:safe, rating:questionable, rating:explicit
    // find the max probability system tag for restrict type
    int maxIdx = systemTagStartIndex;
    float maxProb = outputTensor[systemTagStartIndex];
    for (int i = systemTagStartIndex + 1; i < tagCount; ++i) {
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
ImageTagResult DefaltAutoTagger::analyzeImage(const std::filesystem::path& imagePath) {
    std::vector<float> inputTensor = preprocessImage(imagePath);
    std::vector<float> outputTensor = predict(inputTensor);
    auto [tagIndexes, restrictType] = getTagIndexesAndRestrictType(outputTensor);
    ImageTagResult result;
    result.tagIndexes = tagIndexes;
    result.restrictType = restrictType;
    result.featureVector = std::move(outputTensor);

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
