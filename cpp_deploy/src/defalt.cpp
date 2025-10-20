#include "../include/autotagger_api.h"
#include "../include/image_preprocesser.h"

class DefaltAutoTagger : public AutoTagger {
public:
    DefaltAutoTagger(const std::filesystem::path& modelPath);
    ~DefaltAutoTagger() override;

    std::vector<std::pair<std::string, bool>> getTagSet() override;
    std::string getModelHashID() override;
    std::vector<std::string> getImageTags(const std::filesystem::path& imagePath) override;
    std::vector<int> getImageTagIndexs(const std::filesystem::path& imagePath) override;
    std::vector<float> getImageFeatureVector(const std::filesystem::path& imagePath) override;
    ModelRestrictType getImageRestrictType(const std::filesystem::path& imagePath) override;
    ImageTagResult analyzeImage(const std::filesystem::path& imagePath) override;
    ImageTagIndexResult analyzeImageWithIndex(const std::filesystem::path& imagePath) override;

private:
    std::filesystem::path modelPath_;
    // Add ONNX Runtime session and other members here
};