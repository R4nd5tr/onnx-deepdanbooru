#pragma once
#include <filesystem>
#include <string>
#include <vector>

enum class ModelRestrictType { Unknown, General, Sensitive, Questionable, Explicit };

struct ImageTagResult {
    std::vector<std::string> tags;
    ModelRestrictType restrictType;
    std::vector<float> featureVector;
};
struct ImageTagIndexResult {
    std::vector<int> tagIndexes;
    ModelRestrictType restrictType;
    std::vector<float> featureVector;
};

class AutoTagger {
public:
    virtual ~AutoTagger() = default;
    virtual std::vector<std::pair<std::string, bool>> getTagSet(); // pair<tag, is_character_tag> tags are in index order
    virtual std::string getModelHashID();
    virtual std::vector<std::string> getImageTags(const std::filesystem::path& imagePath);
    virtual std::vector<int> getImageTagIndexs(const std::filesystem::path& imagePath);
    virtual std::vector<float> getImageFeatureVector(const std::filesystem::path& imagePath);
    virtual ModelRestrictType getImageRestrictType(const std::filesystem::path& imagePath);
    virtual ImageTagResult analyzeImage(const std::filesystem::path& imagePath);
    virtual ImageTagIndexResult analyzeImageWithIndex(const std::filesystem::path& imagePath);
};