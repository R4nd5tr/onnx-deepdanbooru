#pragma once
#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#ifdef AUTOTAGGER_EXPORTS
#define AUTOTAGGER_API __declspec(dllexport)
#else
#define AUTOTAGGER_API __declspec(dllimport)
#endif
#else
#define AUTOTAGGER_API
#endif

enum class ModelRestrictType { Unknown, General, Sensitive, Questionable, Explicit };

struct ImageTagResult {
    std::vector<int> tagIndexes;
    ModelRestrictType restrictType;
    std::vector<float> featureVector;
};

class AutoTagger {
public:
    AutoTagger() = default;
    virtual ~AutoTagger() = default;
    virtual std::vector<std::pair<std::string, bool>> getTagSet() = 0; // pair<tag, is_character_tag> tags are in index order
    virtual std::string getModelName() = 0;
    virtual ImageTagResult analyzeImage(const std::filesystem::path& imagePath) = 0;

    virtual bool gpuAvailable() = 0;
    virtual std::string getLog() = 0;
};

extern "C" {
AUTOTAGGER_API AutoTagger* createAutoTagger();
AUTOTAGGER_API void destroyAutoTagger(AutoTagger* ptr);
}
