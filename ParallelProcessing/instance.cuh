#ifndef INSTANCE_H
#define INSTANCE_H

#include <string>
#include <vector>

class Instance
{
public:
    enum class ErrorType
    {
        NONE,
        NEGATIVE_RANDOM,
        NEGATIVE_REPEAT,
        POSITIVE_RANDOM,
        POSITIVE_WRONG_ENDING
    };

    std::string filepath;
    ErrorType errorType = ErrorType::NONE;
    int numErrors = 0;
    int n = 0; // dna sequence length
    int s = 0; // number of oligonucleotides
    int l = 0; // oligonucleotide length
    std::string name = "";
    std::vector<std::vector<char>> oligs;

    Instance(const std::string& filepath);
    Instance();

    void ExtractInstanceInfo();
    std::string toString();

private:
    std::string GetFileName(const std::string& path);
};

#endif // INSTANCE_H