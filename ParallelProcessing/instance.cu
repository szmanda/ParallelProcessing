#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include "instance.cuh"

Instance::Instance(const std::string& filepath) {
    this->filepath = filepath;
    ExtractInstanceInfo();
}

Instance::Instance() {}

void Instance::ExtractInstanceInfo() {
    name = GetFileName(filepath);
    l = 10;

    size_t dotPos = name.find('.');
    size_t plusPos = name.find('+');
    size_t minusPos = name.find('-');
    int numErrorsOffset = 0;

    if (plusPos != std::string::npos)
        numErrorsOffset = plusPos;
    else if (minusPos != std::string::npos)
        numErrorsOffset = minusPos;
    else
        throw std::runtime_error("Unexpected filename");

    s = std::stoi(name.substr(dotPos + 1, numErrorsOffset - dotPos - 1));
    n = s + l - 1;
    numErrors = std::stoi(name.substr(numErrorsOffset + 1, name.length() - numErrorsOffset - 1));

    if (minusPos != std::string::npos && numErrors >= 40)
        errorType = ErrorType::NEGATIVE_RANDOM;
    else if (minusPos != std::string::npos)
        errorType = ErrorType::NEGATIVE_REPEAT;
    else if (plusPos != std::string::npos && numErrors >= 80)
        errorType = ErrorType::POSITIVE_RANDOM;
    else if (plusPos != std::string::npos)
        errorType = ErrorType::POSITIVE_WRONG_ENDING;
}

std::string Instance::toString() {
    std::string oligonucleotideText = "";
    if (!oligs.empty())
        oligonucleotideText = std::string(oligs[0].begin(), oligs[0].end());

    return "Instance " + name + ": " + std::to_string(n) + " " + std::to_string(s) + " " + std::to_string(l) + " : " + std::to_string(oligs.size()) + " olinonucleotides, first: " + oligonucleotideText;
}

std::string Instance::GetFileName(const std::string& path)
{
    size_t slashPos = path.find_last_of("/\\");
    return (slashPos != std::string::npos) ? path.substr(slashPos + 1) : path;
}
