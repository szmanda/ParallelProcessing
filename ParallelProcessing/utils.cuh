#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include "instance.cuh"

class Utils
{
public:
    static std::string projectDirectory;

public:
    static std::vector<Instance> LoadInstances();
    static void SaveToFile(const std::string& filepath, const std::string& data);

private:
    static std::string Trim(const std::string& str);
};

#endif // UTILS_H