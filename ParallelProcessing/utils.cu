#include "Utils.cuh"
#include <fstream>
#include <iostream>
#include <filesystem>

std::string Utils::projectDirectory = "";

std::vector<Instance> Utils::LoadInstances()
{
    std::vector<Instance> instances;
    std::string path = projectDirectory + "/instances";
    std::cout << "Loading instances from " << path << std::endl;

    /*for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        Instance instance(entry.path());
        int counter = 0;
        std::ifstream file(entry.path());
        std::string line;

        while (std::getline(file, line))
        {
            line = Trim(line);
            if (line.length() == instance.l)
            {
                std::vector<char> olig(line.begin(), line.end());
                instance.oligs.push_back(olig);
            }
        }

        instances.push_back(instance);
    }*/

    return instances;
}

void Utils::SaveToFile(const std::string& filepath, const std::string& data)
{
    std::string fullPath = projectDirectory + "/" + filepath;

    try
    {
        std::ofstream file(fullPath, std::ios_base::app);
        file << data << std::endl;
        file.close();
        std::cout << "SAVED: result to the file" << std::endl;
    }
    catch (const std::exception&)
    {
        std::cout << "WARNING: Saving to file failed!" << std::endl;
    }
}

std::string Utils::Trim(const std::string& str)
{
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}