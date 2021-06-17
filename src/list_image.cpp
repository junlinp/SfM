#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <filesystem>

#include "sfm_data.hpp"
#include "sfm_data_io.hpp"

#include "internal/function_programming.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::printf("Usage: %s path/to/image path/to/output\n", argv[0]);
        return 1;
    }
    std::filesystem::path path{argv[1]};
    SfMData sfm_data;
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        std::vector<std::string> vec_path;
        for(auto & p : std::filesystem::directory_iterator(path)) {
            vec_path.push_back(p.path());
        }
        auto filter = [](const std::string& s) {
            if(s.rfind(".jpg") != std::string::npos) {
                return true;
            }
            if(s.rfind(".png") != std::string::npos) {
                return true;
            }
            if(s.rfind(".jpeg") != std::string::npos) {
                return true;
            }
            if(s.rfind(".tif") != std::string::npos) {
                return true;
            }
            return false;
        };
        // Filter 
        // should use function programming 
        std::vector<std::string> result;
        std::copy_if(vec_path.begin(), vec_path.end(), std::back_inserter(result), filter);

        for(std::string s : result) {
            View view;
            view.image_path = s;
            sfm_data.views[sfm_data.views.size()] = view;
        }
        std::cout << "List " << result.size() << std::endl;
    } else {
        std::printf("Path [%s] may not exist or a directory\n", argv[1]);
        return 1;
    }

    bool save_ret = Save(sfm_data, argv[2]);

    return save_ret ? 0 : 1;
}