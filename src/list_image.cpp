#include <cstdlib>
#include <cstdio>
#include <filesystem>

#include "sfm_data.hpp"
#include "sfm_data_io.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::printf("Usage: %s path/to/image path/to/output\n", argv[0]);
        return 1;
    }
    std::filesystem::path path{argv[1]};
    SfMData sfm_data;
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for(auto & p : std::filesystem::directory_iterator(path)) {
            //std::printf("%s\n", p.path().c_str());
            View view;
            view.image_path = p.path();
            sfm_data.views[sfm_data.views.size()] = view;
        }
    } else {
        std::printf("Path [%s] may not exist or a directory\n", argv[1]);
        return 1;
    }

    bool save_ret = Save(sfm_data, argv[2]);

    return save_ret ? 0 : 1;
}