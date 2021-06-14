#include <filesystem>
#include <cstdlib>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::printf("Usage: %s path/to/image path/to/output\n", argv[0]);
        return 1;
    }
    std::filesystem::path path{argv[1]};

    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for(auto & p : std::filesystem::directory_iterator(path)) {
            std::printf("%s\n", p.path().c_str());
        }
        // TODO: filter only images.
        // And Write to a structure to store in disk.
    } else {
        std::printf("Path [%s] may not exist or a directory\n", argv[1]);
        return 1;
    }
    return 0;
}