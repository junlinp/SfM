#include "sfm_data_io.hpp"

#include <fstream>

#include "cereal/archives/binary.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/string.hpp"

template<class Archive>
void serialize(Archive& ar, View& view) {
    ar(view.image_path);
}

template<class Archive>
void serialize(Archive& ar, SfMData& sfm_data) {
    ar(sfm_data.views);
}

bool Save(const SfMData& sfm_data, const std::string path) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);

    if (!ofs.is_open()) {
      // TODO: Error Warning
      return false;
    }
    cereal::BinaryOutputArchive archive(ofs);
    archive(sfm_data);
    return true;
}

bool Load(SfMData& sfm_data, const std::string path) {
    std::ifstream ifs(path, std::ios::binary);

    if (!ifs.is_open()) {
        return false;
    }

    cereal::BinaryInputArchive archive(ifs);
    archive(sfm_data);
    return true;
}