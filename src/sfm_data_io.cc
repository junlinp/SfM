#include "sfm_data_io.hpp"

#include <fstream>

#include "cereal/archives/binary.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/utility.hpp"
#include "cereal/types/vector.hpp"

template<class Archive>
void serialize(Archive& ar, View& view) {
    ar(view.image_path);
}
template<class Archive>
void serialize(Archive& ar, KeyPoint& key_point) {
    ar(cereal::make_nvp("x", key_point.x));
    ar(cereal::make_nvp("y", key_point.y));
}

template<class Archive>
void serialize(Archive& ar, Descriptors& descriptors){
    ar(descriptors.getRaw_Data());
}
template<class Archive>
void serialize(Archive& ar, SfMData& sfm_data) {
    ar(sfm_data.views);
    ar(sfm_data.key_points);
    ar(sfm_data.descriptors);
    ar(sfm_data.matches);
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