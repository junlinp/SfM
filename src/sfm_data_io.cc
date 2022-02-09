#include "sfm_data_io.hpp"

#include <fstream>

#include "cereal/archives/binary.hpp"
#include "cereal/archives/json.hpp"

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

template<typename Archive>
void serialize(Archive& ar, Match& match) {
    // serialize the Eigen::Vector2d fails
    // because cereal can't find the serialize function.
    ar(match.lhs_observation.x());
    ar(match.lhs_observation.y());
    ar(match.rhs_observation.x());
    ar(match.rhs_observation.y());
    ar(match.lhs_idx);
    ar(match.rhs_idx);
}
template<class Archive>
void serialize(Archive& ar, SfMData& sfm_data) {
    ar(cereal::make_nvp("Image_Width", sfm_data.image_width));
    ar(cereal::make_nvp("Image_Height", sfm_data.image_height));
    ar(cereal::make_nvp("Views", sfm_data.views));
    ar(cereal::make_nvp("KeyPoint", sfm_data.key_points));
    ar(cereal::make_nvp("Descriptors", sfm_data.descriptors));
    ar(cereal::make_nvp("Matches", sfm_data.matches));
}
template<typename Archive, typename T>
bool SaveCereal(T&& data, const std::string path) {

    size_t index =  path.find_last_of(".");
    std::string extersion = path.substr(index + 1);
    bool is_binary = (extersion == "bin");
    std::ofstream ofs(path, (is_binary ? (std::ios::trunc | std::ios::binary) : std::ios::trunc));
    if (!ofs.is_open()) {
      // TODO: Error Warning
      std::printf("Path %s can't open.\n", path.c_str());
      return false;
    }
    Archive archive(ofs);
    archive(data);
    return true;
}


bool Save(const SfMData& sfm_data, const std::string path) {
    //cereal::BinaryOutputArchive archive(ofs);
    size_t index =  path.find_last_of(".");
    std::string extersion = path.substr(index + 1);

    if (extersion == "bin") {
        return SaveCereal<cereal::BinaryOutputArchive>(sfm_data, path);
    } else if (extersion == "json") {
        return SaveCereal<cereal::JSONOutputArchive>(sfm_data, path);
    }
    return false;
}


template<typename Archive, typename T>
bool LoadCereal(T&& data, const std::string path) {
    size_t index =  path.find_last_of(".");
    std::string extersion = path.substr(index + 1);
    bool is_binary = (extersion == "bin");
    std::ifstream ifs(path, (is_binary ? (std::ios::in | std::ios::binary) : std::ios::in));
    
    if (!ifs.is_open()) {
        std::printf("Path %s can't open\n", path.c_str());
        return false;
    }
    Archive ar(ifs);
    ar(data);
    return true;
}
bool Load(SfMData& sfm_data, const std::string path) {
    size_t index =  path.find_last_of(".");
    std::string extersion = path.substr(index + 1);

    if (extersion == "bin") {
        return LoadCereal<cereal::BinaryInputArchive>(sfm_data, path);
    } else if (extersion == "json") {
        return LoadCereal<cereal::JSONInputArchive>(sfm_data, path);
    }
    return false;
}
