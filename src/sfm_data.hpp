#ifndef SFM_DATA_H_
#define SFM_DATA_H_

#include <string>
#include <map>

#include "descriptor.hpp"
#include "keypoint.hpp"

using IndexT = int64_t;

struct View {
    std::string image_path;
};

struct SfMData {
    std::map<IndexT, View> views;
    std::map<IndexT, std::vector<KeyPoint>> key_points;
    std::map<IndexT, Descriptors> descriptors;
};

#endif // SFM_DATA_H_