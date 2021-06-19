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
using Pair = std::pair<IndexT, IndexT>;
using Matche = std::pair<size_t, size_t>;
using Matches = std::vector<Matche>;

struct SfMData {
    std::map<IndexT, View> views;

    // Features
    std::map<IndexT, std::vector<KeyPoint>> key_points;
    std::map<IndexT, Descriptors> descriptors;

    // Matches
    std::map<Pair, Matches> matches;
};

#endif // SFM_DATA_H_