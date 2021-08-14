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

struct SparsePoint {
    double x, y, z;
    SparsePoint(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}
};

struct SfMData {
    std::map<IndexT, View> views;

    // Features
    std::map<IndexT, std::vector<KeyPoint>> key_points;
    std::map<IndexT, Descriptors> descriptors;

    // Matches
    std::map<Pair, Matches> matches;

    // structure

    std::vector<SparsePoint> structure_points;
};

#endif // SFM_DATA_H_