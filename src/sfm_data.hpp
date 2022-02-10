#ifndef SFM_DATA_H_
#define SFM_DATA_H_

#include <string>
#include <map>
#include <set>

#include "descriptor.hpp"
#include "keypoint.hpp"
#include "eigen_alias_types.hpp"
#include "types_define.hpp"

struct View {
    std::string image_path;
};

template<typename T>
Observation ToObservation(T other) {
    return Observation(other.x, other.y);
}
struct Match {
    Observation lhs_observation, rhs_observation;
    // used to hash
    IndexT lhs_idx, rhs_idx;
};

using Pair = std::pair<IndexT, IndexT>;

//using Match = std::pair<size_t, size_t>;

using Matches = std::vector<Match>;


struct SparsePoint {
    double x, y, z;
    SparsePoint(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}
    // <ImageID,  obs>
    std::map<IndexT,  Observation> obs;
};


struct SfMData {

    size_t image_width, image_height;
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