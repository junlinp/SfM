#ifndef FEATURE_MATCHER_PAIR_BUILDER_INTERFACE_H_
#define FEATURE_MATCHER_PAIR_BUILDER_INTERFACE_H_
#include "../sfm_data.hpp"
#include <set>

class PairBuilderInterface {
public:
    virtual void BuildPair(SfMData& sfm_data, std::set<Pair>& pair_result) = 0;

    virtual ~PairBuilderInterface() = default;
};

#endif  //FEATURE_MATCHER_PAIR_BUILDER_INTERFACE_H_