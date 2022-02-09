#ifndef FEATURE_MATCHER_FEATURE_MATCHER_INTERFACE_H_
#define FEATURE_MATCHER_FEATURE_MATCHER_INTERFACE_H_
#include "sfm_data.hpp"
#include <set>
class FeatureMatcherInterface {
public:

    virtual void Match(SfMData& sfm_data, const std::set<Pair>& pairs, bool is_corss_valid = true) = 0;

    virtual ~FeatureMatcherInterface() = default;
};
#endif // FEATURE_MATCHER_FEATURE_MATCHER_INTERFACE_H_