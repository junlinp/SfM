#ifndef FEATURE_MATCHER_BRUTE_FORCE_MATCHER_H_
#define FEATURE_MATCHER_BRUTE_FORCE_MATCHER_H_
#include "feature_matcher_interface.hpp"
#include "sfm_data.hpp"
class BruteForceMatcher : public FeatureMatcherInterface {
public:
    void Match(SfMData& sfm_data, const std::set<Pair>& pairs, bool is_cross_valid = true) override;

    virtual ~BruteForceMatcher() = default;
};

#endif  //FEATURE_MATCHER_BRUTE_FORCE_MATCHER_H_