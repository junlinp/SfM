#ifndef FEATURE_MATCHER_EXHAUSTIVE_PAIR_BUILDER_H_
#define FEATURE_MATCHER_EXHAUSTIVE_PAIR_BUILDER_H_

#include "pair_builder_interface.hpp"

class ExhaustivePairBuilder : public PairBuilderInterface {
public:

    void BuildPair(SfMData& sfm_data, std::set<Pair>& pair_result) override;

    virtual ~ExhaustivePairBuilder() = default;
};
#endif  //FEATURE_MATCHER_EXHAUSTIVE_PAIR_BUILDER_H_