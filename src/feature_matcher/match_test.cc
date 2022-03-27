#include "gtest/gtest.h"
#include "cascade_hash_matcher.hpp"
#include "brute_force_matcher.hpp"

#include <random>

class FeatureMatch : public ::testing::Test {
public:
    void SetUp() override {
        std::default_random_engine gen;
        std::normal_distribution<> d;

        size_t lhs_n = 1024;
        size_t rhs_n = 512;
        std::vector<std::array<float, 128>> features;
        for(int i = 0; i < lhs_n; i++) {
            std::array<float, 128> feature;
            for (int j = 0; j < 128; j++) {
                feature[j] = d(gen);
            }
            features.push_back(feature);
        }
        std::vector<std::array<float, 128>> outlier_features;

        for(int i = 0; i < lhs_n - rhs_n; i++) {

            std::array<float, 128> feature;
            for (int j = 0; j < 128; j++) {
                feature[j] = d(gen);
            }
            outlier_features.push_back(feature);
        }

        std::vector<int> shuffle(lhs_n);
        std::iota(shuffle.begin(), shuffle.end(), 0);
        std::shuffle(shuffle.begin(), shuffle.end(), gen);

        Matches matches;
        Descriptors lhs_descriptors;
        Descriptors rhs_descriptors;

        for(int i = 0; i < rhs_n; i++) {
            Match match;
            match.lhs_idx = i;
            match.rhs_idx = shuffle[i];
            rhs_descriptors.push_back(features[shuffle[i]]);
        }

        for (int i = 0; i < lhs_n; i++) {
            lhs_descriptors.push_back(features[i]);
        }

        for (int i = 0; i < lhs_n - rhs_n; i++) {
            rhs_descriptors.push_back(outlier_features[i]);
        }
        puative_match = std::move(matches);
        sfm_data.descriptors[0] = lhs_descriptors;
        sfm_data.descriptors[1] = rhs_descriptors;
        for (int i = 0; i < lhs_n; i++) {
            sfm_data.key_points[0].push_back(KeyPoint());
            sfm_data.key_points[1].push_back(KeyPoint());
        }
    }

    SfMData sfm_data;
    Matches puative_match;
};

TEST_F(FeatureMatch, Exhaustive) {
    auto matcher = std::make_shared<BruteForceMatcher>();
    matcher->Match(sfm_data, {{0, 1}}, true);
    std::unordered_map<IndexT, IndexT> matcher_result;
    for (Match& m : sfm_data.matches[{0, 1}]) {
        matcher_result[m.lhs_idx] = m.rhs_idx;
    }

    for (Match& m : puative_match) {
        EXPECT_TRUE(matcher_result.find(m.lhs_idx) == matcher_result.end());
        EXPECT_EQ(m.rhs_idx, matcher_result[m.lhs_idx]);
    }
}

TEST_F(FeatureMatch, CascadeHash) {

    auto matcher = std::make_shared<CascadeHashMatcher>();
    matcher->Match(sfm_data, {{0, 1}}, true);
    std::unordered_map<IndexT, IndexT> matcher_result;
    for (Match& m : sfm_data.matches[{0, 1}]) {
        matcher_result[m.lhs_idx] = m.rhs_idx;
    }

    for (Match& m : puative_match) {
        EXPECT_TRUE(matcher_result.find(m.lhs_idx) == matcher_result.end());
        EXPECT_EQ(m.rhs_idx, matcher_result[m.lhs_idx]);
    }
}


int main(int argc, char**argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}