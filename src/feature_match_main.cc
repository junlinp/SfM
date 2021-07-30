#include "sfm_data_io.hpp"
#include "sfm_data.hpp"
#include "feature_matcher/exhaustive_pair_builder.hpp"
#include "feature_matcher/brute_force_matcher.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        return 1;
    }

    SfMData sfm_data;
    bool b_load = Load(sfm_data, argv[1]);
    if (b_load) {
        std::printf("Load Sfm Data Finish\n");
    } else {
        std::printf("Load Sfm Data From %s Fails\n", argv[1]);
    }

    auto pair_selector = std::make_shared<ExhaustivePairBuilder>();
    std::set<Pair> pairs;
    pair_selector->BuildPair(sfm_data, pairs);

    std::printf("Need To Match %lu Pairs.\n", pairs.size());

    auto Matcher = std::make_shared<BruteForceMatcher>();
    Matcher->Match(sfm_data, pairs);

    Save(sfm_data, argv[1]);

    return 0;
}