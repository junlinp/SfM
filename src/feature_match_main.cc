#include "sfm_data_io.hpp"
#include "sfm_data.hpp"
#include "feature_matcher/exhaustive_pair_builder.hpp"
#include "feature_matcher/brute_force_matcher.hpp"
#include "feature_matcher/cascade_hash_matcher.hpp"
#include <atomic>
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

    //auto Matcher = std::make_shared<BruteForceMatcher>();
    auto Matcher = std::make_shared<CascadeHashMatcher>();
    size_t total_piar_size = pairs.size();
    std::atomic<size_t> finished_count(0);
    auto call_back = [total_piar_size, &finished_count](Pair& pair, Matches& matches) {
        finished_count++;
        std::printf("Pair [%lld, %lld] With %lu Matches \t\t %f %%\n", pair.first, pair.second, matches.size(), 100.0 * finished_count / total_piar_size);
    };
    Matcher->SetCallBack(call_back);
    Matcher->Match(sfm_data, pairs, true);

    Save(sfm_data, argv[1]);

    return 0;
}