#include "exhaustive_pair_builder.hpp"
#include "internal/function_programming.hpp"
void ExhaustivePairBuilder::BuildPair(SfMData& sfm_data, std::set<Pair>& pair_result) {
    
    auto functor = [](auto pair_item) {
        return pair_item.first;
    };
    std::vector<IndexT> index_vec = sfm_data.views | Transform(functor) | ToVector();
    for (int i = 0;i < index_vec.size(); i++) {
        for (int j = i + 1; j < index_vec.size(); j++) {
            pair_result.insert(
                {
                    std::min(index_vec[i], index_vec[j]),
                    std::max(index_vec[i], index_vec[j])
                }
            );
        }
    }
}