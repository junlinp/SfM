#include "brute_force_matcher.hpp"

#include <iostream>

#include "internal/thread_pool.hpp"
#include "metric.hpp"

void BruteForceMatcher::Match(SfMData& sfm_data, const std::set<Pair>& pairs) {
  // Initial data
  for (Pair pair : pairs) {
    sfm_data.matches[pair] = Matches();
  }
  auto functor = [](Descriptors lhs, Descriptors rhs, std::vector<KeyPoint> lhs_keypoint, std::vector<KeyPoint> rhs_keypoint) {
    size_t lhs_size = lhs.size();
    size_t rhs_size = rhs.size();
    Matches match;
    for (int i = 0; i < lhs_size; i++) {
      size_t min_index = -1;
      float min_distance = std::numeric_limits<float>::max();
      size_t second_min_index = -1;
      float second_min_distance = std::numeric_limits<float>::max();
      float* lhs_ptr = lhs.data();
      float* rhs_ptr = rhs.data();
      for (int j = 0; j < rhs_size; j++) {
        float distance = L2_metric(lhs_ptr + 128 * i, rhs.data() + 128 * j);
        if (distance < second_min_distance) {
          second_min_distance = distance;
          second_min_index = j;
          if (second_min_distance < min_distance) {
            std::swap(second_min_index, min_index);
            std::swap(second_min_distance, min_distance);
          }
        }
      }
      if (min_distance < 0.8 * second_min_distance) {
        struct Match m{ToObservation(lhs_keypoint[i]), ToObservation(rhs_keypoint[min_index]), i, IndexT(min_index)};
        match.push_back(m);
      }
    }
    return match;
  };

  {
    ThreadPool thread_pool;
    std::map<Pair, std::future<Matches>> future_matches;
    for (Pair pair : pairs) {
      Descriptors lhs_descriptor = sfm_data.descriptors[pair.first];
      Descriptors rhs_descriptor = sfm_data.descriptors[pair.second];
      std::vector<KeyPoint> lhs_keypoint = sfm_data.key_points[pair.first];
      std::vector<KeyPoint> rhs_keypoint = sfm_data.key_points[pair.second];

      auto res = thread_pool.Enqueue(functor, lhs_descriptor, rhs_descriptor, lhs_keypoint, rhs_keypoint);
      future_matches.insert({pair, std::move(res)});
    }
    int count = 0;
    int total_size = pairs.size();
    for (Pair pair : pairs) {
        sfm_data.matches[pair] = future_matches[pair].get();
        std::printf("Finish %f %%\n", 100.0 * ++count / total_size);
    }
  }
}