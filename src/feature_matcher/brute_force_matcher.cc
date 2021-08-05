#include "brute_force_matcher.hpp"

#include <cmath>
#include <iostream>

#include "internal/thread_pool.hpp"

float L2_metric(float* lhs, float* rhs) {
  float sum = 0.0f;

  for (int i = 0; i < 128; i++) {
    float temp = lhs[i] - rhs[i];
    sum += temp * temp;
  }
  return std::sqrt(sum);
}

void BruteForceMatcher::Match(SfMData& sfm_data, const std::set<Pair>& pairs) {
  // Initial data
  for (Pair pair : pairs) {
    sfm_data.matches[pair] = Matches();
  }
  auto functor = [](Descriptors lhs, Descriptors rhs) {
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
        match.push_back({i, min_index});
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

      auto res = thread_pool.Enqueue(functor, lhs_descriptor, rhs_descriptor);
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