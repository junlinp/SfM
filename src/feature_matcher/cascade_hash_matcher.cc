#include "cascade_hash_matcher.hpp"

#include <iostream>
#include <random>
#include <unordered_set>

#include "Eigen/Dense"
#include "metric.hpp"

using namespace Eigen;
HashDescriptors CascadeHashMatcher::ConstructHashDescriptors(
    const Descriptors& descriptors, VectorXf zero_mean_vector) {
  size_t n = descriptors.size();

  HashDescriptors hash_descriptors;
  hash_descriptors.descriptor.resize(n);

  for (int i = 0; i < n; i++) {
    VectorXf d = Eigen::Map<VectorXf>(
        const_cast<float*>(descriptors.data() + i * 128), 128);
    d -= zero_mean_vector;

    const VectorXf primary_project = primary_hash_project * d;
    for (int j = 0; j < 128; j++) {
      hash_descriptors.descriptor[i].hash_code[j] = primary_project[j] > 0;
    }

    hash_descriptors.descriptor[i].bucket_ids.resize(bucket_group_size);
    for (int group_index = 0; group_index < bucket_group_size; group_index++) {
      const VectorXf second_project = secondary_hash_project[group_index] * d;
      uint16_t bucket_id = 0;

      for (int k = 0; k < bits_of_group; k++) {
        bucket_id = (bucket_id << 1) + (second_project(k) > 0);
      }

      hash_descriptors.descriptor[i].bucket_ids[group_index] = bucket_id;
    }
  }
  hash_descriptors.bucket_ids.resize(bucket_group_size);
  for (int group_index = 0; group_index < bucket_group_size; group_index++) {
    hash_descriptors.bucket_ids[group_index].resize(group_size);
    for (int i = 0; i < hash_descriptors.descriptor.size(); i++) {
      int bucket_id = hash_descriptors.descriptor[i].bucket_ids[group_index];
      hash_descriptors.bucket_ids[group_index][bucket_id].push_back(i);
    }
  }
  return hash_descriptors;
}

Matches CascadeHashMatcher::DescriptorMath(
    const HashDescriptors& lhs_hash_descriptor,
    const Descriptors& lhs_descriptor,
    const HashDescriptors& rhs_hash_descriptor,
    const Descriptors& rhs_descriptor) {
  Matches matches;
  static const int kNumTopCandidates = 10;
  std::vector<int> candidate_descriptors;
  candidate_descriptors.reserve(rhs_hash_descriptor.descriptor.size());

  std::unordered_map<int, std::vector<IndexT>> candidate_hamming_distance;
  std::unordered_map<Index, bool> used_feature_idx;

  std::vector<std::pair<float, IndexT>> euclidean_distance;

  for (int i = 0; i < lhs_hash_descriptor.descriptor.size(); i++) {
    candidate_descriptors.clear();
    candidate_hamming_distance.clear();
    euclidean_distance.clear();
    used_feature_idx.clear();
    const auto& hashed_descriptor = lhs_hash_descriptor.descriptor[i];
    for (int j = 0; j < bucket_group_size; j++) {
      uint16_t bucket_idx = hashed_descriptor.bucket_ids[j];
      for (auto feature_idx : rhs_hash_descriptor.bucket_ids[j][bucket_idx]) {
        candidate_descriptors.push_back(feature_idx);
        used_feature_idx[feature_idx] = false;
      }
    }

    if (candidate_descriptors.size() < 2) {
      continue;
    }

    for (IndexT candidate_id : candidate_descriptors) {
      if (!used_feature_idx[candidate_id]) {
        used_feature_idx[candidate_id] = true;
        // compute hamming distance
        int hamming_distance = HammingDistance(
            hashed_descriptor.hash_code,
            rhs_hash_descriptor.descriptor.at(candidate_id).hash_code);
        candidate_hamming_distance[hamming_distance].push_back(candidate_id);
      }
    }

    for (int k = 0;
         k <= bits_of_group && euclidean_distance.size() < kNumTopCandidates;
         k++) {
      for (IndexT candidate_id : candidate_hamming_distance[k]) {
        float euclidean = L2_metric(lhs_descriptor.data() + i * 128,
                                    rhs_descriptor.data() + candidate_id * 128);
        euclidean_distance.emplace_back(euclidean, candidate_id);
      }
    }

    if (euclidean_distance.size() >= 2) {
      std::partial_sort(euclidean_distance.begin(),
                        euclidean_distance.begin() + 2,
                        euclidean_distance.end());
      struct Match match;
      match.lhs_idx = i;
      match.rhs_idx = euclidean_distance[0].second;
      matches.push_back(match);
    }
  }
  return matches;
}

void CascadeHashMatcher::Match(SfMData& sfm_data, const std::set<Pair>& pairs,
                               bool is_corss_valid) {
  // Create HashDescriptors;

  bucket_group_size = 6;
  bits_of_group = 10;
  hash_code_size = 128;
  group_size = 1 << bits_of_group;

  std::mt19937 gen(std::mt19937::default_seed);
  std::normal_distribution<> d(0, 1);

  primary_hash_project = MatrixXf(hash_code_size, hash_code_size);
  for (int i = 0; i < hash_code_size; i++) {
    for (int j = 0; j < hash_code_size; j++) {
      primary_hash_project(i, j) = d(gen);
    }
  }

  secondary_hash_project.resize(bucket_group_size);
  for (int t = 0; t < bucket_group_size; t++) {
    secondary_hash_project[t] = MatrixXf(bits_of_group, hash_code_size);

    for (int i = 0; i < bits_of_group; i++) {
      for (int j = 0; j < hash_code_size; j++) {
        secondary_hash_project[t](i, j) = d(gen);
      }
    }
  }

  std::unordered_set<IndexT> index_set;
  for (const Pair& pair_index : pairs) {
    index_set.insert(pair_index.first);
    index_set.insert(pair_index.second);
  }

  VectorXf zero_mean_vector(128);
  zero_mean_vector.setZero();
  int total_size = 0;
  for (IndexT index : index_set) {
    float* raw_data = sfm_data.descriptors[index].data();
    int descriptor_size = sfm_data.descriptors[index].size();
    MatrixXf m = Map<MatrixXf>(raw_data, descriptor_size, 128);

    zero_mean_vector += m.colwise().sum();
    total_size += descriptor_size;
  }
  zero_mean_vector /= total_size;
  std::cout << "mean " << std::endl;
  for (IndexT index : index_set) {
    hash_descriptors_[index] =
        ConstructHashDescriptors(sfm_data.descriptors[index], zero_mean_vector);
  }
  std::cout << "Create Descriptors Finish" << std::endl;

  for (auto [lhs_index, rhs_index] : pairs) {
    Matches matches = DescriptorMath(
        hash_descriptors_[lhs_index], sfm_data.descriptors[lhs_index],
        hash_descriptors_[rhs_index], sfm_data.descriptors[rhs_index]);
    if (is_corss_valid) {
      Matches reverse_matches = DescriptorMath(
          hash_descriptors_[rhs_index], sfm_data.descriptors[rhs_index],
          hash_descriptors_[lhs_index], sfm_data.descriptors[lhs_index]);

      std::unordered_map<IndexT, IndexT> reverse_map;
      for (struct Match& m : reverse_matches) {
        reverse_map[m.lhs_idx] = m.rhs_idx;
      }

      Matches temp(matches.size());
      for (struct Match& m : matches) {
        if (reverse_map.find(m.rhs_idx) != reverse_map.end() &&
            reverse_map.at(m.rhs_idx) == m.lhs_idx) {
          temp.push_back(m);
        }
      }
      matches.swap(temp);
    }
    std::cout << "Piar : [" << lhs_index << "," << rhs_index << "] " << matches.size() << " Matches." << std::endl;
    for (struct Match& m : matches) {
      KeyPoint& lhs_keypoint = sfm_data.key_points[lhs_index][m.lhs_idx];
      KeyPoint& rhs_keypoint = sfm_data.key_points[rhs_index][m.rhs_idx];
      m.lhs_observation = Observation(lhs_keypoint.x, lhs_keypoint.y);
      m.rhs_observation = Observation(rhs_keypoint.x, rhs_keypoint.y);
    }

    sfm_data.matches[{lhs_index, rhs_index}] = matches;
  }
}