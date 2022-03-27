#ifndef SRC_FEATURE_MATCHER_CASCADE_HASH_MATCHER_HPP_
#define SRC_FEATURE_MATCHER_CASCADE_HASH_MATCHER_HPP_

#include <bitset>
#include <unordered_map>
#include <vector>

#include "feature_matcher_interface.hpp"

struct HashDescriptor {
  std::bitset<128> hash_code;

  // bucket_ids[x] = y means this descriptor belongs bucket y of group x
  std::vector<uint16_t> bucket_ids;
};

struct HashDescriptors {
  std::vector<HashDescriptor> descriptor;

  // buckets[bucket_group][bucket_id] = bucket (container of description idx).
  std::vector<std::vector<std::vector<IndexT> > > bucket_ids;
};

class CascadeHashMatcher : public FeatureMatcherInterface {
 public:
  void Match(SfMData& sfm_data, const std::set<Pair>& pairs,
             bool is_corss_valid = true) override;
  virtual ~CascadeHashMatcher() = default;

 private:
  HashDescriptors ConstructHashDescriptors(const Descriptors& descriptors,
                                           Eigen::VectorXf zero_mean_vector);
  Matches DescriptorMath(const HashDescriptors& lhs_hash_descriptor,
                         const Descriptors& lhs_descriptor,
                         const HashDescriptors& rhs_hash_descriptor,
                         const Descriptors& rhs_descriptor);
  std::unordered_map<IndexT, HashDescriptors> hash_descriptors_;
  Eigen::MatrixXf primary_hash_project;
  std::vector<Eigen::MatrixXf> secondary_hash_project;

  int bucket_group_size;
  int bits_of_group;
  int hash_code_size;
  int group_size;
};
#endif  // SRC_FEATURE_MATCHER_CASCADE_HASH_MATCHER_HPP_