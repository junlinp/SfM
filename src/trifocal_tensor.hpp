#ifndef SRC_TRIFOCAL_TENSOr_HPP_
#define SRC_TRIFOCAL_TENSOr_HPP_
#include "eigen_alias_types.hpp"
#include "types_define.hpp"

struct TripleIndex {
  IndexT I, J, K;
};

struct TripleMatch {
  IndexT I_idx;
  Observation I_observation;
  IndexT J_idx;
  Observation J_observation;
  IndexT K_idx;
  Observation K_observation;
};

struct Trifocal {
  Eigen::Matrix3d lhs, middle, rhs;

  friend std::ostream& operator<<(std::ostream& os, Trifocal tirfocal);
};

#endif //SRC_TRIFOCAL_TENSOr_HPP_
