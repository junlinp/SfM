#include "gtest/gtest.h"
#include "trifocal_tensor_solver.hpp"

// Range for [min, max)
template<typename T>
std::vector<T> Range(T min, T max) {

    std::vector<T> range(max - min);
    std::iota(range.begin(), range.end(), min);
    return range;
}

template<typename T>
std::vector<T> Range(T max) {
    return Range(T(0), max);
}

TEST(Tensor, Eipson) {
  int v[3] = {1, 2, 3};

  Eigen::Matrix3i estimate = SkewMatrix(v);

  Eigen::Matrix3d target;
  for(int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      target(row, col) = 0;
      for (int k = 0; k < 3; k++) {
        target(row, col) += EpsilonTensor::at(row, k, col) * v[k];
      }
      
      EXPECT_EQ(target(row, col), estimate(row, col));
    }
  }
}

TEST(TrifocalError, Evaluate) {
  auto rand = []() { 
  std::default_random_engine rg;
  std::normal_distribution<double> normal(0.0, 1.0);
    return normal(rg);
  };

  TripleMatch data_point;
  data_point.I_observation = Observation(rand(), rand());
  data_point.J_observation = Observation(rand(), rand());
  data_point.K_observation = Observation(rand(), rand());

  Trifocal model;
  model.lhs = Eigen::Matrix3d::Random();
  model.middle = Eigen::Matrix3d::Random();
  model.rhs = Eigen::Matrix3d::Random();

  Eigen::VectorXd origin = TrifocalSamponErrorEvaluator::Evaluate(data_point, model);
  auto lhs = SkewMatrix(data_point.J_observation.homogeneous()) ;
  auto middle = (data_point.I_observation(0) * model.lhs + data_point.I_observation(1) * model.middle + 1.0 * model.rhs);
  auto rhs = SkewMatrix(data_point.K_observation.homogeneous());
  auto Target = lhs * middle  * rhs;

  EXPECT_NEAR(Target(0, 0), origin(0), 1e-2);
  EXPECT_NEAR(Target(0, 1), origin(1), 1e-2);
  EXPECT_NEAR(Target(1, 0), origin(2), 1e-2);
  EXPECT_NEAR(Target(1, 1), origin(3), 1e-2);
}

TEST(TrifocalError, SampsonError) {
  Mat34 P1, P2, P3;
  P1 << 1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0;
  P2.setRandom();
  P3.setRandom();

  Trifocal model;
  model.lhs = P2.col(0) * P3.col(3).transpose() - P2.col(3) * P3.col(0).transpose();
  model.middle = P2.col(1) * P3.col(3).transpose() - P2.col(3) * P3.col(1).transpose();
  model.rhs = P2.col(2) * P3.col(3).transpose() - P2.col(3) * P3.col(2).transpose();

  Eigen::Vector3d X;
  X.setRandom();

  TripleMatch data_point;
  data_point.I_observation = (P1 * X.homogeneous()).hnormalized();
  data_point.J_observation = (P2 * X.homogeneous()).hnormalized();
  data_point.K_observation = (P3 * X.homogeneous()).hnormalized();

  double sampson_error = TrifocalSampsonError::Error(data_point, model);
  EXPECT_NEAR(sampson_error, 0, 1e-10);
  std::default_random_engine engine;
  double sigma = 0.5;
  std::normal_distribution<double> d(0.0, sigma);

  data_point.I_observation.x() += d(engine);
  data_point.I_observation.y() += d(engine);
  data_point.J_observation.x() += d(engine);
  data_point.J_observation.y() += d(engine);
  data_point.K_observation.x() += d(engine);
  data_point.K_observation.y() += d(engine);

  sampson_error = TrifocalSampsonError::Error(data_point, model);
  std::printf("Sampson Error : %f\n", sampson_error);
  EXPECT_NEAR(sampson_error, 0, 12.59 * sigma * sigma);
}

TEST(Trifocal, LinearSolver) {
  Mat34 P1, P2, P3;
  P1 << 1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0;
  P2.setRandom();
  P3.setRandom();
  //Trifocal model;
  //model.lhs = P2.col(0) * P3.col(3).transpose() - P2.col(3) * P3.col(0).transpose();
  //model.middle = P2.col(1) * P3.col(3).transpose() - P2.col(3) * P3.col(1).transpose();
  //model.rhs = P2.col(2) * P3.col(3).transpose() - P2.col(3) * P3.col(2).transpose();

  size_t data_size = 20;
  std::vector<TripleMatch> data_set;
  for (size_t i = 0; i < data_size; i++) {
    Eigen::Vector3d X;
    X.setRandom();
    TripleMatch data_point;
    data_point.I_observation = (P1 * X.homogeneous()).hnormalized();
    data_point.J_observation = (P2 * X.homogeneous()).hnormalized();
    data_point.K_observation = (P3 * X.homogeneous()).hnormalized();
    data_set.push_back(data_point);
  }
  Trifocal model;
  LinearSolver::Fit(data_set, &model);

  double error = 0.0;
  for (auto data_point : data_set) {
    double temp = TrifocalSampsonError::Error(data_point, model);
    ASSERT_FALSE(std::isnan(temp));
    error += temp;
  }
  error /= data_size;
  EXPECT_NEAR(error, 0, 12.59 * 0.1 * 0.1);
}

TEST(Trifocal, AlgebraMinimumSolver) {
  Mat34 P1, P2, P3;
  P1 << 1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0;
  P2.setRandom();
  P3.setRandom();
  //Trifocal model;
  //model.lhs = P2.col(0) * P3.col(3).transpose() - P2.col(3) * P3.col(0).transpose();
  //model.middle = P2.col(1) * P3.col(3).transpose() - P2.col(3) * P3.col(1).transpose();
  //model.rhs = P2.col(2) * P3.col(3).transpose() - P2.col(3) * P3.col(2).transpose();

  size_t data_size = 20;
  std::vector<TripleMatch> data_set;
  for (size_t i = 0; i < data_size; i++) {
    Eigen::Vector3d X;
    X.setRandom();
    TripleMatch data_point;
    data_point.I_observation = (P1 * X.homogeneous()).hnormalized();
    data_point.J_observation = (P2 * X.homogeneous()).hnormalized();
    data_point.K_observation = (P3 * X.homogeneous()).hnormalized();
    data_set.push_back(data_point);
  }
  Trifocal model;
  AlgebraMinimumSolver::Fit(data_set, &model);

  double error = 0.0;
  for (auto data_point : data_set) {
    double temp = TrifocalSampsonError::Error(data_point, model);
    ASSERT_FALSE(std::isnan(temp));
    error += temp;
  }
  error /= data_size;
  EXPECT_NEAR(error, 0, 12.59 * 0.1 * 0.1);
}

TEST(Trifocal, BundleRefineSolver) {
  Mat34 P1, P2, P3;
  P1 << 1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0;
  P2.setRandom();
  P3.setRandom();
  //Trifocal model;
  //model.lhs = P2.col(0) * P3.col(3).transpose() - P2.col(3) * P3.col(0).transpose();
  //model.middle = P2.col(1) * P3.col(3).transpose() - P2.col(3) * P3.col(1).transpose();
  //model.rhs = P2.col(2) * P3.col(3).transpose() - P2.col(3) * P3.col(2).transpose();

  size_t data_size = 20;
  std::vector<TripleMatch> data_set;
  for (size_t i = 0; i < data_size; i++) {
    Eigen::Vector3d X;
    X.setRandom();
    TripleMatch data_point;
    data_point.I_observation = (P1 * X.homogeneous()).hnormalized();
    data_point.J_observation = (P2 * X.homogeneous()).hnormalized();
    data_point.K_observation = (P3 * X.homogeneous()).hnormalized();
    data_set.push_back(data_point);
  }
  Trifocal model;
  BundleRefineSolver::Fit(data_set, &model);

  double error = 0.0;
  for (auto data_point : data_set) {
    double temp = TrifocalSampsonError::Error(data_point, model);
    ASSERT_FALSE(std::isnan(temp));
    error += temp;
    EXPECT_NEAR(temp, 0, 12.59 * 0.1 * 0.1);
  }
  EXPECT_NEAR(error, 0, 12.59 * 0.1 * 0.1);
}

TEST(Trifocal, RansacTrifocalSolver) {
  Mat34 P1, P2, P3;
  P1 << 1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0;
  P2.setRandom();
  P3.setRandom();
  //Trifocal model;
  //model.lhs = P2.col(0) * P3.col(3).transpose() - P2.col(3) * P3.col(0).transpose();
  //model.middle = P2.col(1) * P3.col(3).transpose() - P2.col(3) * P3.col(1).transpose();
  //model.rhs = P2.col(2) * P3.col(3).transpose() - P2.col(3) * P3.col(2).transpose();

  size_t inlier_size = 60;
  double outlier_rate = 0.4;
  size_t outlier_size = inlier_size / (1.0 - outlier_rate) * outlier_rate;

  std::vector<TripleMatch> data_set;
  std::vector<TripleMatch> inlier_set(inlier_size);
  for (auto i : Range(inlier_size)) {
    Eigen::Vector3d X;
    X.setRandom();
    TripleMatch data_point;
    data_point.I_observation = (P1 * X.homogeneous()).hnormalized();
    data_point.J_observation = (P2 * X.homogeneous()).hnormalized();
    data_point.K_observation = (P3 * X.homogeneous()).hnormalized();
    data_set.push_back(data_point);
    inlier_set.push_back(data_point);
  }

  for (auto i : Range(outlier_size)) {
      TripleMatch data_point;
      data_point.I_observation.setRandom();
      data_point.J_observation.setRandom();
      data_point.K_observation.setRandom();
      data_set.push_back(data_point);
  }

  Trifocal model;
  RansacTrifocalSolver::Fit(data_set, model);

  double error = 0.0;
  for (auto data_point :inlier_set) {
    double temp = TrifocalSampsonError::Error(data_point, model);
    ASSERT_FALSE(std::isnan(temp));
    error += temp;
  }
  error /= inlier_size;
  EXPECT_NEAR(error, 0, 12.59 * 0.1 * 0.1);
}

int main(int argc, char**argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}