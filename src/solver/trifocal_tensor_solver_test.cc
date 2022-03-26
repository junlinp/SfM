#include "gtest/gtest.h"
#include "trifocal_tensor_solver.hpp"
#include "ceres/rotation.h"

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

  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      ASSERT_NEAR(Target(row, col), origin(row * 3 + col), 1e-2);
    }
  }
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
  ASSERT_LT(sampson_error, 1e-10);
  ASSERT_GT(sampson_error, 0.0);
  std::default_random_engine engine;
  double sigma = 1.0;
  std::normal_distribution<double> d(0.0, sigma);

  data_point.I_observation.x() += d(engine);
  data_point.I_observation.y() += d(engine);
  data_point.J_observation.x() += d(engine);
  data_point.J_observation.y() += d(engine);
  data_point.K_observation.x() += d(engine);
  data_point.K_observation.y() += d(engine);

  sampson_error = TrifocalSampsonError::Error(data_point, model);
  std::printf("Sampson Error : %f\n", sampson_error);
  ASSERT_LT(sampson_error, 12.59 * sigma * sigma);
  ASSERT_GT(sampson_error, 0.0);
}

TEST(Trifocal, RecoveryCameraMatrix) {
  Mat34 P1, P2, P3;
  P1 << 1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0;

  P2.setRandom();
  P3.setRandom();

  Eigen::Vector3d a1 = P2.col(0);
  Eigen::Vector3d a2 = P2.col(1);
  Eigen::Vector3d a3 = P2.col(2);
  Eigen::Vector3d a4 = P2.col(3);

  Eigen::Vector3d b1 = P3.col(0);
  Eigen::Vector3d b2 = P3.col(1);
  Eigen::Vector3d b3 = P3.col(2);
  Eigen::Vector3d b4 = P3.col(3);

  Trifocal true_trifocal;
  true_trifocal.lhs = a1 * b4.transpose() - a4 * b1.transpose();
  true_trifocal.middle = a2 * b4.transpose() - a4 * b2.transpose();
  true_trifocal.rhs = a3 * b4.transpose() - a4 * b3.transpose();

  Mat34 recovery_P1, recovery_P2, recovery_P3;
  RecoveryCameraMatrix(true_trifocal, recovery_P1, recovery_P2, recovery_P3);

  Eigen::Vector2d true_normal = P2.col(3).hnormalized();
  Eigen::Vector2d estimate_normal = recovery_P2.col(3).hnormalized();
  std::cout << "Ground True : " << true_normal << std::endl << " Estimate : " << estimate_normal << std::endl;

  EXPECT_LT((estimate_normal - true_normal).norm(), 0.1 * true_normal.norm());
}
std::tuple<std::vector<TripleMatch>, std::vector<TripleMatch>, Mat34, Mat34> GenerateDataSet(size_t data_size, double sigma) {
  Mat34 P1, P2, P3;
  P1 << 1.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0;
  Eigen::Matrix3d K;
  K << 256.0, 0.0, 512.0,
        0.0,  128.0, 256.0,
        0.0, 0.0, 1.0;
  Eigen::Matrix3d R2, R3;
  Eigen::Vector3d c2, c3;

  std::random_device rd;
  std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
  // Look at (0, 0, 10);
  c2 << uniform_distribution(rd), uniform_distribution(rd), uniform_distribution(rd);
  c3 << uniform_distribution(rd), uniform_distribution(rd), uniform_distribution(rd);
  c2 << 1.0, 2.0, 3.0;
  c3 << 3.0, 4.0, 5.0;
  Eigen::Vector3d c2_hat = Eigen::Vector3d(0, 0, 10) - c2; 
  Eigen::Vector3d c3_hat = Eigen::Vector3d(0, 0, 10) - c3; 
  std::cout << "c2_hat : " << c2_hat << std::endl;
  R2.row(2) = c2_hat / c2_hat.norm();
  auto svd = R2.row(2).bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  std::cout << "RU : " << svd.matrixU() << std::endl;
  std::cout << "RS : " << svd.singularValues() << std::endl;
  std::cout << "RV : " << svd.matrixV() << std::endl;
  R2.row(0) = svd.matrixV().col(1);
  R2.row(1) = svd.matrixV().col(2);

  R3.row(2) = c3_hat / c3_hat.norm();
  auto svd3 = R3.row(2).bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  R3.row(0) = svd3.matrixV().col(1);
  R3.row(1) = svd3.matrixV().col(2);


  std::cout << "Rotation2 : " << R2 << std::endl;

  P2 << R2, - R2 * c2;
  P3 << R3, - R3 * c3;
  P2 =   P2;
  P3 =   P3;

  std::cout << "P2 : " << P2 << std::endl << " P3 : " << P3 << std::endl;


  Eigen::Vector3d a1 = P2.col(0);
  Eigen::Vector3d a2 = P2.col(1);
  Eigen::Vector3d a3 = P2.col(2);
  Eigen::Vector3d a4 = P2.col(3);

  Eigen::Vector3d b1 = P3.col(0);
  Eigen::Vector3d b2 = P3.col(1);
  Eigen::Vector3d b3 = P3.col(2);
  Eigen::Vector3d b4 = P3.col(3);

  Trifocal true_trifocal;
  true_trifocal.lhs = a1 * b4.transpose() - a4 * b1.transpose();
  true_trifocal.middle = a2 * b4.transpose() - a4 * b2.transpose();
  true_trifocal.rhs = a3 * b4.transpose() - a4 * b3.transpose();

  Mat34 recovery_P1, recovery_P2, recovery_P3;
  RecoveryCameraMatrix(true_trifocal, recovery_P1, recovery_P2, recovery_P3);
  std::cout << "Ground True : " << P2.col(3).hnormalized() << std::endl << " Estimate : " << recovery_P2.col(3).hnormalized() << std::endl;

  std::vector<TripleMatch> data_set;
  std::vector<TripleMatch> noised_data_set;
  
  std::normal_distribution<double> normal_distribution(0.0, sigma);

  for (auto i : Range(data_size)) {
    Eigen::Vector3d X (uniform_distribution(rd), uniform_distribution(rd), 10.0 + uniform_distribution(rd));
    TripleMatch data_point;
    TripleMatch noised_data_point;
    data_point.I_observation = (P1 * X.homogeneous()).hnormalized();
    data_point.J_observation = (P2 * X.homogeneous()).hnormalized();
    data_point.K_observation = (P3 * X.homogeneous()).hnormalized();

    noised_data_point.I_observation =
        data_point.I_observation +
        Eigen::Vector2d(normal_distribution(rd), normal_distribution(rd));
    noised_data_point.J_observation =
        data_point.J_observation +
        Eigen::Vector2d(normal_distribution(rd), normal_distribution(rd));
    noised_data_point.K_observation =
        data_point.K_observation +
        Eigen::Vector2d(normal_distribution(rd), normal_distribution(rd));
    data_set.push_back(data_point);
    noised_data_set.push_back(noised_data_point);
  }

  return {noised_data_set, data_set, P2, P3};
}


TEST(Trifocal, LinearSolver) {
  size_t data_size = 7;
  double error_sigma = 0.05;
  auto [noised_data_set, data_set, true_P2, true_P3] = GenerateDataSet(data_size, error_sigma);

  Trifocal model;
  LinearSolver::Fit(noised_data_set, &model);

  for (auto index : Range(data_set.size())) {
    TripleMatch noised_data_point = noised_data_set[index];
    TripleMatch ground_data_point = data_set[index];
    double error = TrifocalError::Error(noised_data_point, model);
    EXPECT_GT(error, 0.0);
    EXPECT_LT(error, 12.592 * error_sigma * error_sigma);
  }
  Mat34 P1, P2, P3;
  RecoveryCameraMatrix(model, P1, P2, P3);
  std::cout << true_P2 << std::endl;
  std::cout  << P2 << std::endl;
    std::cout << true_P2.col(3).hnormalized() << std::endl;
    std::cout << P2.col(3).hnormalized() << std::endl;
  double norm = (true_P2.col(3).hnormalized() - P2.col(3).hnormalized()).norm();
  EXPECT_LT(norm, true_P2.col(3).hnormalized().norm() * 0.1);
}

TEST(Trifocal, NormalizedLinearSolver) {
  size_t data_size = 7;
  double error_sigma = 0.01;
  auto [noised_data_set, data_set, true_P2, true_P3] = GenerateDataSet(data_size, error_sigma);

  Trifocal model;
  NormalizedLinearSolver::Fit(noised_data_set, &model);

  // measure_error + estimate_error >= ground_true
  // but not greater a lot
  for (auto index : Range(data_set.size())) {
    TripleMatch noised_data_point = noised_data_set[index];
    TripleMatch ground_data_point = data_set[index];
    double error = TrifocalError::Error(noised_data_point, model);
    EXPECT_GT(error, 0.0);
    EXPECT_LT(error, 12.592 * error_sigma * error_sigma);
  }
    Mat34 P1, P2, P3;
    RecoveryCameraMatrix(model, P1, P2, P3);
    std::cout << true_P2 << std::endl;
    std::cout  << P2 << std::endl;
    std::cout << true_P2.col(3).hnormalized() << std::endl;
    std::cout << P2.col(3).hnormalized() << std::endl;
  double norm = (true_P2.col(3).hnormalized() - P2.col(3).hnormalized()).norm();
  EXPECT_LT(norm, true_P2.col(3).hnormalized().norm() * 0.1);
}
/*
TEST(Trifocal, AlgebraMinimumSolver) {

  size_t data_size = 32;
  double error_sigma = 0.01;
  auto [noised_data_set, data_set] = GenerateDataSet(data_size, error_sigma);

  Trifocal model;
  AlgebraMinimumSolver::Fit(data_set, &model);

  double measure_error = 0.0;
  double estimate_error = 0.0;
  double ground_true_error = 0.0;
  // measure_error + estimate_error >= ground_true
  // but not greater a lot
  for (auto index : Range(data_set.size())) {
    TripleMatch noised_data_point = noised_data_set[index];
    TripleMatch ground_data_point = data_set[index];

    Eigen::Matrix<double, 6, 1> measure_vector, ground_true_vector;
    measure_vector << noised_data_point.I_observation, noised_data_point.J_observation, noised_data_point.K_observation;
    ground_true_vector << ground_data_point.I_observation, ground_data_point.J_observation, ground_data_point.K_observation;
    Eigen::VectorXd estimate_vector = measure_vector + TrifocalSampsonError::CorrectedPoint(noised_data_point, model);
    ground_true_error += (ground_true_vector - measure_vector).squaredNorm();
    estimate_error += (estimate_vector - ground_true_vector).squaredNorm();
    measure_error += (measure_vector - estimate_vector).squaredNorm();
    EXPECT_LT(TrifocalSampsonError::Error(noised_data_point, model), 12.592 * error_sigma * error_sigma);
  }
  EXPECT_GT(estimate_error + measure_error, ground_true_error);
  EXPECT_NEAR(ground_true_error, estimate_error + measure_error, 225.329 * error_sigma * error_sigma);
}
*/

TEST(Trifocal, BundleRefineSolver) {

  size_t data_size = 7;
  double error_sigma = 0.05;
  auto [noised_data_set, data_set, true_P2, true_P3] = GenerateDataSet(data_size, error_sigma);

  Trifocal model;
  BundleRefineSolver::Fit(noised_data_set, &model);

  // measure_error + estimate_error >= ground_true
  // but not greater a lot
  for (auto index : Range(data_set.size())) {
    TripleMatch noised_data_point = noised_data_set[index];
    TripleMatch ground_data_point = data_set[index];

    double error = TrifocalError::Error(noised_data_point, model);
    EXPECT_GT(error, 0.0);
    EXPECT_LT(error, 12.592 * error_sigma * error_sigma);
  }

  for (auto index : Range(data_size)) {
    TripleMatch outlier;
    outlier.I_observation.setRandom();
    outlier.J_observation.setRandom();
    outlier.K_observation.setRandom();
    double error  = TrifocalError::Error(outlier, model);
    EXPECT_GT(error, 12.592 * error_sigma * error_sigma);
  }

    Mat34 P1, P2, P3;
    RecoveryCameraMatrix(model, P1, P2, P3);
    std::cout << true_P2 << std::endl;
    std::cout  << P2 << std::endl;
    std::cout << true_P2.col(3).hnormalized() << std::endl;
    std::cout << P2.col(3).hnormalized() << std::endl;
  double norm = (true_P2.col(3).hnormalized() - P2.col(3).hnormalized()).norm();
  EXPECT_LT(norm, true_P2.col(3).hnormalized().norm() * 0.1);
}

TEST(Trifocal, RansacTrifocalSolver) {

  size_t data_size = 100;
  double outlier_rate = 0.4;
  double error_sigma = 100.0;

  auto [noised_data_set, ground_data_set, true_P2, true_P3] = GenerateDataSet(data_size, error_sigma);

  size_t inlier_size = 60;
  size_t outlier_size = inlier_size / (1.0 - outlier_rate) * outlier_rate;

  std::vector<TripleMatch> data_set;
  std::vector<TripleMatch> inlier_set(inlier_size);
  std::vector<TripleMatch> outlier_set(outlier_size);

  for (auto i : Range(data_size)) {
    TripleMatch data_point;
    if (i < data_size * ( 1 - outlier_rate)) {
      data_point = ground_data_set[i];
      inlier_set.push_back(data_point);
    } else {
      data_point = noised_data_set[i];
      outlier_set.push_back(data_point); 
    }
    data_set.push_back(data_point);
  }

  Trifocal model;
  RansacTrifocalSolver::Fit(data_set, model);
  std::cout << model << std::endl;
  for (auto data_point :inlier_set) {
    double temp = TrifocalError::Error(data_point, model);
    ASSERT_FALSE(std::isnan(temp));
    EXPECT_GT(temp, 0.0);
    EXPECT_LT(temp,12.59 * 1 * 1);
  }

  for (auto outlier : outlier_set) {
    double temp = TrifocalError::Error(outlier, model);
    ASSERT_FALSE(std::isnan(temp));
    EXPECT_GT(temp,12.59 * 1 * 1);
  }
    Mat34 P1, P2, P3;
    RecoveryCameraMatrix(model, P1, P2, P3);
    std::cout << true_P2 << std::endl;
    std::cout  << P2 << std::endl;
    std::cout << true_P2.col(3).hnormalized() << std::endl;
    std::cout << P2.col(3).hnormalized() << std::endl;
  double norm = (true_P2.col(3).hnormalized() - P2.col(3).hnormalized()).norm();
  EXPECT_LT(norm, true_P2.col(3).hnormalized().norm() * 0.1);
}

int main(int argc, char**argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}