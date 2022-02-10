//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_UNITTEST_HPP_
#define SFM_SRC_UNITTEST_HPP_
#include <random>
#include <vector>

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "internal/thread_pool.hpp"
#include "ransac.hpp"
#include "solver/algebra.hpp"
#include "solver/fundamental_solver.hpp"
#include "solver/triangular_solver.hpp"
#include "solver/trifocal_tensor_solver.hpp"
#include "solver/self_calibration_solver.hpp"

#include "solver/triangular_solver_test.hpp"
TEST(ThreadPool, Enqueue) {
  auto functor = [](int a) { return 2 * a; };
  ThreadPool threadpool;
  for (int i = 0; i < 1024; i++) {
    std::future<int> res = threadpool.Enqueue(functor, i);
    EXPECT_EQ(2 * i, res.get());
  }
}
/*
struct model {
  double a, b;
};
class LineFit {
 private:
  double a_, b_;

 public:
  using DataPointType = std::pair<double, double>;
  using MODEL_TYPE = model;
  static const size_t MINIMUM_DATA_POINT = 2;
  static const size_t MODEL_NUMBER = 1;
  static const size_t MODEL_FREEDOM = 1;
  using sample_type = DataPointType;

  LineFit(double a, double b) : a_(a), b_(b) {}

  static double Error(std::pair<double, double> data_point,
                      const MODEL_TYPE& model) {
    double e = data_point.second - (model.a * data_point.first + model.b);
    return e * e;
  }

  static double Fit(const std::vector<DataPointType>& samples,
                    MODEL_TYPE* model) {
    // y = a * x + b => x1 * a + b = y1
    //  x2 * a + b = y2
    // (x1 - x2) * a = (y1 - y2)
    // a = (y1 - y2) / (x1 - x2)
    // b = y1 - (y1 - y2) / (x1 - x2) * x1
    // LSM
    // A*x = b
    // m > n
    // so (A^T * A) * x = A^T * b
    // x = (A^T * A)^(-1) * A^T * b
    // so error = || b - A * x||^2
    Eigen::MatrixXd A(samples.size(), 2);
    Eigen::VectorXd b(samples.size());
    for (int i = 0; i < samples.size(); i++) {
      A(i, 0) = samples[i].first;
      A(i, 1) = 1;
      b(i) = samples[i].second;
    }

    Eigen::VectorXd x = (A.transpose() * A).inverse() * A.transpose() * b;

    model[0].a = x(0);
    model[0].b = x(1);
    std::printf("Model[0]  = %lf, Model[1] = %lf\n", model[0].a, model[0].b);
    return 0.0;
  }
};

TEST(Ransac, Fit_Line) {
  using DataType = std::pair<double, double>;
  double sigma = 0.001;
  std::default_random_engine engine;
  std::normal_distribution<double> normal_distribution(0, sigma);
  std::uniform_real_distribution<double> uniform_distribution;
  const size_t DataPointNum = 1024;
  std::vector<DataType> data_points;
  double a = 2.3, b = 0.9;
  for (int i = 0; i < DataPointNum; i++) {
    double x = uniform_distribution(engine);
    double y = normal_distribution(engine) + (a * x + b);
    data_points.emplace_back(x, y);
  }

  Ransac<LineFit> ransac;
  std::vector<size_t> inlier_indexs;
  LineFit::MODEL_TYPE models;
  ransac.Inference(data_points, inlier_indexs, &models);
  std::printf("a = %lf, b = %lf\n", models.a, models.b);
  std::printf("Done\n");
  EXPECT_NEAR(models.a, a, 1e-2);
  EXPECT_NEAR(models.b, b, 1e-2);
}

//     | 1  1  0 |
// F = | 0  1  0 |
//     | 0  0  0 |

TEST(Solver, Fundamental_Solver) {
  EightPointFundamentalSolver solver;
  std::vector<KeyPoint> rhs_key_points = {{0, 0}, {1, 0}, {2, 0}, {0, 1},
                                          {1, 1}, {2, 1}, {1, 2}, {2, 2}};

  std::vector<KeyPoint> lhs_key_points = {{1, 1},  {1, -1}, {-1, 1}, {1, 0},
                                          {-2, 1}, {-3, 2}, {3, -1}, {-4, 2}};

  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (int i = 0; i < lhs_key_points.size(); i++) {
    datas.push_back({lhs_key_points[i], rhs_key_points[i]});
  }

  Eigen::Matrix3d F;
  solver.Fit(datas, &F);
  // F << 1.0, 1.0, 0.0,
  //     0.0, 1.0, 0.0,
  //     0.0, 0.0, 0.0;
  std::cout << "F : " << F << std::endl;
  Eigen::Matrix3d true_f;
  true_f << 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  for (auto data_point : datas) {
    std::cout << " Error : " << solver.Error(data_point, F) << std::endl;
    std::cout << " true Error : " << solver.Error(data_point, true_f)
              << std::endl;
  }
}
TEST(Solver, Fundamental_Solver_ERROR_Estimate) {
  EightPointFundamentalSolver solver;
  std::vector<KeyPoint> rhs_key_points = {{5, 10}, {1, 0}, {2, 0}, {0, 1},
                                          {1, 1},  {2, 1}, {1, 2}, {2, 2}};

  std::vector<KeyPoint> lhs_key_points = {{10, 5}, {1, -1}, {-1, 1}, {1, 0},
                                          {-2, 1}, {-3, 2}, {3, -1}, {-4, 2}};

  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (int i = 0; i < lhs_key_points.size(); i++) {
    datas.push_back({lhs_key_points[i], rhs_key_points[i]});
  }

  Eigen::Matrix3d F;
  solver.Fit(datas, &F);
  // F << 1.0, 1.0, 0.0,
  //     0.0, 1.0, 0.0,
  //     0.0, 0.0, 0.0;
  std::cout << "F : " << F << std::endl;
  F = F * 1000;
  for (auto data_point : datas) {
    std::cout << " Error : " << solver.Error(data_point, F) << std::endl;
  }
}

TEST(Solver, Fundamental_Solver_With_Ransac) {
  std::vector<KeyPoint> rhs_key_points = {{5, 10},  {0, 0},   {10, 0},
                                          {20, 0},  {0, 10},  {10, 10},
                                          {20, 10}, {10, 20}, {20, 20}};

  std::vector<KeyPoint> lhs_key_points = {{10, 5},   {10, 10},  {10, -10},
                                          {-10, 10}, {10, 0},   {-20, 10},
                                          {-30, 20}, {30, -10}, {-40, 20}};

  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (int i = 0; i < lhs_key_points.size(); i++) {
    datas.push_back({lhs_key_points[i], rhs_key_points[i]});
  }
  Ransac<EightPointFundamentalSolver> ransac_solver;
  std::vector<size_t> inlier_index;
  Eigen::Matrix3d F;
  ransac_solver.Inference(datas, inlier_index, &F);

  std::cout << "F : " << F << std::endl;

  EightPointFundamentalSolver solver;
  for (auto data_point : datas) {
    std::cout << " Error : " << solver.Error(data_point, F) << std::endl;
  }
}
*/

#include <random>
// f = 35 mm  1920 * 1080  sense_size 36.0 * 23.9 mm
// cx = 1920 / 2 = 960
// cy = 1080 / 2 = 540
// dx = 36.0 / 1920 = 0.01875
// dy = 23.9 / 1080 = 0.02212963
// fx = f / dx = 1866.6667
// fy = f / dy = 1581.58996
struct Scene {
  using Point = Eigen::Vector3d;
  using Observation = Eigen::Vector2d;
  using Mat34 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
  Eigen::Matrix3d K;

  std::vector<Mat34, Eigen::aligned_allocator<Mat34>> Ps;
  std::vector<Point, Eigen::aligned_allocator<Point>> points;
  using AlignedObservation =
      std::vector<Observation, Eigen::aligned_allocator<Observation>>;
  std::map<size_t, AlignedObservation> observations;
  std::map<size_t, AlignedObservation> noised_observations;

  double error_sigma;
  Scene(size_t cameras_num = 2, size_t point_num = 10, double error_sigma = 1.0)
      : error_sigma(error_sigma) {
    K << 1866.666667, 0.0, 960, 0.0, 1581.58996, 540, 0.0, 0.0, 1.0;
    std::random_device rd;
    std::default_random_engine engine(rd());
    std::uniform_real_distribution<double> hundred_distribution(-100.0, 100.0);
    std::uniform_real_distribution<double> two_hundred_distribution(-200.0,
                                                                    200.0);

    for (int i = 0; i < point_num; i++) {
      points.emplace_back(hundred_distribution(engine),
                          hundred_distribution(engine),
                          hundred_distribution(engine));
    }

    for (int i = 0; i < cameras_num; i++) {
      // Init P matrix
      Eigen::Vector3d C(two_hundred_distribution(engine),
                        two_hundred_distribution(engine),
                        two_hundred_distribution(engine));
      Eigen::Vector3d a3 = -C;
      a3.normalize();
      Eigen::Vector3d a1;
      a1.setRandom();
      a1 = a1 - a1.dot(a3) * a3;
      a1.normalize();

      Eigen::Vector3d a2;
      a2.setRandom();
      a2 = a2 - a2.dot(a1) * a1 - a2.dot(a3) * a3;
      a2.normalize();
      Eigen::Matrix3d R;
      R << a1.transpose(), a2.transpose(), a3.transpose();
      if ((K * R).determinant() < 0) {
        R.row(0) *= -1.0;
      }
      // R = R.transpose();
      // std::cout << "a1 : " << a1.transpose() << std::endl;
      // std::cout << "a2 : " << a2.transpose() << std::endl;
      // std::cout << "a3 : " << a3.transpose() << std::endl;
      // std::cout << "R : " << R << std::endl;
      // std::cout << "a1 dot C : " << a1.dot(C) << std::endl;
      // std::cout << "a2 dot C : " << a2.dot(C) << std::endl;
      // std::cout << "a3 dot C : " << a3.dot(C) << std::endl;
      // std::cout << a1.dot(a3) << std::endl;
      // std::cout << "R * C : " << R * C << std::endl;

      // std::cout << R * R.transpose() << std::endl;

      // P = [M | t]
      // v = det(M) * 3rd_row_of(M)
      // the vector v is directed start with the center of camera along the
      // z-axis if the camera stand on C and point to the origin. so v = lambda
      // * C where lambda < 0 this is the way to compute Rotation

      Eigen::Matrix<double, 3, 4> P;
      P << R, -R * C;
      // std::cout << "P : " << P << std::endl;
      P = K * P;

      // std::cout << (P * (-1.1 * C).homogeneous()).hnormalized() << std::endl;
      Ps.push_back(P);
    }

    int count = 0;
    std::normal_distribution<double> normal_distribution(0.0, error_sigma);
    for (auto& P : Ps) {
      for (Point& p : points) {
        Eigen::Vector3d ob = P * p.homogeneous();
        Eigen::Vector2d ob_h = ob.hnormalized();
        observations[count].emplace_back(ob_h);
        // std::cout << ob.hnormalized() << std::endl;
        // add noised
        double noised_x = normal_distribution(engine),
               noised_y = normal_distribution(engine);
        Eigen::Vector2d n(ob_h.x() + noised_x, ob_h.y() + noised_y);
        noised_observations[count].emplace_back(n);
      }
      count++;
    }
  }

  template <typename FundamentalSolver>
  void FundamentalMatrixCompute(Mat33& F) {
    using T = std::pair<Observation, Observation>;
    std::vector<std::pair<Observation, Observation>>
        data_points;
    data_points.resize(noised_observations[0].size());
    for (int i = 0; i < data_points.size(); i++) {
      data_points[i].first = noised_observations[0][i];
      data_points[i].second = noised_observations[1][i];
    }
    FundamentalSolver solver;
    solver.Fit(data_points, F);
  }

  template <typename ErrorEstimator>
  double FundamentalResiduals(const Eigen::Matrix3d& F) {
    double res = 0.0;
    std::vector<std::pair<Observation, Observation>> data_points;
    size_t size = observations[0].size();
    for (int i = 0; i < size; i++) {
      data_points.emplace_back(observations[0][i], observations[1][i]);
    }

    for (auto& data_point : data_points) {
      res += ErrorEstimator::Error(data_point, F);
    }
    size_t d = 8 + 3 * data_points.size();
    size_t n = data_points.size() * 4;
    double error = error_sigma * std::sqrt(double(d) / n);
    std::cout << " Error Lower bound : " << error << std::endl;
    return std::sqrt(res / data_points.size() / 4.0);
  }

  template <typename Functor>
  double TriangularError(Functor&& functor) {
    std::vector<Mat34> p_matrixs;
    for (auto i : Ps) {
      p_matrixs.push_back(i);
    }
    double error = 0.0;
    for (int j = 0; j < observations[0].size(); j++) {
      std::vector<Eigen::Vector3d> obs;
      for (int i = 0; i < Ps.size(); i++) {
        obs.push_back(observations[i][j].homogeneous());
      }

      Eigen::Vector4d X;
      functor(p_matrixs, obs, X);
      Point p = points[j];
      X = X / X(3);
      // std::cout << "X : " << X << " With : " << p << std::endl;
      // std::cout << "P * X : " << p_matrixs[0] * X << std::endl;
      // std::cout << "P * p : " << p_matrixs[0] * p.homogeneous() << std::endl;
      error += (X - p.homogeneous()).norm();
    }

    return error / noised_observations[0].size();
  }

  double TrifocalError() {
    std::vector<TripleMatch> data_points;
    for (int i = 0; i < 7; i++) {
      TripleMatch t = {0, observations[0][i], 1, observations[1][i], 2, observations[2][i]};
      data_points.push_back(t);
    }

    LinearSolver solver;
    Trifocal model;
    solver.Fit(data_points, &model);

    Trifocal model2;
    BundleRefineSolver bundle_solver;
    bundle_solver.Fit(data_points, &model2);

    // std::cout << model << std::endl;

    double error = 0.0;
    double geometry_error = 0.0;
    for (TripleMatch data_point : data_points) {
      error += TrifocalError::Error(data_point, model);
      geometry_error += GeometryError(data_point, model2);
    }
    std::cout << "Geometry Error : " << geometry_error / data_points.size()
              << std::endl;
    return error / data_points.size();
  }

  void SelfCalibration() {
    Eigen::Matrix4d H;
    H.setRandom();
    std::vector<Mat34> P;
    std::copy(Ps.begin(), Ps.end(), std::back_inserter(P));
    for(Mat34& iter : P) {
      iter = iter * H;
    }
    Eigen::Matrix4d Q = IAC(P,1920, 1080);
    std::cout << "Q : " << Q << std::endl;

    Eigen::Matrix3d dual_omega = P[0] * Q * P[0].transpose();
    std::cout << "dual_omega : " << dual_omega << std::endl;
    Eigen::Matrix3d K = RecoveryK(dual_omega, 1920, 1080);
    std::cout << K << std::endl;
    // cx = 1920 / 2 = 960
    // cy = 1080 / 2 = 540
    // dx = 36.0 / 1920 = 0.01875
    // dy = 23.9 / 1080 = 0.02212963
    // fx = f / dx = 1866.6667
    // fy = f / dy = 1581.58996
    Eigen::Matrix3d true_K;
    true_K << 1866.6667, 0.0, 960,
                0.0,  1581.58996, 540,
                0.0, 0.0, 1.0;
    std::cout << "K * K^T : " << true_K * true_K.transpose() << std::endl;
  }
};

TEST(Fundamental, Performance) {
  double res = 0.0;
  size_t test_case = 1;
  for (int i = 0; i < test_case; i++) {
    Scene scene(2, 1024 * 16);
    // We need Solver
    // We need a Error Estimator
    Mat33 F;
    scene.FundamentalMatrixCompute<EightPointFundamentalSolver>(F);
    // std::cout << F << std::endl;
    res += scene.FundamentalResiduals<SampsonError>(F);
  }
  std::cout << res / test_case << std::endl;
}

TEST(Triangular, Performance_Two_Camera) {
  Scene scene(2, 2);
  std::cout << scene.TriangularError(BundleAdjustmentTriangular) << std::endl;
  std::cout << scene.TriangularError(DLT) << std::endl;
}

TEST(Triangular, Performance_Three_Camera) {
  Scene scene(3, 2);
  std::cout << scene.TriangularError(BundleAdjustmentTriangular) << std::endl;
  std::cout << scene.TriangularError(DLT) << std::endl;
}

TEST(Triangular, Performance_Four_Camera) {
  Scene scene(4, 2);
  std::cout << scene.TriangularError(BundleAdjustmentTriangular) << std::endl;
  std::cout << scene.TriangularError(DLT) << std::endl;
}

TEST(Trifocal, Test) {
  Scene scene(3, 7);

  std::cout << scene.TrifocalError() << std::endl;
}

TEST(Self_Calibration, Correct) {
  Scene scene(10, 7);

  scene.SelfCalibration();
}
#endif  // SFM_SRC_UNITTEST_HPP_
