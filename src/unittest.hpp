//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_UNITTEST_HPP_
#define SFM_SRC_UNITTEST_HPP_
#include "gtest/gtest.h"

#include <random>

#include "internal/thread_pool.hpp"
#include "ransac.hpp"

#include "Eigen/Dense"

TEST(ThreadPool, Enqueue) {
  auto functor = [](int a) { return 2 * a; };
  ThreadPool threadpool;
  for (int i = 0; i < 1024; i++) {
    std::future<int> res = threadpool.Enqueue(functor, i);
    EXPECT_EQ(2 * i, res.get());
  }
}

struct model {
  double a, b;
};
class LineFit {
 private:
  double a_, b_;

 public:
  using DataPointType = std::pair<double, double>;
  using model_type = model;
  static const size_t minimum_data_point = 2;
  static const size_t model_number = 1;
  static const size_t model_freedom = 1;
  using sample_type = DataPointType;

  LineFit(double a, double b) : a_(a), b_(b) {}

  static double Error(std::pair<double, double> data_point,const model_type& model) {
      double e = data_point.second - (model.a * data_point.first + model.b);
      return e * e;
  }

  static double Fit(const std::vector<DataPointType>& samples, model_type* model) {
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
    for(int i = 0; i < samples.size(); i++) {
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
  LineFit::model_type models;
  ransac.Inference(data_points, inlier_indexs, &models);
  std::printf("a = %lf, b = %lf\n", models.a, models.b);
  std::printf("Done\n");
  EXPECT_NEAR(models.a, a, 1e-2);
  EXPECT_NEAR(models.b, b, 1e-2);
}

#endif  // SFM_SRC_UNITTEST_HPP_
