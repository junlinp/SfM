//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_UNITTEST_HPP_
#define SFM_SRC_UNITTEST_HPP_
#include <gtest/gtest.h>

#include <random>

#include "internal/thread_pool.hpp"
#include "ransac.hpp"

TEST(ThreadPool, Enqueue) {
  auto functor = [](int a) { return 2 * a; };
  ThreadPool threadpool;
  for (int i = 0; i < 1024; i++) {
    std::future<int> res = threadpool.Enqueue(functor, i);
    EXPECT_EQ(2 * i, res.get());
  }
}
class LineFit {
 private:
  double a_, b_;

 public:
  using DataPointType = std::pair<double, double>;
  using ModelType = double*;
  static size_t minimum_data_point = 2;


  LineFit(double a, double b) : a_(a), b_(b) {}

  double Error(std::pair<double, double> data_point,const ModelType& model) {
      double e = data_point.second - (model[0] * data_point.first - model[1]);
      return e * e;
  }

  double Fit(const std::vector<DataPointType>& samples, ModelType& model) {

  }
};
TEST(Ransac, Fit_Line) {
  using DataType = std::pair<double, double>;
  double sigma = 0.2;
  std::default_random_engine engine;
  std::normal_distribution<double> normal_distribution;
  std::uniform_real_distribution<double> uniform_distribution;
  const size_t DataPointNum = 1024;
  std::vector<DataType> data_points;
  double a = 2.3, b = 0.9;
  for (int i = 0; i < DataPointNum; i++) {
    double x = uniform_distribution(engine);
    double y = normal_distribution(engine) + (a * x + b);
    data_points.emplace_back(x, y);
  }
}

#endif  // SFM_SRC_UNITTEST_HPP_
