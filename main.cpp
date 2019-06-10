#include <iostream>
#include <gtest/gtest.h>
#include "src/feature_extractor.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "ceres/rotation.h"
#include "ceres/Problem.h"
#include "ceres/ceres.h"
#include "unittest.hpp"

struct Cost {

  bool operator()(const double* p, const double* t, const double* point, const double* uv, const double* camera, double* residual) const {
    double pt[3];
    ceres::QuaternionRotatePoint(p, point, pt);
    pt[0] += t[0];
    pt[1] += t[1];
    pt[2] += t[2];
    double f = camera[0];
    double cx = camera[1];
    double cy = camera[2];
    residual[0] = f * pt[0] / pt[2] - cx - uv[0];
    residual[1] = f * pt[1] / pt[2] - cx - uv[1];
    return true;
  }
};
int main(int argc, char** argv) {
  /*
  auto image1 = cv::imread("Img/100_7100.JPG");
  auto image2 = cv::imread("Img/100_7101.JPG");

  feature_extractor ft;
  std::vector<cv::KeyPoint> kp1;
  std::vector<cv::KeyPoint> kp2;
  cv::Mat ds1;
  cv::Mat ds2;
  ft.extract(image1, kp1, ds1);
  std::cout << kp1.size() << std::endl;
  ft.extract(image2, kp2, ds2);
  std::cout << kp2.size() << std::endl;
  auto dmatch = ft.feature_match(ds1, ds2);
  std::cout << dmatch.size() << std::endl;

  double *points = new double[3 * dmatch.size()];
  double *uv1 = new double[2 * dmatch.size()];
  double *uv2 = new double[2 * dmatch.size()];
  double pvec1[4] = {1.0, 0.0, 0.0, 0.0};
  double pvec2[4] = {1.0, 0.0, 0.0, 0.0};
  double tvec1[3] = {0.0, 0.0, 0.0};
  double tvec2[3] = {0.0, 0.0, 0.0};
  double camera1[3] = {1.0, 0.0, 0.0};
  double camera2[3] = {1.0, 0.0, 0.0};

  ceres::Problem problem;
  ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
  for(int i = 0; i < dmatch.size(); i++) {
    points[3 * i + 0] = points[3 * i + 1] = points[3 * i + 2] = 1.0;
    uv1[2 * i + 0] = kp1[dmatch[i].queryIdx].pt.x;
    uv1[2 * i + 1] = kp1[dmatch[i].queryIdx].pt.y;
    uv2[2 * i + 0] = kp2[dmatch[i].trainIdx].pt.x;
    uv2[2 * i + 1] = kp2[dmatch[i].trainIdx].pt.y;

    ceres::CostFunction* cost1 =
        new ceres::NumericDiffCostFunction<Cost, ceres::CENTRAL, 2,4,3,3,2, 3>(new Cost);
    ceres::CostFunction* cost2 =
        new ceres::NumericDiffCostFunction<Cost, ceres::CENTRAL, 2,4,3,3,2, 3>(new Cost);
    problem.AddResidualBlock(cost1, loss_function,pvec1, tvec1, points + 3 * i, uv1 + 2 * i, camera1);
    problem.AddResidualBlock(cost2, loss_function,pvec2, tvec2, points + 3 * i, uv2 + 2 * i, camera2);
  }

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  */
  std::shared_ptr<database> ptr_database = std::make_shared<database>("database");
  bool rc = ptr_database->create();
  std::shared_ptr<image> img = std::make_shared<image>();
  rc = img->load_image("Img/100_7100.JPG");
  rc = ptr_database->insert_image(img->get_name(), img->get_keypoints(), img->get_descriptors(), simple_pinhole_camera_model());

  rc = img->load_image("Img/100_7101.JPG");
  rc = ptr_database->insert_image(img->get_name(), img->get_keypoints(), img->get_descriptors(), simple_pinhole_camera_model());

  rc = img->load_image("Img/100_7102.JPG");
  rc = ptr_database->insert_image(img->get_name(), img->get_keypoints(), img->get_descriptors(), simple_pinhole_camera_model());

  rc = img->load_image("Img/100_7103.JPG");
  rc = ptr_database->insert_image(img->get_name(), img->get_keypoints(), img->get_descriptors(), simple_pinhole_camera_model());
  rc = img->load_image("Img/100_7104.JPG");
  rc = ptr_database->insert_image(img->get_name(), img->get_keypoints(), img->get_descriptors(), simple_pinhole_camera_model());
  rc = img->load_image("Img/100_7105.JPG");
  rc = ptr_database->insert_image(img->get_name(), img->get_keypoints(), img->get_descriptors(), simple_pinhole_camera_model());
  rc = img->load_image("Img/100_7106.JPG");
  rc = ptr_database->insert_image(img->get_name(), img->get_keypoints(), img->get_descriptors(), simple_pinhole_camera_model());
  rc = img->load_image("Img/100_7107.JPG");
  rc = ptr_database->insert_image(img->get_name(), img->get_keypoints(), img->get_descriptors(), simple_pinhole_camera_model());

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}