//
// Created by junlinp on 2019-05-21.
//

#include "feature_extractor.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "vector"
#include "iostream"
bool feature_extractor::extract(cv::Mat image, std::vector<cv::KeyPoint> &kp, cv::Mat &descriptor) {
  cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(
      1 << 17,
      4);

  if (!image.data) {
    fprintf(stderr, "Image is NULL\n");
    return false;
  }
  sift->detectAndCompute(image, cv::Mat(), kp, descriptor);
  return true;
}
std::vector<cv::DMatch> feature_extractor::feature_match(const cv::Mat &descriptor_lhs, const cv::Mat &descriptor_rhs) {
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher->knnMatch(descriptor_lhs, descriptor_rhs, knn_matches, 2);
  std::vector<cv::DMatch> good;

  for(auto& dmatch : knn_matches) {
    //std::cout << dmatch[0].distance << " : " << dmatch[1].distance << std::endl;
     if (dmatch[0].distance < 0.8 * dmatch[1].distance) {
       good.emplace_back(dmatch[0]);
     }
  }
  return good;
}
