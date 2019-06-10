//
// Created by junlinp on 2019-05-21.
//

#ifndef SFM_SRC_FEATURE_EXTRACTOR_HPP_
#define SFM_SRC_FEATURE_EXTRACTOR_HPP_
#include "opencv2/core.hpp"

class feature_extractor {

 public:
  bool extract(cv::Mat image, std::vector<cv::KeyPoint> &kp, cv::Mat &descriptor);
  std::vector<cv::DMatch> feature_match(const cv::Mat &descriptor_lhs,
                                                                  const cv::Mat &descriptor_rhs
  );
};



#endif //SFM_SRC_FEATURE_EXTRACTOR_HPP_
