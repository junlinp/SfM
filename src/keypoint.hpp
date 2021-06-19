//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_KEYPOINT_HPP_
#define SFM_SRC_KEYPOINT_HPP_

struct KeyPoint {
  double x, y;

  KeyPoint() = default;
  KeyPoint(double x, double y) : x(x), y(y){}
};
#endif //SFM_SRC_KEYPOINT_HPP_
