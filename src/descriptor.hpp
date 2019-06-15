//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_DESCRIPTOR_HPP_
#define SFM_SRC_DESCRIPTOR_HPP_
#include "opencv2/core.hpp"
#include <memory>

class descriptor
{

private:
    float *vec;
    size_t _size;

public:
    explicit descriptor(const float *input, int n)
    {
        vec = new float[n];
        memcpy(vec, input, n * sizeof(float));
        _size = n;
    }

    explicit descriptor(const cv::Mat &cvMat)
    {
        _size = cvMat.cols;
        vec = new float[_size];

        for (int i = 0; i < _size; i++)
        {
            vec[i] = cvMat.at<float>(0, i);
        }
    }

    size_t size() {
      return this->_size;
    }

    float* data() {
      return vec;
    }
    double distance(std::shared_ptr<descriptor> des) {
      double res = 0.0;
      for (int i = 0; i < _size; i++) {
        double v = (this->vec[i] - des->vec[i]);
        res += sqrt(v * v);
      }
      return res;
    }

    double distance(const descriptor& des) {
      double res = 0.0;
      for (int i = 0; i < _size; i++) {
        double v = (this->vec[i] - des.vec[i]);
        res += sqrt(v * v);
      }
      return res;
    }
};

#endif //SFM_SRC_DESCRIPTOR_HPP_
