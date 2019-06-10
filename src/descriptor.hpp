//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_DESCRIPTOR_HPP_
#define SFM_SRC_DESCRIPTOR_HPP_
#include "opencv4/opencv2/core.hpp"


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
};

#endif //SFM_SRC_DESCRIPTOR_HPP_
