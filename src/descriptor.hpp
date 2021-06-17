//
// Created by junlinp on 2019-06-09.
//

#ifndef SFM_SRC_DESCRIPTOR_HPP_
#define SFM_SRC_DESCRIPTOR_HPP_
#include <memory>
#include <cmath>
#include <cstring>

class Descriptor
{

private:
    float *vec;
    size_t _size;

public:
    explicit Descriptor(const float *input, int n)
    {
        vec = new float[n];
        memcpy(vec, input, n * sizeof(float));
        _size = n;
    }

    size_t size() {
      return this->_size;
    }

    float* data() {
      return vec;
    }
    double distance(std::shared_ptr<Descriptor> des) {
      double res = 0.0;
      for (int i = 0; i < _size; i++) {
        double v = (this->vec[i] - des->vec[i]);
        res += sqrt(v * v);
      }
      return res;
    }

    double distance(const Descriptor& des) {
      double res = 0.0;
      for (int i = 0; i < _size; i++) {
        double v = (this->vec[i] - des.vec[i]);
        res += sqrt(v * v);
      }
      return res;
    }
};

#endif //SFM_SRC_DESCRIPTOR_HPP_
