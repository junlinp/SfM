#ifndef SRC_METRIC_HPP_
#define SRC_METRIC_HPP_

#include <cmath>
#include "immintrin.h"
#include <bitset>
struct support_avx {
    #ifdef __AVX2__
    static const bool value = true;
    #else
    static const bool value = false;
    #endif
};

inline float L2_metric_trivial(const float* lhs,const float* rhs) {
  float sum = 0.0f;
  for (int i = 0; i < 128; i++) {
    float temp = lhs[i] - rhs[i];
    sum += temp * temp;
  }
  return std::sqrt(sum);
}
/*
float L2_metric_avx(float* lhs, float* rhs) {
  __m256 lhs_256 = _mm256_load_ps(lhs);
  __m256 rhs_256 = _mm256_load_ps(rhs);
  // 8 float per epoch
  // 128 / 8 = 32
  __m256 res = _mm256_setzero_ps();
  for (int i = 0; i < 32; i++) {
    __m256 sub = _mm256_sub_ps(lhs_256, rhs_256);
    __m256 mul = _mm256_mul_ps(sub, sub);
    res = _mm256_add_ps(res, mul);
    lhs += 8;
    rhs += 8;
    lhs_256 = _mm256_load_ps(lhs);
    rhs_256 = _mm256_load_ps(rhs);
  }
  alignas(32) float temp[8];
  // TODO junlinp@qq.com
  // There can be optimized.
  _mm256_store_ps(temp, res);
  float t = 0.0f;
  for (int i = 0; i < 8; i++) {
    t += temp[i];
  }
  return std::sqrtf(t);
}
*/

inline float L2_metric_aux(const float* lhs,const float* rhs) {
    //if (constexpr support_avx::value) {
    //    return L2_metric_avx(lhs, rhs);
    //}
    return L2_metric_trivial(lhs, rhs);
}

inline float L2_metric(const float* lhs,const float* rhs) {
    return L2_metric_aux(lhs, rhs);
}
template<unsigned long N>
int HammingDistance(std::bitset<N> lhs, std::bitset<N> rhs) {
  int count = 0;
  for (int i = 0; i < N; i++) {
    count += lhs[i] ^ rhs[i];
  }
  return count;
}
#endif  // SRC_METRIC_HPP_
