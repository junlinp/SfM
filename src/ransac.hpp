#ifndef RANSAC_H_
#define RANSAC_H_
// We will choose a distance threshold t to make
// the probability of the inliers is alpha.
// Since the freedom of the model is n, and the distance of
// our measure of the error belongs to gaussian distribution
// N(0, sigma), we have the distance meets the X^2 distribution
//
// To make sure that there is no outliers in a sample which contains
// s items under the probability of p.Normally, we assume that p equals to
// 0.99.
// And w is the probability of sample selected randomly.
// the minimum of sample times N is computed as follow
// (1 - w^s)^N = 1 - p
// N = log(1 - p) / log(1 - w^s)
//
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
class NormalSampler {
 public:
  template <int MINIMUM_SAMPLE, class T>
  static inline bool Sample(const std::vector<T>& samples,
                            std::vector<int>& sample_index) {
    if (samples.size() < MINIMUM_SAMPLE) {
      std::cerr << "samples with " << samples.size() << " can't select "
                << MINIMUM_SAMPLE << " items." << std::endl;
      return false;
    }

    sample_index.resize(samples.size());
    std::default_random_engine engine;
    std::iota(sample_index.begin(), sample_index.end(), 0);
    std::shuffle(sample_index.begin(), sample_index.end(), engine);
    sample_index.resize(MINIMUM_SAMPLE);
    return true;
  }
};

/**
 * @brief Ransac Model
 *
 * Concept requiredment:
 *  - function signature: void Fit<std::vector<DataPointType>&, MODEL_TYPE* models)
 *  - function signature: double Error(const DataPointType&, const MODEL_TYPE& model);
 * @tparam Kernel
 */
template <class Kernel>
class Ransac {
  static constexpr double chi_square_distribute[] = {
      0.0, 3.84, 5.99, 7.82, 9.49, 11.07, 12.59, 14.07, 15.51, 16.92, 18.31};
  using DataPointType = typename Kernel::DataPointType;

  static constexpr decltype(Kernel::MINIMUM_DATA_POINT) MINIMUM_DATA_POINT =
      Kernel::MINIMUM_DATA_POINT;
  static constexpr decltype(Kernel::MODEL_NUMBER) MODEL_FIT_NUMBER =
      Kernel::MODEL_NUMBER;
  static constexpr decltype(Kernel::MODEL_FREEDOM) MODEL_FREEDOM =
      Kernel::MODEL_FREEDOM;

  using MODEL_TYPE = typename Kernel::MODEL_TYPE;

 public:
  bool Inference(const std::vector<DataPointType>& samples,
                 std::vector<size_t> inlier_indexs, MODEL_TYPE* result_models) {
    std::size_t N = std::numeric_limits<std::size_t>::max();
    std::size_t sample_count = 0;
    while (N > sample_count) {
        std::printf("Sample %lu of Total %lu\n", sample_count, N);
      std::vector<int> sample_index;
      // Sample
      NormalSampler::Sample<MINIMUM_DATA_POINT>(samples, sample_index);

      MODEL_TYPE models[MODEL_FIT_NUMBER];
      std::vector<DataPointType> temp_sample;
      for (int index : sample_index) {
        temp_sample.push_back(samples[index]);
      }
      // Compute model
      Kernel::Fit(temp_sample, models);
      double sigma = 0.1;
      double threshold = chi_square_distribute[MODEL_FREEDOM] * sigma * sigma;
      // Compute inliers
      for (MODEL_TYPE model_candicate : models) {
        std::vector<size_t> inliner_index;
        inliner_index.reserve(samples.size());
        for (int i = 0; i < samples.size(); i++) {
          double error = Kernel::Error(samples[i], model_candicate);
          if (error < threshold) {
            inliner_index.push_back(i);
          }
        }
        // epsilon = 1 - (inliners) / total_size
        // p = 0.99 N = log(1 - p) / log(1 - (1 - epsilon)^s))
        if (inliner_index.size() == samples.size()) {
          std::swap(inlier_indexs, inliner_index);
          std::swap(*result_models, model_candicate);
          return true;
        }
        double epsilon =
            (samples.size() - inliner_index.size()) * 1.0 / (samples.size());
            std::printf("SAMPLE SIZE = %lu, inliner_index SIZE = %lu\n", samples.size(), inliner_index.size());
        std::printf("EPSILON = %lf\n", epsilon);
        size_t temp_N =
            std::ceil(std::log(1 - 0.99) /
                      std::log(1 - std::pow(1 - epsilon, MODEL_FREEDOM)));
        N = std::min(N, temp_N);
        if (inliner_index.size() > inlier_indexs.size()) {
          std::swap(inlier_indexs, inliner_index);
          std::swap(*result_models, model_candicate);
        }
      }
      sample_count++;
    }
    return true;
  }
};
#endif  // RANSAC_H_