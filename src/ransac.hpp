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
#include <iterator>
#include <numeric>
#include <random>
#include <vector>
#include <set>

class NormalSampler {
 public:
  size_t index_size_;
  std::vector<size_t> shuffle_index_;
  NormalSampler(size_t index_size)
      : index_size_(index_size), shuffle_index_(index_size) {
    std::iota(shuffle_index_.begin(), shuffle_index_.end(), 0);
  }

  template <int MINIMUM_SAMPLE, class Container, class T = typename std::decay_t<Container>::value_type>
  inline bool Sample(const Container& samples,
                     std::vector<int>& sample_index) {
    assert(samples.size() == index_size_);
    if (samples.size() < MINIMUM_SAMPLE) {
      //std::cerr << "samples with " << samples.size() << " can't select "
      //          << MINIMUM_SAMPLE << " items." << std::endl;
      return false;
    }
    sample_index.reserve(index_size_);
    std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
    std::uniform_int_distribution<size_t> start_distribu(0, index_size_);
    std::random_device rd;
    std::mt19937 engine(rd());
    size_t offset = start_distribu(engine);
    // selection sampling
    size_t selected_count = 0;
    for (size_t i = 0; i < index_size_; i++) {
      double select_probability =
          double(MINIMUM_SAMPLE - selected_count) / (index_size_ - i);
      double sample_probability = uniform_distribution(engine);
      if (sample_probability < select_probability) {
        sample_index.push_back((offset + i) % index_size_);
        selected_count++;
      }
    }
    assert(sample_index.size() == MINIMUM_SAMPLE);
    return true;
  }
};

/**
 * @brief Ransac Model
 *
 * Concept requiredment:
 *  - function signature: void Fit<std::vector<DataPointType>&, MODEL_TYPE*
 * models)
 *  - function signature: double Error(const DataPointType&, const MODEL_TYPE&
 * model);
 * @tparam Kernel
 */
template <class Kernel, class ErrorEstimator,
// Concept
typename DataPointType = typename Kernel::DataPointType,
auto MINIMUM_DATA_POINT = Kernel::MINIMUM_DATA_POINT,
auto MODEL_FIT_NUMBER = Kernel::MODEL_NUMBER,
auto MODEL_FREEDOM = Kernel::MODEL_FREEDOM,
typename MODEL_TYPE = typename Kernel::ModelType,
typename _Fit = std::invoke_result_t<decltype(Kernel::Fit), const std::vector<DataPointType>&,MODEL_TYPE*>,
typename _Error = std::invoke_result_t<decltype(ErrorEstimator::Error), const DataPointType&,const MODEL_TYPE&>
>
class Ransac {
  static constexpr double chi_square_distribute[] = {
      0.0, 3.84, 5.99, 7.82, 9.49, 11.07, 12.59, 14.07, 15.51, 16.92, 18.31};
/*
  using DataPointType = typename Kernel::DataPointType;
  static constexpr decltype(Kernel::MINIMUM_DATA_POINT) MINIMUM_DATA_POINT =
      Kernel::MINIMUM_DATA_POINT;
  static constexpr decltype(Kernel::MODEL_NUMBER) MODEL_FIT_NUMBER =
      Kernel::MODEL_NUMBER;
  static constexpr decltype(Kernel::MODEL_FREEDOM) MODEL_FREEDOM =
      Kernel::MODEL_FREEDOM;
  using MODEL_TYPE = typename Kernel::ModelType;
*/

 public:
  static bool Inference(const std::vector<DataPointType>& samples,
                 std::vector<size_t>& inlier_indexs, MODEL_TYPE* result_models) {
    //std::size_t N = std::numeric_limits<std::size_t>::max();
    constexpr std::size_t MaxEpoch = 4096;
    std::size_t N = MaxEpoch;
    std::size_t sample_count = 0;
    NormalSampler sampler(samples.size());
    while (N > sample_count) {
      // std::printf("Sample %lu of Total %lu\n", sample_count, N);
      std::vector<int> sample_index;
      // Sample
      sampler.Sample<MINIMUM_DATA_POINT>(samples, sample_index);

      MODEL_TYPE models[MODEL_FIT_NUMBER];
      std::vector<DataPointType> temp_sample;
      for (int index : sample_index) {
        temp_sample.push_back(samples[index]);
      }
      // Compute model
      Kernel::Fit(temp_sample, models);
      double sigma = 1.0;
      double threshold = chi_square_distribute[MODEL_FREEDOM] * sigma * sigma;
      // Compute inliers
      for (MODEL_TYPE model_candicate : models) {
        std::vector<size_t> inliner_index;
        inliner_index.reserve(samples.size());
        inliner_index.assign(sample_index.begin(), sample_index.end());
        std::set<size_t> inlier_set;
        inlier_set.insert(sample_index.begin(), sample_index.end());
        for (int i = 0; i < samples.size(); i++) {
          if (inlier_set.find(i) == inlier_set.end()) {
            double error = ErrorEstimator::Error(samples[i], model_candicate);
            if (error < threshold) {
              inliner_index.push_back(i);
              inlier_set.insert(i);
            }
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
    return inlier_indexs.size() >= MINIMUM_DATA_POINT ? true : false;
  }

  static void Fit(const std::vector<DataPointType>& data_points, MODEL_TYPE& model) {
    std::vector<size_t> placeholder_;
    Inference(data_points, placeholder_, &model);
  }

  static double Error(const DataPointType& data_point,const MODEL_TYPE& model) {
    return Kernel::Error(data_point, model);
  }
};
#endif  // RANSAC_H_