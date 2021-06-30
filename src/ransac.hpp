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
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
class NormalSampler {

    template<int MINIMUM_SAMPLE, class T>
    static inline bool Sample(const std::vector<T>& samples, std::vector<int>& sample_index) {
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
    }
};
template<class Kernel>
class Ransac {
    using sample_type = typename Kernel::sample_type;
    static constexpr decltype(Kernel::minimum_data_point) MINIMUM_DATA_POINT = Kernel::minimum_data_point;
    static constexpr decltype(Kernel::model_number) MODEL_NUMBER = Kernel::model_number;
    static constexpr decltype(Kernel::model_freedom) MODEL_FREEDOM = Kernel::model_freedom;
    using MODEL_TYPE = typename Kernel::model_type;

    public:
        bool Inference(const std::vector<sample_type>& samples,
        std::vector<size_t> inlier_indexs,
        MODEL_TYPE* models
        ) {
            std::size_t N = std::numeric_limits<std::size_t>::max();
            std::size_t sample_count = 0;
            while(N > sample_count) {
                std::vector<int> sample_index;
                // Sample
                NormalSampler::Sample<MINIMUM_DATA_POINT>(samples, sample_index);

                MODEL_TYPE models[MODEL_NUMBER];
                // Compute model
                Kernel::Fit(sample_index, models);
                double sigma = 0.1;
                double threshold = std::sqrt(3.84) * sigma; 
                // Compute inliers
                for(MODEL_TYPE model_candicate : models) {
                    std::vector<size_t> inliner_index;
                    inliner_index.reserve(samples.size());
                    for(int i = 0; i < samples.size(); i++) {
                        double error = Kernel::Error(model_candicate, samples[i]);
                        if (error < threshold) {
                            inliner_index.push_back(i);
                        }
                    }
                    // epsilon = 1 - (inliners) / total_size
                    // p = 0.99 N = log(1 - p) / log(1 - (1 - epsilon)^s))
                    double epsilon = (samples.size() - inliner_index.size()) * 1.0 / (samples.size());

                    size_t temp_N = std::ceil(
                        std::log(1 - 0.99) /
                        std::log(1 - std::pow(1 - epsilon, MODEL_FREEDOM)));
                    N = std::min(N, temp_N);
                    if (inliner_index.size() > inlier_indexs.size()) {
                        std::swap(inlier_indexs, inliner_index);
                        std::swap(*models, model_candicate);
                    }
                }
                sample_count++;
            }
        }
};
#endif  // RANSAC_H_