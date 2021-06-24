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
template<class Kernel>
class Ransac {
    using sample_type = Kernel::sample_type;
    using model_type = Kernel::model_type;
    using model_number = Kernel::model_number;
    public:
        Inference(const std::vector<sample_type>& samples,
        std::vector<size_t> inlier_indexs,
        model_type* models
        );
};
#endif  // RANSAC_H_