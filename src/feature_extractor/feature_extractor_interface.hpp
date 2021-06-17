#ifndef FEATURE_EXTRACTOR_INTERFACE_H_
#define FEATURE_EXTRACTOR_INTERFACE_H_
#include "../sfm_data.hpp"
class FeatureExtractorInterface {
    public:
    virtual bool FeatureExtractor(SfMData& sfm_data) = 0;
};
#endif  // FEATURE_EXTRACTOR_INTERFACE_H_