#ifndef FEATURE_EXTRACTOR_VLFEAT_FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_VLFEAT_FEATURE_EXTRACTOR_H_

#include "feature_extractor_interface.hpp"

class VlfeatFeatureExtractor : public FeatureExtractorInterface {
public:
    virtual bool FeatureExtractor(SfMData& sfm_data) override;

    virtual ~VlfeatFeatureExtractor() = default;

};
#endif  // FEATURE_EXTRACTOR_VLFEAT_FEATURE_EXTRACTOR_H_