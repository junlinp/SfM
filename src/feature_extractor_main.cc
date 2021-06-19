#include <iostream>

#include "sfm_data.hpp"
#include "sfm_data_io.hpp"
#include "feature_extractor/vlfeat_feature_extractor.hpp"
int main(int argc, char** argv) {
    if (argc != 2) {
      return 1;
    }
    SfMData sfm_data;
    Load(sfm_data, argv[1]);

    auto extractor = VlfeatFeatureExtractor();

    extractor.FeatureExtractor(sfm_data);

    Save(sfm_data, argv[1]);
    return 0;

}