#include <iostream>

#include "sfm_data.hpp"
#include "sfm_data_io.hpp"
#include "feature_extractor/vlfeat_feature_extractor.hpp"
int main(int argc, char** argv) {
    if (argc != 2) {
      std::printf("Usage: %s /path/to/sfm_data\n", argv[0]);
      return 1;
    }
    SfMData sfm_data;
    bool load_data = Load(sfm_data, argv[1]);
    if (!load_data) {
      std::printf("Load %s Fails.\n", argv[1]);
      return 1;
    } else {
      std::printf("Load %s Finish.\n", argv[1]);
    }

    auto extractor = VlfeatFeatureExtractor();

    extractor.FeatureExtractor(sfm_data);

    bool save_data = Save(sfm_data, argv[1]);
    return save_data ? 0 : 1;

}