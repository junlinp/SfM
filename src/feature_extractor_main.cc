#include <iostream>

#include "sfm_data.hpp"
#include "sfm_data_io.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
      return 1;
    }
    SfMData sfm_data;
    Load(sfm_data, argv[1]);

    auto extractor;

    extractor->FeatureExtract(sfm_data);

    Save(SfmData);
    return 0;

}