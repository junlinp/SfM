#include "vlfeat_feature_extractor.hpp"
#include "opencv2/imgcodecs.hpp"

extern "C" {
#include "sift.h"
}

bool VlfeatFeatureExtractor::FeatureExtractor(SfMData& sfm_data) {

    for(auto view_iter : sfm_data.views) {
        std::string image_path = view_iter.second.image_path;
        cv::Mat image = cv::imread(image_path);
        VlSiftFilt* sift_handle = vl_sift_new(image.cols, image.rows, 5, 5, 0);
        // convert to gray
        std::vector<float> gray_image;
        vl_sift_process_first_octave(sift_handle, gray_image.data());
        do {
            vl_sift_detect(sift_handle);
        } while(vl_sift_process_next_octave(sift_handle));

        vl_sift_delete(sift_handle);
    }

    return true;
}
