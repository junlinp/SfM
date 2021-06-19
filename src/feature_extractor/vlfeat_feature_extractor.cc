#include "vlfeat_feature_extractor.hpp"
#include "opencv2/imgcodecs.hpp"

extern "C" {
#include "sift.h"
}
void Gray(cv::Mat image, std::vector<float>& gray_image) {
    int height = image.rows;
    int weight = image.cols;
    gray_image.resize(height * weight);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < weight; col++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
            gray_image[row * weight + col] = 
                pixel[2] * 0.11 + pixel[1] * 0.59 + pixel[0] * 0.30;
        }
    }
}
bool VlfeatFeatureExtractor::FeatureExtractor(SfMData& sfm_data) {

    for(auto view_iter : sfm_data.views) {
        std::string image_path = view_iter.second.image_path;
        cv::Mat image = cv::imread(image_path);
        VlSiftFilt* sift_handle = vl_sift_new(image.cols, image.rows, 5, 5, 0);
        // convert to gray
        std::vector<float> gray_image;
        Gray(image, gray_image);
        vl_sift_process_first_octave(sift_handle, gray_image.data());
        do {
            vl_sift_detect(sift_handle);
            // get keypoint
            // get descriptor
            int keypoint_num = vl_sift_get_nkeypoints(sift_handle);
            const VlSiftKeypoint* keypoint_ptr = vl_sift_get_keypoints(sift_handle);
            for(int i = 0; i < keypoint_num; i++) {
                double angles[4] = {0.0, 0.0, 0.0, 0.0};
                int angle_num = vl_sift_calc_keypoint_orientations(sift_handle, angles, keypoint_ptr + i);
                for (int j = 0; j < angle_num; j++) {
                    std::array<float, 128> descriptor;
                    vl_sift_calc_keypoint_descriptor(sift_handle, descriptor.data(), keypoint_ptr + i, angles[j]);
                    // output
                    IndexT view_id = view_iter.first;
                    sfm_data.key_points[view_id].push_back(KeyPoint(keypoint_ptr[i].x, keypoint_ptr[i].y));
                    sfm_data.descriptors[view_id].push_back(descriptor);
                }
            }
        } while(vl_sift_process_next_octave(sift_handle) == 0);

        vl_sift_delete(sift_handle);
    }

    return true;
}
