#include "vlfeat_feature_extractor.hpp"
#include "opencv2/imgcodecs.hpp"
#include "../internal/thread_pool.hpp"

#include <algorithm>
#include <numeric>
// Bibliography:
// [1] R. Arandjelović, A. Zisserman.
// Three things everyone should know to improve object retrieval. CVPR2012.

extern "C" {
#include "sift.h"
}

void rootSIFT(std::array<float, 128>& descriptor) {
    // L1 Normalized
    float sum = std::accumulate(descriptor.begin(), descriptor.end(), 0.0);

    for(float& f : descriptor) {
        f = std::sqrt(f /sum);
    }
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

size_t VlFeatureExtractor_(const std::string& path, std::vector<KeyPoint>& keypoints, Descriptors& descriptors) {
  cv::Mat image = cv::imread(path);
  VlSiftFilt* sift_handle = vl_sift_new(image.cols, image.rows, 6, 3, 0);
  vl_sift_set_edge_thresh(sift_handle, 10.0);
  vl_sift_set_peak_thresh(sift_handle, 255 * 0.04 / 3);

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
    for (int i = 0; i < keypoint_num; i++) {
      double angles[4] = {0.0, 0.0, 0.0, 0.0};
      int angle_num = vl_sift_calc_keypoint_orientations(sift_handle, angles,
                                                         keypoint_ptr + i);
      for (int j = 0; j < angle_num; j++) {
        std::array<float, 128> descriptor;
        vl_sift_calc_keypoint_descriptor(sift_handle, descriptor.data(),
                                         keypoint_ptr + i, angles[j]);
        // output
        keypoints.push_back(
            KeyPoint(keypoint_ptr[i].x, keypoint_ptr[i].y));
        rootSIFT(descriptor);
        descriptors.push_back(descriptor);
      }
    }
  } while (vl_sift_process_next_octave(sift_handle) == 0);
  vl_sift_delete(sift_handle);
  return keypoints.size();
}

bool VlfeatFeatureExtractor::FeatureExtractor(SfMData& sfm_data) {
    size_t count = 0;
    ThreadPool thread_pool;
    size_t view_size = sfm_data.views.size();
    for (auto view_iter : sfm_data.views) {
      sfm_data.key_points[view_iter.first] = std::vector<KeyPoint>();
      sfm_data.descriptors[view_iter.first] = Descriptors();
    }
    std::vector<std::future<size_t>> t;
    for(auto view_iter : sfm_data.views) {
        std::string image_path = view_iter.second.image_path;

        auto res = thread_pool.Enqueue(VlFeatureExtractor_, std::move(image_path),
                            std::ref(sfm_data.key_points[view_iter.first]),
                            std::ref(sfm_data.descriptors[view_iter.first]));

        t.emplace_back(std::move(res));
    }
    for (auto& tt : t) {
        size_t key_point_count = tt.get();
        std::printf("Feature Extractor %lu Keypoint --- %f %%\n", key_point_count, 100.0 * ++count / view_size);
    }

    return true;
}
