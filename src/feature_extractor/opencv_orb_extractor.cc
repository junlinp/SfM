#include "opencv_orb_extractor.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "iostream"
bool OpenCVORBFeatureExtractor::FeatureExtractor(SfMData& sfm_data) {
    auto orb_ptr = cv::ORB::create(1024);
    for(auto view_iter : sfm_data.views) {
        std::string image_path = view_iter.second.image_path;

        std::vector<cv::KeyPoint> opencv_keypoints;
        cv::Mat opencv_descriptor;
        cv::Mat image = cv::imread(image_path);
        orb_ptr->detect(image, opencv_keypoints);
        orb_ptr->compute(image, opencv_keypoints, opencv_descriptor);

        IndexT view_id = view_iter.first;

        for(size_t i = 0; i < opencv_keypoints.size(); i++) {
          cv::KeyPoint o_keypoint = opencv_keypoints[i];
          sfm_data.key_points[view_id].emplace_back(o_keypoint.pt.x,
                                                    o_keypoint.pt.y);
          std::array<float, 128> descriptor;
          std::fill_n(descriptor.begin(), 128, 0.0);
          for(int j = 0; j < 32; j++) {
              descriptor[j] = opencv_descriptor.at<unsigned char>(i, j);
          }
          sfm_data.descriptors[view_id].push_back(descriptor);
        }
        //std::cout << "Descriptor : " << opencv_descriptor.rows << " , " << opencv_descriptor.cols << std::endl;
    }
 
    return true;
}