#include <filesystem>

#include "internal/function_programming.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "sfm_data.hpp"
#include "sfm_data_io.hpp"

cv::KeyPoint ConvertKeyPoint(KeyPoint k) {
  cv::KeyPoint cv_k;
  cv_k.pt.x = k.x;
  cv_k.pt.y = k.y;
  return cv_k;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::printf("Usage: %s path/to/sfm_data path/to/output_directory\n",
                argv[0]);
    return 1;
  }

  SfMData sfm_data;
  bool b_load = Load(sfm_data, argv[1]);
  if (b_load) {
    std::printf("Load Sfm Data Finish\n");
  } else {
    std::printf("Load Sfm Data From %s Fails\n", argv[1]);
  }

  // if not exist create
  std::filesystem::path output_directory_path(argv[2]);
  if (!std::filesystem::exists(output_directory_path)) {
    if (!std::filesystem::create_directory(output_directory_path)) {
      std::printf("Create %s fails\n", output_directory_path.c_str());
      return 1;
    }
  }

  if (!std::filesystem::is_directory(output_directory_path)) {
    std::printf("%s is not a path to a directory\n",
                output_directory_path.c_str());
    return 1;
  }

  for (auto m : sfm_data.matches) {
    Pair pair = m.first;
    Matches matches = m.second;

    std::string image_path = output_directory_path.c_str() + std::string("/") +
                             std::to_string(pair.first) + std::string("_") +
                             std::to_string(pair.second) + std::string(".jpg");

    cv::Mat lhs_image = cv::imread(sfm_data.views.at(pair.first).image_path);
    cv::Mat rhs_image = cv::imread(sfm_data.views.at(pair.second).image_path);

    std::vector<cv::DMatch> cv_matches;
    for (auto match : matches) {
      cv::DMatch d(match.lhs_idx, match.rhs_idx, 0);
      cv_matches.push_back(d);
    }

    std::vector<cv::KeyPoint> lhs_keypoints = sfm_data.key_points[pair.first] |
                                              Transform(ConvertKeyPoint) |
                                              ToVector();
    std::vector<cv::KeyPoint> rhs_keypoints = sfm_data.key_points[pair.second] |
                                              Transform(ConvertKeyPoint) |
                                              ToVector();

    cv::Mat output_image;
    cv::drawMatches(lhs_image, lhs_keypoints, rhs_image, rhs_keypoints,
                    cv_matches, output_image);

    cv::imwrite(image_path, output_image);
  }

  return 0;
}