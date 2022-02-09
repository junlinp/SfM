#include <filesystem>

#include "internal/function_programming.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "sfm_data.hpp"
#include "sfm_data_io.hpp"
#include "solver/fundamental_solver.hpp"

cv::KeyPoint ConvertKeyPoint(KeyPoint k) {
  cv::KeyPoint cv_k;
  cv_k.pt.x = k.x;
  cv_k.pt.y = k.y;
  return cv_k;
}

bool ComputeFundamentalMatrix(const std::vector<Observation>& lhs_keypoint,
                              const std::vector<Observation>& rhs_keypoint,
                              Mat33* fundamental_matrix,
                              std::vector<size_t>* inlier_index_ptr) {
  assert(lhs_keypoint.size() == rhs_keypoint.size());
  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    //Eigen::Vector2d lhs_temp(lhs_keypoint[i].x, lhs_keypoint[i].y);
    //Eigen::Vector2d rhs_temp(rhs_keypoint[i].x, rhs_keypoint[i].y);
    datas.push_back({ lhs_keypoint[i], rhs_keypoint[i]});
  }

  EightPointFundamentalSolver ransac_solver;

  ransac_solver.Fit(datas, *fundamental_matrix);
  double threshold = 3.84;

  std::vector<size_t> inlier_indexs;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    //Eigen::Vector2d lhs_temp(lhs_keypoint[i].x, lhs_keypoint[i].y);
    //Eigen::Vector2d rhs_temp(rhs_keypoint[i].x, rhs_keypoint[i].y);
    double error =
        //SampsonError::Error({lhs_temp, rhs_temp}, *fundamental_matrix);
        EpipolarLineError::Error({lhs_keypoint[i], rhs_keypoint[i]}, *fundamental_matrix);
    if (error < threshold) {
      inlier_indexs.push_back(i);
    }
  }
  std::printf("inlier : %lu\n", inlier_indexs.size());
  bool ans = inlier_indexs.size() > 30;
  if (inlier_index_ptr != nullptr) {
    *inlier_index_ptr = std::move(inlier_indexs);
  }
  return ans;
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

    std::vector<Observation> lhs, rhs;
    for (auto match: matches) {
        lhs.push_back(match.lhs_observation);
        rhs.push_back(match.rhs_observation);
    }

    Mat33 fundamental_matrix;
    std::vector<size_t> inlier;
    ComputeFundamentalMatrix(lhs, rhs, &fundamental_matrix, &inlier);

    Matches match_filtered;
    for (size_t index : inlier) {
        match_filtered.push_back(matches[index]);
    }

    std::vector<cv::DMatch> cv_matches;

    for (auto match : match_filtered) {
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