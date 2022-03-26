#include <fstream>
#include <iostream>
#include <queue>

#include "Eigen/Dense"
#include "euclidean_constructure.hpp"
#include "internal/function_programming.hpp"
#include "ransac.hpp"
#include "sfm_data.hpp"
#include "sfm_data_io.hpp"
#include "solver/algebra.hpp"
#include "solver/fundamental_solver.hpp"
#include "solver/self_calibration_solver.hpp"
#include "solver/triangular_solver.hpp"
#include "solver/trifocal_tensor_solver.hpp"

bool ComputeFundamentalMatrix(const std::vector<Observation>& lhs_keypoint,
                              const std::vector<Observation>& rhs_keypoint,
                              Mat33* fundamental_matrix,
                              std::vector<size_t>* inlier_index_ptr) {
  assert(lhs_keypoint.size() == rhs_keypoint.size());
  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    Eigen::Vector2d lhs_temp(lhs_keypoint[i].x(), lhs_keypoint[i].y());
    Eigen::Vector2d rhs_temp(rhs_keypoint[i].x(), rhs_keypoint[i].y());
    datas.push_back({lhs_temp, rhs_temp});
  }
  EightPointFundamentalSolver ransac_solver;

  ransac_solver.Fit(datas, *fundamental_matrix);
  double threshold = 3.84;

  std::vector<size_t> inlier_indexs;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    Eigen::Vector2d lhs_temp(lhs_keypoint[i].x(), lhs_keypoint[i].y());
    Eigen::Vector2d rhs_temp(rhs_keypoint[i].x(), rhs_keypoint[i].y());
    double error =
        // SampsonError::Error({lhs_temp, rhs_temp}, *fundamental_matrix);
        EpipolarLineError::Error({lhs_temp, rhs_temp}, *fundamental_matrix);
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

bool ComputeFundamentalMatrix(const std::vector<KeyPoint>& lhs_keypoint,
                              const std::vector<KeyPoint>& rhs_keypoint,
                              Mat33* fundamental_matrix,
                              std::vector<size_t>* inlier_index_ptr) {
  assert(lhs_keypoint.size() == rhs_keypoint.size());
  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    Eigen::Vector2d lhs_temp(lhs_keypoint[i].x, lhs_keypoint[i].y);
    Eigen::Vector2d rhs_temp(rhs_keypoint[i].x, rhs_keypoint[i].y);
    datas.push_back({lhs_temp, rhs_temp});
  }
  EightPointFundamentalSolver ransac_solver;

  ransac_solver.Fit(datas, *fundamental_matrix);
  double threshold = 3.84;

  std::vector<size_t> inlier_indexs;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    Eigen::Vector2d lhs_temp(lhs_keypoint[i].x, lhs_keypoint[i].y);
    Eigen::Vector2d rhs_temp(rhs_keypoint[i].x, rhs_keypoint[i].y);
    double error =
        // SampsonError::Error({lhs_temp, rhs_temp}, *fundamental_matrix);
        EpipolarLineError::Error({lhs_temp, rhs_temp}, *fundamental_matrix);
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

bool ComputeFundamentalMatrix(const Matches& match, Mat33* fundamental_matrix,
                              Matches* inlier_match) {
  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (auto&& m : match) {
    datas.push_back({m.lhs_observation, m.rhs_observation});
  }
  EightPointFundamentalSolver ransac_solver;
  ransac_solver.Fit(datas, *fundamental_matrix);
  double threshold = 3.48;
  for (auto&& m : match) {
    // double error = SampsonError::Error({m.lhs_observation,
    // m.rhs_observation},
    //                                    *fundamental_matrix);
    double error = EpipolarLineError::Error(
        {m.lhs_observation, m.rhs_observation}, *fundamental_matrix);
    if (error < threshold) {
      inlier_match->push_back(m);
    }
  }
  return inlier_match->size() > 30;
}

// a. choosing the initialize pair
// b. initialze the premary structure from the initialize pair
//
// c. Find the next image to register
// d. Find the 2D-3D corresponse between the registering image and the structure
// has build.
// e. Estimating the pose of registering image.
// f. Triangularing new sparse point.
// g. if there are some images need to register, then goto c, exit
// otherwise.

double Error(const Mat34& P1, const Observation& ob1, Eigen::Vector4d& X) {
  Eigen::Vector3d uv = P1 * X;
  return (ob1 - uv.hnormalized()).squaredNorm();
}

double Error(const Mat34& P1, const Observation& ob1, const Mat34& P2,
             const Observation& ob2, Eigen::Vector4d& X) {
  return (Error(P1, ob1, X) + Error(P2, ob2, X)) / 2.0;
}

void FCheck(Mat33 fundamental_matrix, Matches inliers_matches) {
  Mat34 P1, P2;
  P1 << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  Eigen::Vector3d e12 = NullSpace(fundamental_matrix.transpose().eval());
  // std::cout << fundamental_matrix.transpose() * e12 << std::endl;
  P2 << SkewMatrix(e12) * fundamental_matrix, e12;

  for (Match match : inliers_matches) {
    // std::cout << "Match Compute : " << std::endl << std::endl;

    // std::cout << "lhs observation :" << match.lhs_observation << std::endl;
    // std::cout << "rhs observation : " << match.rhs_observation << std::endl;
    // std::cout << "P1 : " << P1 << std::endl;
    // std::cout << "P2 : " << P2 << std::endl;

    // std::cout << "x' * F * x : " <<
    // match.rhs_observation.homogeneous().dot(fundamental_matrix *
    // match.lhs_observation.homogeneous()) << std::endl;
    Eigen::Vector3d epipolar_line_2 =
        fundamental_matrix * match.lhs_observation.homogeneous();
    Eigen::Vector3d epipolar_line_1 =
        fundamental_matrix.transpose() * match.rhs_observation.homogeneous();
    auto squared = [](auto num) { return num * num; };
    // std::cout << "Line2 square error : " <<
    // squared(epipolar_line_2.dot(match.rhs_observation.homogeneous())) /
    // (epipolar_line_2.x()*epipolar_line_2.x() + epipolar_line_2.y() *
    // epipolar_line_2.y())<< std::endl; std::cout << "Line1 square error : " <<
    // squared(epipolar_line_1.dot(match.lhs_observation.homogeneous())) /
    // (epipolar_line_1.x()*epipolar_line_1.x() + epipolar_line_1.y() *
    // epipolar_line_1.y())<< std::endl;

    // std::cout << "Line2 error : " <<
    // (epipolar_line_2.dot(match.rhs_observation.homogeneous())) /
    // sqrt(epipolar_line_2.x()*epipolar_line_2.x() + epipolar_line_2.y() *
    // epipolar_line_2.y())<< std::endl; std::cout << "Line1 error : " <<
    // (epipolar_line_1.dot(match.lhs_observation.homogeneous())) /
    // sqrt(epipolar_line_1.x()*epipolar_line_1.x() + epipolar_line_1.y() *
    // epipolar_line_1.y())<< std::endl;

    Eigen::Vector3d b_ =
        P2.block(0, 0, 3, 3) * match.lhs_observation.homogeneous();
    Eigen::MatrixXd A(3, 2);
    A.col(0) = match.rhs_observation.homogeneous();
    A.col(1) = -P2.col(3);
    Eigen::Vector2d solution =
        (A.transpose() * A).inverse() * A.transpose() * b_;
    // std::cout << "depth : " << solution(1) << std::endl;
    Eigen::Vector4d X = (match.lhs_observation.homogeneous()).homogeneous();
    X(3) *= solution(1);

    Eigen::Vector3d uv = P2 * X;
    Eigen::Vector3d l = SkewMatrix(e12) * uv;

    // std::cout << "l : " << l << std::endl;
    // std::cout << "l dot t : " << l.dot(match.rhs_observation.homogeneous())
    // << std::endl; std::cout << "sqrt(A**2 + B**2) : " << l(0) * l(0) + l(1) *
    // l(1) << std::endl; std::cout << "Estimate Line error : " <<
    // squared(l.dot(match.rhs_observation.homogeneous())) / (l(0) * l(0) + l(1)
    // * l(1)) << std::endl;
    const Observation& rhs_ob = match.rhs_observation;
    double a = l.x();
    double b = l.y();
    double c = l.z();
    double c_ = b * rhs_ob.x() - a * rhs_ob.y();
    double y = (-c_ - c * b / a) * a / (a * a + b * b);
    double x = (-c - b * y) / a;
    Eigen::Vector4d dlt_X;
    DLT({P1, P2},
        {match.lhs_observation.homogeneous(),
         match.rhs_observation.homogeneous()},
        dlt_X);
    if (Error(P1, match.lhs_observation, P2, match.rhs_observation, dlt_X) >
        3.84) {
      std::cout << "Geometry Error : "
                << Error(P1, match.lhs_observation, P2, match.rhs_observation,
                         dlt_X)
                << std::endl;
      std::cout << "X : " << X / X(3) << std::endl;
      std::cout << "The Point : " << x << ", " << y << std::endl;
      std::cout << "uv : " << uv.hnormalized() << std::endl;
      std::cout << "ob : " << match.rhs_observation << std::endl;
      std::cout << "P1 * X : " << (P1 * X).hnormalized() << std::endl;
      std::cout << "ob1 : " << match.lhs_observation << std::endl;
      std::cout << "P2 * X : " << (P2 * X).hnormalized() << std::endl;
      std::cout << "ob2 : " << match.rhs_observation << std::endl;
    }
  }
}

template <typename Point>
void ToPly(const std::vector<Point>& points, const std::string& output) {
  std::ofstream ofs(output);
  // header
  ofs << "ply" << std::endl;
  ofs << "format ascii 1.0" << std::endl;
  ofs << "element vertex " << points.size() << std::endl;
  ofs << "property float x" << std::endl;
  ofs << "property float y" << std::endl;
  ofs << "property float z" << std::endl;
  ofs << "end_header" << std::endl;

  // body
  for (const Point& p : points) {
    ofs << p.x << " " << p.y << " " << p.z << std::endl;
  }

  ofs.close();
}

template <typename Point>
void ToPly2(const std::vector<Point>& points, const std::string& output) {
  std::ofstream ofs(output);
  // header
  ofs << "ply" << std::endl;
  ofs << "format ascii 1.0" << std::endl;
  ofs << "element vertex " << points.size() << std::endl;
  ofs << "property float x" << std::endl;
  ofs << "property float y" << std::endl;
  ofs << "property float z" << std::endl;
  ofs << "end_header" << std::endl;

  // body
  for (const Point& p : points) {
    ofs << p.x() << " " << p.y() << " " << p.z() << std::endl;
  }

  ofs.close();
}

void GeometryFilter(SfMData& sfm_data) {
  std::cout << sfm_data.matches.size() << "Match " << std::endl;
  std::map<Pair, Matches> filter_matches;
  for (auto&& [pair, matches] : sfm_data.matches) {
    std::vector<Observation> lhs, rhs;
    for (Match match : matches) {
      lhs.push_back(match.lhs_observation);
      rhs.push_back(match.rhs_observation);
    }

    Mat33 fundamental;
    std::vector<size_t> inlier;
    bool ans = ComputeFundamentalMatrix(lhs, rhs, &fundamental, &inlier);
    if (ans) {
      Matches t;
      for (size_t index : inlier) {
        t.push_back(matches[index]);
      }
      filter_matches[pair] = t;
    }
  }

  sfm_data.matches = filter_matches;
}

// Step 1 : Geometry Filter
// Step 2 : Trifocal Init
// Step 3 : self calibration
// Step 4 : Euclidean Reconstruction
// Step 5 : Output Result
int main(int argc, char** argv) {
  if (argc != 2) {
    return 1;
  }

  SfMData sfm_data;
  bool b_load = Load(sfm_data, argv[1]);
  if (b_load) {
    std::printf("Load Sfm Data Finish\n");
  } else {
    std::printf("Load Sfm Data From %s Fails\n", argv[1]);
    return 1;
  }

  GeometryFilter(sfm_data);

  EuclideanStructure euclidean_structure(sfm_data);

  euclidean_structure.StartReconstruction();
  auto tracks = euclidean_structure.GetTracks();
  std::vector<Eigen::Vector3d> points;
  for (Track track : tracks) {
    points.push_back(track.X);
  }
  ToPly2(points, "./sparse_point.ply");
  // ToPly(sfm_data.structure_points, "./sparse_point.ply");

  // Save(sfm_data, argv[1]);

  return 0;
}
