#include <iostream>

#include "Eigen/Dense"
#include "internal/function_programming.hpp"
#include "ransac.hpp"
#include "sfm_data.hpp"
#include "sfm_data_io.hpp"
#include "solver/fundamental_solver.hpp"

bool ComputeFundamentalMatrix(const std::vector<KeyPoint>& lhs_keypoint,
                              const std::vector<KeyPoint>& rhs_keypoint,
                              Eigen::Matrix3d* fundamental_matrix) {
  assert(lhs_keypoint.size() == rhs_keypoint.size());

  std::vector<typename EightPointFundamentalSolver::DataPointType> datas;
  for (int i = 0; i < lhs_keypoint.size(); i++) {
    datas.push_back({lhs_keypoint[i], rhs_keypoint[i]});
  }
  Ransac<EightPointFundamentalSolver> ransac_solver;
  std::vector<size_t> inlier_indexs;
  ransac_solver.Inference(datas, inlier_indexs, fundamental_matrix);
  std::printf("inlier : %lu\n", inlier_indexs.size());
  return inlier_indexs.size() > 30;
}
using Mat34 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

Eigen::Matrix3d VectorToSkewMatrix(const Eigen::Vector3d& v) {
  Eigen::Matrix3d res;
  double a1 = v(0), a2 = v(1), a3 = v(2);
  res << 0.0, -a3, a2, a3, 0.0, -a1, -a2, a1, 0.0;
  return res;
}

Eigen::Vector3d NullSpace(const Eigen::Matrix3d& m) {
  Eigen::JacobiSVD svd(m, Eigen::ComputeFullV);
  return svd.matrixV().col(2);
}

bool ComputeProjectiveReconstruction(const Eigen::Matrix3d& F, Mat34& P1,
                                     Mat34& P2) {
  // A. Compute Null Space of the traspose of F

  // B.Construct the {P1, P2} as follow:
  // P1 = [I | 0]
  // P2 = [(e')_x * F | e']
  Eigen::Vector3d e_dot = NullSpace(F);

  P1 << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;
  P2 << VectorToSkewMatrix(e_dot) * F, e_dot;
  return true;
}

#include <queue>

bool ExistOneKey(const std::set<IndexT>& set, std::pair<IndexT, IndexT> item) {
  int lhs_exist = (set.find(item.first) != set.end());
  int rhs_exist = (set.find(item.second) != set.end());
  return lhs_exist ^ rhs_exist;
}

void Solve(const Mat34& A, const Eigen::Vector3d& b, Eigen::Vector4d& x) {
  Eigen::Matrix3d A33 = A.block(0, 0, 3, 3);

  Eigen::Vector3d x_ = A33.inverse() * (b - A.col(3));
  x << x_, 1.0;
}

void ComputeHTransform(const Mat34& sour, const Mat34& dest,
                       Eigen::Matrix4d& H) {
  // sour * H = dest
  // We assume H :
  //   h11  h12  h13  h14
  //   h21  h22  h23  h24
  //   h31  h32  h33  h34
  //   1.0  1.0  1.0  1.0

  Eigen::Vector4d h1, h2, h3, h4;
  Solve(sour, dest.col(0), h1);
  Solve(sour, dest.col(1), h2);
  Solve(sour, dest.col(2), h3);
  Solve(sour, dest.col(3), h4);
  H << h1, h2, h3, h4;
}

void ProjectiveReconstruction(
    const std::map<Pair, Eigen::Matrix3d>& fundamental_matrix,
    std::map<IndexT, Mat34>& projective_reconstruction) {
  using Fundamental_Matrix_Type =
      typename std::map<Pair, Eigen::Matrix3d>::value_type;
  std::queue<Fundamental_Matrix_Type> que;
  for (auto item : fundamental_matrix) {
    que.push(item);
  }
  std::set<IndexT> processed_index;

  // A. Init
  Fundamental_Matrix_Type init_seed = que.front();
  que.pop();
  IndexT lhs_index = init_seed.first.first;
  IndexT rhs_index = init_seed.first.second;
  ComputeProjectiveReconstruction(init_seed.second,
                                  projective_reconstruction[lhs_index],
                                  projective_reconstruction[rhs_index]);
  processed_index.insert(lhs_index);
  processed_index.insert(rhs_index);
  //
  bool found_and_continue = true;

  while (found_and_continue) {
    found_and_continue = false;
    std::queue<Fundamental_Matrix_Type> temp_que;
    while (!que.empty()) {
      Fundamental_Matrix_Type need_to_process = que.front();
      que.pop();

      if (ExistOneKey(processed_index, need_to_process.first)) {
        found_and_continue = true;
        IndexT lhs_index = need_to_process.first.first;
        IndexT rhs_index = need_to_process.first.second;

        Mat34 P1, P2;
        ComputeProjectiveReconstruction(need_to_process.second, P1, P2);
        if (processed_index.find(lhs_index) == processed_index.end()) {
          Eigen::Matrix4d H;
          ComputeHTransform(P2, projective_reconstruction.at(rhs_index), H);
          projective_reconstruction[lhs_index] = P1 * H;

        } else {
          Eigen::Matrix4d H;
          ComputeHTransform(P1, projective_reconstruction.at(lhs_index), H);
          projective_reconstruction[rhs_index] = P2 * H;
        }

        processed_index.insert(lhs_index);
        processed_index.insert(rhs_index);
      } else {
        temp_que.push(need_to_process);
      }
    }
    que = std::move(temp_que);
  }
}
// row and col start with 1 instead of 0
Eigen::Matrix<double, 1, 16> GenerateCoeffient(const Mat34 P, size_t row,
                                               size_t col) {
  Eigen::Matrix<double, 1, 16> res;
  Eigen::Vector4d rhs = P.row(col - 1);
  res << P(row - 1, 0) * rhs.transpose(), P(row - 1, 1) * rhs.transpose(),
      P(row - 1, 2) * rhs.transpose(), P(row - 1, 3) * rhs.transpose();
  return res;
}

// The Matrix K is a 3 x 3 matrix with format as follow:
//  |  alpha_x    0     cx |
//  |     0    alpha_y  cy |
//  |     0       0      1 |
//
// Such that the dual omega = KK^T
//      |   alpha_x * alpha_x + cx * cx        cx * cy      cx |
// KK^T =|     cx * cy           alpha_y * alpha_y * cy * cy cy |
//      |        cx                          cy             1  |
//
//  KK^T =   P * Q * P^T
//  We will solve Q with the constraint about the matrix KK^T
//
void ComputeIntrinsicMatrix(std::map<IndexT, Mat34>& projective_reconstruction,
                            size_t image_width, size_t image_height) {
  // cx = image_width / 2
  // cy = image_height / 2
  // alpha_x / alpha y = image_width / image_height (not used)
  // this is the three constraint to solve a linear function.
  //
  // 16 parameter need to 8 photos to compute at least.
  //

  size_t cx = image_width / 2;
  size_t cy = image_width / 2;
  //  4 * projective_reconstruction * 16
  size_t cameras_size = projective_reconstruction.size();
  Eigen::MatrixXd coeffient(4 * cameras_size, 16);
  Eigen::VectorXd constant(4 * cameras_size);
  size_t count = 0;
  for (auto& item_pair : projective_reconstruction) {
    const Mat34& P_i = item_pair.second;
    coeffient.row(count * 4 + 0) = GenerateCoeffient(P_i, 1, 3);
    coeffient.row(count * 4 + 1) = GenerateCoeffient(P_i, 2, 3);
    coeffient.row(count * 4 + 2) = GenerateCoeffient(P_i, 3, 1);
    coeffient.row(count * 4 + 3) = GenerateCoeffient(P_i, 3, 2);
    constant(count * 4 + 0) = cx;
    constant(count * 4 + 1) = cy;
    constant(count * 4 + 2) = cx;
    constant(count * 4 + 3) = cy;
    count++;
  }
  std::cout << "Generate coeffient Finish" << std::endl;

  // Solve Least-Squares Method
  Eigen::VectorXd Q_coeffient = coeffient.colPivHouseholderQr().solve(constant);
  std::cout << "coeffient : " << Q_coeffient << std::endl;
  Eigen::Matrix4d Q =
      Eigen::Map<Eigen::Matrix<double, 4, 4>>(Q_coeffient.data());
  std::cout << "Solved " << Q << std::endl;
  // SVD Q = HIH with I is diag(1, 1, 1, 0)
  Eigen::JacobiSVD svd(Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector4d diag = svd.singularValues();
  std::cout << "Singular Value : " << diag << std::endl;
  diag(3) = 0.0;

  Q = svd.matrixU() * diag.asDiagonal() * svd.matrixV();
  std::cout << "Q Matrix : " << Q << std::endl;
  svd = Eigen::JacobiSVD(Q, Eigen::ComputeFullV);
  Eigen::Matrix4d H = svd.matrixV();

  for (auto& item : projective_reconstruction) {
    item.second = item.second * H;
  }
}

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
  }
  // Filter With fundamental matrix
  std::map<Pair, Eigen::Matrix3d> fundamental_matrix;
  for (const auto& iter : sfm_data.matches) {
    Pair pair = iter.first;
    if (iter.second.size() < 30) {
      continue;
    }
    std::printf("Compute Fundamental Matrix for pair <%lld, %lld>\n",
                pair.first, pair.second);
    const std::vector<KeyPoint>& lhs_keypoints =
        sfm_data.key_points.at(pair.first);
    const std::vector<KeyPoint>& rhs_keypoints =
        sfm_data.key_points.at(pair.second);

    std::vector<KeyPoint> lhs, rhs;
    lhs.reserve(iter.second.size());
    rhs.reserve(iter.second.size());
    for (const Matche& m : iter.second) {
      lhs.push_back(lhs_keypoints[m.first]);
      rhs.push_back(rhs_keypoints[m.second]);
    }

    Eigen::Matrix3d F;
    if (ComputeFundamentalMatrix(lhs, rhs, &F)) {
      fundamental_matrix.insert({pair, F});
    }
  }
  std::printf("%lu Matches are reserved after fundamental matrix filter\n",
              fundamental_matrix.size());

  std::set<IndexT> id_set;
  auto v = fundamental_matrix |
           Transform([](auto item) { return item.first; }) | ToVector();
  for (auto i : v) {
    id_set.insert(i.first);
    id_set.insert(i.second);
  }
  // whether the graph is connected.
  if (id_set.size() != sfm_data.views.size()) {
    std::printf("Warning: The visible graph is not connected after filter\n");
  }

  // Compute the P matrix for each images
  // using fundamental matrix.
  std::map<IndexT, Mat34> projective_reconstruction;
  ProjectiveReconstruction(fundamental_matrix, projective_reconstruction);
  std::cout << "Reconstruction : " << projective_reconstruction.size()
            << std::endl;
  for (auto iter : projective_reconstruction) {
    std::cout << iter.second << std::endl;
  }

  // Compute camera matrix K with the
  // IAC
  ComputeIntrinsicMatrix(projective_reconstruction, 3072, 2304);

  // triangulation
  //

  // bundle adjustment the parameters rotation and translation

  Save(sfm_data, argv[1]);

  return 0;
}