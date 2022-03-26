#include "epnp_solver.hpp"

#include "algebra.hpp"
#include "gtest/gtest.h"
using namespace Eigen;

TEST(EPnP, Solver) {
  size_t n = 6;
  Eigen::Matrix<double, 3, 6> X;
  X << 0.278498218867048, 0.964888535199277, 0.957166948242946,
      0.141886338627215, 0.792207329559554, 0.0357116785741896,
      0.546881519204984, 0.157613081677548, 0.485375648722841,
      0.421761282626275, 0.959492426392903, 0.849129305868777,
      0.957506835434298, 0.970592781760616, 0.800280468888800,
      0.915735525189067, 0.655740699156587, 0.933993247757551;
  Mat33 K;
  K << 256, 0, 512, 0, 128, 256, 0, 0, 1;
  Eigen::Matrix<double, 2, 6> x;
  x << 744.322102435809, 646.549081181268, 729.176351272558, 726.871473378172,
      898.935253768146, 819.140647148510, 372.849161154134, 419.542520768650,
      455.898274095845, 357.396985862108, 494.785531748331, 363.534593686699;

  Vector3d t = Vector3d::Random();
  Vector3d rand = Vector3d::Random();
  rand /= rand.norm();
  BDCSVD<MatrixXd> bdcSvd(rand.transpose(), ComputeFullV);
  Mat33 V = bdcSvd.matrixV();
  Mat33 R;
  R << rand, V.col(2), V.col(1);

  // MatrixXd x = (K * ((R * X).colwise() + t)).colwise().hnormalized();

  std::vector<std::pair<Vector2d, Vector3d>> data_points;
  for (int i = 0; i < n; i++) {
    data_points.push_back({x.col(i), X.col(i)});
  }

  solver::EPnPSolver solver(K);
  Pose P;
  solver.Fit(data_points, &P);

  std::cout << x << std::endl;
  std::cout
      << (K * ((P.R() * X).colwise() - P.R() * P.C())).colwise().hnormalized()
      << std::endl;
}
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}