#include "gtest/gtest.h"
#include "triangular_solver.hpp"

#include "Eigen/Dense"

TEST(Triangular, DLT) {
    Mat34 P1;
    P1 << 1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0;
    Mat34 P2;
    P2 << -0.0093061, -0.00726141, -0.710643, 0.703353,
          -0.00920808, -0.00718497, -0.70313, -0.710839,
          -1.28578e-5, 1.58039e-5, -0.0178715, -0.00113447;

    Eigen::Vector2d ob1(306.932, 286.974);
    Eigen::Vector2d ob2(308.084, 329.19);

    std::vector<Mat34> p_matrix{P1, P2};
    std::vector<Eigen::Vector3d> obs{ob1.homogeneous(), ob2.homogeneous()};
    Eigen::Vector4d DLT_X;
    DLT(p_matrix, obs, DLT_X);
  
    Eigen::Vector4d target_X(996.178, 931.402, 3.2456, 1.0);
    Eigen::Vector3d diff = DLT_X.hnormalized() - target_X.hnormalized();
    //std::cout << DLT_X.hnormalized() << std::endl << target_X.hnormalized() << std::endl;
    EXPECT_NEAR(diff.norm(), 0, 1e-2);
}