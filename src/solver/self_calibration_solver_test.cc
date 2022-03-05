#include "gtest/gtest.h"
#include <random>
#include "Eigen/Dense"
#include "ceres/rotation.h"
#include "self_calibration_solver.hpp"

using namespace std;

Eigen::Matrix<double, 1, 6> Construct(int row, int col, Eigen::Matrix3d H) {
    Eigen::Matrix<double, 1, 6> res;
    Eigen::Vector3d lhs = H.row(row);
    Eigen::Vector3d rhs = H.row(col);
    res(0, 0) = lhs(0) * rhs(0);
    res(0, 1) = lhs(0) * rhs(1) + lhs(1) * rhs(0);
    res(0, 2) = lhs(0) * rhs(2) + lhs(2) * rhs(0);
    res(0, 3) = lhs(1) * rhs(1);
    res(0, 4) = lhs(1) * rhs(2) + lhs(2) * rhs(1);
    res(0, 5) = lhs(2) * rhs(2);
    return res;
}
void Initial(Eigen::Matrix<double,6 ,6>& A, Eigen::Matrix3d H) {

    A.row(0) = Construct(0, 0, H);
    A.row(1) = Construct(0, 1, H);
    A.row(2) = Construct(0, 2, H);
    A.row(3) = Construct(1, 1, H);
    A.row(4) = Construct(1, 2, H);
    A.row(5) = Construct(2, 2, H);
}

Eigen::Matrix<double, 6, 6> ConstructCoeffient(Eigen::Matrix3d H) {
    Eigen::Matrix<double,6, 6> A;
    Initial(A, H);
    return A;
}

Eigen::Matrix3d AbsoluteConic(std::vector<Mat34> Ps, Eigen::Vector3d p) {
    size_t n = Ps.size();
    Eigen::MatrixXd coeffient(6 * n, 6);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(6 * n);
    size_t i = 0;
    for (Mat34 P : Ps) {
        Eigen::Matrix3d M = P.block(0, 0, 3, 3);
        Eigen::Vector3d a = P.block(0, 3, 3, 1);

        Eigen::Matrix3d H = M - a * p.transpose();
        H /=  std::pow(H.determinant(), 1.0 / 3);
        coeffient.block(6 * i, 0, 6, 6) = ConstructCoeffient(H) - Eigen::Matrix<double, 6, 6>::Identity();
        i++;
    }
    auto svd = coeffient.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 6, 1> solution = svd.matrixV().col(5);
    std::cout << "singular Value : " << svd.singularValues() << endl;
    cout << " Solution : " << solution << std::endl;

    Eigen::Matrix3d res;
    res << solution(0), solution(1) , solution(2),
           solution(1), solution(3), solution(4),
           solution(2), solution(4), solution(5);
    return res;
}


void ConstructFunction(Eigen::Matrix3d H, double a, double b, double c) {
    Eigen::Matrix<double, 5, 2> A;
    A << H(0, 0) - 1.0, H(0, 1) * H(0, 1),
         H(0, 0) * H(1, 0), H(0, 1) * H(1, 1),
         H(0, 0) * H(2, 0), H(0, 1) * H(2, 1),
         H(1, 0) * H(1, 0), H(1, 1) * H(1, 1) - 1.0,
         H(1, 0) * H(2, 0), H(1, 1) * H(2, 1);

    Eigen::Matrix<double, 5, 1> b_;
    b_(0, 0) = H(0, 0) * H(0, 1) * a + H(0, 0) * H(0, 2) * b + H(0, 1) * H(0, 0) * a + H(0, 1) * H(0, 2) * c + H(0, 2) * H(0, 0) * b + H(0, 2) * H(0, 1) * c + H(0, 2) * H(0, 2);
    b_(1, 0) = -a + H(0, 0) * H(1, 1) * a + H(0, 0) * H(1, 2) * b + H(0, 1) * H(1, 0) * a + H(0, 1) * H(1, 2) * c + H(0, 2) * H(1, 0) * b + H(0, 2) * H(1, 1) * c + H(0, 2) * H(1, 2);
    b_(2, 0) = -b * H(0, 0) * H(2, 1) * a + H(0, 0) * H(2, 2) * b + H(0, 1) * H(2, 0) * a + H(0, 1) * H(2, 2) * c + H(0, 2) * H(2, 0) * b + H(0, 2) * H(2, 1) * c + H(0, 2) * H(2, 2);
    b_(3, 0) = H(1, 0) * H(1, 1) * a + H(1, 0) * H(1, 2) * b + H(1, 0) * H(1, 1) * a + H(1, 1) * H(1, 2) * c + H(1, 2) * H(1, 0) * b + H(1, 2) * H(1, 1) * c + H(1, 2) * H(1, 2);
    b_(4, 0) = -c + H(1, 0) * H(2, 1) * a + H(1, 0) * H(2, 2) * b + H(1, 1) * H(2, 0) * a + H(1, 1) * H(2, 2) * c + H(1, 2) * H(2, 0) * b + H(1, 2) * H(2, 1) * c + H(1, 2) * H(2, 2);
    b_ = -b_;
    cout << "2 : " << A << endl;
    Eigen::Vector2d solution = A.bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b_);
    
    solution(0) -= b * b;
    solution(1) -= c * c;
    std::cout << "x :" << solution(0) << " y : " << solution(1) << std::endl;
    cout << "fx : " << std::sqrt(solution(0)) << " fy : " << std::sqrt(solution(1)) << endl;
}

TEST(SELF_CALIBRATION, IAC) {
    
    auto rand = [](double min, double max) {
        random_device rd;
        uniform_real_distribution<double> uniform(min, max);
        return uniform(rd);
    };

    Eigen::Matrix3d K;
    double fx, fy, cx, cy;
    fx = rand(1.0, 1024);
    fy = rand(1.0, 1024);
    cx = rand(1024, 2048);
    cy = rand(1024, 2048);

    K << fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0;

    Eigen::Vector3d p(rand(0, 100.0), rand(0, 100.0), rand(0, 100.0));
    std::cout << "Initial K : " << std::endl << K << std::endl;
    std::cout << "Initial p : " << std::endl << p << std::endl;
    size_t data_size = 16;
    using Mat34 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;
    std::vector<Mat34> data_set;
    data_set.reserve(data_size);

    Eigen::Matrix4d Homogeneous_Matrix_inverse;

    Homogeneous_Matrix_inverse << K.inverse(), Eigen::Vector3d::Zero(),
                                  p.transpose(), 1.0;

    for(size_t i = 0; i < data_size; i++) {
        if (i == 0) {

          Eigen::Matrix3d R;
          Eigen::Vector3d t;
          R.setIdentity();
          t.setZero();
          Eigen::Matrix<double, 3, 4> P;
          P << R, t;
          P = K * P * Homogeneous_Matrix_inverse;
          std::cout << P << std::endl; 
          data_set.push_back(P);
        } else {
          Eigen::Matrix3d R;
          Eigen::Vector3d t;

          double angle_axis[3] = {rand(0.0, 180), rand(0.0, 180),
                                  rand(0.0, 180)};
          ceres::AngleAxisToRotationMatrix(angle_axis, R.data());
          t << rand(0.0, 100), rand(0.0, 100), rand(0, 100);
          Eigen::Matrix<double, 3, 4> P;
          P << R, t;
          P = K * P * Homogeneous_Matrix_inverse;
          data_set.push_back(P);
        }
    }

    IterativeSolver iterative_solver( rand(-100, 100), rand(-100, 100), cx + rand(-10.0, 10.0), cy + rand(-10, 10), 0.0, 0.0, 0.0) ;
    iterative_solver.Solve(data_set);
    Eigen::Matrix3d estimate_K = iterative_solver.K();
    
    std::cout << estimate_K << std::endl;
    // std::cout << "Estimate p : "<< iterative_solver.p() << std::endl;
    // std::cout << data_set[0] * iterative_solver.HomogeneousMatrix() << std::endl;

    // both don't exceed 10%
    EXPECT_NEAR(fx, estimate_K(0, 0), fx * 0.1);
    EXPECT_NEAR(fy, estimate_K(1, 1), fy * 0.1);
    EXPECT_NEAR(cx, estimate_K(0, 2), cx * 0.1);
    EXPECT_NEAR(cy, estimate_K(1, 2), cy * 0.1);

}
int main(int argc, char**argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}