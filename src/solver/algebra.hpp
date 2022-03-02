#ifndef SRC_SOLVER_ALGEBRA_HPP_
#define SRC_SOLVER_ALGEBRA_HPP_

#include <iostream>
#include "Eigen/Dense"

template<typename Derived>
Eigen::VectorXd NullSpace(const Derived& m) {
  size_t col = m.cols();

  Eigen::JacobiSVD svd(m, Eigen::ComputeFullV);

  return svd.matrixV().col(col - 1);
}

template<typename T>
Eigen::VectorXd NullSpace(const Eigen::Transpose<T>& m) {
    T temp = m;
    return NullSpace<T>(temp);
}

template<class T>
Eigen::Matrix3d SkewMatrix(const T& v) {
    Eigen::Matrix3d res;
    res << 0.0, -v(2), v(1),
           v(2), 0.0,  -v(0),
           -v(1), v(0), 0.0;
    return res;
}

template<class T>
Eigen::Matrix<T, 3, 3> SkewMatrix(T* ptr) {
    T data[9] = {
        T(0.0), -ptr[2], ptr[1],
        ptr[2], T(0.0), -ptr[0],
        -ptr[1], ptr[0], T(0.0)
    };
    return Eigen::Map<Eigen::Matrix<T, 3, 3, Eigen::RowMajor>>(data);
}



// solute min |Ax| with |x| = 1
// where symbol |*| means L2-norm
template<typename DerivedMatrix, typename DerivedVector>
void LinearEquationWithNormalSolver(const DerivedMatrix& A, DerivedVector& t) {
    Eigen::JacobiSVD svd(A, Eigen::ComputeFullV);
    auto singular = svd.singularValues();
    std::cout << "Least Singular : " << singular(singular.size() - 1) << std::endl;
    t = svd.matrixV().col(singular.size() - 1);
}

template<typename ConstraintEvaluator, typename ConstraintJacobianEvaluator>
struct SampsonBase {
    template<typename Model, typename DataPoint>
    static double Error(DataPoint data_point, const Model& model) {
        auto epsilon = ConstraintEvaluator::Evaluate(data_point, model);
        //std::cout << "Epsilon in Sampson : " << epsilon << std::endl;
        auto jacobian = ConstraintJacobianEvaluator::Jacobian(data_point, model);
        Eigen::MatrixXd JJT = jacobian * jacobian.transpose();
        //std::cout << "JJT : " << JJT << std::endl;
        //std::cout << "JJT LLT : " << JJT.ldlt().solve(epsilon) << std::endl;
        //std::cout << JJT.bdcSvd().singularValues() << std::endl;
        using Solver = Eigen::LDLT<Eigen::MatrixXd>;
        Solver solver;
        solver.compute(JJT);
        auto solution = solver.solve(epsilon);
        //std::cout << "Solution : " << solution << std::endl;
        //std::cout << "Delta : " << -jacobian.transpose() * solution << std::endl;
        return epsilon.dot(solution);
    }
};

struct EpsilonTensor {
static constexpr double data[3][3][3] = {
{{0.0 , 0.0, 0.0}, {0.0, 0.0, 1}, {0.0, -1, 0.0}},
{{0.0, 0.0, -1}, {0.0, 0.0, 0.0}, {1, 0.0, 0.0}},
{{0.0, 1, 0.0}, {-1, 0.0, 0.0}, {0.0, 0.0, 0.0}}
};
    static double at(int i, int j, int k) {
        //return data[i][j][k];
        int t = (i + 1) * 100 + (j + 1) * 10 + k + 1;
        if (t == 123 || t == 312 || t == 231) {
            return 1.0;
        }

        if (t == 132 || t == 321 || t == 213) {
            return -1.0;
        }
        return 0.0;
    }
};
#endif  // SRC_SOLVER_ALGEBRA_HPP_
