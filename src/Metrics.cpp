#include "Metrics.hpp"
#include <cmath>
#include <limits>

/**
 * Mean squared error between two vectors
 */
double Metrics::mse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("Metrics::mse: vector sizes differ");
    
    const Eigen::VectorXd diff = y_pred - y_true;
    double sumsq = diff.squaredNorm();  // squaredNorm returns sum of squares
    const double n = static_cast<double>(y_true.size());
    double mse_result = sumsq / n;

    return mse_result;
}

/**
 * Mean squared error between two matrices
 */
double Metrics::mse(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& Y_pred) {
    if (Y_true.rows() != Y_pred.rows() || Y_true.cols() != Y_pred.cols())
        throw std::invalid_argument("Metrics::mse (matrix): dimensions differ");

    Eigen::MatrixXd diff = Y_pred - Y_true;
    double sumsq = diff.array().square().sum();
    const double n = static_cast<double>(Y_true.rows()) * static_cast<double>(Y_true.cols());
    double mse_result = sumsq / n;

    return mse_result;
}

/**
 * Root mean squared error between two vectors
 */
double Metrics::rmse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    double rmse_result = std::sqrt(mse(y_true, y_pred));

    return rmse_result;
}

/**
 * Root mean squared error between two matrices
 */
double Metrics::rmse(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& Y_pred) {
    double rmse_result = std::sqrt(mse(Y_true, Y_pred));

    return rmse_result;
}
