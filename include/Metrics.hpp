#ifndef METRICS_HPP
#define METRICS_HPP

#include "eigen-3.4/Eigen/Dense"

class Metrics {
    public:
        //!< Mean Squared Error
        static double mse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);  //!<  MSE between two vectors (element-wise)
        static double mse(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& Y_pred);  //!<  MSE between two matrices (element-wise average across all entries)

        //!< Root Mean Squared Error
        static double rmse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);  //!<  RMSE between two vectors (element-wise)
        static double rmse(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& Y_pred);  //!<  RMSE between two matrices (element-wise average across all entries)

    protected:

    private:

};

#endif // METRICS_HPP
