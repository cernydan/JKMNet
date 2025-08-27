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

        static bool appendMetricsCsv(const std::string &path, 
            const std::vector<std::pair<std::string,
            double>> &metrics, 
            const std::string &id = "",
            bool verbose = true);  //!< Append a labeled row of metrics into CSV file

        static bool computeAndAppendFinalMetrics(const Eigen::MatrixXd &Y_true, 
            const Eigen::MatrixXd &Y_pred,
            const std::string &outCsv, 
            const std::string &id = "",
            bool verbose = true);  //!< Compute final metrics for matrix pair and append a single row into CSV oputput file

    protected:

    private:

};

#endif // METRICS_HPP
