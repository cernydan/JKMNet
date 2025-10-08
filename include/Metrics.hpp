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

        //!< PI - Persistency Index
        static double pi(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred); //!< PI between two vectors (element-wise)

        //!< NI - Nash–Sutcliffe Efficiency
        static double ns(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred); //!< NS between two vectors (element-wise)

        //!< KGE – Kling–Gupta Efficiency
        static double kge(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred); //!< KGE between two vectors

        //!< PBIAS - Percent Bias
        static double pbias(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);

        //!< RSR - RMSE to SD Ratio, (RMSE/ sd(obs))
        static double rsr(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);
        
        static bool appendMetricsCsv(const std::string &path, 
            const std::vector<std::pair<std::string,
            double>> &metrics, 
            const std::string &id = "",
            bool verbose = true);  //!< Append a labeled row of metrics into CSV file

        static bool computeAndAppendFinalMetrics(const Eigen::MatrixXd &Y_true, 
            const Eigen::MatrixXd &Y_pred,
            const std::string &outCsv, 
            const std::string &id = "",
            bool verbose = false);  //!< Compute final metrics for matrix pair and append a single row into CSV oputput file

        static bool appendRunInfoCsv(const std::string &path,
            int iterations,
            double loss,
            bool converged,
            double runtime_sec,
            const std::string &id,
            bool verbose = true);  //!< Save run information into CSV file

        static bool saveErrorsCsv(const std::string &path,
                          const Eigen::MatrixXd &errors,
                          bool verbose = true);  //!< Save MSEs from training and validation into CSV file   
        
        static std::string addRunIdToFilename(const std::string &path, const std::string &run_id);  //!< Helper function for adding run ID into filename

    protected:

    private:

};

#endif // METRICS_HPP
