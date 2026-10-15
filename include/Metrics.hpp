/**
 * @file Metrics.hpp
 * @brief Declares the MetricBuffer struct and the Metrics class with evaluation metrics and CSV export utilities.
 */
#ifndef METRICS_HPP
#define METRICS_HPP

#include "eigen-3.4/Eigen/Dense"
#include <map>
#include <vector>
#include <string>
#include <mutex>

/// Thread-safe in-memory buffer for metrics awaiting export to CSV.
/**
 * Metrics are grouped by metric name (e.g. "MSE_h1"); each name maps to a list
 * of (id, value) pairs in insertion order. Access is guarded by the mutex.
 */
struct MetricBuffer {
    std::map<std::string, std::vector<std::pair<std::string, double>>> data;  ///< Metric name -> list of (id, value) pairs
    std::mutex mtx;  ///< Mutex guarding concurrent access to data
};

/// Static utility class providing error metrics and CSV export helpers.
/**
 * Offers common hydrological/regression metrics (MSE, RMSE, PI, NSE, KGE, PBIAS,
 * RSR) as well as helpers for appending metric rows to CSV files, buffering
 * metrics across threads, and saving error curves. All methods are static.
 */
class Metrics {
    public:
        //!< Mean Squared Error

        /// Mean squared error between two vectors (element-wise).
        /**
         * MSE = sum((y_pred - y_true)^2) / n.
         * @param y_true Vector of observed values.
         * @param y_pred Vector of predicted values (same size as y_true).
         * @return Scalar MSE value.
         * @throw std::invalid_argument If the vector sizes differ.
         */
        static double mse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);

        /// Mean squared error between two matrices (element-wise average across all entries).
        /**
         * @param Y_true Matrix of observed values.
         * @param Y_pred Matrix of predicted values (same dimensions as Y_true).
         * @return MSE averaged over all matrix entries.
         * @throw std::invalid_argument If the matrix dimensions differ.
         */
        static double mse(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& Y_pred);

        //!< Root Mean Squared Error

        /// Root mean squared error between two vectors (element-wise).
        /**
         * @param y_true Vector of observed values.
         * @param y_pred Vector of predicted values (same size as y_true).
         * @return sqrt of the vector MSE.
         * @throw std::invalid_argument If the vector sizes differ.
         */
        static double rmse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);

        /// Root mean squared error between two matrices (element-wise average across all entries).
        /**
         * @param Y_true Matrix of observed values.
         * @param Y_pred Matrix of predicted values (same dimensions as Y_true).
         * @return sqrt of the matrix MSE.
         * @throw std::invalid_argument If the matrix dimensions differ.
         */
        static double rmse(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& Y_pred);

        //!< PI - Persistency Index

        /// Persistence index between two vectors (element-wise).
        /**
         * PI = 1 - MSE(y_true, y_pred) / MSE(y_true, persistence(y_true)), where the
         * persistence forecast repeats the previous observed value.
         * @param y_true Vector of observed values.
         * @param y_pred Vector of predicted values (same size as y_true).
         * @return PI value; 1.0 is a perfect forecast, 0.0 matches the persistence baseline.
         * @throw std::invalid_argument If the vector sizes differ.
         */
        static double pi(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);

        //!< NI - Nash–Sutcliffe Efficiency

        /// Nash–Sutcliffe efficiency between two vectors (element-wise).
        /**
         * NS = 1 - sum((y_pred - y_true)^2) / sum((y_true - mean(y_true))^2).
         * @param y_true Vector of observed values.
         * @param y_pred Vector of predicted values (same size as y_true).
         * @return NSE value; 1.0 is a perfect fit.
         * @throw std::invalid_argument If the vector sizes differ.
         */
        static double ns(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);

        //!< KGE – Kling–Gupta Efficiency

        /// Kling–Gupta efficiency between two vectors.
        /**
         * KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), combining correlation,
         * variability ratio and bias ratio.
         * @param y_true Vector of observed values.
         * @param y_pred Vector of predicted values (same size as y_true).
         * @return KGE value; 1.0 is a perfect fit.
         * @throw std::invalid_argument If the vector sizes differ.
         */
        static double kge(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);

        //!< PBIAS - Percent Bias

        /// Percent bias between two vectors.
        /**
         * PBIAS = 100 * sum(y_pred - y_true) / sum(y_true).
         * @param y_true Vector of observed values.
         * @param y_pred Vector of predicted values (same size as y_true).
         * @return Percent bias; NaN if sum(y_true) is (near) zero.
         * @throw std::invalid_argument If the vector sizes differ.
         */
        static double pbias(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);

        //!< RSR - RMSE to SD Ratio, (RMSE/ sd(obs))

        /// Ratio of RMSE to the standard deviation of the observations.
        /**
         * RSR = RMSE / sd(y_true).
         * @param y_true Vector of observed values.
         * @param y_pred Vector of predicted values (same size as y_true).
         * @return RSR value; NaN for fewer than 2 samples or (near) zero standard deviation.
         * @throw std::invalid_argument If the vector sizes differ.
         */
        static double rsr(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);

        /// Append a labeled row of metrics into a CSV file.
        /**
         * Creates the file with a header ("time,id,<metric names>") if it does not exist yet.
         * @param path Path of the CSV file.
         * @param metrics List of (metric name, value) pairs forming the row.
         * @param id Identifier written into the id column (may be empty).
         * @param verbose Unused; console output is currently disabled.
         * @return True if the row was written successfully.
         */
        static bool appendMetricsCsv(const std::string &path,
            const std::vector<std::pair<std::string,
            double>> &metrics,
            const std::string &id = "",
            bool verbose = true);

        /// Append a labeled row of metrics into CSV files — each metric type in a separate file.
        /**
         * Metric names are expected to carry a type prefix before the first underscore
         * (e.g. "MSE_h1"); metrics are grouped by that prefix and each group is written
         * to "<basePath>_<type>.csv". Headers are written for newly created files.
         * @param basePath Base path (without the "_<type>.csv" suffix).
         * @param metrics List of (metric name, value) pairs.
         * @param id Identifier written into the id column.
         * @param timeStr Timestamp string for the row; "NA" is written if empty.
         * @param verbose If true, warns when no metrics are provided.
         * @return False if metrics is empty, true otherwise (unwritable files are skipped).
         */
        static bool appendMetricsCsvSplit(const std::string &basePath,
            const std::vector<std::pair<std::string,double>> &metrics,
            const std::string &id = "",
            const std::string &timeStr = "",
            bool verbose = true);

        /// Compute final metrics for a matrix pair and append a single row into a CSV output file.
        /**
         * Each column of the matrices is treated as one output horizon; for each column
         * MSE, RMSE, PI, NS, KGE, PBIAS and RSR are computed and appended as one row
         * (header "time,id,MSE_h1,RMSE_h1,..." on first use).
         * @param Y_true Matrix of observed values (rows = samples, cols = horizons).
         * @param Y_pred Matrix of predicted values (same dimensions as Y_true).
         * @param outCsv Path of the CSV output file.
         * @param id Identifier written into the id column.
         * @param verbose Unused; console output is currently disabled.
         * @return True on success, false on shape mismatch or unwritable file.
         */
        static bool computeAndAppendFinalMetrics(const Eigen::MatrixXd &Y_true,
            const Eigen::MatrixXd &Y_pred,
            const std::string &outCsv,
            const std::string &id = "",
            bool verbose = false);

        /// Save run information into a CSV file.
        /**
         * Appends one row "time,id,iterations,loss,converged,runtime_sec"; the header is
         * written on first use. An empty id is replaced by a timestamp-based fallback.
         * @param path Path of the CSV file.
         * @param iterations Number of training iterations performed.
         * @param loss Final loss value.
         * @param converged Whether training converged (written as 1/0).
         * @param runtime_sec Wall-clock runtime in seconds.
         * @param id Run identifier (may be empty).
         * @param verbose If true, prints an error message when writing fails.
         * @return True if the row was written successfully.
         */
        static bool appendRunInfoCsv(const std::string &path,
            int iterations,
            double loss,
            bool converged,
            double runtime_sec,
            const std::string &id,
            bool verbose = true);

        /// Save MSEs from training and validation into a CSV file.
        /**
         * Writes header "epoch,mse_train,mse_valid" followed by one row per matrix row
         * (1-based epoch index in the first column).
         * @param path Path of the CSV file (overwritten).
         * @param errors Matrix with two columns: training and validation MSE per epoch.
         * @param verbose Unused; console output is currently disabled.
         * @return True if the file was written successfully.
         */
        static bool saveErrorsCsv(const std::string &path,
                          const Eigen::MatrixXd &errors,
                          bool verbose = true);

        /// Save a matrix of metric values into a CSV file.
        /**
         * Writes header "epochs_part,h_1,h_2,..." followed by one row per matrix row
         * (1-based epoch index in the first column). Rows end with a trailing comma.
         * @param path Path of the CSV file (overwritten).
         * @param errors Matrix of metric values (rows = epoch parts, cols = horizons).
         * @param verbose Unused; console output is currently disabled.
         * @return True if the file was written successfully.
         */
        static bool saveMetricsCsv(const std::string &path,
                            const Eigen::MatrixXd &errors,
                            bool verbose = true);

        /// Helper function for adding a run ID into a filename.
        /**
         * Inserts "_<run_id>" between the stem and the extension of the path.
         * @param path Original file path.
         * @param run_id Run identifier to insert.
         * @return New path of the form "<stem>_<run_id><ext>".
         */
        static std::string addRunIdToFilename(const std::string &path, const std::string &run_id);

        /// Buffer a set of metrics into a thread-safe in-memory structure for later export.
        /**
         * @param buffer Buffer to append into (locked internally).
         * @param metrics List of (metric name, value) pairs to store.
         * @param id Identifier stored alongside each value.
         * @param timeStr Timestamp string (currently unused by the implementation).
         */
        static void bufferMetrics(MetricBuffer &buffer,
                                  const std::vector<std::pair<std::string,double>> &metrics,
                                  const std::string &id = "",
                                  const std::string &timeStr = "");

        /// Flush the contents of a MetricBuffer to one or more CSV files.
        /**
         * Metric names are grouped by their prefix before the first underscore (e.g.
         * "MSE") and each group is written to "<basePath>_<type>.csv" (overwritten),
         * with the id column followed by one column per metric name.
         * @param buffer Buffer holding the accumulated metrics.
         * @param basePath Base path (without the "_<type>.csv" suffix).
         * @param verbose If true, reports empty buffers and per-file row counts.
         * @return False if the buffer is empty, true otherwise (unwritable files are skipped).
         * @note All metric lists within a group must have equal length; rows per group are
         * determined by the first metric of the group.
         */
        static bool flushMetricsBufferToCsv(const MetricBuffer &buffer,
                                            const std::string &basePath,
                                            bool verbose = true);

        /// Build a per-run metric filename from a base path.
        /**
         * Produces "<stem>_run<run>_<metricName><ext>" in the base path's directory.
         * @param basePath Base file path used for directory, stem and extension.
         * @param run Zero- or one-based run index inserted into the name.
         * @param metricName Metric name inserted into the name.
         * @return The composed file path.
         */
        static std::string makeMetricFilename(const std::string &basePath,int run,const std::string &metricName);

        /// Append one row of values to a CSV file, optionally writing the header first.
        /**
         * Parent directories are created if needed.
         * @param path Path of the CSV file (appended).
         * @param colNames Column names for the header line.
         * @param values Values of the row to append.
         * @param writeHeader If true, write the header line before the row.
         */
        static void saveMetricRow(const std::string &path,
            const std::vector<std::string> &colNames,
            const std::vector<double> &values,
            bool writeHeader = false);

        //Eigen::VectorXd calcMetricsAll(); //<! Calculate all metrics

    protected:

    private:

};

#endif // METRICS_HPP
