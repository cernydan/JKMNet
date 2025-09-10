#include "Metrics.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <ctime>
#include <sstream>
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

/**
 * Append a labeled row of metrics into CSV file
 */
bool Metrics::appendMetricsCsv(const std::string &path,
    const std::vector<std::pair<std::string,double>> &metrics,
    const std::string &id,
    bool verbose) {
    
    // Does file already exist?
    bool file_exists = false;
    {
        std::ifstream ifs(path);
        file_exists = ifs.good();
    }

    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[Metrics::appendMetricsCsv] Cannot open file for writing: " << path << "\n";
        return false;
    }
    ofs << std::setprecision(12);

    // if new file, write header: time,id,<metric names...>
    if (!file_exists) {
        ofs << "time,id";
        if (!metrics.empty()) ofs << ",";
        for (size_t i = 0; i < metrics.size(); ++i) {
            ofs << metrics[i].first;
            if (i + 1 < metrics.size()) ofs << ",";
        }
        ofs << "\n";
    } else {
        // if file exists, we could validate header vs metrics names...
    }

    // Generate timestamp string (human readable)
    using namespace std::chrono;
    auto t = system_clock::now();
    std::time_t tt = system_clock::to_time_t(t);
    std::tm tm{};
#if defined(_MSC_VER)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char timebuf[64];
    std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", &tm);
    std::string timeStr(timebuf);

    // Determine ID to write: if id empty, use compact timestamp as fallback
    std::string idToWrite = id;
    if (idToWrite.empty()) {
        char idbuf[32];
        std::strftime(idbuf, sizeof(idbuf), "%Y%m%d-%H%M%S", &tm);
        idToWrite = std::string(idbuf);
    }

    // write the row: time, id, metric values
    ofs << timeStr << "," << idToWrite;
    if (!metrics.empty()) ofs << ",";
    for (size_t i = 0; i < metrics.size(); ++i) {
        ofs << metrics[i].second;
        if (i + 1 < metrics.size()) ofs << ",";
    }
    ofs << "\n";
    
    // flush and check
    ofs.flush();
    bool ok = !(ofs.fail() || !ofs.good());
    ofs.close();

    // inform about saving of the metrics
    if (verbose) {
        if (ok) {
            std::cout << "[Metrics] Saved metrics to '" << path << "' (time: " << timeStr << ", id: " << idToWrite << ")\n";
        } else {
            std::cerr << "[Metrics] Failed to save metrics to '" << path << "'\n";
        }
    }
    
    return ok;
}


/**
 * Compute final metrics for matrix pair and append a single row into CSV oputput file
 */
bool Metrics::computeAndAppendFinalMetrics(const Eigen::MatrixXd &Y_true,
    const Eigen::MatrixXd &Y_pred,
    const std::string &outCsv,
    const std::string &id,
    bool verbose) {

    if ((Y_true.rows() != Y_pred.rows()) || (Y_true.cols() != Y_pred.cols())) {
        std::cerr << "[Metrics::computeAndAppendFinalMetrics] Shape mismatch\n";
        return false;
    }

    double mse_write = Metrics::mse(Y_true, Y_pred);
    double rmse_write = Metrics::rmse(Y_true, Y_pred);

    std::vector<std::pair<std::string,double>> metrics;
    metrics.emplace_back("MSE", mse_write);
    metrics.emplace_back("RMSE", rmse_write);

    return appendMetricsCsv(outCsv, metrics, id, verbose);
}