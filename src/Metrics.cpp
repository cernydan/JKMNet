#include "Metrics.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <limits>
#include <time.h>
#include <filesystem>

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

double Metrics::pi(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("Metrics::pi: vector sizes differ");

    Eigen::VectorXd y_persist(y_true.size());
    y_persist(0) = y_true(0);
    for (int i = 1; i < y_true.size(); ++i) {
        y_persist(i) = y_true(i - 1);
    }

    double mse_model = mse(y_true, y_pred);
    double mse_persist = mse(y_true, y_persist);

    return 1.0 - (mse_model / mse_persist);
}

double Metrics::ns(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("Metrics::ns: vector sizes differ");

    double mean_y = y_true.mean();

    Eigen::VectorXd diff_model = y_pred - y_true;
    Eigen::VectorXd diff_mean = y_true.array() - mean_y;

    double numerator = diff_model.squaredNorm();       // ∑(y_pred - y_true)^2
    double denominator = diff_mean.squaredNorm();      // ∑(y_true - mean)^2

    return 1.0 - (numerator / denominator);
}

double Metrics::kge(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("Metrics::kge: vector sizes differ");

    const double mean_true = y_true.mean();
    const double mean_pred = y_pred.mean();

    const Eigen::VectorXd centered_true = y_true.array() - mean_true;
    const Eigen::VectorXd centered_pred = y_pred.array() - mean_pred;

    double std_true = std::sqrt(centered_true.squaredNorm() / (y_true.size() - 1));
    double std_pred = std::sqrt(centered_pred.squaredNorm() / (y_pred.size() - 1));

    double r = centered_true.dot(centered_pred) / ((y_true.size() - 1) * std_true * std_pred);
    double alpha = std_pred / std_true;
    double beta = mean_pred / mean_true;

    return 1.0 - std::sqrt(std::pow(r - 1.0, 2) + std::pow(alpha - 1.0, 2) + std::pow(beta - 1.0, 2));
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
#if defined(_WIN32)
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
    double pi_write = Metrics::pi(Y_true.col(0), Y_pred.col(0));
    double ns_write = Metrics::ns(Y_true.col(0), Y_pred.col(0));
    double kge_write = Metrics::kge(Y_true.col(0), Y_pred.col(0));

    std::vector<std::pair<std::string,double>> metrics;
    metrics.emplace_back("MSE", mse_write);
    metrics.emplace_back("RMSE", rmse_write);
    metrics.emplace_back("PI", pi_write);
    metrics.emplace_back("NS", ns_write);
    metrics.emplace_back("KGE", kge_write);

    return appendMetricsCsv(outCsv, metrics, id, verbose);
}

/**
 * Save run information into CSV file
 */
bool Metrics::appendRunInfoCsv(const std::string &path,
                               int iterations,
                               double loss,
                               bool converged,
                               double runtime_sec,
                               const std::string &id,
                               bool verbose) {
    // check if file exists
    bool file_exists = false;
    {
        std::ifstream ifs(path);
        file_exists = ifs.good();
    }

    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[Metrics::appendRunInfoCsv] Cannot open file for writing: " << path << "\n";
        return false;
    }
    ofs << std::setprecision(12);

    // write header if needed
    if (!file_exists) {
        ofs << "time,id,iterations,loss,converged,runtime_sec\n";
    }

    // generate timestamp string
    using namespace std::chrono;
    auto t = system_clock::now();
    std::time_t tt = system_clock::to_time_t(t);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char timebuf[64];
    std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", &tm);
    std::string timeStr(timebuf);

    // fallback ID if empty
    std::string idToWrite = id;
    if (idToWrite.empty()) {
        char idbuf[32];
        std::strftime(idbuf, sizeof(idbuf), "%Y%m%d-%H%M%S", &tm);
        idToWrite = std::string(idbuf);
    }

    // write row
    ofs << timeStr << "," << idToWrite
        << "," << iterations
        << "," << loss
        << "," << (converged ? 1 : 0)
        << "," << runtime_sec << "\n";

    ofs.flush();
    bool ok = !(ofs.fail() || !ofs.good());
    ofs.close();

    if (verbose) {
        if (ok) {
            std::cout << "[Metrics] Saved run info to '" << path
                      << "' (time: " << timeStr << ", id: " << idToWrite << ")\n";
        } else {
            std::cerr << "[Metrics] Failed to save run info to '" << path << "'\n";
        }
    }
    return ok;
}
