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

double Metrics::pbias(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Metrics::pbias: vector sizes differ");
    }

    // PBIAS = 100 * sum(pred - true) / sum(true)
    const double num = (y_pred - y_true).sum();
    const double den = y_true.sum();

    if (std::fabs(den) < 1e-12) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return 100.0 * (num / den);
}

double Metrics::rsr(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Metrics::rsr: vector sizes differ");
    }

    // RSR = RMSE / sd(y_true)
    const double rmse_val = Metrics::rmse(y_true, y_pred);

    const int n = static_cast<int>(y_true.size());
    if (n < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double mean_true = y_true.mean();
    double ssd = 0.0; // sum of squared deviations
    for (int i = 0; i < n; ++i) {
        const double d = y_true(i) - mean_true;
        ssd += d * d;
    }
    const double sd_true = std::sqrt(ssd / (n - 1));

    if (sd_true < 1e-12) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return rmse_val / sd_true;
}

/**
 * Append a labeled row of metrics into CSV file
 */
bool Metrics::appendMetricsCsv(const std::string &path,
                               const std::vector<std::pair<std::string,double>> &metrics,
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
        std::cerr << "[Metrics::appendMetricsCsv] Cannot open file: " << path << "\n";
        return false;
    }
    ofs << std::setprecision(12);

    // write header if needed
    if (!file_exists) {
        ofs << "time,id";
        for (auto &kv : metrics) ofs << "," << kv.first;
        ofs << "\n";
    }

    // generate timestamp
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

    // write row
    ofs << timeStr << "," << id;
    for (auto &kv : metrics) ofs << "," << kv.second;
    ofs << "\n";

    ofs.flush();
    bool ok = !(ofs.fail() || !ofs.good());
    ofs.close();

    if (verbose && ok) {
        // std::cout << "[Metrics] Appended metrics row to " << path << " for id=" << id << "\n";
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
                                           bool verbose) 
{
    if ((Y_true.rows() != Y_pred.rows()) || (Y_true.cols() != Y_pred.cols())) {
        std::cerr << "[Metrics::computeAndAppendFinalMetrics] Shape mismatch\n";
        return false;
    }

    int nCols = Y_true.cols();

    // Generate timestamp string
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

    bool file_exists = false;
    {
        std::ifstream ifs(outCsv);
        file_exists = ifs.good();
    }

    std::ofstream ofs(outCsv, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[Metrics::computeAndAppendFinalMetrics] Cannot open file: " << outCsv << "\n";
        return false;
    }
    ofs << std::setprecision(12);

    // Write header if first time
    if (!file_exists) {
        ofs << "time,id";
        for (int c = 0; c < nCols; ++c) {
            ofs << ",MSE_h" << (c+1)
                << ",RMSE_h" << (c+1)
                << ",PI_h" << (c+1)
                << ",NS_h" << (c+1)
                << ",KGE_h" << (c+1)
                << ",pbias_h" << (c+1)
                << ",rsr_h" << (c+1);
        }
        ofs << "\n";
    }

    // Write one row
    ofs << timeStr << "," << id;
    for (int c = 0; c < nCols; ++c) {
        Eigen::VectorXd yt = Y_true.col(c);
        Eigen::VectorXd yp = Y_pred.col(c);

        double mse = Metrics::mse(yt, yp);
        double rmse = Metrics::rmse(yt, yp);
        double pi = Metrics::pi(yt, yp);
        double ns = Metrics::ns(yt, yp);
        double kge = Metrics::kge(yt, yp);
        double pbias = Metrics::pbias(yt, yp);
        double rsr = Metrics::rsr(yt, yp);

        ofs << "," << mse
            << "," << rmse
            << "," << pi
            << "," << ns
            << "," << kge
            << "," << pbias
            << "," << rsr;
    }
    ofs << "\n";

    ofs.close();
    if (verbose) {
        // std::cout << "[Metrics] Appended per-output metrics to '" << outCsv << "'\n";
    }
    return true;
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
            //std::cout << "[Metrics] Saved run info to '" << path << "' (time: " << timeStr << ", id: " << idToWrite << ")\n";
        } else {
            std::cerr << "[Metrics] Failed to save run info to '" << path << "'\n";
        }
    }
    return ok;
}

/**
 * Save MSEs from training and validation into CSV file 
 */
bool Metrics::saveErrorsCsv(const std::string &path,
                            const Eigen::MatrixXd &errors,
                            bool verbose) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[Metrics::saveErrorsCsv] Cannot open file: " << path << "\n";
        return false;
    }

    ofs << "epoch,mse_train,mse_valid\n";
    ofs << std::setprecision(12);

    for (int i = 0; i < errors.rows(); ++i) {
        ofs << (i + 1) << ","        // epoch index (1-based)
            << errors(i, 0) << ","
            << errors(i, 1) << "\n";
    }

    ofs.close();
    if (verbose) {
        //std::cout << "[Metrics] Saved error curve to '" << path 
        //          << "' with " << errors.rows() << " epochs\n";
    }
    return true;
}

/**
 * Helper function for adding run ID into filename
 */
std::string Metrics::addRunIdToFilename(const std::string &path, const std::string &run_id) {
    std::filesystem::path p(path);
    std::string stem = p.stem().string();
    std::string ext  = p.extension().string();
    return (p.parent_path() / (stem + "_" + run_id + ext)).string();
}

/**
 * Calculate all metrics
 */
//Eigen::VectorXd Metrics::calcMetricsAll() {
    // Eigen::VectorXd allCriteria = Eigen::VectorXd(5);


    // allCriteria(0) = mse(yt, yp);
    // allCriteria(1) = rmse(yt, yp);
    // allCriteria(2) = pi(yt, yp);
    // allCriteria(3) = ns(yt, yp);
    // allCriteria(4) = kge(yt, yp);
    
    // return allCriteria;
//}