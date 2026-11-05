/**
 * @file Metrics.cpp
 * @brief Implements Metrics — evaluation metrics (MSE, RMSE, PI, NSE, KGE, PBIAS, RSR) and CSV export helpers.
 */
/**
 * @file Metrics.cpp
 * @brief Implements Metrics — evaluation metrics (MSE, RMSE, PI, NSE, KGE, PBIAS, RSR) and CSV export helpers.
 */
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
#include <map>
#include <algorithm>

/// Mean squared error between two vectors.
/**
 * @throw std::invalid_argument If the vector sizes differ.
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

/// Mean squared error between two matrices.
/**
 * @throw std::invalid_argument If the matrix dimensions differ.
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

/// Root mean squared error between two vectors (delegates to mse()).
double Metrics::rmse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
    double rmse_result = std::sqrt(mse(y_true, y_pred));

    return rmse_result;
}

/// Root mean squared error between two matrices (delegates to mse()).
double Metrics::rmse(const Eigen::MatrixXd& Y_true, const Eigen::MatrixXd& Y_pred) {
    double rmse_result = std::sqrt(mse(Y_true, Y_pred));

    return rmse_result;
}

/// Persistence index between two vectors (uses a one-step-back persistence baseline).
/**
 * @throw std::invalid_argument If the vector sizes differ.
 */
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

/// Nash–Sutcliffe efficiency between two vectors.
/**
 * @throw std::invalid_argument If the vector sizes differ.
 */
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

/// Kling–Gupta efficiency between two vectors (sample statistics, n-1 denominators).
/**
 * @throw std::invalid_argument If the vector sizes differ.
 */
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

/// Percent bias between two vectors; returns NaN when the observed sum is (near) zero.
/**
 * @throw std::invalid_argument If the vector sizes differ.
 */
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

/// Ratio of RMSE to the sample standard deviation of the observations.
/**
 * Returns NaN for fewer than 2 samples or (near) zero standard deviation.
 * @throw std::invalid_argument If the vector sizes differ.
 */
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

/// Append a labeled row of metrics into a CSV file (header written on first use).
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

/// Append a labeled row of metrics into CSV files — each metric type in a separate file.
bool Metrics::appendMetricsCsvSplit(const std::string &basePath,
                                    const std::vector<std::pair<std::string,double>> &metrics,
                                    const std::string &id,
                                    const std::string &timeStr,
                                    bool verbose)
{
    if (metrics.empty()) {
        if (verbose) std::cerr << "[Metrics::appendMetricsCsvSplit] No metrics provided\n";
        return false;
    }

    // Group by metric type (e.g. MSE, RMSE)
    std::map<std::string, std::vector<std::pair<std::string,double>>> grouped;
    for (const auto &kv : metrics) {
        std::string metricName = kv.first.substr(0, kv.first.find("_"));
        grouped[metricName].push_back(kv);
    }

    // For each metric type
    for (auto &group : grouped) {
        const std::string &metricType = group.first;
        auto &metricPairs = group.second;

        // Sort columns for stable header order
        std::sort(metricPairs.begin(), metricPairs.end(),
                  [](auto &a, auto &b){ return a.first < b.first; });

        std::string outFile = basePath + "_" + metricType + ".csv";

        bool fileExists = std::filesystem::exists(outFile);
        std::ofstream ofs(outFile, std::ios::app);
        if (!ofs.is_open()) {
            std::cerr << "[Metrics::appendMetricsCsvSplit] Cannot open file: " << outFile << "\n";
            continue;
        }

        ofs << std::setprecision(12);

        // Write header if new file
        if (!fileExists) {
            ofs << "time,id";
            for (auto &p : metricPairs) ofs << "," << p.first;
            ofs << "\n";
        }

        // Write row
        std::string ts = timeStr.empty() ? "NA" : timeStr;
        ofs << ts << "," << id;
        for (auto &p : metricPairs) ofs << "," << p.second;
        ofs << "\n";

        ofs.close();

        if (verbose) {
            //std::cout << "[Metrics::appendMetricsCsvSplit] Appended metrics for "
            //          << metricType << " into " << outFile << "\n";
        }
    }

    return true;
}

/// Compute per-column metrics for a matrix pair and append a single row into a CSV output file.
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

/// Save run information into a CSV file (header written on first use).
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

/// Save MSEs from training and validation into a CSV file.
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

/// Save a matrix of metric values into a CSV file.
bool Metrics::saveMetricsCsv(const std::string &path,
                            const Eigen::MatrixXd &errors,
                            bool verbose) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[Metrics::saveMetricsCsv] Cannot open file: " << path << "\n";
        return false;
    }

    ofs << "epochs_part";
    for (Eigen::Index i = 0; i < errors.cols(); ++i) {
        ofs << ",h_" << i+1;
    }
    ofs << "\n";
    ofs << std::setprecision(12);

    for (int i = 0; i < errors.rows(); ++i) {
        ofs << (i + 1) << ",";        // epoch index (1-based)

        for (Eigen::Index j = 0; j < errors.cols(); ++j) {
            double v = errors(i, j);
            if (std::isnan(v)) {
                ofs << "NA" << ",";   // e.g. tail rows after early convergence
            } else {
                ofs << v << ",";
            }
        }
        ofs << "\n";
    }

    ofs.close();
    if (verbose) {
        //std::cout << "[Metrics] Saved error curve to '" << path 
        //          << "' with " << errors.rows() << " epochs\n";
    }
    return true;
}

/// Helper function for adding a run ID into a filename.
std::string Metrics::addRunIdToFilename(const std::string &path, const std::string &run_id) {
    std::filesystem::path p(path);
    std::string stem = p.stem().string();
    std::string ext  = p.extension().string();
    return (p.parent_path() / (stem + "_" + run_id + ext)).string();
}

/// Buffer a set of metrics into a thread-safe in-memory structure for later export.
void Metrics::bufferMetrics(MetricBuffer &buffer,
                            const std::vector<std::pair<std::string,double>> &metrics,
                            const std::string &id,
                            const std::string &timeStr)
{
    std::lock_guard<std::mutex> lock(buffer.mtx);

    for (const auto &kv : metrics) {
        // kv.first = "MSE_h1", kv.second = 0.54
        std::string line = timeStr + "," + id + "," + std::to_string(kv.second);
        buffer.data[kv.first].push_back({id, kv.second});
    }
}
/// Flush the contents of a MetricBuffer to one or more CSV files (grouped by metric type).
bool Metrics::flushMetricsBufferToCsv(const MetricBuffer &buffer,
                                      const std::string &basePath,
                                      bool verbose)
{
    if (buffer.data.empty()) {
        if (verbose) std::cerr << "[Metrics::flushMetricsBufferToCsv] Buffer is empty.\n";
        return false;
    }

    // Group by metric type (e.g., MSE, RMSE)
    std::map<std::string, std::vector<std::string>> metricNames;
    for (const auto &kv : buffer.data) {
        std::string metricType = kv.first.substr(0, kv.first.find("_"));
        metricNames[metricType].push_back(kv.first);
    }

    for (auto &g : metricNames) {
        const std::string &metricType = g.first;
        auto &metricKeys = g.second;

        // Sort for stable column order
        std::sort(metricKeys.begin(), metricKeys.end());

        std::string outFile = basePath + "_" + metricType + ".csv";
        std::ofstream ofs(outFile);
        if (!ofs.is_open()) {
            std::cerr << "[Metrics::flushMetricsBufferToCsv] Cannot open file: " << outFile << "\n";
            continue;
        }

        ofs << std::setprecision(12);
        ofs << "id";
        for (const auto &name : metricKeys) {
            ofs << "," << name;
        }
        ofs << "\n";

        // Determine number of rows (based on first metric)
        size_t nRows = buffer.data.at(metricKeys[0]).size();
        for (size_t r = 0; r < nRows; ++r) {
            ofs << buffer.data.at(metricKeys[0])[r].first; // ID
            for (const auto &name : metricKeys) {
                ofs << "," << buffer.data.at(name)[r].second;
            }
            ofs << "\n";
        }

        ofs.close();
        if (verbose) {
            std::cout << "[Metrics] Wrote " << nRows << " rows to " << outFile << "\n";
        }
    }

    return true;
}

/// Build a per-run metric filename ("<stem>_run<run>_<metricName><ext>") from a base path.
std::string Metrics::makeMetricFilename(const std::string &basePath,
                                        int run,
                                        const std::string &metricName)
{
    namespace fs = std::filesystem;
    fs::path p(basePath);
    fs::path dir = p.parent_path();
    std::string stem = p.stem().string();       // e.g. "metrics"
    std::string ext  = p.extension().string();  // e.g. ".csv"

    std::ostringstream oss;
    oss << stem << "_run" << run << "_" << metricName << ext;

    fs::path newPath = dir / oss.str();
    return newPath.string();
}

/// Append one row of values to a CSV file, optionally writing the header first.
/**
 * Creates parent directories if needed.
 */
void Metrics::saveMetricRow(const std::string &path,
                            const std::vector<std::string> &colNames,
                            const std::vector<double> &values,
                            bool writeHeader)
{
    namespace fs = std::filesystem;
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    std::ofstream ofs(path, std::ios::out | std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[Metrics] Failed to open: " << path << "\n";
        return;
    }

    ofs << std::setprecision(10);
    if (writeHeader) {
        for (size_t i = 0; i < colNames.size(); ++i) {
            ofs << colNames[i];
            if (i + 1 < colNames.size()) ofs << ",";
        }
        ofs << "\n";
    }

    for (size_t i = 0; i < values.size(); ++i) {
        ofs << values[i];
        if (i + 1 < values.size()) ofs << ",";
    }
    ofs << "\n";
}

/// Calculate all metrics (currently disabled).
//Eigen::VectorXd Metrics::calcMetricsAll() {
    // Eigen::VectorXd allCriteria = Eigen::VectorXd(5);


    // allCriteria(0) = mse(yt, yp);
    // allCriteria(1) = rmse(yt, yp);
    // allCriteria(2) = pi(yt, yp);
    // allCriteria(3) = ns(yt, yp);
    // allCriteria(4) = kge(yt, yp);

    // return allCriteria;
//}

/**
 * Number of recorded epoch-metric rows:
 * every metricsAfterXEpochs-th epoch plus the final epoch when it is not a multiple
 */
size_t Metrics::epochMetricRows(int maxIterations, int metricsAfterXEpochs) {
    if (metricsAfterXEpochs == 0)
        return 0;  // 0 = epoch-metric recording disabled
    if (maxIterations <= 0 || metricsAfterXEpochs < 0)
        throw std::invalid_argument("Metrics::epochMetricRows: maxIterations must be positive and metricsAfterXEpochs non-negative");
    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("Metrics::epochMetricRows: metricsAfterXEpochs can't be bigger than maxIterations");
    if (metricsAfterXEpochs == 1)
        return static_cast<size_t>(maxIterations);
    size_t rows = static_cast<size_t>(maxIterations / metricsAfterXEpochs);
    if (maxIterations % metricsAfterXEpochs != 0)
        ++rows;
    return rows;
}

/**
 * Record decision for a 1-based epoch: every metricsAfterXEpochs-th epoch,
 * and always the final epoch (so the last row reflects the final model)
 */
bool Metrics::isEpochMetricRecord(int epoch, int metricsAfterXEpochs, int maxIterations) {
    if (metricsAfterXEpochs <= 0)
        return false;  // 0 = epoch-metric recording disabled
    if (epoch < 1 || epoch > maxIterations)
        throw std::invalid_argument("Metrics::isEpochMetricRecord: epoch out of range (1-based)");
    return (epoch % metricsAfterXEpochs == 0) || (epoch == maxIterations);
}

/**
 * Row index for a recording epoch: regular records go to epoch/step - 1,
 * the extra final epoch (non-divisible case) goes to the last row
 */
size_t Metrics::epochMetricRowIndex(int epoch, int metricsAfterXEpochs, int maxIterations) {
    if (!isEpochMetricRecord(epoch, metricsAfterXEpochs, maxIterations))
        throw std::invalid_argument("Metrics::epochMetricRowIndex: epoch is not a recording epoch");
    if (epoch % metricsAfterXEpochs == 0)
        return static_cast<size_t>(epoch / metricsAfterXEpochs - 1);
    return epochMetricRows(maxIterations, metricsAfterXEpochs) - 1;
}

/**
 * Allocate the 14 metric matrices (0-6 calib, 7-13 valid), NaN-prefilled so
 * that tail rows after an early-converged training read as NA, not garbage
 */
std::vector<Eigen::MatrixXd> Metrics::initEpochMetrics(int maxIterations, int metricsAfterXEpochs, int nOutputs) {
    if (metricsAfterXEpochs <= 0)
        return {};  // recording disabled
    size_t rows = epochMetricRows(maxIterations, metricsAfterXEpochs);
    std::vector<Eigen::MatrixXd> resultMetrics;
    resultMetrics.reserve(14);
    for (int i = 0; i < 14; ++i) {
        Eigen::MatrixXd prepMat = Eigen::MatrixXd::Constant(
            static_cast<Eigen::Index>(rows), nOutputs,
            std::numeric_limits<double>::quiet_NaN());
        resultMetrics.push_back(prepMat);
    }
    return resultMetrics;
}

/**
 * Compute the 7 metrics (MSE, RMSE, PI, NS, KGE, PBIAS, RSR) per output column
 * for calibration (matrices 0-6) and validation (7-13) and store them into row `row`
 */
void Metrics::fillEpochMetricRow(std::vector<Eigen::MatrixXd> &resultMetrics, size_t row,
    const Eigen::MatrixXd &YtrueCal, const Eigen::MatrixXd &YpredCal,
    const Eigen::MatrixXd &YtrueVal, const Eigen::MatrixXd &YpredVal)
{
    if (resultMetrics.size() != 14)
        throw std::invalid_argument("Metrics::fillEpochMetricRow: expected 14 metric matrices");
    if (YtrueCal.rows() != YpredCal.rows() || YtrueCal.cols() != YpredCal.cols() ||
        YtrueVal.rows() != YpredVal.rows() || YtrueVal.cols() != YpredVal.cols())
        throw std::invalid_argument("Metrics::fillEpochMetricRow: matrix dimensions must match");
    if (YtrueCal.cols() != resultMetrics[0].cols())
        throw std::invalid_argument("Metrics::fillEpochMetricRow: number of outputs doesn't match allocation");
    if (row >= static_cast<size_t>(resultMetrics[0].rows()))
        throw std::out_of_range("Metrics::fillEpochMetricRow: row out of range");

    for (int c = 0; c < YtrueCal.cols(); ++c) {
        resultMetrics[0](row, c)  = mse(YtrueCal.col(c).eval(), YpredCal.col(c).eval());
        resultMetrics[1](row, c)  = rmse(YtrueCal.col(c).eval(), YpredCal.col(c).eval());
        resultMetrics[2](row, c)  = pi(YtrueCal.col(c).eval(), YpredCal.col(c).eval());
        resultMetrics[3](row, c)  = ns(YtrueCal.col(c).eval(), YpredCal.col(c).eval());
        resultMetrics[4](row, c)  = kge(YtrueCal.col(c).eval(), YpredCal.col(c).eval());
        resultMetrics[5](row, c)  = pbias(YtrueCal.col(c).eval(), YpredCal.col(c).eval());
        resultMetrics[6](row, c)  = rsr(YtrueCal.col(c).eval(), YpredCal.col(c).eval());

        resultMetrics[7](row, c)  = mse(YtrueVal.col(c).eval(), YpredVal.col(c).eval());
        resultMetrics[8](row, c)  = rmse(YtrueVal.col(c).eval(), YpredVal.col(c).eval());
        resultMetrics[9](row, c)  = pi(YtrueVal.col(c).eval(), YpredVal.col(c).eval());
        resultMetrics[10](row, c) = ns(YtrueVal.col(c).eval(), YpredVal.col(c).eval());
        resultMetrics[11](row, c) = kge(YtrueVal.col(c).eval(), YpredVal.col(c).eval());
        resultMetrics[12](row, c) = pbias(YtrueVal.col(c).eval(), YpredVal.col(c).eval());
        resultMetrics[13](row, c) = rsr(YtrueVal.col(c).eval(), YpredVal.col(c).eval());
    }
}

/**
 * Save the 14 "*_epochs" files for one run (one file per metric per split)
 */
void Metrics::saveEpochMetricFiles(const std::string &metricsCalBase, const std::string &metricsValBase,
    int runIndex, const std::vector<Eigen::MatrixXd> &resultMetrics)
{
    if (resultMetrics.empty())
        return;  // recording was disabled
    if (resultMetrics.size() != 14)
        throw std::invalid_argument("Metrics::saveEpochMetricFiles: expected 14 metric matrices");

    static const char* names[7] = {"MSE", "RMSE", "PI", "NS", "KGE", "PBIAS", "RSR"};
    for (int i = 0; i < 7; ++i) {
        saveMetricsCsv(makeMetricFilename(metricsCalBase, runIndex, std::string(names[i]) + "_epochs"),
                       resultMetrics[i]);
        saveMetricsCsv(makeMetricFilename(metricsValBase, runIndex, std::string(names[i]) + "_epochs"),
                       resultMetrics[7 + i]);
    }
}