#include "Data.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <limits>

/**
 * Trim helper function
 */
static inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b-a);
}

/**
 * Helper function to parse a CSV line into fields (handles quotes)
 */
void Data::splitCSVLine(const std::string& line, std::vector<std::string>& outFields) {
    outFields.clear();
    outFields.reserve(16);

    std::string cur;
    bool inQuote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"' ) {
            // handle double quotes inside quotes ("")
            if (inQuote && i + 1 < line.size() && line[i+1] == '"') {
                cur.push_back('"');
                ++i; // skip second quote
            } else {
                inQuote = !inQuote;
            }
        } else if (c == ',' && !inQuote) {
            outFields.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    outFields.push_back(cur);
}

/**
 * Loads and filters the CSV file and returns number of loaded rows
 */
size_t Data::loadFilteredCSV(const std::string& path,
  const std::unordered_set<std::string>& idFilter,
  const std::vector<std::string>& keepColumns,  // names of numeric columns to extract (e.g. "T1","T2","T3","moisture")
  const std::string& timestampCol,  // name of timestamp column (e.g "hour_start", "date") 
  const std::string& idCol)  // ID of the selected sensor
  
  {
    // Clear or set needed variables
    m_timestamps.clear();  
    m_data.resize(0,0);
    m_colNames = keepColumns;

    // Deal with errors of the CSV file
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open CSV: " + path);
    }

    std::string headerLine;
    if (!std::getline(ifs, headerLine)) {
        throw std::runtime_error("CSV file empty: " + path);
    }

    // Parse header line into fields
    std::vector<std::string> headers;
    splitCSVLine(headerLine, headers);
  
    // Build header to index map (trim headers, reserve space, warn on duplicates)
    std::unordered_map<std::string, size_t> idxMap;
    idxMap.reserve(headers.size());
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h = trim(headers[i]);  // remove surrounding whitespace
      
        // replace original headers with trimmed form so later code uses same names
        headers[i] = h;

        auto it = idxMap.find(h);
        if (it != idxMap.end()) {
            std::cerr << "[Warning]: Duplicate header '" << h << "'; using last occurrence at column " << i << "\n";
        }
        idxMap[h] = i;
    }

    // Check if required columns exist
    if (idxMap.find(idCol) == idxMap.end()) {
        throw std::runtime_error("ID column not found: " + idCol);
    }
    if (idxMap.find(timestampCol) == idxMap.end()) {
        throw std::runtime_error("Timestamp column not found: " + timestampCol);
    }
    std::vector<size_t> keepIdx;
    keepIdx.reserve(keepColumns.size());
    for (auto &name : keepColumns) {
        auto it = idxMap.find(name);
        if (it == idxMap.end()) {
            throw std::runtime_error("Requested column not found in CSV: " + name);
        }
        keepIdx.push_back(it->second);
    }

    // Prepare a container for rows (we don't know the count up front)
    //std::vector<std::array<double, 1>> dummy;
    std::vector<std::vector<double>> rows;
    rows.reserve(1024);
    std::vector<std::string> times;
    times.reserve(1024);

    std::string line;
    std::vector<std::string> fields;
    size_t lineNo = 1;
    while (std::getline(ifs, line)) {
        ++lineNo;
        if (line.empty()) continue;
        splitCSVLine(line, fields);
        
        // Deal with NAs or missing values
        static bool warned_short_row = false;
        if (fields.size() < headers.size()) {
            if (!warned_short_row) {
                std::cerr << "[Warning]: Some rows have fewer fields than header.\n";
                warned_short_row = true;
            }
            fields.resize(headers.size());
        }

        // Filter rows by ID and skip row if its ID is not in idFilter
        std::string idValue = trim(fields[idxMap.at(idCol)]);
        if (!idFilter.empty() && idFilter.find(idValue) == idFilter.end()) {
            continue;
        }

        // Read and trim timestamp and parse selected numeric columns (empty or bad -> NaN).
        std::string ts = trim(fields[idxMap.at(timestampCol)]);
        std::vector<double> numeric;
        numeric.reserve(keepIdx.size());
        for (size_t j = 0; j < keepIdx.size(); ++j) {
            std::string cell = trim(fields[keepIdx[j]]);
            if (cell.empty()) {
                numeric.push_back(std::numeric_limits<double>::quiet_NaN());
            } else {
                try {
                    double v = std::stod(cell);
                    numeric.push_back(v);
                } catch (...) {
                    // if parse fails, push NaN and continue
                    numeric.push_back(std::numeric_limits<double>::quiet_NaN());
                }
            }
        }

        // Append to storage
        times.push_back(ts);
        rows.push_back(std::move(numeric));
    }

    // Fill Eigen matrix with the data
    const size_t nrows = rows.size();
    const size_t ncols = keepIdx.size();
    m_data.resize(nrows, ncols);
    for (size_t r = 0; r < nrows; ++r) {
        for (size_t c = 0; c < ncols; ++c) {
            m_data(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = rows[r][c];
        }
    }
    m_timestamps = std::move(times);

    // Initialize scaler vectors to match number of columns
    Eigen::Index C = m_data.cols();
    m_scaler.min = Eigen::VectorXd::Zero(C);
    m_scaler.max = Eigen::VectorXd::Zero(C);
    m_scaler.fitted = false;

    return nrows;
}

/**
 * Getter for the timestamps
 */
std::vector<std::string> Data::timestamps() const {
  return m_timestamps;
} 

/**
 *  Getter for the data numeric matrix
 */
Eigen::MatrixXd Data::numericData() const {
  return m_data;
} 

/**
 * Getter for the names of numeric columns
 */
std::vector<std::string> Data::numericColNames() const {
  return m_colNames;
} 

/**
 * Print header line, i.e. timestamp + numeric column names
 */
void Data::printHeader(const std::string& timestampColName) const {
    std::cout << "Header: " << timestampColName;
    const auto& cols = numericColNames();

    if (!cols.empty()) std::cout << " | ";

    for (size_t i = 0; i < cols.size(); ++i) {
        std::cout << cols[i];
        if (i + 1 < cols.size()) std::cout << " | ";
    }
    std::cout << "\n";
}

/**
 * Return a copy of the values in a selected column by name
 */
std::vector<double> Data::getColumnValues(const std::string& name) const {
    const auto& cols = numericColNames();
    auto it = std::find(cols.begin(), cols.end(), name);

    if (it == cols.end()) throw std::out_of_range("Column not found: " + name);

    size_t idx = std::distance(cols.begin(), it);
    const auto& mat = numericData();
    std::vector<double> out;
    out.reserve(mat.rows());
    
    for (int r = 0; r < mat.rows(); ++r) out.push_back(mat(r, static_cast<int>(idx)));

    return out;
}

/**
 * Set which transform to apply (applies to all numeric columns)
 */
void Data::setTransform(transform_type t, double alpha, bool excludeLastCol) {
    m_transform = t;
    m_alpha = alpha;
    m_excludeLastCol = excludeLastCol;

    // ensure scaler vectors have correct size (will be filled when applying MINMAX)
    Eigen::Index cols = m_data.cols();
    if (cols <= 0) {
        m_scaler.min.resize(0);
        m_scaler.max.resize(0);
        m_scaler.fitted = false;
    } else {
        m_scaler.min = Eigen::VectorXd::Zero(cols);
        m_scaler.max = Eigen::VectorXd::Zero(cols);
        m_scaler.fitted = false;
    }
}

/**
 * Apply the previously configured transform to m_data
 */
void Data::applyTransform() {
    if (m_transform == transform_type::NONE) return;
    const Eigen::Index R = m_data.rows();
    const Eigen::Index C = m_data.cols();
    if (R == 0 || C == 0) return;

    // helper to decide whether to operate on column c
    auto shouldTransformCol = [&](int c)->bool {
        if (!m_excludeLastCol) return true;
        return c != static_cast<int>(C) - 1;
    };

    switch (m_transform) {
        
        case transform_type::MINMAX:
        {
            // compute min/max and scale each column to [0,1]
            for (Eigen::Index c = 0; c < C; ++c) {
                if (!shouldTransformCol(static_cast<int>(c))) {
                    m_scaler.min(c) = 0.0;
                    m_scaler.max(c) = 1.0;
                    continue;
                }
                double mn = std::numeric_limits<double>::infinity();
                double mx = -std::numeric_limits<double>::infinity();
                std::size_t cnt = 0;
                for (Eigen::Index r = 0; r < R; ++r) {
                    double v = m_data(r, c);
                    if (std::isfinite(v)) { mn = std::min(mn, v); mx = std::max(mx, v); ++cnt; }
                }
                if (cnt == 0) { mn = 0.0; mx = 1.0; }
                m_scaler.min(c) = mn;
                m_scaler.max(c) = mx;
                double span = (mx - mn); if (span == 0.0) span = 1.0;
                for (Eigen::Index r = 0; r < R; ++r) {
                    double &x = m_data(r, c);
                    if (std::isfinite(x)) x = (x - mn) / span;
                }
            }
            m_scaler.fitted = true;
            break;
        }

        case transform_type::NONLINEAR:
        {
            double alpha = m_alpha;
            for (Eigen::Index c = 0; c < C; ++c) {
                if (!shouldTransformCol(static_cast<int>(c))) continue;
                for (Eigen::Index r = 0; r < R; ++r) {
                    double v = m_data(r, c);
                    if (std::isfinite(v)) {
                        double t = 1.0 - std::exp(-alpha * v);
                        m_data(r, c) = std::isfinite(t) ? t : std::numeric_limits<double>::quiet_NaN();
                    }
                }
            }
            break;
        }

        default:
            break;
    }
}

/**
 * Inverse the global transform (to bring predictions back)
 */
void Data::inverseTransform() {
    if (m_transform == transform_type::NONE) return;
    const Eigen::Index R = m_data.rows();
    const Eigen::Index C = m_data.cols();
    if (R == 0 || C == 0) return;

    // small tolerances for numeric clipping
    //const double eps_clip_low = 1e-12;    // allow tiny negative numbers
    //const double eps_clip_high = 1e-12;   // allow tiny >1 numbers to be clamped below 1.0

    auto shouldTransformCol = [&](int c)->bool {
        if (!m_excludeLastCol) return true;
        return c != static_cast<int>(C) - 1;
    };

    switch (m_transform) {
        case transform_type::MINMAX:
        {
            if (!m_scaler.fitted) throw std::runtime_error("Scaler not fitted for inverse");
            for (Eigen::Index c = 0; c < C; ++c) {
                if (!shouldTransformCol(static_cast<int>(c))) continue;
                double mn = m_scaler.min(c);
                double mx = m_scaler.max(c);
                double span = (mx - mn); if (span == 0.0) span = 1.0;
                for (Eigen::Index r = 0; r < R; ++r) {
                    double &x = m_data(r, c);
                    if (std::isfinite(x)) x = x * span + mn;
                }
            }
            break;
        }

        case transform_type::NONLINEAR:
        {
            const double alpha = m_alpha; 
            for (Eigen::Index c = 0; c < C; ++c) {
                if (!shouldTransformCol(static_cast<int>(c))) continue;
                for (Eigen::Index r = 0; r < R; ++r) {
                    double &t = m_data(r, c);
                    if (!std::isfinite(t)) continue;  // preserve NaN/Inf handling
                    // If t >= 1.0 due to rounding/saturation, nudge it below 1.0
                    if (t >= 1.0) t = std::nextafter(1.0, 0.0);
                    // Now safe: compute inverse (allows negative t)
                    double one_minus = 1.0 - t;  // > 0 here
                    if (!(one_minus > 0.0)) throw std::runtime_error("inverseGlobalTransform: 1 - D_trans <= 0");
                    t = -std::log(one_minus) / alpha;
                }
            }
            break;
        }

        default:
            break;
    }
}

/**
 * Create matrix for backpropagation from data matrix
 */
void Data::makeCalibMat(std::vector<int> inpNumsOfVars, int outRows){
    if (outRows <= 0 || std::any_of(inpNumsOfVars.begin(), inpNumsOfVars.end(), [](int x){ return x < 0; }))
        throw std::invalid_argument("inpNumsOfVars values and outRows must be positive");

    // ! Assuming predicted variable is in last column of data
    const auto maxR = *std::max_element(inpNumsOfVars.begin(),inpNumsOfVars.end());    // max number of input variable values
    if (maxR == 0)
        throw std::runtime_error("At least one value in inpNumsOfVars must be greater than 0");

    const int DC = static_cast<int>(m_data.cols());   // number of cols in input data matrix
    if (DC < 1)
        throw std::runtime_error("Data has no columns");
    if (inpNumsOfVars.size() != DC)
        throw std::invalid_argument("inpNumsOfVars size doesnt match data columns");

    const int CR = m_data.rows() - maxR - outRows + 1;   // number of rows of calibration matrix
    if (CR <= 0)
        throw std::runtime_error("Not enough rows to build calibration matrix with given inpNumsOfVars/outRows");
    
    const int CC = std::accumulate(inpNumsOfVars.begin(), inpNumsOfVars.end(), 0) + outRows;  // number of cols of calibration matrix
    calibMat = Eigen::MatrixXd(CR , CC);
    for(int i = 0; i < CR; i++){
        int col_idx = 0;
        for (int j = 0; j < DC; j++) {
            for(int l = 0; l < inpNumsOfVars[j]; l++){
                calibMat(i, col_idx++) = m_data(i + maxR - inpNumsOfVars[j] + l, j);
            }
        }
        for (int j = 0; j < outRows; j++) {
            calibMat(i, col_idx++) = m_data(i + maxR + j, DC - 1);
        }
    }
}

/**
 * Create matrix for backpropagation from data matrix - created by MJ due to errors in Moisture data...
 * na realna data mi to neslo aplikovat, zatim to neresim, jdu delat neco jineho...
 */
void Data::makeCalibMat2(int inpRows, int outRows){
    if (inpRows <= 0 || outRows <= 0)
        throw std::invalid_argument("inpRows and outRows must be positive");

    const int totalCols = static_cast<int>(m_data.cols());
    if (totalCols < 1)
        throw std::runtime_error("Data has no columns");

    const int inputCols = totalCols - 1; // last column is target
    if (inputCols <= 0)
        throw std::runtime_error("Need at least one input column (data must contain at least one feature + target)");

    const int totalRows = static_cast<int>(m_data.rows());
    const int CR = totalRows - inpRows - outRows + 1;   // number of calibration rows
    if (CR <= 0)
        throw std::runtime_error("Not enough rows to build calibration matrix with given inpRows/outRows");

    const int CC = inpRows * inputCols + outRows;  // columns: stacked input-window + outRows
    calibMat = Eigen::MatrixXd(CR, CC);

    for (int i = 0; i < CR; ++i) {
        int col_idx = 0;
        // inputs: for each feature column (except target), stack inpRows values
        for (int j = 0; j < inputCols; ++j) {
            for (int k = 0; k < inpRows; ++k) {
                calibMat(i, col_idx++) = m_data(i + k, j);
            }
        }
        // outputs: take outRows consecutive values from the target column (last column)
        for (int j = 0; j < outRows; ++j) {
            calibMat(i, col_idx++) = m_data(i + inpRows + j, totalCols - 1);
        }
    }
}

/**
 * Create separate calibration inps and outs matrices for backpropagation from data matrix
 */
void Data::makeCalibMatsSplit(std::vector<int> inpNumsOfVars, int outRows){
    if (outRows <= 0 || std::any_of(inpNumsOfVars.begin(), inpNumsOfVars.end(), [](int x){ return x < 0; }))
        throw std::invalid_argument("inpNumsOfVars values and outRows must be positive");

    // ! Assuming predicted variable is in last column of data
    const auto maxR = *std::max_element(inpNumsOfVars.begin(),inpNumsOfVars.end());    // max number of input variable values
    if (maxR == 0)
        throw std::runtime_error("At least one value in inpNumsOfVars must be greater than 0");

    const int DC = static_cast<int>(m_data.cols());   // number of cols in input data matrix
    if (DC < 1)
        throw std::runtime_error("Data has no columns");
    if (inpNumsOfVars.size() != DC)
        throw std::invalid_argument("inpNumsOfVars size doesnt match data columns");

    const int CR = m_data.rows() - maxR - outRows + 1;   // number of rows of matrices
    if (CR <= 0)
        throw std::runtime_error("Not enough rows to build calibration matrices with given inpNumsOfVars/outRows");
    
    const int inpC = std::accumulate(inpNumsOfVars.begin(), inpNumsOfVars.end(), 0);  // number of cols of inps calibration matrix
    calibInpsMat = Eigen::MatrixXd(CR , inpC);
    calibOutsMat = Eigen::MatrixXd(CR , outRows);

    for(int i = 0; i < CR; i++){
        int col_idx = 0;
        for (int j = 0; j < DC; j++) {
            for(int l = 0; l < inpNumsOfVars[j]; l++){
                calibInpsMat(i, col_idx++) = m_data(i + maxR - inpNumsOfVars[j] + l, j);
            }
        }
        for (int j = 0; j < outRows; j++) {
            calibOutsMat(i, j) = m_data(i + maxR + j, DC - 1);
        }
    }
}

/**
 * Split created calibration matrix into separate inps and outs matrices
 */
void Data::splitCalibMat(size_t inpLength){
    if (inpLength <= 0 || inpLength >= calibMat.cols())
        throw std::invalid_argument("inpLength must be greater and 0 and less than calibMat columns");

    calibInpsMat = calibMat.leftCols(inpLength);
    calibOutsMat = calibMat.rightCols(calibMat.cols() - inpLength);
}

/**
 * Getter for calibration matrix
 */
Eigen::MatrixXd Data::getCalibMat(){
    return calibMat;
}

/**
 * Setter for calibration matrix
 */
void Data::setCalibMat(const Eigen::MatrixXd &newMat){
    calibMat = newMat;
}

/**
 * Getter for calibration inputs matrix
 */
Eigen::MatrixXd Data::getCalibInpsMat(){
    return calibInpsMat;
}

/**
 * Setter for calibration inputs matrix
 */
void Data::setCalibInpsMat(const Eigen::MatrixXd &newMat){
    calibInpsMat = newMat;
}

/**
 * Getter for calibration outputs matrix
 */
Eigen::MatrixXd Data::getCalibOutsMat(){
    return calibOutsMat;
}

/**
 * Setter for calibration outputs matrix
 */
void Data::setCalibOutsMat(const Eigen::MatrixXd &newMat){
    calibOutsMat = newMat;
}

/**
 * Create random permutation vector for shuffling
 */
std::vector<int> Data::permutationVector(int length){
    if (length <= 0)
        throw std::invalid_argument("length must be greater than 0");

    std::vector<int> permVec(length);
    std::iota(permVec.begin(), permVec.end(), 0);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(permVec.begin(), permVec.end(), gen);
    
    return permVec;
}

/**
 * Shuffle matrix rows
 */
Eigen::MatrixXd Data::shuffleMatrix(const Eigen::MatrixXd &matrix, const std::vector<int>& permVec){
    if (matrix.rows() != permVec.size())
        throw std::invalid_argument("matrix rows and permVec length dont match");

    Eigen::MatrixXd newmat(matrix.rows(), matrix.cols());
    for (size_t i = 0; i < matrix.rows(); ++i) {
        newmat.row(i) = matrix.row(permVec[i]);
    }
    return newmat;
}

/**
 * Unshuffle matrix rows
 */
Eigen::MatrixXd Data::unshuffleMatrix(const Eigen::MatrixXd &matrix, const std::vector<int>& permVec) {
    if (matrix.rows() != permVec.size())
        throw std::invalid_argument("matrix rows and permVec length dont match");

    Eigen::MatrixXd oldmat(matrix.rows(), matrix.cols());
    for (size_t i = 0; i < matrix.rows(); ++i) {
        oldmat.row(permVec[i]) = matrix.row(i);
    }
    return oldmat;
}


/**
 * Find indices of rows that contain any NaN in numeric data
 */
std::vector<size_t> Data::findRowsWithNa() const {
    std::vector<size_t> out;
    if (m_data.size() == 0) return out;
    const Eigen::Index R = m_data.rows();
    const Eigen::Index C = m_data.cols();
    for (Eigen::Index r = 0; r < R; ++r) {
        bool rowHasNa = false;
        for (Eigen::Index c = 0; c < C; ++c) {
            double v = m_data(r, c);
            if (!std::isfinite(v)) { rowHasNa = true; break; }
        }
        if (rowHasNa) out.push_back(static_cast<size_t>(r));
    }
    return out;
}

/**
 * Remove rows that contain any NaN from m_data and m_timestamps, but keep backups and record removed indices so they can be restored later
 */
void Data::removeRowsWithNa() {
    if (m_has_filtered_rows) {
        // already filtered — do nothing
        return;
    }

    // find rows to remove
    auto naIdx = findRowsWithNa();
    if (naIdx.empty()) {
        // no NAs — nothing to do
        m_has_filtered_rows = false;
        m_na_row_indices.clear();
        return;
    }

    // Backup originals
    m_data_backup = m_data;
    m_timestamps_backup = m_timestamps;

    const Eigen::Index R = m_data.rows();
    const Eigen::Index C = m_data.cols();

    // Build a boolean mask of rows to keep
    std::vector<char> keep(R, 1);
    for (size_t i : naIdx) {
        if (i < static_cast<size_t>(R)) keep[static_cast<size_t>(i)] = 0;
    }

    // Count kept rows
    size_t kept_count = 0;
    for (Eigen::Index r = 0; r < R; ++r) if (keep[static_cast<size_t>(r)]) ++kept_count;

    // Create new matrix of kept rows
    Eigen::MatrixXd newmat(static_cast<Eigen::Index>(kept_count), C);
    std::vector<std::string> newtimes;
    newtimes.reserve(kept_count);

    Eigen::Index rr = 0;
    for (Eigen::Index r = 0; r < R; ++r) {
        if (keep[static_cast<size_t>(r)]) {
            newmat.row(rr) = m_data.row(r);
            newtimes.push_back(m_timestamps[static_cast<size_t>(r)]);
            ++rr;
        }
    }

    // Replace m_data and m_timestamps with filtered versions
    m_data = std::move(newmat);
    m_timestamps = std::move(newtimes);

    // Store removed indices (sorted ascending)
    m_na_row_indices = std::move(naIdx);
    std::sort(m_na_row_indices.begin(), m_na_row_indices.end());

    m_has_filtered_rows = true;
}

/**
 * Restore the original (unfiltered) data/timestamps and clear backups
 */
void Data::restoreOriginalData() {
    if (!m_has_filtered_rows) return;
    m_data = std::move(m_data_backup);
    m_timestamps = std::move(m_timestamps_backup);
    m_data_backup.resize(0,0);
    m_timestamps_backup.clear();
    m_na_row_indices.clear();
    m_has_filtered_rows = false;
}

/**
 * Expand predictions on the filtered dataset back to full-length matrix
 */
Eigen::MatrixXd Data::expandPredictionsToFull(const Eigen::MatrixXd& preds) const {
    if (!m_has_filtered_rows) {
        return preds;
    }

    const Eigen::Index origRows = static_cast<Eigen::Index>(m_data_backup.rows());
    const Eigen::Index validRows = static_cast<Eigen::Index>(m_data.rows());
    if (preds.rows() != validRows) {
        throw std::invalid_argument("expandPredictionsToFull (matrix): preds rows != valid rows");
    }
    const Eigen::Index cols = preds.cols();

    Eigen::MatrixXd full(origRows, cols);
    full.setConstant(std::numeric_limits<double>::quiet_NaN());

    std::size_t idxRemPos = 0;
    Eigen::Index pj = 0;
    for (Eigen::Index ri = 0; ri < origRows; ++ri) {
        bool removed = false;
        if (idxRemPos < m_na_row_indices.size() && static_cast<size_t>(ri) == m_na_row_indices[idxRemPos]) {
            removed = true;
            ++idxRemPos;
        }
        if (!removed) {
            full.row(ri) = preds.row(pj++);
        } // else leave NaNs
    }
    return full;
}

/**
 * Expand predictions produced from calibration matrix
 */
Eigen::MatrixXd Data::expandPredictionsFromCalib(const Eigen::MatrixXd& preds, int inpRows) const {
    if (inpRows < 0) throw std::invalid_argument("expandPredictionsFromCalib: inpRows must be >= 0");
    const Eigen::Index CR = preds.rows();
    const Eigen::Index out_horizon = preds.cols();

    // original full row count (before any filtering)
    const Eigen::Index origRows = static_cast<Eigen::Index>(m_data_backup.rows());
    const Eigen::Index validRows = static_cast<Eigen::Index>(m_data.rows()); // filtered rows

    if (CR == 0 || out_horizon == 0) {
        return Eigen::MatrixXd(0,0);
    }

    // Build mapping filtered_index -> original_index
    std::vector<Eigen::Index> filt2orig;
    filt2orig.reserve(static_cast<size_t>(validRows));
    if (m_na_row_indices.empty()) {
        // identity mapping when no rows were removed
        for (Eigen::Index i = 0; i < validRows; ++i) filt2orig.push_back(i);
    } else {
        // iterate original indices and skip removed ones
        std::size_t remPos = 0;
        for (Eigen::Index orig = 0; orig < origRows; ++orig) {
            if (remPos < m_na_row_indices.size() && static_cast<Eigen::Index>(m_na_row_indices[remPos]) == orig) {
                ++remPos; // this original row was removed
            } else {
                filt2orig.push_back(orig);
            }
        }
        if (static_cast<Eigen::Index>(filt2orig.size()) != validRows) {
            throw std::runtime_error("expandPredictionsFromCalib: internal mapping size mismatch");
        }
    }

    // Prepare full matrix with NaN
    Eigen::MatrixXd full(origRows, out_horizon);
    full.setConstant(std::numeric_limits<double>::quiet_NaN());

    // For each calibration pattern i, place preds(i, j) at filtered-row index (i + inpRows + j)
    for (Eigen::Index i = 0; i < CR; ++i) {
        for (Eigen::Index j = 0; j < out_horizon; ++j) {
            Eigen::Index filteredRow = i + inpRows + j;
            if (filteredRow < 0 || filteredRow >= validRows) {
                // If the pattern maps outside valid rows, skip (safe guard)
                continue;
            }
            Eigen::Index origRow = filt2orig[static_cast<size_t>(filteredRow)];
            full(origRow, j) = preds(i, j);
        }
    }

    return full;
}
