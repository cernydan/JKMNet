#include "Data.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>

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
        bool parseOk = true;
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
                    parseOk = false;
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
 * Create matrix for backpropagation from data matrix
 */
void Data::makeCalibMat(int inpRows, int outRows){

    // ! Assuming predicted variable is in last column of data
    int DC = m_data.cols();                           // number of cols in input data matrix
    int CR = m_data.rows() - inpRows - outRows + 1;   // number of rows of calibration matrix
    int CC = inpRows * DC + outRows;                  // number of cols of calibration matrix

    calibMat = Eigen::MatrixXd(CR , CC);
    for(int i = 0; i < CR; i++){
        int col_idx = 0;
        for (int j = 0; j < DC; j++) {
            for (int k = 0; k < inpRows; k++) {
                calibMat(i, col_idx++) = m_data(i + k, j);
            }
        }
        for (int j = 0; j < outRows; j++) {
            calibMat(i, col_idx++) = m_data(i + inpRows + j, DC - 1);
        }
    }
}

/**
 * Getter for calibration matrix
 */
Eigen::MatrixXd Data::getCalibMat(){
    return calibMat;
}

/**
 * Randomly shuffle matrix rows
 */
std::vector<int> Data::shuffleCalibMat(){
    int r = calibMat.rows();
    std::vector<int> permVec(r);
    std::iota(permVec.begin(), permVec.end(), 0);

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(permVec.begin(), permVec.end(), gen);

    Eigen::MatrixXd newmat(r, calibMat.cols());
    for (int i = 0; i < r; ++i) {
        newmat.row(i) = calibMat.row(permVec[i]);
    }
    calibMat = std::move(newmat);

    return permVec;
}
