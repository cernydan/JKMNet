#ifndef DATA_HPP
#define DATA_HPP

#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <limits> 

#include "eigen-3.4/Eigen/Dense"

enum class transform_type { 
    NONE = 0,
    MINMAX, 
    NONLINEAR
};

struct Scaler {
    Eigen::VectorXd min;  
    Eigen::VectorXd max; 
    bool fitted = false;
};

class Data {
    public:
        Data() = default;  //!< The constructor

        size_t loadFilteredCSV(const std::string& path,
            const std::unordered_set<std::string>& idFilter,
            const std::vector<std::string>& keepColumns,  // names of numeric columns to extract (e.g. "T1","T2","T3","moisture")
            const std::string& timestampCol = "hour_start",  // name of timestamp column (e.g "hour_start", "date")
            const std::string& idCol = "ID");  // ID of the selected sensor
            //!< Returns number of loaded rows

        std::vector<std::string> timestamps() const;  //!< Getter for the timestamps
        Eigen::MatrixXd numericData() const;  //!< Getter for the data numeric matrix
        std::vector<std::string> numericColNames() const;  //!< Getter for the names of numeric columns

        void printHeader(const std::string& timestampColName = "timestamp") const;  //!< Print header line, i.e. timestamp + numeric column names
        std::vector<double> getColumnValues(const std::string& name) const; //!< Return a copy of the values in a selected column by name

        void setTransform(transform_type t, double alpha = 0.015, bool excludeLastCol = false);  //!< Set which transform to apply (applies to all numeric columns)
        void applyTransform();  //!< Apply the previously configured transform to m_data
        void inverseTransform();  //!< Inverse the global transform (to bring predictions back)

        void makeCalibMat(int inpRows, int outRows); //!< Create calibration matrix for backpropagation from data matrix
        void makeCalibMat2(int inpRows, int outRows); //!< Create calibration matrix for backpropagation from data matrix
        Eigen::MatrixXd getCalibMat();  //!< Getter for calibration matrix
        std::vector<int> shuffleCalibMat();  //!< Randomly shuffle calibration matrix rows

    protected:

    private:
        static void splitCSVLine(const std::string& line, std::vector<std::string>& outFields);  //!< Helper function to parse a CSV line into fields (handles quotes)

        std::vector<std::string> m_timestamps;  //!< Time string
        Eigen::MatrixXd m_data;  //!< Matrix with numeric columns (rows x cols) filled with variables
        std::vector<std::string> m_colNames;  //!< Column names for m_data
        Eigen::MatrixXd calibMat; //!< Matrix of inputs and desired outputs for backpropagation

        // global transform config
        transform_type m_transform = transform_type::NONE;
        double m_alpha = 0.015;
        bool m_excludeLastCol = false;
        Scaler m_scaler;   // stores per-column min/max after a MINMAX fit
};

#endif // DATA_HPP