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

        // Deal with NAs in the dataset
        std::vector<size_t> findRowsWithNa() const;  //!< Find indices of rows that contain any NaN in numeric data
        void removeRowsWithNa();  //!< Remove rows that contain any NaN from m_data and m_timestamps, but keep backups and record removed indices so they can be restored later
        void restoreOriginalData();  //< Restore the original (unfiltered) data/timestamps and clear backups
        Eigen::MatrixXd expandPredictionsToFull(const Eigen::MatrixXd& preds) const;  //!< Expand predictions on the filtered dataset back to full-length matrix
        Eigen::MatrixXd expandPredictionsFromCalib(const Eigen::MatrixXd& preds, int inpRows) const;  //!< Expand predictions produced from calibration matrix
        const std::vector<size_t>& removedRowIndices() const { return m_na_row_indices; }   //!< Get indices of rows removed 
        size_t validRowCount() const { return static_cast<size_t>(m_data.rows()); }  //!< Number of valid rows currently in m_data


    protected:

    private:
        static void splitCSVLine(const std::string& line, std::vector<std::string>& outFields);  //!< Helper function to parse a CSV line into fields (handles quotes)

        std::vector<std::string> m_timestamps;  //!< Time string
        Eigen::MatrixXd m_data;  //!< Matrix with numeric columns (rows x cols) filled with variables
        std::vector<std::string> m_colNames;  //!< Column names for m_data
        Eigen::MatrixXd calibMat; //!< Matrix of inputs and desired outputs for backpropagation

        // Global transform config
        transform_type m_transform = transform_type::NONE;
        double m_alpha = 0.015;
        bool m_excludeLastCol = false;
        Scaler m_scaler;   //!< Stores per-column min/max after a MINMAX fit

        // Deal with NAs in the dataset
        Eigen::MatrixXd m_data_backup;  //!< Full original data backup (before filtering)
        std::vector<std::string> m_timestamps_backup;  //!< Timestamps backup
        std::vector<size_t> m_na_row_indices;  //!< Indices of removed rows (in original coordinates)
        bool m_has_filtered_rows = false;  //!< True if removeRowsWithNa() was applied

};

#endif // DATA_HPP