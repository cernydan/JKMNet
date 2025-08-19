#ifndef DATA_HPP
#define DATA_HPP

#pragma once
#include <string>
#include <vector>
#include <unordered_set>

#include "eigen-3.4/Eigen/Dense"

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

    protected:

    private:
        static void splitCSVLine(const std::string& line, std::vector<std::string>& outFields);  //!< Helper function to parse a CSV line into fields (handles quotes)

        std::vector<std::string> m_timestamps;  //!< Time string
        Eigen::MatrixXd m_data;  //!< Matrix with numeric columns (rows x cols) filled with variables
        std::vector<std::string> m_colNames;  //!< Column names for m_data
};

#endif // DATA_HPP