/**
 * @file Data.hpp
 * @brief Declares the Data class (dataset loading, transforms, calibration matrix construction) and the transform_type/Scaler/VarScaler helper types.
 */
#ifndef DATA_HPP
#define DATA_HPP

#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <limits>
#include <ostream>
#include <fstream>
#include <cstdint>

#include "eigen-3.4/Eigen/Dense"

/// Types of per-column data transforms.
enum class transform_type {
    NONE = 0,      //!< No transform
    MINMAX,        //!< Min-max scaling to [0, 1] using per-column min/max
    NONLINEAR,     //!< Nonlinear transform parameterized by alpha
    MINMAXNONLIN,  //!< Nonlinear transform composed with min-max scaling
    ZSCORE         //!< Standardization to zero mean and unit variance
};

/// Per-column min/max statistics fitted by a MINMAX transform.
struct Scaler {
    Eigen::VectorXd min;   //!< Per-column minima
    Eigen::VectorXd max;   //!< Per-column maxima
    bool fitted = false;   //!< True if the statistics were actually computed
};

/// Per-variable transform configuration and fitted statistics.
struct VarScaler {
    transform_type type = transform_type::NONE;  //!< Transform applied to the variable
    double p1 = 0.0;     //!< MINMAX: min   | ZSCORE: mean
    double p2 = 0.0;     //!< MINMAX: max   | ZSCORE: stddev
    double al = 0.0;     //!< NONLINEAR: alpha
    bool fitted = false; //!< True if the statistics were actually computed
};

/// Run configuration (defined elsewhere); used here only for logging settings.
struct RunConfig;

/// Dataset container: CSV loading, transforms, calibration/validation matrix construction.
/**
 * Holds the numeric data matrix together with timestamps and column names, and
 * provides helpers for scaling (transform/inverse), NaN filtering with restore,
 * building calibration matrices for (feed-forward, CNN and LSTM) training,
 * k-fold splitting, shuffling and CSV export.
 */
class Data {
    public:
        Data() = default;  //!< The constructor
        ~Data() = default;  //!< The destructor
        Data(const Data&) = default;  //!< The copy constructor
        Data& operator=(const Data&) = default;   //!< The assignment operator

        /// Load numeric columns of rows matching the ID filter from a CSV file.
        /**
         * @param path Path of the CSV file.
         * @param idFilter Set of accepted IDs; rows whose idCol value is not in the set are skipped.
         * @param keepColumns Names of numeric columns to extract (e.g. "T1","T2","T3","moisture").
         * @param timestampCol Name of the timestamp column (e.g. "hour_start", "date").
         * @param idCol Name of the column containing the sensor/series ID.
         * @return Number of loaded rows.
         */
        size_t loadFilteredCSV(const std::string& path,
            const std::unordered_set<std::string>& idFilter,
            const std::vector<std::string>& keepColumns,  // names of numeric columns to extract (e.g. "T1","T2","T3","moisture")
            const std::string& timestampCol = "hour_start",  // name of timestamp column (e.g "hour_start", "date")
            const std::string& idCol = "ID");  // ID of the selected sensor

        static void cleanDirectory(const std::string &path);  //!< Clean all files in a directory
        static void cleanAllOutputs(const std::string &outDir);  //!< Clean all files in an outputs directory

        /// Write model settings into a log file.
        /**
         * @param os Output stream to write into.
         * @param cfg Run configuration whose settings are logged.
         * @param run_id Identifier of the current run.
         */
        void logRunSettings(std::ostream& os, const RunConfig& cfg, unsigned run_id) const;

        /// Save a matrix into a CSV file.
        /**
         * @param path Path of the CSV file.
         * @param M Matrix to write (one row per line).
         * @param colNames Column names for the header; no header is written if empty.
         * @param inverseOutputs If true, the global output transform is inverted before writing.
         * @return True if the file was written successfully.
         */
        bool saveMatrixCsv(const std::string &path,
            const Eigen::MatrixXd &M,
            const std::vector<std::string> &colNames = {},
            bool inverseOutputs = false) const;

        std::vector<std::string> timestamps() const;  //!< Getter for the timestamps
        Eigen::MatrixXd numericData() const;  //!< Getter for the data numeric matrix
        void setNumericData(const Eigen::MatrixXd &newData); //!< Setter for the data numeric matrix
        std::vector<std::string> numericColNames() const;  //!< Getter for the names of numeric columns

        void printHeader(const std::string& timestampColName = "timestamp") const;  //!< Print header line, i.e. timestamp + numeric column names

        /// Return a copy of the values of a selected column by name.
        /**
         * @param name Name of the numeric column.
         * @return Copy of the column values.
         */
        std::vector<double> getColumnValues(const std::string& name) const;

        /// Set which transforms to apply and their nonlinear parameters.
        /**
         * @param transforms Transform type per numeric column.
         * @param alpha Alpha parameter per column (used by NONLINEAR/MINMAXNONLIN).
         * @param excludeLastCol If true, the last column (typically the output) is left untransformed.
         */
        void setTransform(const std::vector<transform_type>& transforms, std::vector<double> alpha, bool excludeLastCol = false);

        void applyTransform();  //!< Apply the previously configured transform to m_data
        void inverseTransform();  //!< Invert the global transform (to bring predictions back)

        /// Invert the global transform for output columns of a matrix.
        /**
         * @param M Matrix whose columns hold transformed outputs.
         * @return Matrix with the transform inverted.
         */
        Eigen::MatrixXd inverseTransformOutputs(const Eigen::MatrixXd& M) const;

        /// Transform input/output matrices and return the fitted per-variable scalers.
        /**
         * @param in Input matrix.
         * @param out Output matrix.
         * @param transforms Transform type per variable.
         * @param alpha Alpha parameter per variable (used by NONLINEAR/MINMAXNONLIN).
         * @param excludeLastCol If true, the last column of in is left untransformed.
         * @param input_numbers Mapping of input columns to variable scaler indices.
         * @return Tuple of (transformed inputs, transformed outputs, fitted scalers).
         */
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<VarScaler>>
        transformMatsGetScalers(Eigen::MatrixXd in,
                     Eigen::MatrixXd out,
                     const std::vector<transform_type>& transforms,
                     std::vector<double> alpha,
                     bool excludeLastCol,
                     std::vector<std::vector<int>> input_numbers);

        /// Transform input/output matrices using previously fitted scalers.
        /**
         * @param in Input matrix.
         * @param out Output matrix.
         * @param transforms Transform type per variable.
         * @param excludeLastCol If true, the last column of in is left untransformed.
         * @param input_numbers Mapping of input columns to variable scaler indices.
         * @param scalers Scalers previously fitted by transformMatsGetScalers().
         * @return Pair of (transformed inputs, transformed outputs).
         */
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        transformMatsApplyScalers(Eigen::MatrixXd in,
                Eigen::MatrixXd out,
                const std::vector<transform_type>& transforms,
                bool excludeLastCol,
                std::vector<std::vector<int>> input_numbers,
                const std::vector<VarScaler>& scalers);

        /// Invert the transform of output columns using an explicit output scaler.
        /**
         * @param M Matrix whose columns hold transformed outputs.
         * @param outScaler Scaler fitted on the outputs.
         * @return Matrix with the transform inverted.
         */
        Eigen::MatrixXd inverseTransformOutputs(const Eigen::MatrixXd& M, const VarScaler& outScaler);

        std::vector<size_t> calibPatternOriginalIndices() const { return m_calib_pattern_orig_indices; }  //!< Getter for calibration-pattern indices in the original (unfiltered) data
        std::vector<int> calibPatternFilteredIndices() const { return m_calib_pattern_filtered_indices; }  //!< Getter for calibration-pattern indices in the filtered data

        /// Create the calibration matrix (both inputs and outputs) for backpropagation from the data matrix (with NA removal).
        /**
         * @param inpNumsOfVars Number of lagged rows used per input variable.
         * @param outRows Number of output (horizon) rows.
         */
        void makeCalibMat(std::vector<int> inpNumsOfVars, int outRows);

        /// Create separate calibration input and output matrices for backpropagation from the data matrix (with NA removal).
        /**
         * @param inpNumsOfVars Number of lagged rows used per input variable.
         * @param outRows Number of output (horizon) rows.
         */
        void makeCalibMatsSplit(std::vector<int> inpNumsOfVars, int outRows);

        /// Split the created calibration matrix into separate input and output matrices.
        /**
         * @param inpLength Number of leading columns of the calibration matrix that are inputs.
         */
        void splitCalibMat(int inpLength);

        /// Split a matrix horizontally into input and output parts.
        /**
         * @param mat Matrix to split (columns = [inputs | outputs]).
         * @param inpSize Number of input columns.
         * @param outSize Number of output columns.
         * @return Pair of (inputs matrix, outputs matrix).
         */
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        splitInputsOutputs(const Eigen::MatrixXd &mat, int inpSize, int outSize) const;

        /// Split the calibration matrix into train/validation parts and also return the row indices.
        /**
         * @param trainFraction Fraction of rows used for training (default 0.8).
         * @param shuffle Whether to shuffle rows before splitting (default true).
         * @param seed Random seed used for shuffling.
         * @return Tuple of (train matrix, validation matrix, train row indices, validation row indices).
         */
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
        splitCalibMatWithIdx(double trainFraction = 0.8, bool shuffle = true, unsigned seed = 0) const;

        /// Split the raw data rows (m_data) into train/validation and return the row indices.
        /**
         * @param trainFraction Fraction of rows used for training (default 0.8).
         * @param shuffle Whether to shuffle rows before splitting (default true).
         * @param seed Random seed used for shuffling.
         * @return Tuple of (train matrix, validation matrix, train row indices, validation row indices).
         */
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
        splitDataRowsWithIdx(double trainFraction = 0.8, bool shuffle = true, unsigned seed = 0) const;

        Eigen::MatrixXd getCalibMat();  //!< Getter for calibration matrix
        void setCalibMat(const Eigen::MatrixXd &newMat);  //!< Setter for calibration matrix
        Eigen::MatrixXd getCalibInpsMat();  //!< Getter for calibration inputs matrix
        void setCalibInpsMat(const Eigen::MatrixXd &newMat);  //!< Setter for calibration inputs matrix
        Eigen::MatrixXd getCalibOutsMat();  //!< Getter for calibration outputs matrix
        void setCalibOutsMat(const Eigen::MatrixXd &newMat);  //!< Setter for calibration outputs matrix

        /// Create a random permutation vector for shuffling.
        /**
         * @param length Length of the permutation.
         * @return Random permutation of the indices 0 .. length-1.
         */
        std::vector<int> permutationVector(int length);

        /// Shuffle matrix rows according to a permutation vector.
        /**
         * @param matrix Matrix whose rows are shuffled.
         * @param permVec Permutation produced by permutationVector().
         * @return Row-permuted matrix.
         */
        Eigen::MatrixXd shuffleMatrix(const Eigen::MatrixXd &matrix, const std::vector<int>& permVec);

        /// Restore the original row order of a matrix shuffled with permVec.
        /**
         * @param matrix Row-shuffled matrix.
         * @param permVec Permutation that was used for shuffling.
         * @return Matrix with the original row order restored.
         */
        Eigen::MatrixXd unshuffleMatrix(const Eigen::MatrixXd &matrix, const std::vector<int>& permVec);

        // Deal with NAs in the dataset
        std::vector<size_t> findRowsWithNa() const;  //!< Find indices of rows that contain any NaN in numeric data
        void removeRowsWithNa();  //!< Remove rows that contain any NaN from m_data and m_timestamps, but keep backups and record removed indices so they can be restored later
        void restoreOriginalData();  //!< Restore the original (unfiltered) data/timestamps and clear backups

        /// Expand predictions computed on the filtered dataset back to a full-length matrix.
        /**
         * Rows removed by removeRowsWithNa() are reinserted (typically as NaN) so the
         * result again matches the original number of rows.
         * @param preds Predictions for the filtered rows.
         * @return Full-length prediction matrix.
         */
        Eigen::MatrixXd expandPredictionsToFull(const Eigen::MatrixXd& preds) const;

        /// Expand predictions produced from the calibration matrix back to a full-length matrix.
        /**
         * @param preds Predictions for calibration patterns.
         * @param inpRows Number of input (lag) rows per pattern; used for index alignment.
         * @return Full-length prediction matrix.
         */
        Eigen::MatrixXd expandPredictionsFromCalib(const Eigen::MatrixXd& preds, int inpRows) const;

        const std::vector<size_t>& removedRowIndices() const { return m_na_row_indices; }   //!< Get indices of rows removed
        size_t validRowCount() const { return static_cast<size_t>(m_data.rows()); }  //!< Number of valid rows currently in m_data

        /// Make calibration and validation input/output matrices for the current fold in k-fold cross-validation.
        /**
         * @param inpNumsOfVars Number of lagged rows used per input variable.
         * @param outRows Number of output (horizon) rows.
         * @param kFolds Total number of folds.
         * @param foldIdx Index of the current fold (0-based).
         * @param shuffle Whether to shuffle rows before splitting into folds.
         * @param largerPieceCalib If true, the larger part is used for calibration.
         * @param seed Random seed used for shuffling.
         * @return Tuple of (calibration inputs, calibration outputs, validation inputs, validation outputs).
         */
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
        makeKFoldMats(
            std::vector<int> inpNumsOfVars,
            int outRows,
            int kFolds,
            int foldIdx,
            bool shuffle,
            bool largerPieceCalib,
            unsigned seed);

        /// Build train/validation input and output matrices with per-variable lag offsets.
        /**
         * @param inpOffsets Lag offsets used for each input variable.
         * @param outRows Number of output (horizon) rows.
         * @param trainFraction Fraction of patterns used for training.
         * @param shuffleCalib Whether to shuffle calibration patterns before splitting.
         * @param seed Random seed used for shuffling.
         * @return Tuple of (train inputs, train outputs, validation inputs, validation outputs,
         * train pattern indices, validation pattern indices).
         */
        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
        makeMats(const std::vector<std::vector<int>>& inpOffsets,
                          int outRows,
                          double trainFraction,
                          bool shuffleCalib,
                          unsigned seed) const;

        /// Save a vector of integers to a text file, one value per line.
        /**
         * @param v Vector to save.
         * @param path Path of the output file.
         * @return True if the file was written successfully.
         */
        bool saveVector(const std::vector<int>& v, const std::string& path);

        /// Build LSTM train/validation datasets from past time steps only.
        /**
         * @param timeSteps Number of past time steps per input sequence.
         * @param outRows Number of output (horizon) rows.
         * @param trainFraction Fraction of patterns used for training.
         * @param shuffleCalib Whether to shuffle calibration patterns before splitting.
         * @param seed Random seed used for shuffling.
         * @return Tuple of (train input sequences, train outputs, validation input sequences,
         * validation outputs, train indices, validation indices).
         */
        std::tuple<std::vector<Eigen::MatrixXd>, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
        makeLstmPastData(int timeSteps,
                     int outRows,
                     double trainFraction,
                     bool shuffleCalib,
                     unsigned seed) const;

        /// Build LSTM train/validation datasets from past and future time steps.
        /**
         * @param pastTimeSteps Number of past time steps per input sequence.
         * @param futureTimeSteps Number of future time steps per input sequence.
         * @param trainFraction Fraction of patterns used for training.
         * @param shuffleCalib Whether to shuffle calibration patterns before splitting.
         * @param seed Random seed used for shuffling.
         * @return Tuple of (train past sequences, train future sequences, train outputs,
         * validation past sequences, validation future sequences, validation outputs,
         * train indices, validation indices).
         */
        std::tuple<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
        makeLstmPastFutureData(int pastTimeSteps,
                    int futureTimeSteps,
                    double trainFraction,
                    bool shuffleCalib,
                    unsigned seed) const;

        /// Load a previously prepared (e.g. transformed) matrix from a CSV file.
        /**
         * @param path Path of the CSV file.
         * @return Loaded matrix.
         */
        Eigen::MatrixXd loadPreparedMatrix(const std::string& path);

    protected:

    private:
        static void splitCSVLine(const std::string& line, std::vector<std::string>& outFields);  //!< Helper function to parse a CSV line into fields (handles quotes)

        std::vector<std::string> m_timestamps;  //!< Time string
        Eigen::MatrixXd m_data;  //!< Matrix with numeric columns (rows x cols) filled with variables
        std::vector<std::string> m_colNames;  //!< Column names for m_data
        Eigen::MatrixXd calibMat; //!< Matrix of inputs and desired outputs for backpropagation
        Eigen::MatrixXd calibInpsMat; //!< Matrix of inputs for backpropagation
        Eigen::MatrixXd calibOutsMat; //!< Matrix of desired outputs for backpropagation

        std::vector<int> m_calib_pattern_filtered_indices;   //!< Indices in filtered m_data for each calib pattern (reference output row)
        std::vector<size_t> m_calib_pattern_orig_indices;    //!< Indices in original unfiltered data (if available)

        // Global transform config
        std::vector<transform_type> m_transforms;  //!< Transform type configured per numeric column
        std::vector<double> m_alpha;  //!< Alpha parameter per column (used by NONLINEAR transforms)
        bool m_excludeLastCol = false;  //!< True if the last column is excluded from the transform
        Scaler m_scaler;   //!< Stores per-column min/max after a MINMAX fit

        // Deal with NAs in the dataset
        Eigen::MatrixXd m_data_backup;  //!< Full original data backup (before filtering)
        std::vector<std::string> m_timestamps_backup;  //!< Timestamps backup
        std::vector<size_t> m_na_row_indices;  //!< Indices of removed rows (in original coordinates)
        bool m_has_filtered_rows = false;  //!< True if removeRowsWithNa() was applied
};

#endif // DATA_HPP