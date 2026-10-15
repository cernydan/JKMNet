#ifndef CNNLAYER_HPP
#define CNNLAYER_HPP

/**
 * @file CNNLayer.hpp
 * @brief Declares the CNNLayer class — a 1D convolutional layer with pooling, activation, and Adam filter updates — plus the Sizes struct and pool_type enum.
 */

#include "eigen-3.4/Eigen/Dense"
#include "ConfigIni.hpp"

/// Dimensions of the matrices involved in one convolutional layer.
struct Sizes {
    int varSize = 0;    //!< Number of variables values in current input (input rows)
    int numVars = 0;    //!< Number of input variables (input cols)
    int numFilt = 0;     //!< Number of filters in layer (filters rows)
    int filtSize = 0;    //!< Length of filters (filters cols)
    int poolSize = 0;   //!< Length of one non-overlapping pooling window (0 = no pooling)
};

/// Pooling layer types.
enum class pool_type
{
    NONE = 0,   //!< No pooling
    MAX,        //!< Maximum over each pooling window
    AVG         //!< Average over each pooling window

};

/// Convert a string (case-insensitive) to a pool_type.
/**
 * @param s Pool name ("NONE", "MAX", "AVG").
 * @return The matching pool_type.
 * @throw std::runtime_error On an unrecognized name.
 */
inline pool_type strToPoolType(const std::string &s) {
    std::string u;
    for (char c : s) u.push_back(static_cast<char>(std::toupper((unsigned char)c)));
    if (u == "NONE") return pool_type::NONE;
    if (u == "MAX") return pool_type::MAX;
    if (u == "AVG") return pool_type::AVG;

    throw std::runtime_error("Unknown pool: " + s);
}

/// 1D convolutional layer for time-series matrices: convolution, activation, pooling, and backpropagation with Adam updates.
/**
 * The layer owns a filter bank (`filters1D`, shape filtSize x numFilt) and one
 * bias per filter. calculateOutput1D convolves the current input matrix
 * (rows = variable values, cols = variables) with every filter (valid
 * convolution along rows, stride 1, no padding), adds the bias, applies the
 * configured activation function, and then optionally pools the result with
 * non-overlapping windows of `sizes.poolSize` rows (no overlap, so the row
 * count is divided by poolSize). calculateGradients computes the filter, bias,
 * and input deltas from the error signal of the following layer; pooling
 * backpropagation routes max-pool deltas to the argmax positions and spreads
 * average-pool deltas evenly.
 *
 * @see activ_func_type, weight_init_type (Layer.hpp/ConfigIni.hpp), pool_type
 */
class CNNLayer
{
public:
    CNNLayer() = default;  //!< The constructor
    ~CNNLayer() = default;  //!< The destructor
    CNNLayer(const CNNLayer&) = default;  //!< The copy constructor
    CNNLayer& operator=(const CNNLayer&) = default;   //!< The assignment operator
    CNNLayer(CNNLayer&&) = default;   //!< The move constructor
    CNNLayer& operator=(CNNLayer&&) = default;    //!< The move assignment operator

    //!< Initialize a CNN layer
    /// Initialize the layer: dimensions, filter weights, bias, activation, and pooling.
    /**
     * Filters are initialized with the selected weight-init scheme as a
     * filtSize x numFilt matrix; the bias vector (length numFilt) starts at
     * zero. Passing poolSize = 0 forces pool_type::NONE regardless of poolType.
     * @param numberOfFilters Number of filters (columns of the filter matrix).
     * @param filterSize Filter length (rows of the filter matrix); must not exceed inputRows.
     * @param inputRows Number of values per variable in the input (input matrix rows).
     * @param inputCols Number of input variables (input matrix cols).
     * @param poolSize Length of one non-overlapping pooling window (must divide the activation row count unless 0).
     * @param initType Weight initialization name ("RANDOM", "LHS", "LHS2", "HE"; unknown falls back to RANDOM).
     * @param activFunc Activation function name (see activ_func_type).
     * @param minVal Lower bound of the random/uniform initialization range.
     * @param maxVal Upper bound of the random/uniform initialization range.
     * @param poolType Pooling type name ("NONE", "MAX", "AVG").
     * @param rngSeed RNG seed (0 = non-deterministic random_device).
     * @throw std::invalid_argument On non-positive dimensions or filterSize > inputRows.
     * @throw std::runtime_error On an unrecognized poolType.
     */
    void init1DCNNLayer(int numberOfFilters, 
                        int filterSize, 
                        int inputRows,
                        int inputCols,
                        int poolSize,
                        std::string initType = "RANDOM",
                        std::string activFunc = "RELU",
                        double minVal = 0.0,
                        double maxVal = 1.0,
                        std::string poolType = "MAX",
                        int rngSeed = 0);
    void setFilters1D(const Eigen::MatrixXd& newFilters);    //!< Setter for 1D filters matrix
    void setBias1D(const Eigen::VectorXd& newBias);    //!< Setter for 1D bias vector
    Eigen::MatrixXd getFilters1D();    //!< Getter for 1D filters matrix

    void setCurrentInput1D(const Eigen::MatrixXd& currentInp); //!< Setter for current 1D input matrix

    /// 1D valid convolution of an input matrix with a filter bank (stride 1, no padding).
    /**
     * Every filter is slid along the rows of every input column; the output
     * column i * inCols + j holds filter i applied to input column j.
     * @param inputs Input matrix (rows = values, cols = variables).
     * @param filters Filter matrix (rows = filter length, cols = filters).
     * @return Matrix of shape (inputs.rows - filters.rows + 1) x (inputs.cols * filters.cols).
     */
    Eigen::MatrixXd convolution1D(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& filters);
    /// 1D valid convolution pairing each input column with the same-numbered filter column.
    /**
     * @param inputs Input matrix (rows = values, cols = variables).
     * @param filters Filter matrix with the same number of columns as inputs.
     * @return Matrix of shape (inputs.rows - filters.rows + 1) x inputs.cols.
     * @throw std::invalid_argument If inputs.cols() != filters.cols().
     */
    Eigen::MatrixXd convolution1DSeparCols(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& filters);
    /// Max pooling over non-overlapping row windows.
    /**
     * @param inputs Matrix to pool; its row count must be divisible by size.
     * @param size Pooling window length (number of rows per window).
     * @return Tuple of the pooled matrix (inputs.rows/size x inputs.cols) and the
     *         per-window argmax row indices (for backpropagation).
     * @throw std::invalid_argument If size exceeds the row count or does not divide it.
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> maxPool(const Eigen::MatrixXd& inputs, int size);
    /// Average pooling over non-overlapping row windows.
    /**
     * @param inputs Matrix to pool; its row count must be divisible by size.
     * @param size Pooling window length (number of rows per window).
     * @return Pooled matrix (inputs.rows/size x inputs.cols).
     * @throw std::invalid_argument If size exceeds the row count or does not divide it.
     */
    Eigen::MatrixXd averagePool(const Eigen::MatrixXd& inputs, int size);
    /// Reverse row order (flip) and pad with @p pad zero rows on both ends.
    /**
     * Helper for the transposed (full) convolution used in gradient computation.
     * @param mat Input matrix.
     * @param pad Number of zero rows added at the top and bottom.
     * @return Flipped-and-padded matrix of shape (mat.rows + 2*pad) x mat.cols.
     */
    Eigen::MatrixXd flipRowsAndPad(const Eigen::MatrixXd& mat, int pad);
    /// Sum consecutive blocks of columns into single columns.
    /**
     * @param mat Input matrix; its column count must be divisible by blockSize.
     * @param blockSize Number of columns merged per output column.
     * @return Matrix of shape mat.rows x (mat.cols / blockSize).
     */
    Eigen::MatrixXd sumColumnBlocks(const Eigen::MatrixXd& mat, int blockSize);
    void calculateOutput1D();   //!< Calculate activations and output of the layer
    Eigen::MatrixXd getOutput1D();   //!< Getter for the (activated and pooled) output matrix
    Eigen::MatrixXd getActivations1D(); //!< Getter for the pre-activation convolution result (with bias)
    Eigen::VectorXd getBias1D();    //!< Getter for the bias vector
    Eigen::VectorXd getDelta(); //!< Getter for the input delta (error signal for a preceding layer)
    /// Compute filter, bias, and input gradients from the stored forward pass and the next-layer delta.
    /**
     * Applies the activation derivative, backpropagates through pooling
     * (argmax routing for MAX, even spreading for AVG), and produces
     * filtersGradient, biasGradient, inputGradient, and the per-input-variable
     * delta. Must be called after calculateOutput1D and setDeltaFromNextLayer.
     */
    void calculateGradients();
    void setDeltaFromNextLayer(const Eigen::VectorXd& nextDelta);   //!< Set the error delta arriving from the following layer
    /// Apply one Adam update step to the filters and the bias from the stored gradients.
    /**
     * @param learningRate Adam step size.
     * @param iterationNum 1-based step counter for Adam's bias correction.
     * @param beta1 Exponential decay of the first moment estimate.
     * @param beta2 Exponential decay of the second moment estimate.
     * @param epsi Small stabilizer in the denominator.
     */
    void updateFiltersAdam(double learningRate, int iterationNum, double beta1, double beta2, double epsi);

protected:
private:
    Sizes sizes;    //!< Convolution matrices dimensions
    Eigen::MatrixXd filters1D;  //!< Filter matrix of the layer (col = filter)
    Eigen::MatrixXd currentInput1D;  //!< Input matrix (m_data form)
    Eigen::VectorXd bias1D;     //!< Vector of bias values for each filter
    Eigen::MatrixXd activation1D;   //!< Calculated layer activations matrix
    Eigen::MatrixXd output1D;   //!< Calculated layer output matrix
    activ_func_type activ_func = activ_func_type::RELU;  //!< The type of activation function, where default is RELU
    pool_type pool; //!< Pooling type applied to the activated convolution result
    Eigen::VectorXd delta;  //!< Input delta per input variable (error signal passed to a preceding layer)
    Eigen::VectorXd deltaFromNextLayer; //!< Error delta received from the following layer
    Eigen::MatrixXd filtersGradient;    //!< Gradient of the loss w.r.t. the filter matrix
    Eigen::MatrixXd inputGradient;  //!< Gradient of the loss w.r.t. the layer input
    Eigen::VectorXd biasGradient;   //!< Gradient of the loss w.r.t. the bias vector
    Eigen::MatrixXd MtForAdam;  //!< Adam first-moment accumulator for the filters
    Eigen::MatrixXd VtForAdam;  //!< Adam second-moment accumulator for the filters
    Eigen::VectorXd MtForAdamBias;  //!< Adam first-moment accumulator for the bias
    Eigen::VectorXd VtForAdamBias;  //!< Adam second-moment accumulator for the bias
    Eigen::VectorXd avgPoolBpHelp;  //!< Scratch vector distributing average-pool deltas over window rows
    Eigen::VectorXd maxPoolBpHelp;  //!< Scratch vector routing a max-pool delta to the argmax row
    Eigen::MatrixXi maxPoolBpIndicesHelp;   //!< Argmax row indices per pooling window, saved during maxPool forward pass

};

#endif // CNNLAYER_HPP
