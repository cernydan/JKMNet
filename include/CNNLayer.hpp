#ifndef CNNLAYER_HPP
#define CNNLAYER_HPP

#include "eigen-3.4/Eigen/Dense"
#include "ConfigIni.hpp"

struct Sizes {
    int varSize = 0;    //!< Number of variables values in current input (input rows)
    int numVars = 0;    //!< Number of input variables (input cols)
    int numFilt = 0;     //!< Number of filters in layer (filters rows)
    int filtSize = 0;    //!< Length of filters (filters cols)
    int poolSize = 0;
};

enum class pool_type
{
    NONE = 0,
    MAX,
    AVG

}; //!< Pooling layer types

inline pool_type strToPoolType(const std::string &s) {
    std::string u;
    for (char c : s) u.push_back(static_cast<char>(std::toupper((unsigned char)c)));
    if (u == "NONE") return pool_type::NONE;
    if (u == "MAX") return pool_type::MAX;
    if (u == "AVG") return pool_type::AVG;

    throw std::runtime_error("Unknown pool: " + s);
}

class CNNLayer
{
public:
    CNNLayer() = default;  //!< The constructor
    ~CNNLayer() = default;  //!< The destructor 
    CNNLayer(const CNNLayer&) = default;  //!< The copy constructor
    CNNLayer& operator=(const CNNLayer&) = default;   //!< The assignment operator
    CNNLayer(CNNLayer&&) = default;
    CNNLayer& operator=(CNNLayer&&) = default;

    //!< Initialize a CNN layer 
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

    Eigen::MatrixXd convolution1D(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& filters);
    Eigen::MatrixXd convolution1DSeparCols(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& filters);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> maxPool(const Eigen::MatrixXd& inputs, int size);
    Eigen::MatrixXd averagePool(const Eigen::MatrixXd& inputs, int size);
    Eigen::MatrixXd flipRowsAndPad(const Eigen::MatrixXd& mat, int pad);
    Eigen::MatrixXd sumColumnBlocks(const Eigen::MatrixXd& mat, int blockSize);
    void calculateOutput1D();   //!< Calculate activations and output of the layer
    Eigen::MatrixXd getOutput1D(); 
    Eigen::MatrixXd getActivations1D();
    Eigen::VectorXd getBias1D();
    Eigen::VectorXd getDelta();
    void calculateGradients();
    void setDeltaFromNextLayer(const Eigen::VectorXd& nextDelta);
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
    pool_type pool;
    Eigen::VectorXd delta;
    Eigen::VectorXd deltaFromNextLayer;
    Eigen::MatrixXd filtersGradient;
    Eigen::MatrixXd inputGradient;
    Eigen::VectorXd biasGradient;
    Eigen::MatrixXd MtForAdam;
    Eigen::MatrixXd VtForAdam;
    Eigen::VectorXd MtForAdamBias;
    Eigen::VectorXd VtForAdamBias;
    Eigen::VectorXd avgPoolBpHelp;
    Eigen::VectorXd maxPoolBpHelp;
    Eigen::MatrixXi maxPoolBpIndicesHelp;

};

#endif // CNNLAYER_HPP