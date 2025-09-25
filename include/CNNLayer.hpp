#ifndef CNNLAYER_HPP
#define CNNLAYER_HPP

#include "eigen-3.4/Eigen/Dense"
#include "ConfigIni.hpp"

struct Sizes {
    int varSize = 0;    //!< Number of variables values in current input (input rows)
    int numVars = 0;    //!< Number of input variables (input cols)
    int numFilt = 0;     //!< Number of filters in layer (filters rows)
    int filtSize = 0;    //!< Length of filters (filters cols)
};

class CNNLayer
{
public:
    CNNLayer() = default;  //!< The constructor
    ~CNNLayer() = default;  //!< The destructor 
    CNNLayer(const CNNLayer&) = default;  //!< The copy constructor
    CNNLayer& operator=(const CNNLayer&) = default;   //!< The assignment operator

    //!< Initialize a CNN layer 
    void init1DCNNLayer(int numberOfFilters, 
                        int filterSize, 
                        int inputRows,
                        int inputCols,
                        std::string initType = "RANDOM",
                        std::string activFunc = "RELU",
                        double minVal = 0.0,
                        double maxVal = 1.0);
    void setFilters1D(const Eigen::MatrixXd& newFilters);    //!< Setter for 1D filters matrix
    void setBias1D(const Eigen::VectorXd& newBias);    //!< Setter for 1D bias vector
    Eigen::MatrixXd getFilters1D();    //!< Getter for 1D filters matrix

    void setCurrentInput1D(const Eigen::MatrixXd& currentInp); //!< Setter for current 1D input matrix 

    void calculateOutput1D(std::string activFunc);   //!< Calculate activations and output of the layer
    Eigen::MatrixXd getOutput1D(); 
    Eigen::MatrixXd getActivations1D(); 

protected:
private:
    Sizes sizes;    //!< Convolution matrices dimensions
    Eigen::MatrixXd filters1D;  //!< Filter matrix of the layer (row = filter)
    Eigen::MatrixXd currentInput1D;  //!< Input matrix (m_data form)
    Eigen::VectorXd bias1D;     //!< Vector of bias values for each filter
    Eigen::MatrixXd activation1D;   //!< Calculated layer activations matrix
    Eigen::MatrixXd output1D;   //!< Calculated layer output matrix
    activ_func_type activ_func = activ_func_type::RELU;  //!< The type of activation function, where default is RELU
};

#endif // CNNLAYER_HPP