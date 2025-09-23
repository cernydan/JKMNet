#include "CNNLayer.hpp"
#include <iostream>
#include <random>
#include <cmath>

using namespace std;

/**
 *  Setter for 1D filters matrix
 */
void CNNLayer::setFilters1D(const Eigen::MatrixXd& newFilters){
    filters1D = newFilters;
}

/**
 *  Setter for 1D bias vector
 */
void CNNLayer::setBias1D(const Eigen::VectorXd& newBias){
    bias1D = newBias;
}

 /**
 *  Getter for 1D filters matrix
 */
Eigen::MatrixXd CNNLayer::getFilters1D(){
    return filters1D;
}

 /**
 *  Getter: Returns the current architecture (vector of neurons in each layer)
 */
void CNNLayer::setCurrentInput1D(const Eigen::MatrixXd& currentInp){
    currentInput1D = currentInp;
}

 /**
 *  Calculate activations and output of the layer
 */ 
void CNNLayer::calculateOutput1D(std::string activFunc){
    for(int i = 0; i < sizes.numFilt; i++){
        for(int j = 0; j < sizes.varSize - sizes.filtSize + 1; j++){
            activation1D.block(j , i * sizes.numVars, 1 , sizes.numVars) = 
            (filters1D.row(i) * currentInput1D.block(j , 0 , sizes.filtSize,sizes.numVars)).array() + bias1D(i);
        }
    }
    switch (strToActivation(activFunc)) {
        case activ_func_type::RELU:  // f(x) = max(0, x)
            output1D = activation1D.array().max(0.0);
            break;
        
        case activ_func_type::SIGMOID:  // f(x) = 1 / (1 + exp(-x))
            output1D = activation1D.array().unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
            break;

        case activ_func_type::LINEAR:  // f(x) = x
            output1D = activation1D; 
            break;

        case activ_func_type::TANH:  // f(x) = (2 / (1 + exp(-2x))) - 1
            output1D = activation1D.array().unaryExpr([](double x) { return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0; });
            break;
        
        case activ_func_type::GAUSSIAN:  // f(x) = exp(-x^2)
            output1D = activation1D.array().unaryExpr([](double x) { return std::exp(-x * x); });
            break;

        case activ_func_type::IABS:  // f(x) = x / (1 + |x|)
            output1D = activation1D.array().unaryExpr([](double x) { return x / (1.0 + std::abs(x)); });
            break;  

        case activ_func_type::LOGLOG:  // f(x) = exp(-exp(-x))
            output1D = activation1D.array().unaryExpr([](double x) { return std::exp(-1 * std::exp(-x)); });
            break; 
        
        case activ_func_type::CLOGLOG:  // f(x) = 1 - exp(-exp(x))
            output1D = activation1D.array().unaryExpr([](double x) { return 1 - std::exp(- std::exp(x)); });
            break;

        case activ_func_type::CLOGLOGM:  // f(x) = -log(1 - exp(-x)) + 1
            output1D = activation1D.array().unaryExpr([](double x) { return 1 - 2 * std::exp(-0.7 * std::exp(x)); });
            break;

        case activ_func_type::ROOTSIG:  // f(x) = x / ((1 + sqrt(1 + x^2)) * sqrt(1 + x^2))
            output1D = activation1D.array().unaryExpr([](double x) { return x / (1 + std::sqrt(1.0 + std::exp(-x * x))); });
            break;

        case activ_func_type::LOGSIG:  // f(x) = sigmoid(x)^2
            output1D = activation1D.array().unaryExpr([](double x) { return pow((1 / (1 + std::exp(-x))), 2); });
            break;

        case activ_func_type::SECH:  // f(x) = 2 / (exp(x) + exp(-x))
            output1D = activation1D.array().unaryExpr([](double x) { return 2 / (std::exp(x) + std::exp(-x)); });
            break;

        case activ_func_type::WAVE:  // f(x) = (1 − 𝑎2) exp(−𝑎2)
            output1D = activation1D.array().unaryExpr([](double x) { return (1 - x * x) * exp(-x * x); });
            break;

        case activ_func_type::LEAKYRELU:   // f(x) = max(0.01 * x, x)
            output1D = activation1D.array().unaryExpr([](double x) { return x > 0.0 ? x : 0.01 * x; });
            break;

        default:
            std::cerr << "[Error]: Unknown activation function type!" << std::endl;
            break;
    }

    // Detect any NaN or infinite
    if (!output1D.array().isFinite().all()) {
        std::cerr << "[Warning] Non-finite activations detected!\n";
    }
}

Eigen::MatrixXd CNNLayer::getOutput1D(){
    return output1D;
}

Eigen::MatrixXd CNNLayer::getActivations1D(){
    return activation1D;
}