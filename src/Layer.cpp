#include "Layer.hpp"
#include <iostream>
#include "eigen-3.4/Eigen/Dense"

using namespace std;

/**
 * The constructor
 */
Layer::Layer(): weights(),
        inputs(),
        //activations(),
        output(),
        bias() {

}

/**
 * The destructor
 */
Layer::~Layer(){

}

/**
 * The copy constructor
 */
Layer::Layer(const Layer& other): weights(),
        inputs(),
        //activations(),
        output(),
        bias() {

    weights = other.weights;
    inputs = other.inputs;
    //activations= other.activations;
    output= other.output;
    bias = other.bias; 

}

/**
 * The assignment operator
 */
Layer& Layer::operator=(const Layer& other){
    if (this == &other) return *this;
  else {
    weights = other.weights;
    inputs = other.inputs;
    //activations= other.activations;
    output= other.output;
    bias = other.bias; 
  }
  return *this;

}

/**
 * Initialize the layer with the specified number of neurons and input size
 */
void Layer::initLayer(unsigned numInputs, unsigned numNeurons) {
    // Initialize weights
    weights = Eigen::MatrixXd::Random(numNeurons, numInputs);  // Random initialization
    // mozno latinske ctverce
    

    // Initialize inputs
    inputs = Eigen::VectorXd(numInputs);
    inputs.setZero();  // to zero
    //inputs(numInputs) = 1.0;
    
    // Initialize bias
    bias = Eigen::VectorXd(numNeurons);
    bias.setOnes();  // to one

    // Initialize activations
    //activations = Eigen::VectorXd(numNeurons);
    //activations.setZero();  // to zero

    // Initialize output 
    output = Eigen::VectorXd(numNeurons);
    output.setZero();  // to zero

}

/**
 * Getter for the input vector to the layer
 */
Eigen::VectorXd Layer::getInputs() {
    return inputs;  
}

/**
 * Setter for the input vector to the layer
 */
void Layer::setInputs(const Eigen::VectorXd& newInputs) {
    inputs = newInputs; 
}

/**
 * Getter for the weight matrix of the layer
 */
Eigen::MatrixXd Layer::getWeights() {
    return weights; 
}

/**
 * Setter for the weight matrix of the layer
 */
void Layer::setWeights(const Eigen::MatrixXd& newWeights) {
    weights = newWeights;
}

/**
 * Calculate the weighted sum (linear combination of inputs)
 */
Eigen::VectorXd Layer::calculateWeightedSum() {
    //std::cout << "weights dimensions: " << weights.rows() << "x" << weights.cols() << std::endl;
    //std::cout << "inputs dimensions: " << inputs.size() << std::endl;
    //std::cout << "bias dimensions: " << bias.size() << std::endl;

    Eigen::VectorXd activation = weights * inputs + bias;  // Compute the weighted sum
    return activation;
}

/**
 * Apply activation function to the weighted sum
 */
Eigen::VectorXd Layer::applyActivationFunction(const Eigen::VectorXd& weightedSum, activ_func_type activFuncType) {
    Eigen::VectorXd activatedOutput = weightedSum;  // Copy the weighted sum

    // Apply the activation function based on the activation function type
    switch (activFuncType) {
        case activ_func_type::RELU:  // f(x) = max(0, x)
            activatedOutput = activatedOutput.array().max(0.0);
            break;
        
        case activ_func_type::SIGMOID:  // f(x) = 1 / (1 + exp(-x))
            // For loop approach: (might be slower for large datasets)
            // for (int i = 0; i < numActivation; ++i) {
            //     activatedOutput[i] = 1.0 / (1.0 + std::exp(-activatedOutput[i]));
            // }

            // Lambda approach: (might be faster for large datasets)
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
            break;

        case activ_func_type::LINEAR:  // f(x) = x
            activatedOutput = activatedOutput; 
            break;

        case activ_func_type::TANH:  // f(x) = (2 / (1 + exp(-2x))) - 1
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0; });
            break;

        default:
            std::cerr << "Error: Unknown activation function type!" << std::endl;
            break;
    }

    return activatedOutput;
}

/**
 * Calculate complete layer output (weighted sum + activation function)
 */
Eigen::VectorXd Layer::calculateLayerOutput(activ_func_type activFuncType) {
    Eigen::VectorXd weightedSum = calculateWeightedSum();
    output = applyActivationFunction(weightedSum, activFuncType);

    return output;
}


/**
 * Getter for the output vector
 */
Eigen::VectorXd Layer::getOutput() {
    return output; 
}
