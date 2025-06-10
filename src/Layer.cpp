#include "Layer.hpp"
#include <iostream>
#include "eigen-3.4/Eigen/Dense"

using namespace std;

/**
 * The constructor
 */
Layer::Layer(): weights(),
        inputs(),
        activations(),
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
        activations(),
        output(),
        bias() {

    weights = other.weights;
    inputs = other.inputs;
    activations= other.activations;
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
    activations= other.activations;
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
    activations = Eigen::VectorXd(numNeurons);
    activations.setZero();  // to zero

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
 * Calculate activations based on the activation function type
 */
Eigen::VectorXd Layer::calculateActivation(activ_func_type activFuncType) {
    
    //std::cout << "weights dimensions: " << weights.rows() << "x" << weights.cols() << std::endl;
    //std::cout << "inputs dimensions: " << inputs.size() << std::endl;
    //std::cout << "bias dimensions: " << bias.size() << std::endl;

    activations = weights * inputs + bias;  // Compute the weighted sum
    Eigen::VectorXd Activations = activations;  // Create new vector for activations after the activation function

    // numActivation = Activations.size(); 

    // Apply the activation function based on the activation function type
    switch (activFuncType) {
        case activ_func_type::RELU:  // f(x) = max(0, x)
            Activations = Activations.array().max(0.0);
            break;
        
        case activ_func_type::SIGMOID:  // f(x) = 1 / (1 + exp(-x))
            // For loop approach: (might be slower for large datasets)
            // for (int i = 0; i < numActivation; ++i) {
            //     Activations[i] = 1.0 / (1.0 + std::exp(-Activations[i]));
            // }

            // Lambda approach: (might be faster for large datasets)
            Activations = Activations.array().unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
            break;

        case activ_func_type::LINEAR:  // f(x) = k*x ??
            Activations = Activations; 
            break;

        case activ_func_type::TANH:  // f(x) = (2 / (1 + exp(-2x))) - 1
            Activations = Activations.array().unaryExpr([](double x) { return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0; });
            break;

        default:
            std::cerr << "Error: Unknown activation function type!" << std::endl;
            break;
    }

    output = Activations;  // Set the output of the layer
    return Activations; 
    // TODO: aktivace (lin komb) jedna metoda (napr. calculateActivation) + aktivacni fce druha metoda (napr. calculateLayerOutput)
    // TODO: druha metoda vraci jen output, tj. prejmenuj (napr. layerOutput) misto Activations (neni traba oboji)
}

/**
 * Getter for the output vector
 */
Eigen::VectorXd Layer::getOutput() {
    return output; 
}
