#include "Layer.hpp"
#include <iostream>
#include "Eigen/Dense"

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

    // Initialize inputs
    inputs = Eigen::VectorXd(numInputs);
    inputs.setZero();  // to zero
    
    // Initialize the bias
    bias = Eigen::VectorXd(numNeurons);
    bias.setZero();  // to zero

    // Initialize activations
    activations = Eigen::VectorXd(numNeurons);
    activations.setZero();  // to zero

    // Initialize output 
    output = Eigen::VectorXd(numNeurons);
    output.setZero();  // to zero

}

/**
 * Get the input vector to the layer
 */
Eigen::VectorXd Layer::getInputs() const {
    return inputs; 
}

/**
 * Calculate activations based on the activation function type
 */
Eigen::VectorXd Layer::calculateActivation(activ_func_type activFuncType) {
    
    //std::cout << "weights dimensions: " << weights.rows() << "x" << weights.cols() << std::endl;
    //std::cout << "inputs dimensions: " << inputs.size() << std::endl;
    //std::cout << "bias dimensions: " << bias.size() << std::endl;

    Eigen::VectorXd a = weights * inputs + bias;  // Compute the weighted sum
    Eigen::VectorXd Activations = a;  // Create new vector for activations 

    // Apply the activation function based on the activation function type
    switch (activFuncType) {
        case activ_func_type::RELU:  // f(x) = max(0, x)
            //Activations = Activations.array().max(0);
            Activations = Activations;
            break;
        
        case activ_func_type::SIGMOID:  // f(x) = 1 / (1 + exp(-x))
            //Activations = Activations.array().unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
            Activations = Activations;
            break;

        case activ_func_type::LINEAR:  // f(x) = x
            Activations = Activations; 
            break;

        case activ_func_type::TANH:  // f(x) = (2 / (1 + exp(-2x))) - 1
            //Activations = Activations.array().unaryExpr([](double x) { return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0; });
            Activations = Activations;
            break;

        default:
            std::cerr << "Error: Unknown activation function type!" << std::endl;
            break;
    }

    activations = Activations;  // Store the activations
    output = Activations;  // Set the output of the layer
    return Activations;  // Return the activations

}

/**
 * Get the output of the layer after applying the activation function
 */
Eigen::VectorXd Layer::getOutput() const {
    return output; 
}