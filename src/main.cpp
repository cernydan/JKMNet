// ********* 10. 6. 2025 *********
// TODO: The move copy constructor [PM] (in Layer, MLP, JKMNet)
// TODO: The move assignment operator [PM] (in Layer, MLP, JKMNet)
// **DONE**: Change 'bias' to be a part of inputs
// **DONE**(?): Getter and setter for 'weights' (in Layer)
// **DONE**: Getter and setter for 'gradient' (calculation of model's error for backpropagation and optimization of weights) (in Layer)
// **DONE**: Update weights based on the gradient (in Layer)
// **DONE**: Split 'calculateActivation' into two methods (in Layer)
// **DONE**: Method for initialization of weights - 'random' and also 'LHS' and 'LHS2' (in Layer)
// **DONE**: Add more activation functions [MK] (in Layer)
// **DONE**: Getter and setter for 'numInputs', 'numLayers' (in MLP)
// **DONE**: Getter and setter for vector of 'numNeuronsInLayers' (in MLP)
// TODO: Save 'weights' from previous iterations
// *******************************

// ********* 16. 6. 2025 *********
// **DONE**: Detect any NaN or infinite values in acivation (in Layer)
// TODO: Test large values of activations for NA's in f(a), e.g. 'a' in (-10 000, 10 000)
// TODO: Add regularization - weights that give large activations should be penalized (so that the model is not overtrained)
// *******************************

// ********* 24. 6. 2025 *********
// **DONE**: Add bias to inputs (vs. change the last input to bias (currently)) (in Layer)
// TODO: Add 'activ_func' into constructor, etc. (in Layer)
// TODO: Initialize 'activ_func' as some default function (e.g. ReLU) (in Layer)
// TODO: Add 'nNeurons' and 'Inps' as private variables (in MLP)
// TODO: Getter and setter for 'Inps' (in MLP)
// TODO: Test size of 'nNeurons'[0] vs. size of 'Inps' (in MLP)
// TODO: Add vector of activation functions for each neuron (in MLP)
// TODO: Test size of vector of activation functions vs. size of 'nNeurons' (in MLP)
// TODO: Add vector of weights initialization for each neuron (in MLP)
// TODO: Test size of vector of weights initialization vs. size of 'nNeurons' (in MLP)
// TODO: Create 'initMLP' method which initializes layer[0] and then the others in a for loop (in MLP)
// TODO: Create 'runMLP' method which runs the MLP without initialization (in MLP)
// TODO: Test that 'runMLP' produces the same results at all runs (in main)
// TODO: Getter and setter for 'weights' (in MLP)
// *******************************


#include "JKMNet.hpp"
#include "MLP.hpp"
#include "Layer.hpp"

using namespace std;

int main() {

  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "-- Testing the functionalities of JKMNet --" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  JKMNet NNet;
  MLP mlp;
  Layer layer;

  //!! ------------------------------------------------------------
  //!! ARCHITECTURE
  //!! ------------------------------------------------------------

  std::cout << "------------" << std::endl;
  std::cout << "--  MLP --" << std::endl;
  std::cout << "------------" << std::endl;

  // Print the current architecture (using the getter)
  std::cout << "Getter Architecture: ";
  std::vector<unsigned> architecture = mlp.getArchitecture();
  // Output the architecture
  for (size_t i = 0; i < architecture.size(); ++i) {
    std::cout << architecture[i];
    if (i != architecture.size() - 1) {
        std::cout << " -> ";
    }
  } 
  std::cout << std::endl;

  // Print the default architecture
  std::cout << "Default Architecture: ";
  mlp.printArchitecture();

  // Set a new architecture
  std::vector<unsigned> newArchitecture1 = {50, 32, 16, 3}; // Example: 50 inputs, 32/16 hidden neurons, 5 outputs
  mlp.setArchitecture(newArchitecture1);
  std::cout << "Architecture 1: ";
  // Print the updated architecture
  mlp.printArchitecture();

  // Set a new architecture
  std::vector<unsigned> newArchitecture2 = {10, 20, 30, 6, 7}; // Example: 100 inputs, 20/10/6 hidden neurons, 5 outputs
  mlp.setArchitecture(newArchitecture2);
  std::cout << "Architecture 2: ";
  // Print the updated architecture
  mlp.printArchitecture();

  // Get the number of layers
  size_t numLayers = mlp.getNumLayers();
  std::cout << "Number of layers: " << numLayers << std::endl;

  // Get the number of inputs
  unsigned numInputs = mlp.getNumInputs();
  std::cout << "Number of inputs: " << numInputs << std::endl;

  // Set the number of layers
  std::cout << "Number of layers before setting: " << mlp.getNumLayers() << std::endl;
  mlp.setNumLayers(999);
  std::cout << "Number of layers after setting: " << mlp.getNumLayers() << std::endl;

  // Set the number of inputs
  mlp.setNumInputs(100);
  std::cout << "Number of inputs after setting: " << mlp.getNumInputs() << std::endl;

  // Print the updated architecture
  std::cout << "Architecture after setting inputs: ";
  mlp.printArchitecture();

  // Get the number of neurons at specific layer
  unsigned h0 = mlp.getNumNeuronsInLayers(0); 
  unsigned h1 = mlp.getNumNeuronsInLayers(1);   
  unsigned h2 = mlp.getNumNeuronsInLayers(2);   
  std::cout << "Number of neurons in the input layer: " << h0 << std::endl;
  std::cout << "Number of neurons in the 1st hidden layer: " << h1 << std::endl;
  std::cout << "Number of neurons in the 2nd hidden layer: " << h2 << std::endl;

  // Set the number of neurons at specific layer
  mlp.setNumNeuronsInLayers(2, 90);  // 2nd hidden layer change to 90
  std::cout << "Change number of neurons at 2nd hidden layer to: " << mlp.getNumNeuronsInLayers(2) << std::endl;

  // Print the updated architecture
  std::cout << "Architecture after changing neurons at 2nd hidden layer: ";
  mlp.printArchitecture();
  
  //!! ------------------------------------------------------------
  //!! LAYER
  //!! ------------------------------------------------------------

  std::cout << "------------" << std::endl;
  std::cout << "--  LAYER --" << std::endl;
  std::cout << "------------" << std::endl;

  // Initialize the layer: 5 inputs (including bias!), 3 neurons in the layer (others are default)
  //layer.initLayer(5, 3); 

  // Initialize the layer: 5 inputs (including bias!), 3 neurons, random weights between 0.0 and 1.0
  layer.initLayer(5, 3, weight_init_type::RANDOM, 0.0, 1.0);
  // Print the weights matrix
  Eigen::MatrixXd printWeightsR = layer.getWeights();
  std::cout << "Weights initialized to (RAND):\n" << printWeightsR << std::endl;

/*
  layer.initLayer(5, 3, weight_init_type::LHS, 0.0, 1.0);
  // Print the weights matrix
  Eigen::MatrixXd printWeightsL = layer.getWeights();
  std::cout << "Weights initialized to (LHS):\n" << printWeightsL << std::endl;

  layer.initLayer(5, 3, weight_init_type::LHS2, 0.0, 1.0);
  // Print the weights matrix
  Eigen::MatrixXd printWeightsL2 = layer.getWeights();
  std::cout << "Weights initialized to (LHS2):\n" << printWeightsL2 << std::endl;
*/

  // Example input
  Eigen::VectorXd actualInputs(5);
  actualInputs << 10.0, 2.0, 3.0, 4.0, 5.0;
  layer.setInputs(actualInputs);

  // Get and print the inputs to the layer
  Eigen::VectorXd layerInputs = layer.getInputs();
  std::cout << "Layer Inputs: " << layerInputs.transpose() << std::endl;

  // Calculate activations using ReLU
  Eigen::VectorXd reluActivations = layer.calculateLayerOutput(activ_func_type::RELU);
  std::cout << "ReLU Activations: " << reluActivations.transpose() << std::endl;

/*
  // Calculate activations using Sigmoid
  Eigen::VectorXd sigmoidOutput = layer.calculateLayerOutput(activ_func_type::SIGMOID);
  std::cout << "Sigmoid Activations: " << sigmoidOutput.transpose() << std::endl;
  
  // Calculate activations using Linear
  Eigen::VectorXd linearOutput = layer.calculateLayerOutput(activ_func_type::LINEAR);
  std::cout << "Linear Activations: " << linearOutput.transpose() << std::endl;

  // Calculate activations using Tanh
  Eigen::VectorXd tanhOutput = layer.calculateLayerOutput(activ_func_type::TANH);
  std::cout << "Tanh Activations: " << tanhOutput.transpose() << std::endl;

  // Calculate activations using Gaussian
  Eigen::VectorXd gaussianOutput = layer.calculateLayerOutput(activ_func_type::GAUSSIAN);
  std::cout << "Gaussian Activations: " << gaussianOutput.transpose() << std::endl;

  // Calculate activations using IABS
  Eigen::VectorXd iabsOutput = layer.calculateLayerOutput(activ_func_type::IABS);
  std::cout << "IABS Activations: " << iabsOutput.transpose() << std::endl;

  // Calculate activations using LOGLOG
  Eigen::VectorXd loglogOutput = layer.calculateLayerOutput(activ_func_type::LOGLOG);
  std::cout << "LOGLOG Activations: " << loglogOutput.transpose() << std::endl;

  // Calculate activations using CLOGLOG
  Eigen::VectorXd cloglogOutput = layer.calculateLayerOutput(activ_func_type::CLOGLOG);
  std::cout << "CLOGLOG Activations: " << cloglogOutput.transpose() << std::endl;

  // Calculate activations using CLOGLOGM
  Eigen::VectorXd cloglogmOutput = layer.calculateLayerOutput(activ_func_type::CLOGLOGM);
  std::cout << "CLOGLOGM Activations: " << cloglogmOutput.transpose() << std::endl;

  // Calculate activations using ROOTSIG
  Eigen::VectorXd rootsigOutput = layer.calculateLayerOutput(activ_func_type::ROOTSIG);
  std::cout << "ROOTSIG Activations: " << rootsigOutput.transpose() << std::endl;

  // Calculate activations using LOGSIG
  Eigen::VectorXd logsigOutput = layer.calculateLayerOutput(activ_func_type::LOGSIG);
  std::cout << "LOGSIG Activations: " << logsigOutput.transpose() << std::endl;

  // Calculate activations using SECH
  Eigen::VectorXd sechOutput = layer.calculateLayerOutput(activ_func_type::SECH);
  std::cout << "SECH Activations: " << sechOutput.transpose() << std::endl;

  // Calculate activations using WAWE 
  Eigen::VectorXd waveOutput = layer.calculateLayerOutput(activ_func_type::WAVE);
  std::cout << "WAWE Activations: " << waveOutput.transpose() << std::endl;
*/

  // Get and print the output of the layer
  Eigen::VectorXd output = layer.getOutput();
  std::cout << "Layer Outputs: " << output.transpose() << std::endl;

  return 0;
}
