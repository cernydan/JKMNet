// ********* 13. 5. 2025 *********
// **DONE**: Create getter and setter for 'numLayers' (in MLP).
// **DONE**: Create getter and setter for 'weights' and 'inputs' (in Layer) and update the code.
// **DONE**: Define activation functions usinf 'switch()'. 
// **DONE**: Calculate activations as 'weights * inputs + bias'.
// **DONE**: Initialize 'bias' to 1.
// **DONE**: Solve the Eigen library errors.
// **DONE**: Check and update activation functions.
// *******************************

// ********* 10. 6. 2025 *********
// TODO: The move copy constructor [PM] (in Layer, MLP, JKMNet)
// TODO: The move assignment operator [PM] (in Layer, MLP, JKMNet)
// **DONE**: Change 'bias' to be a part of inputs
// TODO: Getter and setter for 'weights' (in Layer)
// TODO: Getter and setter for 'gradient' (calculation of model's error for backpropagation and optimization of weights) (in Layer)
// **DONE**: Split 'calculateActivation' into two methods (in Layer)
// TODO: Method for initialization of weights - 'random' and also 'LHS' (in Layer)
// TODO: Add more activation functions based on Maca's article [MK] (in Layer)
// TODO: Getter and setter for 'numInputs', 'numLayers', vector of 'numNeuronsInLayers' (in MLP)
// TODO: Save 'weights' from previus iterations 
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
  std::vector<unsigned> newArchitecture1 = {50, 32, 16, 5}; // Example: 50 inputs, 32/16 hidden neurons, 5 outputs
  mlp.setArchitecture(newArchitecture1);
  std::cout << "Architecture 1: ";
  // Print the updated architecture
  mlp.printArchitecture();

  // // Set a new architecture
  // std::vector<unsigned> newArchitecture2 = {100, 20, 10, 6, 5}; // Example: 100 inputs, 20/10/6 hidden neurons, 5 outputs
  // mlp.setArchitecture(newArchitecture2);
  // std::cout << "Architecture 2: ";
  // // Print the updated architecture
  // mlp.printArchitecture();

  // Get the number of layers
  size_t numLayers = mlp.getNumLayers();
  std::cout << "Number of layers: " << numLayers << std::endl;

  // Set the number of layers
  mlp.setNumLayers(999);
  std::cout << "Number of layers after setting: " << mlp.getNumLayers() << std::endl;


  //!! ------------------------------------------------------------
  //!! LAYER
  //!! ------------------------------------------------------------

  // Initialize the layer 
  layer.initLayer(5, 3);  // Example: 5 inputs (including bias!), 3 neurons in the layer

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

  // Calculate activations using Sigmoid
  Eigen::VectorXd sigmoidOutput = layer.calculateLayerOutput(activ_func_type::SIGMOID);
  std::cout << "Sigmoid Activations: " << sigmoidOutput.transpose() << std::endl;
  
  // Calculate activations using Linear
  Eigen::VectorXd linearOutput = layer.calculateLayerOutput(activ_func_type::LINEAR);
  std::cout << "Linear Activations: " << linearOutput.transpose() << std::endl;

  // Calculate activations using Tanh
  Eigen::VectorXd tanhOutput = layer.calculateLayerOutput(activ_func_type::TANH);
  std::cout << "Tanh Activations: " << tanhOutput.transpose() << std::endl;

  // Get and print the output of the layer
  Eigen::VectorXd output = layer.getOutput();
  std::cout << "Layer Outputs: " << output.transpose() << std::endl;

  return 0;
}
