// TODO: Join 'setArchitecture()' with 'layer.initLayer()'.

// **DONE**: Create getter and setter for 'numLayers' (in MLP).
// **DONE**: Create getter and setter for 'weights' and 'inputs' (in Layer) and update the code.
// **DONE**: Define activation functions usinf 'switch()'. 
// **DONE**: Calculate activations as 'weights * inputs + bias'.
// **DONE**: Initialize 'bias' to 1.
// **DONE**: Solve the Eigen library errors.
// **DONE**: Check and update activation functions.

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
  mlp.setNumLayers(99);
  std::cout << "Number of layers after setting: " << mlp.getNumLayers() << std::endl;


  //!! ------------------------------------------------------------
  //!! LAYER
  //!! ------------------------------------------------------------

  // Initialize the layer 
  layer.initLayer(5, 3);  // Example: 5 inputs, 3 neurons in the layer

  // Example input
  Eigen::VectorXd inputs(5);
  inputs << 1.0, -2.0, 0.5, 0.3, -0.5;
  layer.setInputs(inputs);

  // Get and print the inputs to the layer
  Eigen::VectorXd layerInputs = layer.getInputs();
  std::cout << "Layer Inputs: " << layerInputs.transpose() << std::endl;

  // Calculate activations using ReLU
  Eigen::VectorXd reluActivations = layer.calculateActivation(activ_func_type::RELU);
  std::cout << "ReLU Activations: " << reluActivations.transpose() << std::endl;

  // Calculate activations using Sigmoid
  Eigen::VectorXd sigmoidOutput = layer.calculateActivation(activ_func_type::SIGMOID);
  std::cout << "Sigmoid Activations: " << sigmoidOutput.transpose() << std::endl;
  
  // Calculate activations using Linear
  Eigen::VectorXd linearOutput = layer.calculateActivation(activ_func_type::LINEAR);
  std::cout << "Linear Activations: " << linearOutput.transpose() << std::endl;

  // Calculate activations using Tanh
  Eigen::VectorXd tanhOutput = layer.calculateActivation(activ_func_type::TANH);
  std::cout << "Tanh Activations: " << tanhOutput.transpose() << std::endl;

  // Get and print the output of the layer
  Eigen::VectorXd output = layer.getOutput();
  std::cout << "Layer Outputs: " << output.transpose() << std::endl;

  return 0;
}
