#include "MLP.hpp"
#include <iostream>

using namespace std;

/**
 * The constructor
 */
MLP::MLP(): nNeurons(){
    numLayers = 0;
}

/**
 * The destructor
 */
MLP::~MLP(){

}


/**
 * The copy constructor
 */
MLP::MLP(const MLP& other): nNeurons(){
    nNeurons = other.nNeurons;
    numLayers = other.numLayers;

}


/**
 * The assignment operator
 */
MLP& MLP::operator=(const MLP& other){
    if (this == &other) return *this;
  else {
    nNeurons = other.nNeurons;
    numLayers = other.numLayers;
  }
  return *this;

}

/**
 *  Getter: Returns the current architecture (vector of neurons in each layer)
 */
std::vector<unsigned> MLP::getArchitecture() {
    return nNeurons;
}

/**
 *  Setter: Sets the architecture from a vector
 */
void MLP::setArchitecture(std::vector<unsigned>& architecture) {
    nNeurons = architecture;
}

/**
 *  Print the architecture
 */
void MLP::printArchitecture() {
    std::cout << "MLP Architecture: ";

    numLayers = nNeurons.size(); 

    for (size_t i = 0; i < numLayers; ++i) {
        std::cout << nNeurons[i];
        if (i != numLayers - 1) {
            std::cout << " -> ";
        }
    }

    std::cout << std::endl;
}

/**
 * Getter for the number of layers
 */
size_t MLP::getNumLayers() {
    return numLayers; 
}

/**
 * Setter for the number of layers
 */
void MLP::setNumLayers(size_t layers) {
    numLayers = layers; 
  
    if (nNeurons.size() != layers) {
        // Error message if number of layers and architecture do not match
        std::cerr << "-> Error: Number of layers does not match the architecture!" << std::endl;
    }
    else { 
        // Positive feedback message
        std::cout << "-> Good: Number of layers matches the architecture." << std::endl;
    }
}

/**
 * Getter for the number of inputs
 */
unsigned MLP::getNumInputs() {
    if (nNeurons.empty()) {
        return 0u;  // an unsigned zero
    }

    return nNeurons.front();  // first element of the vector
}

/**
 * Setter for the number of inputs
 */
void MLP::setNumInputs(unsigned inputs) {
    // if no layers are defined yet, we create the first layer and set its size to 'inputs'
    if (nNeurons.empty()) {
        nNeurons.push_back(inputs);

    // if we already have at least one layer, overwrite the neuron count of that first layer
    } else {
        nNeurons[0] = inputs;
    }

    numLayers = nNeurons.size();
}