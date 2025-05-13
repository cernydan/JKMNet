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
void MLP::setArchitecture(const std::vector<unsigned>& architecture) {
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

// TODO: Update for Layers, not architecture any more
/**
 *  Getter
 */
std::vector<unsigned> MLP::getLayers() {
    return nNeurons;
}

// TODO: Update for Layers, not architecture any more
/**
 *  Setter
 */
void MLP::setLayers(const std::vector<unsigned>& architecture) {
    nNeurons = architecture;
}