#include "MLP.hpp"

#include <iostream>

using namespace std;

/**
 * The constructor
 */
MLP::MLP(): nNeurons(){

}

/**
 * The destructor
 */
MLP::~MLP(){

}

/**
 *  Getter: Returns the current architecture (vector of neurons in each layer)
 */
std::vector<unsigned> MLP::getArchitecture() const {
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
void MLP::printArchitecture() const {
    std::cout << "MLP Architecture: ";

    for (size_t i = 0; i < nNeurons.size(); ++i) {
        std::cout << nNeurons[i];
        if (i != nNeurons.size() - 1) {
            std::cout << " -> ";
        }
    }

    std::cout << std::endl;
}