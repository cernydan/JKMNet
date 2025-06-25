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

    // if user hasn’t supplied activations yet, give them a default (e.g. ReLU)
    if (activFuncs.size() != architecture.size()) {
        activFuncs.assign(architecture.size(), activ_func_type::RELU);
    }
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
 *  Getter: Returns the current activation functions for each layer
 */
std::vector<activ_func_type> MLP::getActivations() {
    return activFuncs;
}

/**
 *  Setter: Sets the activation functions
 */
void MLP::setActivations(std::vector<activ_func_type>& funcs) {
    if (funcs.size() != nNeurons.size()) {
        throw std::invalid_argument("[MLP] Activation vector length must match number of layers");
    }
    activFuncs = funcs;
}

/**
 *  Print the activation functions
 */
void MLP::printActivations() {
    if (activFuncs.empty()) {
        std::cout << "No activation functions set.\n";
        return;
    }
    std::cout << "Activations per layer:\n";
    for (size_t i = 0; i < activFuncs.size(); ++i) {
        std::cout << "  Layer " << i
                  << " (" << nNeurons[i] << " neurons): "
                  << Layer::activationName(activFuncs[i]) 
                  << "\n";
    }
}

/**
 *  Getter: Returns the weight initialization type for each layer
 */
std::vector<weight_init_type> MLP::getWInitType() {
    return wInitTypes;
}

/**
 *  Setter: Sets the weight initialization type
 */
void MLP::setWInitType(std::vector<weight_init_type>& wInits) {
    if (wInits.size() != nNeurons.size()) {
        throw std::invalid_argument(
          "[MLP] wInitTypes length must equal number of layers (" +
          std::to_string(nNeurons.size()) + ")");
    }
    wInitTypes = wInits;
}

/**
 *  Print the weight initialization type
 */
void MLP::printWInitType() {
    if (wInitTypes.empty()) {
        std::cout << "No weight init types set.\n";
        return;
    }
    std::cout << "Weight initialization per layer:\n";
    for (size_t i = 0; i < wInitTypes.size(); ++i) {
        std::cout << "  Layer " << i
                  << " (" << nNeurons[i] << " neurons): "
                  << Layer::wInitTypeName(wInitTypes[i]) 
                  << "\n";
    }
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
        std::cerr << "-> [Error]: Number of layers does not match the architecture!" << std::endl;
    }
    else { 
        // Positive feedback message
        std::cout << "-> [Info]: Number of layers matches the architecture." << std::endl;
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

/**
 * Get the number of neurons at a specific layer index
 */
unsigned MLP::getNumNeuronsInLayers(std::size_t index) {
    if (index >= nNeurons.size()) {
        throw std::out_of_range("[Error]: Layer index out of range");
    }
    return nNeurons[index];
}

/**
 * Set the number of neurons at a specific layer index
 */
void MLP::setNumNeuronsInLayers(std::size_t index, unsigned count) {
    if (index >= nNeurons.size()) {
        throw std::out_of_range("[Error]: Layer index out of range");
    }
    nNeurons[index] = count;
}

/**
 * Getter for the inputs
 */
Eigen::VectorXd& MLP::getInps() {
    return Inps;
}

/**
 * Setter for the inputs
 */
void MLP::setInps(Eigen::VectorXd& inputs) {
    // Resize to (real inputs + bias):
    Inps.resize(inputs.size() + 1);
    Inps.head(inputs.size()) = inputs;       
    Inps(inputs.size()) = 1.0;           
}

/**
 * Validate the size of the inputs compared to nNeurons[0]
 */
bool MLP::validateInputSize() {
    // There has to be an input layer
    if (nNeurons.empty()) {
        std::cerr << "[MLP] No input layer defined!\n";
        return false;
    }

    // Inps must have at least the bias slot
    if (Inps.size() < 1) {
        std::cerr << "[MLP] Inps is empty!\n";
        return false;
    }

    // realInputs = total slots minus the bias slot
    auto realInputs = Inps.size() - 1;

    // Compare to nNeurons[0]
    if (realInputs != nNeurons[0]) {
        std::cerr << "[MLP] Mismatch: nNeurons[0] = "
                  << nNeurons[0] << " but got " << realInputs
                  << " real inputs (Inps.size() = " << Inps.size() << ")\n";
        return false;
    }

    return true;
}

/**
 * Forward pass through all layers
 */
Eigen::VectorXd MLP::initMLP(Eigen::VectorXd& input) {
    if (nNeurons.empty() || activFuncs.size() != nNeurons.size() || wInitTypes.size() != nNeurons.size())
        throw std::logic_error("MLP not fully configured (architecture/activ/weight init mismatch)");

    layers_.clear();
    layers_.reserve(nNeurons.size());

    // Layer[0] from real inputs
    layers_.emplace_back();
    layers_[0].initLayer(
        /*numInputs=*/ input.size(),
        /*numNeurons=*/ nNeurons[0],
        /*wInitType=*/ wInitTypes[0],
        /*func=*/ activFuncs[0]
    );
    layers_[0].setInputs(input);
    layers_[0].calculateLayerOutput(activFuncs[0]);
    Eigen::VectorXd curr = layers_[0].getOutput();

    // Remaining layers in a for loop
    for (size_t i = 1; i < nNeurons.size(); ++i) {
        layers_.emplace_back();
        layers_[i].initLayer(
            curr.size(),
            nNeurons[i],
            wInitTypes[i],
            activFuncs[i]
        );
        layers_[i].setInputs(curr);
        layers_[i].calculateLayerOutput(activFuncs[i]);
        curr = layers_[i].getOutput();
    }
   
    return curr;
}