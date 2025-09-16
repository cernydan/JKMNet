#include "MLP.hpp"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cerrno>
#include <cstring>

using namespace std;
using namespace std::chrono;

// /**
//  * The constructor
//  */
// MLP::MLP(): nNeurons(){
//     numLayers = 0;
// }

// /**
//  * The destructor
//  */
// MLP::~MLP(){

// }

// /**
//  * The copy constructor
//  */
// MLP::MLP(const MLP& other): nNeurons(){
//     nNeurons = other.nNeurons;
//     numLayers = other.numLayers;

// }

// /**
//  * The assignment operator
//  */
// MLP& MLP::operator=(const MLP& other){
//     if (this == &other) return *this;
//   else {
//     nNeurons = other.nNeurons;
//     numLayers = other.numLayers;
//   }
//   return *this;

// }

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
    numLayers = nNeurons.size();
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
size_t MLP::getNumLayers() const {
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
 * Getter for the weights
 */
Eigen::MatrixXd MLP::getWeights(size_t idx) const {
    if (idx >= layers_.size())
      throw std::out_of_range("Layer index out of range in getWeights");
    return layers_[idx].getWeights();
}

/**
 * Setter for the weights
 */
void MLP::setWeights(size_t idx, const Eigen::MatrixXd& W) {
    if (idx >= layers_.size())
      throw std::out_of_range("Layer index out of range in setWeights");
    layers_[idx].setWeights(W);
}

/**
 * Getter for weights vector of MLP
 */
Eigen::VectorXd MLP::getWeightsVectorMlp(){
    return weightsVectorMlp;
}

/**
 * Merge weight vectors of all layers
 */
void MLP::weightsToVectorMlp(){
    int length = 0;
    for(size_t i = 0; i < layers_.size(); i++){
        layers_[i].weightsToVector();
        length += layers_[i].getWeightsVector().size();
    }
    weightsVectorMlp = Eigen::VectorXd(length);
    
    int pos = 0;
    for(size_t i = 0; i < layers_.size(); i++){
        weightsVectorMlp.segment(pos, layers_[i].getWeightsVector().size()) = layers_[i].getWeightsVector();
        pos += layers_[i].getWeightsVector().size();
    }
}

/**
 * Save weights in readable CSV text (per-layer blocks)
 */
bool MLP::saveWeightsCsv(const std::string &path) const {
    namespace fs = std::filesystem;

    if (path.empty()) {
        std::cerr << "[MLP::saveWeightsCsv] Cannot open: path is empty\n";
        return false;
    }

    try {
        fs::path p(path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path()); // no-op if exists
        }
    } catch (const std::exception &e) {
        std::cerr << "[MLP::saveWeightsCsv] Cannot create parent directories for: " << path
                  << "  (" << e.what() << ")\n";
        return false;
    }

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        int e = errno;
        std::cerr << "[MLP::saveWeightsCsv] Cannot open file: " << path
                  << "  errno=" << e << " (" << std::strerror(e) << ")\n";
        return false;
    }

    ofs << std::setprecision(12);
    // Optionally write a small header
    ofs << "# MLP weights CSV\n";
    // Write per-layer weights; format: a comment line with layer index and dimensions, then rows of CSV
    for (size_t li = 0; li < layers_.size(); ++li) {
        const Eigen::MatrixXd &W = layers_[li].getWeights(); // ensure Layer::getWeights() is const
        ofs << "#layer," << li << "," << W.rows() << "," << W.cols() << "\n";
        for (Eigen::Index r = 0; r < W.rows(); ++r) {
            for (Eigen::Index c = 0; c < W.cols(); ++c) {
                ofs << W(r, c);
                if (c + 1 < W.cols()) ofs << ",";
            }
            ofs << "\n";
        }
    }
    ofs.close();
    if (ofs.fail()) {
        std::cerr << "[MLP::saveWeightsCsv] Write failure for: " << path << "\n";
        return false;
    }
    return true;
}

/**
 * Save weights in compact binary
 */
bool MLP::saveWeightsBinary(const std::string &path) const {
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) { std::cerr << "[MLP::saveWeightsBinary] Cannot open: " << path << "\n"; return false; }
        size_t L = getNumLayers();
        ofs.write(reinterpret_cast<const char*>(&L), sizeof(L));
        for (size_t i = 0; i < L; ++i) {
            Eigen::MatrixXd W = getWeights(i);
            uint64_t rows = static_cast<uint64_t>(W.rows());
            uint64_t cols = static_cast<uint64_t>(W.cols());
            ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            // write doubles row-major
            for (Eigen::Index r = 0; r < W.rows(); ++r) {
                for (Eigen::Index c = 0; c < W.cols(); ++c) {
                    double v = W(r,c);
                    ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
                }
            }
        }
        ofs.close();
        return true;
    } catch (const std::exception &ex) {
        std::cerr << "[MLP::saveWeightsBinary] Exception: " << ex.what() << "\n";
        return false;
    }
}

/**
 * Save vector of weights in readable CSV text (per-layer blocks)
 */
bool MLP::saveWeightsVectorCsv(const std::string &path) const {
    namespace fs = std::filesystem;
    if (path.empty()) {
        std::cerr << "[MLP::saveWeightsVectorCsv] Path is empty\n";
        return false;
    }
    try {
        fs::path p(path);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
    } catch (const std::exception &e) {
        std::cerr << "[MLP::saveWeightsVectorCsv] Cannot create parent directories: "
                  << e.what() << "\n";
        return false;
    }

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[MLP::saveWeightsVectorCsv] Cannot open file: " << path << "\n";
        return false;
    }
    ofs << std::setprecision(12);

    // save as column vector
    for (int i = 0; i < weightsVectorMlp.size(); ++i) {
        ofs << weightsVectorMlp[i] << "\n";
    }

    // save as row vector
    // for (int i = 0; i < weightsVectorMlp.size(); ++i) {
    //     ofs << weightsVectorMlp[i];
    //     if (i + 1 < weightsVectorMlp.size()) ofs << ",";
    // }

    ofs << "\n";
    ofs.close();
    return !ofs.fail();
}

/**
 * Save vector of weights in compact binary
 */
bool MLP::saveWeightsVectorBinary(const std::string &path) const {
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) {
            std::cerr << "[MLP::saveWeightsVectorBinary] Cannot open: " << path << "\n";
            return false;
        }
        int64_t len = weightsVectorMlp.size();
        ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
        ofs.write(reinterpret_cast<const char*>(weightsVectorMlp.data()), len * sizeof(double));
        ofs.close();
        return true;
    } catch (const std::exception &ex) {
        std::cerr << "[MLP::saveWeightsVectorBinary] Exception: " << ex.what() << "\n";
        return false;
    }
}

/**
 * Getter for output
 */
Eigen::VectorXd& MLP::getOutput(){
    return output;
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
Eigen::VectorXd MLP::initMLP(const Eigen::VectorXd& input) {
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
    Eigen::VectorXd currentOutput = layers_[0].getOutput();

    // Remaining layers in a for loop
    for (size_t i = 1; i < nNeurons.size(); ++i) {
        layers_.emplace_back();
        layers_[i].initLayer(
            /*numInputs=*/ currentOutput.size(),
            /*numNeurons=*/ nNeurons[i],
            /*wInitType=*/ wInitTypes[i],
            /*func=*/ activFuncs[i]
        );
        layers_[i].setInputs(currentOutput);
        layers_[i].calculateLayerOutput(activFuncs[i]);
        currentOutput = layers_[i].getOutput();
    }
   
    return currentOutput;
}

/**
 * Forward pass reusing existing weights
 */
Eigen::VectorXd MLP::runMLP(const Eigen::VectorXd& input) {
    if (layers_.empty())
        throw std::logic_error("runMLP called before initMLP");

    // First layer
    layers_[0].setInputs(input);
    layers_[0].calculateLayerOutput(activFuncs[0]);
    Eigen::VectorXd currentOutput = layers_[0].getOutput();

    // Remaining layers
    for (size_t i = 1; i < layers_.size(); ++i) {
        layers_[i].setInputs(currentOutput);
        layers_[i].calculateLayerOutput(activFuncs[i]);
        currentOutput = layers_[i].getOutput();
    }
    return currentOutput;
}

/**
 * Compare if 'initMLP' and 'runMLP' produce the same output
 */
bool MLP::compareInitAndRun(const Eigen::VectorXd& input, double tol) const {
    // Make a local copy of *this* so we can init on the copy
    MLP tmp = *this;
    Eigen::VectorXd outInit = tmp.initMLP(input);
    Eigen::VectorXd outRun = tmp.runMLP(input);

    return outInit.isApprox(outRun, tol);
}

/**
 * Test that 'runMLP' produces the same results over times
 */
bool MLP::testRepeatable(const Eigen::VectorXd& input, int repeats, double tol) const {
    // Initialize once to get the baseline output
    MLP tmp = *this;
    Eigen::VectorXd baseline = tmp.initMLP(input);

    // Repeat run several times and compare
    for (int i = 0; i < repeats; ++i) {
        Eigen::VectorXd out = tmp.runMLP(input);
        if (!out.isApprox(baseline, tol)) {
            return false;
        }
    }
    return true;
}

/**
 * Forward pass and update weights with backpropagation (one input)
 */
void MLP::runAndBP(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut, double learningRate) {
    if (layers_.empty())
        throw std::logic_error("runMLP called before initMLP");

    calcOneOutput(input);

    // Output layer BP
    layers_[layers_.size()-1].setDeltas(layers_[layers_.size()-1].getOutput() - obsOut);
    layers_[layers_.size()-1].calculateOnlineGradient();
    layers_[layers_.size()-1].updateWeights(learningRate);

    // Remaining layers BP
    if(layers_.size() > 1){
        for(int i = layers_.size() - 2; i >= 0; --i){
            layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
            layers_[i].calculateOnlineGradient();
            layers_[i].updateWeights(learningRate);
        }
    }
}

/**
 * Online backpropagation - 1 calibration matrix
 */
void MLP::onlineBP(int maxIter, double maxErr, double learningRate, const Eigen::MatrixXd& calMat) {
    if (layers_.empty())
        throw std::logic_error("onlineBP called before initMLP");

    if (maxIter <= 0 || maxErr < 0.0)
        throw std::invalid_argument("maxIter and maxErr must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = calMat.rows();                // number of patterns in calibration matrix
    int inpSize = layers_[0].getInputs().size()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if ((inpSize + outSize) != calMat.cols())
        throw std::runtime_error("Matrix row length doesnt match the initialized input + output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIter + 1; iter++){
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = calMat.row(pat).segment(0,inpSize);
            Eigen::VectorXd currentObs = calMat.row(pat).segment(inpSize,outSize);
            
            calcOneOutput(currentInp);

            // Output layer BP
            layers_[lastLayerIndex].setDeltas(layers_[lastLayerIndex].getOutput() - currentObs);
            layers_[lastLayerIndex].calculateOnlineGradient();
            layers_[lastLayerIndex].updateWeights(learningRate);

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradient();
                    layers_[i].updateWeights(learningRate);
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
        }
        Error = Error / numOfPatterns;
        if(Error <= maxErr){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);
            std::cout<<"Calibration finished on "<<iter<<". iteration after "<<duration.count()<<" seconds."<<endl;
            break;
        }
    }
    if(Error > maxErr){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    std::cout<<"Calibration reached max iterations with error: "<<Error<<" after "<<duration.count()<<" seconds."<<endl;
    }
}

/**
 * Online backpropagation - separete inp out matrices
 */
void MLP::onlineBP(int maxIter, double maxErr, double learningRate, const Eigen::MatrixXd& calInpMat, const Eigen::MatrixXd& calOutMat) {
    if (layers_.empty())
        throw std::logic_error("onlineBP called before initMLP");

    if (maxIter <= 0 || maxErr < 0.0)
        throw std::invalid_argument("maxIter and maxErr must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");
    
    if (calInpMat.rows() != calOutMat.rows())
        throw std::invalid_argument("matrices have different number of rows");

    int numOfPatterns = calInpMat.rows();                // number of patterns in calibration matrix
    int inpSize = layers_[0].getInputs().size()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != calInpMat.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != calOutMat.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIter + 1; iter++){
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = calInpMat.row(pat);
            Eigen::VectorXd currentObs = calOutMat.row(pat);
            
            calcOneOutput(currentInp);

            // Output layer BP
            layers_[lastLayerIndex].setDeltas(layers_[lastLayerIndex].getOutput() - currentObs);
            layers_[lastLayerIndex].calculateOnlineGradient();
            layers_[lastLayerIndex].updateWeights(learningRate);

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradient();
                    layers_[i].updateWeights(learningRate);
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
        }
        Error = Error / numOfPatterns;
        if(Error <= maxErr){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);
            std::cout<<"Calibration finished on "<<iter<<". iteration after "<<duration.count()<<" seconds."<<endl;
            break;
        }
    }
    if(Error > maxErr){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    std::cout<<"Calibration reached max iterations with error: "<<Error<<" after "<<duration.count()<<" seconds."<<endl;
    }
}

/**
 * Online backpropagation using Adam algorithm - 1 calibration matrix
 */
void MLP::onlineAdam(int maxIter, double maxErr, double learningRate, const Eigen::MatrixXd& calMat) {
    if (layers_.empty())
        throw std::logic_error("onlineAdam called before initMLP");

    if (maxIter <= 0 || maxErr < 0.0)
        throw std::invalid_argument("maxIter and maxErr must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = calMat.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getInputs().size()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if ((inpSize + outSize) != calMat.cols())
        throw std::runtime_error("Matrix row length doesnt match the initialized input + output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIter + 1; iter++){
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = calMat.row(pat).segment(0,inpSize);
            Eigen::VectorXd currentObs = calMat.row(pat).segment(inpSize,outSize);
            
            calcOneOutput(currentInp);

            // Output layer BP
            layers_[lastLayerIndex].setDeltas(layers_[lastLayerIndex].getOutput() - currentObs);
            layers_[lastLayerIndex].calculateOnlineGradient();
            layers_[lastLayerIndex].updateAdam(learningRate,iter,0.9, 0.99, 1e-8);

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradient();
                    layers_[i].updateAdam(learningRate,iter,0.9, 0.99, 1e-8);
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
        }
        Error = Error / numOfPatterns;
        if(Error <= maxErr){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);
            std::cout<<"Calibration finished on "<<iter<<". iteration after "<<duration.count()<<" seconds."<<endl;
            break;
        }
    }
    if(Error > maxErr){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    std::cout<<"Calibration reached max iterations with error: "<<Error<<" after "<<duration.count()<<" seconds."<<endl;
    }
}

/**
 * Online backpropagation using Adam algorithm - separete inp out matrices
 */
void MLP::onlineAdam(int maxIter, double maxErr, double learningRate, const Eigen::MatrixXd& calInpMat, const Eigen::MatrixXd& calOutMat) {
    if (layers_.empty())
        throw std::logic_error("onlineAdam called before initMLP");
    
    if (maxIter <= 0 || maxErr < 0.0)
        throw std::invalid_argument("maxIter and maxErr must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");
            
    if (calInpMat.rows() != calOutMat.rows())
        throw std::invalid_argument("matrices have different number of rows");

    int numOfPatterns = calInpMat.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getInputs().size()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != calInpMat.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != calOutMat.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIter + 1; iter++){
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = calInpMat.row(pat);
            Eigen::VectorXd currentObs = calOutMat.row(pat);

            calcOneOutput(currentInp);

            // Output layer BP
            layers_[lastLayerIndex].setDeltas(layers_[lastLayerIndex].getOutput() - currentObs);
            layers_[lastLayerIndex].calculateOnlineGradient();
            layers_[lastLayerIndex].updateAdam(learningRate,iter,0.9, 0.99, 1e-8);

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradient();
                    layers_[i].updateAdam(learningRate,iter,0.9, 0.99, 1e-8);
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
        }
        Error = Error / numOfPatterns;
        if(Error <= maxErr){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);
            std::cout<<"Calibration finished on "<<iter<<". iteration after "<<duration.count()<<" seconds."<<endl;
            break;
        }
    }
    if(Error > maxErr){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    std::cout<<"Calibration reached max iterations with error: "<<Error<<" after "<<duration.count()<<" seconds."<<endl;
    }
}

/**
 * Batch backpropagation using Adam algorithm - 1 calibration matrix
 */
void MLP::batchAdam(int maxIter, double maxErr, int batchSize, double learningRate, const Eigen::MatrixXd& calMat) {
    if (layers_.empty())
        throw std::logic_error("batchAdam called before initMLP");
    
    if (maxIter <= 0 || batchSize <= 0|| maxErr < 0.0)
        throw std::invalid_argument("maxIter, batchSize and maxErr must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = calMat.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getInputs().size()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if ((inpSize + outSize) != calMat.cols())
        throw std::runtime_error("Matrix row length doesnt match the initialized input + output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIter + 1; iter++){
        Error = 0.0;
        for(int batch = 0; batch < (numOfPatterns + batchSize - 1)/batchSize; batch++){
            int start = batch * batchSize;
            int end   = std::min(start + batchSize,numOfPatterns);
            for (int pat = start; pat < end; pat++){

                Eigen::VectorXd currentInp = calMat.row(pat).segment(0,inpSize);
                Eigen::VectorXd currentObs = calMat.row(pat).segment(inpSize,outSize);
                
                calcOneOutput(currentInp);

                // Output layer gradient
                layers_[lastLayerIndex].setDeltas(layers_[lastLayerIndex].getOutput() - currentObs);
                layers_[lastLayerIndex].calculateBatchGradient();

                // Remaining layers BP
                if(layers_.size() > 1){
                    for(int i = lastLayerIndex - 1; i >= 0; --i){
                        layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                        layers_[i].calculateBatchGradient();
                    }
                }
                Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
            }
            for (int i = 0; i <= lastLayerIndex; i++){
                layers_[i].updateAdam(learningRate,iter,0.9, 0.99, 1e-8);
                layers_[i].setGradient(layers_[i].getGradient().setZero());
            }
        }
        Error = Error / numOfPatterns;
        if(Error <= maxErr){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);
            std::cout<<"Calibration finished on "<<iter<<". iteration after "<<duration.count()<<" seconds."<<endl;
            break;
        }
    }
    if(Error > maxErr){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    std::cout<<"Calibration reached max iterations with error: "<<Error<<" after "<<duration.count()<<" seconds."<<endl;
    }
}

/**
 * Batch backpropagation using Adam algorithm  - separete inp out matrices
 */
void MLP::batchAdam(int maxIter, double maxErr, int batchSize, double learningRate, const Eigen::MatrixXd& calInpMat, const Eigen::MatrixXd& calOutMat) {
    if (layers_.empty())
        throw std::logic_error("batchAdam called before initMLP");
    
    if (maxIter <= 0 || batchSize <= 0|| maxErr < 0.0)
        throw std::invalid_argument("maxIter, batchSize and maxErr must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");
        
    if (calInpMat.rows() != calOutMat.rows())
        throw std::invalid_argument("matrices have different number of rows");

    int numOfPatterns = calInpMat.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getInputs().size()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != calInpMat.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != calOutMat.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIter + 1; iter++){
        Error = 0.0;
        for(int batch = 0; batch < (numOfPatterns + batchSize - 1)/batchSize; batch++){
            int start = batch * batchSize;
            int end   = std::min(start + batchSize,numOfPatterns);
            for (int pat = start; pat < end; pat++){

                Eigen::VectorXd currentInp = calInpMat.row(pat);
                Eigen::VectorXd currentObs = calOutMat.row(pat);
                
                calcOneOutput(currentInp);

                // Output layer gradient
                layers_[lastLayerIndex].setDeltas(layers_[lastLayerIndex].getOutput() - currentObs);
                layers_[lastLayerIndex].calculateBatchGradient();

                // Remaining layers BP
                if(layers_.size() > 1){
                    for(int i = lastLayerIndex - 1; i >= 0; --i){
                        layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                        layers_[i].calculateBatchGradient();
                    }
                }
                Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
                }
                for (int i = 0; i <= lastLayerIndex; i++){
                    layers_[i].updateAdam(learningRate,iter,0.9, 0.99, 1e-8);
                    layers_[i].setGradient(layers_[i].getGradient().setZero());
                }
            }
            Error = Error / numOfPatterns;
        if(Error <= maxErr){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);
            std::cout<<"Calibration finished on "<<iter<<". iteration after "<<duration.count()<<" seconds."<<endl;
            break;
        }
    }
    if(Error > maxErr){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    std::cout<<"Calibration reached max iterations with error: "<<Error<<" after "<<duration.count()<<" seconds."<<endl;
    }
}

/**
 * Forward pass reusing existing weights
 */
void MLP::calcOneOutput(const Eigen::VectorXd& inputVec){      
    // First layer
    layers_[0].setInputs(inputVec);
    layers_[0].calculateOutput(activFuncs[0]);

    // Remaining layers
    for (size_t i = 1; i < getNumLayers(); ++i) {
        layers_[i].setInputs(layers_[i-1].getOutput());
        layers_[i].calculateOutput(activFuncs[i]);
    }
    output = layers_[getNumLayers()-1].getOutput();
}

/**
 * Calculate outputs for given matrix of inputs
 */
void MLP::calculateOutputs(const Eigen::MatrixXd& inputMat){
    if (layers_.empty())
        throw std::logic_error("calculateOutputs called before initMLP");

    int inpSize = layers_[0].getInputs().size()-1;
    if (inpSize != inputMat.cols())
        throw std::invalid_argument("Input matrix row length doesnt match the initialized input size");

    if (inputMat.rows() <= 0)
        throw std::invalid_argument("Input matrix is empty");

    outputMat = Eigen::MatrixXd(inputMat.rows(),nNeurons.back());
    for(int i = 0; i < inputMat.rows(); i++){
        calcOneOutput(inputMat.row(i));
        outputMat.row(i) = output;
    }
}

Eigen::MatrixXd MLP::getOutputs() const{
    return outputMat;
}

