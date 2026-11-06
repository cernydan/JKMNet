/**
 * @file MLP.cpp
 * @brief Implements MLP — the multilayer perceptron: forward propagation, backpropagation
 *        (plain gradient descent, momentum, weight penalization, and Adam), batch and
 *        online training with per-epoch metrics, weight I/O, and horizon backpropagation
 *        used to couple the MLP to a preceding LSTM.
 */

#include "MLP.hpp"
#include "ObjectiveFunctions.hpp"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cerrno>
#include <cstring>

using namespace std;
using namespace std::chrono;

/**
 * The constructor
 */
/// Implements the default constructor. All members are default-constructed;
/// layers are created later by initMLP or a loadWeigths* call.
MLP::MLP()
    : nNeurons(),
      numLayers(0),
      Inps(),
      activFuncs(),
      wInitTypes(),
      layers_(),
      output(),
      outputMat(),
      weightsVectorMlp(),
      calibCrit(),
      validCrit(),
      result()
       {
   
}


/**
 * The destructor
 */
MLP::~MLP(){

}

/**
 * The copy constructor
 */
/// Implements the copy constructor. All members are copied; Eigen and
/// std::vector members are deep-copied automatically.
MLP::MLP(const MLP& other)
    : nNeurons(other.nNeurons),
      numLayers(other.numLayers),
      Inps(other.Inps),
      activFuncs(other.activFuncs),
      wInitTypes(other.wInitTypes),
      layers_(other.layers_),
      output(other.output),
      outputMat(other.outputMat),
      weightsVectorMlp(other.weightsVectorMlp),
      calibCrit(other.calibCrit),
      validCrit(other.validCrit),
      result(other.result) {
    // deep copies handled automatically by Eigen and std::vector
}


/**
 * The assignment operator
 */
/// Implements the copy assignment operator. Guarded against self-assignment.
MLP& MLP::operator=(const MLP& other){
    if (this == &other) return *this;
  else {
    nNeurons        = other.nNeurons;
        numLayers       = other.numLayers;
        Inps            = other.Inps;
        activFuncs      = other.activFuncs;
        wInitTypes      = other.wInitTypes;
        layers_         = other.layers_;
        output          = other.output;
        outputMat       = other.outputMat;
        weightsVectorMlp = other.weightsVectorMlp;
        calibCrit = other.calibCrit;
        validCrit = other.validCrit;
        result = other.result;
  }
  return *this;

}

/**
 *  Move constructor
 */
/// Implements the move constructor. Steals all Eigen/vector members from @p other.
MLP::MLP(MLP&& other) noexcept
    : nNeurons(std::move(other.nNeurons)),
      numLayers(other.numLayers),
      Inps(std::move(other.Inps)),
      activFuncs(std::move(other.activFuncs)),
      wInitTypes(std::move(other.wInitTypes)),
      layers_(std::move(other.layers_)),
      output(std::move(other.output)),
      outputMat(std::move(other.outputMat)),
      weightsVectorMlp(std::move(other.weightsVectorMlp)),
      calibCrit(std::move(other.calibCrit)),
      validCrit(std::move(other.validCrit)),
      result(std::move(other.result)) {
    
}

/**
 *  Move assignment operator
 */
/// Implements the move assignment operator. Guarded against self-assignment.
MLP& MLP::operator=(MLP&& other) noexcept {
    if (this != &other) {
        nNeurons       = std::move(other.nNeurons);
        numLayers      = other.numLayers;
        Inps           = std::move(other.Inps);
        activFuncs     = std::move(other.activFuncs);
        wInitTypes     = std::move(other.wInitTypes);
        layers_        = std::move(other.layers_);
        output         = std::move(other.output);
        outputMat      = std::move(other.outputMat);
        weightsVectorMlp = std::move(other.weightsVectorMlp);
        calibCrit = std::move(other.calibCrit);
        validCrit = std::move(other.validCrit);
        result = std::move(other.result);
    }
    return *this;
}

/**
 *  Getter: Returns the current architecture (vector of neurons per layer, including the input layer)
 */
/// Implements getArchitecture.
std::vector<unsigned> MLP::getArchitecture() {
    return nNeurons;
}

/**
 *  Setter: Sets the architecture from a vector
 */
/// Implements setArchitecture. Architectures includes the input layer (nNeurons[0] =
/// number of inputs). Also caches numLayers and, if activations were not yet
/// supplied (or their count no longer matches), fills them with the default ReLU.
void MLP::setArchitecture(const std::vector<unsigned>& architecture) {
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
/// Implements printArchitecture. Prints the layer sizes as "n0 -> n1 -> ... -> nL".
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
/// Implements getActivations.
std::vector<activ_func_type> MLP::getActivations() {
    return activFuncs;
}

/**
 *  Setter: Sets the activation functions
 */
/// Implements setActivations.
/// @throw std::invalid_argument If @p funcs does not have one entry per entry of nNeurons.
void MLP::setActivations(const std::vector<activ_func_type>& funcs) {
    if (funcs.size() != nNeurons.size()) {
        throw std::invalid_argument("[MLP] Activation vector length must match number of layers");
    }
    activFuncs = funcs;
}

/**
 *  Print the activation functions
 */
/// Implements printActivations.
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
/// Implements getWInitType.
std::vector<weight_init_type> MLP::getWInitType() {
    return wInitTypes;
}

/**
 *  Setter: Sets the weight initialization type
 */
/// Implements setWInitType.
/// @throw std::invalid_argument If @p wInits does not have one entry per entry of nNeurons.
void MLP::setWInitType(const std::vector<weight_init_type>& wInits) {
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
/// Implements printWInitType.
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
/// Implements getNumLayers.
size_t MLP::getNumLayers() const {
    return numLayers; 
}

/**
 * Setter for the number of layers
 */
/// Implements setNumLayers. Warns on std::cerr if the new count does not match
/// the size of the already set architecture.
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
/// Implements getNumInputs. The number of inputs equals the size of the first
/// layer (nNeurons[0]); returns 0 when no architecture has been set.
unsigned MLP::getNumInputs() {
    if (nNeurons.empty()) {
        return 0u;  // an unsigned zero
    }

    return nNeurons.front();  // first element of the vector
}

/**
 * Setter for the number of inputs
 */
/// Implements setNumInputs. Creates the first layer if the architecture is still
/// empty, otherwise overwrites the neuron count of the first layer.
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
/// Implements getNumNeuronsInLayers.
/// @throw std::out_of_range If @p index is outside the architecture.
unsigned MLP::getNumNeuronsInLayers(std::size_t index) {
    if (index >= nNeurons.size()) {
        throw std::out_of_range("[Error]: Layer index out of range");
    }
    return nNeurons[index];
}

/**
 * Set the number of neurons at a specific layer index
 */
/// Implements setNumNeuronsInLayers.
/// @throw std::out_of_range If @p index is outside the architecture.
void MLP::setNumNeuronsInLayers(std::size_t index, unsigned count) {
    if (index >= nNeurons.size()) {
        throw std::out_of_range("[Error]: Layer index out of range");
    }
    nNeurons[index] = count;
}

/**
 * Getter for the inputs
 */
/// Implements getInps. Returns the stored input vector including the appended bias slot.
Eigen::VectorXd& MLP::getInps() {
    return Inps;
}

/**
 * Setter for the inputs
 */
/// Implements setInps. Stores the input vector with an extra element equal to 1.0
/// appended at the end, i.e. Inps has size inputs.size() + 1 (real inputs + bias).
void MLP::setInps(Eigen::VectorXd& inputs) {
    // Resize to (real inputs + bias):
    Inps.resize(inputs.size() + 1);
    Inps.head(inputs.size()) = inputs;       
    Inps(inputs.size()) = 1.0;           
}

/**
 * Getter for the weights of one layer
 */
/// Implements getWeights.
/// @throw std::out_of_range If @p idx is not a valid layer index.
Eigen::MatrixXd MLP::getWeights(size_t idx) const {
    if (idx >= layers_.size())
      throw std::out_of_range("Layer index out of range in getWeights");
    return layers_[idx].getWeights();
}

/**
 * Setter for the weights of one layer
 */
/// Implements setWeights.
/// @throw std::out_of_range If @p idx is not a valid layer index.
void MLP::setWeights(size_t idx, const Eigen::MatrixXd& W) {
    if (idx >= layers_.size())
      throw std::out_of_range("Layer index out of range in setWeights");
    layers_[idx].setWeights(W);
}

/**
 * Getter for the flattened weights vector of the whole MLP
 */
/// Implements getWeightsVectorMlp. Call weightsToVectorMlp first to (re)fill it.
Eigen::VectorXd MLP::getWeightsVectorMlp(){
    return weightsVectorMlp;
}

/**
 * Merge weight vectors of all layers into one flat vector
 */
/// Implements weightsToVectorMlp. Flattens each layer's weight matrix via
/// Layer::weightsToVector and concatenates them layer by layer (layer 0 first)
/// into weightsVectorMlp.
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
/// Implements saveWeightsCsv. Creates parent directories when needed. Each layer
/// block is introduced by a "#layer,<idx>,<rows>,<cols>" header line followed by
/// the matrix rows as comma-separated values with a precision of 12 digits.
/// @return true on success, false (with a message on std::cerr) otherwise.
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
/// Implements saveWeightsBinary. Layout: size_t layer count, then for each layer
/// two uint64_t dimension values followed by the doubles in row-major order.
/// @return true on success, false (with a message on std::cerr) otherwise.
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
 * Save the flattened weights vector as readable CSV text
 */
/// Implements saveWeightsVectorCsv. Writes weightsVectorMlp one value per line
/// (a column vector); it does NOT refresh weightsVectorMlp, so
/// weightsToVectorMlp should be called beforehand.
/// @return true on success, false (with a message on std::cerr) otherwise.
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
 * Save the flattened weights vector in compact binary
 */
/// Implements saveWeightsVectorBinary. Layout: int64_t length followed by the
/// raw doubles of weightsVectorMlp.
/// @return true on success, false (with a message on std::cerr) otherwise.
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
 * Save the flattened weights vector inside one shared file for an ensemble run
 */
/// Implements appendWeightsVectorCsv. Collects the weights of several ensemble
/// members in a single CSV file where each column is one member:
/// - on the first run (@p isFirstRun == true) the file is (re)created and the
///   vector is written as the first column;
/// - on subsequent runs the existing file is parsed, one column is appended to
///   each row, and the file is rewritten.
/// @return true on success, false (with a message on std::cerr) otherwise.
/// @note Fails if the row count of the existing file does not match
///       weightsVectorMlp.size().
bool MLP::appendWeightsVectorCsv(const std::string &path, bool isFirstRun) const {
    namespace fs = std::filesystem;
    if (path.empty()) {
        std::cerr << "[MLP::appendWeightsVectorCsv] Path is empty\n";
        return false;
    }
    try {
        fs::path p(path);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
    } catch (const std::exception &e) {
        std::cerr << "[MLP::appendWeightsVectorCsv] Cannot create parent directories: "
                  << e.what() << "\n";
        return false;
    }

    // On first run: create/truncate and write column 1
    if (isFirstRun) {
        std::ofstream ofs(path, std::ios::out | std::ios::trunc);
        if (!ofs.is_open()) {
            std::cerr << "[MLP::appendWeightsVectorCsv] Cannot open file: " << path << "\n";
            return false;
        }
        ofs << std::setprecision(12);
        for (int i = 0; i < weightsVectorMlp.size(); ++i) {
            ofs << weightsVectorMlp[i] << "\n";
        }
        ofs.close();
        return true;
    }

    // For subsequent runs: read, append new column, overwrite
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "[MLP::appendWeightsVectorCsv] Cannot open file for appending: " << path << "\n";
        return false;
    }

    std::vector<std::vector<double>> matrix;
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double val;
        while (ss >> val) {
            row.push_back(val);
            if (ss.peek() == ',') ss.ignore();
        }
        matrix.push_back(row);
    }
    ifs.close();

    if ((int)matrix.size() != weightsVectorMlp.size()) {
        std::cerr << "[MLP::appendWeightsVectorCsv] Mismatch in vector size vs file rows\n";
        return false;
    }

    for (int i = 0; i < weightsVectorMlp.size(); ++i) {
        matrix[i].push_back(weightsVectorMlp[i]);
    }

    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "[MLP::appendWeightsVectorCsv] Cannot reopen file: " << path << "\n";
        return false;
    }
    ofs << std::setprecision(12);
    for (const auto &row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            ofs << row[j];
            if (j + 1 < row.size()) ofs << ",";
        }
        ofs << "\n";
    }
    ofs.close();
    return true;
}

/**
 * Load weights from CSV text (per-layer blocks) and rebuild the MLP
 */
/// Implements loadWeightsCsv. Parses the "#layer,<idx>,<rows>,<cols>" blocks
/// written by saveWeightsCsv, then re-creates all layers with
/// weight_init_type::RANDOM immediately overwritten by the loaded weights.
/// The architecture (nNeurons) is derived from the stored matrix dimensions;
/// @p confPath (an .ini config) provides the activation function name,
/// defaulting to LINEAR when unavailable.
/// @return true on success, false (with a message on std::cerr) otherwise.
bool MLP::loadWeightsCsv(const std::string &wPath, const std::string &confPath) {
    if (wPath.empty()) {
        std::cerr << "[MLP::loadWeightsCsv] wPath is empty\n";
        return false;
    }

    std::ifstream ifs(wPath);
    if (!ifs.is_open()) {
        int e = errno;
        std::cerr << "[MLP::loadWeightsCsv] Cannot open file: " << wPath
                  << " errno=" << e << " (" << std::strerror(e) << ")\n";
        return false;
    }

    // Parse CSV structure
    std::string line;
    struct LayerInfo {
        size_t idx;
        size_t rows;
        size_t cols;
        Eigen::MatrixXd W;
    };
    std::vector<LayerInfo> infos;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (line.rfind("#layer", 0) == 0) {
                std::stringstream ss(line);
                std::string token;
                std::getline(ss, token, ','); // "#layer"
                size_t idx=0, rows=0, cols=0;
                if (!(std::getline(ss, token, ',') && (idx = std::stoul(token), true))) continue;
                if (!(std::getline(ss, token, ',') && (rows = std::stoul(token), true))) continue;
                if (!(std::getline(ss, token, ',') && (cols = std::stoul(token), true))) continue;

                LayerInfo li;
                li.idx = idx;
                li.rows = rows;
                li.cols = cols;
                li.W.resize(rows, cols);

                for (size_t r = 0; r < rows; ++r) {
                    if (!std::getline(ifs, line)) {
                        std::cerr << "[MLP::loadWeightsCsv] Unexpected EOF at layer " << idx << "\n";
                        return false;
                    }
                    std::stringstream ls(line);
                    std::string val;
                    for (size_t c = 0; c < cols; ++c) {
                        if (!std::getline(ls, val, ',')) {
                            std::cerr << "[MLP::loadWeightsCsv] Not enough columns at layer "
                                      << idx << " row " << r << "\n";
                            return false;
                        }
                        li.W(r, c) = std::stod(val);
                    }
                }
                infos.push_back(std::move(li));
            }
            continue;
        }
    }

    if (infos.empty()) {
        std::cerr << "[MLP::loadWeightsCsv] No layer info found in file\n";
        return false;
    }

    // Determine activation function
    std::string loadAct = "LINEAR"; // fallback
    if (!confPath.empty()) {
        auto kv = parseIniToMap(confPath);
        std::string sact;
        std::string key = "activation";
        for (char &c : key)
            c = static_cast<char>(std::tolower((unsigned char)c));

        auto it = kv.find(key);
        if (it != kv.end()) sact = it->second;
        if (!sact.empty()) loadAct = trimStr(sact);
    } else {
        std::cerr << "[MLP::loadWeightsCsv] confPath is empty, using default activation LINEAR\n";
    }

    // Build layers from weights
    layers_.clear();
    layers_.resize(infos.size());
    nNeurons.clear();

    for (const auto &li : infos) {
        if (li.idx >= layers_.size()) {
            std::cerr << "[MLP::loadWeightsCsv] Layer index " << li.idx
                      << " out of range (have " << layers_.size() << ")\n";
            return false;
        }

        layers_[li.idx].initLayer(
            li.cols,
            li.rows,
            weight_init_type::RANDOM,
            strToActivation(loadAct),
            0
        );
        layers_[li.idx].setWeights(li.W);
        nNeurons.push_back(li.rows);
    }

    numLayers = nNeurons.size();
    return true;
}

/**
 * Load weights in compact binary and rebuild the MLP
 */
/// Implements loadWeightsBinary. Reads the layout written by saveWeightsBinary.
/// @p confPath (an .ini config) is required and provides the activation function
/// name for the rebuilt layers.
/// @note Unlike loadWeightsCsv this does not rebuild nNeurons/numLayers.
/// @todo check semantics: comment claims nNeurons is not rebuilt here; verify
///       this is intended (the layer count L from the file drives layers_ only).
/// @return true on success, false (with a message on std::cerr) otherwise.
bool MLP::loadWeightsBinary(const std::string &wPath, const std::string &confPath) {
    try {
        std::ifstream ifs(wPath, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "[MLP::loadWeightsBinary] Cannot open: " << wPath << "\n";
            return false;
        }
        if (confPath.empty()) {
            std::cerr << "[MLP::loadWeightsCsv] confPath is empty\n";
            return false;
        }

        size_t L = 0;
        ifs.read(reinterpret_cast<char*>(&L), sizeof(L));
        if (!ifs) {
            std::cerr << "[MLP::loadWeightsBinary] Failed to read layer count\n";
            return false;
        }

        layers_.clear();
        layers_.resize(L);

        auto kv = parseIniToMap(confPath);
        std::string sact, loadAct;
        {
            std::string key = "activation";
            for (char &c : key)
                c = static_cast<char>(std::tolower((unsigned char)c));

            auto it = kv.find(key);
            if (it != kv.end())
                sact = it->second;
        }
        if (!sact.empty()) loadAct = trimStr(sact);

        for (size_t i = 0; i < L; ++i) {
            uint64_t rows = 0, cols = 0;
            ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            if (!ifs) {
                std::cerr << "[MLP::loadWeightsBinary] Failed to read dims for layer " << i << "\n";
                return false;
            }

            Eigen::MatrixXd W(rows, cols);
            for (uint64_t r = 0; r < rows; ++r) {
                for (uint64_t c = 0; c < cols; ++c) {
                    double v = 0.0;
                    ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
                    if (!ifs) {
                        std::cerr << "[MLP::loadWeightsBinary] Failed to read weight at "
                                  << "layer " << i << " pos (" << r << "," << c << ")\n";
                        return false;
                    }
                    W(r, c) = v;
                }
            }

            layers_[i].initLayer(
                /*numInputs=*/ static_cast<size_t>(cols),
                /*numNeurons=*/ static_cast<size_t>(rows),
                /*wInitType=*/ weight_init_type::RANDOM,
                /*func=*/ strToActivation(loadAct), 
                /*seed=*/ 0
            );
            layers_[i].setWeights(W);
        }

        return true;
    } catch (const std::exception &ex) {
        std::cerr << "[MLP::loadWeightsBinary] Exception: " << ex.what() << "\n";
        return false;
    }
}

/**
 * Load the flattened weights vector from CSV text and rebuild the MLP
 */
/// Implements loadWeightsVectorCsv. The architecture and activation function are
/// taken from the .ini config at @p confPath; the flat vector @p wPath is then
/// split into per-layer weight matrices by walking the architecture backwards
/// (layers are stored in order 0..L-1, each matrix is rows=nNeurons[l],
/// cols=nNeurons[l-1]+1 including the bias column, row-major). Layer 0 takes the
/// remainder of the vector, so its column count (real inputs + 1 bias) is
/// inferred from the data.
/// @throw std::runtime_error If the vector is too short for the architecture or
///        its remainder is not divisible by nNeurons[0].
bool MLP::loadWeightsVectorCsv(const std::string &wPath, const std::string &confPath) {
    if (wPath.empty()) {
        std::cerr << "[MLP::loadWeightsVectorCsv] wPath is empty\n";
        return false;
    }
    if (confPath.empty()) {
        std::cerr << "[MLP::loadWeightsVectorCsv] confPath is empty\n";
        return false;
    }

    std::ifstream ifs(wPath);
    if (!ifs.is_open()) {
        int e = errno;
        std::cerr << "[MLP::loadWeightsVectorCsv] Cannot open file: " << wPath
                  << " errno=" << e << " (" << std::strerror(e) << ")\n";
        return false;
    }

    auto kv = parseIniToMap(confPath);
    auto get = [&](const std::string &k)->std::string {
        std::string kl = k;
        for (char &c : kl) c = static_cast<char>(std::tolower((unsigned char)c));
        auto it = kv.find(kl);
        if (it == kv.end()) return std::string{};
        return it->second;
    };

    std::vector<unsigned> loadArch;
    std::string loadAct;
    std::string sarch = get("architecture");
    if (!sarch.empty()) loadArch = parseUnsignedList(sarch);

    std::string sact = get("activation");
    if (!sact.empty()) loadAct = trimStr(sact);

    nNeurons = loadArch;
    numLayers = nNeurons.size();
    layers_.clear();
    layers_.resize(numLayers);

    std::vector<double> values;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        values.push_back(std::stod(line));
    }
    ifs.close();

    weightsVectorMlp.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        weightsVectorMlp(i) = values[i];
    }

    int pos = weightsVectorMlp.size();

    for (int i = numLayers - 1; i > 0; --i) {
        int rows = nNeurons[i];
        int cols = nNeurons[i - 1] + 1;
        int size = rows * cols;

        pos -= size;
        if (pos < 0) throw std::runtime_error("Vector too short for given architecture");

        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            W(weightsVectorMlp.data() + pos, rows, cols);

        layers_[i].initLayer(
            /*numInputs=*/ static_cast<size_t>(cols),
            /*numNeurons=*/ static_cast<size_t>(rows),
            /*wInitType=*/ weight_init_type::RANDOM,
            /*func=*/ strToActivation(loadAct), 
            /*seed=*/ 0
        );
        layers_[i].setWeights(W);
    }

    int remaining = pos;
    if (remaining % nNeurons[0] != 0)
        throw std::runtime_error("Cannot deduce input size from vector");

    int cols0 = remaining / nNeurons[0];
    int rows0 = nNeurons[0];

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        W0(weightsVectorMlp.data(), rows0, cols0);

    layers_[0].initLayer(
        /*numInputs=*/ static_cast<size_t>(cols0),
        /*numNeurons=*/ static_cast<size_t>(rows0),
        /*wInitType=*/ weight_init_type::RANDOM,
        /*func=*/ strToActivation(loadAct),
        /*seed=*/ 0
    );
    layers_[0].setWeights(W0);

    return true;
}

/**
 * Getter for output
 */
/// Implements getOutput. Returns the last output produced by calcOneOutput.
Eigen::VectorXd& MLP::getOutput(){
    return output;
}

/**
 * Validate the size of the inputs compared to nNeurons[0]
 */
/// Implements validateInputSize. Fails (returns false, message on std::cerr) when
/// no architecture is set, when Inps is empty, or when Inps.size() - 1 (inputs
/// excluding the bias slot) does not equal nNeurons[0].
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
 * Initialize all layers (with freshly randomized weights) and run a forward pass
 */
/// Implements initMLP. (Re)creates every Layer according to nNeurons, activFuncs
/// and wInitTypes, initializing each layer's weights with @p rngSeed, then
/// propagates @p input layer by layer: layer 0 consumes the raw input, every
/// later layer consumes the previous layer's output.
/// @note Weight matrix of layer l has shape nNeurons[l] x (numInputs + 1); the
///       extra column is the bias, added internally by Layer::setInputs.
/// @throw std::logic_error If architecture/activations/weight-init vectors are
///        empty or of mismatched sizes.
/// @return Output of the last layer (length nNeurons.back()).
Eigen::VectorXd MLP::initMLP(const Eigen::VectorXd& input, int rngSeed) {
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
        /*func=*/ activFuncs[0],
        /*seed=*/ rngSeed
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
            /*func=*/ activFuncs[i],
        /*seed=*/ rngSeed
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
/// Implements runMLP. Same layer-by-layer propagation as initMLP but keeps the
/// current weights. Also refreshes the member Inps (with bias) inside each layer.
/// @throw std::logic_error If called before initMLP (layers_ is empty).
/// @return Output of the last layer (length nNeurons.back()).
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
/// Implements compareInitAndRun. Works on a local copy so the object's state is
/// left untouched: initMLP re-initializes weights with @p rngSeed, runMLP then
/// reuses them; outputs compared elementwise with tolerance @p tol.
bool MLP::compareInitAndRun(const Eigen::VectorXd& input, double tol, int rngSeed) const {
    // Make a local copy of *this* so we can init on the copy
    MLP tmp = *this;
    Eigen::VectorXd outInit = tmp.initMLP(input, rngSeed);
    Eigen::VectorXd outRun = tmp.runMLP(input);

    return outInit.isApprox(outRun, tol);
}

/**
 * Test that 'runMLP' produces the same results over times
 */
/// Implements testRepeatable. Works on a local copy: after one initMLP (baseline
/// output), runMLP is executed @p repeats times; each output must match the
/// baseline within @p tol (runMLP must be deterministic and side-effect free
/// with respect to the weights).
bool MLP::testRepeatable(const Eigen::VectorXd& input, int repeats, double tol, int rngSeed) const {
    // Initialize once to get the baseline output
    MLP tmp = *this;
    Eigen::VectorXd baseline = tmp.initMLP(input, rngSeed);

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
/// Implements runAndBP — one online (single-pattern) SGD step:
/// 1. forward pass via calcOneOutput,
/// 2. output-layer deltas from the configured loss gradient
///    (Layer::setDeltasFromLossGradient), then its online gradient,
/// 3. deltas chain-ruled backwards: delta_l = (W_{l+1}^T delta_{l+1}) odot f'(z_l),
///    each hidden layer's online gradient computed on the way,
/// 4. plain weight updates W -= learningRate * grad in every layer.
/// @throw std::logic_error If called before initMLP.
void MLP::runAndBP(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut, double learningRate) {
    if (layers_.empty())
        throw std::logic_error("runMLP called before initMLP");

    calcOneOutput(input);

    // Output layer BP
    layers_[layers_.size()-1].setDeltasFromLossGradient(
        obsOut, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
    );
    layers_[layers_.size()-1].calculateOnlineGradient();

    // Remaining layers BP
    if(layers_.size() > 1){
        for(int i = layers_.size() - 2; i >= 0; --i){
            layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
            layers_[i].calculateOnlineGradient();
        }
    }
    for(size_t i = 0; i < getNumLayers(); i++){
        layers_[i].updateWeights(learningRate);
    }
}

/**
 * Forward pass and update weights with Adam algorithm (one input)
 */
/// Implements runAndBPadam. Same forward + backpropagation order as runAndBP,
/// but the weight update uses Adam (Layer::updateAdam) with fixed hyperparameters
/// beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8.
/// @param iterationNum 1-based step counter passed to Adam for the bias
///        correction of the first and second moment estimates.
void MLP::runAndBPadam(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut, double learningRate, int iterationNum) {
    if (layers_.empty())
        throw std::logic_error("runMLP called before initMLP");

    calcOneOutput(input);

    // Output layer BP
    layers_[layers_.size()-1].setDeltasFromLossGradient(
        obsOut, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
    );
    layers_[layers_.size()-1].calculateOnlineGradient();

    // Remaining layers BP
    if(layers_.size() > 1){
        for(int i = layers_.size() - 2; i >= 0; --i){
            layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
            layers_[i].calculateOnlineGradient();
        }
    }
    for(size_t i = 0; i < getNumLayers(); i++){
        layers_[i].updateAdam(learningRate,iterationNum,0.9, 0.99, 1e-8);
    }
}

/// Implements runAndCalculateBatchGradient. Same forward + backpropagation order
/// as runAndBP, but gradients are only accumulated per layer
/// (Layer::calculateBatchGradient sums dE/dW over the batch) — no weight update
/// happens here; call updateWeights / updateWeightsAdam afterwards to apply and
/// clear the accumulated gradient.
/// @throw std::logic_error If called before initMLP.
void MLP::runAndCalculateBatchGradient(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut) {
    if (layers_.empty())
        throw std::logic_error("runMLP called before initMLP");

    calcOneOutput(input);

    // Output layer BP
    layers_[layers_.size()-1].setDeltasFromLossGradient(
        obsOut, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
    );
    layers_[layers_.size()-1].calculateBatchGradient();

    // Remaining layers BP
    if(layers_.size() > 1){
        for(int i = layers_.size() - 2; i >= 0; --i){
            layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
            layers_[i].calculateBatchGradient();
        }
    }
}

/// Implements updateWeights. Applies one plain SGD step per layer from the
/// accumulated gradient (W -= learningRate * grad) and then zeroes the gradient
/// so the next batch accumulates from scratch.
/// @throw std::logic_error If called before initMLP.
void MLP::updateWeights(double learningRate){
    if (layers_.empty())
        throw std::logic_error("updateWeights called before initMLP");

    for(size_t i = 0; i < getNumLayers(); i++){
        layers_[i].updateWeights(learningRate);
        layers_[i].setGradient(layers_[i].getGradient().setZero());
    }
}

/// Implements updateWeightsAdam. Applies one Adam step per layer (Layer::updateAdam
/// with beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8) from the accumulated gradient
/// and then zeroes the gradient so the next batch accumulates from scratch.
/// @param iterationNum 1-based step counter for Adam's bias correction.
/// @throw std::logic_error If called before initMLP.
void MLP::updateWeightsAdam(double learningRate, int iterationNum){
    if (layers_.empty())
        throw std::logic_error("updateWeights called before initMLP");

    for(size_t i = 0; i < getNumLayers(); i++){
        layers_[i].updateAdam(learningRate,iterationNum,0.9, 0.99, 1e-8);
        layers_[i].setGradient(layers_[i].getGradient().setZero());
    }
}

/// Implements updateWeightsAdamNoClear. Applies one Adam step per layer but
/// intentionally leaves the accumulated gradients in place so further patterns
/// keep accumulating (used by the LSTM horizon training).
/// @param iterationNum 1-based step counter for Adam's bias correction.
/// @throw std::logic_error If called before initMLP.
void MLP::updateWeightsAdamNoClear(double learningRate, int iterationNum){
    if (layers_.empty())
        throw std::logic_error("updateWeightsAdamNoClear called before initMLP");

    for(size_t i = 0; i < getNumLayers(); i++){
        layers_[i].updateAdam(learningRate, iterationNum, 0.9, 0.99, 1e-8);
        // Do NOT clear gradients - allow accumulation
    }
}

/// Implements resetGradientsForLSTM. Resets the Adam moment state (first/second
/// moment accumulators) of every layer; used when the MLP is trained as part of
/// a coupled LSTM->MLP model so a fresh Adam trajectory starts for each window.
void MLP::resetGradientsForLSTM(){
    if (layers_.empty())
        return;

    for(size_t i = 0; i < getNumLayers(); i++){
        // Reset Adam state variables
        layers_[i].resetAdamState();
    }
}

/**
 * Online backpropagation - separete inp out matrices
 */
/// Implements onlineBP. Stochastic gradient descent: each epoch processes all
/// patterns (rows of @p X = inputs, rows of @p Y = target outputs) one by one —
/// forward pass, backpropagation, immediate weight update per pattern.
/// The epoch error is the mean over patterns of the squared norm of the
/// output-layer delta vector; training stops early when it drops below
/// @p maxError. Iteration count, final error, wall-clock time and the convergence
/// flag are stored in `result`.
/// @throw std::logic_error If called before initMLP.
/// @throw std::invalid_argument On empty/mismatched matrices or non-positive
///        maxIterations / negative maxError / learningRate outside (0, 1].
/// @throw std::runtime_error If X.cols()/Y.cols() do not match the initialized
///        input/output sizes.
void MLP::onlineBP(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    if (layers_.empty())
        throw std::logic_error("onlineBP called before initMLP");

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineBP");

    if (X.rows() != Y.rows())
        throw std::invalid_argument("matrices have different number of rows");

    if (maxIterations <= 0 || maxError < 0.0)
        throw std::invalid_argument("maxIterations and maxError must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = X.rows();                // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != X.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Y.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIterations + 1; iter++){
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = X.row(pat);
            Eigen::VectorXd currentObs = Y.row(pat);
            
            calcOneOutput(currentInp);

            // Output layer BP - use custom objective function for deltas
            layers_[lastLayerIndex].setDeltasFromLossGradient(
                currentObs, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
            );
            layers_[lastLayerIndex].calculateOnlineGradient();

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradient();
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
            for(size_t i = 0; i < getNumLayers(); i++){
                layers_[i].updateWeights(learningRate);
            }
        }
        Error = Error / numOfPatterns;
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = iter;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Store results
    result.iterations = maxIterations;
    result.finalLoss = Error;
    result.time = duration.count();
    result.converged = false;
    }
}

/**
 * Online backpropagation using Adam algorithm - separete inp out matrices
 */
/// Implements onlineAdam. Same per-pattern online training loop as onlineBP, but
/// weights are updated with Adam (beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8);
/// the epoch counter is used as Adam's step for the bias correction.
/// Early stopping and the `result` bookkeeping follow onlineBP.
/// @throw std::logic_error If called before initMLP.
/// @throw std::invalid_argument / std::runtime_error Same checks as onlineBP.
void MLP::onlineAdam(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    if (layers_.empty())
        throw std::logic_error("onlineAdam called before initMLP");

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineAdam");
    
    if (X.rows() != Y.rows())
        throw std::invalid_argument("matrices have different number of rows");
    
    if (maxIterations <= 0 || maxError < 0.0)
        throw std::invalid_argument("maxIterations and maxError must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = X.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != X.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Y.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIterations + 1; iter++){
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = X.row(pat);
            Eigen::VectorXd currentObs = Y.row(pat);

            calcOneOutput(currentInp);

            // Output layer BP - use custom objective function for deltas
            layers_[lastLayerIndex].setDeltasFromLossGradient(
                currentObs, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
            );
            layers_[lastLayerIndex].calculateOnlineGradient();

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradient();
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
            for(size_t i = 0; i < getNumLayers(); i++){
                layers_[i].updateAdam(learningRate,iter,0.9, 0.99, 1e-8);
            }
        }
        Error = Error / numOfPatterns;
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = iter;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Store results
    result.iterations = maxIterations;
    result.finalLoss = Error;
    result.time = duration.count();
    result.converged = false;
    }
}

/**
 * Batch backpropagation using Adam algorithm  - separete inp out matrices
 */
/// Implements batchAdam. Each epoch splits the patterns into mini-batches of
/// @p batchSize (the last batch may be smaller); for every pattern in a batch
/// the forward pass and backpropagation only accumulate gradients
/// (calculateBatchGradient), and one Adam update is applied per batch, after
/// which the gradients are zeroed. Adam's bias-correction step is the epoch
/// counter (not the batch counter).
/// Early stopping and the `result` bookkeeping follow onlineBP.
/// @throw std::logic_error If called before initMLP.
/// @throw std::invalid_argument / std::runtime_error Same checks as onlineBP,
///        plus batchSize must be positive.
void MLP::batchAdam(int maxIterations, double maxError, int batchSize, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    if (layers_.empty())
        throw std::logic_error("batchAdam called before initMLP");

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to batchAdam");
            
    if (X.rows() != Y.rows())
        throw std::invalid_argument("matrices have different number of rows");

    if (maxIterations <= 0 || batchSize <= 0|| maxError < 0.0)
        throw std::invalid_argument("maxIterations, batchSize and maxError must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = X.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != X.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Y.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIterations + 1; iter++){
        Error = 0.0;
        for(int batch = 0; batch < (numOfPatterns + batchSize - 1)/batchSize; batch++){
            int start = batch * batchSize;
            int end   = std::min(start + batchSize,numOfPatterns);
            for (int pat = start; pat < end; pat++){

                Eigen::VectorXd currentInp = X.row(pat);
                Eigen::VectorXd currentObs = Y.row(pat);
                
                calcOneOutput(currentInp);

                // Output layer gradient
                layers_[lastLayerIndex].setDeltasFromLossGradient(
                    currentObs, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
                );
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
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = iter;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Store results
    result.iterations = maxIterations;
    result.finalLoss = Error;
    result.time = duration.count();
    result.converged = false;
    }
}

/**
 * Batch backpropagation - separete inp out matrices
 */
/// Implements batchBP. Mini-batch gradient descent: like batchAdam but the
/// batched gradients are applied with a plain SGD step (updateWeights) instead
/// of Adam. Early stopping and the `result` bookkeeping follow onlineBP.
/// @throw std::logic_error If called before initMLP.
/// @throw std::invalid_argument / std::runtime_error Same checks as onlineBP,
///        plus batchSize must be positive.
void MLP::batchBP(int maxIterations, double maxError, int batchSize, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    if (layers_.empty())
        throw std::logic_error("batchBP called before initMLP");

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to batchBP");
            
    if (X.rows() != Y.rows())
        throw std::invalid_argument("matrices have different number of rows");

    if (maxIterations <= 0 || batchSize <= 0|| maxError < 0.0)
        throw std::invalid_argument("maxIterations, batchSize and maxError must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = X.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != X.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Y.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIterations + 1; iter++){
        Error = 0.0;
        for(int batch = 0; batch < (numOfPatterns + batchSize - 1)/batchSize; batch++){
            int start = batch * batchSize;
            int end   = std::min(start + batchSize,numOfPatterns);
            for (int pat = start; pat < end; pat++){

                Eigen::VectorXd currentInp = X.row(pat);
                Eigen::VectorXd currentObs = Y.row(pat);
                
                calcOneOutput(currentInp);

                // Output layer gradient
                layers_[lastLayerIndex].setDeltasFromLossGradient(
                    currentObs, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
                );
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
                    layers_[i].updateWeights(learningRate);
                    layers_[i].setGradient(layers_[i].getGradient().setZero());
                }
            }
            Error = Error / numOfPatterns;
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = iter;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Store results
    result.iterations = maxIterations;
    result.finalLoss = Error;
    result.time = duration.count();
    result.converged = false;
    }
}

/**
 * Online backpropagation with high weights penalization (L2 regularization)
 */
/// Implements onlinePenalizeBP. Online (per-pattern) gradient descent like
/// onlineBP, but gradients are computed by calculateOnlineGradientPenalize,
/// which adds the L2 penalty term 2*lambda*W to dE/dW to discourage large
/// weights (bias column excluded inside Layer).
/// @param lambda Regularization strength, must be in (0, 1].
/// @throw std::logic_error If called before initMLP.
/// @throw std::invalid_argument / std::runtime_error Same checks as onlineBP,
///        plus lambda range check.
void MLP::onlinePenalizeBP(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, double lambda) {
    if (layers_.empty())
        throw std::logic_error("onlineBP called before initMLP");

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineBP");

    if (X.rows() != Y.rows())
        throw std::invalid_argument("matrices have different number of rows");

    if (maxIterations <= 0 || maxError < 0.0)
        throw std::invalid_argument("maxIterations and maxError must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (lambda <= 0.0 || lambda > 1.0)
        throw std::invalid_argument("lambda must be between 0 and 1");

    int numOfPatterns = X.rows();                // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != X.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Y.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIterations + 1; iter++){
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = X.row(pat);
            Eigen::VectorXd currentObs = Y.row(pat);

            calcOneOutput(currentInp);

            // Output layer BP - use custom objective function for deltas
            layers_[lastLayerIndex].setDeltasFromLossGradient(
                currentObs, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
            );
            layers_[lastLayerIndex].calculateOnlineGradientPenalize(lambda);

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradientPenalize(lambda);
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
            for(size_t i = 0; i < getNumLayers(); i++){
                layers_[i].updateWeights(learningRate);
            }
        }
        Error = Error / numOfPatterns;
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = iter;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Store results
    result.iterations = maxIterations;
    result.finalLoss = Error;
    result.time = duration.count();
    result.converged = false;
    }
}

/**
 * Online backpropagation with momentum
 */
/// Implements onlineMomentumBP. Online (per-pattern) gradient descent like
/// onlineBP, but weight updates carry momentum: updateWeightsMomentum applies
/// W -= learningRate * grad + moment * (previous update), which smooths
/// oscillations and accelerates consistent descent directions.
/// @param moment Momentum coefficient, must be in (0, 1].
/// @throw std::logic_error If called before initMLP.
/// @throw std::invalid_argument / std::runtime_error Same checks as onlineBP,
///        plus moment range check.
void MLP::onlineMomentumBP(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, double moment) {
    if (layers_.empty())
        throw std::logic_error("onlineBP called before initMLP");

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineBP");

    if (X.rows() != Y.rows())
        throw std::invalid_argument("matrices have different number of rows");

    if (maxIterations <= 0 || maxError < 0.0)
        throw std::invalid_argument("maxIterations and maxError must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");
        
    if (moment <= 0.0 || moment > 1.0)
        throw std::invalid_argument("momentum must be between 0 and 1");

    int numOfPatterns = X.rows();                // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != X.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Y.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIterations + 1; iter++){
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = X.row(pat);
            Eigen::VectorXd currentObs = Y.row(pat);
            
            calcOneOutput(currentInp);

            // Output layer BP - use custom objective function for deltas
            layers_[lastLayerIndex].setDeltasFromLossGradient(
                currentObs, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
            );
            layers_[lastLayerIndex].calculateOnlineGradient();

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradient();
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
            for(size_t i = 0; i < getNumLayers(); i++){
                layers_[i].updateWeightsMomentum(learningRate, moment);
            }
        }
        Error = Error / numOfPatterns;
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = iter;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Store results
    result.iterations = maxIterations;
    result.finalLoss = Error;
    result.time = duration.count();
    result.converged = false;
    }
}

/// Implements onlineAdamEpochVal — online (per-pattern) Adam training with
/// periodic evaluation of performance metrics on the training and validation
/// sets. Every epoch updates weights per pattern (same loop as onlineAdam),
/// then the seven metrics MSE, RMSE, PI, NS, KGE, PBIAS, RSR are computed:
/// - entry 0..6 of the returned vector are metric matrices for the training
///   set, entry 7..13 for the validation set;
/// - with metricsAfterXEpochs == 1 every epoch produces one row, otherwise one
///   row per metricsAfterXEpochs epochs (the second-to-last epoch is also
///   recorded when the interval does not divide maxIterations).
/// Early stopping below maxError and `result` bookkeeping follow onlineAdam.
/// @param Xtrain Patterns x input-size training inputs.
/// @param Ytrain Patterns x output-size training targets.
/// @param Xval   Validation inputs.
/// @param Yval   Validation targets.
/// @param metricsAfterXEpochs Interval (in epochs) between metric evaluations.
/// @return Vector of 14 matrices (7 train + 7 validation metrics); each matrix
///         has one row per evaluation and one column per output variable.
/// @throw std::logic_error / std::invalid_argument / std::runtime_error Same
///        checks as onlineAdam, plus metricsAfterXEpochs must be in
///        [1, maxIterations].
std::vector<Eigen::MatrixXd> MLP::onlineAdamEpochVal(
    const Eigen::MatrixXd &Xtrain,
    const Eigen::MatrixXd &Ytrain,
    const Eigen::MatrixXd &Xval,
    const Eigen::MatrixXd &Yval,
    int maxIterations,
    double maxError,
    double learningRate,
    int metricsAfterXEpochs,
    const EpochMetricContext *epochCtx)
{
    if (layers_.empty())
        throw std::logic_error("onlineAdamEpochVal called before initMLP");

    if (Xtrain.rows() == 0 || Ytrain.rows() == 0 || Xval.rows() == 0 || Yval.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineAdamEpochVal");
            
    if (Xtrain.rows() != Ytrain.rows())
        throw std::invalid_argument("train matrices have different number of rows");

    if (Xval.rows() != Yval.rows())
        throw std::invalid_argument("validation matrices have different number of rows");

    if (maxIterations <= 0 || metricsAfterXEpochs < 0)
        throw std::invalid_argument("maxIterations must be positive and metricsAfterXEpochs non-negative (0 = epoch-metric recording disabled)");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("metricsAfterXEpochs can't be bigger than maxIterations");

    std::vector<Eigen::MatrixXd> resultMetrics =
        Metrics::initEpochMetrics(maxIterations, metricsAfterXEpochs, static_cast<int>(Ytrain.cols()));

    int numOfPatterns = Xtrain.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != Xtrain.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Ytrain.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    auto start = high_resolution_clock::now();
    double Error;
    for(int epoch = 1; epoch <= maxIterations; epoch++){
        // Run one epoch of online Adam: per-pattern forward + backprop + Adam update
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = Xtrain.row(pat);
            Eigen::VectorXd currentObs = Ytrain.row(pat);

            calcOneOutput(currentInp);

            // Output layer BP - use custom objective function for deltas
            layers_[lastLayerIndex].setDeltasFromLossGradient(
                currentObs, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
            );
            layers_[lastLayerIndex].calculateOnlineGradient();

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradient();
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
            for(size_t i = 0; i < getNumLayers(); i++){
                layers_[i].updateAdam(learningRate,epoch,0.9, 0.99, 1e-8);
            }
        }
        Error = Error / numOfPatterns;
        // Epoch metrics: recorded every metricsAfterXEpochs epochs and at the final epoch;
        // in original units with chronological (unshuffled) calibration order when epochCtx is given
        if (Metrics::isEpochMetricRecord(epoch, metricsAfterXEpochs, maxIterations)) {
            size_t row = Metrics::epochMetricRowIndex(epoch, metricsAfterXEpochs, maxIterations);
            const Eigen::MatrixXd *YcalTrue = &Ytrain;
            const Eigen::MatrixXd *YvalTrue = &Yval;
            if (epochCtx != nullptr) {
                if (epochCtx->YtrueCalOrig.size() > 0) YcalTrue = &epochCtx->YtrueCalOrig;
                if (epochCtx->YtrueValOrig.size() > 0) YvalTrue = &epochCtx->YtrueValOrig;
            }

            calculateOutputs(Xtrain);
            Eigen::MatrixXd YpredCal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepCalPreds) YpredCal = epochCtx->prepCalPreds(YpredCal);

            calculateOutputs(Xval);
            Eigen::MatrixXd YpredVal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepValPreds) YpredVal = epochCtx->prepValPreds(YpredVal);

            Metrics::fillEpochMetricRow(resultMetrics, row, *YcalTrue, YpredCal, *YvalTrue, YpredVal);
        }
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = epoch;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(stop - start);

        result.converged = false;
        result.finalLoss = Error;
        result.iterations = maxIterations;
        result.time = duration.count();
    }
    return resultMetrics;  
}

/// Implements onlineBpEpochVal — online (per-pattern) SGD training with
/// periodic train/validation metric evaluation. Each epoch delegates the weight
/// updates to onlineBP (one iteration per epoch) and stops early when it reports
/// convergence. Metric matrices follow the same 14-entry layout as
/// onlineAdamEpochVal (7 train + 7 validation metrics).
/// @return Vector of 14 metric matrices.
/// @throw std::logic_error / std::invalid_argument / std::runtime_error Same
///        checks as onlineAdamEpochVal.
std::vector<Eigen::MatrixXd> MLP::onlineBpEpochVal(
    const Eigen::MatrixXd &Xtrain,
    const Eigen::MatrixXd &Ytrain,
    const Eigen::MatrixXd &Xval,
    const Eigen::MatrixXd &Yval,
    int maxIterations,
    double maxError,
    double learningRate,
    int metricsAfterXEpochs,
    const EpochMetricContext *epochCtx)
{
    if (layers_.empty())
        throw std::logic_error("onlineBpEpochVal called before initMLP");

    if (Xtrain.rows() == 0 || Ytrain.rows() == 0 || Xval.rows() == 0 || Yval.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineBpEpochVal");
            
    if (Xtrain.rows() != Ytrain.rows())
        throw std::invalid_argument("train matrices have different number of rows");

    if (Xval.rows() != Yval.rows())
        throw std::invalid_argument("validation matrices have different number of rows");

    if (maxIterations <= 0 || metricsAfterXEpochs < 0)
        throw std::invalid_argument("maxIterations must be positive and metricsAfterXEpochs non-negative (0 = epoch-metric recording disabled)");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("metricsAfterXEpochs can't be bigger than maxIterations");

    std::vector<Eigen::MatrixXd> resultMetrics =
        Metrics::initEpochMetrics(maxIterations, metricsAfterXEpochs, static_cast<int>(Ytrain.cols()));

    auto start = high_resolution_clock::now();
    for(int epoch = 0; epoch < maxIterations; epoch++){
        // Run one epoch of online SGD (this also refreshes `result` and may
        // set result.converged)
        try {
            onlineBP(1, maxError, learningRate, Xtrain, Ytrain);
        } catch (const std::exception &ex) {
            std::cerr << "[onlineBP] training failed: " << ex.what() << "\n";
            throw;
        }

        int epoch1 = epoch + 1;  // 1-based epoch for the metric helpers
        // Epoch metrics: recorded every metricsAfterXEpochs epochs and at the final epoch;
        // in original units with chronological (unshuffled) calibration order when epochCtx is given
        if (Metrics::isEpochMetricRecord(epoch1, metricsAfterXEpochs, maxIterations)) {
            size_t row = Metrics::epochMetricRowIndex(epoch1, metricsAfterXEpochs, maxIterations);
            const Eigen::MatrixXd *YcalTrue = &Ytrain;
            const Eigen::MatrixXd *YvalTrue = &Yval;
            if (epochCtx != nullptr) {
                if (epochCtx->YtrueCalOrig.size() > 0) YcalTrue = &epochCtx->YtrueCalOrig;
                if (epochCtx->YtrueValOrig.size() > 0) YvalTrue = &epochCtx->YtrueValOrig;
            }

            calculateOutputs(Xtrain);
            Eigen::MatrixXd YpredCal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepCalPreds) YpredCal = epochCtx->prepCalPreds(YpredCal);

            calculateOutputs(Xval);
            Eigen::MatrixXd YpredVal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepValPreds) YpredVal = epochCtx->prepValPreds(YpredVal);

            Metrics::fillEpochMetricRow(resultMetrics, row, *YcalTrue, YpredCal, *YvalTrue, YpredVal);
        }
        if(result.converged != false){
            result.iterations = epoch;
            break;
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    
    if(result.converged != true){
        result.iterations = maxIterations;
    }
    result.time = duration.count();

    return resultMetrics;
}

/// Implements batchAdamEpochVal — mini-batch Adam training with periodic
/// train/validation metric evaluation. Each epoch splits the training patterns
/// into mini-batches; gradients are accumulated per batch
/// (calculateBatchGradient) and applied with one Adam step per batch, with the
/// epoch counter used as Adam's bias-correction step. Early stopping below
/// maxError and `result` bookkeeping follow onlineAdam; metric matrices follow
/// the same 14-entry layout as onlineAdamEpochVal.
/// @param batchSize Number of patterns per mini-batch (last batch may be smaller).
/// @return Vector of 14 metric matrices.
/// @throw std::logic_error / std::invalid_argument / std::runtime_error Same
///        checks as onlineAdamEpochVal, plus batchSize must be positive.
std::vector<Eigen::MatrixXd> MLP::batchAdamEpochVal(
    const Eigen::MatrixXd &Xtrain,
    const Eigen::MatrixXd &Ytrain,
    const Eigen::MatrixXd &Xval,
    const Eigen::MatrixXd &Yval,
    int batchSize,
    int maxIterations,
    double maxError,
    double learningRate,
    int metricsAfterXEpochs,
    const EpochMetricContext *epochCtx)
{
    if (layers_.empty())
        throw std::logic_error("batchAdamEpochVal called before initMLP");

    if (Xtrain.rows() == 0 || Ytrain.rows() == 0 || Xval.rows() == 0 || Yval.rows() == 0)
        throw std::invalid_argument("Empty training data passed to batchAdamEpochVal");
            
    if (Xtrain.rows() != Ytrain.rows())
        throw std::invalid_argument("train matrices have different number of rows");

    if (Xval.rows() != Yval.rows())
        throw std::invalid_argument("validation matrices have different number of rows");

    if (maxIterations <= 0 || batchSize <= 0 || metricsAfterXEpochs < 0)
        throw std::invalid_argument("maxIterations and batchSize must be positive and metricsAfterXEpochs non-negative (0 = epoch-metric recording disabled)");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("metricsAfterXEpochs can't be bigger than maxIterations");

    std::vector<Eigen::MatrixXd> resultMetrics =
        Metrics::initEpochMetrics(maxIterations, metricsAfterXEpochs, static_cast<int>(Ytrain.cols()));

    int numOfPatterns = Xtrain.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != Xtrain.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Ytrain.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for(int epoch = 1; epoch <= maxIterations; epoch++){
        // Run batch Adam
        Error = 0.0;
        for(int batch = 0; batch < (numOfPatterns + batchSize - 1)/batchSize; batch++){
            int start = batch * batchSize;
            int end   = std::min(start + batchSize,numOfPatterns);
            for (int pat = start; pat < end; pat++){

                Eigen::VectorXd currentInp = Xtrain.row(pat);
                Eigen::VectorXd currentObs = Ytrain.row(pat);
                
                calcOneOutput(currentInp);

                // Output layer gradient
                layers_[lastLayerIndex].setDeltasFromLossGradient(
                    currentObs, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
                );
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
                    layers_[i].updateAdam(learningRate,epoch,0.9, 0.99, 1e-8);
                    layers_[i].setGradient(layers_[i].getGradient().setZero());
                }
            }
            Error = Error / numOfPatterns;

        // Epoch metrics: recorded every metricsAfterXEpochs epochs and at the final epoch;
        // in original units with chronological (unshuffled) calibration order when epochCtx is given
        if (Metrics::isEpochMetricRecord(epoch, metricsAfterXEpochs, maxIterations)) {
            size_t row = Metrics::epochMetricRowIndex(epoch, metricsAfterXEpochs, maxIterations);
            const Eigen::MatrixXd *YcalTrue = &Ytrain;
            const Eigen::MatrixXd *YvalTrue = &Yval;
            if (epochCtx != nullptr) {
                if (epochCtx->YtrueCalOrig.size() > 0) YcalTrue = &epochCtx->YtrueCalOrig;
                if (epochCtx->YtrueValOrig.size() > 0) YvalTrue = &epochCtx->YtrueValOrig;
            }

            calculateOutputs(Xtrain);
            Eigen::MatrixXd YpredCal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepCalPreds) YpredCal = epochCtx->prepCalPreds(YpredCal);

            calculateOutputs(Xval);
            Eigen::MatrixXd YpredVal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepValPreds) YpredVal = epochCtx->prepValPreds(YpredVal);

            Metrics::fillEpochMetricRow(resultMetrics, row, *YcalTrue, YpredCal, *YvalTrue, YpredVal);
        }
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = epoch;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }

    if(Error > maxError){
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(stop - start);

        result.converged = false;
        result.finalLoss = Error;
        result.iterations = maxIterations;
        result.time = duration.count();
    }
    return resultMetrics;
}

/// Implements batchBpEpochVal — mini-batch SGD training with periodic
/// train/validation metric evaluation. Each epoch delegates the weight updates
/// to batchBP (one iteration per epoch) and stops early when it reports
/// convergence. Metric matrices follow the same 14-entry layout as
/// onlineAdamEpochVal (7 train + 7 validation metrics).
/// @param batchSize Number of patterns per mini-batch (last batch may be smaller).
/// @return Vector of 14 metric matrices.
/// @throw std::logic_error / std::invalid_argument / std::runtime_error Same
///        checks as batchAdamEpochVal.
std::vector<Eigen::MatrixXd> MLP::batchBpEpochVal(
    const Eigen::MatrixXd &Xtrain,
    const Eigen::MatrixXd &Ytrain,
    const Eigen::MatrixXd &Xval,
    const Eigen::MatrixXd &Yval,
    int batchSize,
    int maxIterations,
    double maxError,
    double learningRate,
    int metricsAfterXEpochs,
    const EpochMetricContext *epochCtx)
{
    if (layers_.empty())
        throw std::logic_error("batchBpEpochVal called before initMLP");

    if (Xtrain.rows() == 0 || Ytrain.rows() == 0 || Xval.rows() == 0 || Yval.rows() == 0)
        throw std::invalid_argument("Empty training data passed to batchBpEpochVal");
            
    if (Xtrain.rows() != Ytrain.rows())
        throw std::invalid_argument("train matrices have different number of rows");

    if (Xval.rows() != Yval.rows())
        throw std::invalid_argument("validation matrices have different number of rows");

    if (maxIterations <= 0 || batchSize <= 0 || metricsAfterXEpochs < 0)
        throw std::invalid_argument("maxIterations and batchSize must be positive and metricsAfterXEpochs non-negative (0 = epoch-metric recording disabled)");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("metricsAfterXEpochs can't be bigger than maxIterations");

    std::vector<Eigen::MatrixXd> resultMetrics =
        Metrics::initEpochMetrics(maxIterations, metricsAfterXEpochs, static_cast<int>(Ytrain.cols()));

    auto start = high_resolution_clock::now();
    for(int epoch = 0; epoch < maxIterations; epoch++){
        // Run one epoch of mini-batch SGD (this also refreshes `result` and may
        // set result.converged)
        try {
            batchBP(1, maxError, batchSize, learningRate, Xtrain, Ytrain);
        } catch (const std::exception &ex) {
            std::cerr << "[batchBP] training failed: " << ex.what() << "\n";
            throw;
        }

        int epoch1 = epoch + 1;  // 1-based epoch for the metric helpers
        // Epoch metrics: recorded every metricsAfterXEpochs epochs and at the final epoch;
        // in original units with chronological (unshuffled) calibration order when epochCtx is given
        if (Metrics::isEpochMetricRecord(epoch1, metricsAfterXEpochs, maxIterations)) {
            size_t row = Metrics::epochMetricRowIndex(epoch1, metricsAfterXEpochs, maxIterations);
            const Eigen::MatrixXd *YcalTrue = &Ytrain;
            const Eigen::MatrixXd *YvalTrue = &Yval;
            if (epochCtx != nullptr) {
                if (epochCtx->YtrueCalOrig.size() > 0) YcalTrue = &epochCtx->YtrueCalOrig;
                if (epochCtx->YtrueValOrig.size() > 0) YvalTrue = &epochCtx->YtrueValOrig;
            }

            calculateOutputs(Xtrain);
            Eigen::MatrixXd YpredCal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepCalPreds) YpredCal = epochCtx->prepCalPreds(YpredCal);

            calculateOutputs(Xval);
            Eigen::MatrixXd YpredVal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepValPreds) YpredVal = epochCtx->prepValPreds(YpredVal);

            Metrics::fillEpochMetricRow(resultMetrics, row, *YcalTrue, YpredCal, *YvalTrue, YpredVal);
        }
        if(result.converged != false){
            result.iterations = epoch;
            break;
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    
    if(result.converged != true){
        result.iterations = maxIterations;
    }
    result.time = duration.count();

    return resultMetrics;
}

/// Implements onlinePenalizeBpEpochVal — online SGD with L2 weight penalization
/// plus periodic train/validation metric evaluation. Each epoch delegates the
/// weight updates to onlinePenalizeBP (one iteration per epoch) and stops early
/// when it reports convergence. Metric matrices follow the same 14-entry layout
/// as onlineAdamEpochVal (7 train + 7 validation metrics).
/// @param lambda Regularization strength, must be in (0, 1].
/// @return Vector of 14 metric matrices.
/// @throw std::logic_error / std::invalid_argument / std::runtime_error Same
///        checks as onlineAdamEpochVal, plus lambda range check.
std::vector<Eigen::MatrixXd> MLP::onlinePenalizeBpEpochVal(
    const Eigen::MatrixXd &Xtrain,
    const Eigen::MatrixXd &Ytrain,
    const Eigen::MatrixXd &Xval,
    const Eigen::MatrixXd &Yval,
    int maxIterations,
    double maxError,
    double learningRate,
    int metricsAfterXEpochs,
    double lambda,
    const EpochMetricContext *epochCtx)
{
    if (layers_.empty())
        throw std::logic_error("onlineBpEpochVal called before initMLP");

    if (Xtrain.rows() == 0 || Ytrain.rows() == 0 || Xval.rows() == 0 || Yval.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineBpEpochVal");
            
    if (Xtrain.rows() != Ytrain.rows())
        throw std::invalid_argument("train matrices have different number of rows");

    if (Xval.rows() != Yval.rows())
        throw std::invalid_argument("validation matrices have different number of rows");

    if (maxIterations <= 0 || metricsAfterXEpochs < 0)
        throw std::invalid_argument("maxIterations must be positive and metricsAfterXEpochs non-negative (0 = epoch-metric recording disabled)");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("metricsAfterXEpochs can't be bigger than maxIterations");
    
    if (lambda <= 0.0 || lambda > 1.0)
        throw std::invalid_argument("lambda must be between 0 and 1");

    std::vector<Eigen::MatrixXd> resultMetrics =
        Metrics::initEpochMetrics(maxIterations, metricsAfterXEpochs, static_cast<int>(Ytrain.cols()));

    auto start = high_resolution_clock::now();
    for(int epoch = 0; epoch < maxIterations; epoch++){
        // Run one epoch of online SGD with L2 penalization (may set result.converged)
        try {
            onlinePenalizeBP(1, maxError, learningRate, Xtrain, Ytrain,lambda);
        } catch (const std::exception &ex) {
            std::cerr << "[onlinePenalizeBP] training failed: " << ex.what() << "\n";
            throw;
        }

        int epoch1 = epoch + 1;  // 1-based epoch for the metric helpers
        // Epoch metrics: recorded every metricsAfterXEpochs epochs and at the final epoch;
        // in original units with chronological (unshuffled) calibration order when epochCtx is given
        if (Metrics::isEpochMetricRecord(epoch1, metricsAfterXEpochs, maxIterations)) {
            size_t row = Metrics::epochMetricRowIndex(epoch1, metricsAfterXEpochs, maxIterations);
            const Eigen::MatrixXd *YcalTrue = &Ytrain;
            const Eigen::MatrixXd *YvalTrue = &Yval;
            if (epochCtx != nullptr) {
                if (epochCtx->YtrueCalOrig.size() > 0) YcalTrue = &epochCtx->YtrueCalOrig;
                if (epochCtx->YtrueValOrig.size() > 0) YvalTrue = &epochCtx->YtrueValOrig;
            }

            calculateOutputs(Xtrain);
            Eigen::MatrixXd YpredCal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepCalPreds) YpredCal = epochCtx->prepCalPreds(YpredCal);

            calculateOutputs(Xval);
            Eigen::MatrixXd YpredVal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepValPreds) YpredVal = epochCtx->prepValPreds(YpredVal);

            Metrics::fillEpochMetricRow(resultMetrics, row, *YcalTrue, YpredCal, *YvalTrue, YpredVal);
        }
        if(result.converged != false){
            result.iterations = epoch;
            break;
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    
    if(result.converged != true){
        result.iterations = maxIterations;
    }
    result.time = duration.count();

    return resultMetrics;
}

/// Implements onlineMomentumBpEpochVal — online SGD with momentum plus periodic
/// train/validation metric evaluation. Each epoch delegates the weight updates
/// to onlineMomentumBP (one iteration per epoch) and stops early when it reports
/// convergence. Metric matrices follow the same 14-entry layout as
/// onlineAdamEpochVal (7 train + 7 validation metrics).
/// @param moment Momentum coefficient, must be in (0, 1].
/// @return Vector of 14 metric matrices.
/// @throw std::logic_error / std::invalid_argument / std::runtime_error Same
///        checks as onlineAdamEpochVal, plus moment range check.
std::vector<Eigen::MatrixXd> MLP::onlineMomentumBpEpochVal(
    const Eigen::MatrixXd &Xtrain,
    const Eigen::MatrixXd &Ytrain,
    const Eigen::MatrixXd &Xval,
    const Eigen::MatrixXd &Yval,
    int maxIterations,
    double maxError,
    double learningRate,
    int metricsAfterXEpochs,
    double moment,
    const EpochMetricContext *epochCtx)
{
    if (layers_.empty())
        throw std::logic_error("onlineBpEpochVal called before initMLP");

    if (Xtrain.rows() == 0 || Ytrain.rows() == 0 || Xval.rows() == 0 || Yval.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineBpEpochVal");
            
    if (Xtrain.rows() != Ytrain.rows())
        throw std::invalid_argument("train matrices have different number of rows");

    if (Xval.rows() != Yval.rows())
        throw std::invalid_argument("validation matrices have different number of rows");

    if (maxIterations <= 0 || metricsAfterXEpochs < 0)
        throw std::invalid_argument("maxIterations must be positive and metricsAfterXEpochs non-negative (0 = epoch-metric recording disabled)");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("metricsAfterXEpochs can't be bigger than maxIterations");
    
    if (moment <= 0.0 || moment > 1.0)
        throw std::invalid_argument("lambda must be between 0 and 1");

    std::vector<Eigen::MatrixXd> resultMetrics =
        Metrics::initEpochMetrics(maxIterations, metricsAfterXEpochs, static_cast<int>(Ytrain.cols()));

    auto start = high_resolution_clock::now();
    for(int epoch = 0; epoch < maxIterations; epoch++){
        // Run one epoch of online SGD with momentum (may set result.converged)
        try {
            onlineMomentumBP(1, maxError, learningRate, Xtrain, Ytrain, moment);
        } catch (const std::exception &ex) {
            std::cerr << "[onlineMomentumBP] training failed: " << ex.what() << "\n";
            throw;
        }

        int epoch1 = epoch + 1;  // 1-based epoch for the metric helpers
        // Epoch metrics: recorded every metricsAfterXEpochs epochs and at the final epoch;
        // in original units with chronological (unshuffled) calibration order when epochCtx is given
        if (Metrics::isEpochMetricRecord(epoch1, metricsAfterXEpochs, maxIterations)) {
            size_t row = Metrics::epochMetricRowIndex(epoch1, metricsAfterXEpochs, maxIterations);
            const Eigen::MatrixXd *YcalTrue = &Ytrain;
            const Eigen::MatrixXd *YvalTrue = &Yval;
            if (epochCtx != nullptr) {
                if (epochCtx->YtrueCalOrig.size() > 0) YcalTrue = &epochCtx->YtrueCalOrig;
                if (epochCtx->YtrueValOrig.size() > 0) YvalTrue = &epochCtx->YtrueValOrig;
            }

            calculateOutputs(Xtrain);
            Eigen::MatrixXd YpredCal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepCalPreds) YpredCal = epochCtx->prepCalPreds(YpredCal);

            calculateOutputs(Xval);
            Eigen::MatrixXd YpredVal = outputMat;
            if (epochCtx != nullptr && epochCtx->prepValPreds) YpredVal = epochCtx->prepValPreds(YpredVal);

            Metrics::fillEpochMetricRow(resultMetrics, row, *YcalTrue, YpredCal, *YvalTrue, YpredVal);
        }
        if(result.converged != false){
            result.iterations = epoch;
            break;
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    
    if(result.converged != true){
        result.iterations = maxIterations;
    }
    result.time = duration.count();

    return resultMetrics;
}

/**
 * Forward pass reusing existing weights
 */
/// Implements calcOneOutput. Propagates one input vector through all layers
/// (each layer consumes the previous layer's output) and stores the last
/// layer's output in the member `output`. Unlike runMLP this does not return
/// the output and uses Layer::calculateOutput, which keeps the current
/// activations inside the layers.
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
/// Implements calculateOutputs. Runs calcOneOutput for every row of
/// @p inputMat and stores the results in the member `outputMat`
/// (rows = patterns, cols = output neurons).
/// @throw std::logic_error If called before initMLP.
/// @throw std::invalid_argument If inputMat is empty or its row length does not
///        match the initialized input size (bias excluded).
void MLP::calculateOutputs(const Eigen::MatrixXd& inputMat){
    if (layers_.empty())
        throw std::logic_error("calculateOutputs called before initMLP");

    int inpSize = layers_[0].getWeights().cols()-1;
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

/// Implements getOutputs. Returns the output matrix filled by the last
/// calculateOutputs call.
Eigen::MatrixXd MLP::getOutputs() const{
    return outputMat;
}

/// Implements getFirstLayerDeltaSum. Collapses the first layer's delta vector
/// into a 1-element vector holding its sum; used as the error signal passed to
/// a preceding network (e.g. an LSTM feeding this MLP).
Eigen::VectorXd MLP::getFirstLayerDeltaSum(){
    Eigen::VectorXd delt(1);
    delt(0) = layers_[0].getDeltas().sum();
    return delt;
}

/// Implements getFirstLayerInputDelta. Propagates the first layer's deltas back
/// to the MLP's inputs: W_0^T * delta_0, with the bias row dropped
/// (conservativeResize removes the last element). Used to feed the gradient
/// into a preceding network (e.g. an LSTM feeding this MLP).
/// @return Vector of length nNeurons[0] (one entry per MLP input, bias excluded).
Eigen::VectorXd MLP::getFirstLayerInputDelta(){
    Eigen::VectorXd delt;
    delt = layers_[0].getWeights().transpose() * layers_[0].getDeltas();
    delt.conservativeResize(delt.size() - 1);
    return delt;
}

// Objective function configuration methods
/// Implements setObjectiveFunction. Sets the loss used for the output-layer
/// deltas in all training routines (default LEVEL_SLOPE).
void MLP::setObjectiveFunction(objective_func_type func) {
    objectiveFunc_ = func;
}

/// Implements setObjectiveFunction. Converts the name via
/// ObjectiveFunctions::fromString.
void MLP::setObjectiveFunction(const std::string& funcName) {
    objectiveFunc_ = ObjectiveFunctions::fromString(funcName);
}

/// Implements getObjectiveFunction.
objective_func_type MLP::getObjectiveFunction() const {
    return objectiveFunc_;
}

/// Implements setObjectiveAlpha. First blending/weighting coefficient of the
/// objective function (meaning depends on the selected objective).
void MLP::setObjectiveAlpha(double alpha) {
    objectiveAlpha_ = alpha;
}

/// Implements setObjectiveAlpha2. Coefficient of the curvature term of the
/// objective function (serves as the soft-DTW smoothing parameter for DILATE2).
void MLP::setObjectiveAlpha2(double alpha2) {
    objectiveAlpha2_ = alpha2;
}

/// Implements getObjectiveAlpha.
double MLP::getObjectiveAlpha() const {
    return objectiveAlpha_;
}

/// Implements getObjectiveAlpha2.
double MLP::getObjectiveAlpha2() const {
    return objectiveAlpha2_;
}
/// Implements horizonBackprop — multi-step-ahead support for a coupled
/// LSTM->MLP model, assuming a single scalar output per step (output(0)).
/// Algorithm:
/// 1. For every horizon step s: forward pass on stepInputs[s], record the
///    prediction simHorizon(s), then backpropagate a unit output delta to
///    obtain the per-unit-output first-layer input delta dE/dInput for that
///    step (collected in unitDeltas — the Jacobian of the input w.r.t. the
///    output).
/// 2. Compute the gradient of the configured objective function w.r.t. the
///    whole horizon prediction vector (ObjectiveFunctions::computeGradient).
/// 3. For every step again: backpropagate the step's gradient component and
///    ACCUMULATE the per-layer gradients via calculateBatchGradient, so a
///    single subsequent Adam update (usually updateWeightsAdamNoClear)
///    accounts for the whole horizon.
/// @param stepInputs One input vector per horizon step.
/// @param obsHorizon Observed values for the horizon (one per step).
/// @return Matrix (inputs x steps): each step's first-layer input delta scaled
///         by that step's gradient component, for feeding into the preceding
///         LSTM.
/// @throw std::logic_error If called before initMLP.
/// @throw std::invalid_argument If the number of step inputs differs from the
///        horizon length.
Eigen::MatrixXd MLP::horizonBackprop(const std::vector<Eigen::VectorXd> &stepInputs,
                                     const Eigen::VectorXd &obsHorizon) {
    if (layers_.empty())
        throw std::logic_error("horizonBackprop called before initMLP");
    const int n = static_cast<int>(obsHorizon.size());
    if (static_cast<int>(stepInputs.size()) != n)
        throw std::invalid_argument("horizonBackprop: stepInputs count must match horizon length");

    // Forward pass per future step; collect the prediction and the
    // first-layer input delta per unit output (used to scale the horizon
    // gradient when feeding deltas back to the preceding LSTM)
    Eigen::VectorXd simHorizon(n);
    Eigen::MatrixXd unitDeltas(layers_[0].getWeights().cols() - 1, n);
    Eigen::VectorXd dUnit(1);
    for (int s = 0; s < n; ++s) {
        calcOneOutput(stepInputs[s]);
        simHorizon(s) = output(0);
        // Unit delta at the output layer must live at the pre-activation level,
        // so scale by f'(z) of the output activation (chain rule)
        dUnit(0) = layers_[layers_.size()-1].getActivationDerivative()(0);
        layers_[layers_.size()-1].setDeltas(dUnit);
        if (layers_.size() > 1) {
            for (int i = static_cast<int>(layers_.size()) - 2; i >= 0; --i) {
                layers_[i].calculateDeltas(layers_[i+1].getWeights(), layers_[i+1].getDeltas(), activFuncs[i]);
            }
        }
        Eigen::VectorXd delt = layers_[0].getWeights().transpose() * layers_[0].getDeltas();
        unitDeltas.col(s) = delt.head(delt.size() - 1);
    }

    // Gradient of the objective w.r.t. the full horizon prediction vector
    Eigen::VectorXd grad = ObjectiveFunctions::computeGradient(
        obsHorizon, simHorizon, objectiveFunc_, objectiveAlpha_, objectiveAlpha2_, objectiveEpsilon_
    );

    // Backprop each step with its gradient component; calculateBatchGradient
    // accumulates, so all steps contribute before a single Adam update
    Eigen::VectorXd d(1);
    for (int s = 0; s < n; ++s) {
        calcOneOutput(stepInputs[s]);
        // Loss gradient w.r.t. the output, chained through the output activation
        d(0) = grad(s) * layers_[layers_.size()-1].getActivationDerivative()(0);
        layers_[layers_.size()-1].setDeltas(d);
        layers_[layers_.size()-1].calculateBatchGradient();
        if (layers_.size() > 1) {
            for (int i = static_cast<int>(layers_.size()) - 2; i >= 0; --i) {
                layers_[i].calculateDeltas(layers_[i+1].getWeights(), layers_[i+1].getDeltas(), activFuncs[i]);
                layers_[i].calculateBatchGradient();
            }
        }
    }

    // Per-step first-layer input deltas, each scaled by its gradient component
    return unitDeltas.array().rowwise() * grad.transpose().array();
}
