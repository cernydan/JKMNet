#include "CNNLayer.hpp"
#include <iostream>
#include <random>
#include <cmath>

using namespace std;

void CNNLayer::init1DCNNLayer(int numberOfFilters, 
                              int filterSize, 
                              int inputRows,
                              int inputCols,
                              int poolSize,
                              std::string initType,
                              std::string activFunc,
                              double minVal,
                              double maxVal,
                              std::string poolType,
                              int rngSeed)
{
    if (numberOfFilters <= 0 || filterSize <= 0 || inputRows <= 0 || inputCols <= 0)
        throw std::invalid_argument("input and filters dimensions must be positive");

    if (filterSize > inputRows)
        throw std::invalid_argument("filterSize can't exceed inputRows");

    sizes.filtSize = filterSize;
    sizes.numFilt = numberOfFilters;
    sizes.numVars = inputCols;
    sizes.varSize = inputRows;
    sizes.poolSize = poolSize;

    activ_func = strToActivation(activFunc);
    pool = strToPoolType(poolType);

    if(sizes.poolSize == 0){pool = pool_type::NONE;}

    bias1D = Eigen::VectorXd::Zero(sizes.numFilt);
    MtForAdamBias = Eigen::VectorXd::Zero(sizes.numFilt);
    VtForAdamBias = Eigen::VectorXd::Zero(sizes.numFilt);
    biasGradient = Eigen::VectorXd::Zero(sizes.numFilt);

    MtForAdam = Eigen::MatrixXd::Zero(sizes.filtSize, sizes.numFilt);
    VtForAdam = Eigen::MatrixXd::Zero(sizes.filtSize, sizes.numFilt);

    auto weightInit = strToWeightInit(initType);

    switch (weightInit) {
        
        // Random initialization between minVal and maxVal
        case weight_init_type::RANDOM: { 
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<> dist(minVal, maxVal);
            filters1D = Eigen::MatrixXd::Random(sizes.filtSize, sizes.numFilt);         
            filters1D = filters1D.unaryExpr([&](double) { return dist(gen); });
            break;
        }
        
        // Latin Hypercube Sampling initialization
        case weight_init_type::LHS: {
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);

            // use size_t for all sizes and indices
            const std::size_t rows = static_cast<std::size_t>(sizes.numFilt);
            const std::size_t cols = static_cast<std::size_t>(sizes.filtSize);
            const std::size_t totalWeights = rows * cols;
            const double span = maxVal - minVal;

            // reserve space up front
            std::vector<double> samples;
            samples.reserve(totalWeights);

            // for each stratum (subinterval), draw one point
            for (std::size_t i = 0; i < totalWeights; ++i) {
                double a = minVal + span * (double(i) / double(totalWeights));
                double b = minVal + span * (double(i + 1) / double(totalWeights));
                std::uniform_real_distribution<double> dist(a, b);
                samples.push_back(dist(gen));
            }

            // shuffle so weights aren’t tied to particular strata (subinterval)
            std::shuffle(samples.begin(), samples.end(), gen);

            // write back into the matrix (row-major)
            filters1D.resize(rows, cols);
            std::size_t idx = 0;
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    filters1D(r, c) = samples[idx++];
                }
            }

            break;
        }
        // Marta - Latin hypercube sampling by columns
        case weight_init_type::LHS2: {
            float range = maxVal - minVal;
            float range_interval = range / static_cast<float>(sizes.numFilt);

            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            filters1D.resize(sizes.filtSize, sizes.numFilt);  // output matrix

            for (int col = 0; col < sizes.numFilt; ++col) {
                std::vector<float> column(sizes.filtSize);

                // sampling to one column
                for (int i = 0; i < sizes.filtSize; ++i) {
                    float sample = minVal + (i + dist(gen)) * range_interval;
                    column[i] = sample;
                }

                // shuffle the weights in column
                std::shuffle(column.begin(), column.end(), gen);

                // fill the column of weight matrix
                for (int row = 0; row < sizes.filtSize; ++row) {
                    filters1D(row, col) = column[row];
                }
            }

            break;
        }

        case weight_init_type::HE: {
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::normal_distribution<> dist(0.0, std::sqrt(2.0 / (sizes.numVars * sizes.filtSize)));
            filters1D = Eigen::MatrixXd(sizes.filtSize, sizes.numFilt);
            filters1D = filters1D.unaryExpr([&](double) { return dist(gen); });
            break;
        }
               
        default:
            std::cerr << "[Error]: Unknown weight initialization type! Selected RANDOM initialization." << std::endl;
            // Select random initialization
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<> dist(minVal, maxVal);
            filters1D = Eigen::MatrixXd(sizes.filtSize, sizes.numFilt);
            filters1D = filters1D.unaryExpr([&](double) { return dist(gen); });
            break;
    }
}

/**
 *  Setter for 1D filters matrix
 */
void CNNLayer::setFilters1D(const Eigen::MatrixXd& newFilters){
    filters1D = newFilters;
}

/**
 *  Setter for 1D bias vector
 */
void CNNLayer::setBias1D(const Eigen::VectorXd& newBias){
    bias1D = newBias;
}

 /**
 *  Getter for 1D filters matrix
 */
Eigen::MatrixXd CNNLayer::getFilters1D(){
    return filters1D;
}

 /**
 *  Getter: Returns the current architecture (vector of neurons in each layer)
 */
void CNNLayer::setCurrentInput1D(const Eigen::MatrixXd& currentInp){
    currentInput1D = currentInp;
}

Eigen::MatrixXd CNNLayer::convolution1D(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& filters){
    const int inCols = inputs.cols(), filCols = filters.cols(), filRows = filters.rows(), 
                       outRows = inputs.rows() - filRows + 1;
    Eigen::MatrixXd outM = Eigen::MatrixXd(outRows, inCols * filCols);
    double konvo;
    for(int i = 0; i < filCols; i++){
        for(int j = 0; j < inCols; j++){
            for(int k = 0; k < outRows; k++){
                konvo = 0.0;
                for(int l = 0; l < filRows; l++){
                    konvo += filters(l,i) * inputs(k + l,j);
                }
                outM(k , i * inCols + j) = konvo;
            }
        }
    }
    return outM;
}

Eigen::MatrixXd CNNLayer::convolution1DSeparCols(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& filters){
    const int inCols = inputs.cols(), filCols = filters.cols(), filRows = filters.rows(), 
                       outRows = inputs.rows() - filRows + 1;
    if (inCols != filCols)
        throw std::invalid_argument("[convolution1DSeparCols] Col nums don't match");
    Eigen::MatrixXd outM = Eigen::MatrixXd(outRows, inCols);
    double konvo;
    for(int j = 0; j < inCols; j++){
        for(int k = 0; k < outRows; k++){
            konvo = 0.0;
            for(int l = 0; l < filRows; l++){
                konvo += filters(l,j) * inputs(k + l,j);
            }
            outM(k , j) = konvo;
        }
    }
    return outM;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi>CNNLayer::maxPool(const Eigen::MatrixXd& inputs, int size){
    const int inCols = inputs.cols(), inRows = inputs.rows(); 
    if (inRows < size)
        throw std::invalid_argument("pool size can't be larger than input rows");

    if (inRows % size != 0)
        throw std::invalid_argument("number of input rows must be divisible by pool size");
    
    const int outRows = inRows / size;
    Eigen::MatrixXd outM(outRows, inCols);
    Eigen::MatrixXi indices(outRows, inCols);

    for (int i = 0; i < inCols; i++) {
        for (int j = 0; j < outRows; j++) {
            Eigen::VectorXd block = inputs.block(j * size, i, size, 1);
            double maxVal = block.maxCoeff();
            Eigen::Index idx;
            block.maxCoeff(&idx);
            int globalIndex = j * size + idx;
            outM(j, i)  = maxVal;
            indices(j, i) = globalIndex;
        }
    }
    return {outM, indices};
}

Eigen::MatrixXd CNNLayer::averagePool(const Eigen::MatrixXd& inputs, int size){
    const int inCols = inputs.cols(), inRows = inputs.rows();
    if (inRows < size)
        throw std::invalid_argument("pool size can't be larger than input rows");

    if (inRows % size != 0)
        throw std::invalid_argument("number of input rows must be divisible by pool size");

    const int outRows = inRows / size;
    Eigen::MatrixXd outM(outRows, inCols);

    for (int i = 0; i < inCols; i++) {
        for (int j = 0; j < outRows; j++) {
            outM(j, i) = inputs.block(j * size, i, size, 1).mean();
        }
    }
    return outM;
}

Eigen::MatrixXd CNNLayer::flipRowsAndPad(const Eigen::MatrixXd& mat, int pad){
    Eigen::MatrixXd flipped = mat.colwise().reverse();
    Eigen::MatrixXd newMat = Eigen::MatrixXd::Zero(flipped.rows() + 2 * pad, flipped.cols());
    newMat.block(pad, 0, flipped.rows(), flipped.cols()) = flipped;

    return newMat;
}

Eigen::MatrixXd CNNLayer::sumColumnBlocks(const Eigen::MatrixXd& mat, int blockSize)
{
    int rows = mat.rows();
    int groups = mat.cols() / blockSize;
    Eigen::MatrixXd newMat = Eigen::MatrixXd(rows, groups);

    for (int i = 0; i < groups; ++i) {
        newMat.col(i) = mat.block(0, i * blockSize, rows, blockSize).rowwise().sum();
    }
    return newMat;
}

 /**
 *  Calculate activations and output of the layer
 */ 
void CNNLayer::calculateOutput1D(){
    activation1D = convolution1D(currentInput1D,filters1D);

    for(Eigen::Index i = 0; i < filters1D.cols(); i++){
        for(Eigen::Index j = 0; j < currentInput1D.cols(); j++){
            activation1D.col(i * currentInput1D.cols() + j).array() += bias1D(i);
        }
    }

    switch (activ_func) {
        case activ_func_type::RELU:  // f(x) = max(0, x)
            output1D = activation1D.array().max(0.0);
            break;
        
        case activ_func_type::SIGMOID:  // f(x) = 1 / (1 + exp(-x))
            output1D = activation1D.array().unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
            break;

        case activ_func_type::LINEAR:  // f(x) = x
            output1D = activation1D; 
            break;

        case activ_func_type::TANH:  // f(x) = (2 / (1 + exp(-2x))) - 1
            output1D = activation1D.array().unaryExpr([](double x) { return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0; });
            break;
        
        case activ_func_type::GAUSSIAN:  // f(x) = exp(-x^2)
            output1D = activation1D.array().unaryExpr([](double x) { return std::exp(-x * x); });
            break;

        case activ_func_type::IABS:  // f(x) = x / (1 + |x|)
            output1D = activation1D.array().unaryExpr([](double x) { return x / (1.0 + std::abs(x)); });
            break;  

        case activ_func_type::LOGLOG:  // f(x) = exp(-exp(-x))
            output1D = activation1D.array().unaryExpr([](double x) { return std::exp(-1 * std::exp(-x)); });
            break; 
        
        case activ_func_type::CLOGLOG:  // f(x) = 1 - exp(-exp(x))
            output1D = activation1D.array().unaryExpr([](double x) { return 1 - std::exp(- std::exp(x)); });
            break;

        case activ_func_type::CLOGLOGM:  // f(x) = -log(1 - exp(-x)) + 1
            output1D = activation1D.array().unaryExpr([](double x) { return 1 - 2 * std::exp(-0.7 * std::exp(x)); });
            break;

        case activ_func_type::ROOTSIG:  // f(x) = x / ((1 + sqrt(1 + x^2)) * sqrt(1 + x^2))
            output1D = activation1D.array().unaryExpr([](double x) { return x / (1.0 + std::sqrt(1.0 + x * x)); });
            break;

        case activ_func_type::LOGSIG:  // f(x) = sigmoid(x)^2
            output1D = activation1D.array().unaryExpr([](double x) { return pow((1 / (1 + std::exp(-x))), 2); });
            break;

        case activ_func_type::SECH:  // f(x) = 2 / (exp(x) + exp(-x))
            output1D = activation1D.array().unaryExpr([](double x) { return 2 / (std::exp(x) + std::exp(-x)); });
            break;

        case activ_func_type::WAVE:  // f(x) = (1 − 𝑎2) exp(−𝑎2)
            output1D = activation1D.array().unaryExpr([](double x) { return (1 - x * x) * exp(-x * x); });
            break;

        case activ_func_type::LEAKYRELU:   // f(x) = max(0.01 * x, x)
            output1D = activation1D.array().unaryExpr([](double x) { return x > 0.0 ? x : 0.01 * x; });
            break;

        default:
            std::cerr << "[Error]: Unknown activation function type!" << std::endl;
            break;
    }

    switch (pool) {
        case pool_type::NONE:
            break;
        
        case pool_type::MAX:
            std::tie(output1D , maxPoolBpIndicesHelp) = maxPool(output1D,sizes.poolSize);
            break;

        case pool_type::AVG:
            output1D = averagePool(output1D,sizes.poolSize);
            break;

        default:
            std::cerr << "[Error]: Unknown pooling type!" << std::endl;
            break;
    }
}

Eigen::MatrixXd CNNLayer::getOutput1D(){
    return output1D;
}

Eigen::MatrixXd CNNLayer::getActivations1D(){
    return activation1D;
}

Eigen::VectorXd CNNLayer::getBias1D(){
    return bias1D;
}

Eigen::VectorXd CNNLayer::getDelta(){
    return delta;
}

void CNNLayer::calculateGradients(){
    Eigen::MatrixXd activationsDelta;

        switch (activ_func) {
        case activ_func_type::RELU:  // f'(x) = 0 for x <= 0 ; f'(x) = 1 for x > 0
            activationsDelta = activation1D.array().unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0; });
            break;
        
        case activ_func_type::SIGMOID:  // f'(x) = sigmoid(x) * (1 - sigmoid(x))
            activationsDelta = activation1D.array().unaryExpr([](double x) 
            { return (1.0 / (1.0 + std::exp(-x))) * (1.0 - (1.0 / (1.0 + std::exp(-x)))); });
            break;

        case activ_func_type::LINEAR:  // f'(x) = 1
            activation1D.setOnes(); 
            break;

        case activ_func_type::TANH:  // f'(x) = 1 - tanh(x)^2
            activationsDelta = activation1D.array().unaryExpr([](double x) 
            { return 1.0 - ((2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0) * ((2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0); });
            break;
        
        case activ_func_type::GAUSSIAN:  // f'(x) = -2x * exp(-x^2)
            activationsDelta = activation1D.array().unaryExpr([](double x) { return -2.0 * x * std::exp(-x * x); });
            break;

        case activ_func_type::IABS:  // f'(x) = 1 / (1 + |x|)^2  ...not sure // MJ: correct
            activationsDelta = activation1D.array().unaryExpr([](double x) 
            { return 1.0 / ((1.0 + std::abs(x)) * (1.0 + std::abs(x))); });
            break;  

        case activ_func_type::LOGLOG:  // f'(x) = exp(-exp(-x) - x)  // MJ: f'(x) = exp(-exp(-x)) * exp(-x)
            activationsDelta = activation1D.array().unaryExpr([](double x) { return std::exp(-1.0 * std::exp(-x)) * std::exp(-x);});
            break; 
        
        case activ_func_type::CLOGLOG:  // f'(x) = exp(-exp(x) + x)
            activationsDelta = activation1D.array().unaryExpr([](double x) { return std::exp(-1.0 * std::exp(x) + x); });
            break;

        case activ_func_type::CLOGLOGM:  // f'(x) = 7 * exp(x - 0.7 * exp(x)) / 5.0    (for f(x) = 1 - 2 * exp(-0.7 * exp(x)))  
            // MJ: f'(x) = - 1 / (exp(x) - 1) for x > 0
            activationsDelta = activation1D.array().unaryExpr([](double x) 
            { return 1.4 * std::exp(x) * std::exp(-0.7 * std::exp(x)); });
            break;

        case activ_func_type::ROOTSIG:  // f'(x) for f(x) = x / (1 + sqrt(1.0 + exp(-x * x)))  
            // MJ: ( 1 + 2 * sqrt(1 + x^2) - (1 + x^2)^(3/2) ) / ( (1 + x^2)^(3/2) * (1 + sqrt(1 + x^2))^2 )
            activationsDelta = activation1D.array().unaryExpr([](double x) 
            { return 1.0 / ((1.0 + std::sqrt(1.0 + x * x)) * std::sqrt(1.0 + x * x)); });
            break;

        case activ_func_type::LOGSIG:  // f'(x) = 2 * sigmoid(x)^2 * (1 - sigmoid(x))
            activationsDelta = activation1D.array().unaryExpr([](double x) 
            { return 2.0 * std::exp(-x) / std::pow((1 + std::exp(-x)),3); });
            break;

        case activ_func_type::SECH:  // f'(x) = -sech(x) * tanh(h)
            activationsDelta = activation1D.array().unaryExpr([](double x) 
            { return - (2.0 / (std::exp(x) + std::exp(-x))) * ((2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0); });
            break;

        case activ_func_type::WAVE:  // f'(x) = 2x * (x^2 - 2) * exp(-x^2)
            activationsDelta = activation1D.array().unaryExpr([](double x) 
            { return 2.0 * x * (x * x - 2.0) * exp(-x * x); });
            break;

        case activ_func_type::LEAKYRELU:  // f'(x) = 0.01 for x <= 0 ; f'(x) = 1 for x > 0
            activationsDelta = activation1D.array().unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.01; });
            break;

        default:
            std::cerr << "[Error]: Unknown activation function type!" << std::endl;
            break;
    }

    // Detect any NaN or infinite
    if (!activation1D.array().isFinite().all()) {
        std::cerr << "[Warning CNN] Non-finite activations detected!\n";
    }

    switch (pool) {
    case pool_type::NONE:
        activationsDelta.array().colwise() *= deltaFromNextLayer.array();
        break;
    
    case pool_type::MAX:
        for(int i = 0; i < activationsDelta.cols(); i++){
            maxPoolBpHelp = Eigen::VectorXd::Zero(activation1D.rows());
            for(int j = 0; j < deltaFromNextLayer.size(); j++){
                maxPoolBpHelp(maxPoolBpIndicesHelp(j,i)) = deltaFromNextLayer(j);
            }
            activationsDelta.col(i).array() *= maxPoolBpHelp.array();
        }
        break;

    case pool_type::AVG:
        avgPoolBpHelp = Eigen::VectorXd(activation1D.rows());
        for(int i = 0; i < deltaFromNextLayer.size(); i++){
            for(int j = 0; j < sizes.poolSize; j++){
                avgPoolBpHelp(i*sizes.poolSize + j) = deltaFromNextLayer(i) / sizes.poolSize;
            }
        }
        activationsDelta.array().colwise() *= avgPoolBpHelp.array();

        break;

    default:
        std::cerr << "[Error]: Unknown pooling type!" << std::endl;
        break;
    }

    activationsDelta = sumColumnBlocks(activationsDelta, activationsDelta.cols() / sizes.numFilt);

    biasGradient = activationsDelta.colwise().sum();

    filtersGradient = convolution1D(currentInput1D,activationsDelta);
    filtersGradient = sumColumnBlocks(filtersGradient, filtersGradient.cols() / sizes.numFilt);

    inputGradient = convolution1DSeparCols(flipRowsAndPad(filters1D,activationsDelta.rows()-1),activationsDelta);

    delta = inputGradient.rowwise().sum();
}

void CNNLayer::setDeltaFromNextLayer(const Eigen::VectorXd& nextDelta){
    deltaFromNextLayer = nextDelta;
}

void CNNLayer::updateFiltersAdam(double learningRate, int iterationNum, double beta1, double beta2, double epsi){
    MtForAdam = beta1 * MtForAdam.array() + (1 - beta1) * filtersGradient.array();
    VtForAdam = beta2 * VtForAdam.array() + (1 - beta2) * filtersGradient.array() * filtersGradient.array();
    filters1D -= learningRate * (MtForAdam.array() / ((1 - std::pow(beta1, iterationNum)) * 
                (sqrt(VtForAdam.array()/(1 - std::pow(beta2,iterationNum))) + epsi))).matrix();

    MtForAdamBias = beta1 * MtForAdamBias.array() + (1 - beta1) * biasGradient.array();
    VtForAdamBias = beta2 * VtForAdamBias.array() + (1 - beta2) * biasGradient.array() * biasGradient.array();
    bias1D -= learningRate * (MtForAdamBias.array() / ((1 - std::pow(beta1, iterationNum)) * 
               (sqrt(VtForAdamBias.array()/(1 - std::pow(beta2,iterationNum))) + epsi))).matrix();
}
