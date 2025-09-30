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

    bias1D = Eigen::VectorXd(sizes.numFilt);
    bias1D.setZero();

    auto weightInit = strToWeightInit(initType);

    switch (weightInit) {
        
        // Random initialization between minVal and maxVal
        case weight_init_type::RANDOM: { 
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<> dist(minVal, maxVal);
            filters1D = Eigen::MatrixXd::Random(sizes.numFilt, sizes.filtSize);         
            filters1D = filters1D.unaryExpr([&](double) { return dist(gen); });
            break;
        }
        
        // Latin Hypercube Sampling initialization
        case weight_init_type::LHS: {
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);

            // use size_t for all sizes and indices
            const std::size_t rows = static_cast<std::size_t>(sizes.filtSize);
            const std::size_t cols = static_cast<std::size_t>(sizes.numFilt);
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

            filters1D.resize(sizes.numFilt, sizes.filtSize);  // output matrix

            for (int col = 0; col < sizes.filtSize; ++col) {
                std::vector<float> column(sizes.numFilt);

                // sampling to one column
                for (int i = 0; i < sizes.numFilt; ++i) {
                    float sample = minVal + (i + dist(gen)) * range_interval;
                    column[i] = sample;
                }

                // shuffle the weights in column
                std::shuffle(column.begin(), column.end(), gen);

                // fill the column of weight matrix
                for (int row = 0; row < sizes.numFilt; ++row) {
                    filters1D(row, col) = column[row];
                }
            }

            break;
        }

        case weight_init_type::HE: {
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::normal_distribution<> dist(0.0, std::sqrt(2.0 / (sizes.numVars * sizes.filtSize)));
            filters1D = Eigen::MatrixXd(sizes.numFilt,sizes.filtSize);
            filters1D = filters1D.unaryExpr([&](double) { return dist(gen); });
            break;
        }
               
        default:
            std::cerr << "[Error]: Unknown weight initialization type! Selected RANDOM initialization." << std::endl;
            // Select random initialization
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<> dist(minVal, maxVal);
            filters1D = Eigen::MatrixXd(sizes.numFilt,sizes.filtSize);
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
    for(int i = 0; i < filRows ; i++){
        for(int j = 0; j < outRows; j++){
            outM.block(j , i * inCols, 1 , inCols) = filters.row(i) * inputs.block(j , 0 , filCols,inCols);
        }
    }
    return outM;
}

Eigen::MatrixXd CNNLayer::maxPool(const Eigen::MatrixXd& inputs, int size){
    // const int inCols = inputs.cols(), filCols = filters.cols(), filRows = filters.rows(), 
    //                    outRows = inputs.rows() - filRows + 1;
    // Eigen::MatrixXd outM = Eigen::MatrixXd(outRows, inCols * filCols);
    // for(int i = 0; i < filRows ; i++){
    //     for(int j = 0; j < outRows; j++){
    //         outM.block(j , i * inCols, 1 , inCols) = filters.row(i) * inputs.block(j , 0 , filCols,inCols);
    //     }
    // }
    // return outM;
    return inputs;
}

Eigen::MatrixXd CNNLayer::averagePool(const Eigen::MatrixXd& inputs, int size){
    // const int inCols = inputs.cols(), filCols = filters.cols(), filRows = filters.rows(), 
    //                    outRows = inputs.rows() - filRows + 1;
    // Eigen::MatrixXd outM = Eigen::MatrixXd(outRows, inCols * filCols);
    // for(int i = 0; i < filRows ; i++){
    //     for(int j = 0; j < outRows; j++){
    //         outM.block(j , i * inCols, 1 , inCols) = filters.row(i) * inputs.block(j , 0 , filCols,inCols);
    //     }
    // }
    // return outM;
    return inputs;
}

Eigen::MatrixXd CNNLayer::biasAndActivation(){
    Eigen::MatrixXd outM;

    for(Eigen::Index i = 0; i < filters1D.rows(); i++){
        for(Eigen::Index j = 0; j < currentInput1D.cols(); j++){
            activation1D.col(i * filters1D.rows() + j).array() += bias1D(i);
        }
    }

    switch (activ_func) {
        case activ_func_type::RELU:  // f(x) = max(0, x)
            outM = activation1D.array().max(0.0);
            break;
        
        case activ_func_type::SIGMOID:  // f(x) = 1 / (1 + exp(-x))
            outM = activation1D.array().unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
            break;

        case activ_func_type::LINEAR:  // f(x) = x
            outM = activation1D; 
            break;

        case activ_func_type::TANH:  // f(x) = (2 / (1 + exp(-2x))) - 1
            outM = activation1D.array().unaryExpr([](double x) { return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0; });
            break;
        
        case activ_func_type::GAUSSIAN:  // f(x) = exp(-x^2)
            outM = activation1D.array().unaryExpr([](double x) { return std::exp(-x * x); });
            break;

        case activ_func_type::IABS:  // f(x) = x / (1 + |x|)
            outM = activation1D.array().unaryExpr([](double x) { return x / (1.0 + std::abs(x)); });
            break;  

        case activ_func_type::LOGLOG:  // f(x) = exp(-exp(-x))
            outM = activation1D.array().unaryExpr([](double x) { return std::exp(-1 * std::exp(-x)); });
            break; 
        
        case activ_func_type::CLOGLOG:  // f(x) = 1 - exp(-exp(x))
            outM = activation1D.array().unaryExpr([](double x) { return 1 - std::exp(- std::exp(x)); });
            break;

        case activ_func_type::CLOGLOGM:  // f(x) = -log(1 - exp(-x)) + 1
            outM = activation1D.array().unaryExpr([](double x) { return 1 - 2 * std::exp(-0.7 * std::exp(x)); });
            break;

        case activ_func_type::ROOTSIG:  // f(x) = x / ((1 + sqrt(1 + x^2)) * sqrt(1 + x^2))
            outM = activation1D.array().unaryExpr([](double x) { return x / (1 + std::sqrt(1.0 + std::exp(-x * x))); });
            break;

        case activ_func_type::LOGSIG:  // f(x) = sigmoid(x)^2
            outM = activation1D.array().unaryExpr([](double x) { return pow((1 / (1 + std::exp(-x))), 2); });
            break;

        case activ_func_type::SECH:  // f(x) = 2 / (exp(x) + exp(-x))
            outM = activation1D.array().unaryExpr([](double x) { return 2 / (std::exp(x) + std::exp(-x)); });
            break;

        case activ_func_type::WAVE:  // f(x) = (1 − 𝑎2) exp(−𝑎2)
            outM = activation1D.array().unaryExpr([](double x) { return (1 - x * x) * exp(-x * x); });
            break;

        case activ_func_type::LEAKYRELU:   // f(x) = max(0.01 * x, x)
            outM = activation1D.array().unaryExpr([](double x) { return x > 0.0 ? x : 0.01 * x; });
            break;

        default:
            std::cerr << "[Error]: Unknown activation function type!" << std::endl;
            break;
    }
    return outM;
}

 /**
 *  Calculate activations and output of the layer
 */ 
void CNNLayer::calculateOutput1D(std::string activFunc){
    switch (pool) {
        case pool_type::NONE:
            output1D = biasAndActivation();
            break;
        
        case pool_type::MAX:
            output1D = maxPool(biasAndActivation(),sizes.poolSize);
            break;

        case pool_type::AVG:
            output1D = averagePool(biasAndActivation(),sizes.poolSize);
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