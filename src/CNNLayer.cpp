#include "CNNLayer.hpp"
#include <iostream>
#include <random>
#include <cmath>

using namespace std;

void CNNLayer::init1DCNNLayer(int numberOfFilters, 
                              int filterSize,
                              int inputRows,
                              int inputCols,
                              std::string initType,
                              std::string activFunc,
                              double minVal,
                              double maxVal)
{
    if (numberOfFilters <= 0 || filterSize <= 0 || inputRows <= 0 || inputCols <= 0)
        throw std::invalid_argument("input and filters dimensions must be positive");

    if (filterSize > inputRows)
        throw std::invalid_argument("filterSize can't exceed inputRows");

    sizes.filtSize = filterSize;
    sizes.numFilt = numberOfFilters;
    sizes.numVars = inputCols;
    sizes.varSize = inputRows;

    activ_func = strToActivation(activFunc);

    activation1D = Eigen::MatrixXd(sizes.varSize - sizes.filtSize + 1, sizes.numVars * sizes.numFilt);
    bias1D = Eigen::VectorXd(sizes.numFilt);
    bias1D.setZero();

    auto weightInit = strToWeightInit(initType);

    switch (weightInit) {
        
        // Random initialization between minVal and maxVal
        case weight_init_type::RANDOM: { 
            filters1D = Eigen::MatrixXd::Random(sizes.numFilt, sizes.filtSize);
            
            // Scale to desired range
            if (minVal != -1.0 || maxVal != 1.0) {
                filters1D = minVal + (filters1D.array() + 1.0) * 0.5 * (maxVal - minVal);
            }

            break;
        }
        
        // Latin Hypercube Sampling initialization
        case weight_init_type::LHS: {
            // one shared random number generator
            static std::mt19937 gen{std::random_device{}()};
            // static std::mt19937 gen{ 42 };  // always the same seed for debugging

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

            std::random_device rd;
            std::mt19937 gen(rd());
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
            filters1D = Eigen::MatrixXd::Random(sizes.numFilt,sizes.filtSize) * std::sqrt(6.0 / (sizes.numVars * sizes.filtSize));
        }
               
        default:
            std::cerr << "[Error]: Unknown weight initialization type! Selected RANDOM initialization." << std::endl;
            // Select random initialization
            filters1D = Eigen::MatrixXd::Random(sizes.numFilt, sizes.filtSize);
            // Scale to desired range
            filters1D = filters1D.array() * (maxVal - minVal) / 2.0 + (maxVal + minVal) / 2.0;
            
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

 /**
 *  Calculate activations and output of the layer
 */ 
void CNNLayer::calculateOutput1D(std::string activFunc){
    for(int i = 0; i < sizes.numFilt; i++){
        for(int j = 0; j < sizes.varSize - sizes.filtSize + 1; j++){
            activation1D.block(j , i * sizes.numVars, 1 , sizes.numVars) = 
            (filters1D.row(i) * currentInput1D.block(j , 0 , sizes.filtSize,sizes.numVars)).array() + bias1D(i);
        }
    }
    switch (strToActivation(activFunc)) {
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
            output1D = activation1D.array().unaryExpr([](double x) { return x / (1 + std::sqrt(1.0 + std::exp(-x * x))); });
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

    // Detect any NaN or infinite
    if (!output1D.array().isFinite().all()) {
        std::cerr << "[Warning] Non-finite activations detected!\n";
    }
}

Eigen::MatrixXd CNNLayer::getOutput1D(){
    return output1D;
}

Eigen::MatrixXd CNNLayer::getActivations1D(){
    return activation1D;
}