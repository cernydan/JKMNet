#include "LSTMLayer.hpp"
#include <iostream>
#include <random>
#include <cmath>

using namespace std;

Eigen::VectorXd LSTMLayer::sigmoidVector(const Eigen::VectorXd& vec) {
    Eigen::VectorXd res = vec;
    res = res.array().unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    return res;
}

Eigen::VectorXd LSTMLayer::tanhVector(const Eigen::VectorXd& vec) {
    Eigen::VectorXd res = vec;
    res = res.array().unaryExpr([](double x) { return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0; });
    return res;
}

void LSTMLayer::initLSTMLayer(const int numInputs,        
                const int numCells,
                const int numTimeSteps,
                const int numOutputTimeSteps,
                bool isFirstLayer,
                std::string initType,
                int rngSeed,
                double minVal,
                double maxVal){

    settings.cells = numCells;
    settings.cells4gates = 4 * settings.cells;
    settings.inputs = numInputs;
    settings.timeSteps = numTimeSteps;
    settings.outTS = numOutputTimeSteps;
    settings.isFirstLayer = isFirstLayer;
    auto weightInit = strToWeightInit(initType);

    switch (weightInit) {
        
        // Random initialization between minVal and maxVal
        case weight_init_type::RANDOM: { 
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<> dist(minVal, maxVal);
            W = Eigen::MatrixXd(settings.cells4gates, settings.inputs);          
            W = W.unaryExpr([&](double) { return dist(gen); });
            U = Eigen::MatrixXd(settings.cells4gates, settings.cells);          
            U = U.unaryExpr([&](double) { return dist(gen); });
            break;
        }

        case weight_init_type::HE: {
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::normal_distribution<> dist(0.0, std::sqrt(2.0 / (settings.inputs)));
            W = Eigen::MatrixXd(settings.cells4gates, settings.inputs);          
            W = W.unaryExpr([&](double) { return dist(gen); });
            U = Eigen::MatrixXd(settings.cells4gates, settings.cells);          
            U = U.unaryExpr([&](double) { return dist(gen); });
            break;
        }

        case weight_init_type::XG: {
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::normal_distribution<> dist(0.0, std::sqrt(2.0 / (settings.inputs + settings.cells)));
            W = Eigen::MatrixXd(settings.cells4gates, settings.inputs);          
            W = W.unaryExpr([&](double) { return dist(gen); });
            U = Eigen::MatrixXd(settings.cells4gates, settings.cells);          
            U = U.unaryExpr([&](double) { return dist(gen); });
            break;
        }

        default:
            std::cerr << "[Error]: Unknown weight initialization type! Selected RANDOM initialization." << std::endl;
            // Select random initialization
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<> dist(minVal, maxVal);
            W = Eigen::MatrixXd(settings.cells4gates, settings.inputs);          
            W = W.unaryExpr([&](double) { return dist(gen); });
            U = Eigen::MatrixXd(settings.cells4gates, settings.cells);          
            U = U.unaryExpr([&](double) { return dist(gen); });
            break;
    }

    b = Eigen::VectorXd::Zero(settings.cells4gates);
    timeStepsInputs = Eigen::MatrixXd::Zero(settings.timeSteps,settings.inputs);
    gatesActivations = Eigen::MatrixXd::Zero(settings.cells4gates,settings.timeSteps);
    gatesOutputs = Eigen::MatrixXd::Zero(settings.cells4gates,settings.timeSteps);
    cellState = Eigen::MatrixXd::Zero(settings.cells,settings.timeSteps);
    tanhCellState = Eigen::MatrixXd::Zero(settings.cells,settings.timeSteps);
    output = Eigen::MatrixXd::Zero(settings.cells,settings.timeSteps);
    forwardOutput = Eigen::MatrixXd::Zero(settings.cells,settings.outTS);
    Wgradient = Eigen::MatrixXd::Zero(settings.cells4gates, settings.inputs);
    Ugradient = Eigen::MatrixXd::Zero(settings.cells4gates, settings.cells);
    bGradient = Eigen::VectorXd::Zero(settings.cells4gates);
    W_MtForAdam = Eigen::MatrixXd::Zero(settings.cells4gates, settings.inputs);
    W_VtForAdam = Eigen::MatrixXd::Zero(settings.cells4gates, settings.inputs);
    U_MtForAdam = Eigen::MatrixXd::Zero(settings.cells4gates, settings.cells);
    U_VtForAdam = Eigen::MatrixXd::Zero(settings.cells4gates, settings.cells);
    b_MtForAdam = Eigen::VectorXd::Zero(settings.cells4gates);
    b_VtForAdam = Eigen::VectorXd::Zero(settings.cells4gates);
    deltaOutput = Eigen::MatrixXd::Zero(settings.cells,settings.timeSteps);
    deltaCellState = Eigen::MatrixXd::Zero(settings.cells,settings.timeSteps);
    deltaGates = Eigen::MatrixXd::Zero(settings.cells4gates,settings.timeSteps);
    deltaInputs = Eigen::MatrixXd::Zero(settings.inputs, settings.timeSteps);

    b.segment(settings.cells,settings.cells).setOnes();  //Setting forget gate bias to 1 should help training

    settings.initialized = true;
}

void LSTMLayer::setInputTSSegment(const Eigen::MatrixXd& inputSegment){
    if (!settings.initialized)
        throw std::logic_error("[setInputTSSegment] called before [initLSTMLayer]");

    if (inputSegment.cols() != timeStepsInputs.cols())
        throw std::invalid_argument("[setInputTSSegment] Number of inputs (cols) doesn't match the initialized");

    if (inputSegment.rows() != timeStepsInputs.rows())
        std::cerr << "[Warning][setInputTSSegment] Number of time-steps (rows) doesn't match the initialized"<<"\n";

    timeStepsInputs = inputSegment;
}

void LSTMLayer::calculateTimeSteps(){
    if (!settings.initialized)
        throw std::logic_error("[calculateTimeSteps] called before [initLSTMLayer]");
    int c = settings.cells;

    //first TS
    gatesActivations.col(0) = W * timeStepsInputs.row(0).transpose() + b;
    gatesOutputs.col(0).segment(0, c) = tanhVector(gatesActivations.col(0).segment(0, c));   //candidate
    gatesOutputs.col(0).segment(c, 3 * c) = 
                                sigmoidVector(gatesActivations.col(0).segment(c, 3 * c));    //forget,input,output

    cellState.col(0) = gatesOutputs.col(0).segment(2 * c, c).array() *    //input 
                       gatesOutputs.col(0).segment(0, c).array();         //candidate
    
    tanhCellState.col(0) = tanhVector(cellState.col(0));
    
    output.col(0) = gatesOutputs.col(0).segment(3 * c, c).array() *       //output
                    tanhCellState.col(0).array();

    //rest of TS
    for(int i = 1 ; i < timeStepsInputs.rows() ; i++){
        gatesActivations.col(i) = W * timeStepsInputs.row(i).transpose() + b + U * output.col(i-1);
        gatesOutputs.col(i).segment(0, c) = tanhVector(gatesActivations.col(i).segment(0, c));    //candidate
        gatesOutputs.col(i).segment(c, 3 * c) = 
                                sigmoidVector(gatesActivations.col(i).segment(c, 3 * c));         //forget,input,output

        cellState.col(i) = cellState.col(i-1).array() * 
                           gatesOutputs.col(i).segment(c, c).array() +      //forget
                           gatesOutputs.col(i).segment(2 * c, c).array() *  //input  
                           gatesOutputs.col(i).segment(0, c).array();       //candidate
        
        tanhCellState.col(i) = tanhVector(cellState.col(i));

        output.col(i) = gatesOutputs.col(i).segment(3 * c, c).array() *     //output
                        tanhCellState.col(i).array();
    }
    forwardOutput = output.rightCols(settings.outTS);
}

void LSTMLayer::calculateGradients(){
    int t = settings.timeSteps - 1;
    int c = settings.cells;

    // Initialize deltas for all time steps
    deltaGates.setZero();
    deltaCellState.setZero();
    deltaOutput.setZero();

    for(int i = t ; i >= 0 ; i--){
        Eigen::VectorXd delt = Eigen::VectorXd::Zero(c);

        // Add gradient from next layer (only for output time steps)
        if(i >= settings.timeSteps - settings.outTS){
            int idx = i - (settings.timeSteps - settings.outTS);
            delt = nextLayerDelta.col(idx);
        }

        // Add gradient flowing back through recurrent connection
        if(i < t){
            delt += U.transpose() * deltaGates.col(i+1);
        }
        deltaOutput.col(i) = delt;

        // Gradient w.r.t. cell state
        deltaCellState.col(i) = deltaOutput.col(i).array() *
                                gatesOutputs.col(i).segment(3 * c, c).array() *    // output gate
                                (1.0 - tanhCellState.col(i).array().square());     // tanh'

        // Accumulate gradient from future cell state (forget gate path)
        if(i < t){
            deltaCellState.col(i).array() += deltaCellState.col(i+1).array() *
                                            gatesOutputs.col(i+1).segment(c, c).array(); // forget gate
        }

        // Candidate gate gradient (tanh)
        deltaGates.col(i).segment(0, c) =
                                         deltaCellState.col(i).array() *
                                         gatesOutputs.col(i).segment(2 * c, c).array() *
                                         (1.0 - gatesOutputs.col(i).segment(0, c).array().square());

        // Forget gate gradient (sigmoid)
        deltaGates.col(i).segment(c, c) =
                                                    deltaCellState.col(i).array() *
                                                    cellState.col(i).array() *  // Use current cell state, not previous!
                                                    gatesOutputs.col(i).segment(c, c).array() *
                                                    (1.0 - gatesOutputs.col(i).segment(c, c).array());

        // Input gate gradient (sigmoid)
        deltaGates.col(i).segment(2 * c, c) =
                                             deltaCellState.col(i).array() *
                                             gatesOutputs.col(i).segment(0, c).array() *
                                             gatesOutputs.col(i).segment(2 * c, c).array() *
                                             (1.0 - gatesOutputs.col(i).segment(2 * c, c).array());

        // Output gate gradient (sigmoid)
        deltaGates.col(i).segment(3 * c, c) =
                                             deltaOutput.col(i).array() *
                                             tanhCellState.col(i).array() *
                                             gatesOutputs.col(i).segment(3 * c, c).array() *
                                             (1.0 - gatesOutputs.col(i).segment(3 * c, c).array());
    }

    // Compute weight gradients: dW/delta = delta * input^T
    Wgradient = deltaGates * timeStepsInputs;

    // Bias gradient is sum over all time steps
    bGradient = deltaGates.rowwise().sum();

    // Recurrent weight gradient: dU/delta = delta * output^T
    Ugradient = deltaGates * output.transpose();

    // Gradient w.r.t. inputs (for layers before this one)
    if(!settings.isFirstLayer){
        deltaInputs = W.transpose() * deltaGates;
    }
}

void LSTMLayer::updateWeights(double learningRate){
    W -= learningRate * Wgradient;
    U -= learningRate * Ugradient;
    b -= learningRate * bGradient;
}

void LSTMLayer::updateAdam(double learningRate, int iterationNum, double beta1, double beta2, double epsi) {
    W_MtForAdam = beta1 * W_MtForAdam.array() + (1 - beta1) * Wgradient.array();
    U_MtForAdam = beta1 * U_MtForAdam.array() + (1 - beta1) * Ugradient.array();
    b_MtForAdam = beta1 * b_MtForAdam.array() + (1 - beta1) * bGradient.array();

    W_VtForAdam = beta2 * W_VtForAdam.array() + (1 - beta2) * Wgradient.array() * Wgradient.array();
    U_VtForAdam = beta2 * U_VtForAdam.array() + (1 - beta2) * Ugradient.array() * Ugradient.array();
    b_VtForAdam = beta2 * b_VtForAdam.array() + (1 - beta2) * bGradient.array() * bGradient.array();

    double beta1Corr = 1.0 - std::pow(beta1, iterationNum);
    double beta2Corr = 1.0 - std::pow(beta2, iterationNum);

    W -= learningRate * (W_MtForAdam.array() / (beta1Corr * ((W_VtForAdam.array()/beta2Corr).sqrt() + epsi))).matrix();
    U -= learningRate * (U_MtForAdam.array() / (beta1Corr * ((U_VtForAdam.array()/beta2Corr).sqrt() + epsi))).matrix();
    b -= learningRate * (b_MtForAdam.array() / (beta1Corr * ((b_VtForAdam.array()/beta2Corr).sqrt() + epsi))).matrix();
}

void LSTMLayer::eraseMemory(){
    // Clear forward pass memory
    cellState.setZero();
    tanhCellState.setZero();
    output.setZero();
    timeStepsInputs.setZero();
    gatesActivations.setZero();
    gatesOutputs.setZero();
    forwardOutput.setZero();

    // Clear gradient accumulators (keep these if doing batch accumulation)
    // Wgradient.setZero();
    // Ugradient.setZero();
    // bGradient.setZero();

    // Clear backpropagation deltas
    deltaOutput.setZero();
    deltaCellState.setZero();
    deltaGates.setZero();
    nextLayerDelta.setZero();
    if(!settings.isFirstLayer){
        deltaInputs.setZero();
    }
}

void LSTMLayer::setDeltaFromNextLayer(const Eigen::VectorXd& delta){
    if (delta.size() % settings.outTS != 0)
        throw std::logic_error("[setDeltaFromNextLayer] unable to split vector into columns (non-divisible)");
    int numRows = delta.size() / settings.outTS;
    nextLayerDelta = Eigen::Map<const Eigen::MatrixXd>(delta.data(), numRows, settings.outTS);
}

void LSTMLayer::setDeltaFromNextLayer(const Eigen::MatrixXd& delta){
    // Handle both transposed and non-transposed inputs
    if (delta.cols() == settings.outTS && delta.rows() == settings.cells) {
        // Standard case: rows = cells, cols = outTS
        nextLayerDelta = delta;
    } else if (delta.rows() == settings.cells && delta.cols() >= settings.outTS) {
        // Take the last outTS columns if we have more time steps
        nextLayerDelta = delta.rightCols(settings.outTS);
    } else if (delta.cols() == settings.cells && delta.rows() == settings.outTS) {
        // Transposed case: transpose to match expected format
        nextLayerDelta = delta.transpose();
    } else {
        throw std::logic_error("[setDeltaFromNextLayer] dimension mismatch: delta " +
            std::to_string(delta.rows()) + "x" + std::to_string(delta.cols()) +
            " vs expected cells=" + std::to_string(settings.cells) + ", outTS=" + std::to_string(settings.outTS));
    }
}

void LSTMLayer::clearGradients(){
    Wgradient.setZero();
    Ugradient.setZero();
    bGradient.setZero();
}

Eigen::MatrixXd LSTMLayer::getDeltaInputs(){
    return deltaInputs;
}

Eigen::MatrixXd LSTMLayer::getForwardOutput(){
    return forwardOutput.transpose();
}

Eigen::VectorXd LSTMLayer::getForwardOutputVector(){
    // Return flattened output in row-major order (time steps first, then cells)
    return forwardOutput.reshaped<Eigen::RowMajor>();
}

Eigen::VectorXd LSTMLayer::getLastTimeStepOutput(){
    // Return only the last time step output (column) as a vector
    // forwardOutput is (cells x outTS), we want the last column
    if (forwardOutput.cols() == 0) {
        throw std::runtime_error("getLastTimeStepOutput called before forward pass");
    }
    return forwardOutput.col(forwardOutput.cols() - 1);
}

void LSTMLayer::accumulateGradients(const Eigen::MatrixXd& dW, const Eigen::MatrixXd& dU, const Eigen::VectorXd& db){
    Wgradient += dW;
    Ugradient += dU;
    bGradient += db;
}

void LSTMLayer::resetGradients(){
    Wgradient.setZero();
    Ugradient.setZero();
    bGradient.setZero();
}