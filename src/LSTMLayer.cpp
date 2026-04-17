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

Eigen::VectorXd LSTMLayer::sigmoidDerivVector(const Eigen::VectorXd& vec) {
    Eigen::VectorXd res = vec;
    res = res.array().unaryExpr([](double x) { return (1.0 / (1.0 + std::exp(-x))) * (1.0 - (1.0 / (1.0 + std::exp(-x)))); });
    return res;
}

Eigen::VectorXd LSTMLayer::tanhDerivVector(const Eigen::VectorXd& vec) {
    Eigen::VectorXd res = vec;
    res = res.array().unaryExpr([](double x) 
    { return 1.0 - ((2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0) * ((2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0); });
    return res;
}

void LSTMLayer::initLSTMLayer(const int numInputs,        
                const int numCells,
                const int numTimeSteps,
                std::string initType,
                int rngSeed,
                double minVal,
                double maxVal){

    const int numCells4gates = 4 * numCells;
    auto weightInit = strToWeightInit(initType);

    switch (weightInit) {
        
        // Random initialization between minVal and maxVal
        case weight_init_type::RANDOM: { 
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<> dist(minVal, maxVal);
            W = Eigen::MatrixXd(numCells4gates, numInputs);          
            W = W.unaryExpr([&](double) { return dist(gen); });
            U = Eigen::MatrixXd(numCells4gates, numCells);          
            U = U.unaryExpr([&](double) { return dist(gen); });
            break;
        }

        default:
            std::cerr << "[Error]: Unknown weight initialization type! Selected RANDOM initialization." << std::endl;
            // Select random initialization
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<> dist(minVal, maxVal);
            W = Eigen::MatrixXd(numCells4gates, numInputs);          
            W = W.unaryExpr([&](double) { return dist(gen); });
            U = Eigen::MatrixXd(numCells4gates, numCells);          
            U = U.unaryExpr([&](double) { return dist(gen); });
            break;
    }

    b = Eigen::VectorXd::Zero(numCells4gates);
    timeStepsInputs = Eigen::MatrixXd::Zero(numTimeSteps,numInputs);
    activationsWhole = Eigen::MatrixXd::Zero(numCells4gates,numTimeSteps);
    forgetGate = Eigen::MatrixXd::Zero(numCells,numTimeSteps);
    inputGate = Eigen::MatrixXd::Zero(numCells,numTimeSteps);
    candidateGate = Eigen::MatrixXd::Zero(numCells,numTimeSteps);
    outputGate = Eigen::MatrixXd::Zero(numCells,numTimeSteps);
    cellState = Eigen::MatrixXd::Zero(numCells,numTimeSteps);
    output = Eigen::MatrixXd::Zero(numCells,numTimeSteps);
    Wgradient = Eigen::MatrixXd::Zero(numCells4gates, numInputs);
    Ugradient = Eigen::MatrixXd::Zero(numCells, numCells);
    bGradient = Eigen::VectorXd::Zero(numCells4gates);

}

void LSTMLayer::setInputTSSegment(const Eigen::MatrixXd& inputSegment){
    if (inputSegment.cols() != timeStepsInputs.cols())
        throw std::invalid_argument("[setInputTSSegment] Number of inputs (cols) doesn't match the initialized");

    if (inputSegment.rows() != timeStepsInputs.rows())
        std::cerr << "[Warning][setInputTSSegment] Number of time-steps (rows) doesn't match the initialized"<<"\n";

    timeStepsInputs = inputSegment;
}

void LSTMLayer::calculateTimeSteps(){
    int numCells = outputGate.rows();

    //first TS
    activationsWhole.col(0) = W * timeStepsInputs.row(0).transpose() + b;
    forgetGate.col(0) = sigmoidVector(activationsWhole.col(0).segment(0, numCells));
    inputGate.col(0) = sigmoidVector(activationsWhole.col(0).segment(numCells, numCells));
    candidateGate.col(0) = tanhVector(activationsWhole.col(0).segment(2 * numCells, numCells));
    outputGate.col(0) = sigmoidVector(activationsWhole.col(0).segment(3 * numCells, numCells));
    cellState.col(0) += inputGate.col(0).cwiseProduct(candidateGate.col(0));
    output.col(0) = outputGate.col(0).cwiseProduct(tanhVector(cellState.col(0)));

    //rest of TS
    for(int i = 1 ; i < timeStepsInputs.rows() ; i++){
        activationsWhole.col(i) = W * timeStepsInputs.row(i).transpose() + b + U * output.col(i-1);
        forgetGate.col(i) = sigmoidVector(activationsWhole.col(i).segment(0, numCells));
        inputGate.col(i) = sigmoidVector(activationsWhole.col(i).segment(numCells, numCells));
        candidateGate.col(i) = tanhVector(activationsWhole.col(i).segment(2 * numCells, numCells));
        outputGate.col(i) = sigmoidVector(activationsWhole.col(i).segment(3 * numCells, numCells));
        cellState.col(i) = cellState.col(i-1).cwiseProduct(forgetGate.col(i)) + inputGate.col(i).cwiseProduct(candidateGate.col(i));
        output.col(i) = outputGate.col(i).cwiseProduct(tanhVector(cellState.col(i)));
        std::cout<<"timestep "<<i<<"\n\n"<<output<<"\n\n";
    }
}