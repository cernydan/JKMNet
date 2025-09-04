#include "JKMNet.hpp"

#include <random>
#include <iostream>

using namespace std;

/**
 * The constructor
 */
JKMNet::JKMNet(){
    std::cout << "** Initialization of the net **" << std::endl;
}

/**
 * The destructor
 */
JKMNet::~JKMNet(){

}

/**
 * The copy constructor
 */
JKMNet::JKMNet(const JKMNet& other) {
   
}

/**
 * The assignment operator
 */
JKMNet& JKMNet::operator=(const JKMNet& other){
    if (this == &other) return *this;
  else {
        
  }
  return *this;

}

/**
 * Train an MLP with online Adam using calibrated matrices built from Data
 */
TrainingResult JKMNet::trainAdamOnlineSplit(
  MLP &mlp,
  Data &data,
  const std::vector<unsigned> &mlpArchitecture,
  const std::vector<int> &numbersOfPastVarsValues,
  activ_func_type activationType,
  weight_init_type weightsInitType,
  int maxIterations,
  double maxError,
  double learningRate,
  bool shuffle,
  unsigned rngSeed) 
  {
    TrainingResult result;

    // Basic check of architecture
    if (mlpArchitecture.empty())
        throw std::invalid_argument("mlpArchitecture must be non-empty");
    if (numbersOfPastVarsValues.empty())
        throw std::invalid_argument("numbersOfPastVarsValues must be non-empty");
    if (maxIterations <= 0)
        throw std::invalid_argument("maxIterations must be > 0");

    // Build calib mats inside Data
    data.makeCalibMatsSplit(numbersOfPastVarsValues, static_cast<int>(mlpArchitecture.back()));

    // Get calib matrices
    Eigen::MatrixXd X = data.getCalibInpsMat(); 
    Eigen::MatrixXd Y = data.getCalibOutsMat();

    // Shuffle rows
    std::vector<int> perm;
    if (shuffle) {
        perm = data.permutationVector(static_cast<int>(X.rows()));
        X = data.shuffleMatrix(X, perm);
        Y = data.shuffleMatrix(Y, perm);
    }

    // Configure MLP
    mlp.setArchitecture(const_cast<std::vector<unsigned>&>(mlpArchitecture));
    std::vector<activ_func_type> activations(mlpArchitecture.size(), activationType);
    std::vector<weight_init_type> weightInits(mlpArchitecture.size(), weightsInitType);
    mlp.setActivations(activations);
    mlp.setWInitType(weightInits);

    // Initialize MLP
    int inputSize = static_cast<int>(X.cols()); // each pattern is a row of inputs (flattened)
    Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(inputSize);
    mlp.initMLP(zeroIn);

    // Run training with onlineAdam method
    try {
        mlp.onlineAdam(maxIterations, maxError, learningRate, X, Y);
    } catch (const std::exception &ex) {
        std::cerr << "[trainAdamOnlineSplit] training failed: " << ex.what() << "\n";
        throw;
    }

    // Compute final loss on training set (simple MSE)
    Eigen::MatrixXd preds(X.rows(), Y.cols());
    for (int r = 0; r < X.rows(); ++r) {
        Eigen::VectorXd xcol = X.row(r).transpose();
        Eigen::VectorXd yhat = mlp.runMLP(xcol);
        preds.row(r) = yhat.transpose();
    }
    Eigen::MatrixXd diff = preds - Y;
    double mse = diff.array().square().mean();

    result.finalLoss = mse;
    result.iterations = maxIterations;
    result.converged = (mse <= maxError);

    return result;
}