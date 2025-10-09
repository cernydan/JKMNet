#include "JKMNet.hpp"

#include <random>
#include <iostream>

using namespace std;

/**
 * The constructor
 */
JKMNet::JKMNet(){

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
 * Helper function for schuffle
 */
static Eigen::MatrixXd shuffleMatrix(const Eigen::MatrixXd &mat, const std::vector<int> &perm) {
    Eigen::MatrixXd shuffled(mat.rows(), mat.cols());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        shuffled.row(static_cast<int>(i)) = mat.row(perm[i]);
    }
    return shuffled;
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
    mlp.initMLP(zeroIn, rngSeed);

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

/**
 * Train an MLP with batch Adam using calibrated matrices built from Data
 */
TrainingResult JKMNet::trainAdamBatchSplit(
  MLP &mlp,
  Data &data,
  const std::vector<unsigned> &mlpArchitecture,
  const std::vector<int> &numbersOfPastVarsValues,
  activ_func_type activationType,
  weight_init_type weightsInitType,
  int batchSize,
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
    if (maxIterations <= 0 || batchSize <= 0)
        throw std::invalid_argument("maxIterations and batchSize must be > 0");

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
    mlp.initMLP(zeroIn, rngSeed);

    // Run training with batchAdam method
    try {
        mlp.batchAdam(maxIterations, maxError, batchSize, learningRate, X, Y);
    } catch (const std::exception &ex) {
        std::cerr << "[trainAdamBatchSplit] training failed: " << ex.what() << "\n";
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

/**
 * K-fold validation (online Adam) 
 */
void JKMNet::KFold(
    Data &data,
    const std::vector<unsigned> &mlpArchitecture,
    const std::vector<int> &numbersOfPastVarsValues,
    activ_func_type activationType,
    weight_init_type weightsInitType,
    int kFolds,
    bool shuffle,
    bool largerPieceCalib,
    unsigned seed,
    int maxIterations,
    double maxError,
    double learningRate,
    int runsPerFold)
{

    // Basic check of architecture
    if (mlpArchitecture.empty())
        throw std::invalid_argument("mlpArchitecture must be non-empty");
    if (numbersOfPastVarsValues.empty())
        throw std::invalid_argument("numbersOfPastVarsValues must be non-empty");
    if (maxIterations <= 0)
        throw std::invalid_argument("maxIterations must be > 0");

    // Set MLP
    MLP testMlp;
    testMlp.setArchitecture(const_cast<std::vector<unsigned>&>(mlpArchitecture));
    std::vector<activ_func_type> activations(mlpArchitecture.size(), activationType);
    std::vector<weight_init_type> weightInits(mlpArchitecture.size(), weightsInitType);
    testMlp.setActivations(activations);
    testMlp.setWInitType(weightInits);

    std::vector<double> foldsMse;

    for(int foldIdx = 0; foldIdx < kFolds; foldIdx++){
        double meanFoldMse = 0.0; 
        auto [trainInps, trainOuts, validInps, validOuts] = data.makeKFoldMats(numbersOfPastVarsValues,
                                                                    static_cast<int>(mlpArchitecture.back()),
                                                                    kFolds,
                                                                    foldIdx,
                                                                    shuffle,
                                                                    largerPieceCalib,
                                                                    seed);
        for(int run = 0; run < runsPerFold; run++){
            // Initialize MLP
            int inputSize = static_cast<int>(trainInps.cols()); // each pattern is a row of inputs (flattened)
            Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(inputSize);
            testMlp.initMLP(zeroIn, seed);

            // Run training with onlineAdam method
            try {
                testMlp.onlineAdam(maxIterations, maxError, learningRate, trainInps, trainOuts);
            } catch (const std::exception &ex) {
                std::cerr << "[trainAdamOnlineSplit] training failed: " << ex.what() << "\n";
                throw;
            }

            // Validation
            testMlp.calculateOutputs(validInps);
            meanFoldMse += Metrics::mse(data.inverseTransformOutputs(validOuts),
                                        data.inverseTransformOutputs(testMlp.getOutputs()));
        }
        foldsMse.push_back(meanFoldMse / runsPerFold);
    }
    for(int fold = 0; fold < kFolds; fold++){
        std::cout<<"Fold "<<fold<<" validation mean MSE = "<<foldsMse[fold]<<"\n";
    }
}

/**
 * Setter for number of MLPs
 */
void JKMNet::setNmlps(unsigned nmlp){
    Nmlps = nmlp;
}

/**
 * Getter for number of MLPs
 */
unsigned JKMNet::getNmlps(){
    return Nmlps;
}

/**
 * Initialization of MLPs vector
 */
void JKMNet::init_mlps(MLP &mlp){
    //for()
}