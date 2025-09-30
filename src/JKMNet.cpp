#include "JKMNet.hpp"

#include <random>
#include <iostream>

using namespace std;

/**
 * The constructor
 */
JKMNet::JKMNet(){
    std::cout << "** Running the JKMNet **" << std::endl;
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
 * Train an MLP with online Adam without splitting (already done before training)
 */
TrainingResult JKMNet::trainAdamOnline(
    MLP &mlp,
    const Eigen::MatrixXd &X,
    const Eigen::MatrixXd &Y,
    int maxIterations,
    double maxError,
    double learningRate,
    bool shuffle,
    unsigned rngSeed)
{
    TrainingResult result;

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to trainAdamOnline");
    if (X.rows() != Y.rows())
        throw std::invalid_argument("X and Y row counts must match in trainAdamOnline");
    if (maxIterations <= 0)
        throw std::invalid_argument("maxIterations must be > 0");

    Eigen::MatrixXd Xtrain = X;
    Eigen::MatrixXd Ytrain = Y;

    // Shuffle rows if requested
    if (shuffle) {
        std::vector<int> perm(Xtrain.rows());
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937 gen(rngSeed);
        std::shuffle(perm.begin(), perm.end(), gen);

        auto shuffleMatrix = [](const Eigen::MatrixXd &mat, const std::vector<int> &perm) {
            Eigen::MatrixXd shuffled(mat.rows(), mat.cols());
            for (std::size_t i = 0; i < perm.size(); ++i) {
                shuffled.row(static_cast<int>(i)) = mat.row(perm[i]);
            }
            return shuffled;
        };

        Xtrain = shuffleMatrix(Xtrain, perm);
        Ytrain = shuffleMatrix(Ytrain, perm);
    }

    // Initialize MLP with zero input of correct size
    Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(Xtrain.cols());
    mlp.initMLP(zeroIn, rngSeed);

    // Run online Adam
    try {
        mlp.onlineAdam(maxIterations, maxError, learningRate, Xtrain, Ytrain);
    } catch (const std::exception &ex) {
        std::cerr << "[trainAdamOnline] training failed: " << ex.what() << "\n";
        throw;
    }

    // Compute final training loss
    Eigen::MatrixXd preds(Xtrain.rows(), Ytrain.cols());
    for (int r = 0; r < Xtrain.rows(); ++r) {
        preds.row(r) = mlp.runMLP(Xtrain.row(r).transpose()).transpose();
    }
    double mse = (preds - Ytrain).array().square().mean();

    result.finalLoss = mse;
    result.iterations = maxIterations;
    result.converged = (mse <= maxError);

    return result;
}

/**
 * Train an MLP with batch Adam without splitting (already done before training)
 */
TrainingResult JKMNet::trainAdamBatch(
    MLP &mlp,
    const Eigen::MatrixXd &X,
    const Eigen::MatrixXd &Y,
    int batchSize,
    int maxIterations,
    double maxError,
    double learningRate,
    bool shuffle,
    unsigned rngSeed) 
{
    TrainingResult result;

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to trainAdamBatch");
    if (X.rows() != Y.rows())
        throw std::invalid_argument("X and Y row counts must match in trainAdamBatch");
    if (maxIterations <= 0 || batchSize <= 0)
        throw std::invalid_argument("maxIterations and batchSize must be > 0");

    Eigen::MatrixXd Xtrain = X;
    Eigen::MatrixXd Ytrain = Y;

    // Shuffle rows if requested
    if (shuffle) {
        std::vector<int> perm(Xtrain.rows());
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937 gen(rngSeed);
        std::shuffle(perm.begin(), perm.end(), gen);

        Xtrain = shuffleMatrix(Xtrain, perm);
        Ytrain = shuffleMatrix(Ytrain, perm);
    }

    // Initialize MLP with zero input of correct size
    Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(Xtrain.cols());
    mlp.initMLP(zeroIn, rngSeed);

    // Run batch Adam
    try {
        mlp.batchAdam(maxIterations, maxError, batchSize, learningRate, Xtrain, Ytrain);
    } catch (const std::exception &ex) {
        std::cerr << "[trainAdamBatch] training failed: " << ex.what() << "\n";
        throw;
    }

    // Compute final training loss
    Eigen::MatrixXd preds(Xtrain.rows(), Ytrain.cols());
    for (int r = 0; r < Xtrain.rows(); ++r) {
        preds.row(r) = mlp.runMLP(Xtrain.row(r).transpose()).transpose();
    }
    double mse = (preds - Ytrain).array().square().mean();

    result.finalLoss = mse;
    result.iterations = maxIterations;
    result.converged = (mse <= maxError);

    return result;
}

Eigen::MatrixXd JKMNet::trainAdamOnlineEpochVal(
    MLP &mlp,
    const Eigen::MatrixXd &CalInp,
    const Eigen::MatrixXd &CalOut,
    const Eigen::MatrixXd &ValInp,
    const Eigen::MatrixXd &ValOut,
    int maxIterations,
    double maxError,
    double learningRate,
    bool shuffle,
    unsigned rngSeed)
{

    if (CalInp.rows() == 0 || CalOut.rows() == 0 || ValInp.rows() == 0 || ValOut.rows() == 0)
        throw std::invalid_argument("Empty data passed to trainAdamOnlineEpochVal");
    if (CalInp.rows() != CalOut.rows())
        throw std::invalid_argument("CalInp and CalOut row counts must match in trainAdamOnlineEpochVal");
    if (ValInp.rows() != ValOut.rows())
        throw std::invalid_argument("ValInp and ValOut row counts must match in trainAdamOnlineEpochVal");
    if (maxIterations <= 0)
        throw std::invalid_argument("maxIterations must be > 0");

    Eigen::MatrixXd Xtrain = CalInp;
    Eigen::MatrixXd Ytrain = CalOut;
    Eigen::MatrixXd Xval = ValInp;
    Eigen::MatrixXd Yval = ValOut;


    // Shuffle rows if requested
    if (shuffle) {
        std::vector<int> perm(Xtrain.rows());
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937 gen(rngSeed);
        std::shuffle(perm.begin(), perm.end(), gen);

        auto shuffleMatrix = [](const Eigen::MatrixXd &mat, const std::vector<int> &perm) {
            Eigen::MatrixXd shuffled(mat.rows(), mat.cols());
            for (std::size_t i = 0; i < perm.size(); ++i) {
                shuffled.row(static_cast<int>(i)) = mat.row(perm[i]);
            }
            return shuffled;
        };

        Xtrain = shuffleMatrix(Xtrain, perm);
        Ytrain = shuffleMatrix(Ytrain, perm);
    }

    // Initialize MLP with zero input of correct size
    Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(Xtrain.cols());
    mlp.initMLP(zeroIn, rngSeed);

    Eigen::MatrixXd predsCal;
    Eigen::MatrixXd predsVal;
    Eigen::MatrixXd resultErrors = Eigen::MatrixXd(maxIterations,2);

    for(int epoch = 0; epoch < maxIterations; epoch++){
        // Run online Adam
        try {
            mlp.onlineAdam(1, maxError, learningRate, Xtrain, Ytrain);
        } catch (const std::exception &ex) {
            std::cerr << "[trainAdamOnline] training failed: " << ex.what() << "\n";
            throw;
        }

        // Compute final training loss
        mlp.calculateOutputs(Xtrain);
        predsCal = mlp.getOutputs();

        mlp.calculateOutputs(Xval);
        predsVal = mlp.getOutputs();

        resultErrors(epoch,0) = Metrics::mse(Ytrain,predsCal);
        resultErrors(epoch,1) = Metrics::mse(Yval,predsVal);
    }
    return resultErrors;
}

Eigen::MatrixXd JKMNet::trainAdamBatchEpochVal(
    MLP &mlp,
    const Eigen::MatrixXd &CalInp,
    const Eigen::MatrixXd &CalOut,
    const Eigen::MatrixXd &ValInp,
    const Eigen::MatrixXd &ValOut,
    int batchSize,
    int maxIterations,
    double maxError,
    double learningRate,
    bool shuffle,
    unsigned rngSeed)
{

    if (CalInp.rows() == 0 || CalOut.rows() == 0 || ValInp.rows() == 0 || ValOut.rows() == 0)
        throw std::invalid_argument("Empty data passed to trainAdamOnlineEpochVal");
    if (CalInp.rows() != CalOut.rows())
        throw std::invalid_argument("CalInp and CalOut row counts must match in trainAdamOnlineEpochVal");
    if (ValInp.rows() != ValOut.rows())
        throw std::invalid_argument("ValInp and ValOut row counts must match in trainAdamOnlineEpochVal");
    if (maxIterations <= 0)
        throw std::invalid_argument("maxIterations must be > 0");

    Eigen::MatrixXd Xtrain = CalInp;
    Eigen::MatrixXd Ytrain = CalOut;
    Eigen::MatrixXd Xval = ValInp;
    Eigen::MatrixXd Yval = ValOut;


    // Shuffle rows if requested
    if (shuffle) {
        std::vector<int> perm(Xtrain.rows());
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937 gen(rngSeed);
        std::shuffle(perm.begin(), perm.end(), gen);

        auto shuffleMatrix = [](const Eigen::MatrixXd &mat, const std::vector<int> &perm) {
            Eigen::MatrixXd shuffled(mat.rows(), mat.cols());
            for (std::size_t i = 0; i < perm.size(); ++i) {
                shuffled.row(static_cast<int>(i)) = mat.row(perm[i]);
            }
            return shuffled;
        };

        Xtrain = shuffleMatrix(Xtrain, perm);
        Ytrain = shuffleMatrix(Ytrain, perm);
    }

    // Initialize MLP with zero input of correct size
    Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(Xtrain.cols());
    mlp.initMLP(zeroIn, rngSeed);

    Eigen::MatrixXd predsCal;
    Eigen::MatrixXd predsVal;
    Eigen::MatrixXd resultErrors = Eigen::MatrixXd(maxIterations,2);

    for(int epoch = 0; epoch < maxIterations; epoch++){
        // Run online Adam
        try {
            mlp.batchAdam(1, maxError, batchSize, learningRate, Xtrain, Ytrain);
        } catch (const std::exception &ex) {
            std::cerr << "[trainAdamOnline] training failed: " << ex.what() << "\n";
            throw;
        }

        // Compute final training loss
        mlp.calculateOutputs(Xtrain);
        predsCal = mlp.getOutputs();

        mlp.calculateOutputs(Xval);
        predsVal = mlp.getOutputs();

        resultErrors(epoch,0) = Metrics::mse(Ytrain,predsCal);
        resultErrors(epoch,1) = Metrics::mse(Yval,predsVal);
    }
    return resultErrors;
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