#include "JKMNet.hpp"

#include <random>
#include <iostream>

using namespace std;

/**
 * The constructor
 */
JKMNet::JKMNet(){

}

JKMNet::JKMNet(const RunConfig& cfg, unsigned nthreads)
    : cfg_(cfg), nthreads_(nthreads) {}

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

void JKMNet::ensembleRun(MLP &mlp_){
        std::cout << "The number of threads is " << nthreads_ << std::endl;

    // ------------------------------------------------------
    // Load & preprocess data
    // ------------------------------------------------------
    std::unordered_set<std::string> idFilter;
    if (!cfg_.id.empty()) {
        std::vector<std::string> ids = parseStringList(cfg_.id);
        idFilter = std::unordered_set<std::string>(ids.begin(), ids.end());
    }

    std::cout << "-> Loading data..." << std::endl;
    data_.loadFilteredCSV(cfg_.data_file, idFilter, cfg_.columns, cfg_.timestamp, cfg_.id_col);
    std::cout << "-> Data loaded." << std::endl;

    std::cout << "-> Transforming data..." << std::endl;
    data_.setTransform(strToTransformType(cfg_.transform),
                       cfg_.transform_alpha,
                       cfg_.exclude_last_col_from_transform);
    data_.applyTransform();
    std::cout << "-> Data transformed." << std::endl;

    auto [X_train, Y_train, X_valid, Y_valid] = data_.makeMats(cfg_.input_numbers,
                                                              static_cast<int>(cfg_.mlp_architecture.back()),
                                                              cfg_.train_fraction,
                                                              cfg_.shuffle,
                                                              cfg_.seed);

    std::cout << "-> Data split into training and validation sets." << std::endl;

    std::vector<std::string> colNames;
    for (int c = 0; c < Y_train.cols(); ++c) {
        colNames.push_back("h" + std::to_string(c+1));
    }

    // Save real data
    Eigen::MatrixXd Y_true_calib_save = Y_train;
    Eigen::MatrixXd Y_true_valid_save = Y_valid;
    try {
        Y_true_calib_save = data_.inverseTransformOutputs(Y_true_calib_save);
        Y_true_valid_save = data_.inverseTransformOutputs(Y_true_valid_save);
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] inverseTransformOutputs failed (save GT): " << ex.what() << "\n";
    }
    data_.saveMatrixCsv(cfg_.real_calib, Y_true_calib_save, colNames);
    data_.saveMatrixCsv(cfg_.real_valid, Y_true_valid_save, colNames);
    std::cout << "-> Real calibration and validation data saved." << std::endl;

    // ------------------------------------------------------
    // Ensemble loop
    // ------------------------------------------------------
    for (int run = 0; run < cfg_.ensemble_runs; ++run) {
        std::string run_id = std::to_string(run+1);
        std::cout << "\n-------------------------------------------\n";
        std::cout << "Run " << run_id << " starting..." << std::endl;

        // Configure MLP
        mlp_.setArchitecture(cfg_.mlp_architecture);
        mlp_.setActivations(std::vector<activ_func_type>(cfg_.mlp_architecture.size(), strToActivation(cfg_.activation)));
        mlp_.setWInitType(std::vector<weight_init_type>(cfg_.mlp_architecture.size(), strToWeightInit(cfg_.weight_init)));

        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(std::accumulate(cfg_.input_numbers.begin(), cfg_.input_numbers.end(), 0));
        mlp_.initMLP(x0, cfg_.seed);

        // Save init weights
        mlp_.saveWeightsCsv(Metrics::addRunIdToFilename(cfg_.weights_csv_init, run_id));
        mlp_.weightsToVectorMlp();
        mlp_.saveWeightsVectorCsv(Metrics::addRunIdToFilename(cfg_.weights_vec_csv_init, run_id));
        mlp_.appendWeightsVectorCsv(cfg_.weights_vec_csv_init, run == 0);
        std::cout << "-> Initial weights saved." << std::endl;

        // Train
        std::cout << "-> Training starting..." << std::endl;
        Eigen::MatrixXd resultErrors;
        
        if (cfg_.trainer == "online") {
            mlp_.onlineAdam(
                cfg_.max_iterations, cfg_.max_error,
                cfg_.learning_rate,  X_train, Y_train
            );
        } else if (cfg_.trainer == "batch") {
            mlp_.batchAdam(
                cfg_.max_iterations, cfg_.max_error,
                cfg_.batch_size, cfg_.learning_rate,
                X_train, Y_train
            );
        } else if (cfg_.trainer == "online_epoch") {
            resultErrors = mlp_.onlineAdamEpochVal(
                X_train, Y_train, X_valid, Y_valid,
                cfg_.max_iterations, cfg_.learning_rate
            );
            Metrics::saveErrorsCsv(Metrics::addRunIdToFilename(cfg_.errors_csv, run_id), resultErrors);
        } else if (cfg_.trainer == "batch_epoch") {
            resultErrors = mlp_.batchAdamEpochVal(
                X_train, Y_train, X_valid, Y_valid,
                cfg_.batch_size, cfg_.max_iterations,
                cfg_.learning_rate
            );
            Metrics::saveErrorsCsv(Metrics::addRunIdToFilename(cfg_.errors_csv, run_id), resultErrors);
        } else {
            throw std::invalid_argument("Unknown trainer type: " + cfg_.trainer);
        }
        std::cout << "-> Training finished." << std::endl;

        // Evaluate calibration
        std::cout << "-> Evaluating calibration set..." << std::endl;
        mlp_.calculateOutputs(X_train);
        Eigen::MatrixXd Y_pred_calib = mlp_.getOutputs();
        Eigen::MatrixXd Y_true_calib = Y_train;
        try {
            Y_true_calib = data_.inverseTransformOutputs(Y_true_calib);
            Y_pred_calib = data_.inverseTransformOutputs(Y_pred_calib);
        } catch (...) {}
        Metrics::appendRunInfoCsv(cfg_.run_info,
            mlp_.getResult().iterations, mlp_.getResult().finalLoss,
            mlp_.getResult().converged, mlp_.getResult().time,
            cfg_.id + "_run" + run_id);
        Metrics::computeAndAppendFinalMetrics(Y_true_calib, Y_pred_calib, cfg_.metrics_cal, cfg_.id + "_run" + run_id);
        data_.saveMatrixCsv(Metrics::addRunIdToFilename(cfg_.pred_calib, run_id), Y_pred_calib, colNames);
        std::cout << "-> Calibration metrics and predictions saved." << std::endl;

        // Save final weights
        mlp_.saveWeightsCsv(Metrics::addRunIdToFilename(cfg_.weights_csv, run_id));
        mlp_.weightsToVectorMlp();
        mlp_.saveWeightsVectorCsv(Metrics::addRunIdToFilename(cfg_.weights_vec_csv, run_id));
        mlp_.appendWeightsVectorCsv(cfg_.weights_vec_csv, run == 0);
        std::cout << "-> Final weights saved." << std::endl;

        // Evaluate validation
        std::cout << "-> Evaluating validation set..." << std::endl;
        mlp_.calculateOutputs(X_valid);
        Eigen::MatrixXd Y_pred_valid = mlp_.getOutputs();
        Eigen::MatrixXd Y_true_valid = Y_valid;
        try {
            Y_true_valid = data_.inverseTransformOutputs(Y_true_valid);
            Y_pred_valid = data_.inverseTransformOutputs(Y_pred_valid);
        } catch (...) {}
        Metrics::computeAndAppendFinalMetrics(Y_true_valid, Y_pred_valid, cfg_.metrics_val, cfg_.id + "_run" + run_id);
        data_.saveMatrixCsv(Metrics::addRunIdToFilename(cfg_.pred_valid, run_id), Y_pred_valid, colNames);
        std::cout << "-> Validation metrics and predictions saved." << std::endl;

        std::cout << "Run " << run_id << " finished." << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
    }

    std::cout << "\n[I/O] Saved REAL CALIB data to: '" << cfg_.real_calib << "', and PRED CALIB data to '" << cfg_.pred_calib << "'\n";
    std::cout << "[I/O] Saved REAL VALID data to: '" << cfg_.real_valid << "', and PRED VALID data to '" << cfg_.pred_valid << "'\n";
    std::cout << "[I/O] Saved INIT weights vector to: '" << cfg_.weights_vec_csv_init << "'\n";
    std::cout << "[I/O] Saved FINAL weights vector to: '" << cfg_.weights_vec_csv << "'\n";
    std::cout << "[I/O] Saved RUN INFO to: '" << cfg_.run_info << "'\n";
    std::cout << "[I/O] Saved CALIB METRICS to: '" << cfg_.metrics_cal << "'\n";
    std::cout << "[I/O] Saved VALID METRICS to: '" << cfg_.metrics_val << "'\n";

    std::cout << "\n===========================================\n";
    std::cout << " Running Ensemble finished \n";
    std::cout << "===========================================\n";
}