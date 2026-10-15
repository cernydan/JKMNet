/**
 * @file JKMNet.cpp
 * @brief Implements JKMNet — the top-level orchestration class of the library,
 *        driving data loading, single-MLP Adam training, K-fold validation,
 *        and parallel ensemble runs (MLP-only and LSTM-MLP hybrids) from a RunConfig.
 */
#include "JKMNet.hpp"
#include "Metrics.hpp"
#include "ObjectiveFunctions.hpp"

#include <random>
#include <iostream>
#include "eigen-3.4/Eigen/Dense"
#include <filesystem>
#include <fstream>
#include <ostream> 

using namespace std;

/// Default constructor; leaves the object unconfigured.
JKMNet::JKMNet(){

}

/// Constructor storing the run configuration and the OpenMP thread count
/// used by the parallel ensemble loops.
JKMNet::JKMNet(const RunConfig& cfg, unsigned nthreads)
    : cfg_(cfg), nthreads_(nthreads) {
    }

/// Destructor.
JKMNet::~JKMNet(){

}

/// Copy constructor.
/// @todo Empty implementation — does not copy any state; verify intended semantics.
JKMNet::JKMNet(const JKMNet& other) {

}

/// Assignment operator.
/// @todo Guard against self-assignment only — no members are copied; verify intended semantics.
JKMNet& JKMNet::operator=(const JKMNet& other){
    if (this == &other) return *this;
  else {

  }
  return *this;

}

/**
 * Helper function for shuffling matrix rows by a permutation vector.
 * @note Superseded by Data::shuffleMatrix; kept commented out for reference.
 */
// static Eigen::MatrixXd shuffleMatrix(const Eigen::MatrixXd &mat, const std::vector<int> &perm) {
//     Eigen::MatrixXd shuffled(mat.rows(), mat.cols());
//     for (std::size_t i = 0; i < perm.size(); ++i) {
//         shuffled.row(static_cast<int>(i)) = mat.row(perm[i]);
//     }
//     return shuffled;
// }

/// Implements JKMNet::trainAdamOnlineSplit — single-MLP online (SGD-style) Adam training
/// on calibration matrices built on the fly from a Data object.
/**
 * Orchestration flow:
 * 1. Validate the architecture/hyperparameter inputs.
 * 2. Build the calibration input/output matrices inside @p data.
 * 3. Optionally shuffle pattern rows (same permutation for X and Y).
 * 4. Configure the MLP (architecture, activations, weight init) and attach the
 *    objective function from cfg_, falling back to LEVEL_SLOPE on failure.
 * 5. Initialize weights with input size taken from the matrix column count
 *    (each pattern is a flattened row of inputs).
 * 6. Train via MLP::onlineAdam, then re-run the forward pass over the training
 *    set and compute the final MSE to fill the TrainingResult.
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

    // Configure objective function for training (from cfg_)
    try {
        mlp.setObjectiveFunction(cfg_.objective_function);
        mlp.setObjectiveAlpha(cfg_.objective_alpha);
        mlp.setObjectiveAlpha2(cfg_.objective_alpha2);
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] Failed to set objective function: " << ex.what()
                  << ". Using default LEVEL_SLOPE.\n";
        mlp.setObjectiveFunction(objective_func_type::LEVEL_SLOPE);
    }

    // Initialize MLP
    int inputSize = static_cast<int>(X.cols()); // each pattern is a row of inputs (flattened)
    Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(inputSize);
    mlp.initMLP(zeroIn, rngSeed);

    // Train with the online Adam method; any failure is logged and rethrown
    try {
        mlp.onlineAdam(maxIterations, maxError, learningRate, X, Y);
    } catch (const std::exception &ex) {
        std::cerr << "[trainAdamOnlineSplit] training failed: " << ex.what() << "\n";
        throw;
    }

    // Compute the final loss on the training set as plain MSE:
    // run the forward pass for every pattern row and average the squared
    // error over all outputs and patterns.
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

/// Implements JKMNet::trainAdamBatchSplit — single-MLP mini-batch Adam training
/// on calibration matrices built on the fly from a Data object.
/**
 * Same orchestration as trainAdamOnlineSplit, except training runs through
 * MLP::batchAdam with the given @p batchSize instead of pattern-by-pattern
 * online updates.
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

    // Configure objective function for training (from cfg_)
    try {
        mlp.setObjectiveFunction(cfg_.objective_function);
        mlp.setObjectiveAlpha(cfg_.objective_alpha);
        mlp.setObjectiveAlpha2(cfg_.objective_alpha2);
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] Failed to set objective function: " << ex.what()
                  << ". Using default LEVEL_SLOPE.\n";
        mlp.setObjectiveFunction(objective_func_type::LEVEL_SLOPE);
    }

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

    // Compute the final loss on the training set as plain MSE
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

/// Implements JKMNet::KFold — K-fold cross-validation of an MLP trained with online Adam.
/**
 * For each fold, the train/validation matrices are rebuilt by
 * Data::makeKFoldMats (honoring the shuffle and largerPieceCalib options),
 * then the MLP is freshly initialized and trained runsPerFold times from the
 * same seed. Validation MSE is computed on inverse-transformed (original
 * scale) outputs, averaged across runs within the fold, and finally the mean
 * MSE of every fold is printed to stdout.
 *
 * @note The Metrics::mse is computed in the original data units, not in the
 *       transformed space used for training.
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
        // Build this fold's train/validation split; with largerPieceCalib the
        // calibration set takes the larger fraction of the fold boundaries
        auto [trainInps, trainOuts, validInps, validOuts] = data.makeKFoldMats(numbersOfPastVarsValues,
                                                                    static_cast<int>(mlpArchitecture.back()),
                                                                    kFolds,
                                                                    foldIdx,
                                                                    shuffle,
                                                                    largerPieceCalib,
                                                                    seed);
        for(int run = 0; run < runsPerFold; run++){
            // Re-initialize the MLP from the seed each run
            int inputSize = static_cast<int>(trainInps.cols()); // each pattern is a row of inputs (flattened)
            Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(inputSize);
            testMlp.initMLP(zeroIn, seed);

            // Train with online Adam; any failure is logged and rethrown
            try {
                testMlp.onlineAdam(maxIterations, maxError, learningRate, trainInps, trainOuts);
            } catch (const std::exception &ex) {
                std::cerr << "[trainAdamOnlineSplit] training failed: " << ex.what() << "\n";
                throw;
            }

            // Validate on the held-out fold; metrics are computed after
            // inverse-transforming back to the original data units
            testMlp.calculateOutputs(validInps);
            meanFoldMse += Metrics::mse(data.inverseTransformOutputs(validOuts),
                                        data.inverseTransformOutputs(testMlp.getOutputs()));
        }
        foldsMse.push_back(meanFoldMse / runsPerFold);
    }
    // Report the mean validation MSE per fold
    for(int fold = 0; fold < kFolds; fold++){
        std::cout<<"Fold "<<fold<<" validation mean MSE = "<<foldsMse[fold]<<"\n";
    }
}

/// Setter for the number of MLPs in the ensemble.
void JKMNet::setNmlps(unsigned nmlp){
    Nmlps = nmlp;
}

/// Getter for the number of MLPs in the ensemble.
unsigned JKMNet::getNmlps(){
    return Nmlps;
}

/// Implements JKMNet::init_mlps — creates Nmlps MLPs configured from cfg_
/// and initializes their weights in parallel.
/**
 * A prototype MLP is built first (architecture, activation and weight-init
 * types parsed from the config strings, objective function with a LEVEL_SLOPE
 * fallback), then copied into each slot of mlps_ inside an OpenMP loop.
 * The input size is the total number of lag values across all configured
 * input variables (sum of the lengths of cfg_.input_numbers entries).
 */
void JKMNet::init_mlps(){
    mlps_.clear();
    mlps_ = std::vector<MLP>(Nmlps);
    MLP setMlp;
    setMlp.setArchitecture(cfg_.mlp_architecture);
    setMlp.setActivations(strVecToActivationTypes(cfg_.activation));
    setMlp.setWInitType(std::vector<weight_init_type>(cfg_.mlp_architecture.size(), strToWeightInit(cfg_.weight_init)));
    try {
        setMlp.setObjectiveFunction(cfg_.objective_function);
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] Failed to set objective function: " << ex.what()
                  << ". Using default LEVEL_SLOPE.\n";
        setMlp.setObjectiveFunction(objective_func_type::LEVEL_SLOPE);
    }
    setMlp.setObjectiveAlpha(cfg_.objective_alpha);
    setMlp.setObjectiveAlpha2(cfg_.objective_alpha2);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(std::accumulate(cfg_.input_numbers.begin(),cfg_.input_numbers.end(),0,
                        [](int s, const std::vector<int>& v) {return s + static_cast<int>(v.size());}));
    #pragma omp parallel for num_threads(nthreads_)
    for(unsigned i = 0; i < Nmlps; i++){
        mlps_[i] = setMlp;
        mlps_[i].initMLP(x0, cfg_.seed);
    }
}

/// Implements JKMNet::init_mlpsForLSTM — same as init_mlps but the MLP input
/// size is the LSTM output size (number of LSTM cells), because these MLPs act
/// as readout layers on top of the recurrent part, not on raw time-series inputs.
void JKMNet::init_mlpsForLSTM(unsigned lstmOutputSize){
    mlps_.clear();
    mlps_ = std::vector<MLP>(Nmlps);
    MLP setMlp;
    setMlp.setArchitecture(cfg_.mlp_architecture);
    setMlp.setActivations(strVecToActivationTypes(cfg_.activation));
    setMlp.setWInitType(std::vector<weight_init_type>(cfg_.mlp_architecture.size(), strToWeightInit(cfg_.weight_init)));
    // Input size is the LSTM output size (lstm_cells), not the time-series flattened size
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(lstmOutputSize);
    #pragma omp parallel for num_threads(nthreads_)
    for(unsigned i = 0; i < Nmlps; i++){
        mlps_[i] = setMlp;
        mlps_[i].initMLP(x0, cfg_.seed);
    }
}

/// Implements JKMNet::ensembleRunMlpVector — the main config-driven MLP
/// ensemble workflow: load and preprocess data, then train and evaluate
/// Nmlps independent MLPs in parallel (OpenMP).
/**
 * Orchestration overview:
 * -@Phase 1 (data): The CSV is loaded with the optional station-id filter,
 *   split into calibration/validation matrices (makeMats), transformed with
 *   scalers fitted on calibration data only, and the ground-truth outputs are
 *   saved back in original units along with the pattern indices.
 * -@Phase 2 (ensemble loop, parallel over runs): each run writes its own log
 *   file, saves initial weights, trains with the trainer selected by
 *   cfg_.trainer and, for the *_EPOCH variants, additionally dumps per-epoch
 *   calibration/validation metric files. After training, calibration and
 *   validation predictions are produced, inverse-transformed to original
 *   units, and both the predictions and the per-metric rows (MSE, RMSE, PI,
 *   NS, KGE, PBIAS, RSR) are saved. Final weights are stored per run.
 *
 * @note In *EpochVal trainers the resultMetrics vector is indexed 0..13 as
 *   [MSE, RMSE, PI, NS, KGE, PBIAS, RSR] for calibration (0-6) and validation
 *   (7-13) — the order must match MLP::onlineAdamEpochVal and friends.
 */
void JKMNet::ensembleRunMlpVector(){
    std::cout << "The number of threads is " << nthreads_ << std::endl;

    // ------------------------------------------------------
    // Load & preprocess data (shared by all ensemble runs)
    // ------------------------------------------------------
    // Build the optional station-id filter set from the config string
    std::unordered_set<std::string> idFilter;
    if (!cfg_.id.empty()) {
        std::vector<std::string> ids = parseStringList(cfg_.id);
        idFilter = std::unordered_set<std::string>(ids.begin(), ids.end());
    }

    std::cout << "-> Loading data..." << std::endl;
    data_.loadFilteredCSV(cfg_.data_file, idFilter, cfg_.columns, cfg_.timestamp, cfg_.id_col);
    std::cout << "-> Data loaded." << std::endl;

    // Build flattened (inputs, targets) pattern matrices and split them into
    // calibration and validation parts; calIdxForUnshuffle allows restoring
    // the original chronological order of the shuffled calibration patterns
    auto [X_train, Y_train, X_valid, Y_valid, pat_indices, calIdxForUnshuffle] = data_.makeMats(cfg_.input_numbers,
                                                                                static_cast<int>(cfg_.mlp_architecture.back()),
                                                                                cfg_.train_fraction,
                                                                                cfg_.shuffle,
                                                                                cfg_.seed);

    std::cout << "-> Data split into training and validation sets." << std::endl;

    std::cout << "-> Transforming data..." << std::endl;

    // Fit the transformation scalers on calibration data only...
    auto [X_train_trans, Y_train_trans, train_scalers] = data_.transformMatsGetScalers(X_train, Y_train, strVecToTransformTypes(cfg_.transform),
                                                                            cfg_.transform_alpha, cfg_.exclude_last_col_from_transform,
                                                                            cfg_.input_numbers);

    // ...and apply the same scalers to validation to avoid data leakage
    auto [X_valid_trans, Y_valid_trans] = data_.transformMatsApplyScalers(X_valid, Y_valid, strVecToTransformTypes(cfg_.transform),
                                                                         cfg_.exclude_last_col_from_transform,
                                                                            cfg_.input_numbers, train_scalers);

    std::cout << "-> Data transformed." << std::endl;

    // Output column labels "h1", "h2", ... used as CSV headers everywhere
    std::vector<std::string> colNames;
    for (int c = 0; c < Y_train.cols(); ++c) {
        colNames.push_back("h" + std::to_string(c+1));
    }

    // Save ground-truth outputs in original units (inverse-transformed,
    // calibration set unshuffled back to chronological order)
    Eigen::MatrixXd Y_true_calib_save = data_.unshuffleMatrix(Y_train_trans, calIdxForUnshuffle);
    Eigen::MatrixXd Y_true_valid_save = Y_valid_trans;
    try {
        Y_true_calib_save = data_.inverseTransformOutputs(Y_true_calib_save, train_scalers.back());
        Y_true_valid_save = data_.inverseTransformOutputs(Y_true_valid_save, train_scalers.back());
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] inverseTransformOutputs failed (save GT): " << ex.what() << "\n";
    }
    data_.saveMatrixCsv(cfg_.real_calib, Y_true_calib_save, colNames);
    data_.saveMatrixCsv(cfg_.real_valid, Y_true_valid_save, colNames);
    data_.saveVector(pat_indices, cfg_.pattern_indices);
    std::cout << "-> Real calibration and validation data saved." << std::endl;

    // Create and initialize the ensemble of MLPs from the config
    setNmlps(cfg_.ensemble_runs);
    init_mlps();

    std::cout << "-> Ensemble run starting..." << std::endl;
    // ------------------------------------------------------
    // Ensemble loop — one independent MLP per iteration, run in parallel.
    // Each run keeps its own log file and per-run output files, so no shared
    // state is written inside the parallel region.
    // ------------------------------------------------------
    #pragma omp parallel for num_threads(nthreads_)
    for (unsigned run = 0; run < mlps_.size() ; ++run) {
        std::string run_id = std::to_string(run+1);  // 1-based run id as a string (for filenames)
        unsigned run_id_integer = run + 1; // 1-based run id as an integer (for the log)

        // Make sure the log directory exists
        if (!std::filesystem::exists(cfg_.log_dir)) {
            std::filesystem::create_directories(cfg_.log_dir);
        }
        // Open the per-run log file
        std::string filename = cfg_.log_dir + "log_run" + run_id + ".log";
        std::ofstream logFile(filename);

        // Log all configuration settings for this run
        data_.logRunSettings(logFile, cfg_, run_id_integer);

        logFile << "Run " << run_id << " starting...\n";

        // Persist the initial weights (layer-wise CSV and flat vector CSV)
        mlps_[run].saveWeightsCsv(Metrics::addRunIdToFilename(cfg_.weights_csv_init, run_id));
        mlps_[run].weightsToVectorMlp();
        mlps_[run].saveWeightsVectorCsv(Metrics::addRunIdToFilename(cfg_.weights_vec_csv_init, run_id));
        //mlps_[run].appendWeightsVectorCsv(cfg_.weights_vec_csv_init, run == 0);  // all init weights in one file
        logFile << "-> Initial weights saved.\n";

        // Train
        logFile << "-> Training starting...\n";

        std::vector<Eigen::MatrixXd> resultMetrics;
        TrainerType trainerType = strToTrainerType(cfg_.trainer);
        int runIndex = run + 1;

        // Per-epoch metric filenames for the *_EPOCH trainer variants:
        // one file per (split, metric) pair for this run index
        std::string mseCalFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "MSE_epochs");
        std::string rmseCalFile  = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RMSE_epochs");
        std::string piCalFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PI_epochs");
        std::string nsCalFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "NS_epochs");
        std::string kgeCalFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "KGE_epochs");
        std::string pbiasCalFile = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PBIAS_epochs");
        std::string rsrCalFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RSR_epochs");
        std::string mseValFile   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "MSE_epochs");
        std::string rmseValFile  = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RMSE_epochs");
        std::string piValFile    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PI_epochs");
        std::string nsValFile    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "NS_epochs");
        std::string kgeValFile   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "KGE_epochs");
        std::string pbiasValFile = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PBIAS_epochs");
        std::string rsrValFile   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RSR_epochs");

        // Dispatch on the trainer type selected in the config; the *_EPOCH
        // variants return per-epoch metric series which are saved to CSV
        switch (trainerType) {
            case TrainerType::ONLINE_ADAM:
                mlps_[run].onlineAdam(
                    cfg_.max_iterations, cfg_.max_error,
                    cfg_.learning_rate,  X_train_trans, Y_train_trans
                );
                break;

            case TrainerType::BATCH_ADAM:
                mlps_[run].batchAdam(
                    cfg_.max_iterations, cfg_.max_error,
                    cfg_.batch_size, cfg_.learning_rate,
                    X_train_trans, Y_train_trans
                );
                break;

                
            case TrainerType::ONLINE_BP:
                mlps_[run].onlineBP(
                    cfg_.max_iterations, cfg_.max_error,
                    cfg_.learning_rate,  X_train_trans, Y_train_trans
                );
                break;

            case TrainerType::BATCH_BP:
                mlps_[run].batchBP(
                    cfg_.max_iterations, cfg_.max_error,
                    cfg_.batch_size, cfg_.learning_rate,
                    X_train_trans, Y_train_trans
                );
                break;

            case TrainerType::ONLINE_BP_PENALIZE:
                mlps_[run].onlinePenalizeBP(
                    cfg_.max_iterations, cfg_.max_error,
                    cfg_.learning_rate,  X_train_trans, Y_train_trans, 0.0001  //Add lambda choice to config
                );
                break;

            case TrainerType::ONLINE_BP_MOMENTUM:
                mlps_[run].onlineMomentumBP(
                    cfg_.max_iterations, cfg_.max_error,
                    cfg_.learning_rate,  X_train_trans, Y_train_trans,0.9 //Add moment choice to config
                );
                break;

            case TrainerType::ONLINE_ADAM_EPOCH:
                resultMetrics = mlps_[run].onlineAdamEpochVal(
                    X_train_trans, Y_train_trans, X_valid_trans, Y_valid_trans,
                    cfg_.max_iterations, cfg_.max_error, cfg_.learning_rate, cfg_.max_metrics_step
                );
                Metrics::saveMetricsCsv(mseCalFile, resultMetrics[0]);
                Metrics::saveMetricsCsv(rmseCalFile, resultMetrics[1]);
                Metrics::saveMetricsCsv(piCalFile, resultMetrics[2]);
                Metrics::saveMetricsCsv(nsCalFile, resultMetrics[3]);
                Metrics::saveMetricsCsv(kgeCalFile, resultMetrics[4]);
                Metrics::saveMetricsCsv(pbiasCalFile, resultMetrics[5]);
                Metrics::saveMetricsCsv(rsrCalFile, resultMetrics[6]);
                Metrics::saveMetricsCsv(mseValFile, resultMetrics[7]);
                Metrics::saveMetricsCsv(rmseValFile, resultMetrics[8]);
                Metrics::saveMetricsCsv(piValFile, resultMetrics[9]);
                Metrics::saveMetricsCsv(nsValFile, resultMetrics[10]);
                Metrics::saveMetricsCsv(kgeValFile, resultMetrics[11]);
                Metrics::saveMetricsCsv(pbiasValFile, resultMetrics[12]);
                Metrics::saveMetricsCsv(rsrValFile, resultMetrics[13]);
                break;

            case TrainerType::BATCH_ADAM_EPOCH:
                resultMetrics = mlps_[run].batchAdamEpochVal(
                    X_train_trans, Y_train_trans, X_valid_trans, Y_valid_trans,
                    cfg_.batch_size, cfg_.max_iterations, cfg_.max_error,
                    cfg_.learning_rate, cfg_.max_metrics_step
                );
                Metrics::saveMetricsCsv(mseCalFile, resultMetrics[0]);
                Metrics::saveMetricsCsv(rmseCalFile, resultMetrics[1]);
                Metrics::saveMetricsCsv(piCalFile, resultMetrics[2]);
                Metrics::saveMetricsCsv(nsCalFile, resultMetrics[3]);
                Metrics::saveMetricsCsv(kgeCalFile, resultMetrics[4]);
                Metrics::saveMetricsCsv(pbiasCalFile, resultMetrics[5]);
                Metrics::saveMetricsCsv(rsrCalFile, resultMetrics[6]);
                Metrics::saveMetricsCsv(mseValFile, resultMetrics[7]);
                Metrics::saveMetricsCsv(rmseValFile, resultMetrics[8]);
                Metrics::saveMetricsCsv(piValFile, resultMetrics[9]);
                Metrics::saveMetricsCsv(nsValFile, resultMetrics[10]);
                Metrics::saveMetricsCsv(kgeValFile, resultMetrics[11]);
                Metrics::saveMetricsCsv(pbiasValFile, resultMetrics[12]);
                Metrics::saveMetricsCsv(rsrValFile, resultMetrics[13]);
                break;

            case TrainerType::ONLINE_BP_EPOCH:
                resultMetrics = mlps_[run].onlineBpEpochVal(
                    X_train_trans, Y_train_trans, X_valid_trans, Y_valid_trans,
                    cfg_.max_iterations, cfg_.max_error, cfg_.learning_rate, cfg_.max_metrics_step
                );
                Metrics::saveMetricsCsv(mseCalFile, resultMetrics[0]);
                Metrics::saveMetricsCsv(rmseCalFile, resultMetrics[1]);
                Metrics::saveMetricsCsv(piCalFile, resultMetrics[2]);
                Metrics::saveMetricsCsv(nsCalFile, resultMetrics[3]);
                Metrics::saveMetricsCsv(kgeCalFile, resultMetrics[4]);
                Metrics::saveMetricsCsv(pbiasCalFile, resultMetrics[5]);
                Metrics::saveMetricsCsv(rsrCalFile, resultMetrics[6]);
                Metrics::saveMetricsCsv(mseValFile, resultMetrics[7]);
                Metrics::saveMetricsCsv(rmseValFile, resultMetrics[8]);
                Metrics::saveMetricsCsv(piValFile, resultMetrics[9]);
                Metrics::saveMetricsCsv(nsValFile, resultMetrics[10]);
                Metrics::saveMetricsCsv(kgeValFile, resultMetrics[11]);
                Metrics::saveMetricsCsv(pbiasValFile, resultMetrics[12]);
                Metrics::saveMetricsCsv(rsrValFile, resultMetrics[13]);
                break;

            case TrainerType::BATCH_BP_EPOCH:
                resultMetrics = mlps_[run].batchBpEpochVal(
                    X_train_trans, Y_train_trans, X_valid_trans, Y_valid_trans,
                    cfg_.batch_size, cfg_.max_iterations, cfg_.max_error,
                    cfg_.learning_rate, cfg_.max_metrics_step
                );
                Metrics::saveMetricsCsv(mseCalFile, resultMetrics[0]);
                Metrics::saveMetricsCsv(rmseCalFile, resultMetrics[1]);
                Metrics::saveMetricsCsv(piCalFile, resultMetrics[2]);
                Metrics::saveMetricsCsv(nsCalFile, resultMetrics[3]);
                Metrics::saveMetricsCsv(kgeCalFile, resultMetrics[4]);
                Metrics::saveMetricsCsv(pbiasCalFile, resultMetrics[5]);
                Metrics::saveMetricsCsv(rsrCalFile, resultMetrics[6]);
                Metrics::saveMetricsCsv(mseValFile, resultMetrics[7]);
                Metrics::saveMetricsCsv(rmseValFile, resultMetrics[8]);
                Metrics::saveMetricsCsv(piValFile, resultMetrics[9]);
                Metrics::saveMetricsCsv(nsValFile, resultMetrics[10]);
                Metrics::saveMetricsCsv(kgeValFile, resultMetrics[11]);
                Metrics::saveMetricsCsv(pbiasValFile, resultMetrics[12]);
                Metrics::saveMetricsCsv(rsrValFile, resultMetrics[13]);
                break;

            case TrainerType::ONLINE_BP_PENALIZE_EPOCH:
                resultMetrics = mlps_[run].onlinePenalizeBpEpochVal(
                    X_train_trans, Y_train_trans, X_valid_trans, Y_valid_trans,
                    cfg_.max_iterations, cfg_.max_error, cfg_.learning_rate, cfg_.max_metrics_step, 0.0001   //Add lambda choice to config
                );
                Metrics::saveMetricsCsv(mseCalFile, resultMetrics[0]);
                Metrics::saveMetricsCsv(rmseCalFile, resultMetrics[1]);
                Metrics::saveMetricsCsv(piCalFile, resultMetrics[2]);
                Metrics::saveMetricsCsv(nsCalFile, resultMetrics[3]);
                Metrics::saveMetricsCsv(kgeCalFile, resultMetrics[4]);
                Metrics::saveMetricsCsv(pbiasCalFile, resultMetrics[5]);
                Metrics::saveMetricsCsv(rsrCalFile, resultMetrics[6]);
                Metrics::saveMetricsCsv(mseValFile, resultMetrics[7]);
                Metrics::saveMetricsCsv(rmseValFile, resultMetrics[8]);
                Metrics::saveMetricsCsv(piValFile, resultMetrics[9]);
                Metrics::saveMetricsCsv(nsValFile, resultMetrics[10]);
                Metrics::saveMetricsCsv(kgeValFile, resultMetrics[11]);
                Metrics::saveMetricsCsv(pbiasValFile, resultMetrics[12]);
                Metrics::saveMetricsCsv(rsrValFile, resultMetrics[13]);
                break;

            case TrainerType::ONLINE_BP_MOMENTUM_EPOCH:
                resultMetrics = mlps_[run].onlineMomentumBpEpochVal(
                    X_train_trans, Y_train_trans, X_valid_trans, Y_valid_trans,
                    cfg_.max_iterations, cfg_.max_error, cfg_.learning_rate, cfg_.max_metrics_step, 0.9   //Add moment choice to config
                );
                Metrics::saveMetricsCsv(mseCalFile, resultMetrics[0]);
                Metrics::saveMetricsCsv(rmseCalFile, resultMetrics[1]);
                Metrics::saveMetricsCsv(piCalFile, resultMetrics[2]);
                Metrics::saveMetricsCsv(nsCalFile, resultMetrics[3]);
                Metrics::saveMetricsCsv(kgeCalFile, resultMetrics[4]);
                Metrics::saveMetricsCsv(pbiasCalFile, resultMetrics[5]);
                Metrics::saveMetricsCsv(rsrCalFile, resultMetrics[6]);
                Metrics::saveMetricsCsv(mseValFile, resultMetrics[7]);
                Metrics::saveMetricsCsv(rmseValFile, resultMetrics[8]);
                Metrics::saveMetricsCsv(piValFile, resultMetrics[9]);
                Metrics::saveMetricsCsv(nsValFile, resultMetrics[10]);
                Metrics::saveMetricsCsv(kgeValFile, resultMetrics[11]);
                Metrics::saveMetricsCsv(pbiasValFile, resultMetrics[12]);
                Metrics::saveMetricsCsv(rsrValFile, resultMetrics[13]);
                break;

            default:
                throw std::invalid_argument("Unknown trainer type");
        }


        logFile << "-> Training finished.\n";

        // Evaluate calibration: forward predictions on the training set,
        // inverse-transformed back to original units for metric computation
        logFile << "-> Evaluating calibration set...\n";
        mlps_[run].calculateOutputs(X_train_trans);
        Eigen::MatrixXd Y_pred_calib = mlps_[run].getOutputs();
        Eigen::MatrixXd Y_true_calib = Y_train_trans;
        try {
            Y_true_calib = data_.inverseTransformOutputs(Y_true_calib, train_scalers.back());
            Y_pred_calib = data_.inverseTransformOutputs(Y_pred_calib, train_scalers.back());
        } catch (...) {}

        // Compute all metrics per output column (one row per run,
        // written into per-metric CSV files)
        std::vector<double> mseVals, rmseVals, piVals, nsVals, kgeVals, pbiasVals, rsrVals;

        for (int c = 0; c < Y_true_calib.cols(); ++c) {
            mseVals.push_back(Metrics::mse(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            rmseVals.push_back(Metrics::rmse(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            piVals.push_back(Metrics::pi(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            nsVals.push_back(Metrics::ns(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            kgeVals.push_back(Metrics::kge(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            pbiasVals.push_back(Metrics::pbias(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            rsrVals.push_back(Metrics::rsr(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
        }
        // per-metric filenames for this run
        std::string mseFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "MSE");
        std::string rmseFile  = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RMSE");
        std::string piFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PI");
        std::string nsFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "NS");
        std::string kgeFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "KGE");
        std::string pbiasFile = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PBIAS");
        std::string rsrFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RSR");
        // always write header once per file (this is one run)
        Metrics::saveMetricRow(mseFile,   colNames, mseVals,   true);
        Metrics::saveMetricRow(rmseFile,  colNames, rmseVals,  true);
        Metrics::saveMetricRow(piFile,    colNames, piVals,    true);
        Metrics::saveMetricRow(nsFile,    colNames, nsVals,    true);
        Metrics::saveMetricRow(kgeFile,   colNames, kgeVals,   true);
        Metrics::saveMetricRow(pbiasFile, colNames, pbiasVals, true);
        Metrics::saveMetricRow(rsrFile,   colNames, rsrVals,   true);


        data_.saveMatrixCsv(Metrics::addRunIdToFilename(cfg_.pred_calib, run_id), 
                            data_.unshuffleMatrix(Y_pred_calib, calIdxForUnshuffle), 
                            colNames);

        logFile << "-> Calibration metrics and predictions saved.\n";

        // Save final weights (layer-wise CSV and flat vector CSV)
        mlps_[run].saveWeightsCsv(Metrics::addRunIdToFilename(cfg_.weights_csv, run_id));
        mlps_[run].weightsToVectorMlp();
        mlps_[run].saveWeightsVectorCsv(Metrics::addRunIdToFilename(cfg_.weights_vec_csv, run_id));
        // mlps_[run].appendWeightsVectorCsv(cfg_.weights_vec_csv, run == 0);  // all final weights in one file
        logFile << "-> Final weights saved.\n";

        // TODO: prediction should be a separate method
        // Evaluate validation on the held-out split, inverse-transformed to
        // original units, mirroring the calibration evaluation above
        logFile << "-> Evaluating validation set...\n";
        mlps_[run].calculateOutputs(X_valid_trans);
        Eigen::MatrixXd Y_pred_valid = mlps_[run].getOutputs();
        Eigen::MatrixXd Y_true_valid = Y_valid_trans;
        try {
            Y_true_valid = data_.inverseTransformOutputs(Y_true_valid, train_scalers.back());
            Y_pred_valid = data_.inverseTransformOutputs(Y_pred_valid, train_scalers.back());
        } catch (...) {}

        // Compute all metrics per output column (one row per run,
        // written into per-metric CSV files)
        std::vector<double> mseValsV, rmseValsV, piValsV, nsValsV, kgeValsV, pbiasValsV, rsrValsV;

        for (int c = 0; c < Y_true_valid.cols(); ++c) {
            mseValsV.push_back(Metrics::mse(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            rmseValsV.push_back(Metrics::rmse(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            piValsV.push_back(Metrics::pi(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            nsValsV.push_back(Metrics::ns(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            kgeValsV.push_back(Metrics::kge(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            pbiasValsV.push_back(Metrics::pbias(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            rsrValsV.push_back(Metrics::rsr(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
        }

        std::string mseFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "MSE");
        std::string rmseFileV  = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RMSE");
        std::string piFileV    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PI");
        std::string nsFileV    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "NS");
        std::string kgeFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "KGE");
        std::string pbiasFileV = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PBIAS");
        std::string rsrFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RSR");

        Metrics::saveMetricRow(mseFileV,   colNames, mseValsV,   true);
        Metrics::saveMetricRow(rmseFileV,  colNames, rmseValsV,  true);
        Metrics::saveMetricRow(piFileV,    colNames, piValsV,    true);
        Metrics::saveMetricRow(nsFileV,    colNames, nsValsV,    true);
        Metrics::saveMetricRow(kgeFileV,   colNames, kgeValsV,   true);
        Metrics::saveMetricRow(pbiasFileV, colNames, pbiasValsV, true);
        Metrics::saveMetricRow(rsrFileV,   colNames, rsrValsV,   true);

        
        data_.saveMatrixCsv(Metrics::addRunIdToFilename(cfg_.pred_valid, run_id), Y_pred_valid, colNames);
        logFile << "-> Validation metrics and predictions saved.\n";

        logFile << "Run " << run_id << " finished.\n";
        logFile << "-------------------------------------------\n";

    }

    std::cout << "-> Ensemble run finished." << std::endl;

    std::cout << "\n===========================================\n";
    std::cout << " Running Ensemble finished \n";
    std::cout << "===========================================\n";
}

/// Implements JKMNet::predictFromSavedWeights — reconstructs an MLP from the
/// config, loads previously saved weights, and produces predictions on the
/// validation split of the data.
/**
 * The data pipeline mirrors the training path so the network sees the same
 * transformations: CSV load with the id filter, transformation fitted on the
 * whole dataset, and the makeMats split (only the validation part is used).
 * Predictions are inverse-transformed, saved to cfg_.pred_valid, and per-column
 * RMSE is written to cfg_.metrics_val.
 * @throw std::runtime_error If the weights file cannot be loaded.
 */
void JKMNet::predictFromSavedWeights(const std::string &weightsPath)
{
    std::cout << " Prediction from saved weights\n";

    // Load and transform data
    std::unordered_set<std::string> idFilter;
    if (!cfg_.id.empty()) {
        auto ids = parseStringList(cfg_.id);
        idFilter = std::unordered_set<std::string>(ids.begin(), ids.end());
    }

    std::cout << "-> Loading data...\n";
    data_.loadFilteredCSV(cfg_.data_file, idFilter, cfg_.columns, cfg_.timestamp, cfg_.id_col);
    std::cout << "-> Data loaded.\n";

    std::cout << "-> Transforming data...\n";
    data_.setTransform(strVecToTransformTypes(cfg_.transform),
                       cfg_.transform_alpha,
                       cfg_.exclude_last_col_from_transform);
    data_.applyTransform();
    std::cout << "-> Data transformed.\n";

    // Reproduce the same calibration/validation split used during training;
    // prediction runs on the validation part only
    auto [X_train, Y_train, X_valid, Y_valid, pat_indices, calIdxForUnshuffle] = data_.makeMats(cfg_.input_numbers,
                                                                                static_cast<int>(cfg_.mlp_architecture.back()),
                                                                                cfg_.train_fraction,
                                                                                cfg_.shuffle,
                                                                                cfg_.seed);
    Eigen::MatrixXd X_pred = X_valid;
    Eigen::MatrixXd Y_true = Y_valid;

    std::vector<std::string> colNames;
    for (int c = 0; c < Y_true.cols(); ++c) {
        colNames.push_back("h" + std::to_string(c+1));
    }

    // Reconstruct the network
    std::cout << "-> Initializing MLP...\n";
    MLP mlp;
    mlp.setArchitecture(cfg_.mlp_architecture);

    mlp.setActivations(strVecToActivationTypes(cfg_.activation));

    std::vector<weight_init_type> wInits(cfg_.mlp_architecture.size(), strToWeightInit(cfg_.weight_init));
    mlp.setWInitType(wInits);

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(cfg_.mlp_architecture.front());
    mlp.initMLP(x0, cfg_.seed);

    std::cout << "-> Loading weights from: " << weightsPath << "\n";
    if (!mlp.loadWeightsCsv(weightsPath, "")) {
        throw std::runtime_error("Failed to load weights file: " + weightsPath);
    }

    // Predict
    std::cout << "-> Calculating outputs...\n";
    mlp.calculateOutputs(X_pred);
    Eigen::MatrixXd Y_pred = mlp.getOutputs();

    try {
        Y_true = data_.inverseTransformOutputs(Y_true);
        Y_pred = data_.inverseTransformOutputs(Y_pred);
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] inverseTransformOutputs failed: " << ex.what() << "\n";
    }

    // Save results
    std::cout << "-> Saving predictions...\n";
    data_.saveMatrixCsv(cfg_.pred_valid, Y_pred, colNames);

    // Compute metrics
    std::vector<double> rmseVals;
    for (int c = 0; c < Y_true.cols(); ++c) {
        rmseVals.push_back(Metrics::rmse(Y_true.col(c).eval(), Y_pred.col(c).eval()));
    }
    Metrics::saveMetricRow(cfg_.metrics_val, colNames, rmseVals, true);

    std::cout << "-> Prediction completed successfully.\n";
}



/// Implements JKMNet::ensembleLstmFirstTest — an early LSTM-MLP hybrid ensemble
/// in which a single LSTMLayer consumes past observations and an output MLP
/// maps the LSTM forward output to the target horizon.
/**
 * Training alternates per sample: the LSTM runs its time steps, the MLP
 * computes a batch gradient from the LSTM output, the gradient is handed back
 * to the LSTM (setDeltaFromNextLayer), and both parts are updated with Adam.
 * After each sample the LSTM memory is reset (eraseMemory) so samples are
 * treated independently. Calibration/validation evaluation, inverse
 * transformation, and metric saving follow the same flow as in
 * ensembleRunMlpVector.
 *
 * @note The output MLPs are initialized with init_mlps(), i.e. with a
 *       time-series-flattened input size, even though the LSTM output size
 *       (cfg_.lstm_cells) is what actually feeds them — see init_mlpsForLSTM
 *       for the matching variant.
 * @todo check semantics: init_mlps() vs init_mlpsForLSTM(cfg_.lstm_cells) mismatch.
 */
void JKMNet::ensembleLstmFirstTest(){
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
    data_.setTransform(strVecToTransformTypes(cfg_.transform),
                       cfg_.transform_alpha,
                       cfg_.exclude_last_col_from_transform);
    data_.applyTransform();
    std::cout << "-> Data transformed." << std::endl;

    // Build the LSTM windowed dataset: one input segment per pattern of
    // lstm_past_time_steps rows, target = the next outputs
    auto [X_train, Y_train, X_valid, Y_valid, pat_indices, calIdxForUnshuffle] = data_.makeLstmPastData(cfg_.lstm_past_time_steps,
                                                                                cfg_.mlp_architecture.back(),
                                                                                cfg_.train_fraction,
                                                                                cfg_.shuffle,
                                                                                cfg_.seed);

    std::cout << "-> Data split into training and validation sets." << std::endl;

    std::vector<std::string> colNames;
    for (int c = 0; c < Y_train.cols(); ++c) {
        colNames.push_back("h" + std::to_string(c+1));
    }

    // Save real data
    Eigen::MatrixXd Y_true_calib_save = data_.unshuffleMatrix(Y_train, calIdxForUnshuffle);
    Eigen::MatrixXd Y_true_valid_save = Y_valid;
    try {
        Y_true_calib_save = data_.inverseTransformOutputs(Y_true_calib_save);
        Y_true_valid_save = data_.inverseTransformOutputs(Y_true_valid_save);
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] inverseTransformOutputs failed (save GT): " << ex.what() << "\n";
    }
    data_.saveMatrixCsv(cfg_.real_calib, Y_true_calib_save, colNames);
    data_.saveMatrixCsv(cfg_.real_valid, Y_true_valid_save, colNames);
    data_.saveVector(pat_indices, cfg_.pattern_indices);
    std::cout << "-> Real calibration and validation data saved." << std::endl;

    // Configure one LSTM layer per ensemble run
    std::vector<LSTMLayer> lstm_vec(cfg_.ensemble_runs);
    #pragma omp parallel for num_threads(nthreads_)
    for(int i = 0; i < cfg_.ensemble_runs; i++){
        lstm_vec[i].initLSTMLayer(cfg_.columns.size(),cfg_.lstm_cells,cfg_.lstm_past_time_steps,cfg_.lstm_future_time_steps,true,"XG",cfg_.seed);
    }
        // Configure the output MLPs (one per ensemble run)
    setNmlps(cfg_.ensemble_runs);
    init_mlps();

    std::cout << "-> Ensemble run starting..." << std::endl;
    // ------------------------------------------------------
    // Ensemble loop — one independent LSTM+MLP pair per iteration,
    // runs executed in parallel over OpenMP threads
    // ------------------------------------------------------
    #pragma omp parallel for num_threads(nthreads_)
    for (unsigned run = 0; run < lstm_vec.size() ; ++run) {
        std::string run_id = std::to_string(run+1);  // 1-based run id as a string (for filenames)
        unsigned run_id_integer = run + 1; // 1-based run id as an integer (for the log)

        // Make sure the log directory exists
        if (!std::filesystem::exists(cfg_.log_dir)) {
            std::filesystem::create_directories(cfg_.log_dir);
        }
        // Open the per-run log file
        std::string filename = cfg_.log_dir + "log_run" + run_id + ".log";
        std::ofstream logFile(filename);

        // Log all configuration settings for this run
        data_.logRunSettings(logFile, cfg_, run_id_integer);

        logFile << "Run " << run_id << " starting...\n";

        // Train
        logFile << "-> Training starting...\n";

        int runIndex = run + 1;

        // Online training loop over epochs and windowed samples:
        // forward through the LSTM, gradient through the output MLP,
        // backpropagate the MLP's first-layer delta into the LSTM, then
        // Adam-update both parts; the LSTM memory is cleared per sample
        for(int iter = 1; iter <= cfg_.max_iterations ; iter++){
            std::cout<<"\n"<< iter<< "\n";
            for(size_t i = 0; i < X_train.size() ; i++){
                lstm_vec[run].setInputTSSegment(X_train[i]);
                lstm_vec[run].calculateTimeSteps();
                mlps_[run].runAndCalculateBatchGradient(lstm_vec[run].getForwardOutputVector(),Y_train.row(i));
                lstm_vec[run].setDeltaFromNextLayer(mlps_[run].getFirstLayerInputDelta());
                lstm_vec[run].calculateGradients();
                lstm_vec[run].updateAdam(cfg_.learning_rate,iter,0.9, 0.99, 1e-8);
                mlps_[run].updateWeightsAdam(cfg_.learning_rate,iter);
                lstm_vec[run].eraseMemory();
            }
        }

        logFile << "-> Training finished.\n";

        // Evaluate calibration: forward-only passes through LSTM + MLP
        logFile << "-> Evaluating calibration set...\n";
        Eigen::MatrixXd Y_pred_calib = Eigen::MatrixXd(Y_train.rows(),Y_train.cols());
        Eigen::MatrixXd Y_true_calib = Y_train;

        for(size_t i = 0; i < X_train.size() ; i++){
            lstm_vec[run].setInputTSSegment(X_train[i]);
            lstm_vec[run].calculateTimeSteps();
            mlps_[run].calcOneOutput(lstm_vec[run].getForwardOutputVector());
            Y_pred_calib.row(i) = mlps_[run].getOutput();
            lstm_vec[run].eraseMemory();
        }

        try {
            Y_true_calib = data_.inverseTransformOutputs(Y_true_calib);
            Y_pred_calib = data_.inverseTransformOutputs(Y_pred_calib);
        } catch (...) {}

        // One file for each metrics per each run
        std::vector<double> mseVals, rmseVals, piVals, nsVals, kgeVals, pbiasVals, rsrVals;

        for (int c = 0; c < Y_true_calib.cols(); ++c) {
            mseVals.push_back(Metrics::mse(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            rmseVals.push_back(Metrics::rmse(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            piVals.push_back(Metrics::pi(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            nsVals.push_back(Metrics::ns(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            kgeVals.push_back(Metrics::kge(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            pbiasVals.push_back(Metrics::pbias(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            rsrVals.push_back(Metrics::rsr(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
        }
        // per-metric filenames for this run
        std::string mseFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "MSE");
        std::string rmseFile  = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RMSE");
        std::string piFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PI");
        std::string nsFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "NS");
        std::string kgeFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "KGE");
        std::string pbiasFile = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PBIAS");
        std::string rsrFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RSR");
        // always write header once per file (this is one run)
        Metrics::saveMetricRow(mseFile,   colNames, mseVals,   true);
        Metrics::saveMetricRow(rmseFile,  colNames, rmseVals,  true);
        Metrics::saveMetricRow(piFile,    colNames, piVals,    true);
        Metrics::saveMetricRow(nsFile,    colNames, nsVals,    true);
        Metrics::saveMetricRow(kgeFile,   colNames, kgeVals,   true);
        Metrics::saveMetricRow(pbiasFile, colNames, pbiasVals, true);
        Metrics::saveMetricRow(rsrFile,   colNames, rsrVals,   true);


        data_.saveMatrixCsv(Metrics::addRunIdToFilename(cfg_.pred_calib, run_id), 
                            data_.unshuffleMatrix(Y_pred_calib, calIdxForUnshuffle), 
                            colNames);

        logFile << "-> Calibration metrics and predictions saved.\n";

        // Evaluate validation: forward-only passes through LSTM + MLP on the
        // held-out split
        logFile << "-> Evaluating validation set...\n";
        Eigen::MatrixXd Y_pred_valid = Eigen::MatrixXd(Y_valid.rows(),Y_valid.cols());
        Eigen::MatrixXd Y_true_valid = Y_valid;

        for(size_t i = 0; i < X_valid.size() ; i++){
            lstm_vec[run].setInputTSSegment(X_valid[i]);
            lstm_vec[run].calculateTimeSteps();
            mlps_[run].calcOneOutput(lstm_vec[run].getForwardOutputVector());
            Y_pred_valid.row(i) = mlps_[run].getOutput();
            lstm_vec[run].eraseMemory();
        }

        try {
            Y_true_valid = data_.inverseTransformOutputs(Y_true_valid);
            Y_pred_valid = data_.inverseTransformOutputs(Y_pred_valid);
        } catch (...) {}

        // One file for each metrics per each run
        std::vector<double> mseValsV, rmseValsV, piValsV, nsValsV, kgeValsV, pbiasValsV, rsrValsV;

        for (int c = 0; c < Y_true_valid.cols(); ++c) {
            mseValsV.push_back(Metrics::mse(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            rmseValsV.push_back(Metrics::rmse(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            piValsV.push_back(Metrics::pi(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            nsValsV.push_back(Metrics::ns(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            kgeValsV.push_back(Metrics::kge(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            pbiasValsV.push_back(Metrics::pbias(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            rsrValsV.push_back(Metrics::rsr(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
        }

        std::string mseFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "MSE");
        std::string rmseFileV  = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RMSE");
        std::string piFileV    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PI");
        std::string nsFileV    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "NS");
        std::string kgeFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "KGE");
        std::string pbiasFileV = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PBIAS");
        std::string rsrFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RSR");

        Metrics::saveMetricRow(mseFileV,   colNames, mseValsV,   true);
        Metrics::saveMetricRow(rmseFileV,  colNames, rmseValsV,  true);
        Metrics::saveMetricRow(piFileV,    colNames, piValsV,    true);
        Metrics::saveMetricRow(nsFileV,    colNames, nsValsV,    true);
        Metrics::saveMetricRow(kgeFileV,   colNames, kgeValsV,   true);
        Metrics::saveMetricRow(pbiasFileV, colNames, pbiasValsV, true);
        Metrics::saveMetricRow(rsrFileV,   colNames, rsrValsV,   true);

        
        data_.saveMatrixCsv(Metrics::addRunIdToFilename(cfg_.pred_valid, run_id), Y_pred_valid, colNames);
        logFile << "-> Validation metrics and predictions saved.\n";

        logFile << "Run " << run_id << " finished.\n";
        logFile << "-------------------------------------------\n";

    }

    std::cout << "-> Ensemble run finished." << std::endl;

    std::cout << "\n===========================================\n";
    std::cout << " Running Ensemble finished \n";
    std::cout << "===========================================\n";
}

/// Implements JKMNet::ensembleLstmPastFutureTest — a three-LSTM encoder-decoder
/// style ensemble for direct multi-horizon forecasting.
/**
 * Architecture per ensemble run:
 * - lstm_past_vec    encodes the past observation window,
 * - lstm_future_vec  encodes known future forcings (all columns except the
 *                    observed one),
 * - lstm_together_vec merges the two encodings over the combined past+future
 *   time axis, and the output MLP predicts every horizon step.
 *
 * Training runs the horizon-wise objective through MLP::horizonBackprop so
 * losses that need the whole forecast vector (slope/curvature/DTW/PI) see all
 * future steps at once; the resulting delta is then propagated backwards
 * through lstm_together_vec and split back into the past/future encoders.
 * All three LSTMs and the MLP are Adam-updated per sample and the LSTM
 * memories are cleared between samples.
 *
 * @note The between_lstm buffer stacks past encodings in the first rows and
 *   future encodings below, matching the input layout expected by
 *   lstm_together_vec.
 */
void JKMNet::ensembleLstmPastFutureTest(){
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
    data_.setTransform(strVecToTransformTypes(cfg_.transform),
                       cfg_.transform_alpha,
                       cfg_.exclude_last_col_from_transform);
    data_.applyTransform();
    std::cout << "-> Data transformed." << std::endl;

    // Build the windowed dataset: separate past-observation and
    // future-forcing input segments plus the horizon targets
    auto [X_trainPast, X_trainFuture, Y_train, X_validPast, X_validFuture, Y_valid, pat_indices, calIdxForUnshuffle] = 
                                                    data_.makeLstmPastFutureData(cfg_.lstm_past_time_steps,
                                                                                cfg_.lstm_future_time_steps,
                                                                                cfg_.train_fraction,
                                                                                cfg_.shuffle,
                                                                                cfg_.seed);

    std::cout << "-> Data split into training and validation sets." << std::endl;

    std::vector<std::string> colNames;
    for (int c = 0; c < Y_train.cols(); ++c) {
        colNames.push_back("h" + std::to_string(c+1));
    }

    // Save real data
    Eigen::MatrixXd Y_true_calib_save = data_.unshuffleMatrix(Y_train, calIdxForUnshuffle);
    Eigen::MatrixXd Y_true_valid_save = Y_valid;
    try {
        Y_true_calib_save = data_.inverseTransformOutputs(Y_true_calib_save);
        Y_true_valid_save = data_.inverseTransformOutputs(Y_true_valid_save);
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] inverseTransformOutputs failed (save GT): " << ex.what() << "\n";
    }
    data_.saveMatrixCsv(cfg_.real_calib, Y_true_calib_save, colNames);
    data_.saveMatrixCsv(cfg_.real_valid, Y_true_valid_save, colNames);
    data_.saveVector(pat_indices, cfg_.pattern_indices);
    std::cout << "-> Real calibration and validation data saved." << std::endl;

    // Configure the three LSTMLayers per run, plus per-run scratch buffers:
    // between_lstm  — stacked past|future encodings fed to the "together" LSTM,
    // lstm_to_mlp   — per-horizon-step inputs to the output MLP,
    // delta_mlp_to_lstm — MLP input deltas propagated back into the LSTMs,
    // separate_obs  — the observed horizon row reshaped as a single sample
    std::vector<LSTMLayer> lstm_past_vec(cfg_.ensemble_runs);
    std::vector<LSTMLayer> lstm_future_vec(cfg_.ensemble_runs);
    std::vector<LSTMLayer> lstm_together_vec(cfg_.ensemble_runs);
    std::vector<Eigen::MatrixXd> between_lstm(cfg_.ensemble_runs);
    std::vector<Eigen::MatrixXd> lstm_to_mlp(cfg_.ensemble_runs);
    std::vector<Eigen::MatrixXd> delta_mlp_to_lstm(cfg_.ensemble_runs);
    std::vector<Eigen::MatrixXd> separate_obs(cfg_.ensemble_runs);
    #pragma omp parallel for num_threads(nthreads_)
    for(int i = 0; i < cfg_.ensemble_runs; i++){
        lstm_past_vec[i].initLSTMLayer(cfg_.columns.size(),cfg_.lstm_cells,cfg_.lstm_past_time_steps,cfg_.lstm_past_time_steps,true,"XG",cfg_.seed);
        lstm_future_vec[i].initLSTMLayer(cfg_.columns.size()-1,cfg_.lstm_cells,cfg_.lstm_future_time_steps,cfg_.lstm_future_time_steps,true,"XG",cfg_.seed);
        lstm_together_vec[i].initLSTMLayer(cfg_.lstm_cells,cfg_.lstm_cells,cfg_.lstm_past_time_steps + cfg_.lstm_future_time_steps,cfg_.lstm_future_time_steps,false,"XG",cfg_.seed);
        between_lstm[i] = Eigen::MatrixXd(cfg_.lstm_future_time_steps + cfg_.lstm_past_time_steps, cfg_.lstm_cells);
        lstm_to_mlp[i] = Eigen::MatrixXd(cfg_.lstm_cells,cfg_.lstm_future_time_steps);
        delta_mlp_to_lstm[i] = Eigen::MatrixXd(cfg_.lstm_cells,cfg_.lstm_future_time_steps);
        separate_obs[i] = Eigen::MatrixXd(1,cfg_.lstm_future_time_steps);
    }
        // Configure the output MLPs (one per ensemble run)
    setNmlps(cfg_.ensemble_runs);
    init_mlps();

    std::cout << "-> Ensemble run starting..." << std::endl;
    // ------------------------------------------------------
    // Ensemble loop
    // ------------------------------------------------------
    #pragma omp parallel for num_threads(nthreads_)
    for (unsigned run = 0; run < lstm_past_vec.size() ; ++run) {
        std::string run_id = std::to_string(run+1);  // string
        unsigned run_id_integer = run + 1; // integer

        // Check, if exists folder 'outputs/logs'
        if (!std::filesystem::exists(cfg_.log_dir)) {
            std::filesystem::create_directories(cfg_.log_dir);
        }
        // Set the output file for the log messages        
        std::string filename = cfg_.log_dir + "log_run" + run_id + ".log";
        std::ofstream logFile(filename);

        // Log all settings 
        data_.logRunSettings(logFile, cfg_, run_id_integer);

        logFile << "Run " << run_id << " starting...\n";

        // Train
        logFile << "-> Training starting...\n";

        int runIndex = run + 1;

        // Online training loop over epochs and windowed samples
        for(int iter = 1; iter <= cfg_.max_iterations ; iter++){
            for(size_t i = 0; i < X_trainPast.size() ; i++){
                // Forward: encode past observations and future forcings...
                lstm_past_vec[run].setInputTSSegment(X_trainPast[i]);
                lstm_future_vec[run].setInputTSSegment(X_trainFuture[i]);
                lstm_past_vec[run].calculateTimeSteps();
                lstm_future_vec[run].calculateTimeSteps();
                // ...stack both encodings into the "together" LSTM input and merge
                between_lstm[run].block(0,0,cfg_.lstm_past_time_steps,between_lstm[run].cols()) = lstm_past_vec[run].getForwardOutput();
                between_lstm[run].block(cfg_.lstm_past_time_steps,0,cfg_.lstm_future_time_steps,between_lstm[run].cols()) = lstm_future_vec[run].getForwardOutput();
                lstm_together_vec[run].setInputTSSegment(between_lstm[run]);
                lstm_together_vec[run].calculateTimeSteps();
                lstm_to_mlp[run] = lstm_together_vec[run].getForwardOutput().transpose();
                separate_obs[run] = Y_train.row(i);
                {
                    // Objective over the full horizon vector (slope/curvature/
                    // DTW/PI losses need all future steps at once)
                    std::vector<Eigen::VectorXd> stepInputs(lstm_to_mlp[run].cols());
                    for(int s = 0; s < lstm_to_mlp[run].cols(); s++){
                        stepInputs[s] = lstm_to_mlp[run].col(s);
                    }
                    delta_mlp_to_lstm[run] = mlps_[run].horizonBackprop(stepInputs, separate_obs[run].row(0).transpose());
                    mlps_[run].updateWeightsAdam(cfg_.learning_rate,iter);
                }
                // Backward: push the MLP delta into the "together" LSTM and
                // split its input deltas back between the past/future encoders
                lstm_together_vec[run].setDeltaFromNextLayer(delta_mlp_to_lstm[run]);
                lstm_together_vec[run].calculateGradients();
                lstm_past_vec[run].setDeltaFromNextLayer(Eigen::MatrixXd(lstm_together_vec[run].getDeltaInputs().block(0,0,cfg_.lstm_cells,cfg_.lstm_past_time_steps)));
                lstm_future_vec[run].setDeltaFromNextLayer(Eigen::MatrixXd(lstm_together_vec[run].getDeltaInputs().block(0,cfg_.lstm_past_time_steps,cfg_.lstm_cells,cfg_.lstm_future_time_steps)));
                lstm_past_vec[run].calculateGradients();
                lstm_future_vec[run].calculateGradients();
                // Adam-update all three LSTMs; MLP was already updated above;
                // memories are cleared per sample so samples stay independent
                lstm_past_vec[run].updateAdam(cfg_.learning_rate,iter,0.9, 0.99, 1e-8);
                lstm_future_vec[run].updateAdam(cfg_.learning_rate,iter,0.9, 0.99, 1e-8);
                lstm_together_vec[run].updateAdam(cfg_.learning_rate,iter,0.9, 0.99, 1e-8);
                lstm_past_vec[run].eraseMemory();
                lstm_future_vec[run].eraseMemory();
                lstm_together_vec[run].eraseMemory();
            }
        }

        logFile << "-> Training finished.\n";

        // Evaluate calibration: forward-only passes through the three LSTMs,
        // predicting each horizon step separately with the output MLP
        logFile << "-> Evaluating calibration set...\n";
        Eigen::MatrixXd Y_pred_calib = Eigen::MatrixXd(Y_train.rows(),Y_train.cols());
        Eigen::MatrixXd Y_true_calib = Y_train;

        for(size_t i = 0; i < X_trainPast.size() ; i++){
            lstm_past_vec[run].setInputTSSegment(X_trainPast[i]);
            lstm_future_vec[run].setInputTSSegment(X_trainFuture[i]);
            lstm_past_vec[run].calculateTimeSteps();
            lstm_future_vec[run].calculateTimeSteps();
            between_lstm[run].block(0,0,cfg_.lstm_past_time_steps,between_lstm[run].cols()) = lstm_past_vec[run].getForwardOutput();
            between_lstm[run].block(cfg_.lstm_past_time_steps,0,cfg_.lstm_future_time_steps,between_lstm[run].cols()) = lstm_future_vec[run].getForwardOutput();
            lstm_together_vec[run].setInputTSSegment(between_lstm[run]);
            lstm_together_vec[run].calculateTimeSteps();
            lstm_to_mlp[run] = lstm_together_vec[run].getForwardOutput().transpose();
            for(int s = 0; s < lstm_to_mlp[run].cols(); s++){
                mlps_[run].calcOneOutput(lstm_to_mlp[run].col(s));
                Y_pred_calib(i,s) = mlps_[run].getOutput().value();
            }
            lstm_past_vec[run].eraseMemory();
            lstm_future_vec[run].eraseMemory();
            lstm_together_vec[run].eraseMemory();
        }

        try {
            Y_true_calib = data_.inverseTransformOutputs(Y_true_calib);
            Y_pred_calib = data_.inverseTransformOutputs(Y_pred_calib);
        } catch (...) {}

        // One file for each metrics per each run
        std::vector<double> mseVals, rmseVals, piVals, nsVals, kgeVals, pbiasVals, rsrVals;

        for (int c = 0; c < Y_true_calib.cols(); ++c) {
            mseVals.push_back(Metrics::mse(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            rmseVals.push_back(Metrics::rmse(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            piVals.push_back(Metrics::pi(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            nsVals.push_back(Metrics::ns(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            kgeVals.push_back(Metrics::kge(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            pbiasVals.push_back(Metrics::pbias(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            rsrVals.push_back(Metrics::rsr(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
        }
        // per-metric filenames for this run
        std::string mseFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "MSE");
        std::string rmseFile  = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RMSE");
        std::string piFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PI");
        std::string nsFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "NS");
        std::string kgeFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "KGE");
        std::string pbiasFile = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PBIAS");
        std::string rsrFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RSR");
        // always write header once per file (this is one run)
        Metrics::saveMetricRow(mseFile,   colNames, mseVals,   true);
        Metrics::saveMetricRow(rmseFile,  colNames, rmseVals,  true);
        Metrics::saveMetricRow(piFile,    colNames, piVals,    true);
        Metrics::saveMetricRow(nsFile,    colNames, nsVals,    true);
        Metrics::saveMetricRow(kgeFile,   colNames, kgeVals,   true);
        Metrics::saveMetricRow(pbiasFile, colNames, pbiasVals, true);
        Metrics::saveMetricRow(rsrFile,   colNames, rsrVals,   true);


        data_.saveMatrixCsv(Metrics::addRunIdToFilename(cfg_.pred_calib, run_id), 
                            data_.unshuffleMatrix(Y_pred_calib, calIdxForUnshuffle), 
                            colNames);

        logFile << "-> Calibration metrics and predictions saved.\n";

        // Evaluate validation on the held-out split (same forward flow as
        // the calibration evaluation above)
        logFile << "-> Evaluating validation set...\n";
        Eigen::MatrixXd Y_pred_valid = Eigen::MatrixXd(Y_valid.rows(),Y_valid.cols());
        Eigen::MatrixXd Y_true_valid = Y_valid;

        for(size_t i = 0; i < X_validPast.size() ; i++){
            lstm_past_vec[run].setInputTSSegment(X_validPast[i]);
            lstm_future_vec[run].setInputTSSegment(X_validFuture[i]);
            lstm_past_vec[run].calculateTimeSteps();
            lstm_future_vec[run].calculateTimeSteps();
            between_lstm[run].block(0,0,cfg_.lstm_past_time_steps,between_lstm[run].cols()) = lstm_past_vec[run].getForwardOutput();
            between_lstm[run].block(cfg_.lstm_past_time_steps,0,cfg_.lstm_future_time_steps,between_lstm[run].cols()) = lstm_future_vec[run].getForwardOutput();
            lstm_together_vec[run].setInputTSSegment(between_lstm[run]);
            lstm_together_vec[run].calculateTimeSteps();
            lstm_to_mlp[run] = lstm_together_vec[run].getForwardOutput().transpose();
            for(int s = 0; s < lstm_to_mlp[run].cols(); s++){
                mlps_[run].calcOneOutput(lstm_to_mlp[run].col(s));
                Y_pred_valid(i,s) = mlps_[run].getOutput().value();
            }
            lstm_past_vec[run].eraseMemory();
            lstm_future_vec[run].eraseMemory();
            lstm_together_vec[run].eraseMemory();
        }

        try {
            Y_true_valid = data_.inverseTransformOutputs(Y_true_valid);
            Y_pred_valid = data_.inverseTransformOutputs(Y_pred_valid);
        } catch (...) {}

        // One file for each metrics per each run
        std::vector<double> mseValsV, rmseValsV, piValsV, nsValsV, kgeValsV, pbiasValsV, rsrValsV;

        for (int c = 0; c < Y_true_valid.cols(); ++c) {
            mseValsV.push_back(Metrics::mse(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            rmseValsV.push_back(Metrics::rmse(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            piValsV.push_back(Metrics::pi(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            nsValsV.push_back(Metrics::ns(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            kgeValsV.push_back(Metrics::kge(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            pbiasValsV.push_back(Metrics::pbias(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            rsrValsV.push_back(Metrics::rsr(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
        }

        std::string mseFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "MSE");
        std::string rmseFileV  = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RMSE");
        std::string piFileV    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PI");
        std::string nsFileV    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "NS");
        std::string kgeFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "KGE");
        std::string pbiasFileV = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PBIAS");
        std::string rsrFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RSR");

        Metrics::saveMetricRow(mseFileV,   colNames, mseValsV,   true);
        Metrics::saveMetricRow(rmseFileV,  colNames, rmseValsV,  true);
        Metrics::saveMetricRow(piFileV,    colNames, piValsV,    true);
        Metrics::saveMetricRow(nsFileV,    colNames, nsValsV,    true);
        Metrics::saveMetricRow(kgeFileV,   colNames, kgeValsV,   true);
        Metrics::saveMetricRow(pbiasFileV, colNames, pbiasValsV, true);
        Metrics::saveMetricRow(rsrFileV,   colNames, rsrValsV,   true);

        
        data_.saveMatrixCsv(Metrics::addRunIdToFilename(cfg_.pred_valid, run_id), Y_pred_valid, colNames);
        logFile << "-> Validation metrics and predictions saved.\n";

        logFile << "Run " << run_id << " finished.\n";
        logFile << "-------------------------------------------\n";

    }

    std::cout << "-> Ensemble run finished." << std::endl;

    std::cout << "\n===========================================\n";
    std::cout << " Running Ensemble finished \n";
    std::cout << "===========================================\n";
}
