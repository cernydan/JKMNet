// ********* 16. 9. 2025 *********
// **DONE**: Save initial matrix and vector of weights
// **DONE**: Save final vector of weights
// **DONE**: Save other needed params into file, e.g., #iteration, duration, etc.
// **DONE**: Split data into calib and valid dataset
// **DONE**: Create method for loading weights
// **DONE**: Prepare validation run, i.e.,: read data and settings from files, no training
// **DONE**: Split data to calib and valid with or without schuffle, i.e., randomly or historically 
// **DONE**: Clean the code from unused methods
// **DONE**: Use 'main()' only for read data and setting, run, save
// TODO: Use more detailed data, i.e., hourly or 15-min, and prepare all datasets (in R?) 
// TODO: Prepare many MLPs configuration, i.e., 'config_model.ini' (in R?)
// TODO: Prepare tree structure of files and folders for running each MLP configuration with corresponding setting and data
// TODO: Solve the running of scenarios in a loop (with Rcpp or in .sh file?)

// ********* MetaCentrum *********
// TODO: Upload all case folders
// **DONE**: Compile the C++ code on MetaCentrum using an interactive job 
// TODO: Put the bin file inside each case folder
// TODO: Test the code, i.e., run MLP with the same seed many times (approx. 100) (must have the same results!!) - solve seeds!
// TODO: Create .sh file for running qsub job on MetaCentrum
// *******************************

#include <ctime>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <limits>
#include <iomanip>
#include <chrono>

#include "JKMNet.hpp"
#include "MLP.hpp"
#include "Layer.hpp"
#include "Data.hpp"
#include "Metrics.hpp"
#include "ConfigIni.hpp"
#include "CNNLayer.hpp"

using namespace std;

int main() {
  //!! ------------------------------------------------------------
  //!! REAL DATA
  //!! ------------------------------------------------------------

  JKMNet net;
  Data configData;
  MLP configBatchMLP;

  // ------------------------------------------------------
  // Read setting file
  // ------------------------------------------------------  
  RunConfig cfg = parseConfigIni("settings/config_model.ini");

  std::unordered_set<std::string> idFilter;
  if (!cfg.id.empty()) {
    std::vector<std::string> ids = parseStringList(cfg.id); // splits & trims on commas
    idFilter = std::unordered_set<std::string>(ids.begin(), ids.end());
  }

  // ------------------------------------------------------
  // Load and filter data
  // ------------------------------------------------------    
  configData.loadFilteredCSV(cfg.data_file, idFilter, cfg.columns, cfg.timestamp, cfg.id_col);
  
  // ------------------------------------------------------
  // Transform data
  // ------------------------------------------------------  
  transform_type tt = transform_type::NONE;
  try {
      tt = strToTransformType(cfg.transform);
  } catch (const std::exception &ex) {
      std::cerr << "[Warning] Unknown transform in config ('" << cfg.transform 
                << "'). Defaulting to NONE.\n";
      tt = transform_type::NONE;
  }

  configData.setTransform(tt, cfg.transform_alpha, cfg.exclude_last_col_from_transform);

  // Apply transform
  configData.applyTransform();
  
  // ------------------------------------------------------
  // Build calibration matrix and split into train/valid
  // ------------------------------------------------------
  configData.makeCalibMatsSplit(cfg.input_numbers, (int)cfg.mlp_architecture.back());
  Eigen::MatrixXd calibMat = configData.getCalibMat();

  auto [trainMat, validMat, trainIdx, validIdx] = configData.splitCalibMatWithIdx(cfg.train_fraction, cfg.split_shuffle, cfg.seed);

  int inpSize = (int)trainMat.cols() - (int)cfg.mlp_architecture.back();
  int outSize = (int)cfg.mlp_architecture.back();

  auto [X_train, Y_train] = configData.splitInputsOutputs(trainMat, inpSize, outSize);
  auto [X_valid, Y_valid] = configData.splitInputsOutputs(validMat, inpSize, outSize);

  
  // ------------------------------------------------------
  // Configure and initialize MLP
  // ------------------------------------------------------
  configBatchMLP.setArchitecture(cfg.mlp_architecture);
  std::vector<activ_func_type> realActivs(cfg.mlp_architecture.size(), strToActivation(cfg.activation));
  configBatchMLP.setActivations(realActivs);
  std::vector<weight_init_type> realWeightInit(cfg.mlp_architecture.size(), strToWeightInit(cfg.weight_init));
  configBatchMLP.setWInitType(realWeightInit);

  // initialize layers (build weights) using any input of the right length 
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(cfg.input_numbers.size());
  configBatchMLP.initMLP(x0);

 
  // Save initialized weights
  configBatchMLP.saveWeightsCsv(cfg.weights_csv_init);
  configBatchMLP.saveWeightsBinary(cfg.weights_bin_init);
  configBatchMLP.weightsToVectorMlp();
  configBatchMLP.saveWeightsVectorCsv(cfg.weights_vec_csv_init);
  configBatchMLP.saveWeightsVectorBinary(cfg.weights_vec_bin_init);

  // ------------------------------------------------------
  // Run training (on calibration/train set only)
  // ------------------------------------------------------
  TrainingResult trainingResult;
  Eigen::MatrixXd resultErrors;

  if (cfg.trainer == "online") {
      trainingResult = net.trainAdamOnline(
          configBatchMLP,
          X_train, Y_train,
          cfg.max_iterations,
          cfg.max_error,
          cfg.learning_rate,
          cfg.shuffle,
          cfg.seed
      );
  }
  else if (cfg.trainer == "batch") {
      trainingResult = net.trainAdamBatch(
          configBatchMLP,
          X_train, Y_train,
          cfg.batch_size,
          cfg.max_iterations,
          cfg.max_error,
          cfg.learning_rate,
          cfg.shuffle,
          cfg.seed
      );
  }
  else if (cfg.trainer == "online_epoch") {
      resultErrors = net.trainAdamOnlineEpochVal(
          configBatchMLP,
          X_train, Y_train,
          X_valid, Y_valid,
          cfg.max_iterations,
          cfg.max_error,
          cfg.learning_rate,
          cfg.shuffle,
          cfg.seed
      );

      trainingResult.finalLoss = resultErrors(resultErrors.rows() - 1, 1);
      trainingResult.iterations = cfg.max_iterations;
      trainingResult.converged = (trainingResult.finalLoss <= cfg.max_error);

      Metrics::saveErrorsCsv(cfg.errors_csv, resultErrors);
  }
  else if (cfg.trainer == "batch_epoch") {
      resultErrors = net.trainAdamBatchEpochVal(
          configBatchMLP,
          X_train, Y_train,
          X_valid, Y_valid,
          cfg.batch_size,
          cfg.max_iterations,
          cfg.max_error,
          cfg.learning_rate,
          cfg.shuffle,
          cfg.seed
      );

      trainingResult.finalLoss = resultErrors(resultErrors.rows()-1, 0);
      trainingResult.iterations = cfg.max_iterations;
      trainingResult.converged = (trainingResult.finalLoss <= cfg.max_error);

      Metrics::saveErrorsCsv(cfg.errors_csv, resultErrors);
  }
  else {
      throw std::invalid_argument(
          "Unknown trainer type: " + cfg.trainer +
          " (must be online, batch, online_epoch_val, or batch_epoch_val)"
      );
  }



  // ------------------------------------------------------
  // Evaluate calibration/train set
  // ------------------------------------------------------
  configBatchMLP.calculateOutputs(configData.getCalibInpsMat());

  Eigen::MatrixXd Y_true_calib = configData.getCalibOutsMat();   
  Eigen::MatrixXd Y_pred_calib = configBatchMLP.getOutputs();    

  try {
    Y_true_calib = configData.inverseTransformOutputs(Y_true_calib);
    Y_pred_calib = configData.inverseTransformOutputs(Y_pred_calib);
  } catch (const std::exception &ex) {
      std::cerr << "[Warning] inverseTransformOutputs failed: " << ex.what() << "\n"
                << "Saving transformed values instead.\n";
  }

  // Shapes should match
  if (Y_true_calib.rows() != Y_pred_calib.rows() || Y_true_calib.cols() != Y_pred_calib.cols()) {
      std::cerr << "[Warning] shape mismatch: real vs pred ("
                << Y_true_calib.rows() << "x" << Y_true_calib.cols() << ") vs ("
                << Y_pred_calib.rows() << "x" << Y_pred_calib.cols() << ").\n";
  }

  // ------------------------------------------------------
  // Save run info and weights
  // ------------------------------------------------------
  Metrics::appendRunInfoCsv(cfg.run_info,
                          configBatchMLP.getLastIterations(),
                          configBatchMLP.getLastError(),
                          trainingResult.converged,
                          configBatchMLP.getLastRuntimeSec(),
                          cfg.id);

  // Save metrics into CSV file (need to have an existing folder "data/outputs")
  Metrics::computeAndAppendFinalMetrics(Y_true_calib, Y_pred_calib, cfg.metrics_cal, cfg.id);

  // Save real and predicted calib data
  std::vector<std::string> colNames;
  for (int c = 0; c < Y_true_calib.cols(); ++c) {
      colNames.push_back(std::string("h") + std::to_string(c+1));
  }

  bool ok1 = configData.saveMatrixCsv(cfg.real_calib, Y_true_calib, colNames);
  bool ok2 = configData.saveMatrixCsv(cfg.pred_calib, Y_pred_calib, colNames);
  if (ok1 && ok2) {
  } else {
      std::cerr << "[I/O] Saving calib matrices failed\n";
  }

  configBatchMLP.saveWeightsCsv(cfg.weights_csv);
  configBatchMLP.saveWeightsBinary(cfg.weights_bin);
  configBatchMLP.weightsToVectorMlp();
  configBatchMLP.saveWeightsVectorCsv(cfg.weights_vec_csv);
  configBatchMLP.saveWeightsVectorBinary(cfg.weights_vec_bin);

  // ------------------------------------------------------
  // Evaluate validation set
  // ------------------------------------------------------
  configBatchMLP.calculateOutputs(X_valid);

  Eigen::MatrixXd Y_pred_valid = configBatchMLP.getOutputs();
  Eigen::MatrixXd Y_true_valid = Y_valid;

  try {
      Y_true_valid = configData.inverseTransformOutputs(Y_true_valid);
      Y_pred_valid = configData.inverseTransformOutputs(Y_pred_valid);
  } catch (const std::exception &ex) {
      std::cerr << "[Warning] inverseTransformOutputs failed (valid): " << ex.what() << "\n";
  }

  // Save metrics into CSV file 
  Metrics::computeAndAppendFinalMetrics(Y_true_valid, Y_pred_valid, cfg.metrics_val, cfg.id);

  // Save real and predicted calib data
  bool ok1V = configData.saveMatrixCsv(cfg.real_valid, Y_true_valid, colNames);
  bool ok2V = configData.saveMatrixCsv(cfg.pred_valid, Y_pred_valid, colNames);
  if (ok1V && ok2V) {
  } else {
      std::cerr << "[I/O] Saving calib matrices failed\n";
  }

  std::cout << "** JKMNet finished **" << endl;
  
  return 0;
}
