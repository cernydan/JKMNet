#include "EnsembleRunner.hpp"
#include <iostream>
#include <numeric>

EnsembleRunner::EnsembleRunner(const RunConfig& cfg, unsigned nthreads)
    : cfg_(cfg), nthreads_(nthreads) {}

void EnsembleRunner::run() {
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

    data_.makeCalibMatsSplit(cfg_.input_numbers, (int)cfg_.mlp_architecture.back());

    auto [trainMat, validMat, trainIdx, validIdx] = data_.splitCalibMatWithIdx(cfg_.train_fraction, cfg_.split_shuffle, cfg_.seed);

    int inpSize = (int)trainMat.cols() - (int)cfg_.mlp_architecture.back();
    int outSize = (int)cfg_.mlp_architecture.back();

    auto [X_train, Y_train] = data_.splitInputsOutputs(trainMat, inpSize, outSize);
    auto [X_valid, Y_valid] = data_.splitInputsOutputs(validMat, inpSize, outSize);

    std::cout << "-> Data split into training and validation sets." << std::endl;

    std::vector<std::string> colNames;
    for (int c = 0; c < Y_train.cols(); ++c) {
        colNames.push_back("h" + std::to_string(c+1));
    }

    // Save real data
    Eigen::MatrixXd Y_true_calib_save = data_.getCalibOutsMat();
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
        TrainingResult trainingResult;
        Eigen::MatrixXd resultErrors;
        
        if (cfg_.trainer == "online") {
            trainingResult = net_.trainAdamOnline(
                mlp_, X_train, Y_train,
                cfg_.max_iterations, cfg_.max_error,
                cfg_.learning_rate, cfg_.shuffle, cfg_.seed + run
            );
        } else if (cfg_.trainer == "batch") {
            trainingResult = net_.trainAdamBatch(
                mlp_, X_train, Y_train,
                cfg_.batch_size, cfg_.max_iterations,
                cfg_.max_error, cfg_.learning_rate,
                cfg_.shuffle, cfg_.seed + run
            );
        } else if (cfg_.trainer == "online_epoch") {
            resultErrors = net_.trainAdamOnlineEpochVal(
                mlp_, X_train, Y_train, X_valid, Y_valid,
                cfg_.max_iterations, cfg_.max_error,
                cfg_.learning_rate, cfg_.shuffle, cfg_.seed + run
            );
            trainingResult.finalLoss = resultErrors(resultErrors.rows()-1, 1);
            trainingResult.iterations = cfg_.max_iterations;
            trainingResult.converged = (trainingResult.finalLoss <= cfg_.max_error);
            Metrics::saveErrorsCsv(Metrics::addRunIdToFilename(cfg_.errors_csv, run_id), resultErrors);
        } else if (cfg_.trainer == "batch_epoch") {
            resultErrors = net_.trainAdamBatchEpochVal(
                mlp_, X_train, Y_train, X_valid, Y_valid,
                cfg_.batch_size, cfg_.max_iterations,
                cfg_.max_error, cfg_.learning_rate,
                cfg_.shuffle, cfg_.seed + run
            );
            trainingResult.finalLoss = resultErrors(resultErrors.rows()-1, 0);
            trainingResult.iterations = cfg_.max_iterations;
            trainingResult.converged = (trainingResult.finalLoss <= cfg_.max_error);
            Metrics::saveErrorsCsv(Metrics::addRunIdToFilename(cfg_.errors_csv, run_id), resultErrors);
        } else {
            throw std::invalid_argument("Unknown trainer type: " + cfg_.trainer);
        }
        std::cout << "-> Training finished." << std::endl;

        // Evaluate calibration
        std::cout << "-> Evaluating calibration set..." << std::endl;
        mlp_.calculateOutputs(data_.getCalibInpsMat());
        Eigen::MatrixXd Y_pred_calib = mlp_.getOutputs();
        Eigen::MatrixXd Y_true_calib = data_.getCalibOutsMat();
        try {
            Y_true_calib = data_.inverseTransformOutputs(Y_true_calib);
            Y_pred_calib = data_.inverseTransformOutputs(Y_pred_calib);
        } catch (...) {}
        Metrics::appendRunInfoCsv(cfg_.run_info,
            mlp_.getLastIterations(), mlp_.getLastError(),
            trainingResult.converged, mlp_.getLastRuntimeSec(),
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
