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

//JKMNet 8 //8 is number of threads
int main(int argc, char** argv) {
    unsigned nthreads = 1;  // default value

    if (argc > 1) {
        try {
            int valueThread = std::stoi(argv[1]);
            if (valueThread > 0) {
                nthreads = static_cast<unsigned>(valueThread);
            } else {
                std::cerr << "[Warning] Invalid thread number: " << argv[1] 
                          << ". Using default (1).\n";
            }
        } catch (const std::exception &e) {
            std::cerr << "[Warning] Cannot parse thread number: " << argv[1] 
                      << " (" << e.what() << "). Using default (1).\n";
        }
    }

    std::cout << "The numbers of thread is " << nthreads << std::endl;

    JKMNet net;
    Data Data;
    MLP MLP;

    // ------------------------------------------------------
    // Read setting file
    // ------------------------------------------------------  
    RunConfig cfg = parseConfigIni("settings/config_model.ini");

    // Clean output directories before training
    Data::cleanAllOutputs(cfg.out_dir);

    std::unordered_set<std::string> idFilter;
    if (!cfg.id.empty()) {
        std::vector<std::string> ids = parseStringList(cfg.id); 
        idFilter = std::unordered_set<std::string>(ids.begin(), ids.end());
    }
    std::cout << "-> Settings loaded." << std::endl;
    std::cout << "-> Output directory cleaned." << std::endl;

    // ------------------------------------------------------
    // Load and filter data
    // ------------------------------------------------------    
    Data.loadFilteredCSV(cfg.data_file, idFilter, cfg.columns, cfg.timestamp, cfg.id_col);
    std::cout << "-> Data loaded." << std::endl;

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
    Data.setTransform(tt, cfg.transform_alpha, cfg.exclude_last_col_from_transform);
    Data.applyTransform();

    std::cout << "-> Data transformed." << std::endl;
  
    // ------------------------------------------------------
    // Build calibration matrix and split into train/valid
    // ------------------------------------------------------
    Data.makeCalibMatsSplit(cfg.input_numbers, (int)cfg.mlp_architecture.back());
    Eigen::MatrixXd calibMat = Data.getCalibMat();

    auto [trainMat, validMat, trainIdx, validIdx] = 
        Data.splitCalibMatWithIdx(cfg.train_fraction, cfg.split_shuffle, cfg.seed);

    int inpSize = (int)trainMat.cols() - (int)cfg.mlp_architecture.back();
    int outSize = (int)cfg.mlp_architecture.back();

    auto [X_train, Y_train] = Data.splitInputsOutputs(trainMat, inpSize, outSize);
    auto [X_valid, Y_valid] = Data.splitInputsOutputs(validMat, inpSize, outSize);

    std::cout << "-> Data splitted to training and testing sets." << std::endl;

    // ------------------------------------------------------
    // Save real calib and valid data
    // ------------------------------------------------------
    std::vector<std::string> colNames;
    for (int c = 0; c < Y_train.cols(); ++c) {
        colNames.push_back(std::string("h") + std::to_string(c+1));
    }

    Eigen::MatrixXd Y_true_calib_save = Data.getCalibOutsMat();
    Eigen::MatrixXd Y_true_valid_save = Y_valid;

    try {
        Y_true_calib_save = Data.inverseTransformOutputs(Y_true_calib_save);
        Y_true_valid_save = Data.inverseTransformOutputs(Y_true_valid_save);
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] inverseTransformOutputs failed when saving ground-truth: " 
                  << ex.what() << "\n";
    }

    Data.saveMatrixCsv(cfg.real_calib, Y_true_calib_save, colNames);
    Data.saveMatrixCsv(cfg.real_valid, Y_true_valid_save, colNames);

    std::cout << "-> Real training and testing data saved." << std::endl;

    // ------------------------------------------------------
    // Ensemble loop
    // ------------------------------------------------------
    for (int run = 0; run < cfg.ensemble_runs; ++run) {
        std::cout << "\n-------------------------------------------" << std::endl;
        std::cout << "Run " <<  (run+1) << " starting..." << std::endl;
  
        std::string run_id = std::to_string(run + 1);

        // ------------------------------------------------------
        // Configure and initialize MLP
        // ------------------------------------------------------
        MLP.setArchitecture(cfg.mlp_architecture);
        std::vector<activ_func_type> realActivs(cfg.mlp_architecture.size(), strToActivation(cfg.activation));
        MLP.setActivations(realActivs);
        std::vector<weight_init_type> realWeightInit(cfg.mlp_architecture.size(), strToWeightInit(cfg.weight_init));
        MLP.setWInitType(realWeightInit);

        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(std::accumulate(cfg.input_numbers.begin(), cfg.input_numbers.end(), 0));
        MLP.initMLP(x0,cfg.seed);

        // Save initialized weights (per run)
        MLP.saveWeightsCsv(Metrics::addRunIdToFilename(cfg.weights_csv_init, run_id));
        MLP.weightsToVectorMlp();
        MLP.saveWeightsVectorCsv(Metrics::addRunIdToFilename(cfg.weights_vec_csv_init, run_id));

        // Save initialized weights into one combined file
        MLP.appendWeightsVectorCsv(cfg.weights_vec_csv_init, run == 0);

        // ------------------------------------------------------
        // Run training
        // ------------------------------------------------------
        std::cout << "-> Training starting..." << std::endl;

        TrainingResult trainingResult;
        Eigen::MatrixXd resultErrors;

        if (cfg.trainer == "online") {
            trainingResult = net.trainAdamOnline(
                MLP,
                X_train, Y_train,
                cfg.max_iterations,
                cfg.max_error,
                cfg.learning_rate,
                cfg.shuffle,
                cfg.seed + run
            );
        }
        else if (cfg.trainer == "batch") {
            trainingResult = net.trainAdamBatch(
                MLP,
                X_train, Y_train,
                cfg.batch_size,
                cfg.max_iterations,
                cfg.max_error,
                cfg.learning_rate,
                cfg.shuffle,
                cfg.seed + run
            );
        }
        else if (cfg.trainer == "online_epoch") {
            resultErrors = net.trainAdamOnlineEpochVal(
                MLP,
                X_train, Y_train,
                X_valid, Y_valid,
                cfg.max_iterations,
                cfg.max_error,
                cfg.learning_rate,
                cfg.shuffle,
                cfg.seed + run
            );
            trainingResult.finalLoss = resultErrors(resultErrors.rows() - 1, 1);
            trainingResult.iterations = cfg.max_iterations;
            trainingResult.converged = (trainingResult.finalLoss <= cfg.max_error);

            Metrics::saveErrorsCsv(Metrics::addRunIdToFilename(cfg.errors_csv, run_id), resultErrors);
        }
        else if (cfg.trainer == "batch_epoch") {
            resultErrors = net.trainAdamBatchEpochVal(
                MLP,
                X_train, Y_train,
                X_valid, Y_valid,
                cfg.batch_size,
                cfg.max_iterations,
                cfg.max_error,
                cfg.learning_rate,
                cfg.shuffle,
                cfg.seed + run
            );
            trainingResult.finalLoss = resultErrors(resultErrors.rows()-1, 0);
            trainingResult.iterations = cfg.max_iterations;
            trainingResult.converged = (trainingResult.finalLoss <= cfg.max_error);

            Metrics::saveErrorsCsv(Metrics::addRunIdToFilename(cfg.errors_csv, run_id), resultErrors);
        }
        else {
            throw std::invalid_argument(
                "Unknown trainer type: " + cfg.trainer +
                " (must be online, batch, online_epoch, or batch_epoch)"
            );
        }

        std::cout << "-> Training finished." << std::endl;
  

        // ------------------------------------------------------
        // Evaluate calibration/train set
        // ------------------------------------------------------
        std::cout << "-> Calculating metrics..." << std::endl;

        MLP.calculateOutputs(Data.getCalibInpsMat());

        Eigen::MatrixXd Y_true_calib = Data.getCalibOutsMat();   
        Eigen::MatrixXd Y_pred_calib = MLP.getOutputs();    

        try {
            Y_true_calib = Data.inverseTransformOutputs(Y_true_calib);
            Y_pred_calib = Data.inverseTransformOutputs(Y_pred_calib);
        } catch (const std::exception &ex) {
            std::cerr << "[Warning] inverseTransformOutputs failed: " << ex.what() << "\n";
        }

        Metrics::appendRunInfoCsv(
            cfg.run_info,
            MLP.getLastIterations(),
            MLP.getLastError(),
            trainingResult.converged,
            MLP.getLastRuntimeSec(),
            cfg.id + "_run" + run_id
        );

        Metrics::computeAndAppendFinalMetrics(
            Y_true_calib, Y_pred_calib,
            cfg.metrics_cal,   // same file across runs
            cfg.id + "_run" + run_id
        );

        // Save predicted calib outputs
        Data.saveMatrixCsv(Metrics::addRunIdToFilename(cfg.pred_calib, run_id), Y_pred_calib, colNames);

        // Save final weights
        MLP.saveWeightsCsv(Metrics::addRunIdToFilename(cfg.weights_csv, run_id));
        MLP.weightsToVectorMlp();
        MLP.saveWeightsVectorCsv(Metrics::addRunIdToFilename(cfg.weights_vec_csv, run_id));

        // Save final weights into one combined file
        MLP.appendWeightsVectorCsv(cfg.weights_vec_csv, run == 0);

        // ------------------------------------------------------
        // Evaluate validation set
        // ------------------------------------------------------
        MLP.calculateOutputs(X_valid);

        Eigen::MatrixXd Y_pred_valid = MLP.getOutputs();
        Eigen::MatrixXd Y_true_valid = Y_valid;

        try {
            Y_true_valid = Data.inverseTransformOutputs(Y_true_valid);
            Y_pred_valid = Data.inverseTransformOutputs(Y_pred_valid);
        } catch (const std::exception &ex) {
            std::cerr << "[Warning] inverseTransformOutputs failed (valid): " << ex.what() << "\n";
        }

        Metrics::computeAndAppendFinalMetrics(
            Y_true_valid, Y_pred_valid,
            cfg.metrics_val,   // same file across runs
            cfg.id + "_run" + run_id
        );

        Data.saveMatrixCsv(Metrics::addRunIdToFilename(cfg.pred_valid, run_id), Y_pred_valid, colNames);

        std::cout << "-> Metrics saved." << std::endl;

        std::cout << "Run " << (run+1) << " finished." << "\n";
        std::cout << "-------------------------------------------" << std::endl;

    }

    std::cout << "\n[I/O] Saved REAL CALIB data to: '" << cfg.real_calib << "', and PRED CALIB data to '" << cfg.pred_calib << "' \n";
    std::cout << "[I/O] Saved REAL VALID data to: '" << cfg.real_valid << "', and PRED VALID data to '" << cfg.pred_valid<< "' \n";
    std::cout << "[I/O] Saved INIT weights vector to: '" << cfg.weights_vec_csv_init << "'\n";
    std::cout << "[I/O] Saved FINAL weights vector to: '" << cfg.weights_vec_csv << "'\n";
    std::cout << "[I/O] Saved RUN INFO to: '" << cfg.run_info << "' \n";
    std::cout << "[I/O] Saved CALIB METRICS to: '" << cfg.metrics_cal << "' \n";
    std::cout << "[I/O] Saved VALID METRICS to: '" << cfg.metrics_val << "' \n";

    std::cout << "\n-------------------------------------------" << std::endl;
    std::cout << "-- JKMNet FINISHED --" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    
    return 0;
}
