// ********* 16. 10. 2025 *********
// **DONE**: Change 'std::cout' into 'clog'
// **DONE**: Change 'if (cfg_.trainer == "online"){}' into 'switch(){}'
// **DONE**: Make 'JKMNet::init_mlps()' parallel, but has to be without 'push_back()'
// **DONE**: Add calculation of all metrics during training (and validation), not only MSE
// **DONE**: Choice in config for saving metrics for all epochs, last epoch, every x-th epoch, ...
// TODO: Choice in config for saving predicted values for all epochs, last epoch, every x-th epoch, ... 
// **DONE**: Put 'metricsAfterXEpochs' value into 'config_model.ini'
// **DONE**: Create separate method for predictions (validation), i.e., read final weights from file and calculate outputs
// TODO: Update saving weights in predictions
// **DONE**: Catch if predict is run without any weights saved yet
// TODO: Activ func in predictation mode from config file

// ********* [PSO] *********
// TODO: [PSO] Save PSO best hyperparams into 'config_model.ini' for MLP ensemble run
// TODO: [PSO] Add all activation functions into PSO optim
// TODO: [PSO] Add more hyperparams into PSO optim, i.e., architecture, weight_init, trainer, ...
// TODO: [PSO] Read settings of the optimization from file, e.g., 'settings/settings_pso.ini'
// TODO: [PSO] Change randomization in PSO using seed from 'config_model.ini' (?)
// TODO: [PSO] Increase params of PSO, i.e., swarm size, max iteration, ... (in HyperparamOptimizer.cpp)


#include "ConfigIni.hpp"
#include "Data.hpp"
#include "JKMNet.hpp"
#include "PSO.hpp"
#include "HyperparamObjective.hpp"
#include "HyperparamOptimizer.hpp"
#include "CNN.hpp"

#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char** argv) {
    unsigned nthreads = 1;
    bool predictMode = false;
    std::string weightsPath;

    // -------------------------------------------------------
    // Parse CLI arguments
    // -------------------------------------------------------
    if (argc > 1) {
        std::string arg1 = argv[1];

        // Check if the first argument is "predict"
        if (arg1 == "predict") {  // RUN: ./bin/JKMNet predict
            predictMode = true;

            // Optional second argument: path to weights
            if (argc > 2) {   // RUN: ./bin/JKMNet predict data/outputs/weights/weights_final_1.csv  // TODO: ensemble for all
                weightsPath = argv[2];
            }

            // Optional third argument: number of threads
            if (argc > 3) {  // RUN: ./bin/JKMNet predict data/outputs/weights/weights_final_1.csv 4
                try {
                    int valueThread = std::stoi(argv[3]);
                    if (valueThread > 0) nthreads = valueThread;
                } catch (...) {
                    std::cerr << "[Warning] Invalid thread argument. Using 1.\n";
                }
            }
        } 
        else {
            // Not predict mode, so treat argv[1] as thread count
            try {
                int valueThread = std::stoi(arg1);
                if (valueThread > 0) nthreads = valueThread;
            } catch (...) {
                std::cerr << "[Warning] Invalid thread argument. Using 1.\n";
            }
        }
    }

    // -------------------------------------------------------
    // Load configuration
    // -------------------------------------------------------
    std::string cfg_path = "settings/config_model.ini";
    RunConfig cfg = parseConfigIni(cfg_path);

    // -------------------------------------------------------
    // PREDICTION MODE
    // -------------------------------------------------------
    if (predictMode) {
        if (weightsPath.empty()) {
            weightsPath = cfg.weights_csv; // fallback to default from ini
        }

        std::cout << "\n===========================================\n";
        std::cout << " Prediction mode\n";
        std::cout << "===========================================\n";

        // Check if weights file exists
        if (!std::filesystem::exists(weightsPath)) {
            std::cerr << "[Error] Weights file not found: " << weightsPath << "\n";
            std::cerr << "        Please train the model first or specify a valid weights path.\n";
            std::cerr << "        Hint: ./bin/JKMNet [threads]\n";
            return 1;
        }

        try {
            JKMNet net_(cfg, nthreads);
            net_.predictFromSavedWeights(weightsPath);
        } catch (const std::exception &ex) {
            std::cerr << "[Error] Prediction failed: " << ex.what() << "\n";
            return 1;
        }

        return 0;
    }

    // -------------------------------------------------------
    // TRAINING MODE (ENSEMBLE)
    // -------------------------------------------------------
    if (cfg.pso_optimize) {
        cfg = optimizeHyperparams(cfg);
    }

    Data::cleanAllOutputs(cfg.out_dir);

    std::cout << "\n===========================================\n";
    std::cout << " Running Ensemble\n";
    std::cout << "===========================================\n";
    JKMNet net_(cfg, nthreads);
    net_.ensembleRunMlpVector();
    
    return 0;
}