// ********* 16. 10. 2025 *********
// **DONE**: Change 'std::cout' into 'clog'
// TODO: Change 'if (cfg_.trainer == "online"){}' into 'switch(){}'
// TODO: Make 'JKMNet::init_mlps()' parallel, but has to be without 'push_back()'
// TODO: Add calculation of all metrics during training (and validation), not only MSE
// TODO: Choice in config for saving metrics for all epochs, last epoch, every x-th epoch, ...
// TODO: Choice in config for saving predicted values for all epochs, last epoch, every x-th epoch, ... 
// TODO: Create separate method for predictions (validation), i.e., read final weights from file and calculate outputs

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
#include <iostream>

int main(int argc, char** argv) {

    // Read and/or set the number of threads for parallelization
    unsigned nthreads = 1;
    if (argc > 1) {
        try {
            int valueThread = std::stoi(argv[1]);
            if (valueThread > 0) nthreads = valueThread;
        } catch (...) {
            std::cerr << "[Warning] Invalid thread argument. Using 1.\n";
        }
    }

    // Read configuration file
    RunConfig cfg = parseConfigIni("settings/config_model.ini");

    // Run PSO optimization
    if (cfg.pso_optimize) {
        cfg = optimizeHyperparams(cfg);
    }

    // Clean all files in the output folder
    Data::cleanAllOutputs(cfg.out_dir);

    std::cout << "\n===========================================\n";
    std::cout << " Running Ensemble\n";
    std::cout << "===========================================\n";
    JKMNet net_(cfg, nthreads);
    net_.ensembleRunMlpVector();

    return 0;
}

