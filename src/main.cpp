// ********* 6. 10. 2025 *********
// TODO: Add parallelization of the model using openML
// TODO: [PSO] Save PSO best hyperparams into 'config_model.ini' for MLP ensemble run
// TODO: [PSO] Add all activation functions into PSO optim
// TODO: [PSO] Add more hyperparams into PSO optim, i.e., architecture, weight_init, trainer, ...
// TODO: [PSO] Read settings of the optimization from file, e.g., 'settings/settings_pso.ini'
// TODO: [PSO] Change randomization in PSO using seed from 'config_model.ini' (?)
// TODO: [PSO] Increase params of PSO, i.e., swarm size, max iteration, ... (in HyperparamOptimizer.cpp)


#include "ConfigIni.hpp"
#include "Data.hpp"
#include "EnsembleRunner.hpp"
#include "PSO.hpp"
#include "HyperparamObjective.hpp"
#include "HyperparamOptimizer.hpp"
#include <iostream>

int main(int argc, char** argv) {
    unsigned nthreads = 1;
    if (argc > 1) {
        try {
            int valueThread = std::stoi(argv[1]);
            if (valueThread > 0) nthreads = valueThread;
        } catch (...) {
            std::cerr << "[Warning] Invalid thread argument. Using 1.\n";
        }
    }

    RunConfig cfg = parseConfigIni("settings/config_model.ini");
    // cfg = optimizeHyperparams(cfg);

    Data::cleanAllOutputs(cfg.out_dir);

    std::cout << "\n===========================================\n";
    std::cout << " Running Ensemble with Optimized Parameters\n";
    std::cout << "===========================================\n";
    EnsembleRunner runner(cfg, nthreads);
    runner.run();

    return 0;
}

