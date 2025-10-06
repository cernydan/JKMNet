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

#include "ConfigIni.hpp"
#include "Data.hpp"
#include "EnsembleRunner.hpp"
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
    Data::cleanAllOutputs(cfg.out_dir);

    EnsembleRunner runner(cfg, nthreads);
    runner.run();

    return 0;
}
