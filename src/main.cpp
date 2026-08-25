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
#include "LSTMLayer.hpp"

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

    // Select model type from config
    std::string modelType = cfg.model_type;
    std::transform(modelType.begin(), modelType.end(), modelType.begin(), ::tolower);

    if (modelType == "mlp") {
        std::cout << "[INFO] Using MLP model\n";
        net_.ensembleRunMlpVector();
    } else if (modelType == "lstm_first" || modelType == "lstmfirst") {
        std::cout << "[INFO] Using LSTM model (past data only)\n";
        net_.ensembleLstmFirstTest();
    } else if (modelType == "lstm_past_future" || modelType == "lstmpastfuture") {
        std::cout << "[INFO] Using LSTM model (past + future data)\n";
        net_.ensembleLstmPastFutureTest();
    } else {
        std::cerr << "[Error] Unknown model_type: " << cfg.model_type
                  << ". Valid options: mlp, lstm_first, lstm_past_future\n";
        return 1;
    }

    return 0;


// //// LSTM TEST

//     std::vector<std::string> vars = {"T3","ET","prec", "moisture"};
//     std::vector<std::string> trans = {"MINMAX","MINMAX","MINMAX","NONLINEAR"};
//     std::unordered_set<std::string> idt = {"93148340"};
//     int histts = 30;
//     int futts = 3;
//     int firstpartouts = 50;
//     std::vector<unsigned int> mlparch = {50,1};
//     std::vector<activ_func_type> mlpact = {activ_func_type::SIGMOID,activ_func_type::RELU};
//     std::vector<weight_init_type> mlpinit = {weight_init_type::XG, weight_init_type::HE};

//     std::cout << "-> Loading data..." << std::endl;
//     Data data_;
//     data_.loadFilteredCSV("data/inputs/data_all_daily_eddy.csv", idt , vars, "date", "ID");
//     std::cout << "-> Data loaded." << std::endl;

//     std::cout << "-> Transforming data..." << std::endl;
//     data_.setTransform(strVecToTransformTypes(trans),
//                        5.0,
//                        false);
//     data_.applyTransform();
//     std::cout << "-> Data transformed." << std::endl;
//     std::cout<<data_.numericData().rows();

//     LSTMLayer hist;
//     hist.initLSTMLayer(vars.size(),firstpartouts,histts,histts,true,"XG",0);
//     LSTMLayer fut;
//     fut.initLSTMLayer(vars.size() - 1,firstpartouts,futts,futts,true,"XG",0);
//     LSTMLayer toget;
//     toget.initLSTMLayer(firstpartouts,firstpartouts,histts + futts,futts,false,"XG",0);
//     MLP final;
    
//     final.setArchitecture(mlparch);
//     final.setActivations(mlpact);
//     final.setWInitType(mlpinit);
//     Eigen::VectorXd x0 = Eigen::VectorXd::Zero(firstpartouts);
//     final.initMLP(x0, 0);

//     for(int it = 0; it < 500; it++){
//         for(int i = 0; i < 800; i++){
//             Eigen::MatrixXd hinp = data_.numericData().block(0+i,0,histts,vars.size());
//             Eigen::MatrixXd finp = data_.numericData().block(histts+i,0,futts,vars.size() - 1);
//             Eigen::MatrixXd obs = data_.numericData().block(histts+i,vars.size() - 1,futts,1);

//             if (!(hinp.array().isFinite().all() &&
//                 finp.array().isFinite().all() &&
//                 obs.array().isFinite().all())) {
//                 continue;
//             }
            
//             double lr = 0.01;
//             if (it > 350){ lr = 0.00005; }
//             else if (it > 300){ lr = 0.0005; }
//             else if (it > 200){ lr = 0.001; }
//             else if (it > 10) { lr = 0.005; }
//             else              { lr = 0.01;  }

//             hist.setInputTSSegment(hinp);
//             fut.setInputTSSegment(finp);
//             hist.calculateTimeSteps();
//             fut.calculateTimeSteps();
//             Eigen::MatrixXd betw = Eigen::MatrixXd::Zero(histts + futts,firstpartouts);
//             betw.block(0,0,histts,firstpartouts) = hist.getForwardOutput();
//             betw.block(histts,0,futts,firstpartouts) = fut.getForwardOutput();
//             toget.setInputTSSegment(betw);
//             toget.calculateTimeSteps();

//             Eigen::MatrixXd ltom = toget.getForwardOutput().transpose();
//             Eigen::MatrixXd deltaMtol = Eigen::MatrixXd(ltom.rows(), ltom.cols());

//             for(int p = 0; p < ltom.cols(); p++){
//                 final.runAndBP(ltom.col(p),obs.row(p),lr);
//                 deltaMtol.col(p) = final.getFirstLayerInputDelta();
//                 std::cout<< "mod "<<p<<" : "<<final.getOutput()<<"\n";
//             }
 
//             toget.setDeltaFromNextLayer(deltaMtol);
//             toget.calculateGradients();
//             Eigen::MatrixXd futdelt = toget.getDeltaInputs().block(0,histts,firstpartouts,futts);
//             Eigen::MatrixXd histdelt = toget.getDeltaInputs().block(0,0,firstpartouts,histts);
//             fut.setDeltaFromNextLayer(futdelt);
//             hist.setDeltaFromNextLayer(histdelt);
//             fut.calculateGradients();
//             hist.calculateGradients();
//             toget.updateWeights(lr);
//             fut.updateWeights(lr);
//             hist.updateWeights(lr);
//             toget.eraseMemory();
//             fut.eraseMemory();
//             hist.eraseMemory();

//             std::cout<<"obs:  "<<obs.transpose()<<"\n\n";
//             //std::cout<<"lstmout:  "<<ltom<<"\n\n";
//         }
//     }

//More ids test

    

    // Data jendat;
    // auto Xcal = jendat.loadPreparedMatrix("data/inputs/Xcal.bin");
    // auto Xval = jendat.loadPreparedMatrix("data/inputs/Xval.bin");
    // auto Ycal = jendat.loadPreparedMatrix("data/inputs/Ycal.bin");
    // auto Yval = jendat.loadPreparedMatrix("data/inputs/Yval.bin");

    // MLP final;
    // final.setArchitecture({30,3});
    // final.setActivations({activ_func_type::SIGMOID,activ_func_type::SIGMOID});
    // final.setWInitType({weight_init_type::XG,weight_init_type::XG});
    // Eigen::VectorXd x0 = Eigen::VectorXd::Zero(Xcal.cols());
    // final.initMLP(x0, 0);

    //     final.onlineAdam(10,0.0,0.01,Xcal,Ycal);
    //     std::vector<int> perm;
    //     perm = jendat.permutationVector(static_cast<int>(Xcal.rows()));
    //     Xcal = jendat.shuffleMatrix(Xcal, perm);
    //     Ycal = jendat.shuffleMatrix(Ycal, perm);

    //     final.onlineAdam(20,0.0,0.005,Xcal,Ycal);
    //     Xcal = jendat.shuffleMatrix(Xcal, perm);
    //     Ycal = jendat.shuffleMatrix(Ycal, perm);

    //     final.onlineAdam(30,0.0,0.001,Xcal,Ycal);
    //     Xcal = jendat.shuffleMatrix(Xcal, perm);
    //     Ycal = jendat.shuffleMatrix(Ycal, perm);

    //     final.onlineAdam(40,0.0,0.0005,Xcal,Ycal);
    //     Xcal = jendat.shuffleMatrix(Xcal, perm);
    //     Ycal = jendat.shuffleMatrix(Ycal, perm);

    //     final.onlineAdam(50,0.0,0.0001,Xcal,Ycal);
    //     Xcal = jendat.shuffleMatrix(Xcal, perm);
    //     Ycal = jendat.shuffleMatrix(Ycal, perm);


    // final.calculateOutputs(Xval);
    // Eigen::MatrixXd mod_out = final.getOutputs();
    // Eigen::VectorXd bef = Eigen::VectorXd::Zero(3);

    // for (int c = 0; c < Yval.cols(); ++c) {
    //     bef(c) = (Metrics::pi(Yval.col(c).eval(), mod_out.col(c).eval()));
    // }

    // std::cout<<bef<<"\n\n";

    // jendat.saveMatrixCsv(Metrics::addRunIdToFilename("data/outputs/model.csv", "1"),mod_out,{"h1","h2","h3"}); 
    // jendat.saveMatrixCsv(Metrics::addRunIdToFilename("data/outputs/real.csv", "1"),Yval,{"h1","h2","h3"}); 
}
