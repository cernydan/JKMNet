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
    net_.ensembleRunMlpVector();
    
    return 0;


/// LSTM TEST

    // std::cout << "-> Loading data..." << std::endl;
    // Data data_;
    // data_.loadFilteredCSV("data/inputs/data_all_daily.csv", {"93148340"}, {"T1", "T2", "T3", "moisture"}, "date", "ID");
    // std::cout << "-> Data loaded." << std::endl;

    // std::cout << "-> Transforming data..." << std::endl;
    // data_.setTransform(strVecToTransformTypes({"MINMAX","MINMAX","MINMAX","NONLINEAR"}),
    //                    0.9,
    //                    false);
    // data_.applyTransform();
    // std::cout << "-> Data transformed." << std::endl;
    // std::cout<<data_.numericData().rows();

    // LSTMLayer hist;
    // hist.initLSTMLayer(4,50,30,30,true,"RANDOM",0,0.0,0.1);
    // LSTMLayer fut;
    // fut.initLSTMLayer(3,50,3,3,true,"RANDOM",0,0.0,0.1);
    // LSTMLayer toget;
    // toget.initLSTMLayer(50,50,33,3,false,"RANDOM",0,0.0,0.1);
    // MLP final;
    
    // final.setArchitecture({20,20,3});
    // final.setActivations({activ_func_type::SIGMOID,activ_func_type::SIGMOID,activ_func_type::SIGMOID});
    // final.setWInitType({weight_init_type::HE,weight_init_type::HE,weight_init_type::HE});
    // Eigen::VectorXd x0 = Eigen::VectorXd::Zero(150);
    // final.initMLP(x0, 0);

    // for(int it = 0; it < 500; it++){
    //     for(int i = 0; i < 800; i++){
    //         Eigen::MatrixXd hinp = data_.numericData().block(0+i,0,30,4);
    //         Eigen::MatrixXd finp = data_.numericData().block(30+i,0,3,3);
    //         Eigen::VectorXd obs = data_.numericData().block(30+i,3,3,1);

    //         if (!(hinp.array().isFinite().all() &&
    //             finp.array().isFinite().all() &&
    //             obs.array().isFinite().all())) {
    //             continue;
    //         }
            
    //         int lr = 0.01;
    //         if (it > 10){
    //             lr = 0.005;
    //         } else if(it > 200){
    //             lr = 0.001;
    //         } else if(it > 300){
    //             lr = 0.0005;
    //         }else if(it > 350){
    //             lr = 0.00005;
    //         }

    //         hist.setInputTSSegment(hinp);
    //         fut.setInputTSSegment(finp);
    //         hist.calculateTimeSteps();
    //         fut.calculateTimeSteps();
    //         Eigen::MatrixXd betw = Eigen::MatrixXd::Zero(33,50);
    //         betw.block(0,0,30,50) = hist.getForwardOutput();
    //         betw.block(30,0,3,50) = fut.getForwardOutput();
    //         toget.setInputTSSegment(betw);
    //         toget.calculateTimeSteps();
    //         Eigen::VectorXd ltom = Eigen::VectorXd(150);
    //         ltom = toget.getForwardOutputVector();
    //         final.runAndBP(ltom,obs,lr);  
    //         toget.setDeltaFromNextLayer(final.getFirstLayerInputDelta());
    //         toget.calculateGradients();
    //         Eigen::MatrixXd futdelt = toget.getDeltaInputs().block(0,30,50,3);
    //         Eigen::MatrixXd histdelt = toget.getDeltaInputs().block(0,0,50,30);
    //         fut.setDeltaFromNextLayer(futdelt);
    //         hist.setDeltaFromNextLayer(histdelt);
    //         fut.calculateGradients();
    //         hist.calculateGradients();
    //         toget.updateWeights(lr);
    //         fut.updateWeights(lr);
    //         hist.updateWeights(lr);
    //         toget.eraseMemory();
    //         fut.eraseMemory();
    //         hist.eraseMemory();

    //         //std::cout<<"lstmout:  "<<ltom.transpose()<<"\n";
    //         std::cout<<"obs:  "<<obs.transpose()<<"\n\n";
    //         std::cout<<"mod:  "<<final.getOutput().transpose()<<"\n\n\n";
    //     }
    // }

//More ids test

    // std::cout << "-> Loading data..." << std::endl;
    // int trainIds = 7;
    // std::vector<std::unordered_set<std::string>> ids_vec = {{"94951006"} ,{"94208511"},{"94951014"},{"95228849"},{"95228106"},{"94208530"},{"94212708"},
    //                                                         {"95228846"},{"94212735"},{"94951047"},{"94249013"},{"95228110"},{"93148340"},{"94212724"}};
    // std::vector<std::string> keep_cols = {"T3","prec","moisture"};
    // std::vector<std::string> transform_cols = {"MINMAX","MINMAX","NONLINEAR"};
    // std::vector<std::vector<int>> inps = {{-1,0,1,2},{-3,-2,-1,0,1,2},{-7,-6,-5,-4,-3,-2,-1}};
    // std::vector<Data> data_vec(ids_vec.size());
    // std::vector<Eigen::MatrixXd> Xs(ids_vec.size());
    // std::vector<Eigen::MatrixXd> Ys(ids_vec.size());
    // for(size_t i = 0; i < ids_vec.size() ; i++){
    //     data_vec[i].loadFilteredCSV("data/inputs/data_all_daily_eddy.csv", ids_vec[i], keep_cols, "date", "ID");
    //     data_vec[i].setTransform(strVecToTransformTypes(transform_cols), 5.0, false);
    //     data_vec[i].applyTransform();
    //     if(i < trainIds){
    //         auto mats = data_vec[i].makeMats(inps, 3, 0.99, true, 0);
    //         Xs[i] = std::get<0>(mats);
    //         Ys[i] = std::get<1>(mats);
    //     } else {
    //         auto mats = data_vec[i].makeMats(inps, 3, 0.01, true, 0);
    //         Xs[i] = std::get<2>(mats);
    //         Ys[i] = std::get<3>(mats);
    //     }

    // }
    // std::cout << "-> Data loaded, transformed,split into training and validation sets." << std::endl;
    
    // MLP final;
    // final.setArchitecture({20,20,3});
    // final.setActivations({activ_func_type::SIGMOID,activ_func_type::SIGMOID,activ_func_type::RELU});
    // final.setWInitType({weight_init_type::HE,weight_init_type::HE,weight_init_type::HE});
    // Eigen::VectorXd x0 = Eigen::VectorXd::Zero(17);
    // final.initMLP(x0, 0);

    // for(int i = 0; i < 2 ; i++){
    //     for(int j = 0 ; j < trainIds ; j++){
    //         final.onlineAdam(50,0.0,0.005,Xs[j],Ys[j]);
    //     }
    //     for(int j = trainIds - 1 ; j >= 0; j--){
    //         final.onlineAdam(50,0.0,0.005,Xs[j],Ys[j]);
    //     }
    // }

    // for (size_t i = trainIds ; i < ids_vec.size() ; i++ ){
    //     final.calculateOutputs(Xs[i]);
    //     Eigen::MatrixXd mod_out = final.getOutputs();
    //     Eigen::VectorXd bef = Eigen::VectorXd::Zero(3);
    //     Eigen::VectorXd aft = Eigen::VectorXd::Zero(3);
    //     Eigen::MatrixXd Y_val = Ys[i];
    //     for (int c = 0; c < Y_val.cols(); ++c) {
    //         bef(c) = (Metrics::pi(Y_val.col(c).eval(), mod_out.col(c).eval()));
    //     }

    //     Y_val = data_vec[i].inverseTransformOutputs(Y_val);
    //     mod_out = data_vec[i].inverseTransformOutputs(mod_out);

    //     for (int c = 0; c < Y_val.cols(); ++c) {
    //         aft(c) = (Metrics::pi(Y_val.col(c).eval(), mod_out.col(c).eval()));
    //     }
    //     std::cout<<"val id "<<i<<"\n\n";
    //     std::cout<<bef<<"\n\n";
    //     std::cout<<aft<<"\n\n";

    //     data_vec[i].saveMatrixCsv(Metrics::addRunIdToFilename("data/outputs/real.csv", std::to_string(i)),Y_val,{"h1","h2","h3"});
    //     data_vec[i].saveMatrixCsv(Metrics::addRunIdToFilename("data/outputs/model.csv", std::to_string(i)),mod_out,{"h1","h2","h3"});
    // }
}