#include "HyperparamObjective.hpp"
#include "ConfigIni.hpp"
#include "Data.hpp"
#include "Metrics.hpp"
#include "JKMNet.hpp"
#include "MLP.hpp"
#include <iostream>
#include <cmath>
#include <unordered_set>


/**
 * Helper to put values into range
 */
double scale(double x, double xmin, double xmax) { 
    return xmin + x * (xmax - xmin);
}

/**
 * Evaluate a single MLP configuration defined by PSO params
 */
double evaluateMLPwithParams(const Eigen::VectorXd &params, const RunConfig &cfg) {
    try {
        // ------------------------------------------------------------
        // Decode PSO parameters
        // ------------------------------------------------------------
        double lr = scale(params[0], 0.0001, 0.01);
        int hidden_neurons = static_cast<int>(std::round(scale(params[1], 4, 50)));
        int act_idx = static_cast<int>(std::round(scale(params[2], 0, 2)));

        activ_func_type act;
        switch (act_idx) {
            case 0: act = activ_func_type::RELU; break;
            case 1: act = activ_func_type::TANH; break;
            default: act = activ_func_type::SIGMOID; break;
        }

        // ------------------------------------------------------------
        // Load and preprocess data (same as EnsembleRunner.cpp)
        // ------------------------------------------------------------
        Data data;
        std::unordered_set<std::string> idFilter;
        if (!cfg.id.empty()) {
            auto ids = parseStringList(cfg.id);
            idFilter = std::unordered_set<std::string>(ids.begin(), ids.end());
        }

        data.loadFilteredCSV(cfg.data_file, idFilter, cfg.columns, cfg.timestamp, cfg.id_col);
        data.setTransform(strToTransformType(cfg.transform),
                          cfg.transform_alpha,
                          cfg.exclude_last_col_from_transform);
        data.applyTransform();

        data.makeCalibMatsSplit(cfg.input_numbers, static_cast<int>(cfg.mlp_architecture.back()));
        auto [trainMat, validMat, _, __] = data.splitCalibMatWithIdx(cfg.train_fraction,
                                                                     cfg.split_shuffle,
                                                                     cfg.seed);

        int inp = static_cast<int>(trainMat.cols()) - static_cast<int>(cfg.mlp_architecture.back());
        int out = static_cast<int>(cfg.mlp_architecture.back());

        auto [X_train, Y_train] = data.splitInputsOutputs(trainMat, inp, out);
        auto [X_valid, Y_valid] = data.splitInputsOutputs(validMat, inp, out);

        // ------------------------------------------------------------
        // Build and train MLP
        // ------------------------------------------------------------
        MLP mlp;
        JKMNet net;

        std::vector<unsigned> arch = {static_cast<unsigned>(inp),
                                      static_cast<unsigned>(hidden_neurons),
                                      static_cast<unsigned>(out)};
        mlp.setArchitecture(arch);
        mlp.setActivations(std::vector<activ_func_type>(arch.size(), act));
        mlp.setWInitType(std::vector<weight_init_type>(arch.size(), strToWeightInit(cfg.weight_init)));

        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(inp);
        mlp.initMLP(x0, cfg.seed);

        TrainingResult tr;
        if (cfg.trainer == "online") {
            tr = net.trainAdamOnline(mlp, X_train, Y_train,
                                     cfg.max_iterations, cfg.max_error,
                                     lr, cfg.shuffle, cfg.seed);
        } else {
            tr = net.trainAdamBatch(mlp, X_train, Y_train,
                                    cfg.batch_size, cfg.max_iterations,
                                    cfg.max_error, lr,
                                    cfg.shuffle, cfg.seed);
        }

        // ------------------------------------------------------------
        // Evaluate on validation data
        // ------------------------------------------------------------
        mlp.calculateOutputs(X_valid);
        Eigen::MatrixXd Y_pred = mlp.getOutputs();

        double rmse = Metrics::rmse(Y_valid, Y_pred);

        if (!std::isfinite(rmse))
            rmse = 1e6; // penalize invalid runs

        std::cout << "[evaluateMLPwithParams] Particle: lr = " << lr
                  << ", hidden = " << hidden_neurons
                  << ", act = " << act_idx
                  << ", RMSE = " << rmse << "\n";

        return rmse;
    } catch (const std::exception &e) {
        std::cerr << "[evaluateMLPwithParams] Error: " << e.what() << std::endl;
        return 1e6;
    }
}
