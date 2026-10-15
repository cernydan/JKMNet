/**
 * @file HyperparamObjective.cpp
 * @brief Implements the objective wrapper that evaluates one MLP configuration for the PSO optimizer.
 */

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
 * Helper to put values into range.
 * Performs a linear map of a normalized coordinate x from [0, 1] onto [xmin, xmax].
 * @param x Normalized coordinate, expected in [0, 1].
 * @param xmin Lower bound of the target range.
 * @param xmax Upper bound of the target range.
 * @return The value xmin + x * (xmax - xmin).
 */
double scale(double x, double xmin, double xmax) {
    return xmin + x * (xmax - xmin);
}

/**
 * Evaluate a single MLP configuration defined by PSO params.
 * Decodes the normalized PSO vector into concrete hyperparameters — learning
 * rate in [0.0001, 0.01], number of hidden neurons in [4, 50], and activation
 * function (0 = RELU, 1 = TANH, 2 = SIGMOID) — then loads and transforms the
 * data according to cfg, builds a three-layer MLP
 * (inputs - hidden_neurons - outputs), trains it with the trainer selected in
 * cfg (online or batch Adam), and returns the validation MSE as the objective
 * value to minimize.
 * @param params Normalized PSO position of size 3 with components in [0, 1].
 * @param cfg Run configuration supplying the data file, transforms, trainer
 *        settings, and the input/output architecture.
 * @return Validation MSE of the trained MLP; returns +infinity if evaluation fails.
 * @note Uses cfg.learning_rate rather than the decoded learning rate
 *       @todo check semantics
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
        data.setTransform(strVecToTransformTypes(cfg.transform),
                          cfg.transform_alpha,
                          cfg.exclude_last_col_from_transform);
        data.applyTransform();

        int out = static_cast<int>(cfg.mlp_architecture.back());

        auto [X_train, Y_train, X_valid, Y_valid, pat_indices, calIdxForUnshuffle] = data.makeMats(cfg.input_numbers,
                                                                  static_cast<int>(cfg.mlp_architecture.back()),
                                                                  cfg.train_fraction,
                                                                  cfg.shuffle,
                                                                  cfg.seed);

        int inp = static_cast<int>(X_train.cols());

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

        if (cfg.trainer == "online") {
            mlp.onlineAdam(
                cfg.max_iterations, cfg.max_error,
                cfg.learning_rate,  X_train, Y_train
            );
        } else {
            mlp.batchAdam(
                cfg.max_iterations, cfg.max_error,
                cfg.batch_size, cfg.learning_rate,
                X_train, Y_train
            );

        }

        // ------------------------------------------------------------
        // Evaluate on validation data
        // ------------------------------------------------------------
        mlp.calculateOutputs(X_valid);
        Eigen::MatrixXd Y_pred = mlp.getOutputs();

        double mse = Metrics::mse(Y_valid, Y_pred);

        return mse;

    } catch (const std::exception &e) {
        std::cerr << "[PSO] Evaluation failed: " << e.what() << "\n";
        return std::numeric_limits<double>::infinity(); // penalize invalid configs
    }
}
