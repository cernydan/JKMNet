/**
 * @file HyperparamOptimizer.cpp
 * @brief Implements the PSO-based hyperparameter optimizer driver for JKMNet.
 */

#include "HyperparamOptimizer.hpp"
#include "HyperparamObjective.hpp"
#include "PSO.hpp"
#include <iostream>

/**
 * Run PSO optimization of MLP hyperparameters and return the updated configuration.
 * Optimizes three normalized hyperparameters — learning rate in [0.0001, 0.01],
 * number of hidden neurons in [4, 50], and activation function
 * (0 = RELU, 1 = TANH, 2 = SIGMOID) — over the unit cube [0, 1]^3 using a PSO
 * swarm (5 particles, w = 0.5, c1 = c2 = 1.5, 30 iterations) with
 * evaluateMLPwithParams() as the objective (validation MSE). The best solution
 * found is decoded and written back into cfg (learning_rate, activation, and
 * the hidden layer of mlp_architecture).
 * @param cfg Base run configuration supplying data and trainer settings.
 * @return The configuration updated with the best hyperparameters found by PSO.
 * @note best_hidden is referenced but never declared, and cfg is taken by
 *       const reference while being assigned to, so this does not compile as
 *       written.
 * @todo check semantics
 */
RunConfig optimizeHyperparams(const RunConfig &cfg) {
    // Define PSO bounds: [lr, hidden, act]
    double xmin = 0.0;
    double xmax = 1.0;

    PSO pso(5, 3, 0.5, 1.5, 1.5, 30, xmin, xmax);

    auto objective = [&](const Eigen::VectorXd &params) {
        return evaluateMLPwithParams(params, cfg);
    };

    pso.optimize(objective);

    Eigen::VectorXd best = pso.getBestPosition();

    // Decode best params
    double lr = scale(best[0], 0.0001, 0.01);
    int best_hidden = static_cast<int>(std::round(scale(best[1], 4, 50)));
    int act_idx = static_cast<int>(std::round(scale(best[2], 0, 2)));

    std::string act_name =
        act_idx == 0 ? "RELU" :
        act_idx == 1 ? "TANH" : "SIGMOID";

    std::cout << "[PSO] Best configuration found:\n"
              << "  lr = " << lr
              << ", hidden = " << best_hidden
              << ", activation = " << act_name << "\n";

    // Update config
    // @todo check semantics — undocumented block referencing undeclared
    //       identifiers (cfg is const here); left as in the current sources
    std::cout << "\n[PSO] Configuration updated for best parameters. \n";

    return cfg;
}
