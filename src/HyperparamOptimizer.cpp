#include "HyperparamOptimizer.hpp"
#include "HyperparamObjective.hpp"
#include "PSO.hpp"
#include "ConfigIni.hpp"
#include <iostream>
#include <cmath>

/**
 * Run PSO optimization and update config accordingly
 */
RunConfig optimizeHyperparams(const RunConfig& cfg_in) {
    std::cout << "===========================================\n";
    std::cout << " Starting PSO hyperparameter optimization\n";
    std::cout << "===========================================\n";

    RunConfig cfg = cfg_in;

    // Set up PSO
    int dim = 3; // // number of hyperparameters being optimized - learning_rate, hidden_neurons, activation
    PSO pso(
        3,    // swarm size
        dim,
        0.7,  // inertia weight (w)
        1.4,  // cognitive coefficient (c1)
        1.4,  // social coefficient  (c2)
        2,   // max iterations
        0.0001,  // xmin
        0.1      // xmax
    );

    auto objective = [&](const Eigen::VectorXd &x) {
        return evaluateMLPwithParams(x, cfg);
    };

    pso.optimize(objective);

    Eigen::VectorXd best = pso.getBestPosition();

    double best_lr = scale(best[0], 0.0001, 0.01);
    int best_hidden = (int)std::round(scale(best[1], 4, 50));
    int act_idx = (int)std::round(scale(best[2], 0, 2));

    std::string act_name = "RELU";
    if (act_idx == 1) act_name = "TANH";
    else if (act_idx == 2) act_name = "SIGMOID";

    std::cout << "\n[PSO] Best parameters found:\n";
    std::cout << " - Learning rate: " << best_lr << "\n";
    std::cout << " - Hidden neurons: " << best_hidden << "\n";
    std::cout << " - Activation: " << act_name << "\n";
    std::cout << " - RMSE: " << pso.getBestValue() << "\n";

    // Update config
    cfg.learning_rate = best_lr;
    cfg.mlp_architecture = {
        (unsigned)cfg.input_numbers.size(),
        (unsigned)best_hidden,
        (unsigned)cfg.mlp_architecture.back()
    };
    cfg.activation = act_name;

    std::cout << "\n[PSO] Configuration updated for best parameters. \n";

    return cfg;
}
