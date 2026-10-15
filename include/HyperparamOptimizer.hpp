/**
 * @file HyperparamOptimizer.hpp
 * @brief Declares the PSO-based hyperparameter optimizer driver for JKMNet.
 */

#ifndef HYPERPARAM_OPT_HPP
#define HYPERPARAM_OPT_HPP

#include "ConfigIni.hpp"

RunConfig optimizeHyperparams(const RunConfig &cfg);  //!< Run PSO optimization of MLP hyperparameters and return the updated configuration

#endif // HYPERPARAM_OPT_HPP
