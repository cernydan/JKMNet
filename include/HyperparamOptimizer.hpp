#ifndef HYPERPARAM_OPTIM_HPP
#define HYPERPARAM_OPTIM_HPP

#include "ConfigIni.hpp"
#include "PSO.hpp"

RunConfig optimizeHyperparams(const RunConfig& cfg_in);  //!< Run PSO optimization and update config accordingly

#endif // HYPERPARAM_OPTIM_HPP