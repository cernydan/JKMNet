#ifndef HYPERPARAM_OBJ_HPP
#define HYPERPARAM_OBJ_HPP

#include "eigen-3.4/Eigen/Dense"
#include "JKMNet.hpp"

double scale(double x, double xmin, double xmax);
double evaluateMLPwithParams(const Eigen::VectorXd &params, const RunConfig& cfg);  //!< Evaluate a single MLP configuration defined by PSO params

#endif // HYPERPARAM_OBJ_HPP