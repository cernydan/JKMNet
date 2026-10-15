/**
 * @file PSO.hpp
 * @brief Declares the PSO class implementing particle swarm optimization for hyperparameter search.
 */

#ifndef PSO_HPP
#define PSO_HPP

#include "eigen-3.4/Eigen/Dense"
#include <vector>
#include <functional>
#include <random>
#include <limits>
#include <iostream>

/// Particle swarm optimizer that minimizes a scalar objective over a bounded box [xmin, xmax]^dim.
class PSO {
public:
    PSO(int swarmSize, int dim, double w, double c1, double c2,
        int maxIter, double xmin, double xmax);  //!< The constructor

    void optimize(const std::function<double(const Eigen::VectorXd&)>& objective);  //!< PSO optimization method
    Eigen::VectorXd getBestPosition() const { return globalBestPos_; }  //!< Get the global best position in the search space
    double getBestValue() const { return globalBestVal_; }  //!< Get the global best value

    std::string decodeBestParams() const;  //!< Decode PSO vector into human-readable hyperparameters

private:
    int swarmSize_;      ///< Number of particles in the swarm.
    int dim_;            ///< Dimension of the search space (number of optimized hyperparameters).
    double w_;           ///< Inertia weight applied to the previous velocity.
    double c1_;          ///< Cognitive (personal best) acceleration coefficient.
    double c2_;          ///< Social (global best) acceleration coefficient.
    int maxIter_;        ///< Number of swarm update iterations.
    double xmin_;        ///< Lower bound of every search-space dimension.
    double xmax_;        ///< Upper bound of every search-space dimension.
    std::vector<Eigen::VectorXd> pos_;      ///< Current positions of the particles (swarmSize_ x dim_).
    std::vector<Eigen::VectorXd> vel_;      ///< Current velocities of the particles (swarmSize_ x dim_).
    std::vector<Eigen::VectorXd> pBest_;    ///< Personal best positions of the particles.
    std::vector<double> pBestVal_;          ///< Objective values at the personal best positions.
    Eigen::VectorXd globalBestPos_;         ///< Global best position found so far.
    double globalBestVal_;                  ///< Objective value at the global best position.

    std::mt19937 rng_;                                  ///< Random number generator for the swarm updates.
    std::uniform_real_distribution<double> unif_{0.0, 1.0};  ///< Uniform distribution on [0, 1) used for r1 and r2.
};

#endif // PSO_HPP
