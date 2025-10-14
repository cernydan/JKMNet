#ifndef PSO_HPP
#define PSO_HPP

#include "eigen-3.4/Eigen/Dense"
#include <vector>
#include <functional>
#include <random>
#include <limits>
#include <iostream>

class PSO {
public:
    PSO(int swarmSize, int dim, double w, double c1, double c2,
        int maxIter, double xmin, double xmax);  //!< The constructor
    
    void optimize(const std::function<double(const Eigen::VectorXd&)>& objective);  //!< PSO optimization method
    Eigen::VectorXd getBestPosition() const { return globalBestPos_; }  //!< Get the global best position in the search space
    double getBestValue() const { return globalBestVal_; }  //!< Get the global best value

    std::string decodeBestParams() const;  //!< Decode PSO vector into human-readable hyperparameters

private:
    int swarmSize_, dim_;
    double w_, c1_, c2_; 
    int maxIter_;
    double xmin_, xmax_;
    std::vector<Eigen::VectorXd> pos_, vel_, pBest_;
    std::vector<double> pBestVal_;
    Eigen::VectorXd globalBestPos_;
    double globalBestVal_;

    std::mt19937 rng_;
    std::uniform_real_distribution<double> unif_{0.0, 1.0};
};

#endif // PSO_HPP