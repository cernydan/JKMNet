#include "PSO.hpp"


/**
 * The constructor
 */
PSO::PSO(int swarmSize, int dim, double w, double c1, double c2,
         int maxIter, double xmin, double xmax)
    : swarmSize_(swarmSize), dim_(dim), w_(w), c1_(c1), c2_(c2),
      maxIter_(maxIter), xmin_(xmin), xmax_(xmax),
      globalBestVal_(std::numeric_limits<double>::infinity()) {
    
    rng_.seed(std::random_device{}());
    pos_.resize(swarmSize_);
    vel_.resize(swarmSize_);
    pBest_.resize(swarmSize_);
    pBestVal_.resize(swarmSize_);

    for (int i = 0; i < swarmSize_; ++i) {
        pos_[i] = Eigen::VectorXd::NullaryExpr(dim_, [&](){ return xmin_ + unif_(rng_) * (xmax_ - xmin_); });
        vel_[i] = Eigen::VectorXd::Zero(dim_);
        pBest_[i] = pos_[i];
        pBestVal_[i] = std::numeric_limits<double>::infinity();
    }
}

/**
 * PSO optimization method
 */
void PSO::optimize(const std::function<double(const Eigen::VectorXd&)>& objective) {
    for (int iter = 0; iter < maxIter_; ++iter) {
        for (int i = 0; i < swarmSize_; ++i) {
            double val = objective(pos_[i]);
            if (val < pBestVal_[i]) {
                pBest_[i] = pos_[i];
                pBestVal_[i] = val;
                if (val < globalBestVal_) {
                    globalBestVal_ = val;
                    globalBestPos_ = pos_[i];
                }
            }
        }

        for (int i = 0; i < swarmSize_; ++i) {
            for (int d = 0; d < dim_; ++d) {
                double r1 = unif_(rng_);
                double r2 = unif_(rng_);
                vel_[i][d] = w_ * vel_[i][d]
                           + c1_ * r1 * (pBest_[i][d] - pos_[i][d])
                           + c2_ * r2 * (globalBestPos_[d] - pos_[i][d]);
                pos_[i][d] += vel_[i][d];
                if (pos_[i][d] < xmin_) pos_[i][d] = xmin_;
                if (pos_[i][d] > xmax_) pos_[i][d] = xmax_;
            }
        }

        std::cout << "[PSO] Iter " << iter + 1
          << " | best RMSE = " << globalBestVal_
          << " | " << decodeBestParams() << "\n";
    }

    std::cout << "\n[PSO] Optimization finished. Best value = " << globalBestVal_ << "\n";
}

/**
 * Decode PSO vector into human-readable hyperparameters
 */
std::string PSO::decodeBestParams() const {
    std::ostringstream oss;
    oss << "Learning rate = " << (0.0001 + globalBestPos_[0] * (0.01 - 0.0001))
        << ", Hidden neurons = " << std::round(4 + globalBestPos_[1] * (50 - 4))
        << ", Activation = " << std::round(globalBestPos_[2] * 2);
    return oss.str();
}
