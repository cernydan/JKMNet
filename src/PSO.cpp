/**
 * @file PSO.cpp
 * @brief Implements PSO — particle swarm optimization over a bounded hyperparameter box.
 */

#include "PSO.hpp"


/**
 * The constructor.
 * Initializes the swarm with uniformly distributed random positions inside
 * the bounds [xmin, xmax]^dim and zero velocities. The personal best of each
 * particle starts at its initial position with an infinite objective value,
 * and the global best value starts at infinity (minimization problem).
 * @param swarmSize Number of particles in the swarm.
 * @param dim Dimension of the search space (number of optimized hyperparameters).
 * @param w Inertia weight applied to the previous velocity.
 * @param c1 Cognitive (personal best) acceleration coefficient.
 * @param c2 Social (global best) acceleration coefficient.
 * @param maxIter Number of swarm update iterations performed by optimize().
 * @param xmin Lower bound of every search-space dimension.
 * @param xmax Upper bound of every search-space dimension.
 * @note The random number generator is seeded from std::random_device, so
 *       runs are not reproducible unless this is changed to a fixed seed.
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
 * PSO optimization method.
 * Runs maxIter_ iterations of the swarm. Each iteration first evaluates the
 * objective at every particle position and updates the personal and global
 * bests, then updates each particle's velocity using
 *   v = w * v + c1 * r1 * (pBest - pos) + c2 * r2 * (globalBest - pos)
 * with r1, r2 ~ U(0, 1), advances the position by the new velocity, and
 * clamps the position to the bounds [xmin_, xmax_]. The current global best
 * value and the decoded best hyperparameters are printed after each iteration.
 * @param objective Objective function to minimize over the search space.
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
 * Decode PSO vector into human-readable hyperparameters.
 * Maps the normalized particle coordinates from [0, 1]^3 to the concrete
 * hyperparameter ranges: learning rate in [0.0001, 0.01], number of hidden
 * neurons in [4, 50] (rounded), and activation function index as
 * round(pos[2] * 2) (0 = RELU, 1 = TANH, 2 = SIGMOID).
 * @return Human-readable string describing the global best hyperparameters.
 * @note The decoding must stay in sync with evaluateMLPwithParams(); if the
 *       global best was never updated, the result is empty/undefined.
 */
std::string PSO::decodeBestParams() const {
    std::ostringstream oss;
    oss << "Learning rate = " << (0.0001 + globalBestPos_[0] * (0.01 - 0.0001))
        << ", Hidden neurons = " << std::round(4 + globalBestPos_[1] * (50 - 4))
        << ", Activation = " << std::round(globalBestPos_[2] * 2);
    return oss.str();
}
