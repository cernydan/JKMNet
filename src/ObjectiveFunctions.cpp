#include "ObjectiveFunctions.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iostream>

objective_func_type ObjectiveFunctions::fromString(const std::string& s) {
    std::string upper;
    upper.reserve(s.size());
    for (char c : s) {
        upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }

    if (upper == "MSE" || upper == "DEFAULT") return objective_func_type::MSE;
    if (upper == "DIFF_SLOPE" || upper == "SLOPE" || upper == "L1") return objective_func_type::DIFF_SLOPE;
    if (upper == "LEVEL_SLOPE" || upper == "COMBINED" || upper == "L2") return objective_func_type::LEVEL_SLOPE;
    if (upper == "DIFF_CURVATURE" || upper == "CURVATURE" || upper == "L3") return objective_func_type::DIFF_CURVATURE;
    if (upper == "LEVEL_SLOPE_CURVATURE" || upper == "COMBINED_ALL" || upper == "L3PRIME") return objective_func_type::LEVEL_SLOPE_CURVATURE;
    if (upper == "NORMALIZED_SLOPE" || upper == "SCALED_SLOPE" || upper == "L4") return objective_func_type::NORMALIZED_SLOPE;
    if (upper == "PERSISTENCE_INDEX" || upper == "PI_LOSS" || upper == "L5") return objective_func_type::PERSISTENCE_INDEX;
    if (upper == "DILATE") return objective_func_type::DILATE;

    throw std::runtime_error("Unknown objective function: " + s);
}

std::string ObjectiveFunctions::toString(objective_func_type func) {
    switch (func) {
        case objective_func_type::MSE: return "MSE";
        case objective_func_type::DIFF_SLOPE: return "DIFF_SLOPE";
        case objective_func_type::LEVEL_SLOPE: return "LEVEL_SLOPE";
        case objective_func_type::DIFF_CURVATURE: return "DIFF_CURVATURE";
        case objective_func_type::LEVEL_SLOPE_CURVATURE: return "LEVEL_SLOPE_CURVATURE";
        case objective_func_type::NORMALIZED_SLOPE: return "NORMALIZED_SLOPE";
        case objective_func_type::PERSISTENCE_INDEX: return "PERSISTENCE_INDEX";
        case objective_func_type::DILATE: return "DILATE";
        default: return "UNKNOWN";
    }
}

Eigen::VectorXd ObjectiveFunctions::firstDifference(const Eigen::VectorXd& x) {
    int n = x.size();
    if (n < 2) {
        return Eigen::VectorXd(0);
    }
    Eigen::VectorXd diff(n - 1);
    for (int i = 1; i < n; ++i) {
        diff(i - 1) = x(i) - x(i - 1);
    }
    return diff;
}

Eigen::VectorXd ObjectiveFunctions::secondDifference(const Eigen::VectorXd& x) {
    int n = x.size();
    if (n < 3) {
        return Eigen::VectorXd(0);
    }
    Eigen::VectorXd diff(n - 2);
    for (int i = 2; i < n; ++i) {
        diff(i - 2) = x(i) - 2.0 * x(i - 1) + x(i - 2);
    }
    return diff;
}

Eigen::VectorXd ObjectiveFunctions::persistenceForecast(const Eigen::VectorXd& y) {
    int n = y.size();
    Eigen::VectorXd persist(n);
    persist(0) = y(0);
    for (int i = 1; i < n; ++i) {
        persist(i) = y(i - 1);
    }
    return persist;
}

double ObjectiveFunctions::diffSlopeLoss(const Eigen::VectorXd& obs, const Eigen::VectorXd& sim) {
    if (obs.size() != sim.size()) {
        throw std::invalid_argument("ObjectiveFunctions::diffSlopeLoss: vector sizes must match");
    }

    Eigen::VectorXd dObs = firstDifference(obs);
    Eigen::VectorXd dSim = firstDifference(sim);

    if (dObs.size() == 0) {
        return 0.0;
    }

    double sumSq = 0.0;
    for (int i = 0; i < dObs.size(); ++i) {
        double diff = dObs(i) - dSim(i);
        sumSq += diff * diff;
    }

    return sumSq / static_cast<double>(dObs.size());
}

double ObjectiveFunctions::levelSlopeLoss(
    const Eigen::VectorXd& obs,
    const Eigen::VectorXd& sim,
    double alpha
) {
    if (obs.size() != sim.size()) {
        throw std::invalid_argument("ObjectiveFunctions::levelSlopeLoss: vector sizes must match");
    }

    // MSE term (level)
    double mse = 0.0;
    for (int i = 0; i < obs.size(); ++i) {
        double diff = obs(i) - sim(i);
        mse += diff * diff;
    }
    mse /= static_cast<double>(obs.size());

    // Slope term
    double slopeLoss = diffSlopeLoss(obs, sim);

    return alpha * mse + (1.0 - alpha) * slopeLoss;
}

double ObjectiveFunctions::diffCurvatureLoss(const Eigen::VectorXd& obs, const Eigen::VectorXd& sim) {
    if (obs.size() != sim.size()) {
        throw std::invalid_argument("ObjectiveFunctions::diffCurvatureLoss: vector sizes must match");
    }

    Eigen::VectorXd ddObs = secondDifference(obs);
    Eigen::VectorXd ddSim = secondDifference(sim);

    if (ddObs.size() == 0) {
        return 0.0;
    }

    double sumSq = 0.0;
    for (int i = 0; i < ddObs.size(); ++i) {
        double diff = ddObs(i) - ddSim(i);
        sumSq += diff * diff;
    }

    return sumSq / static_cast<double>(ddObs.size());
}

double ObjectiveFunctions::levelSlopeCurvatureLoss(
    const Eigen::VectorXd& obs,
    const Eigen::VectorXd& sim,
    double alpha1,
    double alpha2
) {
    if (obs.size() != sim.size()) {
        throw std::invalid_argument("ObjectiveFunctions::levelSlopeCurvatureLoss: vector sizes must match");
    }

    double alpha3 = 1.0 - alpha1 - alpha2;

    // MSE term (level)
    double mse = 0.0;
    for (int i = 0; i < obs.size(); ++i) {
        double diff = obs(i) - sim(i);
        mse += diff * diff;
    }
    mse /= static_cast<double>(obs.size());

    // Slope term
    double slopeLoss = diffSlopeLoss(obs, sim);

    // Curvature term
    double curvatureLoss = diffCurvatureLoss(obs, sim);

    return alpha1 * mse + alpha2 * slopeLoss + alpha3 * curvatureLoss;
}

double ObjectiveFunctions::normalizedSlopeLoss(
    const Eigen::VectorXd& obs,
    const Eigen::VectorXd& sim,
    double epsilon
) {
    if (obs.size() != sim.size()) {
        throw std::invalid_argument("ObjectiveFunctions::normalizedSlopeLoss: vector sizes must match");
    }

    Eigen::VectorXd dObs = firstDifference(obs);
    Eigen::VectorXd dSim = firstDifference(sim);

    if (dObs.size() == 0) {
        return 0.0;
    }

    // Compute standard deviation of observed differences
    double meanDObs = dObs.mean();
    double varDObs = 0.0;
    for (int i = 0; i < dObs.size(); ++i) {
        double diff = dObs(i) - meanDObs;
        varDObs += diff * diff;
    }
    varDObs /= static_cast<double>(dObs.size());
    double stdDObs = std::sqrt(varDObs);

    // Normalized slope loss
    double sumSq = 0.0;
    for (int i = 0; i < dObs.size(); ++i) {
        double normalizedDiff = (dObs(i) - dSim(i)) / (stdDObs + epsilon);
        sumSq += normalizedDiff * normalizedDiff;
    }

    return sumSq / static_cast<double>(dObs.size());
}

double ObjectiveFunctions::persistenceIndexLoss(
    const Eigen::VectorXd& obs,
    const Eigen::VectorXd& sim,
    double epsilon
) {
    if (obs.size() != sim.size()) {
        throw std::invalid_argument("ObjectiveFunctions::persistenceIndexLoss: vector sizes must match");
    }

    // MSE between obs and sim
    double mseModel = 0.0;
    for (int i = 0; i < obs.size(); ++i) {
        double diff = obs(i) - sim(i);
        mseModel += diff * diff;
    }
    mseModel /= static_cast<double>(obs.size());

    // MSE between obs and persistence forecast of obs
    Eigen::VectorXd persist = persistenceForecast(obs);
    double msePersist = 0.0;
    for (int i = 0; i < obs.size(); ++i) {
        double diff = obs(i) - persist(i);
        msePersist += diff * diff;
    }
    msePersist /= static_cast<double>(obs.size());

    // L5 = MSE_model / (MSE_persist + epsilon) = 1 - PI
    return mseModel / (msePersist + epsilon);
}

double ObjectiveFunctions::softDTW(
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& yhat,
    double gamma
) {
    // Simplified soft-DTW implementation
    // For a full implementation, see Le Guen & Thome (2019)
    int n = y.size();
    int m = yhat.size();

    if (n == 0 || m == 0) {
        return 0.0;
    }

    // Cost matrix: D[i,j] = (y[i] - yhat[j])^2
    Eigen::MatrixXd D(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double diff = y(i) - yhat(j);
            D(i, j) = diff * diff;
        }
    }

    // Soft-DTW dynamic programming
    Eigen::MatrixXd R(n + 1, m + 1);
    R.setConstant(std::numeric_limits<double>::infinity());
    R(0, 0) = 0.0;

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            double minPrev = std::min({R(i - 1, j), R(i, j - 1), R(i - 1, j - 1)});
            R(i, j) = D(i - 1, j - 1) + gamma * std::exp((minPrev - D(i - 1, j - 1)) / gamma);
        }
    }

    return R(n, m);
}

double ObjectiveFunctions::temporalPenalty(
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& yhat
) {
    // Simple temporal penalty based on alignment
    // In full DILATE, this would use the optimal alignment path
    int n = std::min(y.size(), yhat.size());
    if (n == 0) return 0.0;

    double penalty = 0.0;
    for (int i = 0; i < n; ++i) {
        // Quadratic penalty for temporal deviation from diagonal
        double tDev = static_cast<double>(i) - static_cast<double>(i);
        penalty += tDev * tDev;
    }

    return penalty / static_cast<double>(n);
}

double ObjectiveFunctions::dilateLoss(
    const Eigen::VectorXd& obs,
    const Eigen::VectorXd& sim,
    double delta,
    double gamma
) {
    if (obs.size() != sim.size()) {
        throw std::invalid_argument("ObjectiveFunctions::dilateLoss: vector sizes must match");
    }

    double shapeTerm = softDTW(obs, sim, gamma);
    double temporalTerm = temporalPenalty(obs, sim);

    return delta * shapeTerm + (1.0 - delta) * temporalTerm;
}

double ObjectiveFunctions::computeLoss(
    const Eigen::VectorXd& obs,
    const Eigen::VectorXd& sim,
    objective_func_type lossType,
    double alpha,
    double alpha2,
    double epsilon
) {
    switch (lossType) {
        case objective_func_type::MSE: {
            double sumSq = 0.0;
            for (int i = 0; i < obs.size(); ++i) {
                double diff = obs(i) - sim(i);
                sumSq += diff * diff;
            }
            return sumSq / static_cast<double>(obs.size());
        }
        case objective_func_type::DIFF_SLOPE:
            return diffSlopeLoss(obs, sim);
        case objective_func_type::LEVEL_SLOPE:
            return levelSlopeLoss(obs, sim, alpha);
        case objective_func_type::DIFF_CURVATURE:
            return diffCurvatureLoss(obs, sim);
        case objective_func_type::LEVEL_SLOPE_CURVATURE:
            return levelSlopeCurvatureLoss(obs, sim, alpha, alpha2);
        case objective_func_type::NORMALIZED_SLOPE:
            return normalizedSlopeLoss(obs, sim, epsilon);
        case objective_func_type::PERSISTENCE_INDEX:
            return persistenceIndexLoss(obs, sim, epsilon);
        case objective_func_type::DILATE:
            return dilateLoss(obs, sim, alpha, epsilon);
        default:
            return levelSlopeLoss(obs, sim, alpha);
    }
}

Eigen::VectorXd ObjectiveFunctions::computeGradient(
    const Eigen::VectorXd& obs,
    const Eigen::VectorXd& sim,
    objective_func_type lossType,
    double alpha,
    double alpha2,
    double epsilon
) {
    int n = obs.size();
    Eigen::VectorXd grad(n);

    switch (lossType) {
        case objective_func_type::MSE: {
            // d/d(sim_i) MSE = 2 * (sim_i - obs_i) / n
            for (int i = 0; i < n; ++i) {
                grad(i) = 2.0 * (sim(i) - obs(i)) / static_cast<double>(n);
            }
            break;
        }
        case objective_func_type::DIFF_SLOPE: {
            // Gradient of slope loss
            // For interior points: contributes to both Δ_i and Δ_{i+1}
            for (int i = 0; i < n; ++i) {
                double g = 0.0;
                if (i > 0) {
                    g += 2.0 * ((sim(i) - sim(i - 1)) - (obs(i) - obs(i - 1))) / static_cast<double>(n - 1);
                }
                if (i < n - 1) {
                    g -= 2.0 * ((sim(i + 1) - sim(i)) - (obs(i + 1) - obs(i))) / static_cast<double>(n - 1);
                }
                grad(i) = g;
            }
            break;
        }
        case objective_func_type::LEVEL_SLOPE: {
            // Combined gradient: α * MSE_grad + (1-α) * slope_grad
            Eigen::VectorXd mseGrad = computeGradient(obs, sim, objective_func_type::MSE, alpha, alpha2, epsilon);
            Eigen::VectorXd slopeGrad = computeGradient(obs, sim, objective_func_type::DIFF_SLOPE, alpha, alpha2, epsilon);
            for (int i = 0; i < n; ++i) {
                grad(i) = alpha * mseGrad(i) + (1.0 - alpha) * slopeGrad(i);
            }
            break;
        }
        case objective_func_type::DIFF_CURVATURE: {
            // Gradient of curvature loss
            for (int i = 0; i < n; ++i) {
                double g = 0.0;
                if (i >= 2) {
                    g += 2.0 * ((sim(i) - 2*sim(i-1) + sim(i-2)) - (obs(i) - 2*obs(i-1) + obs(i-2))) / static_cast<double>(n - 2);
                }
                if (i >= 1 && i < n - 1) {
                    g -= 4.0 * ((sim(i+1) - 2*sim(i) + sim(i-1)) - (obs(i+1) - 2*obs(i) + obs(i-1))) / static_cast<double>(n - 2);
                }
                if (i < n - 2) {
                    g += 2.0 * ((sim(i+2) - 2*sim(i+1) + sim(i)) - (obs(i+2) - 2*obs(i+1) + obs(i))) / static_cast<double>(n - 2);
                }
                grad(i) = g;
            }
            break;
        }
        case objective_func_type::LEVEL_SLOPE_CURVATURE: {
            // Combined gradient for all three terms
            Eigen::VectorXd mseGrad = computeGradient(obs, sim, objective_func_type::MSE, alpha, alpha2, epsilon);
            Eigen::VectorXd slopeGrad = computeGradient(obs, sim, objective_func_type::DIFF_SLOPE, alpha, alpha2, epsilon);
            Eigen::VectorXd curvGrad = computeGradient(obs, sim, objective_func_type::DIFF_CURVATURE, alpha, alpha2, epsilon);
            double alpha3 = 1.0 - alpha - alpha2;
            for (int i = 0; i < n; ++i) {
                grad(i) = alpha * mseGrad(i) + alpha2 * slopeGrad(i) + alpha3 * curvGrad(i);
            }
            break;
        }
        case objective_func_type::NORMALIZED_SLOPE: {
            // Compute normalization factor
            Eigen::VectorXd dObs = firstDifference(obs);
            double meanDObs = dObs.size() > 0 ? dObs.mean() : 0.0;
            double varDObs = 0.0;
            for (int i = 0; i < dObs.size(); ++i) {
                double diff = dObs(i) - meanDObs;
                varDObs += diff * diff;
            }
            varDObs /= static_cast<double>(dObs.size());
            double stdDObs = std::sqrt(varDObs) + epsilon;

            // Normalized gradient
            for (int i = 0; i < n; ++i) {
                double g = 0.0;
                if (i > 0) {
                    double normDiff = ((sim(i) - sim(i - 1)) - (obs(i) - obs(i - 1))) / stdDObs;
                    g += 2.0 * normDiff / static_cast<double>(n - 1) / stdDObs;
                }
                if (i < n - 1) {
                    double normDiff = ((sim(i + 1) - sim(i)) - (obs(i + 1) - obs(i))) / stdDObs;
                    g -= 2.0 * normDiff / static_cast<double>(n - 1) / stdDObs;
                }
                grad(i) = g;
            }
            break;
        }
        case objective_func_type::PERSISTENCE_INDEX: {
            // Gradient of PI-based loss
            Eigen::VectorXd persist = persistenceForecast(obs);

            // MSE_model gradient
            double mseModel = 0.0;
            for (int i = 0; i < n; ++i) {
                double diff = obs(i) - sim(i);
                mseModel += diff * diff;
            }
            mseModel /= static_cast<double>(n);

            // MSE_persist (constant w.r.t. sim)
            double msePersist = 0.0;
            for (int i = 0; i < n; ++i) {
                double diff = obs(i) - persist(i);
                msePersist += diff * diff;
            }
            msePersist /= static_cast<double>(n);

            // d/d(sim_i) [mseModel / (msePersist + eps)] = 2*(sim_i - obs_i) / (n * (msePersist + eps))
            double denom = msePersist + epsilon;
            for (int i = 0; i < n; ++i) {
                grad(i) = 2.0 * (sim(i) - obs(i)) / (static_cast<double>(n) * denom);
            }
            break;
        }
        case objective_func_type::DILATE: {
            // Approximate gradient for DILATE (using soft-DTW approximation)
            // This is a simplified version - full implementation requires more complex DTW backprop
            // For now, fall back to level-slope as a practical approximation
            Eigen::VectorXd mseGrad = computeGradient(obs, sim, objective_func_type::MSE, alpha, alpha2, epsilon);
            Eigen::VectorXd slopeGrad = computeGradient(obs, sim, objective_func_type::DIFF_SLOPE, alpha, alpha2, epsilon);
            for (int i = 0; i < n; ++i) {
                grad(i) = alpha * mseGrad(i) + (1.0 - alpha) * slopeGrad(i);
            }
            break;
        }
        default: {
            // Default to MSE gradient
            for (int i = 0; i < n; ++i) {
                grad(i) = 2.0 * (sim(i) - obs(i)) / static_cast<double>(n);
            }
        }
    }

    return grad;
}

double ObjectiveFunctions::computeBatchLoss(
    const Eigen::MatrixXd& Y_obs,
    const Eigen::MatrixXd& Y_sim,
    objective_func_type lossType,
    double alpha,
    double alpha2,
    double epsilon
) {
    if (Y_obs.rows() != Y_sim.rows() || Y_obs.cols() != Y_sim.cols()) {
        throw std::invalid_argument("ObjectiveFunctions::computeBatchLoss: matrix dimensions must match");
    }

    double totalLoss = 0.0;
    int nRows = Y_obs.rows();

    for (int i = 0; i < nRows; ++i) {
        Eigen::VectorXd obsRow = Y_obs.row(i);
        Eigen::VectorXd simRow = Y_sim.row(i);
        totalLoss += computeLoss(obsRow, simRow, lossType, alpha, alpha2, epsilon);
    }

    return totalLoss / static_cast<double>(nRows);
}
