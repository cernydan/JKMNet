/**
 * @file ObjectiveFunctions.hpp
 * @brief Declares the objective_func_type enum and the ObjectiveFunctions class with loss functions for training.
 */
#ifndef OBJECTIVE_FUNCTIONS_HPP
#define OBJECTIVE_FUNCTIONS_HPP

#include "eigen-3.4/Eigen/Dense"
#include <array>
#include <string>
#include <vector>

/// Types of objective (loss) functions available for training.
enum class objective_func_type {
    MSE,                    //!< Standard Mean Squared Error (default)
    DIFF_SLOPE,             //!< L1: Pure First-Difference (Slope) Loss
    LEVEL_SLOPE,            //!< L2: Combined Level + Slope Loss (recommended baseline)
    DIFF_CURVATURE,         //!< L3: Second-Difference (Curvature) Loss
    LEVEL_SLOPE_CURVATURE,  //!< L3': Combined Level + Slope + Curvature Loss
    NORMALIZED_SLOPE,       //!< L4: Normalized/Scaled Slope Loss
    PERSISTENCE_INDEX,      //!< L5: Persistence-Index-Based Loss
    DILATE,                 //!< DILATE: Shape and Temporal Alignment Loss (simplified surrogate gradient)
    DILATE2                 //!< DILATE2: Faithful DILATE (Le Guen & Thome 2019) with exact soft-DTW + TDI gradients
};

/// Provides various loss functions and their gradients for network training.
/**
 * These loss functions are designed to explicitly penalize temporal misalignment
 * rather than only pointwise magnitude error, in order to correct lag while
 * preserving predictive accuracy. All methods are static; the class is a
 * stateless utility.
 */
class ObjectiveFunctions {
public:
    /// Parse an objective function type from a string (case-insensitive).
    /**
     * Accepts several aliases per type, e.g. "MSE"/"DEFAULT", "DIFF_SLOPE"/"SLOPE"/"L1".
     * @param s Name of the objective function (case-insensitive).
     * @return The matching objective_func_type value.
     * @throw std::runtime_error If the name does not match any known type.
     */
    static objective_func_type fromString(const std::string& s);

    /// Convert an objective function type to its canonical string name.
    /**
     * @param func The objective function type.
     * @return Canonical name (e.g. "MSE", "DIFF_SLOPE"), or "UNKNOWN" for invalid values.
     */
    static std::string toString(objective_func_type func);

    /// Compute the selected loss between observed and simulated values.
    /**
     * Dispatches to the individual loss function selected by lossType.
     * @param obs Observed values (one row of observations).
     * @param sim Simulated/predicted values (one row of predictions, same size as obs).
     * @param lossType Type of loss function to apply.
     * @param alpha Weighting parameter for combined losses (default 0.5).
     * @param alpha2 Second weighting parameter for the curvature term (default 0.33).
     * @param epsilon Small constant to prevent division by zero (default 1e-8).
     * @return Scalar loss value.
     * @note For DILATE2, alpha is the shape/time trade-off and alpha2 is used
     *       as the soft-DTW smoothing parameter (gamma).
     */
    static double computeLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        objective_func_type lossType,
        double alpha = 0.5,
        double alpha2 = 0.33,
        double epsilon = 1e-8
    );

    /// Compute the gradient of the selected loss for backpropagation.
    /**
     * @param obs Observed values.
     * @param sim Simulated/predicted values (same size as obs).
     * @param lossType Type of loss function.
     * @param alpha Weighting parameter for combined losses (default 0.5).
     * @param alpha2 Second weighting parameter for the curvature term (default 0.33).
     * @param epsilon Small constant to prevent division by zero (default 1e-8).
     * @return Gradient vector of the loss with respect to sim (same size as sim).
     * @note The DILATE gradient is only approximated (falls back to a level+slope combination);
     *       DILATE2 uses the exact gradient of its loss (alpha2 as soft-DTW gamma).
     */
    static Eigen::VectorXd computeGradient(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        objective_func_type lossType,
        double alpha = 0.5,
        double alpha2 = 0.33,
        double epsilon = 1e-8
    );

    /// Compute the loss for a matrix of samples (average over all rows).
    /**
     * @param Y_obs Matrix of observed values (one sample per row).
     * @param Y_sim Matrix of simulated values, same dimensions as Y_obs.
     * @param lossType Type of loss function to apply row-wise.
     * @param alpha Weighting parameter for combined losses (default 0.5).
     * @param alpha2 Second weighting parameter for the curvature term (default 0.33).
     * @param epsilon Small constant to prevent division by zero (default 1e-8).
     * @return Mean loss over all rows.
     * @throw std::invalid_argument If the matrix dimensions do not match.
     * @note For DILATE2, alpha2 is used as the soft-DTW smoothing parameter (gamma).
     */
    static double computeBatchLoss(
        const Eigen::MatrixXd& Y_obs,
        const Eigen::MatrixXd& Y_sim,
        objective_func_type lossType,
        double alpha = 0.5,
        double alpha2 = 0.33,
        double epsilon = 1e-8
    );

    // Individual loss functions (for testing/documentation)

    /// L1: Pure First-Difference (Slope) Loss.
    /**
     * Penalizes mismatch in rate of change: mean((dOBS - dSIM)^2), where d denotes
     * the first difference.
     * @param obs Observed values.
     * @param sim Simulated values (same size as obs).
     * @return Mean squared error of the first differences; 0.0 for inputs shorter than 2.
     * @throw std::invalid_argument If the vector sizes do not match.
     */
    static double diffSlopeLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim
    );

    /// L2: Combined Level + Slope Loss.
    /**
     * L2 = alpha * MSE(OBS,SIM) + (1-alpha) * mean((dOBS - dSIM)^2).
     * @param obs Observed values.
     * @param sim Simulated values (same size as obs).
     * @param alpha Weight of the level (MSE) term; (1-alpha) weights the slope term.
     * @return Combined scalar loss.
     * @throw std::invalid_argument If the vector sizes do not match.
     */
    static double levelSlopeLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double alpha = 0.5
    );

    /// L3: Second-Difference (Curvature) Loss.
    /**
     * Penalizes mismatch in acceleration at inflection points using second differences.
     * @param obs Observed values.
     * @param sim Simulated values (same size as obs).
     * @return Mean squared error of the second differences; 0.0 for inputs shorter than 3.
     * @throw std::invalid_argument If the vector sizes do not match.
     */
    static double diffCurvatureLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim
    );

    /// L3': Combined Level + Slope + Curvature Loss.
    /**
     * L3' = alpha1*MSE + alpha2*slopeLoss + alpha3*curvatureLoss,
     * where alpha3 = 1 - alpha1 - alpha2.
     * @param obs Observed values.
     * @param sim Simulated values (same size as obs).
     * @param alpha1 Weight of the level (MSE) term (default 0.34).
     * @param alpha2 Weight of the slope term (default 0.33).
     * @return Combined scalar loss.
     * @throw std::invalid_argument If the vector sizes do not match.
     */
    static double levelSlopeCurvatureLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double alpha1 = 0.34,
        double alpha2 = 0.33
    );

    /// L4: Normalized/Scaled Slope Loss.
    /**
     * Normalizes slope differences by the standard deviation of the observed differences.
     * @param obs Observed values.
     * @param sim Simulated values (same size as obs).
     * @param epsilon Small constant added to the standard deviation to prevent division by zero.
     * @return Mean squared normalized slope difference.
     * @throw std::invalid_argument If the vector sizes do not match.
     */
    static double normalizedSlopeLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double epsilon = 1e-8
    );

    /// L5: Persistence-Index-Based Loss.
    /**
     * Minimizing this loss is equivalent to maximizing the persistence index (PI):
     * L5 = MSE(OBS,SIM) / (MSE(OBS, persistence(OBS)) + epsilon) = 1 - PI.
     * @param obs Observed values.
     * @param sim Simulated values (same size as obs).
     * @param epsilon Small constant added to the denominator to prevent division by zero.
     * @return Model MSE relative to the persistence-forecast MSE.
     * @throw std::invalid_argument If the vector sizes do not match.
     */
    static double persistenceIndexLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double epsilon = 1e-8
    );

    /// DILATE Loss (simplified version).
    /**
     * Combines a shape term (soft-DTW) and a temporal alignment term:
     * L = delta * softDTW + (1-delta) * temporalPenalty.
     * @param obs Observed values.
     * @param sim Simulated values (same size as obs).
     * @param delta Weight of the shape term; (1-delta) weights the temporal term.
     * @param gamma Smoothing parameter of soft-DTW.
     * @return Combined scalar loss.
     * @throw std::invalid_argument If the vector sizes do not match.
     */
    static double dilateLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double delta = 0.5,
        double gamma = 0.1
    );

    /// DILATE2 Loss — faithful DILATE (Le Guen & Thome, NeurIPS 2019).
    /**
     * L = delta * softDTW_gamma(OBS, SIM) + (1-delta) * TDI, where
     * softDTW is the soft dynamic time warping discrepancy over the squared
     * Euclidean cost matrix D[i,j] = (obs_i - sim_j)^2 (shape term), and
     * TDI is the temporal distortion index: the soft-DTW alignment matrix E
     * weighted by the squared index distances Omega[i,j] = (i-j)^2, normalized
     * by N^2 (temporal term). This is the loss actually minimized by
     * dilate2Gradient.
     * @param obs Observed values (one row of observations).
     * @param sim Simulated/predicted values (same size as obs).
     * @param delta Weight of the shape (soft-DTW) term; (1-delta) weights the temporal term.
     * @param gamma Smoothing parameter of the soft-minimum (gamma -> 0 recovers hard DTW).
     * @return Combined scalar loss.
     * @throw std::invalid_argument If the vector sizes do not match or gamma <= 0.
     */
    static double dilate2Loss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double delta = 0.5,
        double gamma = 0.1
    );

    /// Gradient of the DILATE2 loss w.r.t. sim (exact, via soft-DTW Hessian-vector product).
    /**
     * The shape gradient uses the soft-DTW alignment matrix E = d softDTW / dD.
     * The temporal gradient differentiates TDI through E using the soft-DTW
     * Hessian-vector product (forward/backward differential DP, following the
     * reference implementation of the DILATE paper).
     * @param obs Observed values.
     * @param sim Simulated/predicted values (same size as obs).
     * @param delta Weight of the shape term; (1-delta) weights the temporal term.
     * @param gamma Smoothing parameter of the soft-minimum.
     * @return Gradient vector (same size as sim).
     * @throw std::invalid_argument If the vector sizes do not match or gamma <= 0.
     */
    static Eigen::VectorXd dilate2Gradient(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double delta = 0.5,
        double gamma = 0.1
    );

private:
    /// Compute the first difference (dx_i = x_i - x_{i-1}).
    /**
     * @param x Input vector of size n.
     * @return Vector of size n-1 with the consecutive differences; empty for n < 2.
     */
    static Eigen::VectorXd firstDifference(const Eigen::VectorXd& x);

    /// Compute the second difference (d2x_i = x_i - 2*x_{i-1} + x_{i-2}).
    /**
     * @param x Input vector of size n.
     * @return Vector of size n-2 with the second differences; empty for n < 3.
     */
    static Eigen::VectorXd secondDifference(const Eigen::VectorXd& x);

    /// Compute the persistence forecast of a series.
    /**
     * The persistence forecast repeats the previous value: y_persist[0] = y[0],
     * y_persist[i] = y[i-1].
     * @param y Input series.
     * @return Persistence forecast, same size as y.
     */
    static Eigen::VectorXd persistenceForecast(const Eigen::VectorXd& y);

    /// Soft-DTW computation for the DILATE shape term.
    /**
     * Simplified soft dynamic time warping over the squared-distance cost matrix.
     * @param y First series.
     * @param yhat Second series.
     * @param gamma Smoothing parameter of the soft-minimum.
     * @return Soft-DTW discrepancy value; 0.0 if either series is empty.
     */
    static double softDTW(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& yhat,
        double gamma = 0.1
    );

    /// Temporal penalty term for DILATE.
    /**
     * Quadratic penalty for alignment deviation from the diagonal.
     * @note Currently a placeholder: the deviation is always zero, so it returns 0.0.
     * @todo check semantics — implement the alignment-path-based penalty from full DILATE.
     */
    static double temporalPenalty(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& yhat
    );

    /// Shared soft-DTW dynamic programming for DILATE2 (forward DP + alignment backtrace).
    /**
     * Computes the cost matrix D, the soft-DTW forward DP V (with the three
     * soft-min weights per cell collected in Q), and the alignment matrix E
     * (= d softDTW / dD, via the backward DP seeded at the last cell).
     * Padded indexing follows the reference DILATE implementation: V is
     * (N+1)x(N+1) with V(0,0)=0 and borders 1e10; Q[k] and E are (N+2)x(N+2)
     * with the seed Q[k](N+1,N+1)=1, E(N+1,N+1)=1.
     * @param obs Observed values.
     * @param sim Simulated values (same size as obs).
     * @param gamma Soft-min smoothing parameter (must be > 0).
     * @param[out] D NxN cost matrix D(i,j) = (obs(i)-sim(j))^2.
     * @param[out] V (N+1)x(N+1) soft-DTW forward DP; V(N,N) is the soft-DTW value.
     * @param[out] Q Three (N+2)x(N+2) soft-min weight matrices per DP cell.
     * @param[out] E (N+2)x(N+2) alignment matrix; E(i+1,j+1) = d softDTW / dD(i,j).
     */
    static void dilate2ComputeDP(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double gamma,
        Eigen::MatrixXd& D,
        Eigen::MatrixXd& V,
        std::array<Eigen::MatrixXd, 3>& Q,
        Eigen::MatrixXd& E
    );

    /// Soft-DTW Hessian-vector product for the DILATE2 temporal gradient.
    /**
     * Given the upstream gradient Z = d loss / dE (NxN, cropped), computes
     * E_dot = d (sum(Z * E)) / dD (NxN, cropped) via the differential forward
     * DP (V_dot), the soft-min Hessian product (Q_dot) and the differential
     * backward DP (E_dot). Requires Q and E from a prior dilate2ComputeDP call.
     * @param Z NxN upstream gradient w.r.t. the alignment matrix E (cropped).
     * @param Q Soft-min weights from dilate2ComputeDP.
     * @param E Alignment matrix from dilate2ComputeDP.
     * @param gamma Soft-min smoothing parameter (must be > 0).
     * @return NxN matrix E_dot = d (Z . E) / dD.
     */
    static Eigen::MatrixXd dilate2HessianProduct(
        const Eigen::MatrixXd& Z,
        const std::array<Eigen::MatrixXd, 3>& Q,
        const Eigen::MatrixXd& E,
        double gamma
    );
};

#endif // OBJECTIVE_FUNCTIONS_HPP
