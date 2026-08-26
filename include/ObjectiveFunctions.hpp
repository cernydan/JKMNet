#ifndef OBJECTIVE_FUNCTIONS_HPP
#define OBJECTIVE_FUNCTIONS_HPP

#include "eigen-3.4/Eigen/Dense"
#include <string>
#include <vector>

/**
 * Enum for objective function types
 */
enum class objective_func_type {
    MSE,                    //!< Standard Mean Squared Error (default)
    DIFF_SLOPE,             //!< L1: Pure First-Difference (Slope) Loss
    LEVEL_SLOPE,            //!< L2: Combined Level + Slope Loss (recommended baseline)
    DIFF_CURVATURE,         //!< L3: Second-Difference (Curvature) Loss
    LEVEL_SLOPE_CURVATURE,  //!< L3': Combined Level + Slope + Curvature Loss
    NORMALIZED_SLOPE,       //!< L4: Normalized/Scaled Slope Loss
    PERSISTENCE_INDEX,      //!< L5: Persistence-Index-Based Loss
    DILATE                  //!< DILATE: Shape and Temporal Alignment Loss
};

/**
 * ObjectiveFunctions class - provides various loss functions for training
 *
 * These loss functions are designed to explicitly penalize temporal misalignment
 * rather than only pointwise magnitude error, in order to correct lag while
 * preserving predictive accuracy.
 */
class ObjectiveFunctions {
public:
    /**
     * Get objective function type from string
     */
    static objective_func_type fromString(const std::string& s);

    /**
     * Convert objective function type to string
     */
    static std::string toString(objective_func_type func);

    /**
     * Compute loss between observed and simulated values
     *
     * @param obs Observed values (row of observations)
     * @param sim Simulated/predicted values (row of predictions)
     * @param alpha Weighting parameter for combined losses (default 0.5)
     * @param alpha2 Second weighting parameter for curvature term (default 0.33)
     * @param epsilon Small constant to prevent division by zero (default 1e-8)
     * @return Loss value
     */
    static double computeLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        objective_func_type lossType,
        double alpha = 0.5,
        double alpha2 = 0.33,
        double epsilon = 1e-8
    );

    /**
     * Compute gradient for backpropagation
     *
     * @param obs Observed values
     * @param sim Simulated/predicted values
     * @param lossType Type of loss function
     * @param alpha Weighting parameter for combined losses (default 0.5)
     * @param alpha2 Second weighting parameter for curvature term (default 0.33)
     * @param epsilon Small constant to prevent division by zero (default 1e-8)
     * @return Gradient vector (derivative w.r.t. sim)
     */
    static Eigen::VectorXd computeGradient(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        objective_func_type lossType,
        double alpha = 0.5,
        double alpha2 = 0.33,
        double epsilon = 1e-8
    );

    /**
     * Compute loss for a matrix of samples (average over all rows)
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

    /**
     * L1: Pure First-Difference (Slope) Loss
     * Penalizes mismatch in rate of change: sum((ΔOBS - ΔSIM)^2)
     */
    static double diffSlopeLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim
    );

    /**
     * L2: Combined Level + Slope Loss
     * L2 = α * MSE(OBS,SIM) + (1-α) * sum((ΔOBS - ΔSIM)^2)
     */
    static double levelSlopeLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double alpha = 0.5
    );

    /**
     * L3: Second-Difference (Curvature) Loss
     * Penalizes mismatch in acceleration at inflection points
     */
    static double diffCurvatureLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim
    );

    /**
     * L3': Combined Level + Slope + Curvature Loss
     * L3' = α1*MSE + α2*|ΔOBS-ΔSIM|^2 + α3*|Δ²OBS-Δ²SIM|^2
     */
    static double levelSlopeCurvatureLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double alpha1 = 0.34,
        double alpha2 = 0.33
    );

    /**
     * L4: Normalized/Scaled Slope Loss
     * Normalizes slope differences by standard deviation of observed differences
     */
    static double normalizedSlopeLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double epsilon = 1e-8
    );

    /**
     * L5: Persistence-Index-Based Loss
     * Minimizing this is equivalent to maximizing PI
     * L5 = MSE(OBS,SIM) / MSE(OBS, persistence(OBS))
     */
    static double persistenceIndexLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double epsilon = 1e-8
    );

    /**
     * DILATE Loss (simplified version)
     * Combines shape (soft-DTW) and temporal alignment terms
     */
    static double dilateLoss(
        const Eigen::VectorXd& obs,
        const Eigen::VectorXd& sim,
        double delta = 0.5,
        double gamma = 0.1
    );

private:
    /**
     * Compute first difference (Δx_i = x_i - x_{i-1})
     * Returns vector of size n-1
     */
    static Eigen::VectorXd firstDifference(const Eigen::VectorXd& x);

    /**
     * Compute second difference (Δ²x_i = x_i - 2*x_{i-1} + x_{i-2})
     * Returns vector of size n-2
     */
    static Eigen::VectorXd secondDifference(const Eigen::VectorXd& x);

    /**
     * Compute persistence forecast (y_persist[0] = y[0], y_persist[i] = y[i-1])
     */
    static Eigen::VectorXd persistenceForecast(const Eigen::VectorXd& y);

    /**
     * Soft-DTW computation for DILATE shape term
     */
    static double softDTW(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& yhat,
        double gamma = 0.1
    );

    /**
     * Temporal penalty term for DILATE
     */
    static double temporalPenalty(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& yhat
    );
};

#endif // OBJECTIVE_FUNCTIONS_HPP
