/**
 * @file Layer.hpp
 * @brief Declares the Layer class (one fully-connected layer of an MLP) together with the weight initialization and activation function enums.
 */
#ifndef LAYER_HPP
#define LAYER_HPP

#include "eigen-3.4/Eigen/Dense"
#include "ObjectiveFunctions.hpp"

/**
 * @brief All weight initialization techniques.
 */
enum class weight_init_type
{
    RANDOM,  //!< Uniform random initialization in [minVal, maxVal]
    LHS,     //!< Latin hypercube sampling over all weights of the matrix
    LHS2,    //!< Latin hypercube sampling performed column by column
    HE,      //!< He initialization (normal, std = sqrt(2 / numInputs))
    XG       //!< Xavier/Glorot initialization (normal, std = sqrt(2 / (numInputs + numNeurons)))
};

/**
 * @brief All activation functions.
 */
enum class activ_func_type
{
    RELU,      //!< f(x) = max(0, x)
    SIGMOID,   //!< f(x) = 1 / (1 + exp(-x))
    LINEAR,    //!< f(x) = x
    TANH,      //!< f(x) = 2 / (1 + exp(-2x)) - 1
    GAUSSIAN,  //!< f(x) = exp(-x^2)
    IABS,      //!< f(x) = x / (1 + |x|)
    LOGLOG,    //!< f(x) = exp(-exp(-x))
    CLOGLOG,   //!< f(x) = 1 - exp(-exp(x))
    CLOGLOGM,  //!< f(x) = 1 - 2 * exp(-0.7 * exp(x))
    ROOTSIG,   //!< f(x) = x / (1 + sqrt(1 + x^2))
    LOGSIG,    //!< f(x) = sigmoid(x)^2
    SECH,      //!< f(x) = 2 / (exp(x) + exp(-x))
    WAVE,      //!< f(x) = (1 - x^2) * exp(-x^2)
    LEAKYRELU  //!< f(x) = max(0.01 * x, x)
}; //!< All activation functions

/// Array containing every activation function, e.g. for iterating over all of them during hyperparameter search.
const activ_func_type all_activ_func[]{
    activ_func_type::RELU,
    activ_func_type::SIGMOID,
    activ_func_type::LINEAR,
    activ_func_type::TANH,
    activ_func_type::GAUSSIAN,
    activ_func_type::IABS,
    activ_func_type::LOGLOG,
    activ_func_type::CLOGLOG,
    activ_func_type::CLOGLOGM,
    activ_func_type::ROOTSIG,
    activ_func_type::LOGSIG,
    activ_func_type::SECH,
    activ_func_type::WAVE,
    activ_func_type::LEAKYRELU
};
const unsigned numActivFunc = std::size(all_activ_func); //!< Total number of activation functions

/**
 * @brief One fully-connected layer of a multilayer perceptron.
 *
 * Stores the weight matrix (including a bias column), the layer inputs,
 * activations, outputs and backpropagation deltas, and provides the
 * forward pass, gradient computation and weight update rules
 * (plain gradient descent, momentum, penalized gradient descent and Adam).
 *
 * @note Index/dimension conventions: the weight matrix has shape
 *       (numNeurons x (numInputs + 1)); the inputs vector always carries
 *       a trailing 1.0 element that acts as the bias input.
 */
class Layer
{
public:
    Layer();                              //!< The constructor
    ~Layer();                             //!< The destructor
    Layer(const Layer &other);            //!< The copy constructor
    Layer &operator=(const Layer &other); //!< The assignment operator
    Layer(Layer&& other) noexcept;        //!< The move copy constructor
    Layer& operator=(Layer&& other) noexcept;  //!< The move assignment operator

    /// Initialize a fully-connected layer.
    /**
     * Allocates all internal vectors/matrices, appends the bias input and
     * initializes the weights with the chosen technique.
     * @param numFeatures Number of real input features (without the bias input).
     * @param numNeurons Number of neurons in the layer.
     * @param initType Weight initialization technique.
     * @param activFunc Activation function used by the layer.
     * @param rngSeed Seed for the random generator; 0 selects a random seed.
     * @param minVal Lower bound of the initialization range (used by RANDOM/LHS/LHS2).
     * @param maxVal Upper bound of the initialization range (used by RANDOM/LHS/LHS2).
     */
    void initLayer(unsigned numFeatures,
        unsigned numNeurons,
        weight_init_type initType = weight_init_type::RANDOM,
        activ_func_type  activFunc = activ_func_type::RELU,
        int rngSeed = 0,
        double minVal = 0.0,
        double maxVal = 1.0);

    /// Initialize weights using the specified technique.
    /**
     * @param numNeurons Number of rows of the weight matrix.
     * @param numInputs Number of columns of the weight matrix (including the bias column).
     * @param initType Weight initialization technique.
     * @param minVal Lower bound of the initialization range (used by RANDOM/LHS/LHS2).
     * @param maxVal Upper bound of the initialization range (used by RANDOM/LHS/LHS2).
     * @param rngSeed Seed for the random generator; 0 selects a random seed.
     * @note An unknown initType falls back to RANDOM and prints an error to stderr.
     */
    void initWeights(unsigned numNeurons,
        unsigned numInputs,
        weight_init_type initType,
        double minVal,
        double maxVal,
        int rngSeed);

    /// Get the input vector of the layer (including the trailing bias element).
    Eigen::VectorXd getInputs();

    /// Set the input vector; a trailing 1.0 bias element is appended automatically.
    /**
     * @param newInputs Real inputs, without the bias element.
     */
    void setInputs(const Eigen::VectorXd &newInputs);

    /// Get the backpropagation gradient matrix (dE/dW).
    Eigen::MatrixXd getGradient();

    /// Set the backpropagation gradient matrix (dE/dW).
    void setGradient(const Eigen::MatrixXd& grad);

    /// Calculate the gradient matrix for online backpropagation: grad = deltas * inputs^T.
    void calculateOnlineGradient();

    /// Accumulate the gradient matrix for batch backpropagation: grad += deltas * inputs^T.
    void calculateBatchGradient();

    /// Calculate the gradient matrix for online backpropagation with L2 weight penalization: grad = deltas * inputs^T + lambda * weights.
    /**
     * @param lambda Weight penalization (regularization) coefficient.
     */
    void calculateOnlineGradientPenalize(double lambda = 0.0001);

    /// Get the weight matrix of the layer.
    Eigen::MatrixXd getWeights() const;

    /// Set the weight matrix of the layer.
    void setWeights(const Eigen::MatrixXd &newWeights);

    /// Apply a gradient descent step: W = W - learningRate * (dE/dW).
    /**
     * @param learningRate Step size eta.
     */
    void updateWeights(double learningRate);

    /// Apply a gradient descent step using the Adam algorithm (with bias correction).
    /**
     * @param learningRate Step size.
     * @param iterationNum 1-based training iteration index, used for bias correction.
     * @param beta1 Exponential decay rate for the first moment estimate.
     * @param beta2 Exponential decay rate for the second moment estimate.
     * @param epsi Small constant for numerical stability.
     */
    void updateAdam(double learningRate, int iterationNum, double beta1, double beta2, double epsi);

    /// Apply a gradient descent step with momentum: W = W - learningRate * (dE/dW) + moment * (W - W_prev).
    /**
     * @param learningRate Step size eta.
     * @param moment Momentum coefficient.
     */
    void updateWeightsMomentum(double learningRate, double moment = 0.9);

    /// Calculate the weighted sum (linear combination of inputs), i.e. the pre-activations.
    /**
     * @return Vector of pre-activation values, one per neuron.
     * @throw std::runtime_error When weights.cols() does not match inputs.size().
     * @note Does not store the result in the activations member (unlike calculateOutput()).
     */
    Eigen::VectorXd calculateWeightedSum();

    /// Apply an activation function element-wise to the given weighted sum.
    /**
     * @param weightedSum Pre-activation values.
     * @param activFuncType Activation function to apply.
     * @return Activated output vector of the same size as @p weightedSum.
     */
    Eigen::VectorXd setActivationFunction(const Eigen::VectorXd &weightedSum, activ_func_type activFuncType);

    /// Apply the derivative of an activation function element-wise to the given weighted sum.
    /**
     * @param weightedSum Pre-activation values.
     * @param activFuncType Activation function whose derivative is applied.
     * @return Vector of derivative values of the same size as @p weightedSum.
     */
    Eigen::VectorXd setActivFunDeriv(const Eigen::VectorXd &weightedSum, activ_func_type activFuncType);

    /// Calculate the complete layer output (weighted sum + activation function).
    /**
     * @param activFuncType Activation function to apply.
     * @return Output vector of the layer (also stored in the output member).
     * @note Does not store the pre-activations in the activations member (unlike calculateOutput()).
     */
    Eigen::VectorXd calculateLayerOutput(activ_func_type activFuncType);

    /// Calculate the layer output, storing both the pre-activations and the activated output in the layer state.
    /**
     * @param activFuncType Activation function to apply.
     */
    void calculateOutput(activ_func_type activFuncType);

    /// Calculate the layer deltas for backpropagation from the deltas and weights of the next layer.
    /**
     * The bias column of @p nextWeights is excluded from the backpropagated product.
     * Requires calculateOutput() to have been called first so that the pre-activations are available.
     * @param nextWeights Weight matrix of the next (closer to the output) layer.
     * @param nextDeltas Delta vector of the next layer.
     * @param activFuncType Activation function of this layer (its derivative is used).
     */
    void calculateDeltas(const Eigen::MatrixXd &nextWeights, const Eigen::VectorXd &nextDeltas, activ_func_type activFuncType);

    /// Get the output vector of the layer.
    Eigen::VectorXd getOutput();

    /// Get the deltas vector of the layer.
    Eigen::VectorXd getDeltas();

    /// Set the deltas vector of the layer.
    void setDeltas(const Eigen::VectorXd &newDeltas);

    /// Set the deltas of an output layer from the gradient of the chosen loss function w.r.t. the layer output.
    /**
     * The current output vector is treated as the prediction. The gradient of
     * the loss w.r.t. the output is computed by ObjectiveFunctions::computeGradient
     * (MSE included, with its 2*(sim-obs)/n normalization) and then multiplied by
     * the derivative of this layer's activation function, so the deltas are
     * correct at the pre-activation level.
     * @param obs Observed (target) values with the same size as the output.
     * @param lossType Objective (loss) function type.
     * @param alpha Weighting parameter for combined losses.
     * @param alpha2 Second weighting parameter (curvature term).
     * @param epsilon Small constant for numerical stability.
     * @note For DILATE2, alpha2 doubles as the soft-DTW smoothing parameter (gamma).
     */
    void setDeltasFromLossGradient(const Eigen::VectorXd &obs, objective_func_type lossType, double alpha = 0.5, double alpha2 = 0.33, double epsilon = 1e-8);

    /// Get the derivative of this layer's activation function at the stored pre-activations.
    /**
     * Requires calculateOutput() to have been called so the pre-activations are
     * current. Used when setting output-layer deltas manually (e.g. MLP::horizonBackprop)
     * to apply the same activation chain rule as setDeltasFromLossGradient.
     * @return Vector of f'(z) values, same size as the layer output.
     */
    Eigen::VectorXd getActivationDerivative();

    /// Reset the Adam optimizer state variables (MtForAdam and VtForAdam) to zero.
    void resetAdamState();

    /// Map an activ_func_type value to its string name.
    /**
     * @return Name of the activation function, or "Unknown" for an unhandled value.
     */
    static std::string activationName(activ_func_type f);

    /// Map a weight_init_type value to its string name.
    /**
     * @return Name of the initialization technique, or "Unknown" for an unhandled value.
     */
    static std::string wInitTypeName(weight_init_type w);

    /// Get the weight matrix flattened into a vector (must call weightsToVector() first).
    Eigen::VectorXd getWeightsVector();

    /// Flatten the weight matrix row-by-row into the weightsVector member.
    void weightsToVector();

protected:
private:
    Eigen::MatrixXd weights;  //!< The weight matrix for the layer
    Eigen::VectorXd weightsVector;  //!< The weight matrix arranged into vector by rows
    Eigen::VectorXd inputs;  //!< The input vector to the layer
    Eigen::VectorXd output;  //!< The output vector of the layer
    Eigen::VectorXd activations;  //!< The activation vector of the layer
    // Eigen::VectorXd bias;  //!< The bias vector
    Eigen::VectorXd deltas; //!< The deltas vector of the layer
    activ_func_type activ_func = activ_func_type::RELU;  //!< The type of activation function, where default is RELU
    Eigen::MatrixXd weightGrad;  //!< The backpropagation gradient matrix for the layer (∂E/∂W)
    Eigen::MatrixXd MtForAdam;  //!< Mt parameter for every weight in ADAM algorithm
    Eigen::MatrixXd VtForAdam;  //!< Vt parameter for every weight in ADAM algorithm
    Eigen::MatrixXd previousWeights; //!< The weight matrix saved from the previous training step (used for momentum)
};

#endif // LAYER_HPP
