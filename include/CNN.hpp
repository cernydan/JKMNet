#ifndef CNN_HPP
#define CNN_HPP

#include "eigen-3.4/Eigen/Dense"
#include "CNNLayer.hpp"

/**
 * @file CNN.hpp
 * @brief Declares the CNN class, a 1D convolutional neural network composed of stacked CNNLayer objects.
 */

/// One-dimensional convolutional neural network built from a stack of CNNLayer objects.
/**
 * The network is configured by an architecture description, initialized on an
 * example input, and then used for 1D forward passes and backpropagation
 * through all of its layers. It is typically chained with an MLP head: the CNN
 * reduces the input to a vector (one row) that can be fed into an MLP, and the
 * MLP's backpropagated delta is handed back to the CNN for filter updates.
 */
class CNN
{
public:
    CNN() = default;  //!< The constructor
    ~CNN() = default;  //!< The destructor
    CNN(const CNN&) = default;  //!< The copy constructor
    CNN& operator=(const CNN&) = default;   //!< The assignment operator

    /// Set the layer architecture of the network.
    /**
     * Each inner vector describes one layer as {filterSize, numberOfFilters,
     * poolSize}; the order of entries defines the order of layers.
     * @param newArc Architecture description: one {filterSize, numFilters, poolSize} triple per layer.
     */
    void setArchitecture(const std::vector<std::vector<int>>& newArc);

    /// Create and initialize all layers from the stored architecture.
    /**
     * Builds one CNNLayer per entry of the architecture. The first layer is
     * sized against the given input matrix; each subsequent layer is sized
     * against the output of the previous one. One forward pass is performed
     * during initialization so that all layer outputs have valid shapes.
     * @param input Example 1D input (rows = variable values/time steps, cols = variables) used to derive layer shapes.
     * @param rngSeed Seed for the random number generator used to initialize filters (0 = random_device).
     */
    void initCNN(const Eigen::MatrixXd& input, int rngSeed);

    /// Run a 1D forward pass through all layers.
    /**
     * @param input 1D input matrix (rows = values/time steps, cols = variables).
     * @throw std::logic_error If called before initCNN, or if the result does not collapse to a single row.
     * @note The final layer output must have exactly one row; that row becomes the network output vector.
     */
    void runCNN1D(const Eigen::MatrixXd& input);
    Eigen::VectorXd getOutput();

    /// Backpropagate a delta from the following MLP through all CNN layers and update filters with Adam.
    /**
     * Propagates the error from the last layer to the first, computing filter,
     * bias and input gradients in each layer, then applies one Adam update step
     * (beta1 = 0.9, beta2 = 0.99, eps = 1e-8) to every layer.
     * @param deltaFromMlp Error signal of the next (MLP) layer, one entry per element of the CNN output vector.
     * @param learningRate Step size for the Adam update.
     * @param iterationNum Current training iteration (1-based), used for Adam bias correction.
     */
    void bpAdam1input(Eigen::VectorXd deltaFromMlp, double learningRate, int iterationNum);

    /// Compute network outputs for multiple 1D inputs at once.
    /**
     * @param inputs Matrix whose rows hold the individual 1D input samples.
     * @todo Currently unimplemented: the loop body is empty and neither `outputs` nor `output` is set.
     */
    void calculateOutputs1D(const Eigen::MatrixXd& inputs);

protected:
private:
    std::vector<CNNLayer> layers_;  //!< Ordered stack of CNN layers (index 0 = first/ input layer)
    std::vector<std::vector<int>> architecture;  //!< Layer architecture: one {filterSize, numFilters, poolSize} triple per layer
    Eigen::VectorXd output;  //!< Output vector of the whole network (single row of the last layer's output)
    Eigen::MatrixXd outputs; //!< Matrix of network outputs for multiple inputs (filled by calculateOutputs1D)
};

#endif // CNN_HPP
