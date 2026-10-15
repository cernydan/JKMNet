#ifndef LSTMLAYER_HPP
#define LSTMLAYER_HPP

/**
 * @file LSTMLayer.hpp
 * @brief Declares the LSTMLayer class — an LSTM layer with forward pass over a time-step segment and backpropagation through time (BPTT) — plus the LSTMSettings struct.
 */

#include "eigen-3.4/Eigen/Dense"
#include "ConfigIni.hpp"

/// Dimensions and state flags of one LSTM layer.
struct LSTMSettings{
  int cells = 0; ///< number of cells (neurons) in layer
  int cells4gates = 0; ///< number of gates in all cells (cells * 4)
  int inputs = 0; ///< number of input values to layer
  int timeSteps = 0; ///< number of time steps in one computation segment of layer
  int outTS = 0; ///< number of last time steps used as actual output
  bool initialized = false; ///< true after initLSTMLayer() has sized all matrices
  bool isFirstLayer = false; ///< true for a layer fed directly by the input data (no input deltas are computed)
};

/// LSTM layer processing a segment of time steps: forward pass, BPTT gradients, and weight updates.
/**
 * All four gates are stored stacked by rows in the order candidate, forget,
 * input, output, so W and U have 4*cells rows and the bias b has 4*cells
 * entries. Per time step t the forward pass computes, with x_t the input row,
 * h_{t-1} the previous output, and c the cell count:
 * - gatesActivations = W * x_t + b + U * h_{t-1}
 * - candidate g = tanh(act[0 : c)), forget f = sig(act[c : 2c)),
 *   input i = sig(act[2c : 3c)), output o = sig(act[3c : 4c))
 * - cellState_t = f * cellState_{t-1} + i * g
 * - output h_t = o * tanh(cellState_t)
 *
 * forwardOutput keeps only the last settings.outTS columns of the output
 * (cells x outTS). calculateGradients performs BPTT from the last time step
 * backwards, combining the next-layer delta with the recurrent gradient
 * (U^T * deltaGates) and the forget-gate path into the cell state, producing
 * Wgradient = deltaGates * timeStepsInputs, Ugradient = deltaGates * h^T,
 * bGradient = rowwise sum of deltaGates, and (for non-first layers)
 * deltaInputs = W^T * deltaGates for the preceding layer. The forget-gate
 * bias is initialized to 1 to help training.
 *
 * @see weight_init_type (Layer.hpp/ConfigIni.hpp)
 */
class LSTMLayer
{
public:
  LSTMLayer() = default;  //!< The constructor
  ~LSTMLayer() = default;  //!< The destructor
  LSTMLayer(const LSTMLayer&) = default;  //!< The copy constructor
  LSTMLayer& operator=(const LSTMLayer&) = default;  //!< The assignment operator
  LSTMLayer(LSTMLayer&&) = default;  //!< The move constructor
  LSTMLayer& operator=(LSTMLayer&&) = default;  //!< The move assignment operator

  Eigen::VectorXd sigmoidVector(const Eigen::VectorXd& vec);  //!< Sigmoid function for all elements in vector
  Eigen::VectorXd tanhVector(const Eigen::VectorXd& vec);     //!< Hyperbolic tangent function for all elements in vector

  /// Initialize LSTMLayer, its weights using specified technique.
  /**
   * Sizes all weight, state, and gradient matrices; W and U are filled with
   * the selected scheme ("RANDOM" uniform in [minVal, maxVal], "HE", "XG";
   * unknown falls back to RANDOM), the bias starts at zero with the
   * forget-gate segment set to 1.
   * @param numInputs Number of input values per time step (columns of an input segment).
   * @param numCells Number of LSTM cells (output dimension per time step).
   * @param numTimeSteps Number of time steps in one input segment.
   * @param numOutputTimeSteps Number of trailing time steps exposed as the forward output.
   * @param isFirstLayer True when this layer consumes raw inputs (skips computing input deltas).
   * @param initType Weight initialization name.
   * @param rngSeed RNG seed (0 = non-deterministic random_device).
   * @param minVal Lower bound of the RANDOM initialization range.
   * @param maxVal Upper bound of the RANDOM initialization range.
   */
  void initLSTMLayer(const int numInputs,        
                     const int numCells,
                     const int numTimeSteps,
                     const int numOutputTimeSteps,
                     bool isFirstLayer,
                     std::string initType = "RANDOM",
                     int rngSeed = 0,
                     double minVal = 0.0,
                     double maxVal = 1.0);

  /// Set inputs for n time-steps.
  /**
   * @param inputSegment Matrix with one row per time step and numInputs columns.
   * @throw std::logic_error If called before initLSTMLayer.
   * @throw std::invalid_argument If the column count differs from the initialized input size.
   */
  void setInputTSSegment(const Eigen::MatrixXd& inputSegment);
  void calculateTimeSteps();      //!< Calculate outputs for n time-steps
  /// Backpropagation through time: compute gradients of W, U, b and the input deltas from the stored forward pass.
  /**
   * Walks from the last time step backwards, combining the next-layer delta
   * (only at the outTS output steps) with the recurrent gradient and the
   * forget-gate cell-state path. Must be called after calculateTimeSteps and
   * setDeltaFromNextLayer.
   */
  void calculateGradients();
  /// Add externally computed gradients to the stored gradient accumulators (for batching).
  /**
   * @param dW Input-weight gradient (4*cells x inputs).
   * @param dU Recurrent-weight gradient (4*cells x cells).
   * @param db Bias gradient (4*cells).
   */
  void accumulateGradients(const Eigen::MatrixXd& dW, const Eigen::MatrixXd& dU, const Eigen::VectorXd& db);
  void resetGradients();    //!< Zero the W, U, and b gradient accumulators
  /// Apply one plain SGD step (W/U/b -= learningRate * gradient) from the stored gradients.
  void updateWeights(double learningRate);
  /// Apply one Adam update step to W, U, and b from the stored gradients.
  /**
   * @param learningRate Adam step size.
   * @param iterationNum 1-based step counter for Adam's bias correction.
   * @param beta1 Exponential decay of the first moment estimate.
   * @param beta2 Exponential decay of the second moment estimate.
   * @param epsi Small stabilizer in the denominator.
   */
  void updateAdam(double learningRate, int iterationNum, double beta1, double beta2, double epsi);
  /// Clear the forward-pass states, inputs, and deltas (weight gradients and Adam moments are kept).
  /**
   * Called between samples in online training so samples stay independent.
   */
  void eraseMemory();
  /// Set the error delta arriving from the next layer, reshaping the flat vector to (delta.size()/outTS) x outTS.
  /**
   * @param delta Flattened delta whose length must be divisible by outTS.
   * @throw std::logic_error If the length is not divisible by outTS.
   */
  void setDeltaFromNextLayer(const Eigen::VectorXd& delta);
  /// Set the error delta arriving from the next layer as a matrix.
  /**
   * Accepts rows = cells with cols = outTS (or more, in which case the last
   * outTS columns are used), or the transposed outTS x cells layout.
   * @param delta Delta matrix.
   * @throw std::logic_error On a dimension mismatch with cells/outTS.
   */
  void setDeltaFromNextLayer(const Eigen::MatrixXd& delta);
  Eigen::MatrixXd getDeltaInputs(); //!< Getter for the input deltas (inputs x timeSteps) for a preceding layer
  Eigen::MatrixXd getForwardOutput(); //!< Getter for the forward output transposed to (outTS x cells)
  Eigen::VectorXd getForwardOutputVector();   //!< Getter for the forward output flattened column-major (all cells of step t, then t+1, ...)
  Eigen::VectorXd getLastTimeStepOutput();  //!< Get only the last time step output (for sequence-to-one)
  void clearGradients();  //!< Clear accumulated gradients

private:
  LSTMSettings settings;  //!< Settings of the layer
  Eigen::MatrixXd W;  //!< The input weight matrix (all gates - by rows candidate, forget, input, output)
  Eigen::MatrixXd U;  //!< The hidden state (short memory) gate weight matrix (all gates - by rows candidate, forget, input, output)
  Eigen::VectorXd b;  //!< The bias vector (all gates...)
  Eigen::MatrixXd timeStepsInputs;  //!< The matrix of inputs for n time-steps (rows = time-steps)
  Eigen::MatrixXd gatesActivations;  //!< Computed activations all gates (by rows) all time steps (by columns)
  Eigen::MatrixXd gatesOutputs;   //!< Computed outputs all gates (by rows) all time steps (by columns)
  Eigen::MatrixXd cellState;      //!< Computed cell state (long memory) all cells (rows) all time steps (columns)
  Eigen::MatrixXd tanhCellState;      //!< Computed cell state (long memory) all cells (rows) all time steps (columns) after tanh function
  Eigen::MatrixXd output;         //!< Computed outputs (short memory) all cells (rows) all time steps (columns)
  Eigen::MatrixXd forwardOutput;  //!< Computed outputs all cells (rows) only outTS time steps (columns)
  Eigen::MatrixXd Wgradient;  //!< Gradient of the loss w.r.t. the input weight matrix W
  Eigen::MatrixXd Ugradient;  //!< Gradient of the loss w.r.t. the recurrent weight matrix U
  Eigen::VectorXd bGradient;  //!< Gradient of the loss w.r.t. the bias vector b
  Eigen::MatrixXd W_MtForAdam;    //!< Adam first-moment accumulator for W
  Eigen::MatrixXd W_VtForAdam;    //!< Adam second-moment accumulator for W
  Eigen::MatrixXd U_MtForAdam;    //!< Adam first-moment accumulator for U
  Eigen::MatrixXd U_VtForAdam;    //!< Adam second-moment accumulator for U
  Eigen::VectorXd b_MtForAdam;    //!< Adam first-moment accumulator for b
  Eigen::VectorXd b_VtForAdam;    //!< Adam second-moment accumulator for b
  Eigen::MatrixXd deltaOutput;    //!< Delta of the layer output h per cell (rows) and time step (columns)
  Eigen::MatrixXd deltaCellState; //!< Delta of the cell state per cell (rows) and time step (columns)
  Eigen::MatrixXd deltaGates; //!< Delta of the gate activations (4*cells rows, gates stacked, by time-step columns)
  Eigen::MatrixXd nextLayerDelta; //!< Error delta received from the following layer (cells x outTS)
  Eigen::MatrixXd deltaInputs;    //!< Delta w.r.t. the layer inputs (inputs x timeSteps), computed only for non-first layers
};

#endif // LSTMLAYER_HPP
