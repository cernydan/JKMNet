/**
 * @file MLP.hpp
 * @brief Declares the MLP class (a multilayer perceptron built from Layer objects) and the TrainingResult struct.
 */
#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include <functional>
#include "eigen-3.4/Eigen/Dense"

#include "Layer.hpp"
#include "ConfigIni.hpp"
#include "Metrics.hpp"

using namespace std;

/// Outcome of one training run of an MLP.
struct TrainingResult {
    double finalLoss = std::numeric_limits<double>::quiet_NaN();  //!< Final training loss (NaN until the trainer writes it)
    int iterations = 0;        //!< Number of iterations actually executed
    bool converged = false;    //!< True when training stopped early by reaching maxError
    int time = 0;              //!< Wall-clock time of the training run
};

/// A multilayer perceptron built from a sequence of Layer objects.
/**
 * The network is configured with an architecture vector (neurons per layer),
 * one activation function and one weight-initialization scheme per layer, and
 * an objective function used by all training routines. Layers (and weights)
 * are materialized by initMLP(); afterwards runMLP() performs forward passes
 * and the training methods (onlineAdam, batchAdam, onlineBP, ...) update the
 * weights.
 * @see activ_func_type, weight_init_type (Layer.hpp), objective_func_type
 *      (ObjectiveFunctions.hpp)
 */

/**
 * Optional context for the *EpochVal trainers controlling in which space the
 * epoch metrics are computed. When given, calibration/validation metrics are
 * computed on original-unit (inverse-transformed) data, with the calibration
 * side in chronological (unshuffled) order so that order-sensitive metrics
 * such as PI get a meaningful persistence baseline. When null (default), the
 * metrics are computed directly on the passed (transformed) matrices as before.
 */
struct EpochMetricContext {
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> prepCalPreds;  //!< Postprocess calibration predictions (inverse transform + unshuffle) before metrics
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> prepValPreds;  //!< Postprocess validation predictions (inverse transform) before metrics
    Eigen::MatrixXd YtrueCalOrig;  //!< Calibration ground truth in original units, chronological order; empty = use Ytrain
    Eigen::MatrixXd YtrueValOrig;  //!< Validation ground truth in original units; empty = use Yval
};

class MLP {

    public:
        MLP();  //!< The constructor
        ~MLP();  //!< The destructor
        MLP(const MLP& other);  //!< The copy constructor
        MLP& operator=(const MLP& other);  //!< The assignment operator
        MLP(MLP&& other) noexcept;        //!< The move copy constructor
        MLP& operator=(MLP&& other) noexcept;  //!< The move assignment operator

        //MLP() = default;  //!< The constructor
       //~MLP() = default;  //!< The destructor 
        //MLP(const MLP&) = default;  //!< The copy constructor
       //MLP& operator=(const MLP&) = default;   //!< The assignment operator

        //!< Architecture (neurons per layer)
        std::vector<unsigned> getArchitecture();  //!< Getter for the architecture
        /// Set the network architecture (number of neurons per layer).
        /**
         * Resizes the per-layer configuration vectors (activations, weight
         * init types); the value is validated against them here.
         * @param architecture Neuron count of each layer, first to last.
         */
        void setArchitecture(const std::vector<unsigned>& architecture);
        void printArchitecture();  //!< Print the architecture

        // Activations for each layer
        std::vector<activ_func_type> getActivations();  //!< Getter for the activation function
        //void setActivations(std::vector<activ_func_type>& funcs);  //!< Setter for the activation function
        void printActivations();  //!< Print the activation function

        /// Set the activation function of each layer.
        /**
         * @param funcs One activation per layer; must match the architecture size.
         * @throw std::invalid_argument On a size mismatch.
         * @see activ_func_type
         */
        void setActivations(const std::vector<activ_func_type>& funcs);
        /// Set the weight initialization scheme of each layer.
        /**
         * @param wInits One init type per layer; must match the architecture size.
         * @throw std::invalid_argument On a size mismatch.
         * @see weight_init_type
         */
        void setWInitType(const std::vector<weight_init_type>& wInits);

         // Weight‐init types per layer
        std::vector<weight_init_type> getWInitType();  //!< Getter for the weight initialization type
        //void setWInitType(std::vector<weight_init_type>& wInits);  //!< Setter for the weight initialization type
        void printWInitType();  //!< Print the weight initialization type

        //!< Number of layers
        size_t getNumLayers() const;  //!< Getter for the number of layers
        void setNumLayers(size_t layers);  //!< Setter for the number of layers

        //!< Number of inputs (neurons in first layer)
        unsigned getNumInputs();  //!< Getter for the number of inputs
        void setNumInputs(unsigned inputs);  //!< Setter for the number of inputs

        //!< Number of neurons at specific layer
        unsigned getNumNeuronsInLayers(std::size_t index);  //!< Getter for the number of neurons at a specific layer
        void setNumNeuronsInLayers(std::size_t index, unsigned count);  //!< Setter for the number of neurons at a specific layer

        // Getter and Setter for inputs
        Eigen::VectorXd& getInps();  //!< Getter for the inputs
        void setInps(Eigen::VectorXd& inputs);   //!< Setter for the inputs

        // Getter and Setter for weigths
        /// Get the weight matrix of one layer (including the bias row/column).
        /**
         * @param layerIndex Index of the layer (0-based).
         * @return Weight matrix of the layer.
         * @throw std::out_of_range If the network is not initialized or the index is invalid.
         */
        Eigen::MatrixXd getWeights(size_t layerIndex) const;
        /// Replace the weight matrix of one layer (specifying new weights for Adam update too).
        /**
         * @param layerIndex Index of the layer (0-based).
         * @param W New weight matrix; must match the existing layer's weight dimensions.
         * @throw std::out_of_range On an invalid index.
         * @throw std::invalid_argument On a shape mismatch.
         */
        void setWeights(size_t layerIndex, const Eigen::MatrixXd& W);

        Eigen::VectorXd getWeightsVectorMlp(); //!< Getter for the flattened weights vector of the MLP
        /// Merge the weight vectors of all layers into one flat vector (weightsVectorMlp).
        void weightsToVectorMlp();

        /// Save per-layer weight matrices in readable CSV text (per-layer blocks, with config header).
        /**
         * @param path Destination file path.
         * @return True on success, false if the file cannot be opened.
         */
        bool saveWeightsCsv(const std::string &path) const;
        /// Save per-layer weight matrices in a compact binary format (with config header).
        /**
         * @param path Destination file path.
         * @return True on success, false if the file cannot be opened.
         */
        bool saveWeightsBinary(const std::string &path) const;
        /// Save the flattened weight vector (weightsVectorMlp) in readable CSV text.
        /**
         * @param path Destination file path.
         * @return True on success, false if the file cannot be opened.
         */
        bool saveWeightsVectorCsv(const std::string &path) const;
        /// Save the flattened weight vector (weightsVectorMlp) in a compact binary format.
        /**
         * @param path Destination file path.
         * @return True on success, false if the file cannot be opened.
         */
        bool saveWeightsVectorBinary(const std::string &path) const;
        /// Append the flattened weight vector to one shared CSV file (used for ensemble runs).
        /**
         * @param path Destination file path (shared across ensemble runs).
         * @param isFirstRun True for the first ensemble run (writes the header, truncates); false appends.
         * @return True on success, false if the file cannot be opened.
         */
        bool appendWeightsVectorCsv(const std::string &path, bool isFirstRun) const;

        /// Load per-layer weights from a CSV text file, validating them against an optional config file.
        /**
         * @param wPath Path to the weights CSV file.
         * @param confPath Path to the expected config header file ("" to skip the check).
         * @return True on success; false on open/parse/validation failure.
         */
        bool loadWeightsCsv(const std::string &wPath, const std::string &confPath);
        /// Load per-layer weights in compact binary format, validating them against an optional config file.
        /**
         * @param wPath Path to the binary weights file.
         * @param confPath Path to the expected config header file ("" to skip the check).
         * @return True on success; false on open/parse/validation failure.
         */
        bool loadWeightsBinary(const std::string &wPath, const std::string &confPath);
        /// Load the flattened weight vector from CSV text and distribute it into the layers.
        /**
         * @param wPath Path to the weights-vector CSV file.
         * @param confPath Path to the expected config header file ("" to skip the check).
         * @return True on success; false on open/parse/validation failure.
         */
        bool loadWeightsVectorCsv(const std::string &wPath, const std::string &confPath);
        Eigen::VectorXd& getOutput();   //!< Getter for the output vector (last forward pass result)
        /// Check that the current input vector size matches the first layer's input size.
        /**
         * @return True when Inps.size() equals the first layer weight-matrix input dimension.
         */
        bool validateInputSize();

        /// Initialize (or re-initialize) all layer weights and perform a forward pass.
        /**
         * Builds the Layer objects from the configured architecture, seeding
         * each weight matrix with @p rngSeed, then propagates @p input through
         * all layers so the member state reflects a valid forward pass.
         * @param input Input vector (its length defines the first layer input size).
         * @param rngSeed RNG seed for weight initialization (0 = non-deterministic).
         * @return Output of the last layer (length nNeurons.back()).
         * @throw std::logic_error If architecture/activations/weight-init sizes mismatch.
         */
        Eigen::VectorXd initMLP(const Eigen::VectorXd& input, int rngSeed);
        /// Forward pass reusing the existing weights.
        /**
         * @param input Input vector; must match the first layer input size.
         * @return Output of the last layer (length nNeurons.back()).
         * @throw std::logic_error If called before initMLP (layers_ is empty).
         */
        Eigen::VectorXd runMLP(const Eigen::VectorXd& input);
        /// Compare whether initMLP followed by runMLP reproduces the initMLP output.
        /**
         * Works on an internal copy, leaving this object's state untouched.
         * @param input Input vector.
         * @param tol Elementwise comparison tolerance.
         * @param rngSeed Seed passed to the re-initialization.
         * @return True if both outputs agree within @p tol.
         */
        bool compareInitAndRun(const Eigen::VectorXd& input, double tol = 1e-6, int rngSeed = 0) const;
        /// Repeatability check: runMLP is deterministic after initMLP.
        /**
         * @param input Input vector.
         * @param repeats Number of repeated forward passes.
         * @param tol Elementwise comparison tolerance.
         * @param rngSeed Seed passed to the re-initialization.
         * @return True if all @p repeats outputs agree within @p tol.
         */
        bool testRepeatable(const Eigen::VectorXd& input, int repeats = 10, double tol = 1e-8, int rngSeed = 0) const;
        /// Forward pass for one pattern followed by an immediate backpropagation weight update.
        /**
         * @param input Input vector.
         * @param obsOut Target (observed) output vector.
         * @param learningRate Step size of the plain SGD update.
         * @throw std::logic_error If called before initMLP.
         */
        void runAndBP(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut, double learningRate);
        /// Forward pass for one pattern followed by an immediate Adam weight update.
        /**
         * @param input Input vector.
         * @param obsOut Target (observed) output vector.
         * @param learningRate Adam step size.
         * @param iterationNum 1-based step counter for Adam's bias correction.
         * @throw std::logic_error If called before initMLP.
         */
        void runAndBPadam(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut, double learningRate, int iterationNum);
        /// Forward pass for one pattern; accumulate the per-layer batch gradients without updating weights.
        /**
         * Call repeatedly over a batch, then apply updateWeights()/updateWeightsAdam().
         * @param input Input vector.
         * @param obsOut Target (observed) output vector.
         * @throw std::logic_error If called before initMLP.
         */
        void runAndCalculateBatchGradient(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut);
        /// Apply one plain SGD step (W -= learningRate * gradient) from the accumulated batch gradient, then clear it.
        /**
         * @param learningRate Step size.
         * @throw std::logic_error If called before initMLP.
         */
        void updateWeights(double learningRate);
        /// Apply one Adam step per layer from the accumulated batch gradient, then clear it.
        /**
         * Uses beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8.
         * @param learningRate Adam step size.
         * @param iterationNum 1-based step counter for Adam's bias correction.
         * @throw std::logic_error If called before initMLP.
         */
        void updateWeightsAdam(double learningRate, int iterationNum);
        /// Same as updateWeightsAdam but keeps the accumulated gradients, so further patterns keep accumulating.
        /**
         * Used by the LSTM horizon training where several steps contribute before clearing.
         * @param learningRate Adam step size.
         * @param iterationNum 1-based step counter for Adam's bias correction.
         * @throw std::logic_error If called before initMLP.
         */
        void updateWeightsAdamNoClear(double learningRate, int iterationNum);
        /// Reset the Adam moment accumulators of every layer (first/second moments).
        /**
         * Used when the MLP is trained as part of a coupled LSTM->MLP model so
         * each window starts a fresh Adam trajectory.
         */
        void resetGradientsForLSTM();

        /// Online (per-pattern) backpropagation trainer - separate input/output matrices.
        /**
         * Each epoch processes all patterns (rows of @p X = inputs, rows of
         * @p Y = targets) one by one - forward pass, backpropagation,
         * immediate weight update per pattern. Training stops early when the
         * mean epoch error falls below @p maxError; statistics go to `result`.
         * @param maxIterations Maximum number of epochs (> 0).
         * @param maxError Early-stopping error threshold (>= 0).
         * @param learningRate SGD step size in (0, 1].
         * @param X Input matrix (rows = patterns, cols = inputs).
         * @param Y Target output matrix (rows = patterns, cols = outputs).
         * @throw std::logic_error If called before initMLP.
         * @throw std::invalid_argument On empty/mismatched matrices or invalid hyperparameters.
         * @throw std::runtime_error If X.cols()/Y.cols() do not match the initialized sizes.
         */
        void onlineBP(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
        /// Online backpropagation using the Adam optimizer - separate input/output matrices.
        /**
         * Same per-pattern loop as onlineBP, but weights are updated by one
         * Adam step per pattern. Training stops early below @p maxError.
         * @param maxIterations Maximum number of epochs (> 0).
         * @param maxError Early-stopping error threshold.
         * @param learningRate Adam step size.
         * @param X Input matrix (rows = patterns, cols = inputs).
         * @param Y Target output matrix (rows = patterns, cols = outputs).
         * @throw std::logic_error If called before initMLP.
         * @throw std::invalid_argument On empty/mismatched matrices or invalid hyperparameters.
         */
        void onlineAdam(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
        /// Mini-batch backpropagation using the Adam optimizer - separate input/output matrices.
        /**
         * Each epoch processes @p batchSize-sized batches: gradients are
         * accumulated over the batch, then one Adam step is applied.
         * @param maxIterations Maximum number of epochs (> 0).
         * @param maxError Early-stopping error threshold.
         * @param batchSize Number of patterns per batch.
         * @param learningRate Adam step size.
         * @param X Input matrix (rows = patterns, cols = inputs).
         * @param Y Target output matrix (rows = patterns, cols = outputs).
         * @throw std::logic_error If called before initMLP.
         * @throw std::invalid_argument On empty/mismatched matrices or invalid hyperparameters.
         */
        void batchAdam(int maxIterations, double maxError, int batchSize, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
        /// Mini-batch backpropagation trainer - separate input/output matrices.
        /**
         * Same batch loop as batchAdam, but applies plain SGD updates.
         * @param maxIterations Maximum number of epochs (> 0).
         * @param maxError Early-stopping error threshold.
         * @param batchSize Number of patterns per batch.
         * @param learningRate SGD step size.
         * @param X Input matrix (rows = patterns, cols = inputs).
         * @param Y Target output matrix (rows = patterns, cols = outputs).
         * @throw std::logic_error If called before initMLP.
         * @throw std::invalid_argument On empty/mismatched matrices or invalid hyperparameters.
         */
        void batchBP(int maxIterations, double maxError, int batchSize, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
        /// Online backpropagation with large-weight penalization (L2-like weight decay on the gradient).
        /**
         * Same loop as onlineBP; each weight update additionally subtracts
         * @p lambda * W, penalizing large weights.
         * @param maxIterations Maximum number of epochs (> 0).
         * @param maxError Early-stopping error threshold.
         * @param learningRate SGD step size.
         * @param X Input matrix (rows = patterns, cols = inputs).
         * @param Y Target output matrix (rows = patterns, cols = outputs).
         * @param lambda Weight penalization coefficient.
         * @throw std::logic_error If called before initMLP.
         * @throw std::invalid_argument On empty/mismatched matrices or invalid hyperparameters.
         */
        void onlinePenalizeBP(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, double lambda = 0.0001);
        /// Online backpropagation with a momentum term.
        /**
         * Same loop as onlineBP; each weight update keeps a fraction
         * @p moment of the previous update direction.
         * @param maxIterations Maximum number of epochs (> 0).
         * @param maxError Early-stopping error threshold.
         * @param learningRate SGD step size.
         * @param X Input matrix (rows = patterns, cols = inputs).
         * @param Y Target output matrix (rows = patterns, cols = outputs).
         * @param moment Momentum coefficient in [0, 1).
         * @throw std::logic_error If called before initMLP.
         * @throw std::invalid_argument On empty/mismatched matrices or invalid hyperparameters.
         */
        void onlineMomentumBP(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, double moment = 0.9);

        /// Online Adam trainer that also evaluates calibration/validation metrics every n-th epoch.
        /**
         * @param Xtrain Calibration inputs (rows = patterns).
         * @param Ytrain Calibration targets (rows = patterns).
         * @param Xval Validation inputs.
         * @param Yval Validation targets.
         * @param maxIterations Maximum number of epochs.
         * @param maxError Early-stopping error threshold (on calibration).
         * @param learningRate Adam step size.
         * @param metricsAfterXEpochs Metric evaluation interval in epochs.
         * @return Vector with row-per-epoch metric matrices; index 0..13 =
         *         [MSE, RMSE, PI, NS, KGE, PBIAS, RSR] for calibration (0-6)
         *         and validation (7-13).
         */
        std::vector<Eigen::MatrixXd> onlineAdamEpochVal(
            const Eigen::MatrixXd &Xtrain,
            const Eigen::MatrixXd &Ytrain,
            const Eigen::MatrixXd &Xval,
            const Eigen::MatrixXd &Yval,
            int maxIterations,
            double maxError,
            double learningRate,
            int metricsAfterXEpochs,
            const EpochMetricContext *epochCtx = nullptr);

        /// Batch Adam trainer that also evaluates calibration/validation metrics every n-th epoch.
        /**
         * @param Xtrain Calibration inputs (rows = patterns).
         * @param Ytrain Calibration targets (rows = patterns).
         * @param Xval Validation inputs.
         * @param Yval Validation targets.
         * @param batchSize Number of patterns per batch.
         * @param maxIterations Maximum number of epochs.
         * @param maxError Early-stopping error threshold (on calibration).
         * @param learningRate Adam step size.
         * @param metricsAfterXEpochs Metric evaluation interval in epochs.
         * @return Same layout as onlineAdamEpochVal().
         */
        std::vector<Eigen::MatrixXd> batchAdamEpochVal(
            const Eigen::MatrixXd &Xtrain,
            const Eigen::MatrixXd &Ytrain,
            const Eigen::MatrixXd &Xval,
            const Eigen::MatrixXd &Yval,
            int batchSize,
            int maxIterations,
            double maxError,
            double learningRate,
            int metricsAfterXEpochs,
            const EpochMetricContext *epochCtx = nullptr);

        /// Online backpropagation trainer that also evaluates calibration/validation metrics every n-th epoch.
        /**
         * @param Xtrain Calibration inputs (rows = patterns).
         * @param Ytrain Calibration targets (rows = patterns).
         * @param Xval Validation inputs.
         * @param Yval Validation targets.
         * @param maxIterations Maximum number of epochs.
         * @param maxError Early-stopping error threshold (on calibration).
         * @param learningRate SGD step size.
         * @param metricsAfterXEpochs Metric evaluation interval in epochs.
         * @return Same layout as onlineAdamEpochVal().
         */
        std::vector<Eigen::MatrixXd> onlineBpEpochVal(
            const Eigen::MatrixXd &Xtrain,
            const Eigen::MatrixXd &Ytrain,
            const Eigen::MatrixXd &Xval,
            const Eigen::MatrixXd &Yval,
            int maxIterations,
            double maxError,
            double learningRate,
            int metricsAfterXEpochs,
            const EpochMetricContext *epochCtx = nullptr);

        /// Batch backpropagation trainer that also evaluates calibration/validation metrics every n-th epoch.
        /**
         * @param Xtrain Calibration inputs (rows = patterns).
         * @param Ytrain Calibration targets (rows = patterns).
         * @param Xval Validation inputs.
         * @param Yval Validation targets.
         * @param batchSize Number of patterns per batch.
         * @param maxIterations Maximum number of epochs.
         * @param maxError Early-stopping error threshold (on calibration).
         * @param learningRate SGD step size.
         * @param metricsAfterXEpochs Metric evaluation interval in epochs.
         * @return Same layout as onlineAdamEpochVal().
         */
        std::vector<Eigen::MatrixXd> batchBpEpochVal(
            const Eigen::MatrixXd &Xtrain,
            const Eigen::MatrixXd &Ytrain,
            const Eigen::MatrixXd &Xval,
            const Eigen::MatrixXd &Yval,
            int batchSize,
            int maxIterations,
            double maxError,
            double learningRate,
            int metricsAfterXEpochs,
            const EpochMetricContext *epochCtx = nullptr);

        /// Online penalized backpropagation trainer that also evaluates metrics every n-th epoch.
        /**
         * @param Xtrain Calibration inputs (rows = patterns).
         * @param Ytrain Calibration targets (rows = patterns).
         * @param Xval Validation inputs.
         * @param Yval Validation targets.
         * @param maxIterations Maximum number of epochs.
         * @param maxError Early-stopping error threshold (on calibration).
         * @param learningRate SGD step size.
         * @param metricsAfterXEpochs Metric evaluation interval in epochs.
         * @param lambda Weight penalization coefficient.
         * @return Same layout as onlineAdamEpochVal().
         */
        std::vector<Eigen::MatrixXd> onlinePenalizeBpEpochVal(
            const Eigen::MatrixXd &Xtrain,
            const Eigen::MatrixXd &Ytrain,
            const Eigen::MatrixXd &Xval,
            const Eigen::MatrixXd &Yval,
            int maxIterations,
            double maxError,
            double learningRate,
            int metricsAfterXEpochs,
            double lambda = 0.0001,
            const EpochMetricContext *epochCtx = nullptr);

        /// Online momentum backpropagation trainer that also evaluates metrics every n-th epoch.
        /**
         * @param Xtrain Calibration inputs (rows = patterns).
         * @param Ytrain Calibration targets (rows = patterns).
         * @param Xval Validation inputs.
         * @param Yval Validation targets.
         * @param maxIterations Maximum number of epochs.
         * @param maxError Early-stopping error threshold (on calibration).
         * @param learningRate SGD step size.
         * @param metricsAfterXEpochs Metric evaluation interval in epochs.
         * @param moment Momentum coefficient in [0, 1).
         * @return Same layout as onlineAdamEpochVal().
         */
        std::vector<Eigen::MatrixXd> onlineMomentumBpEpochVal(
            const Eigen::MatrixXd &Xtrain,
            const Eigen::MatrixXd &Ytrain,
            const Eigen::MatrixXd &Xval,
            const Eigen::MatrixXd &Yval,
            int maxIterations,
            double maxError,
            double learningRate,
            int metricsAfterXEpochs,
            double moment = 0.9,
            const EpochMetricContext *epochCtx = nullptr);

        /// Forward pass for one input vector; stores the result in the member `output`.
        /**
         * @param inputVec Input vector; must match the first layer input size.
         * @throw std::logic_error If called before initMLP.
         */
        void calcOneOutput(const Eigen::VectorXd& inputVec);
        /// Calculate outputs for all rows of the input matrix; stores them in `outputMat`.
        /**
         * @param inputMat Input matrix (rows = patterns, cols = inputs).
         * @throw std::logic_error If called before initMLP.
         */
        void calculateOutputs(const Eigen::MatrixXd& inputMat);
        Eigen::MatrixXd getOutputs() const;  //!< Getter for the output matrix filled by calculateOutputs()
        TrainingResult getResult() const {return result;}  //!< Getter for the last training run's result
        /// Sum the deltas of the first-layer neurons into a 1-element vector for preceding networks.
        Eigen::VectorXd getFirstLayerDeltaSum();
        /// Propagate the first layer's deltas back to the MLP inputs (W_0^T * delta_0, bias dropped).
        /**
         * Used to feed the gradient into a preceding network (e.g. an LSTM feeding this MLP).
         * @return Vector of length nNeurons[0] (one entry per MLP input, bias excluded).
         */
        Eigen::VectorXd getFirstLayerInputDelta();

        // Objective function configuration
        /// Set the objective (loss) function used by all training routines.
        /**
         * @param func Objective type (see ObjectiveFunctions.hpp).
         */
        void setObjectiveFunction(objective_func_type func);
        /// Set the objective function by name.
        /**
         * @param funcName Name such as "MSE", "LEVEL_SLOPE" (see ObjectiveFunctions.hpp).
         * @throw std::invalid_argument For an unrecognized name.
         */
        void setObjectiveFunction(const std::string& funcName);
        objective_func_type getObjectiveFunction() const;  //!< Getter for the current objective function
        void setObjectiveAlpha(double alpha);  //!< Setter for the alpha parameter of combined losses
        void setObjectiveAlpha2(double alpha2);  //!< Setter for the alpha2 parameter of the curvature term
        double getObjectiveAlpha() const;  //!< Getter for the alpha parameter
        double getObjectiveAlpha2() const;  //!< Getter for the alpha2 parameter

        // Horizon-vector objective support (for LSTM trainers)
        /// Compute the objective gradient w.r.t. the full prediction horizon and backpropagate it.
        /**
         * Runs a forward pass per horizon step, computes the gradient of the
         * objective on the whole horizon vector (slope/curvature/PI losses see
         * all steps at once), backpropagates each step's gradient component
         * (accumulating batch gradients), and returns the per-step first-layer
         * input deltas scaled by the horizon gradient, for feeding deltas back
         * into a preceding LSTM.
         * @param stepInputs Per-step input vectors (one per horizon step).
         * @param obsHorizon Observed horizon values (length = number of steps).
         * @return Matrix (rows = MLP inputs, cols = horizon steps) of input deltas.
         * @throw std::logic_error If called before initMLP.
         * @throw std::invalid_argument If stepInputs.size() != obsHorizon.size().
         */
        Eigen::MatrixXd horizonBackprop(const std::vector<Eigen::VectorXd> &stepInputs,
                                        const Eigen::VectorXd &obsHorizon);

    protected:

    private:
        std::vector<unsigned> nNeurons;  //!< The vector of number of neurons per layer
        size_t numLayers;  //!< Cache of nNeurons.size()
        Eigen::VectorXd Inps;  //!< The vector of inputs
        std::vector<activ_func_type> activFuncs;  //!< Vector of activation functions for each layer
        std::vector<weight_init_type> wInitTypes;   //!< Vector of weights initialization for each layer
        std::vector<Layer> layers_;  //!< Per-layer state (weights, outputs, deltas, gradients)
        Eigen::VectorXd output;  //!< The output vector of the MLP (result of the last forward pass)
        Eigen::MatrixXd outputMat; //!< The output matrix (one row per pattern) filled by calculateOutputs()
        Eigen::VectorXd weightsVectorMlp;  //!< The flattened weights vector of all layers
        Eigen::MatrixXd calibCrit;  //>! Matrix of all calibration criteria
        Eigen::MatrixXd validCrit;  //>! Matrix of all validation criteria
        TrainingResult result;  //!< Statistics of the last completed training run

        // Objective function settings for training
        objective_func_type objectiveFunc_ = objective_func_type::LEVEL_SLOPE;  //!< Loss used by all training routines
        double objectiveAlpha_ = 0.5;     //!< Alpha weighting parameter for combined losses
        double objectiveAlpha2_ = 0.33;   //!< Second alpha parameter for the curvature term
        double objectiveEpsilon_ = 1e-8;  //!< Numerical stabilizer used inside the objective gradients
};

#endif // MLP_H
