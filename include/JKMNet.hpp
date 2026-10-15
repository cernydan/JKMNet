#ifndef JKMNet_HPP
#define JKMNet_HPP

/**
 * @file JKMNet.hpp
 * @brief Declares the JKMNet class — the top-level orchestration class driving data loading, MLP training, K-fold validation, and parallel ensemble runs from a RunConfig.
 */

#include "MLP.hpp"
#include "Layer.hpp"
#include "Data.hpp"
#include "Metrics.hpp"
#include "LSTMLayer.hpp"

#include <stdio.h>
#include <iostream>



/// Top-level orchestrator of the library: config-driven training, evaluation, and ensemble workflows.
/**
 * Holds the RunConfig and the OpenMP thread count, owns the ensemble of MLPs
 * and a Data object, and provides the high-level workflows: single-MLP Adam
 * training on calibration matrices, K-fold cross-validation, a config-driven
 * parallel ensemble run, prediction from saved weights, and LSTM-MLP hybrid
 * ensemble experiments.
 */
class JKMNet {

    public:
        JKMNet();  //!< The constructor
        /// Constructor storing the run configuration and the OpenMP thread count used by the parallel ensemble loops.
        /**
         * @param cfg Run configuration (paths, architecture, hyperparameters).
         * @param nthreads Number of OpenMP threads for the ensemble loops.
         */
        JKMNet(const RunConfig& cfg, unsigned nthreads);
        ~JKMNet();  //!< The destructor
        // virtual ~JKMNet();
        JKMNet(const JKMNet& other);  //!< The copy constructor
        JKMNet& operator=(const JKMNet& other);  //!< The assignment operator

        //!< Train MLP with online Adam using calibrated matrices built from Data
        /// Train a single MLP with online (per-pattern) Adam on calibration matrices built from a Data object.
        /**
         * Builds the calibration matrices via Data::makeCalibMatsSplit,
         * optionally shuffles pattern rows, configures the MLP (architecture,
         * activations, weight init, objective from cfg_), initializes weights,
         * trains via MLP::onlineAdam, and fills the result with the final
         * training MSE.
         * @param mlp MLP to configure and train.
         * @param data Data object providing the calibration matrices.
         * @param mlpArchitecture Neuron counts per layer; the first entry is the input size, the last is the output size.
         * @param numbersOfPastVarsValues Number of past values per input variable (window construction for Data).
         * @param activationType Activation function used for every layer.
         * @param weightsInitType Weight initialization type used for every layer.
         * @param maxIterations Maximum number of epochs (> 0).
         * @param maxError Early-stopping error threshold.
         * @param learningRate Adam step size.
         * @param shuffle Whether to shuffle pattern rows before training.
         * @param rngSeed Seed for weight initialization.
         * @return TrainingResult with the final training MSE and the convergence flag.
         * @throw std::invalid_argument On empty architecture/past-values or non-positive maxIterations.
         * @see activ_func_type, weight_init_type
         */
        TrainingResult trainAdamOnlineSplit(
            MLP &mlp,
            Data &data,
            const std::vector<unsigned> &mlpArchitecture,
            const std::vector<int> &numbersOfPastVarsValues,
            activ_func_type activationType,
            weight_init_type weightsInitType,
            int maxIterations,
            double maxError,
            double learningRate,
            bool shuffle = true,
            unsigned rngSeed = 42
        );

        //!< Train MLP with batch Adam using calibrated matrices built from Data
        /// Train a single MLP with mini-batch Adam on calibration matrices built from a Data object.
        /**
         * Same orchestration as trainAdamOnlineSplit, except training runs
         * through MLP::batchAdam with the given batch size.
         * @param mlp MLP to configure and train.
         * @param data Data object providing the calibration matrices.
         * @param mlpArchitecture Neuron counts per layer; the first entry is the input size, the last is the output size.
         * @param numbersOfPastVarsValues Number of past values per input variable (window construction for Data).
         * @param activationType Activation function used for every layer.
         * @param weightsInitType Weight initialization type used for every layer.
         * @param batchSize Number of patterns per mini-batch (> 0).
         * @param maxIterations Maximum number of epochs (> 0).
         * @param maxError Early-stopping error threshold.
         * @param learningRate Adam step size.
         * @param shuffle Whether to shuffle pattern rows before training.
         * @param rngSeed Seed for weight initialization.
         * @return TrainingResult with the final training MSE and the convergence flag.
         * @throw std::invalid_argument On empty architecture/past-values or non-positive maxIterations/batchSize.
         * @see activ_func_type, weight_init_type
         */
        TrainingResult trainAdamBatchSplit(
            MLP &mlp,
            Data &data,
            const std::vector<unsigned> &mlpArchitecture,
            const std::vector<int> &numbersOfPastVarsValues,
            activ_func_type activationType,
            weight_init_type weightsInitType,
            int batchSize,
            int maxIterations,
            double maxError,
            double learningRate,
            bool shuffle = true,
            unsigned rngSeed = 42
        );

        //!< K-fold validation (online Adam)
        /// K-fold cross-validation of an MLP trained with online Adam.
        /**
         * For each fold the train/validation matrices are rebuilt by
         * Data::makeKFoldMats, the MLP is freshly initialized and trained
         * runsPerFold times from the same seed, and the validation MSE
         * (computed in the original data units after inverse transformation)
         * is averaged across runs; the mean MSE of every fold is printed.
         * @param data Data object used to build the fold matrices.
         * @param mlpArchitecture Neuron counts per layer; the first entry is the input size, the last is the output size.
         * @param numbersOfPastVarsValues Number of past values per input variable (window construction for Data).
         * @param activationType Activation function used for every layer.
         * @param weightsInitType Weight initialization type used for every layer.
         * @param kFolds Number of folds.
         * @param shuffle Whether to shuffle patterns inside the fold construction.
         * @param largerPieceCalib When true, the calibration set takes the larger fraction of the fold boundaries.
         * @param seed Seed for weight initialization and fold shuffling.
         * @param maxIterations Maximum number of epochs per run (> 0).
         * @param maxError Early-stopping error threshold.
         * @param learningRate Adam step size.
         * @param runsPerFold Number of independent training runs averaged per fold.
         * @throw std::invalid_argument On empty architecture/past-values or non-positive maxIterations.
         * @see activ_func_type, weight_init_type
         */
        void KFold(
            Data &data,
            const std::vector<unsigned> &mlpArchitecture,
            const std::vector<int> &numbersOfPastVarsValues,
            activ_func_type activationType,
            weight_init_type weightsInitType,
            int kFolds,
            bool shuffle,
            bool largerPieceCalib,
            unsigned seed,
            int maxIterations,
            double maxError,
            double learningRate,
            int runsPerFold
        );

        void setNmlps(unsigned nmlp);  //!< Setter for number of MLPs in the ensemble
        unsigned getNmlps();  //!< Getter for number of MLPs in the ensemble

        /// Create Nmlps MLPs configured from cfg_ and initialize their weights in parallel (OpenMP).
        /**
         * A prototype MLP is built from the config (architecture, activations,
         * weight init, objective with LEVEL_SLOPE fallback) and copied into
         * each slot of mlps_; the input size is the total number of lag values
         * across all configured input variables.
         */
        void init_mlps();
        /// Same as init_mlps, but the MLP input size is the LSTM output size (MLPs act as readout layers on top of an LSTM).
        /**
         * @param lstmOutputSize Number of values produced by the preceding LSTM (= number of LSTM cells).
         */
        void init_mlpsForLSTM(unsigned lstmOutputSize);

        //void ensembleRun(MLP &mlp_);
        /// Main config-driven MLP ensemble workflow: load and preprocess data, then train and evaluate Nmlps independent MLPs in parallel.
        /**
         * Loads the CSV (with optional station-id filter), builds the
         * calibration/validation matrices, fits transformations on the
         * calibration part, runs per-run training with the trainer selected by
         * cfg_.trainer (saving per-epoch metric files for the *_EPOCH
         * variants), and writes predictions, metrics, and initial/final
         * weights per run.
         */
        void ensembleRunMlpVector();

        /// Reconstruct an MLP from the config, load previously saved weights, and predict on the validation split.
        /**
         * @param weightsPath Path to the per-layer weights CSV file.
         * @throw std::runtime_error If the weights file cannot be loaded.
         */
        void predictFromSavedWeights(const std::string &weightsPath);

        /// Early LSTM-MLP hybrid ensemble: one LSTMLayer consumes past observations and an output MLP maps its forward output to the target horizon.
        /**
         * Training alternates per sample: the LSTM runs its time steps, the
         * MLP computes a batch gradient from the LSTM output, the gradient is
         * handed back to the LSTM, and both parts are Adam-updated; the LSTM
         * memory is reset per sample.
         */
        void ensembleLstmFirstTest();
        /// Three-LSTM encoder-decoder style ensemble for direct multi-horizon forecasting.
        /**
         * A past-observation LSTM and a future-forcing LSTM feed a third
         * "together" LSTM over the combined time axis; the output MLP predicts
         * every horizon step, trained through MLP::horizonBackprop so
         * horizon-wise losses see all future steps at once.
         */
        void ensembleLstmPastFutureTest();

    protected:

    private:
        std::vector<MLP> mlps_;  //!< The vector of MLPs
        unsigned Nmlps;  //!< Total number of MLPs
        RunConfig cfg_;  //!< The run configuration (paths, architecture, hyperparameters)
        unsigned nthreads_;  //!< Number of OpenMP threads for the parallel ensemble loops
        Data data_;  //!< Data object used for loading, transforming, and splitting the dataset

};

#endif // JKMNet_H
