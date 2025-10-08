#ifndef JKMNet_HPP
#define JKMNet_HPP

#include "MLP.hpp"
#include "Layer.hpp"
#include "Data.hpp"
#include "Metrics.hpp"

#include <stdio.h>
#include <iostream>

struct TrainingResult {
    double finalLoss = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    bool converged = false;
};

class JKMNet {

    public:
        JKMNet();  //!< The constructor
        ~JKMNet();  //!< The destructor 
        // virtual ~JKMNet();
        JKMNet(const JKMNet& other);  //!< The copy constructor
        JKMNet& operator=(const JKMNet& other);  //!< The assignment operator

        //!< Train MLP with online Adam using calibrated matrices built from Data
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

        //!< Train an MLP with online Adam without splitting (already done before training)
        TrainingResult trainAdamOnline(
            MLP &mlp,
            const Eigen::MatrixXd &X,
            const Eigen::MatrixXd &Y,
            int maxIterations,
            double maxError,
            double learningRate,
            bool shuffle,
            unsigned rngSeed
        );

        //!< Train an MLP with batch Adam without splitting (already done before training)
        TrainingResult trainAdamBatch(
            MLP &mlp,
            const Eigen::MatrixXd &X,
            const Eigen::MatrixXd &Y,
            int batchSize,
            int maxIterations,
            double maxError,
            double learningRate,
            bool shuffle,
            unsigned rngSeed
        );

        Eigen::MatrixXd trainAdamOnlineEpochVal(
            MLP &mlp,
            const Eigen::MatrixXd &CalInp,
            const Eigen::MatrixXd &CalOut,
            const Eigen::MatrixXd &ValInp,
            const Eigen::MatrixXd &ValOut,
            int maxIterations,
            double maxError,
            double learningRate,
            bool shuffle,
            unsigned rngSeed);

        Eigen::MatrixXd trainAdamBatchEpochVal(
            MLP &mlp,
            const Eigen::MatrixXd &CalInp,
            const Eigen::MatrixXd &CalOut,
            const Eigen::MatrixXd &ValInp,
            const Eigen::MatrixXd &ValOut,
            int batchSize,
            int maxIterations,
            double maxError,
            double learningRate,
            bool shuffle,
            unsigned rngSeed);

        //!< K-fold validation (online Adam) 
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

        void setNmlps(unsigned nmlp);  //>! Setter for number of MLPs
        unsigned getNmlps();  //>! Getter for number of MLPs

        void init_mlps(MLP &mlp); //>! Initialization of MLPs vector

    protected:

    private:
        std::vector<MLP> mlps_;  //>! The vector of MLPs
        unsigned Nmlps;  //>! Total number of MLPs

};

#endif // JKMNet_H