#ifndef JKMNet_HPP
#define JKMNet_HPP

#include "MLP.hpp"
#include "Layer.hpp"
#include "Data.hpp"
#include "Metrics.hpp"

#include <stdio.h>
#include <iostream>



class JKMNet {

    public:
        JKMNet();  //!< The constructor
        JKMNet(const RunConfig& cfg, unsigned nthreads);
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

        void init_mlps(); //>! Initialization of MLPs vector

        void ensembleRun(MLP &mlp_);
        void ensembleRunMlpVector();

    protected:

    private:
        std::vector<MLP> mlps_;  //>! The vector of MLPs
        unsigned Nmlps;  //>! Total number of MLPs
        RunConfig cfg_;
        unsigned nthreads_;
        Data data_;

};

#endif // JKMNet_H