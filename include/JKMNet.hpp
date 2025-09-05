#ifndef JKMNet_HPP
#define JKMNet_HPP

#include "MLP.hpp"
#include "Layer.hpp"
#include "Data.hpp"

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

    protected:

    private:

};

#endif // JKMNet_H