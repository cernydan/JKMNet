#ifndef ENSEMBLE_RUN_HPP
#define ENSEMBLE_RUN_HPP

#include "JKMNet.hpp"
#include "MLP.hpp"
#include "Data.hpp"
#include "Metrics.hpp"
#include "ConfigIni.hpp"

class EnsembleRunner {
public:
    EnsembleRunner(const RunConfig& cfg, unsigned nthreads);
    void run();

private:
    RunConfig cfg_;
    unsigned nthreads_;
    JKMNet net_;
    Data data_;
    MLP mlp_;
};

#endif // ENSEMBLE_RUN_HPP