#ifndef LSTMLAYER_HPP
#define LSTMLAYER_HPP

#include "eigen-3.4/Eigen/Dense"
#include "ConfigIni.hpp"

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
    Eigen::VectorXd tanhVector(const Eigen::VectorXd& vec);     //!< Sigmoid derivative function for all elements in vector
    Eigen::VectorXd sigmoidDerivVector(const Eigen::VectorXd& vec);     //!< Tanh function for all elements in vector
    Eigen::VectorXd tanhDerivVector(const Eigen::VectorXd& vec);    //!< Tanh derivative function for all elements in vector

    void initLSTMLayer(const int numInputs,        
                    const int numCells,
                    const int numTimeSteps,
                    std::string initType = "RANDOM",
                    int rngSeed = 0,
                    double minVal = 0.0,
                    double maxVal = 1.0);   //!< Initialize LSTMLayer, its weights using specified technique

    void setInputTSSegment(const Eigen::MatrixXd& inputSegment);    //!< Set inputs for n time-steps
    void calculateTimeSteps();      //!< Calculate outputs for n time-steps

private:
    Eigen::MatrixXd W;  //!< The input weight matrix (all gates - by rows forget, input, candidate, output)
    Eigen::MatrixXd U;  //!< The hidden state (short memory) gate weight matrix (all gates - by rows forget, input, candidate, output)
    Eigen::VectorXd b;  //!< The bias vector (all gates...)
    Eigen::MatrixXd timeStepsInputs;  //!< The matrix of inputs for n time-steps (rows = time-steps)
    Eigen::MatrixXd activationsWhole;  //!< Computed activations all gates (by rows) all time steps (by columns)
    Eigen::MatrixXd forgetGate;     //!< Computed forget gate outputs all cells (rows) all time steps (columns)
    Eigen::MatrixXd inputGate;      //!< Computed input gate outputs all cells (rows) all time steps (columns)
    Eigen::MatrixXd candidateGate;  //!< Computed candidate gate outputs all cells (rows) all time steps (columns)
    Eigen::MatrixXd outputGate;     //!< Computed output gate outputs all cells (rows) all time steps (columns)
    Eigen::MatrixXd cellState;      //!< Computed cell state (long memory) all cells (rows) all time steps (columns)
    Eigen::MatrixXd output;         //!< Computed outputs (short memory) all cells (rows) all time steps (columns)
    Eigen::MatrixXd Wgradient;
    Eigen::MatrixXd Ugradient;
    Eigen::VectorXd bGradient;

};

#endif // LSTMLAYER_HPP