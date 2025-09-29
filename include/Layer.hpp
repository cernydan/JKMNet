#ifndef LAYER_HPP
#define LAYER_HPP

#include "eigen-3.4/Eigen/Dense"

enum class weight_init_type
{
    RANDOM,
    LHS,
    LHS2,
    HE
}; //!< All weight initialization techniques

enum class activ_func_type
{
    RELU,
    SIGMOID,
    LINEAR,
    TANH,
    GAUSSIAN,
    IABS,
    LOGLOG,
    CLOGLOG,
    CLOGLOGM,
    ROOTSIG,
    LOGSIG,
    SECH,
    WAVE,
    LEAKYRELU
}; //!< All activation functions

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

class Layer
{
public:
    Layer();                              //!< The constructor
    ~Layer();                             //!< The destructor
    Layer(const Layer &other);            //!< The copy constructor
    Layer &operator=(const Layer &other); //!< The assignment operator
                                          //!< The move copy constructor
                                          //!< The move assignment operator

    //!< Initialize a fully-connected layer 
    void initLayer(unsigned numFeatures, 
        unsigned numNeurons,
        weight_init_type initType = weight_init_type::RANDOM, 
        activ_func_type  activFunc = activ_func_type::RELU,
        int rngSeed = 0,
        double minVal = 0.0,
        double maxVal = 1.0);   
    void initWeights(unsigned numNeurons, 
        unsigned numInputs, 
        weight_init_type initType, 
        double minVal, 
        double maxVal, 
        int rngSeed);   //!< Initialize weights using specified technique

    Eigen::VectorXd getInputs();  //!< Getter for inputs
    void setInputs(const Eigen::VectorXd &newInputs);  //!< Setter for inputs

    Eigen::MatrixXd getGradient();  //!< Getter for gradient
    void setGradient(const Eigen::MatrixXd& grad);  //!< Getter for gradient
    void calculateOnlineGradient(); //!< Calculation of the gradient matrix for online backpropagation
    void calculateBatchGradient(); //!< Calculation of the gradient matrix for batch backpropagation

    Eigen::MatrixXd getWeights() const;  //!< Getter for weights
    void setWeights(const Eigen::MatrixXd &newWeights);  //!< Setter for weights
    void updateWeights(double learningRate);  //!< Apply a gradient calculation: W = W – η·(∂E/∂W)
    void updateAdam(double learningRate, int iterationNum, double beta1, double beta2, double epsi); //!<  Apply a gradient calculation using ADAM algorithm

    Eigen::VectorXd calculateWeightedSum();  //!< Calculate the weighted sum (linear combination) - calculate activations
    Eigen::VectorXd setActivationFunction(const Eigen::VectorXd &weightedSum, activ_func_type activFuncType); //!< Apply activation function to weighted sum
    Eigen::VectorXd setActivFunDeriv(const Eigen::VectorXd &weightedSum, activ_func_type activFuncType); //!< Apply derivative of activation function to weighted sum
    Eigen::VectorXd calculateLayerOutput(activ_func_type activFuncType);                                        //!< Calculate complete layer output
    void calculateOutput(activ_func_type activFuncType);    //!< Calculate layer output
    void calculateDeltas(const Eigen::MatrixXd &nextWeights, const Eigen::VectorXd &nextDeltas, activ_func_type activFuncType);  //!< Calculate layer output and deltas for BP

    Eigen::VectorXd getOutput(); //!< Getter for output
    Eigen::VectorXd getDeltas(); //!< Getter for deltas
    void setDeltas(const Eigen::VectorXd &newDeltas); //!< Setter for deltas

    static std::string activationName(activ_func_type f);  //!< Mapping activ_func_type from enum to string
    static std::string wInitTypeName(weight_init_type w);  //!< Mapping weight_init_type from enum to string

    Eigen::VectorXd getWeightsVector(); //!< Getter for weights vector
    void weightsToVector(); //!< Arrange weight matrix into vector

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
};

#endif // LAYER_HPP
