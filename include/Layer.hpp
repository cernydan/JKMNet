#ifndef LAYER_HPP
#define LAYER_HPP

#include "eigen-3.4/Eigen/Dense"

enum class weight_init_type
{
    RANDOM,
    LHS,
    LHS2
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
    WAVE
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
    activ_func_type::WAVE
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

    // void initLayer(unsigned numInputs, unsigned numNeurons);  //!< Initialize the layer with the specified number of neurons and input size
    void initLayer(unsigned numInputs, unsigned numNeurons, weight_init_type initType = weight_init_type::RANDOM, double minVal = -1.0, double maxVal = 1.0); //!< Initialize the layer with chosen weight initialization technique
    void initWeights(unsigned numNeurons, unsigned numInputs, weight_init_type initType, double minVal, double maxVal);                                       //!< Initialize weights using specified technique

    Eigen::VectorXd getInputs();  //!< Getter for inputs
    void setInputs(const Eigen::VectorXd &newInputs);  //!< Setter for inputs

    Eigen::MatrixXd getGradient();  //!< Getter for gradient
    void setGradient(const Eigen::MatrixXd& grad);  //!< Getter for gradient 

    Eigen::MatrixXd getWeights();  //!< Getter for weights
    void setWeights(const Eigen::MatrixXd &newWeights);  //!< Setter for weights
    void updateWeights(double learningRate);  //!< Apply a gradient calculation: W = W – η·(∂E/∂W)

    Eigen::VectorXd calculateWeightedSum();  //!< Calculate the weighted sum (linear combination) - calculate activations
    Eigen::VectorXd setActivationFunction(const Eigen::VectorXd &weightedSum, activ_func_type activFuncType); //!< Apply activation function to weighted sum
    Eigen::VectorXd calculateLayerOutput(activ_func_type activFuncType);                                        //!< Calculate complete layer output

    Eigen::VectorXd getOutput(); //!< Getter for output

    static std::string activationName(activ_func_type f);  //!< Mapping activ_func_type from enum to string
    static std::string wInitTypeName(weight_init_type w);  //!< Mapping weight_init_type from enum to string

protected:
private:
    Eigen::MatrixXd weights;  //!< The weight matrix for the layer
    Eigen::VectorXd inputs;  //!< The input vector to the layer
    Eigen::VectorXd output;  //!< The output vector of the layer
    // Eigen::VectorXd activations;  //!< The activation vector of the layer
    // Eigen::VectorXd bias;  //!< The bias vector

    activ_func_type activ_func = activ_func_type::RELU;  //!< The type of activation function, where default is RELU

    Eigen::MatrixXd weightGrad;  //!< The backpropagation gradient matrix for the layer (∂E/∂W)
};

#endif // LAYER_HPP
