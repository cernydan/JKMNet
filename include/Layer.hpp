#ifndef LAYER_HPP
#define LAYER_HPP

#include "eigen-3.4/Eigen/Dense"
#include <cmath> 

enum class activ_func_type{RELU, SIGMOID, LINEAR, TANH};
const activ_func_type all_activ_func[]{activ_func_type::RELU, activ_func_type::SIGMOID, activ_func_type::LINEAR, activ_func_type::TANH};
const unsigned numActivFunc = std::size(all_activ_func);   //!< Total number of activation functions 

class Layer {
    public:
        Layer();  //!< The constructor
        ~Layer();  //!< The destructor     
        Layer(const Layer& other);  //!< The copy constructor
        Layer& operator=(const Layer& other);  //!< The assignment operator
            //!< The move copy constructor
            //!< The move assignment operator

        void initLayer(unsigned numInputs, unsigned numNeurons);  //!< Initialize the layer with the specified number of neurons and input size    
        
        Eigen::VectorXd getInputs();  //!< Getter for inputs
        void setInputs(const Eigen::VectorXd& newInputs);  //!< Setter for inputs

        Eigen::MatrixXd getWeights();  //!< Getter for weights
        void setWeights(const Eigen::MatrixXd& newWeights);  //!< Setter for weights
                
        Eigen::VectorXd calculateWeightedSum(); //!< Calculate the weighted sum (linear combination)
        Eigen::VectorXd applyActivationFunction(const Eigen::VectorXd& weightedSum, activ_func_type activFuncType); //!< Apply activation function to weighted sum
        Eigen::VectorXd calculateLayerOutput(activ_func_type activFuncType); //!< Calculate complete layer output

        Eigen::VectorXd getOutput();  //!< Getter for output       

    protected:

    private:
        Eigen::MatrixXd weights;  //!< The weight matrix for the layer
        Eigen::VectorXd inputs;  //!< The input vector to the layer
        Eigen::VectorXd output;  //!< The output vector of the layer
        //Eigen::VectorXd activations;  //!< The activation vector of the layer    
        //Eigen::VectorXd bias;  //!< The bias vector
};

#endif // LAYER_H