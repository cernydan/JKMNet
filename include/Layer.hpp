#ifndef LAYER_HPP
#define LAYER_HPP

#include "Eigen/Dense"
#include <cmath> 

enum class activ_func_type{RELU, SIGMOID, LINEAR, TANH};
const activ_func_type all_activ_func[]{activ_func_type::RELU, activ_func_type::SIGMOID, activ_func_type::LINEAR, activ_func_type::TANH};
const unsigned numActivFunc = 4;

class Layer {
    public:
        Layer();  //!< The constructor
        ~Layer();  //!< The destructor     
        Layer(const Layer& other);  //!< The copy constructor
        Layer& operator=(const Layer& other);  //!< The assignment operator

        // void getActivations
        // getOutput - argumentem je typ act func
        // initLayer - definice velikosti objektu, tj, pocet radku ve w, ...
        // getInputs - minuele vystupy jsou vstupem do dalsiho
        
        void initLayer(unsigned numInputs, unsigned numNeurons);  //!< Initialize the layer with the specified number of neurons and input size
        Eigen::VectorXd getInputs();  //!< Get the input vector to the layer
        Eigen::VectorXd calculateActivation(activ_func_type activFuncType);  //!< Calculate activations based on the activation function type
        Eigen::VectorXd getOutput();  //!< Get the output of the layer after applying the activation function
        
        // put the following to 'private' after creating getter and setter
        Eigen::MatrixXd weights;  //!< The weight matrix for the layer
        Eigen::VectorXd inputs;  //!< The input vector to the layer
        Eigen::VectorXd activations;  //!< The activations
        Eigen::VectorXd output;  //!< The output vector of the layer
        Eigen::VectorXd bias;  //!< The bias vector

    protected:

    private:
        

        
};

#endif // LAYER_H