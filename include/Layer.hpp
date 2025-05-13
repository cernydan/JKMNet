#ifndef LAYER_HPP
#define LAYER_HPP

#include "Eigen/Dense"

enum class activ_func_type{RELU, SIGMOID};
const activ_func_type all_activ_func[]{activ_func_type::RELU, activ_func_type::SIGMOID};
const unsigned numActivFunc = 2;

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

    protected:

    private:
        Eigen::MatrixXd weights;  //!< The weight matrix for the layer
        Eigen::VectorXd inputs;  //!< The input vector to the layer
        Eigen::VectorXd activations;  //!< The activations after applying the activation function
        Eigen::VectorXd output;  //!< The output vector of the layer
};

#endif // LAYER_H