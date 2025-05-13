#ifndef LAYER_HPP
#define LAYER_HPP

#include "Eigen/Dense"

//enum act func

class Layer {
    public:
        Layer();  //!< The constructor
        ~Layer();  //!< The destructor     
        Layer(const Layer& other);  //!< The copy constructor
        Layer& operator=(const Layer& other);  //!< The assignment operator

        // void getActivations
        // getOutput - argumentem je typ act func

    protected:

    private:
        Eigen::MatrixXd weights;
        Eigen::VectorXd inputs;
        Eigen::VectorXd activations;
        Eigen::VectorXd output;
};

#endif // LAYER_H