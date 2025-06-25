#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include "eigen-3.4/Eigen/Dense"

#include "Layer.hpp" 

using namespace std;

class MLP {

    public:
        MLP();  //!< The constructor
        ~MLP();  //!< The destructor     
        MLP(const MLP& other);  //!< The copy constructor
        MLP& operator=(const MLP& other);  //!< The assignment operator

        //!< Architecture (neurons per layer)
        std::vector<unsigned> getArchitecture();  //!< Getter for the architecture
        void setArchitecture(std::vector<unsigned>& architecture);  //!< Setter for the architecture
        void printArchitecture();  //!< Print the architecture

        // Activations for each layer
        std::vector<activ_func_type> getActivations();  //!< Getter for the activation function
        void setActivations(std::vector<activ_func_type>& funcs);  //!< Setter for the activation function
        void printActivations();  //!< Print the activation function

        //!< Number of layers
        size_t getNumLayers();  //!< Getter for the number of layers
        void setNumLayers(size_t layers);  //!< Setter for the number of layers

        //!< Number of inputs (neurons in first layer)
        unsigned getNumInputs();  //!< Getter for the number of inputs
        void setNumInputs(unsigned inputs);  //!< Setter for the number of inputs

        //!< Number of neurons at specific layer 
        unsigned getNumNeuronsInLayers(std::size_t index);  //!< Getter for the number of neurons at specific layer
        void setNumNeuronsInLayers(std::size_t index, unsigned count);  //!< Setter for the number of neurons at specific layer

        // Getter and Setter for inputs
        Eigen::VectorXd& getInps();  //!< Getter for the inputs
        void setInps(Eigen::VectorXd& inputs);   //!< Setter for the inputs

        bool validateInputSize();  //!< Validate the size of the inputs compared to nNeurons[0]

    protected:

    private:
        std::vector<unsigned> nNeurons;  //!< The vector of number of neurons per layer
        size_t numLayers;  //!< Cache of nNeurons.size()
        Eigen::VectorXd Inps;  //!< The vector of inputs  
        std::vector<activ_func_type> activFuncs;  //!< Vector of activation functions for each layer 

};

#endif // MLP_H
