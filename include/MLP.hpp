#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include "eigen-3.4/Eigen/Dense"

#include "Layer.hpp" 

using namespace std;

class MLP {

    public:
        //MLP();  //!< The constructor
        //~MLP();  //!< The destructor     
        //MLP(const MLP& other);  //!< The copy constructor
        //MLP& operator=(const MLP& other);  //!< The assignment operator

        MLP() = default;  //!< The constructor
        ~MLP() = default;  //!< The destructor 
        MLP(const MLP&) = default;  //!< The copy constructor
        MLP& operator=(const MLP&) = default;   //!< The assignment operator

        //!< Architecture (neurons per layer)
        std::vector<unsigned> getArchitecture();  //!< Getter for the architecture
        void setArchitecture(std::vector<unsigned>& architecture);  //!< Setter for the architecture
        void printArchitecture();  //!< Print the architecture

        // Activations for each layer
        std::vector<activ_func_type> getActivations();  //!< Getter for the activation function
        void setActivations(std::vector<activ_func_type>& funcs);  //!< Setter for the activation function
        void printActivations();  //!< Print the activation function

         // Weight‐init types per layer 
        std::vector<weight_init_type> getWInitType();  //!< Getter for the weight initialization type
        void setWInitType(std::vector<weight_init_type>& wInits);  //!< Setter for the weight initialization type
        void printWInitType();  //!< Print the weight initialization type

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

        // Getter and Setter for weigths
        Eigen::MatrixXd getWeights(size_t layerIndex);  //!< Getter for weights
        void setWeights(size_t layerIndex, const Eigen::MatrixXd& W);  //!< Setter for weights

        bool validateInputSize();  //!< Validate the size of the inputs compared to nNeurons[0]

        Eigen::VectorXd initMLP(const Eigen::VectorXd& input);  //!< Forward pass through all layers     
        Eigen::VectorXd runMLP(const Eigen::VectorXd& input);  //!< Forward pass reusing existing weights
        bool compareInitAndRun(const Eigen::VectorXd& input, double tol = 1e-6) const;  //!< Compare if 'initMLP' and 'runMLP' produce the same output
        bool testRepeatable(const Eigen::VectorXd& input, int repeats = 10, double tol = 1e-8) const; //!< Repeatability check for 'runMLP'

    protected:

    private:
        std::vector<unsigned> nNeurons;  //!< The vector of number of neurons per layer
        size_t numLayers;  //!< Cache of nNeurons.size()
        Eigen::VectorXd Inps;  //!< The vector of inputs  
        std::vector<activ_func_type> activFuncs;  //!< Vector of activation functions for each layer 
        std::vector<weight_init_type> wInitTypes;   //!< Vector of weights initialization for each layer
        std::vector<Layer> layers_;  //!< Private member of the class Layer to store each layer’s state

};

#endif // MLP_H
