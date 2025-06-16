#ifndef MLP_HPP
#define MLP_HPP

#include <vector>

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

        //!< Number of layers
        size_t getNumLayers();  //!< Getter for the number of layers
        void setNumLayers(size_t layers);  //!< Setter for the number of layers

        //!< Number of inputs (neurons in first layer)
        unsigned getNumInputs();  //!< Getter for the number of inputs
        void setNumInputs(unsigned inputs);  //!< Setter for the number of inputs

        //!< Number of neurons at specific layer 
        unsigned getNumNeuronsInLayers(std::size_t index);  //!< Getter for the number of neurons at specific layer
        void setNumNeuronsInLayers(std::size_t index, unsigned count);  //!< Setter for the number of neurons at specific layer

    protected:

    private:
        std::vector<unsigned> nNeurons;  //!< Number of neurons per layer
        size_t numLayers;  //!< Cache of nNeurons.size()

};

#endif // MLP_H
