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

        std::vector<unsigned> getArchitecture();  //!< Getter for the architecture (returns number of neurons in each layer)  
        void setArchitecture(const std::vector<unsigned>& architecture);  //!< Setter for the architecture (modifies number of neurons in each layer)
        void printArchitecture();  //!< Print the architecture

        //get, set numLayers
        // pokud mam arch a modifikuji nNeurons, tak zkontroluj, ze je dobre zadany vektor 
        // UPDATE:
        std::vector<unsigned> getLayers();  //!< Getter for the layers 
        void setLayers(const std::vector<unsigned>& architecture);  //!< Setter for the layers

    protected:
    private:
        std::vector<unsigned> nNeurons;
        size_t numLayers;

};

#endif // MLP_H