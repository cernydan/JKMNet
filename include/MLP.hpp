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

        std::vector<unsigned> getArchitecture();  //!< Getter for the architecture
        void setArchitecture(std::vector<unsigned>& architecture);  //!< Setter for the architecture
        void printArchitecture();  //!< Print the architecture

        size_t getNumLayers();  //!< Getter for the number of layers
        void setNumLayers(size_t layers);  //!< Setter for the number of layers

    protected:

    private:
        std::vector<unsigned> nNeurons;
        size_t numLayers;

};

#endif // MLP_H