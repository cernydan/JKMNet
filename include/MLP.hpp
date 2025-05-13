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

        // Getter for the architecture (returns number of neurons in each layer)
        std::vector<unsigned> getArchitecture();
        // Setter for the architecture (modifies number of neurons in each layer)
        void setArchitecture(const std::vector<unsigned>& architecture);
        // Print the architecture
        void printArchitecture();


        //get, set numLayers
        // pokud mam arch a modifikuji nNeurons, tak zkontroluj, ze je dobre zadany vektor 

    protected:
    private:
        std::vector<unsigned> nNeurons;
        size_t numLayers;

};

#endif // MLP_H