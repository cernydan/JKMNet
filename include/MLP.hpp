#include <vector>

using namespace std;

class MLP {

    public:
    MLP();  //!< The constructor
    ~MLP();  //!< The destructor     
    // MLP(const MLP& other);  //!< The copy constructor
    // MLP& operator=(const MLP& other);  //!< The assignment operator

    // Getter for the architecture (returns number of neurons in each layer)
    std::vector<unsigned> getArchitecture() const;
    // Setter for the architecture (modifies number of neurons in each layer)
    void setArchitecture(const std::vector<unsigned>& architecture);
    // Print the architecture
    void printArchitecture() const;

    protected:
    private:
    std::vector<unsigned> nNeurons;

};