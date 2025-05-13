#ifndef JKMNet_HPP
#define JKMNet_HPP

#include "MLP.hpp"
#include "Layer.hpp"
#include <stdio.h>
#include <iostream>

class JKMNet {

    public:
        JKMNet();  //!< The constructor
        ~JKMNet();  //!< The destructor 
        // virtual ~JKMNet();
        JKMNet(const JKMNet& other);  //!< The copy constructor
        JKMNet& operator=(const JKMNet& other);  //!< The assignment operator

    protected:
    private:

};

#endif // JKMNet_H