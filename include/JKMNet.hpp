#ifndef JKMNet_HPP
#define JKMNet_HPP
#include <stdio.h>
#include <iostream>

#include "MLP.hpp"
#include "Layer.hpp"


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