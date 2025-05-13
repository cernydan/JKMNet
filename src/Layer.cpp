#include "Layer.hpp"

#include <iostream>

using namespace std;


/**
 * The constructor
 */
Layer::Layer(): weights(),
        inputs(),
        activations(),
        output() {

}

/**
 * The destructor
 */
Layer::~Layer(){

}


/**
 * The copy constructor
 */
Layer::Layer(const Layer& other): weights(),
        inputs(),
        activations(),
        output() {

    weights = other.weights;
    inputs = other.inputs;
    activations= other.activations;
    output= other.output;
   

}


/**
 * The assignment operator
 */
Layer& Layer::operator=(const Layer& other){
    if (this == &other) return *this;
  else {
    weights = other.weights;
    inputs = other.inputs;
    activations= other.activations;
    output= other.output;
  }
  return *this;

}