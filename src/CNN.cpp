#include "CNN.hpp"
#include <iostream>
#include <random>
#include <cmath>

using namespace std;

void CNN::setArchitecture(const std::vector<std::vector<int>>& newArc){
    architecture = newArc;
}

void CNN::initCNN(const Eigen::MatrixXd& input, int rngSeed){
    layers_.clear();
    layers_.reserve(architecture.size());

    layers_.emplace_back();
    layers_[0].init1DCNNLayer(architecture[0][1], 
                    architecture[0][0], 
                    input.rows(),
                    input.cols(),
                    architecture[0][2],
                    //PRIDELAT INIT ATD Z VEKTORU!
                    )
    layers_[0].setCurrentInput(input);
    layers_[0].calculateOutput1D();

    for(size_t i = 1; i < architecture.size(); i++){
        layers_.emplace_back();
        layers_[i].init1DCNNLayer(architecture[i][1], 
                        architecture[i][0], 
                        layers[i-1].getOutput().rows(),
                        layers[i-1].getOutput().cols(),
                        architecture[0][2],
                    //PRIDELAT INIT ATD Z VEKTORU!
                    )
        layers_[i].setCurrentInput(layers[i-1].getOutput());
        layers_[i].calculateOutput1D();
    }
}

void CNN::runCNN1D(const Eigen::MatrixXd& input){
    if (layers_.empty())
        throw std::logic_error("runCNN1D called before initCNN");

    // First layer
    layers_[0].setCurrentInput1D(input);
    layers_[0].calculateOutput1D();

    // Remaining layers
    for (size_t i = 1; i < layers_.size(); ++i) {
        layers_[i].setCurrentInput1D(layers_[i-1].getOutput1D());
        layers_[i].calculateOutput1D();
    }
    if(layers_[layers_.size()-1].getOutput1D().rows() != 1)
        throw std::logic_error("final output of runCNN1D doesn't have 1 row");

    output = layers_[layers_.size()-1].getOutput1D().row(0);
}

Eigen::VectorXd CNN::getOutput(){
    return output;
}