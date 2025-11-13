#ifndef CNN_HPP
#define CNN_HPP

#include "eigen-3.4/Eigen/Dense"
#include "CNNLayer.hpp"

class CNN
{
public:
    CNN() = default;  //!< The constructor
    ~CNN() = default;  //!< The destructor 
    CNN(const CNN&) = default;  //!< The copy constructor
    CNN& operator=(const CNN&) = default;   //!< The assignment operator

    void setArchitecture(const std::vector<std::vector<int>>& newArc);
    void initCNN(const Eigen::MatrixXd& input, int rngSeed);
    void runCNN1D(const Eigen::MatrixXd& input);
    Eigen::VectorXd getOutput();

protected:
private:
    std::vector<CNNLayer> layers_;
    std::vector<std::vector<int>> architecture;
    Eigen::VectorXd output;
};

#endif // CNN_HPP