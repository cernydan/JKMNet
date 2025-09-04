// ********* 27. 8. 2025 *********
// **DONE**: Create class for metrics
// TODO: Add more metrics (PI, NS, KGE, ...)
// **DONE**: Deal with NAs in the dataset
// TODO: Skip windows (horizon) that include any NA in the data (not only the exact timestamp) (?)
// **DONE**: Split calibration matrix into one for inputs and one for targets
// TODO: Preprocessing of calibration matrix in R (?)
// **DONE**: Split data for calibration and validation set (+ testing set?) (chronologically or randomly as a new method in MLP) 
// **DONE**: Change 'Testing ADAM' section in 'main.cpp' as a new method 
// TODO: Prepare more scenarios for running
// TODO: Prepare vector of weights from matrix of weights (for global optimization)
// *******************************

#include <ctime>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <limits>

#include "JKMNet.hpp"
#include "MLP.hpp"
#include "Layer.hpp"
#include "Data.hpp"
#include "Metrics.hpp"

using namespace std;

void testAdamOnlineSplitD(MLP& adamlp,
                          Data& dataA, 
                          std::vector<unsigned> mlpArchitecture,
                          const std::vector<int>& numbersOfPastVarsValues,
                          activ_func_type activationType,
                          weight_init_type weightsInitType,
                          int maxIterations,
                          double maxError,
                          double learningRate){           
  dataA.makeCalibMatsSplit(numbersOfPastVarsValues,mlpArchitecture.back());
  std::vector<int>permutVector = dataA.permutationVector(dataA.getCalibInpsMat().rows());
  dataA.setCalibInpsMat(dataA.shuffleMatrix(dataA.getCalibInpsMat(),permutVector));
  dataA.setCalibOutsMat(dataA.shuffleMatrix(dataA.getCalibOutsMat(),permutVector));
  adamlp.setArchitecture(mlpArchitecture);
  std::vector<activ_func_type> activations;
  std::vector<weight_init_type> weightInits;
  for(size_t i = 0; i < mlpArchitecture.size(); i++){
    activations.push_back(activationType);
    weightInits.push_back(weightsInitType);
  }
  adamlp.setActivations(activations);
  adamlp.setWInitType(weightInits);
  adamlp.initMLP(dataA.getCalibInpsMat().row(0));
  adamlp.onlineAdam(maxIterations, maxError, learningRate, dataA.getCalibInpsMat(), dataA.getCalibOutsMat());
}

int main() {

  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "-- Testing the functionalities of JKMNet --" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  JKMNet NNet;
  MLP mlp;
  Layer layer;

  //!! ------------------------------------------------------------
  //!! ARCHITECTURE
  //!! ------------------------------------------------------------

  std::cout << "------------" << std::endl;
  std::cout << "--  MLP --" << std::endl;
  std::cout << "------------" << std::endl;

  // Print the current architecture (using the getter)
  std::cout << "Getter Architecture: ";
  std::vector<unsigned> architecture = mlp.getArchitecture();
  // Output the architecture
  for (size_t i = 0; i < architecture.size(); ++i) {
    std::cout << architecture[i];
    if (i != architecture.size() - 1) {
        std::cout << " -> ";
    }
  } 
  std::cout << std::endl;

  // Print the default architecture
  std::cout << "Default Architecture: ";
  mlp.printArchitecture();

  // Set a new architecture
  std::vector<unsigned> newArchitecture1 = {50, 32, 16, 3}; // Example: 50 inputs, 32/16 hidden neurons, 5 outputs
  mlp.setArchitecture(newArchitecture1);
  std::cout << "Architecture 1: ";
  // Print the updated architecture
  mlp.printArchitecture();

  // Set a new architecture
  std::vector<unsigned> newArchitecture2 = {10, 20, 30, 6, 7}; // Example: 100 inputs, 20/10/6 hidden neurons, 5 outputs
  mlp.setArchitecture(newArchitecture2);
  std::cout << "Architecture 2: ";
  // Print the updated architecture
  mlp.printArchitecture();

  // Get the number of layers
  size_t numLayers = mlp.getNumLayers();
  std::cout << "Number of layers: " << numLayers << std::endl;

  // Get the number of inputs
  unsigned numInputs = mlp.getNumInputs();
  std::cout << "Number of inputs: " << numInputs << std::endl;

  // Set the number of layers
  std::cout << "Number of layers before setting: " << mlp.getNumLayers() << std::endl;
  mlp.setNumLayers(999);
  std::cout << "Number of layers after setting: " << mlp.getNumLayers() << std::endl;

  // Set the number of inputs
  mlp.setNumInputs(100);
  std::cout << "Number of inputs after setting: " << mlp.getNumInputs() << std::endl;

  // Print the updated architecture
  std::cout << "Architecture after setting inputs: ";
  mlp.printArchitecture();

  // Get the number of neurons at specific layer
  unsigned h0 = mlp.getNumNeuronsInLayers(0); 
  unsigned h1 = mlp.getNumNeuronsInLayers(1);   
  unsigned h2 = mlp.getNumNeuronsInLayers(2);   
  std::cout << "Number of neurons in the input layer: " << h0 << std::endl;
  std::cout << "Number of neurons in the 1st hidden layer: " << h1 << std::endl;
  std::cout << "Number of neurons in the 2nd hidden layer: " << h2 << std::endl;

  // Set the number of neurons at specific layer
  mlp.setNumNeuronsInLayers(2, 90);  // 2nd hidden layer change to 90
  std::cout << "Change number of neurons at 2nd hidden layer to: " << mlp.getNumNeuronsInLayers(2) << std::endl;

  // Print the updated architecture
  std::cout << "Architecture after changing neurons at 2nd hidden layer: ";
  mlp.printArchitecture();
  
  std::cout << "------------" << std::endl;
  std::cout << "Testing real inputs " << std::endl;
  std::cout << "------------" << std::endl;

  // Define new architecture
  std::vector<unsigned> MyArchitecture = {8, 6, 2};
  mlp.setArchitecture(MyArchitecture);

  // Define activation functions for each layer
  std::vector<activ_func_type> funcs = {
        activ_func_type::RELU,    // input layer
        activ_func_type::TANH,    // 1st hidden layer
        //activ_func_type::SIGMOID,  // 2nd hidden layer
        activ_func_type::SIGMOID  // output layer
  };
  mlp.setActivations(funcs);

  // Define weight initialization for each layer
  std::vector<weight_init_type> wInits = {
        weight_init_type::RANDOM,
        weight_init_type::LHS,
        //weight_init_type::LHS,
        weight_init_type::LHS2
  };
  mlp.setWInitType(wInits);

  // Print the updated architecture, activation functions and weight initialization types
  std::cout << "Real Architecture: ";
  mlp.printArchitecture();
  mlp.printActivations();
  mlp.printWInitType();

  // Create random input vector of given length 
  std::srand(static_cast<unsigned>(std::time(nullptr)));  // seed using the current time; comment if debugging
  Eigen::VectorXd MyInps = Eigen::VectorXd::Random(8);
  // Eigen::VectorXd MyInps = Eigen::VectorXd::Random(10);

  // Call the setter to load inputs (bias will be appended automatically)
  mlp.setInps(MyInps);

  // Retrieve the internal Inps (real inputs + 1 bias)
  const Eigen::VectorXd& inpsWithBias = mlp.getInps();

  // Validate input size
  if (!mlp.validateInputSize()) {
          std::cerr << "[Error]: Input vector size does not match architecture!\n";
          return 1;
  }
  std::cout << "[Ok]: Input size validated.\n";

  // Print input size and values
  std::cout << "Inps size (with bias): " << inpsWithBias.size() << "\n";
  std::cout << "Inps values:\n " << inpsWithBias.transpose() << std::endl;

  // Forward pass - initialize and run
  Eigen::VectorXd MyInitOut = mlp.initMLP(MyInps);
  Eigen::VectorXd MyRunOut = mlp.runMLP(MyInps);
  
  // Compare 'initMLP' and 'runMLP' outputs
  if (!mlp.compareInitAndRun(MyInps)) {
    std::cerr << "[Error]: initMLP vs runMLP outputs disagree \n";
  } else {
    std::cout << "[Ok]: initMLP and runMLP outputs match \n";
  }

  // Print output size and values
  std::cout << "Init output size: " << MyInitOut.size() << "\n";
  std::cout << "Run output size: " << MyRunOut.size() << "\n";
  std::cout << "Init output values:\n " << MyInitOut.transpose() << std::endl;
  std::cout << "Run output values:\n " << MyRunOut.transpose() << std::endl;

  // Check if 'runMLP' produces the same results over several times
  if (mlp.testRepeatable(MyInps, 20, 1e-8)) {  // run 20x
      std::cout << "[Ok]: runMLP is repeatable \n";
  } else {
      std::cerr << "[Error]: runMLP produced different outputs\n";
  }

  std::cout << "------------" << std::endl;
  std::cout << "Observing weights " << std::endl;
  std::cout << "------------" << std::endl;

  std::cout << "Architecture: ";
  mlp.printArchitecture();

  // Get and print each layer’s weights
  size_t L = MyArchitecture.size();
  for (size_t i = 0; i < L; ++i) {
      Eigen::MatrixXd W = mlp.getWeights(i);
      std::cout << "Layer " << i 
                << " weight matrix (" << W.rows() << "×" << W.cols() << "):\n"  
                // W.rows(): number of neurons in the current layer
                // W.cols(): number of inputs coming into that layer (i.e. the size of the previous layer’s output) plus one extra column for the bias weight
                << W << "\n\n";
  }

  // // Modify the first layer’s weights (e.g., scale them) and verify the change
  // Eigen::MatrixXd W0 = mlp.getWeights(0);
  // W0 *= 0.5;  // scale all weights by 0.5
  // mlp.setWeights(0, W0);  // write them back
  // Eigen::MatrixXd W0b = mlp.getWeights(0);
  // std::cout << "Layer 0 weights after scaling:\n" << W0b << "\n";


  //!! ------------------------------------------------------------
  //!! LAYER
  //!! ------------------------------------------------------------

  std::cout << "------------" << std::endl;
  std::cout << "--  LAYER --" << std::endl;
  std::cout << "------------" << std::endl;

  // Initialize the layer: 5 inputs (including bias!), 3 neurons in the layer (others are default)
  //layer.initLayer(5, 3); 

  // Initialize the layer: 5 inputs (including bias!), 3 neurons, random weights between 0.0 and 1.0
  layer.initLayer(5, 3, weight_init_type::RANDOM,  activ_func_type::RELU, 0.0, 1.0);
  // Print the weights matrix
  Eigen::MatrixXd printWeightsR = layer.getWeights();
  std::cout << "Weights initialized to (RAND):\n" << printWeightsR << std::endl;

/*
  layer.initLayer(5, 3, weight_init_type::LHS, 0.0, 1.0);
  // Print the weights matrix
  Eigen::MatrixXd printWeightsL = layer.getWeights();
  std::cout << "Weights initialized to (LHS):\n" << printWeightsL << std::endl;

  layer.initLayer(5, 3, weight_init_type::LHS2, 0.0, 1.0);
  // Print the weights matrix
  Eigen::MatrixXd printWeightsL2 = layer.getWeights();
  std::cout << "Weights initialized to (LHS2):\n" << printWeightsL2 << std::endl;
*/

  // Example input
  Eigen::VectorXd actualInputs(5);
  actualInputs << 10.0, 2.0, 3.0, 4.0, 5.0;
  layer.setInputs(actualInputs);

  // Get and print the inputs to the layer
  Eigen::VectorXd layerInputs = layer.getInputs();
  std::cout << "Layer Inputs: " << layerInputs.transpose() << std::endl;

  // Calculate activations using ReLU
  Eigen::VectorXd reluActivations = layer.calculateLayerOutput(activ_func_type::RELU);
  std::cout << "ReLU Activations: " << reluActivations.transpose() << std::endl;

/*
  // Calculate activations using Sigmoid
  Eigen::VectorXd sigmoidOutput = layer.calculateLayerOutput(activ_func_type::SIGMOID);
  std::cout << "Sigmoid Activations: " << sigmoidOutput.transpose() << std::endl;
  
  // Calculate activations using Linear
  Eigen::VectorXd linearOutput = layer.calculateLayerOutput(activ_func_type::LINEAR);
  std::cout << "Linear Activations: " << linearOutput.transpose() << std::endl;

  // Calculate activations using Tanh
  Eigen::VectorXd tanhOutput = layer.calculateLayerOutput(activ_func_type::TANH);
  std::cout << "Tanh Activations: " << tanhOutput.transpose() << std::endl;

  // Calculate activations using Gaussian
  Eigen::VectorXd gaussianOutput = layer.calculateLayerOutput(activ_func_type::GAUSSIAN);
  std::cout << "Gaussian Activations: " << gaussianOutput.transpose() << std::endl;

  // Calculate activations using IABS
  Eigen::VectorXd iabsOutput = layer.calculateLayerOutput(activ_func_type::IABS);
  std::cout << "IABS Activations: " << iabsOutput.transpose() << std::endl;

  // Calculate activations using LOGLOG
  Eigen::VectorXd loglogOutput = layer.calculateLayerOutput(activ_func_type::LOGLOG);
  std::cout << "LOGLOG Activations: " << loglogOutput.transpose() << std::endl;

  // Calculate activations using CLOGLOG
  Eigen::VectorXd cloglogOutput = layer.calculateLayerOutput(activ_func_type::CLOGLOG);
  std::cout << "CLOGLOG Activations: " << cloglogOutput.transpose() << std::endl;

  // Calculate activations using CLOGLOGM
  Eigen::VectorXd cloglogmOutput = layer.calculateLayerOutput(activ_func_type::CLOGLOGM);
  std::cout << "CLOGLOGM Activations: " << cloglogmOutput.transpose() << std::endl;

  // Calculate activations using ROOTSIG
  Eigen::VectorXd rootsigOutput = layer.calculateLayerOutput(activ_func_type::ROOTSIG);
  std::cout << "ROOTSIG Activations: " << rootsigOutput.transpose() << std::endl;

  // Calculate activations using LOGSIG
  Eigen::VectorXd logsigOutput = layer.calculateLayerOutput(activ_func_type::LOGSIG);
  std::cout << "LOGSIG Activations: " << logsigOutput.transpose() << std::endl;

  // Calculate activations using SECH
  Eigen::VectorXd sechOutput = layer.calculateLayerOutput(activ_func_type::SECH);
  std::cout << "SECH Activations: " << sechOutput.transpose() << std::endl;

  // Calculate activations using WAWE 
  Eigen::VectorXd waveOutput = layer.calculateLayerOutput(activ_func_type::WAVE);
  std::cout << "WAWE Activations: " << waveOutput.transpose() << std::endl;
*/

  // Get and print the output of the layer
  Eigen::VectorXd output = layer.getOutput();
  std::cout << "Layer Outputs: " << output.transpose() << std::endl;

  std::cout << "\n-------------------------------------------" << std::endl;
  std::cout << "-- Upload data from file --" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  Data data;

  // Filter data by single ID "94206029"
  std::unordered_set<std::string> ids = {"94206029"};

  // Select which columns to keep
  std::vector<std::string> keepCols = {"T1", "T2", "T3", "moisture"};

  try {
      //size_t n = data.loadFilteredCSV("data/inputs/data_all_hourly.csv",  // has "hour_start" column
      size_t n = data.loadFilteredCSV("data/inputs/data_all_daily.csv",  // has "date" column
        ids,
        keepCols,
        "date", //"hour_start",
        "ID");

      // Print total number of loaded row
      std::cout << "Total number of rows: " << n << "\n";

      // Print header names
      data.printHeader("date");   // or "hour_start"

      const auto& times = data.timestamps();
      const auto& mat = data.numericData();

      // Print couple of the first rows (5)
      for (size_t i = 0; i < std::min<size_t>(5, n); ++i) {
          std::cout << times[i] << " | ";
          for (int c = 0; c < mat.cols(); ++c) {
              std::cout << mat(i, c) << (c+1<mat.cols()? " | " : "");
          }
          std::cout << "\n";
      }
  } catch (std::exception &ex) {
      std::cerr << "[Error]: " << ex.what() << "\n";
  }

  // Select only one column (moisture)
  auto moistureData = data.getColumnValues("moisture");
  size_t nMoistureData = moistureData.size();
  if (nMoistureData == 0) { 
    std::cout << "no data\n"; 
    return {}; 
  }
  Eigen::RowVectorXd firstMoisture;
  size_t N = std::min<size_t>(3, nMoistureData);
  {
      Eigen::Map<const Eigen::VectorXd> mview(moistureData.data(), static_cast<Eigen::Index>(N));
      firstMoisture = mview.transpose(); 
  }
  std::cout << "First 3 moisture values: " << firstMoisture << "\n";

  // Detect and remove rows with NaNs
  auto naIdx = data.findRowsWithNa();
  std::cout << "Number of rows with any NaN: " << naIdx.size() << "\n";
  if (!naIdx.empty()) {
      data.removeRowsWithNa();
  }

  std::cout << "\n-------------------------------------------" << std::endl;
  std::cout << "-- Transformation of data --" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  std::cout << "Before transform (first row): " << data.numericData().row(0) << "\n";
  data.setTransform(transform_type::MINMAX);
  data.applyTransform();
  std::cout << "After MINMAX transform (first row):  " << data.numericData().row(0) << "\n";
  data.inverseTransform();
  std::cout << "After inverse (first row):   " << data.numericData().row(0) << "\n";

  std::cout << "Before transform (first row): " << data.numericData().row(0) << "\n";
  data.setTransform(transform_type::NONLINEAR, 0.015, false);
  data.applyTransform();
  std::cout << "After NONLINEAR transform (first row):  " << data.numericData().row(0) << "\n";
  data.inverseTransform();
  std::cout << "After inverse (first row):   " << data.numericData().row(0) << "\n";

  std::cout << "\n-------------------------------------------" << std::endl;
  std::cout << "--  Testing criteria (no real data!) --" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
  
  // Outputs as vector
  Eigen::VectorXd y_true(4), y_pred(4);
  y_true << 1.0, 2.0, 0.0, 4.0;
  y_pred << 0.9, 2.1, 0.01, 3.9;

  double mse_v = Metrics::mse(y_true, y_pred);
  double rmse_v = Metrics::rmse(y_true, y_pred);
  std::cout << "Outputs as vector: MSE = " << mse_v << ", RMSE = " << rmse_v << "\n";
  
  // Outputs as matrix
  Eigen::MatrixXd Y_true(2,3), Y_pred(2,3);
  Y_true << 1.0, 2.0, 0.0, 4.0, 2.5, 1.9;
  Y_pred << 0.9, 2.1, 0.01, 3.9, 3.1, 0.8;

  double mse_m = Metrics::mse(Y_true, Y_pred);
  double rmse_m = Metrics::rmse(Y_true, Y_pred);
  std::cout << "Outputs as matrix: MSE = " << mse_m << ", RMSE = " << rmse_m << "\n";

  // Save metrics into CSV file (need to have an existing folder "data/outputs")
  Metrics::computeAndAppendFinalMetrics(Y_true, Y_pred, "data/outputs/final_metrics.csv");


  std::cout << "\n-------------------------------------------" << std::endl;
  std::cout << "--  Testing NAs removal - vector (no real data!) --" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  // Ensure folder exists: "data/inputs"
  const std::string path = "data/inputs/test_artificial.csv";

  // Create a small CSV with some missing values (empty fields and "NA")
  {
      std::ofstream ofs(path);
      if (!ofs.is_open()) {
          std::cerr << "Cannot create test CSV at " << path << "\n";
          return 1;
      }
      // header: ID, date, then numeric cols T1,T2,moisture
      ofs << "ID,date,T1,T2,moisture\n";
      ofs << "1,2020-01-01,1.0,2.0,0.50\n";       // OK
      ofs << "2,2020-01-02,1.10,,0.55\n";         // missing T2 -> NaN
      ofs << "3,2020-01-03,,2.20,0.60\n";         // missing T1 -> NaN
      ofs << "4,2020-01-04,1.30,2.30,\n";         // missing moisture -> NaN
      ofs << "5,2020-01-05,1.40,2.40,0.65\n";     // OK
      ofs << "6,2020-01-06,NA,2.50,0.70\n";       // "NA" -> NaN
      ofs.close();
      std::cout << "Wrote test CSV: " << path << "\n";
  }

  // Load the CSV using Data
  Data dataArtificial;
  std::vector<std::string> keepColsArtificial = {"T1", "T2", "moisture"};

  try {
      // Empty idFilter to accept all IDs
      std::unordered_set<std::string> idFilter;
      size_t nrows = dataArtificial.loadFilteredCSV(path, idFilter, keepColsArtificial, "date", "ID");
      std::cout << "Loaded rows: " << nrows << "\n";

      // Store original timestamps (so we can print full timeline later)
      std::vector<std::string> original_times = dataArtificial.timestamps();

      // Print the loaded numeric matrix
      Eigen::MatrixXd MArtificial = dataArtificial.numericData();
      std::cout << "Loaded numeric data (" << MArtificial.rows() << " x " << MArtificial.cols() << "):\n";
      for (int r = 0; r < MArtificial.rows(); ++r) {
          std::cout << " row " << r << ": ";
          for (int c = 0; c < MArtificial.cols(); ++c) {
              double v = MArtificial(r,c);
              if (std::isfinite(v)) std::cout << v;
              else std::cout << "NaN";
              if (c + 1 < MArtificial.cols()) std::cout << " | ";
          }
          std::cout << "\n";
      }

      // Find rows with any NaN
      auto naRows = dataArtificial.findRowsWithNa();
      std::cout << "Rows containing NaN: ";
      if (naRows.empty()) std::cout << "(none)";
      for (auto idx : naRows) std::cout << idx << " ";
      std::cout << "\n";

      // Remove rows with NaN and show filtered data
      dataArtificial.removeRowsWithNa();
      Eigen::MatrixXd M_filtered = dataArtificial.numericData();
      std::cout << "After removeRowsWithNa(): data has " << M_filtered.rows() << " rows\n";
      for (int r = 0; r < M_filtered.rows(); ++r) {
          std::cout << " kept row " << r << " : ";
          for (int c = 0; c < M_filtered.cols(); ++c) {
              double v = M_filtered(r,c);
              if (std::isfinite(v)) std::cout << v;
              else std::cout << "NaN";
              if (c + 1 < M_filtered.cols()) std::cout << " | ";
          }
          std::cout << "\n";
      }

      // Simulate predictions on the filtered rows (one output per row)
      int validRows = static_cast<int>(M_filtered.rows());
      // Produce one predicted value per valid row, e.g. predicted moisture = last column of filtered row + 0.1
      Eigen::MatrixXd preds_valid(validRows, 1);
      for (int i = 0; i < validRows; ++i) {
          double observed_moisture = M_filtered(i, M_filtered.cols()-1);
          preds_valid(i,0) = std::isfinite(observed_moisture) ? (observed_moisture + 0.1) : std::numeric_limits<double>::quiet_NaN();
      }

      // Expand predictions back to full timeline (original length)
      Eigen::MatrixXd preds_full = dataArtificial.expandPredictionsToFull(preds_valid);
      std::cout << "Expanded preds_full rows: " << preds_full.rows() << " cols: " << preds_full.cols() << "\n";

      // Print the full timeline alongside timestamps (using original_times captured earlier)
      std::cout << "Full timeline (timestamp | predicted):\n";
      for (int r = 0; r < preds_full.rows(); ++r) {
          std::cout << " " << original_times[r] << " | ";
          double v = preds_full(r, 0);
          if (std::isfinite(v)) std::cout << v;
          else std::cout << "NaN";
          std::cout << "\n";
      }

      // Restore original data in memory
      dataArtificial.restoreOriginalData();
  }
  catch (std::exception &ex) {
      std::cerr << "[Error]: " << ex.what() << "\n";
      return 1;
  }


  std::cout << "\n-------------------------------------------" << std::endl;
  std::cout << "--  Testing NAs removal - matrix (no real data!) --" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  try {
    const std::string pathA = "data/inputs/test_multi_horizon.csv";
    {
        std::ofstream ofs(pathA);
        ofs << "ID,date,T1,T2,moisture\n";
        ofs << "1,2020-01-01,1.0,2.0,0.50\n";       // OK
        ofs << "2,2020-01-02,1.10,1.9,0.55\n";      // OK
        ofs << "3,2020-01-03,1.17,2.20,0.60\n";     // OK
        ofs << "4,2020-01-04,1.30,2.30,0.62\n";     // OK
        ofs << "5,2020-01-05,1.40,2.40,0.65\n";     // OK
        ofs << "5,2020-01-06,1.35,2.30,0.55\n";     // OK
        ofs << "5,2020-01-07,1.20,    ,0.60\n";         // missing T2 -> NaN
        ofs << "5,2020-01-08,1.80,2.10,0.68\n";     // OK
        ofs << "5,2020-01-09,1.80,2.60,0.65\n";     // OK
        ofs << "5,2020-01-10,1.72,2.50,0.58\n";     // OK
        ofs << "5,2020-01-11,1.40,2.50,0.52\n";     // OK
        //ofs << "5,2020-01-12,1.40,2.50,0.52\n";     // OK
        //ofs << "5,2020-01-13,1.40,2.50,0.52\n";     // OK
        ofs.close();
        std::cout << "Wrote test CSV: " << pathA << "\n";
    }

    Data dataMultiOut;
    std::vector<std::string> keepA = {"T1","T2","moisture"};
    std::unordered_set<std::string> idFilterA; // accept all
    size_t nA = dataMultiOut.loadFilteredCSV(pathA, idFilterA, keepA, "date", "ID");
    std::cout << "Loaded rows: " << nA << "\n";

    // save original timestamps BEFORE filtering
    std::vector<std::string> timesA = dataMultiOut.timestamps();

    auto naRowsA = dataMultiOut.findRowsWithNa();
    std::cout << "Number of rows with any NaN: " << naRowsA.size() << "\n";
    std::cout << "Row index with NaN: ";
    for (auto r : naRowsA) std::cout << r << " ";
    std::cout << "\n";

    // remove rows that contain any NaN
    dataMultiOut.removeRowsWithNa();
    Eigen::MatrixXd MfA = dataMultiOut.numericData();
    std::cout << "Filtered dataset: " << MfA.rows() << " (rows) x " << MfA.cols() << " (cols) \n";

    // Build calibration matrix for multi-horizon outputs (inpRows = 2, out_horizon = 2)
    int inpRows = 2;
    int out_horizon = 2;
    dataMultiOut.makeCalibMat({2,2,2}, out_horizon);
    Eigen::MatrixXd C = dataMultiOut.getCalibMat(); // CR x (inpRows*inputCols + out_horizon)
    const Eigen::Index CR = C.rows();
    const Eigen::Index CC = C.cols();
    const Eigen::Index inputSize = CC - out_horizon;

    // Split data into calibration and validation sets
    double trainFrac = 0.8;
    bool shuffle = true;
    unsigned seed = 42;  // choose 0 for non-deterministic

    Eigen::MatrixXd trainMat, validMat;
    std::vector<int> trainIdx, validIdx;
    std::tie(trainMat, validMat, trainIdx, validIdx) = dataMultiOut.splitCalibMatWithIdx(trainFrac, shuffle, seed);

    // calib rows = filteredRows - inpRows - out_horizon + 1
    std::cout << "calib rows: " << dataMultiOut.getCalibMat().rows()
              << " -> train: " << trainMat.rows() << ", valid: " << validMat.rows() << "\n";

    // If you saved original timestamps earlier (before calling removeRowsWithNa()):
    auto times_all = dataMultiOut.timestamps();

    // Print first 10 validation rows and their original indices
    int toPrint = std::min<int>(10, static_cast<int>(validIdx.size()));
    for (int i = 0; i < toPrint; ++i) {
        int calibRow = validIdx[i]; // index into calibMat 
        std::cout << "valid row " << i << " : calibIndex=" << calibRow;
        // If you have a mapping calibIndex -> original_time_index (e.g. m_calib_pattern_orig_indices),
        // map that here to get the timestamp. Example (pseudocode):
        // size_t origRow = m_calib_pattern_orig_indices[calibRow];
        // std::cout << " origRow=" << origRow << " time=" << times_all[origRow];
        std::cout << "\n";
    }

    // Extract truth_valid (CR x out_horizon) and synthetic preds_valid
    Eigen::MatrixXd Y_true_valid = C.block(0, inputSize, CR, out_horizon);
    Eigen::MatrixXd preds_valid = Y_true_valid;
    // synthetic shift so preds != truth
    preds_valid.array() += 0.1;

    // compute a quick metric on valid rows
    double mseA = Metrics::mse(Y_true_valid, preds_valid);
    std::cout << "MSE (multi-horizon, valid rows) = " << mseA << "\n";

    // Expand predictions back to full timeline (use inpRows from makeCalibMat)
    Eigen::MatrixXd preds_full = dataMultiOut.expandPredictionsFromCalib(preds_valid, inpRows);
    std::cout << "Expanded preds_full: " << preds_full.rows() << " x " << preds_full.cols() << "\n";

    // Print full timeline (timestamps from timesA)
    std::cout << "Timestamp | pred_h1 | pred_h2\n";
    for (int r = 0; r < preds_full.rows(); ++r) {
        std::cout << " " << timesA[r] << " | ";
        for (int c = 0; c < preds_full.cols(); ++c) {
            double v = preds_full(r,c);
            if (std::isfinite(v)) std::cout << v;
            else std::cout << "NaN";
            if (c+1 < preds_full.cols()) std::cout << " | ";
        }
        std::cout << "\n";
    }

    // Restore original data if you want to reuse dataMultiOut later
    dataMultiOut.restoreOriginalData();

  } catch (const std::exception &ex) {
    std::cerr << "[Error]: " << ex.what() << "\n";
    return 1;
  }

  


  // std::cout << "\n-------------------------------------------" << std::endl;
  // std::cout << "--   Testing split calibMats, shuffling  --" << std::endl;
  // std::cout << "-------------------------------------------" << std::endl;
  // std::cout<<"data: \n"<<data.numericData().topRows(5)<<endl<<"\n";
  // data.makeCalibMat({1,3,2,1},2);     // input length = 7
  // std::cout<<"calibMat: \n"<<data.getCalibMat().topRows(3)<<endl<<"\n";
  // data.splitCalibMat(7);
  // std::cout<<"split mats from calibMat: \n"<<data.getCalibInpsMat().topRows(3)<<endl<<"\n";
  // std::cout<<data.getCalibOutsMat().topRows(3)<<endl<<"\n";
  // data.makeCalibMatsSplit({1,3,2,1},2);
  // std::cout<<"directly created split mats: \n"<<data.getCalibInpsMat().topRows(3)<<endl<<"\n";
  // std::cout<<data.getCalibOutsMat().topRows(3)<<endl<<"\n";
  // std::vector<int>permut = data.permutationVector(data.getCalibInpsMat().rows());
  // data.setCalibInpsMat(data.shuffleMatrix(data.getCalibInpsMat(),permut));
  // data.setCalibOutsMat(data.shuffleMatrix(data.getCalibOutsMat(),permut));
  // std::cout<<"shuffled split mats: \n"<<data.getCalibInpsMat().topRows(3)<<endl<<"\n";
  // std::cout<<data.getCalibOutsMat().topRows(3)<<endl<<"\n";
  // data.setCalibInpsMat(data.unshuffleMatrix(data.getCalibInpsMat(),permut));
  // data.setCalibOutsMat(data.unshuffleMatrix(data.getCalibOutsMat(),permut));
  // std::cout<<"unshuffled split mats: \n"<<data.getCalibInpsMat().topRows(3)<<endl<<"\n";
  // std::cout<<data.getCalibOutsMat().topRows(3)<<endl<<"\n";

  std::cout << "\n-------------------------------------------" << std::endl;
  std::cout << "--      Testing testing online Adam      --" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;

  MLP test123;
  testAdamOnlineSplitD(test123,data,{2,1},{0,0,1,2},activ_func_type::RELU,weight_init_type::RANDOM,500,0.002,0.001);
  test123.calculateOutputs(data.getCalibInpsMat().topRows(5));
  std::cout<<"calibrated outputs: \n"<<data.getCalibOutsMat().topRows(5)<<"\n\n";
  std::cout<<"modelled outputs: \n"<<test123.getOutputs()<<"\n\n";

  std::cout<<"first layer weight mat: \n"<<test123.getWeights(0)<<"\n\n";
  std::cout<<"second layer weight mat: \n"<<test123.getWeights(1)<<"\n\n";
  test123.weightsToVectorMlp();
  std::cout<<"all weights vector: \n"<<test123.getWeightsVectorMlp().transpose();


  std::cout << "\n-------------------------------------------" << std::endl;
  std::cout << "-- Testing online Adam from JKMNet --" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
  
  JKMNet net;
  MLP mymlp;
  auto resAdam = net.trainAdamOnlineSplit(
      mymlp, data, {2,1}, {0,0,1,2},
      activ_func_type::RELU, weight_init_type::RANDOM,
      500, 0.002, 0.001, true, 42);

  mymlp.calculateOutputs(data.getCalibInpsMat().topRows(5));
  std::cout<<"calibrated outputs: \n"<<data.getCalibOutsMat().topRows(5)<<"\n\n";
  std::cout<<"modelled outputs: \n"<<mymlp.getOutputs()<<"\n\n";

  std::cout << "Training finished; final MSE=" << resAdam.finalLoss
            << ", iterations=" << resAdam.iterations
            << ", converged=" << std::boolalpha << resAdam.converged << "\n";


  return 0;
}
