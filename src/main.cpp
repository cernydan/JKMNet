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
  std::cout << "First 5 moisture values: " << firstMoisture << "\n";

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
  std::cout << "--  Testing Adam for matrix data --" << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
  
  int numInpVar = 3;             // number of values of each variable used in each pattern
  unsigned int numOuts = 2;      // number of outputs in each pattern = output neurons

  data.makeCalibMat(numInpVar, numOuts);
  data.shuffleCalibMat();

  MLP mlpbp;
  std::vector<unsigned> arch = {2,3,numOuts};
  mlpbp.setArchitecture(arch);
  mlpbp.printArchitecture();
  std::vector<weight_init_type> weiInits = {
        weight_init_type::RANDOM,weight_init_type::RANDOM,weight_init_type::RANDOM
  };
  mlpbp.setWInitType(weiInits);
  mlpbp.printActivations();
  mlpbp.printWInitType();
  Eigen::VectorXd initvec(data.getCalibMat().cols() - numOuts); // number of inputs = calibration matrix row length - number of outputs
  mlpbp.initMLP(initvec);

  Eigen::MatrixXd M;
  M = (data.getCalibMat().array().isNaN()).select(0.0, data.getCalibMat()); // change NaN to 0 ... for now
  mlpbp.batchAdam(200,20,0.001, M);

  Eigen::VectorXd jedna(12);
  Eigen::VectorXd dva(12);
  Eigen::VectorXd tri(12);
  jedna<< 5.35807,5.16797,5.02604,4.79622,4.83268,4.58984,1.64323,2.87695,2.00586,0.243134,0.242252,0.241349;
  dva<< 12.3822,12.056,11.9596,12.8229,12.459,12.3763,13.6309,13.3034,12.9238,0.148254,0.144221,0.141342;
  tri<< 9.04362,9.42383,9.86198,9.57552,10.1908,10.696,11.4434,13.3431,14.2383,0.2519,0.248034,0.24362;
  std::cout<<"\n Testing 3 input vectors from calibration data: \n";
  std::cout<<"vec1: "<<jedna.transpose()<<"\n";
  std::cout<<"vec1: "<<dva.transpose()<<"\n";
  std::cout<<"vec1: "<<tri.transpose()<<"\n \n";
  std::cout<<"Their measured outputs: \n";
  std::cout<<"0.240465   0.240071 \n";
  std::cout<<"0.139811   0.138863 \n";
  std::cout<<"0.239087   0.235801 \n \n";
  std::cout<<"Modelled outputs: \n";
  std::cout<<mlpbp.runMLP(jedna).transpose()<<"\n";
  std::cout<<mlpbp.runMLP(dva).transpose()<<"\n";
  std::cout<<mlpbp.runMLP(tri).transpose()<<"\n";


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
        ofs << "5,2020-01-07,1.20,,0.60\n";         // missing T2 -> NaN
        ofs << "5,2020-01-08,1.80,2.10,0.68\n";     // OK
        ofs << "5,2020-01-09,1.80,2.60,0.65\n";     // OK
        ofs << "5,2020-01-10,1.72,2.50,0.58\n";     // OK
        ofs << "5,2020-01-11,1.40,2.50,0.52\n";     // OK
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
    std::cout << "Rows with NaN: ";
    for (auto r : naRowsA) std::cout << r << " ";
    std::cout << "\n";

    // remove rows that contain any NaN
    dataMultiOut.removeRowsWithNa();
    Eigen::MatrixXd MfA = dataMultiOut.numericData();
    std::cout << "Filtered rows: " << MfA.rows() << " x " << MfA.cols() << "\n";

    // Build calibration matrix for multi-horizon outputs (inpRows = 2, out_horizon = 2)
    int inpRows = 2;
    int out_horizon = 2;
    dataMultiOut.makeCalibMat(inpRows, out_horizon);
    Eigen::MatrixXd C = dataMultiOut.getCalibMat(); // CR x (inpRows*inputCols + out_horizon)
    const Eigen::Index CR = C.rows();
    const Eigen::Index CC = C.cols();
    const Eigen::Index inputSize = CC - out_horizon;

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


  return 0;
}
