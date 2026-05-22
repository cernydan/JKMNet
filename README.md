# JKMNet: A configurable neural-network framework for environmental time-series prediction

`JKMNet` is a C++20 project for training and applying **feed-forward neural networks (MLP)** to environmental and time-series data.  
The framework supports **parallel ensemble training**, **Adam optimizer**, and **post-hoc prediction from saved weights**.

---

## ✨ Main Features
- Multi-layer perceptron (MLP) implementation with customizable architecture  
- Parallel ensemble training using OpenMP  
- Multiple training modes: online, batch, epoch-based  
- Transformations of inputs/outputs (e.g. MinMax, Z-score)  
- Saving and reloading of model weights for later predictions  
- CSV-based input/output for easy integration with other tools

---

## 📦 Requirements
- Linux or macOS (tested on Ubuntu 22.04)  
- `g++` with C++20 support  
- [OpenMP](https://www.openmp.org/) for parallel processing  
- [Eigen](https://eigen.tuxfamily.org/) for linear algebra (header-only library)  
- `make`

---

## 🧰 Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/JKMNet.git
cd JKMNet
```

### 2. Compile the project
```bash
make
```
This will:
- create the `obj` and `bin` directories
- build the executable at `bin/JKMNet`

If you want to clean the build:
```bash
make clean
```

---

## 🧪 Running the program

### 🔹 1. Training + Testing Mode

To train an ensemble of MLPs and test it on other data:

```bash
./bin/JKMNet [threads]
```

Example:
```bash
./bin/JKMNet 4
```
This will train using 4 threads and save:
- trained weights  
- calibration and validation predictions  
- metrics per run  
- log files in `data/outputs/`

---

### 🔹 2. Prediction Mode

Once the model has been trained and weights are saved, you can run the prediction mode:

```bash
./bin/JKMNet predict
```
This will use the **default weights file** path specified in `settings/config_model.ini`.

You can also specify a custom weights file:

```bash
./bin/JKMNet predict data/outputs/weights/weights_final_1.csv
```

Optionally, you can specify the number of threads:
```bash
./bin/JKMNet predict data/outputs/weights/weights_final_1.csv 4
```

If the weight file doesn’t exist, the program will exit cleanly with a clear error message.

---

## ⚙️ Configuration

All model and training settings are specified in:
```
settings/config_model.ini
```

Example parameters:
```ini
[data]
data_file = data/inputs/input_data.csv
columns = T1, T2, T3, moisture

[model]
architecture = 8, 6, 4
activation = RELU
trainer = online

[training]
ensemble_runs = 20
max_iterations = 500
learning_rate = 0.001
train_fraction = 0.8

[paths]
weights_csv = data/outputs/weights/weights_final.csv
```

---

## 🧠 Example Workflow

1. Prepare your dataset as a CSV file.  
2. Configure the model architecture and paths in `config_model.ini`.  
3. Run training and testing:
   ```bash
   ./bin/JKMNet 4
   ```
4. Use the trained weights for prediction:
   ```bash
   ./bin/JKMNet predict
   ```
5. Check outputs in `data/outputs/`:
   - `weights/` — saved model weights  
   - `metrics/` — calibration and validation metrics  
   - `logs/` — per-run log files  
   - `*.csv` — predicted values

---

## 🧭 Project Structure

```
JKMNet/
├── src/                    # C++ source files
├── include/                # Header files
├── data/
│   ├── inputs/             # Input data
│   └── outputs/            # Outputs, logs, metrics
├── settings/               # Configuration files
├── Makefile
├── README.md
└── bin/                    # Compiled binary
```

---

## 🧑‍💻 Development Notes
- The modular structure allows easy extension with new optimizers or layer types (e.g., CNN or hybrid models).  
- Parallelization is handled via OpenMP (`#pragma omp parallel for`) for ensemble training.

---

## 📜 License
MIT License.  

---
