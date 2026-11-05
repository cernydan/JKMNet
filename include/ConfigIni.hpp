/**
 * @file ConfigIni.hpp
 * @brief Declares the RunConfig struct and the helpers that parse the JKMNet .ini configuration file.
 */
#ifndef CONFIG_HPP
#define CONFIG_HPP

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "Data.hpp"   
#include "MLP.hpp"    

/**
 * Helper functions
 */

/// Strip leading and trailing whitespace from a string.
/**
 * @param s Input string.
 * @return Copy of @p s without leading/trailing whitespace characters.
 */
static inline std::string trimStr(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b-a);
}

// static std::string join(const std::vector<std::string>& v, const std::string& sep = ", ") {
//     std::string out;
//     for (size_t i = 0; i < v.size(); ++i) {
//         out += v[i];
//         if (i + 1 < v.size()) out += sep;
//     }
//     return out;
// }

/// Split a comma-separated config value into trimmed, non-empty tokens.
/**
 * @param s Raw value string, e.g. "8, 6, 2".
 * @return Vector of trimmed tokens; empty entries are dropped.
 */
static inline std::vector<std::string> splitCommaList(const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream iss(s);
    while (std::getline(iss, cur, ',')) {
        std::string t = trimStr(cur);
        if (!t.empty()) out.push_back(t);
    }
    return out;
}

/// Split a string by a single delimiter, trimming each resulting token.
/**
 * @param s Input string.
 * @param delim Delimiter character.
 * @return Vector of trimmed tokens (empty tokens are kept, unlike splitCommaList).
 */
static inline std::vector<std::string>
splitList(const std::string& s, char delim)
{
    std::vector<std::string> out;
    size_t start = 0;

    while (true) {
        size_t pos = s.find(delim, start);
        if (pos == std::string::npos) {
            out.push_back(trimStr(s.substr(start)));
            break;
        }
        out.push_back(trimStr(s.substr(start, pos - start)));
        start = pos + 1;
    }

    return out;
}

/// Parse a boolean from a config string.
/**
 * Accepts (case-insensitive): 1/TRUE/YES/ON and 0/FALSE/NO/OFF.
 * @param s Value string from the ini file.
 * @return The parsed boolean value.
 * @throw std::runtime_error If the string matches none of the accepted forms.
 */
static inline bool parseBool(const std::string &s) {
    std::string x; x.reserve(s.size());
    for (char c : s) x.push_back(static_cast<char>(std::toupper((unsigned char)c)));
    if (x=="1" || x=="TRUE" || x=="YES" || x=="ON") return true;
    if (x=="0" || x=="FALSE" || x=="NO" || x=="OFF") return false;
    throw std::runtime_error("parseBool: cannot parse boolean from '" + s + "'");
}

/// Configuration of one program run, parsed from the .ini file by parseConfigIni().
/**
 * Every member corresponds to a user-facing ini key of the same name.
 * Members with default values are optional in the ini file; the ones checked
 * at the end of parseConfigIni() (data_file, architecture, input_numbers) are
 * required.
 * @note A single-element per-column list (activation, transform,
 *       transform_alpha) is broadcast to the number of layers/columns.
 */
struct RunConfig {
    // model / training
    std::string trainer = "online";                 ///< Training variant ("online" or "batch"); selects the MLP trainer used by the ensemble run
    std::vector<unsigned> mlp_architecture;         ///< Number of neurons per MLP layer, e.g. [8,6,2] (required, ini key: architecture)
    std::vector<std::vector<int>> input_numbers;    ///< Time offsets used as inputs per data column, one |-separated list per column, e.g. "-2,-1,0 | -1,0,1,2 | | -2,-1" (required, ini key: input_numbers)
    std::vector<std::string> activation = {"RELU"}; ///< Activation function name(s) per layer; a single value is broadcast to all layers (see activ_func_type in Layer.hpp)
    std::string weight_init = "RANDOM";             ///< Weight initialization scheme for all layers (RANDOM, LHS, LHS2, HE, XG)

    int ensemble_runs = 25;      ///< Number of independent MLPs trained in the ensemble run
    int max_iterations = 500;    ///< Maximum number of training iterations (epochs/pattern passes, trainer-dependent)
    int max_metrics_step = 99;   ///< Compute and save metrics every n-th epoch in the *_EPOCH trainers
    double max_error = 0.002;    ///< Target mean error; training stops early once the epoch error falls below it
    double learning_rate = 0.001;///< Step size of the weight updates (Adam / backpropagation)
    bool shuffle = true;         ///< Whether to shuffle pattern order during training
    unsigned seed = 42;          ///< RNG seed for weight initialization and shuffling (0 means non-deterministic where supported)
    int batch_size = 30;         ///< Number of patterns per batch in the batch trainers
    double train_fraction = 0.8; ///< Fraction of patterns used for calibration; the rest forms the validation set
    bool split_shuffle = true;   ///< Whether the calibration/validation split shuffles the pattern order

    int lstm_past_time_steps = 30;   ///< Length of the past observation window fed to the LSTM encoder
    int lstm_future_time_steps = 10; ///< Length of the known future forcing window (lstm_past_future model)
    int lstm_cells = 10;             ///< Number of LSTM cells (hidden state size) per LSTM layer
    std::string model_type = "mlp";  ///< Model type selection: "mlp", "lstm_first", or "lstm_past_future"
    bool save_epoch_metrics = false; ///< Compute and save per-epoch metrics (MLP *_epoch trainers and LSTM models; also gates the extra forward passes)

    // objective function for training
    std::string objective_function = "LEVEL_SLOPE";  ///< Objective (loss) function: MSE, DIFF_SLOPE, LEVEL_SLOPE, DIFF_CURVATURE, LEVEL_SLOPE_CURVATURE, NORMALIZED_SLOPE, PERSISTENCE_INDEX, DILATE
    double objective_alpha = 0.5;    ///< Weighting parameter for combined losses (alpha in L2)
    double objective_alpha2 = 0.33;  ///< Second weighting parameter for the curvature term

    // transforms
    std::vector<std::string> transform;      ///< Data transform per column (NONE, MINMAX, NONLINEAR, MINMAXNONLIN, ZSCORE); a single value is broadcast to all columns
    std::vector<double> transform_alpha;     ///< Transform parameter (alpha) per column; must match the number of columns
    bool exclude_last_col_from_transform = false; ///< If true, the last data column is left untransformed
    bool remove_na_before_calib = true;      ///< If true, rows containing NaN are dropped before building the calibration matrices

    // data selection
    std::string data_file;              ///< Path to the input CSV data file (required)
    std::string id = "";                ///< Optional station id filter (comma-separated list); empty means use all ids
    std::string id_col = "ID";          ///< Name of the CSV column holding the station id
    std::string timestamp = "date";     ///< Name of the CSV column holding the timestamp
    std::vector<std::string> columns;   ///< Names of the CSV columns used as model inputs/target; the last one is the modeled variable

    // optimization
    bool pso_optimize = true;           ///< Whether hyperparameters are optimized with particle swarm optimization (PSO)

    // output paths (parsed from [paths] section)
    std::string out_dir = "";            ///< Output directory for the run results
    std::string log_dir = "";            ///< Directory for per-run log files
    std::string calib_mat = "";          ///< Path to save the calibration input matrix
    std::string weights_csv_init = "";   ///< Path for initial weights in readable CSV text (per-layer blocks)
    std::string weights_bin_init = "";   ///< Path for initial weights in compact binary format
    std::string weights_vec_csv_init = "";///< Path for initial flattened weight vector in readable CSV text
    std::string weights_vec_bin_init = "";///< Path for initial flattened weight vector in compact binary format
    std::string weights_csv = "";        ///< Path for final weights in readable CSV text (per-layer blocks)
    std::string weights_bin = "";        ///< Path for final weights in compact binary format
    std::string weights_vec_csv = "";    ///< Path for final flattened weight vector in readable CSV text
    std::string weights_vec_bin = "";    ///< Path for final flattened weight vector in compact binary format
    std::string real_calib = "";         ///< Path to save real (observed) calibration outputs in original units
    std::string pred_calib = "";         ///< Path to save predicted calibration outputs in original units
    std::string real_valid = "";         ///< Path to save real (observed) validation outputs in original units
    std::string pred_valid = "";         ///< Path to save predicted validation outputs in original units
    std::string metrics_cal = "";        ///< Path to the calibration metrics file (one row per metric)
    std::string metrics_val = "";        ///< Path to the validation metrics file (one row per metric)
    std::string run_info = "";           ///< Path for the run info/summary file
    std::string errors_csv;              ///< Path for the per-iteration error history
    std::string pattern_indices = "";    ///< Path to save the indices of the patterns used
};

/**
 * Mapping helpers: string -> enum conversions used while reading the config.
 */

/// Convert an activation function name (case-insensitive) to activ_func_type.
/**
 * @param s Name such as "RELU", "TANH", "SIGMOID" (see activ_func_type in Layer.hpp).
 * @return The matching enum value.
 * @throw std::runtime_error For an unrecognized name.
 * @see activ_func_type
 */
inline activ_func_type strToActivation(const std::string &s) {
    std::string u;
    for (char c : s) u.push_back(static_cast<char>(std::toupper((unsigned char)c)));
    if (u == "RELU") return activ_func_type::RELU;
    if (u == "LEAKYRELU") return activ_func_type::LEAKYRELU;
    if (u == "TANH") return activ_func_type::TANH;
    if (u == "SIGMOID") return activ_func_type::SIGMOID;
    if (u == "LINEAR") return activ_func_type::LINEAR;
    if (u == "IABS") return activ_func_type::IABS;
    if (u == "GAUSSIAN") return activ_func_type::GAUSSIAN;
    if (u == "LOGLOG") return activ_func_type::LOGLOG;
    if (u == "CLOGLOG") return activ_func_type::CLOGLOG;
    if (u == "CLOGLOGM") return activ_func_type::CLOGLOGM;
    if (u == "ROOTSIG") return activ_func_type::ROOTSIG;
    if (u == "LOGSIG") return activ_func_type::LOGSIG;
    if (u == "SECH") return activ_func_type::SECH;
    if (u == "WAVE") return activ_func_type::WAVE;

    throw std::runtime_error("Unknown activation: " + s);
}

/// Convert a weight initialization name (case-insensitive) to weight_init_type.
/**
 * @param s One of "RANDOM", "LHS", "LHS2", "HE", "XG".
 * @return The matching enum value.
 * @throw std::runtime_error For an unrecognized name.
 * @see weight_init_type
 */
inline weight_init_type strToWeightInit(const std::string &s) {
    std::string u;
    for (char c : s) u.push_back(static_cast<char>(std::toupper((unsigned char)c)));
    if (u == "RANDOM") return weight_init_type::RANDOM;
    if (u == "LHS") return weight_init_type::LHS;
    if (u == "LHS2") return weight_init_type::LHS2;
    if (u == "HE") return weight_init_type::HE;
    if (u == "XG") return weight_init_type::XG;

    throw std::runtime_error("Unknown weight_init: " + s);
}

/// Convert a data transform name (case-insensitive) to transform_type.
/**
 * @param s One of "NONE", "MINMAX", "NONLINEAR", "MINMAXNONLIN", "ZSCORE".
 * @return The matching enum value.
 * @throw std::runtime_error For an unrecognized name.
 * @see transform_type (declared in Data.hpp)
 */
inline transform_type strToTransformType(const std::string &s) {
    std::string u;
    for (char c : s) u.push_back(static_cast<char>(std::toupper((unsigned char)c)));
    if (u == "NONE") return transform_type::NONE;
    if (u == "MINMAX") return transform_type::MINMAX;
    if (u == "NONLINEAR") return transform_type::NONLINEAR;
    if (u == "MINMAXNONLIN") return transform_type::MINMAXNONLIN;
    if (u == "ZSCORE") return transform_type::ZSCORE;

    throw std::runtime_error("Unknown transform: " + s);
}

/// Convert a list of transform names to transform_type values.
/**
 * @param strs Vector of transform names (one per data column).
 * @return Vector of matching enum values, same order and length as @p strs.
 * @throw std::runtime_error If any entry is invalid; the message includes its index.
 */
inline std::vector<transform_type>
strVecToTransformTypes(const std::vector<std::string>& strs)
{
    std::vector<transform_type> out;
    out.reserve(strs.size());
    for (std::size_t i = 0; i < strs.size(); ++i) {
        try {
            out.push_back(strToTransformType(strs[i]));
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Invalid transform at index " + std::to_string(i) +
                ": " + strs[i]
            );
        }
    }
    return out;
}

/// Convert a list of activation names to activ_func_type values.
/**
 * @param strs Vector of activation names (one per layer).
 * @return Vector of matching enum values, same order and length as @p strs.
 * @throw std::runtime_error If any entry is invalid; the message includes its index.
 */
inline std::vector<activ_func_type>
strVecToActivationTypes(const std::vector<std::string>& strs)
{
    std::vector<activ_func_type> out;
    out.reserve(strs.size());
    for (std::size_t i = 0; i < strs.size(); ++i) {
        try {
            out.push_back(strToActivation(strs[i]));
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Invalid activation at index " + std::to_string(i) +
                ": " + strs[i]
            );
        }
    }
    return out;
}


/// Training algorithm variants selectable through the RunConfig::trainer key.
/**
 * Suffix "_EPOCH" marks the variants that additionally evaluate and save
 * calibration/validation metrics every RunConfig::max_metrics_step epochs.
 */
enum class TrainerType {
    ONLINE_ADAM,              ///< Adam with per-pattern (online) updates
    BATCH_ADAM,               ///< Adam with per-batch updates
    ONLINE_ADAM_EPOCH,        ///< Online Adam with periodic epoch-level metric evaluation
    BATCH_ADAM_EPOCH,         ///< Batch Adam with periodic epoch-level metric evaluation
    ONLINE_BP,                ///< Plain backpropagation with per-pattern updates
    BATCH_BP,                 ///< Plain backpropagation with per-batch updates
    ONLINE_BP_EPOCH,          ///< Online backpropagation with periodic epoch-level metric evaluation
    BATCH_BP_EPOCH,           ///< Batch backpropagation with periodic epoch-level metric evaluation
    ONLINE_BP_MOMENTUM,       ///< Online backpropagation with a momentum term
    ONLINE_BP_PENALIZE,       ///< Online backpropagation penalizing large weights
    ONLINE_BP_MOMENTUM_EPOCH, ///< Online momentum backpropagation with epoch-level metric evaluation
    ONLINE_BP_PENALIZE_EPOCH  ///< Online penalized backpropagation with epoch-level metric evaluation
};

/// Convert a lowercase trainer string to TrainerType.
/**
 * @param s Trainer name, e.g. "online_adam", "batch_bp_epoch"
 *          (must be lowercase).
 * @return The matching enum value.
 * @throw std::invalid_argument For an unrecognized name.
 */
inline TrainerType strToTrainerType(const std::string& s) {
    if (s == "online_adam")       return TrainerType::ONLINE_ADAM;
    if (s == "batch_adam")        return TrainerType::BATCH_ADAM;
    if (s == "online_adam_epoch") return TrainerType::ONLINE_ADAM_EPOCH;
    if (s == "batch_adam_epoch")  return TrainerType::BATCH_ADAM_EPOCH;
    if (s == "online_bp")       return TrainerType::ONLINE_BP;
    if (s == "batch_bp")        return TrainerType::BATCH_BP;
    if (s == "online_bp_epoch") return TrainerType::ONLINE_BP_EPOCH;
    if (s == "batch_bp_epoch")  return TrainerType::BATCH_BP_EPOCH;
    if (s == "online_bp_momentum")       return TrainerType::ONLINE_BP_MOMENTUM;
    if (s == "online_bp_penalize")       return TrainerType::ONLINE_BP_PENALIZE;
    if (s == "online_bp_momentum_epoch")       return TrainerType::ONLINE_BP_MOMENTUM_EPOCH;
    if (s == "online_bp_penalize_epoch")       return TrainerType::ONLINE_BP_PENALIZE_EPOCH;
    throw std::invalid_argument("Unknown trainer type: " + s);
}

/// Parse an ini file into a flat key->value map.
/**
 * Blank lines and ';'/'#' comment lines are skipped, [section] headers are
 * ignored, inline comments after a value are stripped, and keys are
 * lowercased.
 * @param path Path to the ini file.
 * @return Map from lowercase key to trimmed value.
 * @throw std::runtime_error If the file cannot be opened.
 */
inline std::unordered_map<std::string, std::string> parseIniToMap(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open INI file: " + path);
    std::unordered_map<std::string,std::string> kv;
    std::string line;
    while (std::getline(ifs, line)) {
        std::string s = trimStr(line);
        if (s.empty()) continue;
        if (s[0] == ';' || s[0] == '#') continue;
        if (s.front() == '[' && s.back() == ']') continue; // skip sections
        // find '='
        auto pos = s.find('=');
        if (pos == std::string::npos) continue;
        std::string key = trimStr(s.substr(0, pos));
        std::string val = trimStr(s.substr(pos+1));
        // remove inline comment after value ( ; or # )
        size_t cpos = val.find_first_of(";#");
        if (cpos != std::string::npos) {
            val = trimStr(val.substr(0, cpos));
        }
        // lowercase key for convenience
        std::string key_lower;
        key_lower.reserve(key.size());
        for (char ch : key) key_lower.push_back(static_cast<char>(std::tolower((unsigned char)ch)));
        kv[key_lower] = val;
    }
    return kv;
}

/**
 * Convert string lists to numeric lists.
 */

/// Parse a comma-separated list into non-negative integers.
/**
 * @param s String like "8, 6, 2".
 * @return Vector of unsigned values, one per token.
 * @throw std::runtime_error On a negative or non-numeric token.
 */
inline std::vector<unsigned> parseUnsignedList(const std::string &s) {
    std::vector<unsigned> out;
    for (auto &tok : splitCommaList(s)) {
        try {
            int tmp = std::stoi(tok);
            if (tmp < 0) throw std::runtime_error("negative integer in unsigned list");
            out.push_back(static_cast<unsigned>(tmp));
        } catch (...) {
            throw std::runtime_error("parseUnsignedList: invalid integer '" + tok + "'");
        }
    }
    return out;
}

/// Parse a comma-separated list into non-negative doubles.
/**
 * @param s String like "0.5, 0.33".
 * @return Vector of double values, one per token.
 * @throw std::runtime_error On a negative or non-numeric token.
 */
inline std::vector<double> parseUnsignedListDouble(const std::string &s) {
    std::vector<double> out;
    for (auto &tok : splitCommaList(s)) {
        try {
            double tmp = std::stod(tok);
            if (tmp < 0) throw std::runtime_error("negative double in unsigned list");
            out.push_back(static_cast<double>(tmp));
        } catch (...) {
            throw std::runtime_error("parseUnsignedListDouble: invalid double '" + tok + "'");
        }
    }
    return out;
}

/// Parse one integer range token into the expanded list of offsets.
/**
 * Accepts "a" (single value), "a:b" (inclusive range, step 1), or
 * "a:b:step" (inclusive range with explicit step); descending ranges are
 * supported when a > b.
 * @param tok Token such as "-2", "-2:0", or "0:10:2".
 * @return Expanded vector of offsets.
 * @throw std::runtime_error On a malformed token or non-positive step.
 */
inline std::vector<int> parseRangeToken(const std::string& tok)
{
    auto parts = splitList(tok, ':');

    try {
        if (parts.size() == 1) {
            return { std::stoi(parts[0]) };
        }
        else if (parts.size() == 2 || parts.size() == 3) {
            int a = std::stoi(parts[0]);
            int b = std::stoi(parts[1]);
            int step = (parts.size() == 3) ? std::stoi(parts[2]) : 1;
            if (step <= 0)
                throw std::runtime_error("step must be positive");

            std::vector<int> out;
            if (a <= b) {
                for (int i = a; i <= b; i += step) out.push_back(i);
            } else {
                for (int i = a; i >= b; i -= step) out.push_back(i);
            }
            return out;
        }
    } catch (...) {
        throw std::runtime_error("Invalid range token '" + tok + "'");
    }

    throw std::runtime_error("Invalid range token '" + tok + "'");
}

/// Parse a comma-separated expression of offset ranges into a sorted, unique list.
/**
 * @param s Expression like "-2,-1,0:2".
 * @return Sorted vector of distinct time offsets.
 */
inline std::vector<int> parseOffsetExpr(const std::string& s)
{
    std::vector<int> out;

    for (auto& tok : splitList(s, ',')) {
        if (tok.empty()) continue;
        auto v = parseRangeToken(tok);
        out.insert(out.end(), v.begin(), v.end());
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());

    return out;
}

/// Parse the '|'-separated input_numbers value into one offset list per data column.
/**
 * Blocks separated by '|' correspond to the data columns; an empty block
 * yields an empty offset list for that column.
 * @param s String like "-2,-1,0 | -1,0,1,2 | | -2,-1".
 * @return Vector of offset vectors, one per column.
 * @see RunConfig::input_numbers
 */
inline std::vector<std::vector<int>>
parseOffsetLists(const std::string& s)
{
    std::vector<std::vector<int>> out;

    for (auto& block : splitList(s, '|')) {
        if (block.empty())
            out.emplace_back();
        else
            out.push_back(parseOffsetExpr(block));
    }

    return out;
}

/// Parse a comma-separated list of strings (trimmed, empty tokens dropped).
/**
 * @param s String like "P, Q, h".
 * @return Vector of trimmed tokens.
 */
inline std::vector<std::string> parseStringList(const std::string &s) {
    return splitCommaList(s);
}

/// Read an ini file and build the validated RunConfig.
/**
 * Missing keys fall back to the RunConfig default values. A single-element
 * activation/transform/transform_alpha list is broadcast to the number of
 * layers/columns respectively.
 * @param path Path to the ini configuration file.
 * @return Populated and validated RunConfig.
 * @throw std::runtime_error If the file is missing/unreadable, a required key
 *        (data_file, architecture, input_numbers) is absent, list lengths
 *        mismatch, or a value cannot be parsed.
 */
inline RunConfig parseConfigIni(const std::string &path) {
    auto kv = parseIniToMap(path);
    RunConfig cfg;

    auto get = [&](const std::string &k)->std::string {
        std::string kl = k;
        for (char &c : kl) c = static_cast<char>(std::tolower((unsigned char)c));
        auto it = kv.find(kl);
        if (it == kv.end()) return std::string{};
        return it->second;
    };

    // required/important
    std::string sdata = get("data_file");
    if (sdata.empty()) throw std::runtime_error("config: data_file is required");
    cfg.data_file = sdata;

    std::string sid = get("id");
    if (!sid.empty()) cfg.id = trimStr(sid);

    std::string sidcol = get("id_col");
    if (!sidcol.empty()) cfg.id_col = trimStr(sidcol);

    std::string stimestamp = get("timestamp");
    if (!stimestamp.empty()) cfg.timestamp = trimStr(stimestamp);

    std::string scols = get("columns");
    if (!scols.empty()) cfg.columns = parseStringList(scols);

    std::string strainer = get("trainer");
    if (!strainer.empty()) cfg.trainer = trimStr(strainer);

    std::string sarch = get("architecture");
    if (!sarch.empty()) cfg.mlp_architecture = parseUnsignedList(sarch);

    std::string sinpnums = get("input_numbers");
    if (!sinpnums.empty()) cfg.input_numbers = parseOffsetLists(sinpnums);

    std::string sact = get("activation");
    if (!sact.empty()) cfg.activation = parseStringList(sact);
    if (cfg.activation.size() == 1 && cfg.mlp_architecture.size() > 1) {
        cfg.activation.resize(cfg.mlp_architecture.size(), cfg.activation[0]);
    }

    std::string swinit = get("weight_init");
    if (!swinit.empty()) cfg.weight_init = trimStr(swinit);

    std::string sensemblerun = get("ensemble_runs");
    if (!sensemblerun.empty()) cfg.ensemble_runs = std::stoi(sensemblerun);

    std::string smaxit = get("max_iterations");
    if (!smaxit.empty()) cfg.max_iterations = std::stoi(smaxit);

    std::string smaxmetricstep = get("max_metrics_step");
    if (!smaxmetricstep.empty()) cfg.max_metrics_step = std::stoi(smaxmetricstep);

    std::string smaxerr = get("max_error");
    if (!smaxerr.empty()) cfg.max_error = std::stod(smaxerr);

    std::string slr = get("learning_rate");
    if (!slr.empty()) cfg.learning_rate = std::stod(slr);

    std::string sshuffle = get("shuffle");
    if (!sshuffle.empty()) cfg.shuffle = parseBool(sshuffle);

    std::string sseed = get("seed");
    if (!sseed.empty()) cfg.seed = static_cast<unsigned>(std::stoul(sseed));

    std::string sbsize = get("batch_size");
    if (!sbsize.empty()) cfg.batch_size = std::stoi(sbsize);

    std::string strainf = get("train_fraction");
    if (!strainf.empty()) cfg.train_fraction = std::stod(strainf);

    std::string spasttimesteps = get("lstm_past_time_steps");
    if (!spasttimesteps.empty()) cfg.lstm_past_time_steps = std::stoi(spasttimesteps);

    std::string sfuturetimesteps = get("lstm_future_time_steps");
    if (!sfuturetimesteps.empty()) cfg.lstm_future_time_steps = std::stoi(sfuturetimesteps);

    std::string scells = get("lstm_cells");
    if (!scells.empty()) cfg.lstm_cells = std::stoi(scells);

    std::string smodeltype = get("model_type");
    if (!smodeltype.empty()) cfg.model_type = trimStr(smodeltype);

    std::string ssaveepmet = get("save_epoch_metrics");
    if (!ssaveepmet.empty()) cfg.save_epoch_metrics = parseBool(ssaveepmet);

    std::string ssplitshuffle = get("split_shuffle");
    if (!ssplitshuffle.empty()) cfg.split_shuffle = parseBool(ssplitshuffle);

    std::string strans = get("transform");
    if (!strans.empty()) cfg.transform = parseStringList(strans);
    if (cfg.transform.size() == 1 && cfg.columns.size() > 1) {
        cfg.transform.resize(cfg.columns.size(), cfg.transform[0]);
    }

    std::string sa = get("transform_alpha");
    if (!sa.empty()) cfg.transform_alpha = parseUnsignedListDouble(sa);
    if (cfg.transform_alpha.size() == 1 && cfg.columns.size() > 1) {
        cfg.transform_alpha.resize(cfg.columns.size(), cfg.transform_alpha[0]);
    }

    std::string sexc = get("exclude_last_col_from_transform");
    if (!sexc.empty()) cfg.exclude_last_col_from_transform = parseBool(sexc);

    std::string sremna = get("remove_na_before_calib");
    if (!sremna.empty()) cfg.remove_na_before_calib = parseBool(sremna);

    std::string soptim = get("pso_optimize");
    if (!soptim.empty()) cfg.pso_optimize = parseBool(soptim);

    // objective function settings
    std::string sobjfunc = get("objective_function");
    if (!sobjfunc.empty()) cfg.objective_function = trimStr(sobjfunc);

    std::string sobjalpha = get("objective_alpha");
    if (!sobjalpha.empty()) cfg.objective_alpha = std::stod(sobjalpha);

    std::string sobjalpha2 = get("objective_alpha2");
    if (!sobjalpha2.empty()) cfg.objective_alpha2 = std::stod(sobjalpha2);

    // paths (optional)
    std::string sout = get("out_dir");
    if (!sout.empty()) cfg.out_dir = trimStr(sout);
    std::string slogdir = get("log_dir");
    if (!slogdir.empty()) cfg.log_dir = trimStr(slogdir);
    std::string scalib = get("calib_mat");
    if (!scalib.empty()) cfg.calib_mat = trimStr(scalib);
    std::string sinitweil = get("weights_csv_init");
    if (!sinitweil.empty()) cfg.weights_csv_init = trimStr(sinitweil);
    std::string sinitwellbin = get("weights_bin_init");
    if (!sinitwellbin.empty()) cfg.weights_bin_init = trimStr(sinitwellbin);
    std::string sinitweilvec = get("weights_vec_csv_init");
    if (!sinitweilvec.empty()) cfg.weights_vec_csv_init = trimStr(sinitweilvec);
    std::string sinitwellbinvec = get("weights_vec_bin_init");
    if (!sinitwellbinvec.empty()) cfg.weights_vec_bin_init = trimStr(sinitwellbinvec);
    std::string sweil = get("weights_csv");
    if (!sweil.empty()) cfg.weights_csv = trimStr(sweil);
    std::string swellbin = get("weights_bin");
    if (!swellbin.empty()) cfg.weights_bin = trimStr(swellbin);
    std::string sweilvec = get("weights_vec_csv");
    if (!sweilvec.empty()) cfg.weights_vec_csv = trimStr(sweilvec);
    std::string swellbinvec = get("weights_vec_bin");
    if (!swellbinvec.empty()) cfg.weights_vec_bin = trimStr(swellbinvec);
    std::string srealc = get("real_calib");
    if (!srealc.empty()) cfg.real_calib = trimStr(srealc);
    std::string spredc = get("pred_calib");
    if (!spredc.empty()) cfg.pred_calib = trimStr(spredc);
    std::string srealv = get("real_valid");
    if (!srealv.empty()) cfg.real_valid = trimStr(srealv);
    std::string spredv = get("pred_valid");
    if (!spredv.empty()) cfg.pred_valid = trimStr(spredv);
    std::string smetcal = get("metrics_cal");
    if (!smetcal.empty()) cfg.metrics_cal = trimStr(smetcal);
    std::string smetval = get("metrics_val");
    if (!smetval.empty()) cfg.metrics_val = trimStr(smetval);
    std::string sruninfo = get("run_info");
    if (!sruninfo.empty()) cfg.run_info = trimStr(sruninfo);
    std::string serrors = get("errors_csv");
    if (!serrors.empty()) cfg.errors_csv = trimStr(serrors);
    std::string sindices = get("pattern_indices");
    if (!sindices.empty()) cfg.pattern_indices = trimStr(sindices);
    
    // Basic validation
    if (cfg.mlp_architecture.empty()) throw std::runtime_error("config: architecture is required (e.g. architecture = 8,6,2)");
    if (cfg.input_numbers.empty()) throw std::runtime_error("config: input_numbers is required and length should match number of columns");

    // check that input_numbers length matches columns (if columns provided)
    if (!cfg.columns.empty() && cfg.columns.size() != cfg.input_numbers.size()) {
        throw std::runtime_error("config: columns length and input_numbers length must match");
    }

    if(cfg.columns.size() != cfg.transform_alpha.size()){
        throw std::runtime_error("config: columns length and transform_alpha length must match");
    }

    return cfg;
}


#endif // CONFIG_HPP
