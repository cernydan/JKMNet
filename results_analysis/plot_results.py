# === IMPORT LIBRARIES ===
import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIG ===
base_dir = os.path.join("..", "data", "outputs")
calib_real_file = os.path.join(base_dir, "calib_real.csv")
calib_pred_file = os.path.join(base_dir, "calib_pred.csv")
valid_real_file = os.path.join(base_dir, "valid_real.csv")
valid_pred_file = os.path.join(base_dir, "valid_pred.csv")
out_dir = "plot_results_compare"  # this will be created inside results_analysis/


# === MAKE OUTPUT DIR ===
os.makedirs(out_dir, exist_ok=True)

# === LOAD DATA (numeric only) ===
calib_real = pd.read_csv(calib_real_file, header=None).apply(pd.to_numeric, errors="coerce")
calib_pred = pd.read_csv(calib_pred_file, header=None).apply(pd.to_numeric, errors="coerce")
valid_real = pd.read_csv(valid_real_file, header=None).apply(pd.to_numeric, errors="coerce")
valid_pred = pd.read_csv(valid_pred_file, header=None).apply(pd.to_numeric, errors="coerce")

# shape checks
if calib_real.shape != calib_pred.shape:
    raise ValueError(f"Calibration mismatch: real={calib_real.shape}, pred={calib_pred.shape}")
if valid_real.shape != valid_pred.shape:
    raise ValueError(f"Validation mismatch: real={valid_real.shape}, pred={valid_pred.shape}")
if calib_real.shape[1] != valid_real.shape[1]:
    raise ValueError("Calibration and validation have different number of output columns")

n_outputs = calib_real.shape[1]
print(f"Detected {n_outputs} output columns")

# === LOOP OVER OUTPUT COLUMNS ===
for i in range(n_outputs):
    cr, cp = calib_real.iloc[:, i], calib_pred.iloc[:, i]
    vr, vp = valid_real.iloc[:, i], valid_pred.iloc[:, i]

    # --- TIME SERIES PLOTS (side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    axes[0].plot(cr.values, label="Measured", marker="o", markersize=3, linestyle="-")
    axes[0].plot(cp.values, label="Predicted", marker="x", markersize=3, linestyle="--")
    axes[0].set_title(f"Calibration (col {i})")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Value")
    axes[0].legend()  # legend only on calibration

    axes[1].plot(vr.values, marker="o", markersize=3, linestyle="-")
    axes[1].plot(vp.values, marker="x", markersize=3, linestyle="--")
    axes[1].set_title(f"Validation (col {i})")
    axes[1].set_xlabel("Sample index")

    plt.suptitle(f"Time Series: Measured vs Predicted (col {i})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"timeseries_col{i}.png"))
    plt.close()

    # --- SCATTER PLOTS (side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    min_val = min(cr.min(), cp.min(), vr.min(), vp.min())
    max_val = max(cr.max(), cp.max(), vr.max(), vp.max())

    axes[0].scatter(cr, cp, alpha=0.7, label="Points")
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 line")
    axes[0].set_title(f"Calibration (col {i})")
    axes[0].set_xlabel("Measured")
    axes[0].set_ylabel("Predicted")
    axes[0].legend()  # legend only on calibration

    axes[1].scatter(vr, vp, alpha=0.7)
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--")
    axes[1].set_title(f"Validation (col {i})")
    axes[1].set_xlabel("Measured")

    plt.suptitle(f"Scatter: Measured vs Predicted (col {i})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"scatter_col{i}.png"))
    plt.close()

    # --- RESIDUAL PLOTS (side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    res_cal = cr - cp
    res_val = vr - vp

    axes[0].scatter(cr, res_cal, alpha=0.7, label="Residuals")
    axes[0].axhline(0, color="red", linestyle="--", label="Zero line")
    axes[0].set_title(f"Calibration (col {i})")
    axes[0].set_xlabel("Measured")
    axes[0].set_ylabel("Residual (Measured - Predicted)")
    axes[0].legend()  # legend only on calibration

    axes[1].scatter(vr, res_val, alpha=0.7)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title(f"Validation (col {i})")
    axes[1].set_xlabel("Measured")

    plt.suptitle(f"Residuals (col {i})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"residuals_col{i}.png"))
    plt.close()

# === HISTOGRAMS OF RESIDUALS PER COLUMN ===
residuals_calib = [calib_real.iloc[:, i] - calib_pred.iloc[:, i] for i in range(n_outputs)]
residuals_valid = [valid_real.iloc[:, i] - valid_pred.iloc[:, i] for i in range(n_outputs)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Calibration residuals
axes[0].hist(residuals_calib, bins=30, stacked=True, label=[f"Col {i+1}" for i in range(n_outputs)], alpha=0.7)
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_title("Calibration residuals distribution")
axes[0].set_xlabel("Residual (Measured - Predicted)")
axes[0].set_ylabel("Frequency")
axes[0].legend()  # legend only on calibration

# Validation residuals
axes[1].hist(residuals_valid, bins=30, stacked=True, label=[f"Col {i+1}" for i in range(n_outputs)], alpha=0.7)
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Validation residuals distribution")
axes[1].set_xlabel("Residual (Measured - Predicted)")

plt.suptitle("Residual histograms per output column")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "residuals_hist.png"))
plt.close()

print("Calibration and validation plots saved in:", out_dir)
