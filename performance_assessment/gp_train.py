#!/usr/bin/env python
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from gollum.utils.config import instantiate_class
import torch._dynamo
import matplotlib.cm as cm
from sklearn.metrics import r2_score, mean_squared_error

torch._dynamo.config.suppress_errors = True

featurizer_configs = [
    {
        "name": "T5Base",
        "config": {
            "class_path": "gollum.data.module.Featurizer",
            "init_args": {
                "representation": "get_huggingface_embeddings",
                "model_name": "t5-base",
                "pooling_method": "average",
                "normalize_embeddings": False
            }
        }
    },
    {
        "name": "BERT",
        "config": {
            "class_path": "gollum.data.module.Featurizer",
            "init_args": {
                "representation": "get_huggingface_embeddings",
                "model_name": "bert-base-uncased",
                "pooling_method": "average",
                "normalize_embeddings": False
            }
        }
    },
    {
        "name": "SciBERT",
        "config": {
            "class_path": "gollum.data.module.Featurizer",
            "init_args": {
                "representation": "get_huggingface_embeddings",
                "model_name": "allenai/scibert_scivocab_uncased",
                "pooling_method": "average",
                "normalize_embeddings": False
            }
        }
    },
    {
        "name": "ChemBERTa-MTR",
        "config": {
            "class_path": "gollum.data.module.Featurizer",
            "init_args": {
                "representation": "get_huggingface_embeddings",
                "model_name": "DeepChem/ChemBERTa-77M-MTR",
                "pooling_method": "average",
                "normalize_embeddings": False
            }
        }
    },
    {
        "name": "ModernBERT",
        "config": {
            "class_path": "gollum.data.module.Featurizer",
            "init_args": {
                "representation": "get_huggingface_embeddings",
                "model_name": "answerdotai/ModernBERT-base",
                "pooling_method": "cls",
                "normalize_embeddings": False
            }
        }
    },
    {
        "name": "CheModernBERT",
        "config": {
            "class_path": "gollum.data.module.Featurizer",
            "init_args": {
                "representation": "get_huggingface_embeddings",
                "model_name": "/home/david/modernbert_chemistry/fineweb/fine-web-modernbert-base-8192-multi-tok-new-procedure-60-epochs-notags",
                "pooling_method": "cls",
                "normalize_embeddings": False
            }
        }
    }
]

gp_regression_config = {
    "class_path": "gollum.surrogate_models.gp.GP",
    "init_args": {
        "standardize": True,
        "normalize": False,
        "initial_noise_val": 1e-4,
        "noise_constraint": 1e-5,
        "initial_outputscale_val": 1.0,
        "initial_lengthscale_val": 1.0,
        "gp_lr": 0.02
    }
}

# Regression: continuous target
# DATA_PATH     = "/home/david/modernbert_chemistry/fineweb/dataset/folder_data22/Properties of Monomers.csv"
# INPUT_COLUMN  = "SMILES"
# TARGET_COLUMN = "E_coh (MPa)"  

# DATA_PATH = "/home/david/modernbert_chemistry/fineweb/dataset/folder_data22/Properties of Monomers.csv"
# INPUT_COLUMN = "SMILES"
# TARGET_COLUMN = "T_g (K)"

# DATA_PATH = "/home/david/modernbert_chemistry/fineweb/dataset/folder_data22/Properties of Monomers.csv"
# INPUT_COLUMN = "SMILES"
# TARGET_COLUMN = "Densities (kg/m^3)"

DATA_PATH = "/home/david/modernbert_chemistry/fineweb/dataset/folder_data22/Properties of Monomers.csv"
INPUT_COLUMN = "SMILES"
TARGET_COLUMN = "R_gyr (A^2)"

n_bins         = 10
data_regimes   = [50, 100, 150, None]   
num_iterations = 10

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded with {len(df)} samples.")

inputs  = df[INPUT_COLUMN].tolist()
targets = df[TARGET_COLUMN].values.astype(np.float64)

indices = np.arange(len(inputs))
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
y_train_full = targets[train_idx]
y_test_np    = targets[test_idx]

n_test                = len(y_test_np)
theoretical_quantiles = stats.norm.ppf((np.arange(1, n_test + 1) - 0.5) / n_test)
nominal_levels        = np.linspace(0.5, 0.99, n_bins)

embeddings = {}
for fe in featurizer_configs:
    name = fe["name"]
    print(f"\n=== Computing embeddings with {name} ===")
    featurizer = instantiate_class(fe["config"])
    E = featurizer.featurize(inputs)
    if not isinstance(E, torch.Tensor):
        E = torch.tensor(E, dtype=torch.float64)
    embeddings[name] = E

calibration_results = { regime: {} for regime in data_regimes }
performance_results  = { regime: {} for regime in data_regimes }

n_feats       = len(featurizer_configs)
plasma_r_cmap = cm.get_cmap("plasma_r")
palette_plasma_r = [
    plasma_r_cmap(0.1 + 0.9 * i / (n_feats - 1))
    for i in range(n_feats)
]

LABEL_FONTSIZE  = 13
TICK_FONTSIZE   = 13
TITLE_FONTSIZE  = 13
LEGEND_FONTSIZE = 13

for regime in data_regimes:
    actual_n = (len(train_idx) if regime is None else regime)
    print(f"\nData regime: {actual_n} samples")
    all_idxs = []
    n_train  = len(train_idx)
    for _ in range(num_iterations):
        perm = torch.randperm(n_train)
        all_idxs.append(None if regime is None else perm[:regime])

    for fe in featurizer_configs:
        name = fe["name"]
        print(f"\nFeaturizer: {name} | Regime = {actual_n}")

        E_full      = embeddings[name]
        X_train_all = E_full[train_idx]
        X_test      = E_full[test_idx]

        qq_list       = []
        coverage_list = []

        for it, idxs in enumerate(all_idxs):
            print(f"  Iteration {it+1}/{num_iterations}", end="")
            if idxs is None:
                X_sub = X_train_all
                y_sub = y_train_full
            else:
                X_sub = X_train_all[idxs]
                y_sub = y_train_full[idxs.cpu().numpy()]
            Xc = X_sub.to("cuda")
            yc = torch.tensor(y_sub, dtype=torch.float64).unsqueeze(-1).to("cuda")
            cfg = copy.deepcopy(gp_regression_config)
            cfg["init_args"]["train_x"] = Xc
            cfg["init_args"]["train_y"] = yc
            gp_model = instantiate_class(cfg).to("cuda")

            gp_model.fit()

            with torch.no_grad():
                pred_mean, pred_var = gp_model.predict(X_test.to("cuda"))
                pred_mean = pred_mean.cpu().numpy().flatten()
                pred_std  = torch.sqrt(pred_var).cpu().numpy().flatten()

            std_res    = (y_test_np - pred_mean) / pred_std
            sorted_res = np.sort(std_res)
            qq_list.append(sorted_res)

            cov_iter = []
            for level in nominal_levels:
                z  = stats.norm.ppf(0.5 + level / 2)
                lb = pred_mean - z * pred_std
                ub = pred_mean + z * pred_std
                cov_iter.append(np.mean((y_test_np >= lb) & (y_test_np <= ub)))
            coverage_list.append(cov_iter)

            print("  → done")

        avg_qq  = np.mean(np.vstack(qq_list), axis=0)
        avg_cov = np.mean(np.vstack(coverage_list), axis=0)
        calib_error = np.mean(np.abs(avg_cov - nominal_levels))

        calibration_results[regime][name] = {
            "qq": avg_qq,
            "coverage": avg_cov
        }
        performance_results[regime][name] = {
            "calib_error": calib_error
        }

fig, axs = plt.subplots(
    2, len(data_regimes),
    figsize=(4 * len(data_regimes), 8),
    sharex=False, sharey=False
)
for col, regime in enumerate(data_regimes):
    actual_n = (len(train_idx) if regime is None else regime)
    ax_qq  = axs[0, col]
    ax_cov = axs[1, col]

    ax_qq.plot(theoretical_quantiles, theoretical_quantiles, "k--", lw=1)
    ax_cov.plot(nominal_levels, nominal_levels, "k--", lw=1)

    for j, fe in enumerate(featurizer_configs):
        name    = fe["name"]
        color   = palette_plasma_r[j]
        avg_qq  = calibration_results[regime][name]["qq"]
        avg_cov = calibration_results[regime][name]["coverage"]

        ax_qq.plot(
            theoretical_quantiles, avg_qq,
            marker="o", linestyle="-",
            markersize=4, linewidth=0.5,
            color=color, label=name
        )
        ax_cov.plot(
            nominal_levels, avg_cov,
            marker="o", linestyle="-",
            markersize=4, linewidth=2,
            color=color
        )

    ax_qq.set_title(f"PIT Q–Q (n={actual_n})", fontsize=TITLE_FONTSIZE)
    ax_qq.set_xlabel("Theoretical Quantiles", fontsize=LABEL_FONTSIZE)
    ax_qq.set_ylabel("Avg. Ordered Residuals", fontsize=LABEL_FONTSIZE)
    ax_qq.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    ax_cov.set_title(f"Calibration (n={actual_n})", fontsize=TITLE_FONTSIZE)
    ax_cov.set_xlabel("Nominal Level", fontsize=LABEL_FONTSIZE)
    ax_cov.set_ylabel("Empirical Coverage", fontsize=LABEL_FONTSIZE)
    ax_cov.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    if col == 0:
        ax_qq.legend(fontsize=LEGEND_FONTSIZE - 2, labelspacing=0.2, loc="upper left")

plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.5, bottom=0.35)
plt.savefig(
    "/home/david/gollum_modernbert/gollum/results/new_results/GPREG_FORREG_CALIB_R_gyr.pdf",
    dpi=300
)
plt.show()

for regime in data_regimes:
    actual_n = (len(train_idx) if regime is None else regime)
    print(f"\n===== Computing R² & MSE for regime: {actual_n} =====")

    all_idxs = []
    n_train  = len(train_idx)
    for _ in range(num_iterations):
        perm = torch.randperm(n_train)
        all_idxs.append(None if regime is None else perm[:regime])

    for fe in featurizer_configs:
        name = fe["name"]
        print(f"--- Featurizer: {name} | Regime = {actual_n} ---")

        E_full      = embeddings[name]
        X_train_all = E_full[train_idx]
        X_test      = E_full[test_idx]

        r2_list  = []
        mse_list = []

        for it, idxs in enumerate(all_idxs):
            print(f"  Iteration {it+1}/{num_iterations}", end="")

            if idxs is None:
                X_sub = X_train_all
                y_sub = y_train_full
            else:
                X_sub = X_train_all[idxs]
                y_sub = y_train_full[idxs.cpu().numpy()]

            Xc = X_sub.to("cuda")
            yc = torch.tensor(y_sub, dtype=torch.float64).unsqueeze(-1).to("cuda")

            cfg = copy.deepcopy(gp_regression_config)
            cfg["init_args"]["train_x"] = Xc
            cfg["init_args"]["train_y"] = yc
            gp_model = instantiate_class(cfg).to("cuda")
            gp_model.fit()

            with torch.no_grad():
                pred_mean, _ = gp_model.predict(X_test.to("cuda"))
                pred_mean = pred_mean.cpu().numpy().flatten()

            
            r2_list.append(r2_score(y_test_np, pred_mean))
       
            mse_list.append(mean_squared_error(y_test_np, pred_mean))

            print(f"  → R²={r2_list[-1]:.4f}  MSE={mse_list[-1]:.4f}")

        avg_r2 = np.mean(r2_list)
        std_r2 = np.std(r2_list)
        avg_mse = np.mean(mse_list)
        std_mse = np.std(mse_list)

        performance_results[regime][name]["r2_mean"]  = avg_r2
        performance_results[regime][name]["r2_std"]   = std_r2
        performance_results[regime][name]["mse_mean"] = avg_mse
        performance_results[regime][name]["mse_std"]  = std_mse

fig2, axs2 = plt.subplots(
    2, len(data_regimes),
    figsize=(4 * len(data_regimes), 8),
    sharex=False, sharey=False
)
axs2 = np.atleast_2d(axs2)

for col, regime in enumerate(data_regimes):
    actual_n = (len(train_idx) if regime is None else regime)

    ax_r2 = axs2[0, col]
    feats    = list(performance_results[regime].keys())
    r2_vals  = [performance_results[regime][f]["r2_mean"] for f in feats]
    r2_stds  = [performance_results[regime][f]["r2_std"]  for f in feats]
    x_pos    = np.arange(len(feats))

    best_idx   = np.argmax(r2_vals)
    bar_colors = ["salmon"] * len(feats)
    bar_colors[best_idx] = "gold"

    lower_err = np.array([min(std, abs(m)) for m, std in zip(r2_vals, r2_stds)])
    upper_err = np.array(r2_stds)
    yerr      = [lower_err, upper_err]

    ax_r2.bar(
        x_pos, r2_vals,
        yerr=yerr, capsize=5,
        color=bar_colors
    )
    ax_r2.set_title(f"R² (n={actual_n})", fontsize=TITLE_FONTSIZE)
    ax_r2.set_ylabel("R² Score", fontsize=LABEL_FONTSIZE)
    ax_r2.set_xticks(x_pos)
    ax_r2.set_xticklabels(feats, rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    ax_r2.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    # Bottom row: MSE
    ax_mse = axs2[1, col]
    mse_vals = [performance_results[regime][f]["mse_mean"] for f in feats]
    mse_stds = [performance_results[regime][f]["mse_std"]  for f in feats]

    best_idx_mse = np.argmin(mse_vals)
    bar_colors_mse = ["salmon"] * len(feats)
    bar_colors_mse[best_idx_mse] = "gold"

    lower_err_mse = np.array([min(std, abs(m)) for m, std in zip(mse_vals, mse_stds)])
    upper_err_mse = np.array(mse_stds)
    yerr_mse      = [lower_err_mse, upper_err_mse]

    ax_mse.bar(
        x_pos, mse_vals,
        yerr=yerr_mse, capsize=5,
        color=bar_colors_mse
    )
    ax_mse.set_title(f"MSE (n={actual_n})", fontsize=TITLE_FONTSIZE)
    ax_mse.set_ylabel("Mean Squared Error", fontsize=LABEL_FONTSIZE)
    ax_mse.set_xticks(x_pos)
    ax_mse.set_xticklabels(feats, rotation=45, ha="right", fontsize=TICK_FONTSIZE)
    ax_mse.tick_params(axis="both", labelsize=TICK_FONTSIZE)

plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=1.0, bottom=0.35)
plt.savefig(
    "/home/david/gollum_modernbert/gollum/results/new_results/GPREG_FORREG_R2_MSE_R_gyr.pdf",
    dpi=300
)
plt.show()

rows = []
for regime in data_regimes:
    actual_n = (len(train_idx) if regime is None else regime)
    for name in calibration_results[regime].keys():
        r2_mean   = performance_results[regime][name]["r2_mean"]
        r2_std    = performance_results[regime][name]["r2_std"]
        mse_mean  = performance_results[regime][name]["mse_mean"]
        mse_std   = performance_results[regime][name]["mse_std"]
        calib_err = performance_results[regime][name]["calib_error"]
        rows.append({
            "regime": actual_n,
            "featurizer": name,
            "R2_mean": r2_mean,
            "R2_std": r2_std,
            "MSE_mean": mse_mean,
            "MSE_std": mse_std,
            "calibration_error": calib_err
        })

df_scores = pd.DataFrame(rows)
csv_path = "/home/david/gollum_modernbert/gollum/results/new_results/GPREG_FORREG_CALIB_R_gyr_scores.csv"
df_scores.to_csv(csv_path, index=False)
print(f"Saved calibration & performance scores (including R² and MSE) to {csv_path}")
