"""
Batch Visualization & Metrics for dLSEB Parameter Optimization Experiments
===========================================================================
For each (experiment, site) combination, generates:
  1. losses.{svg,pdf}  — loss & parameter evolution across epochs (in svg/ & pdf/)
  2. model_output.{svg,pdf} — energy-balance time-series (obs vs EST vs optimised)
  3. metrics.npy  — RMSE & correlation coefficients for each variable

Optionally, also computes EST vs obs metrics (est_metrics_{site}.npy).

Usage
-----
All experiments, all sites:
    python "2.batch_visualization.py"

Single experiment, single site:
    python "2.batch_visualization.py" --exp ALL --site Huazhaizi

Single experiment, all sites:
    python "2.batch_visualization.py" --exp RSL --all

EST metrics only:
    python "2.batch_visualization.py" --est-only

Skip certain plots:
    python "2.batch_visualization.py" --no-loss --no-energy
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
# (ScalarFormatter not needed — ticklabel_format handles sci notation)

# ── Path resolution ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_BASE = os.path.join(REPO_ROOT, "data", "2.data_selection")
PARAMS_DIR = os.path.join(REPO_ROOT, "data", "3.estimate_parameters")
EST_DIR = os.path.join(SCRIPT_DIR, "0.EST")

SITES = ["Huazhaizi", "Ejin", "Shenshawo"]
EXP_MAP = {"ALL": "1.ALL", "RSL": "2.RSL", "RHS": "3.RHS", "RHT": "4.RHT"}
VALID_EXPS = ["RSL", "RHS", "RHT", "ALL"]  # ALL last for progressive readability

# ── Matplotlib defaults ──────────────────────────────────────────────────────
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 15,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "legend.title_fontsize": 20,
    "savefig.dpi": 300,
})


# =============================================================================
# Data loaders
# =============================================================================
def load_obs_data(site_name):
    """Load observations for a given site.

    Returns dict with: Rsu, Rlu, Hs, G, G6, Train_T, T_sfc
    """
    site_dir = os.path.join(DATA_BASE, site_name)

    data = pd.read_csv(os.path.join(site_dir, "meteo_var.csv"))
    flux = pd.read_csv(os.path.join(site_dir, "ec_var.csv"))
    soil_data = pd.read_csv(os.path.join(site_dir, "soil.csv"))

    Rsu = np.where(
        data["UR"].astype(float).values > 0,
        data["UR"].astype(float).values,
        0.0,
    )
    Rlu = data["ULR_Cor"].astype(float).values
    Hs = flux["Hs"].astype(float).values

    G = -1 * soil_data["Gs_1"].astype(float).values
    G6 = -1 * np.array([
        soil_data["Gs_1"].astype(float).values,
        soil_data["Gs_2"].astype(float).values,
        soil_data["Gs_3"].astype(float).values,
    ]).mean(axis=0)

    Train_T = (
        np.array([
            soil_data["Ts_2cm"].astype(float).values,
            soil_data["Ts_4cm"].astype(float).values,
            soil_data["Ts_10cm"].astype(float).values,
            soil_data["Ts_20cm"].astype(float).values,
        ]).T
        + 273.15
    )

    T_sfc = soil_data["Ts_0cm"].astype(float).values + 273.15

    return {
        "Rsu": Rsu, "Rlu": Rlu, "Hs": Hs, "G": G, "G6": G6,
        "Train_T": Train_T, "T_sfc": T_sfc,
    }


def load_experiment_data(exp_mode, site_name):
    """Load optimized experiment results.

    Returns dict with:
        params_hat, T_loss, EB_result (r_s, r_l, h, G_out), T_result
    """
    exp_dir = os.path.join(SCRIPT_DIR, EXP_MAP[exp_mode])
    site_dir = os.path.join(exp_dir, site_name)

    params_hat = np.load(os.path.join(site_dir, "params_hat_values.npy"))
    T_loss = np.load(os.path.join(site_dir, "T_loss_values.npy"))
    EB_result = np.load(os.path.join(site_dir, "EB_result.npy"))
    T_result = np.load(os.path.join(site_dir, "T_result.npy"))

    return {
        "params_hat": params_hat,
        "T_loss": T_loss,
        "EB_result": EB_result,
        "T_result": T_result,
    }


def load_est_data(site_name):
    """Load EST (estimated, not optimized) simulation results."""
    eb = np.load(os.path.join(EST_DIR, f"EB_result_{site_name}.npy"))
    T = np.load(os.path.join(EST_DIR, f"T_result_{site_name}.npy"))
    return {"EB_result": eb, "T_result": T}


def load_est_k(site_name):
    """Load the k parameter used in EST run."""
    npy_path = os.path.join(PARAMS_DIR, f"{site_name}.npy")
    params_dict = np.load(npy_path, allow_pickle=True).item()
    return float(params_dict["k"])


def load_est_params(site_name):
    """Load all EST parameters for a site: (alpha, sigma, z0m, k, C)."""
    npy_path = os.path.join(PARAMS_DIR, f"{site_name}.npy")
    params_dict = np.load(npy_path, allow_pickle=True).item()
    return (
        float(params_dict["alpha"]),
        float(params_dict["sigma"]),
        float(params_dict["z0m"]),
        float(params_dict["k"]),
        float(params_dict["C"]),
    )


def load_site_date_range(site_name):
    """Parse final_periods.log to get the date range for a site.

    Returns
    -------
    xtick_labels : list of str  — "MM-DD" labels for each of the 10 days
    xlabel : str               — e.g. "Date (2024-03-26 – 2024-04-04)"
    """
    log_path = os.path.join(DATA_BASE, "final_periods.log")
    with open(log_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == site_name:
                start_str = parts[2]
                end_str = parts[3]
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
                n_days = (end_date - start_date).days + 1
                dates = [start_date + timedelta(days=i) for i in range(n_days)]
                xtick_labels = [d.strftime("%m-%d") for d in dates]
                xlabel = f"Date ({start_date.year})"
                return xtick_labels, xlabel
    # fallback
    return [str(i) for i in range(1, 11)], "Date"


# =============================================================================
# Metrics
# =============================================================================
def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def compute_metrics(obs, pred_dict, prefix=""):
    """Compute RMSE & correlation for all variables.

    Parameters
    ----------
    obs : dict  — observation arrays
    pred_dict : dict — predicted arrays {"Rsu": ..., "Rlu": ..., ...}
    prefix : str — identifier for the prediction source (e.g. "EST_")

    Returns
    -------
    dict with keys like: RMSE_Rsu, RMSE_Rlu, ..., Corr_Rsu, Corr_Rlu, ...
    """
    metrics = {}
    # Variables: (obs_key, pred_key, label)
    vars_ = [
        ("Rsu", "Rsu", "Rsu"), ("Rlu", "Rlu", "Rlu"),
        ("Hs", "H", "H"), ("G6", "G6", "G6"),
        ("Train_T", "T2cm", "T2cm"), ("Train_T", "T10cm", "T10cm"),
    ]
    for obs_key, pred_key, label in vars_:
        if obs_key == "Train_T":
            o = obs[obs_key][:, 0] if label == "T2cm" else obs[obs_key][:, 2]
        else:
            o = obs[obs_key]
        p = pred_dict[pred_key]
        metrics[f"RMSE_{prefix}{label}"] = rmse(o, p)
        metrics[f"Corr_{prefix}{label}"] = corr(o, p)
    return metrics


# =============================================================================
# Figure 1: Loss & parameter evolution (plot_loss_multi)
# =============================================================================
def plot_loss_multi(out_dir, exp_name, site_name, T_loss, params_hat):
    """6-panel figure: loss + 5 parameters across epochs."""
    plt.style.use("seaborn-v0_8-white")
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 15,
        "axes.linewidth": 1.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "legend.fontsize": 18,
        "legend.title_fontsize": 20, "savefig.dpi": 300,
    })

    alpha_hat = params_hat[:, 0]
    sigma_hat = params_hat[:, 1]
    z0m_hat = params_hat[:, 2]
    k_hat = params_hat[:, 3]
    C_hat = np.exp(params_hat[:, 4])  # log → linear

    data_list = [T_loss, alpha_hat, sigma_hat, z0m_hat, k_hat, C_hat]
    labels = [
        f"{exp_name} Losses",
        r"$\hat{\alpha}$",
        r"$\hat{\sigma}$",
        r"$\hat{z}_{0m}$ (m)",
        r"$\hat{k}$ (W m$^{-1}$ K$^{-1}$)",
        r"$\hat{C}$ (J m$^{-3}$ K$^{-1}$)",
    ]
    nums = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    sci_flags = [False, False, False, True, False, True]

    n_epochs = len(T_loss)
    x = np.arange(n_epochs)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()

    for i, (data, label, num, sci) in enumerate(
        zip(data_list, labels, nums, sci_flags)
    ):
        ax = axes[i]
        ax.plot(x, data, color="black", linewidth=2)
        if sci:
            ax.ticklabel_format(
                style="sci", axis="y", scilimits=(-1, 0), useMathText=True
            )
            ax.yaxis.get_offset_text().set_fontsize(15)
            ax.yaxis.get_offset_text().set_position((-0.15, 0.8))
        ax.set_ylabel(label)
        ax.set_title(num, fontsize=20, pad=10, loc="left")
        ax.set_xlim(0, n_epochs - 1)

    for j in range(len(data_list), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    # Save to both svg/ and pdf/
    svg_dir = os.path.join(out_dir, "svg")
    pdf_dir = os.path.join(out_dir, "pdf")
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    plt.savefig(os.path.join(svg_dir, "losses.svg"), format="svg", dpi=300)
    print(f"    [✓] svg/losses.svg saved")
    plt.savefig(os.path.join(pdf_dir, "losses.pdf"), format="pdf", dpi=300)
    print(f"    [✓] pdf/losses.pdf saved")
    plt.close(fig)


# =============================================================================
# Figure 2: Energy balance time-series (combined_line_plots)
# =============================================================================
def plot_line_subplot(ax, obs, opt, est, title, ylabel, color_gray, color2, color3,
                      xtick_labels=None):
    """Single subplot: observations + optimized + EST."""
    ax.plot(
        obs, color=color_gray, marker="o", markerfacecolor="none",
        linestyle="None", markersize=5, label="Obs",
    )
    ax.plot(est, color=color3, linewidth=2, label="EST")
    ax.plot(opt, color=color2, linewidth=2, label="OPT")

    ax.tick_params(axis="y", labelsize=16, labelcolor="black")
    ax.set_ylabel(ylabel, color="#000000")
    ax.set_xticks(np.arange(1, 11) * 24 * 2 - 24 * 1)
    if xtick_labels is None:
        xtick_labels = [str(i) for i in range(1, 11)]
    ax.set_xticklabels(xtick_labels, fontsize=10, rotation=30, ha="right")
    ax.set_title(title, fontsize=20, loc="left", pad=8)

    # Adaptive y-axis with margin
    ax.relim()
    ax.autoscale_view()
    ax.margins(y=0.12)


def plot_energy_balance(out_dir, obs, opt_data, est_data, k_opt, k_est,
                        site_name="Huazhaizi"):
    """6-panel time-series: Rsu, Rlu, H, G6, T2cm, T10cm."""
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 15,
        "axes.linewidth": 1.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 8, "ytick.major.size": 8,
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "legend.fontsize": 12,
        "savefig.dpi": 300,
    })

    # Load site-specific date labels from final_periods.log
    xtick_labels, xlabel = load_site_date_range(site_name)

    r_s, r_l, h, G_out = opt_data["EB_result"]
    c_r_s, c_r_l, c_H, c_G = est_data["EB_result"]
    T_opt = opt_data["T_result"]
    T_est = est_data["T_result"]

    # G6 at 6cm: k * ΔT / 0.06  (use all 480 time steps)
    G6_opt = k_opt * (T_opt[:, 2] - T_opt[:, 1]) / 0.06
    G6_est = k_est * (T_est[:, 2] - T_est[:, 1]) / 0.06

    color_gray = "dimgray"
    color2 = "#FF4B4BAF"
    color3 = "#3256B3AF"

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), dpi=300)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, bottom=0.08)

    subplot_info = [
        {
            "obs": obs["Rsu"], "opt": np.array(r_s), "est": c_r_s,
            "title": r"(a) R$_{su}$", "ylabel": r"W m$^{-2}$",
        },
        {
            "obs": obs["Rlu"], "opt": np.array(r_l), "est": c_r_l,
            "title": r"(b) R$_{lu}$", "ylabel": r"W m$^{-2}$",
        },
        {
            "obs": obs["Hs"], "opt": h, "est": c_H,
            "title": r"(c) H", "ylabel": r"W m$^{-2}$",
        },
        {
            "obs": obs["G6"], "opt": G6_opt, "est": G6_est,
            "title": r"(d) G$_{6cm}$", "ylabel": r"W m$^{-2}$",
        },
        {
            "obs": obs["Train_T"][:, 0], "opt": T_opt[:, 0], "est": T_est[:, 0],
            "title": r"(e) $T_{2cm}$", "ylabel": r"K",
        },
        {
            "obs": obs["Train_T"][:, 2], "opt": T_opt[:, 2], "est": T_est[:, 2],
            "title": r"(f) $T_{10cm}$", "ylabel": r"K",
        },
    ]

    for i in range(3):
        for j in range(2):
            idx = i * 2 + j
            if idx >= len(subplot_info):
                continue
            info = subplot_info[idx]
            plot_line_subplot(
                axes[i, j], info["obs"], info["opt"], info["est"],
                info["title"], info["ylabel"],
                color_gray, color2, color3,
                xtick_labels=xtick_labels if i == 2 else [""] * 10,
            )
            if i == 2:
                axes[i, j].set_xlabel(xlabel)

    axes[2, 1].legend(loc="lower right", ncol=3, frameon=False)
    plt.tight_layout(pad=1.0, rect=[0, 0, 1, 1])

    # Save to both svg/ and pdf/
    svg_dir = os.path.join(out_dir, "svg")
    pdf_dir = os.path.join(out_dir, "pdf")
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    plt.savefig(os.path.join(svg_dir, "model_output.svg"), format="svg", dpi=300)
    print(f"    [✓] svg/model_output.svg saved")
    plt.savefig(os.path.join(pdf_dir, "model_output.pdf"), format="pdf", dpi=300)
    print(f"    [✓] pdf/model_output.pdf saved")
    plt.close(fig)


# =============================================================================
# Compute & save metrics
# =============================================================================
def save_metrics(obs, opt_data, est_data, k_opt, k_est, save_dir, exp_name):
    """Compute metrics for both EST and OPT vs observations, save as .npy."""
    r_s, r_l, h, G_out = opt_data["EB_result"]
    c_r_s, c_r_l, c_H, c_G = est_data["EB_result"]
    T_opt = opt_data["T_result"]
    T_est = est_data["T_result"]

    G6_opt = k_opt * (T_opt[:, 2] - T_opt[:, 1]) / 0.06
    G6_est = k_est * (T_est[:, 2] - T_est[:, 1]) / 0.06

    opt_pred = {
        "Rsu": np.array(r_s), "Rlu": np.array(r_l), "H": h,
        "G6": G6_opt, "T2cm": T_opt[:, 0], "T10cm": T_opt[:, 2],
    }
    est_pred = {
        "Rsu": c_r_s, "Rlu": c_r_l, "H": c_H,
        "G6": G6_est, "T2cm": T_est[:, 0], "T10cm": T_est[:, 2],
    }

    opt_metrics = compute_metrics(obs, opt_pred, prefix="OPT_")
    est_metrics = compute_metrics(obs, est_pred, prefix="EST_")

    all_metrics = {**opt_metrics, **est_metrics}

    metrics_path = os.path.join(save_dir, f"metrics_{exp_name}.npy")
    np.save(metrics_path, all_metrics)
    print(f"    [✓] metrics_{exp_name}.npy saved")

    return all_metrics


def save_est_metrics_only(site_name):
    """Compute and save EST vs obs metrics (standalone)."""
    obs = load_obs_data(site_name)
    est_data = load_est_data(site_name)
    k_est = load_est_k(site_name)

    c_r_s, c_r_l, c_H, c_G = est_data["EB_result"]
    T_est = est_data["T_result"]
    G6_est = k_est * (T_est[:, 2] - T_est[:, 1]) / 0.06

    est_pred = {
        "Rsu": c_r_s, "Rlu": c_r_l, "H": c_H,
        "G6": G6_est, "T2cm": T_est[:, 0], "T10cm": T_est[:, 2],
    }
    metrics = compute_metrics(obs, est_pred, prefix="EST_")

    save_path = os.path.join(EST_DIR, f"est_metrics_{site_name}.npy")
    np.save(save_path, metrics)
    print(f"  [✓] est_metrics_{site_name}.npy saved")
    return metrics


def print_metrics_table(metrics, title):
    """Pretty-print a metrics dictionary."""
    print(f"\n  {title}")
    print(f"  {'Variable':<8s} {'RMSE':>12s}  {'Corr':>8s}")
    print(f"  {'-'*8} {'-'*12}  {'-'*8}")
    for var in ["Rsu", "Rlu", "H", "G6", "T2cm", "T10cm"]:
        rmse_key = [k for k in metrics if k.startswith("RMSE_") and k.endswith(var)]
        corr_key = [k for k in metrics if k.startswith("Corr_") and k.endswith(var)]
        rmse_val = metrics[rmse_key[0]] if rmse_key else float("nan")
        corr_val = metrics[corr_key[0]] if corr_key else float("nan")
        print(f"  {var:<8s} {rmse_val:>12.3f}  {corr_val:>8.4f}")


# =============================================================================
# Main runner
# =============================================================================
def run_visualization(site_name, exp_mode, plot_loss=True, plot_energy=True,
                      compute_m=True):
    """Run all visualizations & metrics for one (site, experiment) pair."""
    exp_dir = os.path.join(SCRIPT_DIR, EXP_MAP[exp_mode])
    site_out_dir = os.path.join(exp_dir, site_name)

    print(f"\n  [{exp_mode}] {site_name}")
    print(f"  {'-'*40}")

    # Load observation & experiment data (shared across plots)
    obs = load_obs_data(site_name)
    opt_data = load_experiment_data(exp_mode, site_name)
    est_data = load_est_data(site_name)
    k_opt = float(opt_data["params_hat"][-1, 3])
    k_est = load_est_k(site_name)

    if plot_loss:
        plot_loss_multi(
            site_out_dir,
            exp_mode, site_name,
            opt_data["T_loss"], opt_data["params_hat"],
        )

    if plot_energy:
        plot_energy_balance(
            site_out_dir,
            obs, opt_data, est_data, k_opt, k_est,
            site_name=site_name,
        )

    opt_metrics = None
    est_metrics = None
    if compute_m:
        all_m = save_metrics(
            obs, opt_data, est_data, k_opt, k_est, site_out_dir, exp_mode,
        )
        opt_metrics = {k: v for k, v in all_m.items() if "OPT_" in k}
        est_metrics = {k: v for k, v in all_m.items() if "EST_" in k}
        print_metrics_table(opt_metrics, "OPT Metrics")
        print_metrics_table(est_metrics, "EST Metrics")

    # Final optimized parameters (last row of params_hat)
    final_p = opt_data["params_hat"][-1, :]
    final_params = (
        float(final_p[0]),         # alpha
        float(final_p[1]),         # sigma
        float(final_p[2]),         # z0m
        float(final_p[3]),         # k
        float(np.exp(final_p[4])), # C (log → linear)
    )

    return site_name, exp_mode, opt_metrics, est_metrics, final_params


# =============================================================================
# Summary: RMSE_Cor_Param.log
# =============================================================================
def write_rmse_cor_log(all_opt, all_est, all_params, sites, experiments):
    """Write a formatted RMSE & Correlation summary to RMSE_Cor_Param.log.

    For each site, prints:
      1. Parameter Results per experiment (α, σ, z₀ₘ, k, C, λ=k/C)
      2. RMSE & Correlation per experiment

    Parameters
    ----------
    all_opt : dict  — {(site, exp): metrics_dict} for optimized experiments
    all_est : dict  — {site: metrics_dict} for EST (no optimization)
    all_params : dict — {(site, exp): (α, σ, z₀ₘ, k, C)} for each experiment
    sites : list
    experiments : list
    """
    vars_ = ["Rsu", "Rlu", "H", "G6", "T2cm", "T10cm"]
    log_path = os.path.join(SCRIPT_DIR, "RMSE_Cor_Param.log")

    lines = []
    lines.append("RMSE & Correlation Summary")
    lines.append("=" * 100)

    for site in sites:
        lines.append(f"\nSite: {site}")
        lines.append("-" * 100)

        # ── 1. Parameter Results ──────────────────────────────────────
        lines.append("  Parameter Results")
        lines.append(f"  {'Exp':<6s} |  {'α':>8s}  {'σ':>8s}  {'z₀ₘ':>10s}  {'k':>8s}  {'C':>12s}  {'λ':>12s}")
        lines.append(f"  {'-'*6}-+-{'-'*64}")

        # EST params row
        est_params = load_est_params(site)
        est_lambda = est_params[3] / est_params[4]  # k / C
        lines.append(
            f"  {'EST':<6s} |  {est_params[0]:>8.4f}  {est_params[1]:>8.4f}  "
            f"{est_params[2]:>10.2e}  {est_params[3]:>8.4f}  {est_params[4]:>12.2e}  "
            f"{est_lambda:>12.2e}"
        )

        # Experiment params rows
        for exp in experiments:
            key = (site, exp)
            if key not in all_params:
                continue
            p = all_params[key]
            p_lambda = p[3] / p[4]  # k / C
            lines.append(
                f"  {exp:<6s} |  {p[0]:>8.4f}  {p[1]:>8.4f}  "
                f"{p[2]:>10.2e}  {p[3]:>8.4f}  {p[4]:>12.2e}  "
                f"{p_lambda:>12.2e}"
            )

        # ── 2. Metrics ─────────────────────────────────────────────────
        lines.append(f"\n  RMSE & Correlation")
        hdr = f"  {'Exp':<6s} |"
        hdr += " " + "RMSE".center(7 * len(vars_) + len(vars_) - 1)
        hdr += " | " + "Corr".center(7 * len(vars_) + len(vars_) - 1)
        lines.append(hdr)

        sub_hdr = f"  {'':<6s} |"
        sub_hdr += " " + " ".join(f"{v:>6s}" for v in vars_)
        sub_hdr += " | " + " ".join(f"{v:>6s}" for v in vars_)
        lines.append(sub_hdr)
        lines.append(f"  {'-'*6}-+-{'-'*46}-+-{'-'*46}")

        # EST row
        if site in all_est:
            est_m = all_est[site]
            row = f"  {'EST':<6s} |"
            row_rmse = " ".join(
                f"{est_m.get(f'RMSE_EST_{v}', float('nan')):>6.2f}" for v in vars_
            )
            row_corr = " ".join(
                f"{est_m.get(f'Corr_EST_{v}', float('nan')):>6.3f}" for v in vars_
            )
            row += " " + row_rmse + " | " + row_corr
            lines.append(row)

        # Experiment rows
        for exp in experiments:
            key = (site, exp)
            if key not in all_opt:
                continue
            opt_m = all_opt[key]
            row = f"  {exp:<6s} |"
            row_rmse = " ".join(
                f"{opt_m.get(f'RMSE_OPT_{v}', float('nan')):>6.2f}" for v in vars_
            )
            row_corr = " ".join(
                f"{opt_m.get(f'Corr_OPT_{v}', float('nan')):>6.3f}" for v in vars_
            )
            row += " " + row_rmse + " | " + row_corr
            lines.append(row)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[✓] RMSE_Cor_Param.log written → {log_path}")


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch visualization & metrics for dLSEB experiments."
    )
    parser.add_argument(
        "--exp", nargs="+", default=VALID_EXPS, choices=VALID_EXPS,
        help="Experiment(s) to process (default: all 4)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--site", type=str, help=f"Single site: {', '.join(SITES)}")
    group.add_argument("--all", action="store_true",
                       help="Process all 3 sites")
    parser.add_argument("--est-only", action="store_true",
                        help="Only compute EST vs obs metrics, skip experiments")
    parser.add_argument("--no-loss", action="store_true",
                        help="Skip loss/parameter evolution plot")
    parser.add_argument("--no-energy", action="store_true",
                        help="Skip energy-balance time-series plot")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Skip RMSE/correlation computation")

    args = parser.parse_args()

    # ── EST-only mode ────────────────────────────────────────────────────
    if args.est_only:
        print("EST vs Observations Metrics")
        print("=" * 40)
        est_sites = SITES if args.all else ([args.site] if args.site else SITES)
        est_all = {}
        for s in est_sites:
            m = save_est_metrics_only(s)
            est_all[s] = m
            print_metrics_table(m, f"EST — {s}")
        write_rmse_cor_log({}, est_all, {}, est_sites, [])
        sys.exit(0)

    # ── Experiment visualisation mode ────────────────────────────────────
    sites = SITES if (args.all or args.site is None) else ([args.site] if args.site in SITES else [])
    if not sites:
        print(f"Error: unknown site '{args.site}'. Choose from {SITES}")
        sys.exit(1)

    experiments = args.exp

    all_opt = {}    # {(site, exp): opt_metrics}
    all_est = {}    # {site: est_metrics}
    all_params = {} # {(site, exp): (α, σ, z₀ₘ, k, C)}

    total = len(sites) * len(experiments)
    count = 0
    for site in sites:
        # Compute EST metrics once per site
        if not args.no_metrics and site not in all_est:
            obs = load_obs_data(site)
            est_data = load_est_data(site)
            k_est = load_est_k(site)
            c_r_s, c_r_l, c_H, c_G = est_data["EB_result"]
            T_est = est_data["T_result"]
            G6_est = k_est * (T_est[:, 2] - T_est[:, 1]) / 0.06
            est_pred = {
                "Rsu": c_r_s, "Rlu": c_r_l, "H": c_H,
                "G6": G6_est, "T2cm": T_est[:, 0], "T10cm": T_est[:, 2],
            }
            est_metrics = compute_metrics(obs, est_pred, prefix="EST_")
            all_est[site] = est_metrics

        for exp in experiments:
            count += 1
            print(f"\n{'~'*50}")
            print(f"  [{count}/{total}]  {site} / {exp}")
            print(f"{'~'*50}")
            _, _, opt_m, _, final_params = run_visualization(
                site, exp,
                plot_loss=not args.no_loss,
                plot_energy=not args.no_energy,
                compute_m=not args.no_metrics,
            )
            if opt_m:
                all_opt[(site, exp)] = opt_m
            all_params[(site, exp)] = final_params

    # Write summary log
    if not args.no_metrics:
        write_rmse_cor_log(all_opt, all_est, all_params, sites, experiments)

    print(f"\n{'='*50}")
    print(f"  All visualizations complete.")
    print(f"{'='*50}")
