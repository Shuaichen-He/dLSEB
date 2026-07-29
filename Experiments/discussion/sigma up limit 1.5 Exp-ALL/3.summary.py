"""
实验结果汇总可视化 — RMSE, Correlation, Std Ratio & Bias bar / line charts.

Layout (3×2 grid per figure, 3 sites grouped per experiment):
  RMSE:      (a) Rsu  |  (b) Rlu  |  (c) H
             (d) G6   |  (e) T2cm |  (f) T10cm
  Cor:       (a) Rsu  |  (b) Rlu  |  (c) H
             (d) G6   |  (e) T2cm |  (f) T10cm
  Std Ratio: (a) Rsu  |  (b) Rlu  |  (c) H
             (d) G6   |  (e) T2cm |  (f) T10cm
  Bias:      (a) Rsu  |  (b) Rlu  |  (c) H
             (d) G6   |  (e) T2cm |  (f) T10cm

Three grey shades represent Huazhaizi, Ejin, Shenshawo.

Usage:
    python 3.summary.py                     # RMSE & Cor bar charts
    python 3.summary.py --lines             # RMSE & Cor line charts
    python 3.summary.py --std-bias          # RMSE, Cor, Std Ratio & Bias bars
    python 3.summary.py --lines --all       # all four line charts
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

# ── Path resolution ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..",".."))
DATA_BASE = os.path.join(REPO_ROOT, "data", "2.data_selection")
PARAMS_DIR = os.path.join(REPO_ROOT, "data", "3.estimate_parameters")
EST_DIR = os.path.join(SCRIPT_DIR, "0.EST")

EXP_MAP = {"EST": "0.EST", "ALL": "1.ALL", "RSL": "2.RSL", "RHS": "3.RHS", "RHT": "4.RHT"}

# ── Matplotlib style (unified with other experiments) ──
plt.style.use("seaborn-v0_8-white")
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 18,
    "axes.linewidth": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.fontsize": 16,
    "legend.title_fontsize": 20,
    "savefig.dpi": 300,
})

# ── Constants ──
SITES = ["Huazhaizi", "Ejin", "Shenshawo"]
EXPS = ["EST", "RSL", "RHS", "RHT", "ALL"]  # ALL last for progressive readability
VARS = ["Rsu", "Rlu", "H", "G6", "T2cm", "T10cm"]


# =============================================================================
# Parse RMSE_Cor_Param.log
# =============================================================================
def _parse_rmse_cor_log(log_path):
    """Parse RMSE_Cor.log → (rmse_dict, cor_dict).

    Each dict: {site: {exp: [6 values]}} following VARS order.
    """
    rmse = {}
    cor = {}
    current_site = None
    in_rmse_cor_section = False

    with open(log_path, "r") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("Site:"):
                current_site = stripped.split(":")[1].strip()
                in_rmse_cor_section = False
                continue

            if current_site is None:
                continue

            if "RMSE & Correlation" in stripped:
                in_rmse_cor_section = True
                continue

            if not in_rmse_cor_section:
                continue

            if "|" in stripped and not stripped.startswith("---"):
                parts = stripped.split("|")
                if len(parts) != 3:
                    continue

                exp_name = parts[0].strip()
                if exp_name not in EXPS:
                    continue

                rmse_vals = [float(v) for v in parts[1].strip().split()]
                cor_vals = [float(v) for v in parts[2].strip().split()]

                if current_site not in rmse:
                    rmse[current_site] = {}
                    cor[current_site] = {}

                rmse[current_site][exp_name] = rmse_vals
                cor[current_site][exp_name] = cor_vals

    return rmse, cor


RMSE, COR = _parse_rmse_cor_log(os.path.join(SCRIPT_DIR, "RMSE_Cor_Param.log"))

# ── Site colours & markers ──
SITE_COLORS = ["#1A1A1A", "#7A7A7A", "#CCCCCC"]  # dark → medium → light
SITE_LABELS = ["Huazhaizi", "Ejin", "Shenshawo"]
SITE_MARKERS = ["o", "s", "^"]  # circle, square, triangle

# ── Variable display labels ──
VAR_LABELS = [
    r"$R_{su}$",
    r"$R_{lu}$",
    r"$H$",
    r"$G_{6cm}$",
    r"$T_{2cm}$",
    r"$T_{10cm}$",
]

RMSE_YLABELS = [
    r"RMSE($R_{su}$, $\hat{R}_{su}$) (W m$^{-2}$)",
    r"RMSE($R_{lu}$, $\hat{R}_{lu}$) (W m$^{-2}$)",
    r"RMSE($H$, $\hat{H}$) (W m$^{-2}$)",
    r"RMSE($G_{6cm}$, $\hat{G}_{6cm}$) (W m$^{-2}$)",
    r"RMSE($T_{2cm}$, $\hat{T}_{2cm}$) (K)",
    r"RMSE($T_{10cm}$, $\hat{T}_{10cm}$) (K)",
]

COR_YLABELS = [
    r"Cor($R_{su}$, $\hat{R}_{su}$)",
    r"Cor($R_{lu}$, $\hat{R}_{lu}$)",
    r"Cor($H$, $\hat{H}$)",
    r"Cor($G_{6cm}$, $\hat{G}_{6cm}$)",
    r"Cor($T_{2cm}$, $\hat{T}_{2cm}$)",
    r"Cor($T_{10cm}$, $\hat{T}_{10cm}$)",
]

STD_YLABELS = [
    r"Std Ratio ($\sigma_{\hat{R}_{su}}/\sigma_{R_{su}}$)",
    r"Std Ratio ($\sigma_{\hat{R}_{lu}}/\sigma_{R_{lu}}$)",
    r"Std Ratio ($\sigma_{\hat{H}}/\sigma_{H}$)",
    r"Std Ratio ($\sigma_{\hat{G}_{6cm}}/\sigma_{G_{6cm}}$)",
    r"Std Ratio ($\sigma_{\hat{T}_{2cm}}/\sigma_{T_{2cm}}$)",
    r"Std Ratio ($\sigma_{\hat{T}_{10cm}}/\sigma_{T_{10cm}}$)",
]

BIAS_YLABELS = [
    r"Bias (W m$^{-2}$)", r"Bias (W m$^{-2}$)", r"Bias (W m$^{-2}$)",
    r"Bias (W m$^{-2}$)", r"Bias (K)", r"Bias (K)",
]


# =============================================================================
# Auto-determined Y-limits
# =============================================================================
def _compute_ylim(data_dict, pad=0.12):
    """Compute per-var y-limits with proportional padding."""
    lims = []
    for vi in range(6):
        vals = []
        for site in SITES:
            for exp in EXPS:
                vals.append(data_dict[site][exp][vi])
        lo, hi = np.min(vals), np.max(vals)
        span = hi - lo
        lims.append((lo - pad * span, hi + pad * span))
    return lims


# =============================================================================
# Data loaders for Std Ratio & Bias computation
# =============================================================================
def load_obs_data(site_name):
    """Load observations for a given site (mirrors 2.batch_visualization.py)."""
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
    return {"Rsu": Rsu, "Rlu": Rlu, "Hs": Hs, "G6": G6, "Train_T": Train_T}


def load_sim_data(exp_mode, site_name):
    """Load simulation output (EB_result, T_result, k) for a given (exp, site)."""
    exp_dir = os.path.join(SCRIPT_DIR, EXP_MAP[exp_mode])

    if exp_mode == "EST":
        eb = np.load(os.path.join(exp_dir, f"EB_result_{site_name}.npy"))
        T = np.load(os.path.join(exp_dir, f"T_result_{site_name}.npy"))
        k_path = os.path.join(PARAMS_DIR, f"{site_name}.npy")
        k = float(np.load(k_path, allow_pickle=True).item()["k"])
    else:
        site_dir = os.path.join(exp_dir, site_name)
        eb = np.load(os.path.join(site_dir, "EB_result.npy"))
        T = np.load(os.path.join(site_dir, "T_result.npy"))
        params_hat = np.load(os.path.join(site_dir, "params_hat_values.npy"))
        k = float(params_hat[-1, 3])

    return {"EB_result": eb, "T_result": T, "k": k}


def _extract_sim_vars(eb, T, k):
    """Extract predicting variables from simulation outputs."""
    r_s, r_l, h, G_out = eb
    G6 = k * (T[:, 2] - T[:, 1]) / 0.06
    return {
        "Rsu": np.array(r_s),
        "Rlu": np.array(r_l),
        "H": np.array(h),
        "G6": G6,
        "T2cm": T[:, 0],
        "T10cm": T[:, 2],
    }


def _extract_obs_vars(obs):
    """Extract observation variables matching simulation order."""
    return {
        "Rsu": obs["Rsu"],
        "Rlu": obs["Rlu"],
        "H": obs["Hs"],
        "G6": obs["G6"],
        "T2cm": obs["Train_T"][:, 0],
        "T10cm": obs["Train_T"][:, 2],
    }


def compute_std_bias():
    """Compute Std Ratio & Bias for all (site, exp, var) from npy files."""
    std_ratio = {site: {} for site in SITES}
    bias = {site: {} for site in SITES}
    obs_cache = {}

    for site in SITES:
        if site not in obs_cache:
            obs_cache[site] = load_obs_data(site)
        obs_data = obs_cache[site]
        obs_vars = _extract_obs_vars(obs_data)

        for exp in EXPS:
            sim_data = load_sim_data(exp, site)
            sim_vars = _extract_sim_vars(
                sim_data["EB_result"], sim_data["T_result"], sim_data["k"]
            )

            vals_std, vals_bias = [], []
            for var in VARS:
                o = obs_vars[var]
                s = sim_vars[var]
                vals_std.append(np.std(s) / np.std(o))
                vals_bias.append(np.mean(s) - np.mean(o))

            std_ratio[site][exp] = vals_std
            bias[site][exp] = vals_bias

        print(f"  [✓] {site} — std ratio & bias computed")

    return std_ratio, bias


# =============================================================================
# Shared y-axis logic
# =============================================================================
def _apply_ylim(ax, vi, data_dict, ylims, ylim_cap, hline_y, legend_headroom):
    """Apply unified or per-variable y-axis limits to a subplot."""
    if ylim_cap is not None:
        all_vals = []
        for _vi in range(6):
            for site in SITES:
                for exp in EXPS:
                    all_vals.append(data_dict[site][exp][_vi])
        glo = np.min(all_vals)
        _ylo, _yhi = glo, ylim_cap * 1.06
        if hline_y is not None:
            _ylo = min(_ylo, hline_y)
            _yhi = max(_yhi, hline_y)
        if vi == 0:
            _yhi = ylim_cap + legend_headroom * (ylim_cap - glo)
        ax.set_ylim(_ylo, _yhi)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
        ticks = [t for t in ax.get_yticks()
                 if t <= ylim_cap + 1e-8 and t >= _ylo - 1e-8]
        ax.set_yticks(ticks)
    else:
        ylo, yhi = ylims[vi]
        if hline_y is not None:
            ylo = min(ylo, hline_y)
            yhi = max(yhi, hline_y)
        if vi == 0:
            span = yhi - ylo
            yhi = yhi + legend_headroom * span
        ax.set_ylim(ylo, yhi)


# =============================================================================
# Grouped bar chart function
# =============================================================================
def plot_grouped_bars(data_dict, ylabels, ylims, titles, outpath,
                      data_label="RMSE", hline_y=None, ylim_cap=None,
                      legend_headroom=0.40):
    """3×2 grouped bar charts with 3 sites per experiment group."""
    n_exps = len(EXPS)
    n_sites = len(SITES)
    bar_width = 0.16
    x = np.arange(n_exps)
    offsets = np.linspace(-bar_width, bar_width, n_sites)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=300)
    axes = axes.flatten()

    for vi, ax in enumerate(axes):
        for si, site in enumerate(SITES):
            vals = [data_dict[site][exp][vi] for exp in EXPS]
            ax.bar(
                x + offsets[si], vals, bar_width,
                color=SITE_COLORS[si],
                edgecolor="black",
                alpha=0.85,
                linewidth=1.2,
                label=SITE_LABELS[si] if vi == 0 else "",
            )

        if hline_y is not None:
            ax.axhline(y=hline_y, color="black", linewidth=2.0, zorder=3)

        ax.set_title(titles[vi], fontsize=20, loc="left", pad=8,
                     fontweight="normal")
        ax.set_ylabel(ylabels[vi], fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(EXPS, fontsize=18)

        _apply_ylim(ax, vi, data_dict, ylims, ylim_cap, hline_y, legend_headroom)
        ax.tick_params(axis="both", labelsize=18)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels, loc="upper left", ncol=1,
        frameon=False, fontsize=16,
    )

    for j in range(len(VARS), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 1])
    plt.savefig(outpath, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {data_label} bar chart saved → {outpath}")


# =============================================================================
# Grouped line chart function
# =============================================================================
def plot_grouped_lines(data_dict, ylabels, ylims, titles, outpath,
                       data_label="RMSE", hline_y=None, ylim_cap=None,
                       legend_headroom=0.40):
    """3×2 grouped line charts with 3 sites as separate line series."""
    n_exps = len(EXPS)
    x = np.arange(n_exps)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=300)
    axes = axes.flatten()

    for vi, ax in enumerate(axes):
        for si, site in enumerate(SITES):
            vals = [data_dict[site][exp][vi] for exp in EXPS]
            ax.plot(
                x, vals,
                color=SITE_COLORS[si],
                marker=SITE_MARKERS[si],
                markerfacecolor="white" if si == 0 else SITE_COLORS[si],
                markeredgecolor=SITE_COLORS[si],
                markeredgewidth=1.5,
                markersize=10,
                linewidth=2.0,
                alpha=0.85,
                label=SITE_LABELS[si] if vi == 0 else "",
            )

        if hline_y is not None:
            ax.axhline(y=hline_y, color="black", linewidth=2.0, zorder=3)

        ax.set_title(titles[vi], fontsize=20, loc="left", pad=8,
                     fontweight="normal")
        ax.set_ylabel(ylabels[vi], fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(EXPS, fontsize=18)

        _apply_ylim(ax, vi, data_dict, ylims, ylim_cap, hline_y, legend_headroom)
        ax.tick_params(axis="both", labelsize=18)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels, loc="upper left", ncol=1,
        frameon=False, fontsize=16,
    )

    for j in range(len(VARS), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 1])
    plt.savefig(outpath, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {data_label} line chart saved → {outpath}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summary chart visualization for dLSEB experiments."
    )
    parser.add_argument(
        "--std-bias", "--all",
        dest="compute_std_bias",
        action="store_true",
        help="Also compute and plot Std Ratio & Bias from npy experiment files.",
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="Generate line charts instead of bar charts (default: bar charts).",
    )
    args = parser.parse_args()

    chart_kind = "lines" if args.lines else "bars"
    plot_fn = plot_grouped_lines if args.lines else plot_grouped_bars

    # ── Titles ──
    rmse_titles = [f"({c}) {v}" for c, v in zip("abcdef", VAR_LABELS)]
    cor_titles = [f"({c}) {v}" for c, v in zip("abcdef", VAR_LABELS)]
    std_titles = [f"({c}) {v}" for c, v in zip("abcdef", VAR_LABELS)]
    bias_titles = [f"({c}) {v}" for c, v in zip("abcdef", VAR_LABELS)]

    # ── Y-limits ──
    RMSE_YLIMS = _compute_ylim(RMSE, pad=0.10)
    COR_YLIMS = _compute_ylim(COR, pad=0.10)

    print("=" * 60)
    print(f"Generating {chart_kind} charts...")

    # ── Plot RMSE ──
    plot_fn(
        RMSE, RMSE_YLABELS, RMSE_YLIMS, rmse_titles,
        os.path.join(SCRIPT_DIR, f"rmse_{chart_kind}.svg"), data_label="RMSE",
    )

    # ── Plot Correlation ──
    plot_fn(
        COR, COR_YLABELS, COR_YLIMS, cor_titles,
        os.path.join(SCRIPT_DIR, f"cor_{chart_kind}.svg"), data_label="Correlation",
        ylim_cap=1.0, legend_headroom=0.60,
    )

    # ── Optionally: Std Ratio & Bias ──
    if args.compute_std_bias:
        print("=" * 60)
        print("Computing Std Ratio & Bias from experiment npy files...")
        std_ratio, bias = compute_std_bias()

        STD_YLIMS = _compute_ylim(std_ratio, pad=0.10)
        BIAS_YLIMS = _compute_ylim(bias, pad=0.10)

        plot_fn(
            std_ratio, STD_YLABELS, STD_YLIMS, std_titles,
            os.path.join(SCRIPT_DIR, f"std_{chart_kind}.svg"), data_label="Std Ratio",
        )
        plot_fn(
            bias, BIAS_YLABELS, BIAS_YLIMS, bias_titles,
            os.path.join(SCRIPT_DIR, f"bias_{chart_kind}.svg"), data_label="Bias",
            hline_y=0, legend_headroom=0.60,
        )

    print("=" * 60)
    print(f"All {chart_kind} charts generated.")
