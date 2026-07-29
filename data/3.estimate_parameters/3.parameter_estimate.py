"""
Parameter estimation for land surface model (LSM) energy balance closure.

Estimated parameters:
  a) Surface shortwave albedo                — α   (alpha)
  b) Surface longwave emissivity              — σ   (sigma)
  c) Surface momentum roughness length        — z₀ₘ (z0m)
  d) Soil thermal conductivity                — k
  (+) Soil thermal diffusivity                 — λ   (lambda)
  (+) Volumetric soil heat capacity            — C

Usage
-----
Single site:
    python "3.parameter_estimate.py" --site Huazhaizi

Batch (all 3 sites):
    python "3.parameter_estimate.py" --all
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize, curve_fit
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
import seaborn as sns

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# Paths & site configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "2.data_selection"))
OUTPUT_BASE = SCRIPT_DIR

SITES = ["Huazhaizi", "Ejin", "Shenshawo"]

# ============================================================
# Physical constants
# ============================================================
SIGMA0 = 5.670374419e-8   # Stefan-Boltzmann constant (W·m⁻²·K⁻⁴)
KAM = 0.4                 # von Kármán constant
G = 9.81                  # gravitational acceleration (m·s⁻²)
RHO = 1.225               # air density (kg·m⁻³)
CP = 1004.0               # specific heat capacity of air (J·kg⁻¹·K⁻¹)
Z_ATM = 4.5               # measurement height (m)
D_PLANE = 0               # zero-plane displacement height (m)

# Soil layer depths (m)
SOIL_DEPTHS = np.array([0.0, 0.02, 0.04, 0.1, 0.2, 0.4, 0.6, 1.0])
SOIL_DZ_6CM = 0.06        # spacing between 4 cm and 10 cm sensors (m)

# Sinusoidal fitting
OMEGA = 2 * np.pi / 86400  # Earth rotation angular frequency (rad/s)


# ============================================================
# Plot style
# ============================================================
def set_style():
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 15,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "legend.fontsize": 15,
        "legend.title_fontsize": 20,
        "savefig.dpi": 300,
    })


# ============================================================
# Data loader
# ============================================================
def load_data(data_dir):
    """Load all required CSV files from a site's data directory."""
    meteo = pd.read_csv(os.path.join(data_dir, "meteo_var.csv"))
    soil = pd.read_csv(os.path.join(data_dir, "soil.csv"))
    flux = pd.read_csv(os.path.join(data_dir, "ec_var.csv"))
    return meteo, soil, flux


# ============================================================
# a) Surface albedo  α
# ============================================================
def estimate_albedo(meteo):
    """Estimate surface shortwave albedo (α) from daytime radiation.

    Returns
    -------
    alpha : float
        Overall mean albedo α = ΣR_su / ΣR_sd.
    info : dict
        hours, hourly_mean_albedo, albedo_daily, sun_rsu, sun_rsd
    """
    Rsd = meteo["DR"].astype(float).values  # downward shortwave
    Rsu = meteo["UR"].astype(float).values  # upward shortwave

    # Extract 10 days of daytime data (08:00–18:00, 2 points per hour)
    sun_rsu, sun_rsd = [], []
    for i in range(10):
        start = 2 * 8 + i * 2 * 24
        end = start + 2 * 11
        sun_rsu.append(Rsu[start:end])
        sun_rsd.append(Rsd[start:end])
    sun_rsu = np.array(sun_rsu)
    sun_rsd = np.array(sun_rsd)

    albedo = sun_rsu / sun_rsd
    alpha = np.sum(sun_rsu) / np.sum(sun_rsd)

    hours = np.arange(8, 19)
    hourly_mean = []
    for i in range(len(hours)):
        hour_data = albedo[:, i * 2:(i + 1) * 2].flatten()
        hourly_mean.append(np.mean(hour_data))

    info = {
        "hours": hours,
        "hourly_mean": hourly_mean,
        "albedo_daily": albedo,
        "sun_rsu": sun_rsu,
        "sun_rsd": sun_rsd,
    }
    return alpha, info


# ============================================================
# b) Surface emissivity  σ
# ============================================================
def estimate_emissivity(meteo, soil):
    """Estimate surface longwave emissivity (σ) via regression through origin.

    Model:  R_lu↑ − R_ld↓ = σ (σ₀ T_s⁴ − R_ld↓)

    Returns
    -------
    sigma : float
    r2 : float
    info : dict (x, y, T_sfc_K)
    """
    Rlu = meteo["ULR_Cor"].astype(float).values
    Rld = meteo["DLR_Cor"].astype(float).values
    T_sfc = soil["Ts_0cm"].astype(float).values + 273.15

    y = Rlu - Rld
    x = SIGMA0 * T_sfc ** 4 - Rld

    model = LinearRegression(fit_intercept=False)
    model.fit(x.reshape(-1, 1), y)
    sigma = model.coef_[0]
    r2 = model.score(x.reshape(-1, 1), y)

    info = {"x": x, "y": y, "T_sfc_K": T_sfc, "Rlu": Rlu, "Rld": Rld,
            "model": model}
    return sigma, r2, info


# ============================================================
# c) Surface roughness length  z₀ₘ
# ============================================================
def _psi_m(zeta):
    """Businger-Dyer stability function for momentum (unstable)."""
    x = (1 - 16 * zeta) ** 0.25
    return (2 * np.log((1 + x) / 2)
            + np.log((1 + x ** 2) / 2)
            - 2 * np.arctan(x)
            + np.pi / 2)


def _dlg_m(zeta, z0m, L):
    """Dimensionless wind gradient integrated over layer z0m → z_atm."""
    if zeta < -1.574:
        return (np.log(-1.574 * L / z0m)
                - _psi_m(-1.574)
                + 1.14 * ((-zeta) ** (1 / 3) - 1.574 ** (1 / 3))
                + _psi_m(z0m / L))
    elif zeta < 0:
        return np.log((Z_ATM - D_PLANE) / z0m) - _psi_m(zeta) + _psi_m(z0m / L)
    elif zeta <= 1:
        return np.log((Z_ATM - D_PLANE) / z0m) + 5 * zeta - 5 * z0m / L
    else:
        return np.log(L / z0m) + 5 + 5 * np.log(zeta) + zeta - 1 - 5 * z0m / L


def _solve_z0m_single(u_obs, u_star, L):
    """Solve z₀ₘ for a single time step via MOST inversion."""
    zeta = Z_ATM / L

    def objective(log_z0):
        z0 = np.exp(log_z0)
        u_pred = (u_star / KAM) * _dlg_m(zeta, z0, L)
        return (u_pred - u_obs) ** 2

    res = minimize(objective, x0=np.log(0.01),
                   bounds=[(np.log(1e-7), np.log(1e5))])
    return np.exp(res.x[0])


def estimate_z0m(flux):
    """Estimate surface momentum roughness length (z₀ₘ) via MOST.

    Returns
    -------
    z0m_kde_max : float
        z₀ₘ at the maximum of the KDE (ln-space).
    z0m_array : ndarray (cleaned)
    info : dict
    """
    u = flux["Wnd"].astype(float).values
    u_star = flux["Ustar"].astype(float).values
    L_vals = flux["L"].astype(float).values

    z0m_store = []
    for i in range(u.shape[0]):
        z0m_pre = _solve_z0m_single(u[i], u_star[i], L_vals[i])
        z0m_store.append(z0m_pre)

    # Clean outliers
    x = np.array(np.log(z0m_store))
    x = np.where(x < -15, np.nan, x)
    x = np.where(x > 1e5, np.nan, x)
    x_clean = x[~np.isnan(x)]

    kde = gaussian_kde(x_clean)
    x_grid = np.linspace(x_clean.min(), x_clean.max(), len(x_clean))
    density = kde(x_grid)
    kde_max_log = x_grid[np.argmax(density)]
    z0m_max = np.exp(kde_max_log)

    info = {
        "log_z0m_clean": x_clean,
        "kde_x": x_grid,
        "kde_density": density,
        "kde_max_log": kde_max_log,
    }
    return z0m_max, np.exp(x_clean), info


# ============================================================
# d) Soil thermal conductivity  k
# ============================================================
def estimate_soil_k(soil):
    """Estimate soil thermal conductivity (k) via regression through origin.

    Model:  −G₆cm = k · (ΔT / Δz)

    Returns
    -------
    k : float
        Soil thermal conductivity (W·m⁻¹·K⁻¹).
    r2 : float
    info : dict
    """
    Ts_4 = soil["Ts_4cm"].astype(float).values
    Ts_10 = soil["Ts_10cm"].astype(float).values

    Gs = np.mean([
        soil["Gs_1"].astype(float).values,
        soil["Gs_2"].astype(float).values,
        soil["Gs_3"].astype(float).values,
    ], axis=0)

    y = -Gs                                    # upward positive
    x = (Ts_10 - Ts_4) / SOIL_DZ_6CM           # K/m  (np.diff convention)

    model = LinearRegression(fit_intercept=False)
    model.fit(x.reshape(-1, 1), y)
    k_val = model.coef_[0]
    r2 = model.score(x.reshape(-1, 1), y)

    info = {"x": x, "y": y, "Gs_mean": Gs, "Ts_4": Ts_4, "Ts_10": Ts_10,
            "model": model}
    return k_val, r2, info


# ============================================================
# Thermal diffusivity  λ  & heat capacity  C
# ============================================================
def _sin_func(t, T_mean, A, beta):
    """Sinusoidal temperature model."""
    return T_mean + A * np.sin(OMEGA * t + beta)


def estimate_thermal_diffusivity(soil):
    """Estimate soil thermal diffusivity (λ) via amplitude & phase methods.

    Returns
    -------
    lmt_A : ndarray (7,)
        λ from amplitude damping between consecutive layers.
    lmt_phi : ndarray (7,)
        λ from phase lag between consecutive layers.
    fit_T : ndarray (8, 480)
        Fitted sinusoidal soil temperatures.
    info : dict
    """
    # Build soil temperature array (8 layers × N time points)
    T_cols = ["Ts_0cm", "Ts_2cm", "Ts_4cm", "Ts_10cm",
              "Ts_20cm", "Ts_40cm", "Ts_60cm", "Ts_100cm"]
    soil_T = np.array(
        [soil[col].astype(float).values for col in T_cols]
    ) + 273.15

    n_layers, n_pts = soil_T.shape
    xdata = np.linspace(600, 86400 * 10, n_pts)

    from scipy.signal import detrend as _detrend

    A_store, beta_store, popt_store = [], [], []
    for i in range(n_layers):
        popt, _ = curve_fit(
            _sin_func,
            xdata,
            _detrend(soil_T[i, :], type="linear"),
            bounds=([-1, 0, -np.pi], [1, 30, np.pi]),
            method="trf",
        )
        A_store.append(popt[1])
        beta_store.append(popt[2])
        popt_store.append(popt)

    # λ from amplitude ratio
    def _lam_A(A1, A2, z1, z2, omega):
        return omega / 2 * ((z2 - z1) / np.log(A1 / A2)) ** 2

    # λ from phase lag
    def _lam_phi(phi1, phi2, z1, z2, omega):
        return (omega / 2) * ((z1 - z2) / (phi2 - phi1)) ** 2

    lmt_A, lmt_phi = [], []
    for i in range(n_layers - 1):
        lmt_A.append(_lam_A(A_store[i], A_store[i + 1],
                            SOIL_DEPTHS[i], SOIL_DEPTHS[i + 1], OMEGA))
        lmt_phi.append(_lam_phi(beta_store[i], beta_store[i + 1],
                                SOIL_DEPTHS[i], SOIL_DEPTHS[i + 1], OMEGA))

    # Reconstruct fitted temperatures
    fit_T = np.array([
        _sin_func(xdata, *popt_store[i]) for i in range(n_layers)
    ])

    info = {
        "soil_T_obs": soil_T,
        "fit_T": fit_T,
        "popt_store": popt_store,
        "xdata": xdata,
    }
    return np.array(lmt_A), np.array(lmt_phi), fit_T, info


def soil_heat_capacity(k, lam_phi, lam_A, thickness_weights=None):
    """Compute volumetric soil heat capacity C = k / λ̄.

    λ̄ is the arithmetic mean of thickness-weighted λ from amplitude and phase methods.
    """
    if thickness_weights is None:
        # Normalized thickness of top 4 layers (0–20 cm) → [0.02, 0.02, 0.06, 0.1] / 0.2
        thickness_weights = [0.1, 0.1, 0.3, 0.5]

    w = np.asarray(thickness_weights)
    lam_phi_w = np.sum(np.asarray(lam_phi) * w)
    lam_A_w   = np.sum(np.asarray(lam_A) * w)
    lam_mean = (lam_phi_w + lam_A_w) / 2
    return k / lam_mean


# ============================================================
# Summary 4-panel figure → SVG
# ============================================================
def _adaptive_lim(data, pad=0.05):
    """Return (lo, hi) with proportional padding around data range."""
    lo, hi = np.min(data), np.max(data)
    span = hi - lo
    return lo - pad * span, hi + pad * span


def plot_summary_figure(alpha, alb_info,
                        sigma, r2_sigma, emis_info,
                        z0m_info,
                        k_val, r2_k, k_info,
                        site_name, output_dir):
    """Combined 4-panel figure (a–d) saved as SVG."""
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    # ---- a) Albedo ----
    ax = axes[0, 0]
    hours = alb_info["hours"]
    albedo = alb_info["albedo_daily"]
    for i, h in enumerate(hours):
        hour_data = albedo[:, i * 2:(i + 1) * 2].flatten()
        ax.scatter([h] * len(hour_data), hour_data,
                   color="black", alpha=0.5, s=18,
                   label="Obs" if i == 0 else "")
    ax.axhline(alpha, color="black", linestyle="--", linewidth=1.5, label="Mean")
    ax.text(13, alpha + 0.05, rf"$\alpha = {alpha:.3f}$",
            fontsize=15, ha="center", va="bottom")
    ax.set_xlabel("Hour (Local Time)", labelpad=6)
    ax.set_ylabel("Albedo", labelpad=6)
    ax.set_xticks(hours)
    ax.set_yticks(np.arange(0.1, 0.51, 0.1))
    ax.set_xlim(7.5, 18.5)
    ax.set_ylim(0.1, 0.5)
    ax.set_title("(a)", loc="left", fontsize=20, pad=10)
    ax.grid(False)
    ax.legend(loc="upper left", prop={"size": 12})

    # ---- b) Emissivity ----
    ax = axes[0, 1]
    x_e = emis_info["x"]
    y_e = emis_info["y"]
    model_e = LinearRegression(fit_intercept=False)
    model_e.fit(x_e.reshape(-1, 1), y_e)
    ax.scatter(x_e, y_e, color="#000000", label="Obs",
               marker="o", facecolor="none", s=45, alpha=0.6)
    x_lo, x_hi = _adaptive_lim(x_e, pad=0.05)
    y_lo, y_hi = _adaptive_lim(y_e, pad=0.05)
    x_line = np.arange(x_lo, x_hi)
    ax.plot(x_line, model_e.predict(x_line.reshape(-1, 1)),
            color="#FF4B4BAF", label="Least Square Fitting",
            linewidth=2, linestyle="--", alpha=1)
    ax.plot(x_line, x_line, color="#3256B3AF", label="1:1 Line",
            linewidth=2, linestyle="--", alpha=1)
    ax.text(0.96, 0.55, rf"$\sigma = {sigma:.3f}$",
            fontsize=15, ha="right", va="center", transform=ax.transAxes)
    ax.text(0.96, 0.47, rf"$R^2 = {r2_sigma:.3f}$",
            fontsize=15, ha="right", va="center", transform=ax.transAxes)
    ax.set_xlabel(r"$\sigma_0 T_s^4 - R_{ld}\downarrow\ (\mathrm{W\ m^{-2}})$")
    ax.set_ylabel(r"$R_{lu}\uparrow - R_{ld}\downarrow\ (\mathrm{W\ m^{-2}})$")
    ax.set_title("(b)", loc="left", fontsize=20, pad=10)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="upper left")

    # ---- c) Roughness length ----
    ax = axes[1, 0]
    data_clean = z0m_info["log_z0m_clean"]
    x_grid = z0m_info["kde_x"]
    density = z0m_info["kde_density"]
    kde_max_log = z0m_info["kde_max_log"]
    sns.histplot(data_clean, kde=False, stat="density", bins=40,
                 color="#000000", edgecolor="white", alpha=0.5, ax=ax)
    ax.plot(x_grid, density, color="#E64D4D", linewidth=2, label="KDE", alpha=0.8)
    ax.axvline(kde_max_log, color="#000000", linestyle="--", linewidth=2,
               label="Max Density")
    ax.text(0.04, 0.62, rf"$\ln z_{{0m}} = {kde_max_log:.2f}$",
            color="#000000", fontsize=15, ha="left", va="center",
            transform=ax.transAxes)
    z0m_val = np.exp(kde_max_log)
    z0m_str = _fmt_sci(z0m_val)
    ax.text(0.04, 0.53, f"$z_{{0m}} = {z0m_str}$ m",
            color="#000000", fontsize=15, ha="left", va="center",
            transform=ax.transAxes)
    ax.set_xlabel(r"$\ln z_{0m}$", fontsize=18)
    ax.set_xticks(np.arange(-15, 4, 3))
    ax.set_ylabel("PDF")
    ax.legend(loc="upper left")
    ax.set_title("(c)", loc="left", fontsize=20, pad=10)

    # ---- d) Soil thermal conductivity ----
    ax = axes[1, 1]
    x_k = k_info["x"]
    y_k = k_info["y"]
    model_k = LinearRegression(fit_intercept=False)
    model_k.fit(x_k.reshape(-1, 1), y_k)
    ax.scatter(x_k, y_k, color="#000000", label="Obs",
               marker="o", facecolor="none", s=45, alpha=0.6)
    xk_lo, xk_hi = _adaptive_lim(x_k, pad=0.16)
    yk_lo, yk_hi = _adaptive_lim(y_k, pad=0.30)
    x_line_k = np.arange(xk_lo, xk_hi)
    ax.plot(x_line_k, model_k.predict(x_line_k.reshape(-1, 1)),
            color="#E64D4D", label="Least Square Fitting",
            linewidth=2, linestyle="--", alpha=1)
    ax.hlines(0, xk_lo, xk_hi, colors="#000000FF", linestyles="--", linewidth=2)
    ax.vlines(0, yk_lo, yk_hi, colors="#000000FF", linestyles="--", linewidth=2)
    ax.text(0.04, 12, f"$k = {k_val:.2f}$ W m$^{{-1}}$ K$^{{-1}}$",
            fontsize=15, ha="left", va="bottom",
            transform=ax.get_yaxis_transform())
    ax.text(0.04, 1, rf"$R^2 = {r2_k:.3f}$",
            fontsize=15, ha="left", va="bottom",
            transform=ax.get_yaxis_transform())
    ax.set_xlabel(r"$\Delta T / \Delta z$ (K/m)")
    ax.set_ylabel(r"$G_{6cm}$ (W m$^{-2}$)")
    ax.set_xlim(xk_lo, xk_hi)
    ax.set_ylim(yk_lo, yk_hi)
    ax.set_title("(d)", loc="left", fontsize=20, pad=10)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    svg_path = os.path.join(output_dir, "parameters_summary.svg")
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Summary figure saved: {svg_path}")
    return svg_path


# ============================================================
# Exploratory figures → JPG
# ============================================================
def plot_soil_T_fit(soil_info, site_name, output_dir):
    """Exploratory: detrended observed vs sinusoidally fitted soil temperature."""
    from scipy.signal import detrend as _detrend

    soil_T_obs = soil_info["soil_T_obs"]
    fit_T = soil_info["fit_T"]
    obs_det = np.array([_detrend(soil_T_obs[i, :], type="linear")
                        for i in range(soil_T_obs.shape[0])])

    # --- 0–2 cm ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(obs_det[0, :], color="#000000", linewidth=2, linestyle="-",
            label="Obs-0cm")
    ax.plot(obs_det[1, :], color="#000000", linewidth=2, linestyle="--",
            label="Obs-2cm")
    ax.plot(fit_T[0, :], color="#FF6347", linewidth=2, linestyle="-",
            label="Fit-0cm")
    ax.plot(fit_T[1, :], color="#FF6347", linewidth=2, linestyle="--",
            label="Fit-2cm")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"{site_name} — 0–2 cm soil T: obs (detrended) vs sin-fit")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    fig.tight_layout()
    path = os.path.join(output_dir, "soil_T_fit_0-10cm.jpg")
    fig.savefig(path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- 20–100 cm ---
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["20 cm", "40 cm", "60 cm", "100 cm"]
    linestyles = ["-", "--", "-.", ":"]
    for j, (ls, lbl) in enumerate(zip(linestyles, labels)):
        idx = j + 4  # layers 4,5,6,7
        ax.plot(obs_det[idx, :], color="#000000", linewidth=2, linestyle=ls,
                label=f"Obs-{lbl}")
        ax.plot(fit_T[idx, :], color="#6495ED", linewidth=2, linestyle=ls,
                label=f"Fit-{lbl}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"{site_name} — 20–100 cm soil T: obs (detrended) vs sin-fit")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    fig.tight_layout()
    path = os.path.join(output_dir, "soil_T_fit_20-100cm.jpg")
    fig.savefig(path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("  [✓] Soil T fit figures saved.")


def _make_sci_ticks(ylo, yhi, n_ticks=5):
    """Return tick locations using clean scientific-notation steps."""
    raw_step = (yhi - ylo) / n_ticks
    exp = np.floor(np.log10(raw_step))
    mantissa = raw_step / 10**exp
    if mantissa <= 1.5:
        step = 10**exp
    elif mantissa <= 3.5:
        step = 2 * 10**exp
    else:
        step = 5 * 10**exp
    start = np.ceil(ylo / step) * step
    stop = np.floor(yhi / step) * step
    return np.arange(start, stop + step / 2, step)


def plot_thermal_diffusivity(lmt_phi, lmt_A, site_name, output_dir):
    """Exploratory: λ estimated from phase-lag vs amplitude-damping methods.

    Y-axis uses e-notation, each point labelled with its value.
    """
    from matplotlib.ticker import FuncFormatter

    all_vals = np.concatenate([lmt_phi, lmt_A])
    x_idx = np.arange(len(lmt_phi))
    depth_labels = ["0", "2", "4", "10", "20", "40", "60"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_idx, lmt_phi, color="#6495ED", linewidth=2, marker="^",
            markersize=7, label=r"$\varphi \rightarrow \lambda$")
    ax.plot(x_idx, lmt_A, color="#FF6347", linewidth=2, marker="^",
            markersize=7, label=r"$A \rightarrow \lambda$")

    # Annotate each point with e-notation
    for i in x_idx:
        ax.annotate(f"{lmt_phi[i]:.1e}",
                    (i, lmt_phi[i]), textcoords="offset points",
                    xytext=(8, 6), fontsize=8.5,
                    color="#6495ED", ha="left", va="bottom")
        ax.annotate(f"{lmt_A[i]:.1e}",
                    (i, lmt_A[i]), textcoords="offset points",
                    xytext=(8, -12), fontsize=8.5,
                    color="#FF6347", ha="left", va="top")

    ylo, yhi = _adaptive_lim(all_vals, pad=0.12)
    yticks = _make_sci_ticks(ylo, yhi, n_ticks=6)
    ax.set_yticks(yticks)
    ax.set_ylim(ylo, yhi)

    # Y-axis in e-notation (no offset text)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1e}"))

    ax.set_xticks(x_idx)
    ax.set_xticklabels(depth_labels, fontsize=12)
    ax.set_xlabel("Depth (cm)", fontsize=14)
    ax.set_ylabel(r"$\lambda$ (m$^{2}$ s$^{-1}$)", fontsize=14)
    ax.set_title(f"{site_name} — Thermal diffusivity from φ and A", fontsize=15)
    ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=11)

    fig.tight_layout()
    path = os.path.join(output_dir, "thermal_diffusivity.jpg")
    fig.savefig(path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [✓] Thermal diffusivity figure saved.")


# ============================================================
# Helpers
# ============================================================
def _fmt_sci(val, n_dec=2):
    """Format a number in LaTeX ×10^{exponent} notation.

    E.g.  1.234e-3  →  1.23 × 10^{-3}
    """
    s = f"{val:.{n_dec}e}"
    base, exp = s.split("e")
    exp_int = int(exp)  # removes leading zeros / sign
    return f"{base} \\times 10^{{{exp_int}}}"


# ============================================================
# Combined 3-site × 4-parameter figure
# ============================================================
def plot_combined_figure(all_plot_data, site_names, output_dir):
    """3 columns (sites) × 4 rows (parameters) combined summary figure."""
    set_style()
    # Larger font sizes for the combined multi-panel figure
    rcParams["font.size"] = 24
    rcParams["legend.fontsize"] = 18
    rcParams["xtick.labelsize"] = 22
    rcParams["ytick.labelsize"] = 22
    rcParams["axes.labelsize"] = 26
    rcParams["xtick.major.size"] = 10
    rcParams["ytick.major.size"] = 10
    rcParams["axes.linewidth"] = 2.0

    n_rows = 4
    n_cols = len(site_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 34))

    # Continuous lettering: a) through l)
    letters = [f"({c})" for c in "abcdefghijkl"]
    param_names = ["Albedo", "Emissivity", "Roughness length", "Soil conductivity"]

    AXIS_FS = 26   # axis label fontsize
    TEXT_FS = 24   # annotation text fontsize
    TITLE_FS = 30  # subplot title fontsize
    LEG_FS = 20    # legend fontsize

    for col_idx, site in enumerate(site_names):
        d = all_plot_data[site]
        alb_info = d["alb_info"]
        emis_info = d["emis_info"]
        z0m_info = d["z0m_info"]
        k_info = d["k_info"]

        # ---- Row 1: Albedo (a, b, c) ----
        ax = axes[0, col_idx]
        hours = alb_info["hours"]
        albedo = alb_info["albedo_daily"]
        for i, h in enumerate(hours):
            hour_data = albedo[:, i * 2:(i + 1) * 2].flatten()
            ax.scatter([h] * len(hour_data), hour_data,
                       color="black", alpha=0.5, s=28,
                       label="Obs" if i == 0 else "")
        ax.axhline(d["alpha"], color="black", linestyle="--", linewidth=2.0, label="Mean")
        ax.text(13, d["alpha"] + 0.08, rf"$\alpha = {d['alpha']:.3f}$",
                fontsize=TEXT_FS, ha="center", va="bottom")
        ax.set_xticks(hours)
        ax.set_yticks(np.arange(0.1, 0.51, 0.1))
        ax.set_xlim(7.5, 18.5)
        ax.set_ylim(0.1, 0.5)
        ax.set_title(f"{letters[col_idx]} {param_names[0]} — {site}",
                     loc="left", fontsize=TITLE_FS, pad=14)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=22, width=1.5, length=8)
        if col_idx == 0:
            ax.set_ylabel("Albedo", fontsize=AXIS_FS, labelpad=10)
            ax.legend(loc="upper left", prop={"size": LEG_FS})
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Hour (Local Time)", fontsize=AXIS_FS, labelpad=10)

        # ---- Row 2: Emissivity (d, e, f) ----
        ax = axes[1, col_idx]
        x_e = emis_info["x"]
        y_e = emis_info["y"]
        model_e = emis_info["model"]
        ax.scatter(x_e, y_e, color="#000000", label="Obs",
                   marker="o", facecolor="none", s=55, alpha=0.6)
        x_lo, x_hi = _adaptive_lim(x_e, pad=0.05)
        y_lo, y_hi = _adaptive_lim(y_e, pad=0.05)
        x_line = np.arange(x_lo, x_hi)
        ax.plot(x_line, model_e.predict(x_line.reshape(-1, 1)),
                color="#FF4B4BAF", label="Least Square Fitting",
                linewidth=2.5, linestyle="--", alpha=1)
        ax.plot(x_line, x_line, color="#3256B3AF", label="1:1 Line",
                linewidth=2.5, linestyle="--", alpha=1)
        ax.text(0.96, 0.58, rf"$\sigma = {d['sigma']:.3f}$",
                fontsize=TEXT_FS, ha="right", va="center", transform=ax.transAxes)
        ax.text(0.96, 0.47, rf"$R^2 = {d['r2_sig']:.3f}$",
                fontsize=TEXT_FS, ha="right", va="center", transform=ax.transAxes)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(f"{letters[3 + col_idx]} {param_names[1]} — {site}",
                     loc="left", fontsize=TITLE_FS, pad=14)
        ax.tick_params(axis="both", labelsize=22, width=1.5, length=8)
        if col_idx == 0:
            ax.set_ylabel(r"$R_{lu}\uparrow - R_{ld}\downarrow\ (\mathrm{W\ m^{-2}})$",
                          fontsize=AXIS_FS, labelpad=10)
            ax.legend(loc="upper left", prop={"size": LEG_FS})
        else:
            ax.set_ylabel("")
        ax.set_xlabel(r"$\sigma_0 T_s^4 - R_{ld}\downarrow\ (\mathrm{W\ m^{-2}})$",
                      fontsize=AXIS_FS, labelpad=10)

        # ---- Row 3: Roughness length (g, h, i) ----
        ax = axes[2, col_idx]
        data_clean = z0m_info["log_z0m_clean"]
        x_grid = z0m_info["kde_x"]
        density = z0m_info["kde_density"]
        kde_max_log = z0m_info["kde_max_log"]
        sns.histplot(data_clean, kde=False, stat="density", bins=50,
                     color="#000000", edgecolor="white", alpha=0.5, ax=ax)
        ax.plot(x_grid, density, color="#E64D4D", linewidth=2.5, label="KDE", alpha=0.8)
        ax.axvline(kde_max_log, color="#000000", linestyle="--", linewidth=2.5,
                   label="Max Density")
        z0m_val = np.exp(kde_max_log)
        z0m_str = _fmt_sci(z0m_val)
        ax.text(0.04, 0.62, rf"$\ln z_{{0m}} = {kde_max_log:.2f}$",
                fontsize=TEXT_FS, ha="left", va="center", transform=ax.transAxes)
        ax.text(0.04, 0.53, f"$z_{{0m}} = {z0m_str}$ m",
                fontsize=TEXT_FS, ha="left", va="center", transform=ax.transAxes)
        ax.set_xticks(np.arange(-15, 4, 3))
        ax.set_title(f"{letters[6 + col_idx]} {param_names[2]} — {site}",
                     loc="left", fontsize=TITLE_FS, pad=14)
        ax.tick_params(axis="both", labelsize=22, width=1.5, length=8)
        if col_idx == 0:
            ax.set_ylabel("PDF", fontsize=AXIS_FS, labelpad=10)
            ax.legend(loc="upper left", prop={"size": LEG_FS})
        else:
            ax.set_ylabel("")
        ax.set_xlabel(r"$\ln z_{0m}$", fontsize=AXIS_FS, labelpad=10)

        # ---- Row 4: Soil thermal conductivity (j, k, l) ----
        ax = axes[3, col_idx]
        x_k = k_info["x"]
        y_k = k_info["y"]
        model_k = k_info["model"]
        ax.scatter(x_k, y_k, color="#000000", label="Obs",
                   marker="o", facecolor="none", s=55, alpha=0.6)
        xk_lo, xk_hi = _adaptive_lim(x_k, pad=0.16)
        yk_lo, yk_hi = _adaptive_lim(y_k, pad=0.30)
        x_line_k = np.arange(xk_lo, xk_hi)
        ax.plot(x_line_k, model_k.predict(x_line_k.reshape(-1, 1)),
                color="#E64D4D", label="Least Square Fitting",
                linewidth=2.5, linestyle="--", alpha=1)
        ax.hlines(0, xk_lo, xk_hi, colors="#000000FF", linestyles="--", linewidth=2.5)
        ax.vlines(0, yk_lo, yk_hi, colors="#000000FF", linestyles="--", linewidth=2.5)
        y_span = yk_hi - yk_lo
        ax.text(0.04, y_span * 0.06, f"$k = {d['k_val']:.2f}$ W m$^{{-1}}$ K$^{{-1}}$",
                fontsize=TEXT_FS, ha="left", va="bottom", transform=ax.get_yaxis_transform())
        ax.text(0.04, -y_span * 0.06, rf"$R^2 = {d['r2_k']:.3f}$",
                fontsize=TEXT_FS, ha="left", va="top", transform=ax.get_yaxis_transform())
        ax.set_xlim(xk_lo, xk_hi)
        ax.set_ylim(yk_lo, yk_hi)
        ax.set_title(f"{letters[9 + col_idx]} {param_names[3]} — {site}",
                     loc="left", fontsize=TITLE_FS, pad=14)
        ax.tick_params(axis="both", labelsize=22, width=1.5, length=8)
        if col_idx == 0:
            ax.set_ylabel(r"$G_{6cm}$ (W m$^{-2}$)", fontsize=AXIS_FS, labelpad=10)
            ax.legend(loc="upper left", prop={"size": LEG_FS})
        else:
            ax.set_ylabel("")
        ax.set_xlabel(r"$\Delta T / \Delta z$ (K/m)", fontsize=AXIS_FS, labelpad=10)

    fig.tight_layout(pad=2.5)
    svg_path = os.path.join(output_dir, "parameters_summary_combined.svg")
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [✓] Combined summary figure saved: {svg_path}")
    return svg_path


# ============================================================
# Main pipeline for a single site
# ============================================================
def run_site(site_name, data_base=None, output_base=None):
    """Run full parameter estimation for one site.

    Parameters
    ----------
    site_name : str
        One of {"Huazhaizi", "Ejin", "Shenshawo"}.
    data_base : str, optional
        Root directory containing per-site CSV folders.
    output_base : str, optional
        Root output directory (subfolder per site created automatically).
    """
    if data_base is None:
        data_base = DATA_BASE
    if output_base is None:
        output_base = OUTPUT_BASE

    data_dir = os.path.join(data_base, site_name)
    out_dir = os.path.join(output_base, site_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Site: {site_name}")
    print(f"  Data: {data_dir}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Load data
    meteo, soil, flux = load_data(data_dir)

    # --- (a) Albedo ---
    alpha, alb_info = estimate_albedo(meteo)
    print(f"  (a) Albedo  α  = {alpha:.4f}")

    # --- (b) Emissivity ---
    sigma, r2_sig, emis_info = estimate_emissivity(meteo, soil)
    print(f"  (b) Emissivity  σ  = {sigma:.4f}  (R² = {r2_sig:.4f})")

    # --- (c) Roughness length ---
    z0m_max, _, z0m_info = estimate_z0m(flux)
    print(f"  (c) Roughness length  z₀ₘ  = {z0m_max:.4e} m")

    # --- (d) Soil thermal conductivity ---
    k_val, r2_k, k_info = estimate_soil_k(soil)
    print(f"  (d) Soil conductivity  k  = {k_val:.4f} W m^{-1} K^{-1}  (R² = {r2_k:.4f})")

    # --- Thermal diffusivity & heat capacity ---
    lmt_A, lmt_phi, _fit_T, soil_info = estimate_thermal_diffusivity(soil)
    C_val = soil_heat_capacity(k_val, lmt_phi[:4], lmt_A[:4])
    # thickness-weighted average (2, 2, 6, 10 cm → 0.1, 0.1, 0.3, 0.5)
    w = np.array([0.1, 0.1, 0.3, 0.5])
    lam_A_w = np.sum(lmt_A[:4] * w)
    lam_phi_w = np.sum(lmt_phi[:4] * w)
    print(f"  (+) λ̄_A   = {lam_A_w:.4e} m^{{2}} s^{{-1}}")
    print(f"  (+) λ̄_φ   = {lam_phi_w:.4e} m^{{2}} s^{{-1}}")
    print(f"  (+) Heat capacity  C  = {C_val:.2e} J m^{{-3}} K^{{-1}}")

    # --- Summary figure (SVG) ---
    plot_summary_figure(alpha, alb_info,
                        sigma, r2_sig, emis_info,
                        z0m_info,
                        k_val, r2_k, k_info,
                        site_name, out_dir)

    # --- Exploratory figures (JPG) ---
    plot_soil_T_fit(soil_info, site_name, out_dir)
    plot_thermal_diffusivity(lmt_phi, lmt_A, site_name, out_dir)

    # --- Print summary ---
    print(f"\n  {'─'*50}")
    print(f"  PARAMETER SUMMARY — {site_name}")
    print(f"  {'─'*50}")
    print(f"  α   (albedo)            = {alpha:.4f}")
    print(f"  σ   (emissivity)        = {sigma:.4f}")
    print(f"  z₀ₘ (roughness length)  = {z0m_max:.4e} m")
    print(f"  k   (conductivity)      = {k_val:.4f} W m^{{-1}} K^{{-1}}")
    print(f"  C   (heat capacity)     = {C_val:.2e} J m^{{-3}} K^{{-1}}")
    print(f"  {'─'*50}")

    # --- Save estimated parameters as npy ---
    results = {
        "site": site_name,
        "alpha": alpha,
        "sigma": sigma,
        "sigma_r2": r2_sig,
        "z0m": z0m_max,
        "k": k_val,
        "k_r2": r2_k,
        "C": C_val,
        "lam_A_mean": lam_A_w,
        "lam_phi_mean": lam_phi_w,
        "lmt_A": lmt_A,          # full 7-layer amplitude-method λ
        "lmt_phi": lmt_phi,      # full 7-layer phase-method λ
    }
    npy_path = os.path.join(output_base, f"{site_name}.npy")
    np.save(npy_path, results)
    print(f"  [✓] Parameters saved: {npy_path}")

    # Package plot data for combined figure
    plot_data = {
        "alpha": alpha,
        "alb_info": alb_info,
        "sigma": sigma,
        "r2_sig": r2_sig,
        "emis_info": emis_info,
        "z0m_info": z0m_info,
        "k_val": k_val,
        "r2_k": r2_k,
        "k_info": k_info,
    }

    return results, plot_data


def run_all(data_base=None, output_base=None):
    """Batch-run all 3 sites."""
    from datetime import datetime

    if data_base is None:
        data_base = DATA_BASE
    if output_base is None:
        output_base = OUTPUT_BASE

    results = {}
    all_plot_data = {}
    for site in SITES:
        res, plot_data = run_site(site, data_base, output_base)
        results[site] = res
        all_plot_data[site] = plot_data

    # Cross-site combined summary figure
    plot_combined_figure(all_plot_data, SITES, output_base)

    # Cross-site summary table (print + log)
    header = f"{'Site':<16} {'α':>8} {'σ':>8} {'z₀ₘ (m)':>14} {'k':>8} {'C (J m^{{-3}} K^{{-1}})':>22}"
    sep = "-" * 80

    lines = []
    lines.append("=" * 80)
    lines.append("  LSM PARAMETER ESTIMATION LOG")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    # Per-site parameter summaries
    for site in SITES:
        r = results[site]
        lines.append(f"{'─'*60}")
        lines.append(f"  PARAMETER SUMMARY — {site}")
        lines.append(f"{'─'*60}")
        lines.append(f"  α   (albedo)            = {r['alpha']:.4f}")
        lines.append(f"  σ   (emissivity)        = {r['sigma']:.4f}    (R² = {r['sigma_r2']:.4f})")
        lines.append(f"  z₀ₘ (roughness length)  = {r['z0m']:.4e} m")
        lines.append(f"  k   (conductivity)      = {r['k']:.4f} W m^{{-1}} K^{{-1}}    (R² = {r['k_r2']:.4f})")
        lines.append(f"  C   (heat capacity)     = {r['C']:.2e} J m^{{-3}} K^{{-1}}")
        lines.append(f"  λ̄_A (diffusivity, amp)  = {r['lam_A_mean']:.4e} m^{{2}} s^{{-1}}")
        lines.append(f"  λ̄_φ (diffusivity, phase) = {r['lam_phi_mean']:.4e} m^{{2}} s^{{-1}}")
        lines.append(f"  {'─'*60}")
        lines.append(f"  λ per-layer estimates:")
        lines.append(f"  {'Layer (cm)':<16} {'λ_amp (m² s⁻¹)':>18} {'λ_phase (m² s⁻¹)':>18}")
        lines.append(f"  {'─'*16} {'─'*16} {'─'*16}")
        # Layer depth labels for 7 consecutive pairs
        layer_pairs = [
            "0 → 2", "2 → 4", "4 → 10",
            "10 → 20", "20 → 40", "40 → 60", "60 → 100",
        ]
        for i, (z_str, la, lp) in enumerate(zip(layer_pairs, r["lmt_A"], r["lmt_phi"])):
            lines.append(f"  {z_str:<16} {la:16.4e} {lp:16.4e}")
        lines.append(f"{'─'*60}")
        lines.append("")

    # Cross-site comparison table
    lines.append("=" * 80)
    lines.append("  CROSS-SITE COMPARISON")
    lines.append("=" * 80)
    lines.append(header)
    lines.append(sep)
    for site in SITES:
        r = results[site]
        lines.append(f"{site:<16} {r['alpha']:8.4f} {r['sigma']:8.4f} "
                     f"{r['z0m']:14.4e} {r['k']:8.4f} {r['C']:14.2e}")
    lines.append("=" * 80)

    log_text = "\n".join(lines)

    # Print to console
    print(f"\n{log_text}")

    # Write to log file
    log_path = os.path.join(output_base, "parameter_est.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text + "\n")
    print(f"  [✓] Parameter estimation log saved: {log_path}")

    return results


# ============================================================
# CLI entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSM parameter estimation for desert stations.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--site", type=str,
                       help="Run a single site (Huazhaizi, Ejin, Shenshawo)")
    group.add_argument("--all", action="store_true",
                       help="Run all 3 sites in batch")
    parser.add_argument("--data", type=str, default=None,
                        help="Override data base directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output base directory")
    args = parser.parse_args()

    if args.all:
        run_all(data_base=args.data, output_base=args.output)
    else:
        if args.site not in SITES:
            print(f"Error: unknown site '{args.site}'. Choose from {SITES}")
            sys.exit(1)
        _, _ = run_site(args.site, data_base=args.data, output_base=args.output)
