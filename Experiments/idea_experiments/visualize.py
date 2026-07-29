"""
Visualization module: 5-panel comparison plot.

Layout (2×3 equal-sized grid, bottom-left empty):
  a) Analytical λ  |  b) Direct λ  |  c) MLP λ
                      |  d) Direct error |  e) MLP error

Usage:
  python visualize.py --direct ./output/direct_lmd --nn ./output/nn_lmd
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.tri import Triangulation
from matplotlib import rcParams

from common import theta_lim, n_lim


# ── Matplotlib defaults (unified with parameter estimation & optimization experiments) ──
rcParams.update(
    {
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
    }
)


# ── Common axis decoration ──
def _setup_ax(ax, xlabel=False, ylabel=False):
    ax.set_xlim(*theta_lim)
    ax.set_ylim(*n_lim)
    ax.set_yticks(np.arange(0.2, 0.9, 0.2))
    ax.grid(True, alpha=0.15, ls="-")
    if xlabel:
        ax.set_xlabel(r"$\theta$  (m$^{3}$ m$^{-3}$)", fontsize=16)
    if ylabel:
        ax.set_ylabel(r"$n$  (m$^{3}$ m$^{-3}$)", fontsize=16)


def plot_comparison(
    theta,
    n,
    lmd_true,
    lmd_direct,
    lmd_nn,
    outpath="./comparison.png",
):
    """
    5-panel comparison plot.

    Left — Analytical λ (analytical)
    Top-right left — Direct λ estimate
    Top-right right — MLP λ estimate
    Bottom-right left — Direct |relative error|
    Bottom-right right — MLP |relative error|
    """
    theta = np.asarray(theta)
    n = np.asarray(n)
    lmd_t = np.asarray(lmd_true) * 1e6
    lmd_d = np.asarray(lmd_direct) * 1e6
    lmd_m = np.asarray(lmd_nn) * 1e6
    rel_d = np.abs((lmd_d - lmd_t) / lmd_t) * 100  # %
    rel_m = np.abs((lmd_m - lmd_t) / lmd_t) * 100

    # Print statistics
    def _print_stats(name, rel):
        mask_high = lmd_true > 5e-7
        mask_low = lmd_true < 2e-7
        print(f"\n── {name} ──")
        print(f"  Mean relative error:                 {np.mean(rel):.2f} %")
        print(f"  Median relative error:               {np.median(rel):.2f} %")
        print(f"  Samples with error < 5%:             {np.mean(rel < 5) * 100:.1f} %")
        print(f"  Mean error (λ > 5×10⁻⁷ m² s⁻¹):     {np.mean(rel[mask_high]):.2f} %  (n={np.sum(mask_high)})")
        print(f"  Mean error (λ < 2×10⁻⁷ m² s⁻¹):     {np.mean(rel[mask_low]):.2f} %  (n={np.sum(mask_low)})")

    _print_stats("Direct", rel_d)
    _print_stats("MLP", rel_m)

    tri = Triangulation(theta, n)

    # Common parameters
    levels = np.linspace(0.05, 1.05, 21)
    contour_levels = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    lmd_label = r"$\lambda$  ($\times 10^{-6}$ m$^{2}$ s$^{-1}$)"
    err_label = r"$|\Delta\lambda / \lambda|$  (%)"

    # ── Layout: 4×3 grid, a) vertically centered, matching right-subplot height ──
    # Rows 0–1 → top-right b) c); Rows 2–3 → bottom-right d) e)
    # Rows 1–2 → left a) vertically centered
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(4, 3, hspace=0.45, wspace=0.35)

    # ────────────────────────────────
    # Left vertical center: Analytical λ
    # ────────────────────────────────
    ax_a = fig.add_subplot(gs[1:3, 0])
    cf_a = ax_a.tricontourf(tri, lmd_t, levels=levels, cmap="Greys", extend="both")
    cs = ax_a.tricontour(
        tri, lmd_t, levels=contour_levels, colors="k", linewidths=0.8, linestyles="-"
    )
    ax_a.clabel(cs, inline=True, fontsize=9, fmt="%.1f")
    cbar_a = fig.colorbar(cf_a, ax=ax_a, shrink=0.85, pad=0.02)
    cbar_a.set_label(lmd_label, fontsize=15)
    _setup_ax(ax_a, xlabel=True, ylabel=True)
    ax_a.set_title(r"(a) Analytical $\lambda$", fontsize=16, loc="left", pad=8)

    # ────────────────────────────────
    # Top-right left: Direct λ
    # ────────────────────────────────
    ax_b = fig.add_subplot(gs[0:2, 1])
    cf_b = ax_b.tricontourf(tri, lmd_d, levels=levels, cmap="Greys", extend="both")
    cs = ax_b.tricontour(
        tri, lmd_d, levels=contour_levels, colors="k", linewidths=0.6, linestyles="-"
    )
    ax_b.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
    _setup_ax(ax_b, ylabel=True)
    ax_b.set_title(r"(b) Direct $\lambda$", fontsize=16, loc="left", pad=8)

    # ────────────────────────────────
    # Top-right right: MLP λ
    # ────────────────────────────────
    ax_c = fig.add_subplot(gs[0:2, 2])
    cf_c = ax_c.tricontourf(tri, lmd_m, levels=levels, cmap="Greys", extend="both")
    cs = ax_c.tricontour(
        tri, lmd_m, levels=contour_levels, colors="k", linewidths=0.6, linestyles="-"
    )
    ax_c.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
    _setup_ax(ax_c)
    ax_c.set_title(r"(c) MLP $\lambda$", fontsize=16, loc="left", pad=8)

    # b) c) share colorbar
    cbar_bc = fig.colorbar(cf_c, ax=[ax_b, ax_c], shrink=0.85, pad=0.02)
    cbar_bc.set_label(lmd_label, fontsize=15)

    # ────────────────────────────────
    # Bottom-right left: Direct relative error
    # ────────────────────────────────
    err_vmax = 25
    err_levels = np.linspace(0, err_vmax, 21)

    ax_d = fig.add_subplot(gs[2:4, 1])
    cf_d = ax_d.tricontourf(tri, rel_d, levels=err_levels, cmap="Greys", extend="max")
    _setup_ax(ax_d, xlabel=True, ylabel=True)
    ax_d.set_title(
        r"(d) Direct $|\Delta\lambda/\lambda|$", fontsize=16, loc="left", pad=8
    )

    # ────────────────────────────────
    # Bottom-right right: MLP relative error
    # ────────────────────────────────
    ax_e = fig.add_subplot(gs[2:4, 2])
    cf_e = ax_e.tricontourf(tri, rel_m, levels=err_levels, cmap="Greys", extend="max")
    _setup_ax(ax_e, xlabel=True)
    ax_e.set_title(r"(e) MLP $|\Delta\lambda/\lambda|$", fontsize=16, loc="left", pad=8)

    # ────────────────────────────────
    # d) e) mean relative deviation annotations in lower-triangle blank area
    # ────────────────────────────────
    mean_rel_d = np.mean(rel_d)
    mean_rel_m = np.mean(rel_m)
    tx, ty = 0.35, 0.10  # data坐标，位于 n < θ 空白三角区

    ax_d.text(tx, ty, f"Mean relative error:\n{mean_rel_d:.1f} %",
              fontsize=13, ha="center", va="center")
    ax_e.text(tx, ty, f"Mean relative error:\n{mean_rel_m:.1f} %",
              fontsize=13, ha="center", va="center")

    # d) e) share colorbar
    cbar_de = fig.colorbar(cf_e, ax=[ax_d, ax_e], shrink=0.85, pad=0.02)
    cbar_de.set_ticks(np.arange(0, err_vmax + 1, 5))
    cbar_de.set_label(err_label, fontsize=15)

    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved to: {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot 5-panel comparison of two methods"
    )
    parser.add_argument(
        "--direct",
        type=str,
        default="./output/direct_lmd",
        help="Direct method npy directory",
    )
    parser.add_argument(
        "--nn",
        type=str,
        default="./output/nn_lmd",
        help="MLP method npy directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/comparison.svg",
        help="Output image path",
    )
    args = parser.parse_args()

    dir_direct = Path(args.direct)
    dir_nn = Path(args.nn)

    # Both methods share the same true λ, read from direct directory
    theta = np.load(dir_direct / "theta_valid.npy")
    n = np.load(dir_direct / "n_valid.npy")
    lmd_true = np.load(dir_direct / "lmd_true.npy")

    lmd_direct = np.load(dir_direct / "lmd_est.npy")
    lmd_nn = np.load(dir_nn / "lmd_est.npy")

    plot_comparison(theta, n, lmd_true, lmd_direct, lmd_nn, outpath=args.output)
