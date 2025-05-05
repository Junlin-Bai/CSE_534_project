# -*- coding: utf-8 -*-
r"""
Extended plotting script: GHZ **and** matching CHSH-based estimates
===========================================================
This keeps the original API and variable names while adding:

* **process_chsh_results** – scans sister folders with suffix `_chsh`, extracts Run 1 `s_value` and computes lower/upper CHSH‐based fidelity bounds.
* **plot_figure** – overlays Actual vs. GHZ-protocol Estimated Fidelity (and CHSH bounds when available). For **Change Depolar Rate**, discards any rates ≤ 2000 Hz and only plots starting at 4000 Hz.
* **plot_error_rate_triplet** – computes and plots three error‐rate curves:
    - GHZ‐protocol Error Rate = (GHZ_est − Actual) / Actual
    - CHSH‐based Lower Bound Error Rate = (CHSH_low − Actual) / Actual
    - CHSH‐based Upper Bound Error Rate = (CHSH_up − Actual) / Actual
  It also draws a horizontal zero‐line and matches each curve’s color to the corresponding fidelity curve.

Six figures in total (three fidelity + three error‐rate) are saved under `./figures`.
"""

import json
import os
import re
from math import sqrt
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------
# Matplotlib global style
# --------------------------------------------------------------------
WIDTH = 8.5
HEIGHT = WIDTH * (np.sqrt(5) - 1.0) / 2.5
plt.rcParams.update({
    "figure.figsize": (WIDTH, HEIGHT),
    "font.size": 12,
    "axes.axisbelow": True,
})

# --------------------------------------------------------------------
# GHZ result processing (original behaviour)
# --------------------------------------------------------------------
def process_results(file_path: str,
                    re_expressions: str
                   ) -> Tuple[List[float], Optional[List[float]], List[float], List[float]]:
    """Parse GHZ JSON files in a folder (single-run datasets)."""
    factors: List[float] = []
    second_vals_dict: dict = {}
    actual_dict: dict = {}
    est_dict: dict = {}
    multi_factor = False

    for root, _, files in os.walk(file_path):
        for fname in files:
            m = re.search(re_expressions, fname)
            if not m:
                continue
            factor = float(m.group(1))
            if len(m.groups()) > 1:
                second_vals_dict[factor] = float(m.group(2))
                multi_factor = True
            else:
                factors.append(factor)

            with open(Path(root, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            actual = np.mean(list(data["actual_fid"].values())[0])
            p_ghz  = list(data["p_ghz"].values())[0]
            est    = 2 * p_ghz - 1

            actual_dict[factor] = actual
            est_dict[factor] = est

    if multi_factor:
        factors = sorted(second_vals_dict.keys())
        second_vals = [second_vals_dict[x] for x in factors]
    else:
        factors = sorted(factors)
        second_vals = None

    actual = [actual_dict[x] for x in factors]
    est    = [est_dict[x]    for x in factors]
    return factors, second_vals, actual, est

# --------------------------------------------------------------------
# CHSH companion processing
# --------------------------------------------------------------------
def process_chsh_results(folder_chsh: str,
                         re_expressions: str,
                         factors_order: List[float]
                        ) -> Tuple[List[float], List[float]]:
    """Compute CHSH-based lower/upper fidelity bounds aligned to factors_order."""
    low_d, up_d = {}, {}
    if not os.path.isdir(folder_chsh):
        nan = [np.nan] * len(factors_order)
        return nan, nan

    for root, _, files in os.walk(folder_chsh):
        for fname in files:
            m = re.search(re_expressions, fname)
            if not m:
                continue
            factor = float(m.group(1))
            with open(Path(root, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            s = list(data["s_value"].values())[0]
            low_d[factor] = s / (2 * sqrt(2))
            up_d[factor]  = s / (4 * sqrt(2)) + 0.5

    low = [low_d.get(x, np.nan) for x in factors_order]
    up  = [up_d.get(x, np.nan)  for x in factors_order]
    return low, up

# --------------------------------------------------------------------
# Generic line-plot helper
# --------------------------------------------------------------------
def plot_lines(xs: Sequence[Sequence[float]],
               ys: Sequence[Sequence[float]],
               legends: Sequence[str],
               title: str,
               x_label: str,
               y_label: str,
               save_dir: str = "./figures",
               annotation: Optional[Sequence[float]] = None,
               ncols: int = 1):
    """Draw multiple lines with markers, legends, and optional annotations."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()
    for x, y, leg in zip(xs, ys, legends):
        ax.plot(x, y, label=leg, marker="o")
        if annotation is not None:
            for i, (xi, yi) in enumerate(zip(x, y)):
                if i < len(annotation):
                    ax.annotate(f"{annotation[i]}", (xi, yi),
                                textcoords="offset points", xytext=(0, 5),
                                ha="center", fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(ncols=ncols, fontsize=10)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{title}.png")
    plt.close()

# --------------------------------------------------------------------
# Error-rate plotting helper (three curves) with zero‐line & matched colors
# --------------------------------------------------------------------
def plot_error_rate_triplet(factors: List[float],
                            actual: List[float],
                            ghz_est: List[float],
                            chsh_low: List[float],
                            chsh_up: List[float],
                            title: str,
                            x_label: str,
                            save_dir: str = "./figures"):
    """
    Compute and plot three error‐rate curves:
      - GHZ‐protocol Error Rate
      - CHSH‐based Lower Bound Error Rate
      - CHSH‐based Upper Bound Error Rate
    Draws a horizontal zero‐line and matches colors to the fidelity plot.
    """
    os.makedirs(save_dir, exist_ok=True)

    # compute error‐rates
    err_ghz = [(e - a) / a for a, e in zip(actual, ghz_est)]
    err_lo  = [(l - a) / a for a, l in zip(actual, chsh_low)]
    err_hi  = [(u - a) / a for a, u in zip(actual, chsh_up)]

    # get default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots()
    # zero reference line
    ax.axhline(0, linestyle='--', linewidth=1, color='gray')

    # plot with matching colors:
    ax.plot(factors, err_ghz, label="GHZ-protocol Error Rate",
            marker="o", color=colors[1])
    ax.plot(factors, err_lo,  label="CHSH-based Lower Bound Error Rate",
            marker="o", color=colors[2])
    ax.plot(factors, err_hi,  label="CHSH-based Upper Bound Error Rate",
            marker="o", color=colors[3])

    ax.set_xlabel(x_label)
    ax.set_ylabel("Error Rate")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{title}_error.png")
    plt.close()

# --------------------------------------------------------------------
# High-level fidelity routine
# --------------------------------------------------------------------
def plot_figure(folder: str,
                re_expr: str,
                title: str,
                x_label: str,
                save_dir: str = "./figures"):
    """Plot fidelity curves; custom handling for sample size & depolar rate."""
    factors, second_vals, actual, est = process_results(folder, re_expr)
    low, up = process_chsh_results(f"{folder}_chsh", re_expr, factors)

    # for Change Depolar Rate: drop ≤2000 Hz, keep ≥4000 Hz
    if title == "Change Depolar Rate":
        idx = [i for i, f in enumerate(factors) if f >= 4000]
        factors    = [factors[i]    for i in idx]
        actual     = [actual[i]     for i in idx]
        est        = [est[i]        for i in idx]
        low        = [low[i]        for i in idx]
        up         = [up[i]         for i in idx]
        if second_vals is not None:
            second_vals = [second_vals[i] for i in idx]

    ys      = [actual, est]
    legends = ["Actual Fidelity", "GHZ-protocol Estimated Fidelity"]
    if not np.all(np.isnan(low)):
        ys      += [low, up]
        legends += ["CHSH-based Lower Bound", "CHSH-based Upper Bound"]
    annotation = second_vals if second_vals is not None else None

    if title == "Change Sample Size":
        # custom sample-size x-axis
        fig, ax = plt.subplots()
        for x, y, leg in zip([factors]*len(ys), ys, legends):
            ax.plot(x, y, label=leg, marker="o")
        ax.set_xlabel("Sample Number")
        ax.set_ylabel("Fidelity")
        ax.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6])
        ax.set_xticklabels(["5k","10k","15k","20k","25k","30k"])
        ax.legend(ncols=2, fontsize=10)
        plt.tight_layout()
        plt.savefig(Path(save_dir) / f"{title}_fid.png")
        plt.close()
    else:
        plot_lines([factors]*len(ys), ys, legends,
                   f"{title}_fid", x_label, "Fidelity",
                   save_dir=save_dir, annotation=annotation, ncols=2)

# --------------------------------------------------------------------
# Wrapper: fidelity + error for twin folders
# --------------------------------------------------------------------
def plot_all_twin(folder1: str,
                  folder2: str,
                  re_expr: str,
                  title: str,
                  x_label: str,
                  save_dir: str = "./figures"):
    # fidelity plots
    plot_figure(folder1, re_expr, title, x_label, save_dir)
    plot_figure(folder2, re_expr, title, x_label, save_dir)

    # error‐rate plot (once)
    factors, _, actual, est = process_results(folder1, re_expr)
    low, up = process_chsh_results(f"{folder1}_chsh", re_expr, factors)
    if title == "Change Depolar Rate":
        idx = [i for i, f in enumerate(factors) if f >= 4000]
        factors = [factors[i] for i in idx]
        actual  = [actual[i]  for i in idx]
        est     = [est[i]     for i in idx]
        low     = [low[i]     for i in idx]
        up      = [up[i]      for i in idx]

    plot_error_rate_triplet(factors, actual, est, low, up,
                            title, x_label, save_dir)

# --------------------------------------------------------------------
# CLI invocation
# --------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Depolarization rate (fidelity + error)
    plot_all_twin(
        "./change_channel_rate_result_0.1_60000",
        "./change_channel_rate_result_0.1_60000",
        r"channel_([\d\.]+)_",
        "Change Depolar Rate",
        r"$R_c$ (Hz)"
    )
    # 2. Node distance (fidelity + error)
    plot_all_twin(
        "./change_distance_result_0.1_60000",
        "./change_distance_result_0.1_60000",
        r"distance_([\d\.]+)_",
        "Change Node Distance",
        r"$L$ (Km)"
    )
    # 3. Sample size (fidelity + error)
    plot_all_twin(
        "./change_sample_size_result_0.1_50000",
        "./change_sample_size_result_0.1_50000",
        r"sample_([\d\.]+)_alpha",
        "Change Sample Size",
        "Sample Number"
    )
