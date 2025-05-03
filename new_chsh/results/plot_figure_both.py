# -*- coding: utf-8 -*-
r"""
Extended plotting script: GHZ **and** matching CHSH estimates
===========================================================
This keeps the original API and variable names while adding:

* **process_chsh_results** – scans sister folders with the suffix
  ``_chsh``, extracts the *Run 1* ``s_value``, and computes lower/upper
  CHSH‑based fidelity bounds:

  .. math::

     F_{\text{lower}} = \frac{s}{2\sqrt{2}}, \qquad
     F_{\text{upper}} = \frac{s}{4\sqrt{2}} + 0.5

* **plot_figure** now overlays these bounds on the existing *Actual* vs.
  *Estimated* fidelity plot. If the companion folder is missing, the
  script falls back to the GHZ curves only.

All other functions, filenames, and CLI examples remain unchanged.
"""

import json
import os
import re
from math import sqrt
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------
# Matplotlib global style
# --------------------------------------------------------------------
WIDTH = 7
HEIGHT = WIDTH * (np.sqrt(5) - 1.0) / 2.5
plt.rcParams.update({
    "figure.figsize": (WIDTH, HEIGHT),
    "font.size": 12,
    "axes.axisbelow": True,
})

# --------------------------------------------------------------------
# GHZ result processing (original behaviour)
# --------------------------------------------------------------------

def process_results(file_path: str, re_expressions: str) -> Tuple[
    List[float],               # primary x‑axis factor values (sorted)
    Optional[List[float]],     # optional secondary factor values
    List[float],               # actual fidelities (mean)
    List[float],               # estimated fidelities (2·p_ghz−1)
]:
    """Parse a folder of GHZ JSON files (single‑run datasets)."""

    factors, second_vals_dict = [], {}
    actual_dict, est_dict = {}, {}
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
            p_ghz = list(data["p_ghz"].values())[0]
            est = 2 * p_ghz - 1

            actual_dict[factor] = actual
            est_dict[factor] = est

    if multi_factor:
        factors = sorted(second_vals_dict.keys())
        second_vals = [second_vals_dict[x] for x in factors]
    else:
        factors = sorted(factors)
        second_vals = None

    actual = [actual_dict[x] for x in factors]
    est = [est_dict[x] for x in factors]
    return factors, second_vals, actual, est

# --------------------------------------------------------------------
# CHSH companion processing (new)
# --------------------------------------------------------------------

def process_chsh_results(folder_chsh: str, re_expressions: str,
                         factors_order: List[float]) -> Tuple[List[float], List[float]]:
    """Return CHSH lower/upper fidelity bounds aligned with *factors_order*."""
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
            s_val = list(data["s_value"].values())[0]  # Run‑1
            low_d[factor] = s_val / (2 * sqrt(2))
            up_d[factor] = s_val / (4 * sqrt(2)) + 0.5

    low = [low_d.get(x, np.nan) for x in factors_order]
    up = [up_d.get(x, np.nan) for x in factors_order]
    return low, up

# --------------------------------------------------------------------
# Generic line‑plot helper
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
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()
    for x, y, leg in zip(xs, ys, legends):
        ax.plot(x, y, label=leg, marker="o")
        if annotation is not None:
            for i, (xi, yi) in enumerate(zip(x, y)):
                if i < len(annotation):
                    ax.annotate(f"{annotation[i]}", (xi, yi),
                                textcoords="offset points", xytext=(0, 5), ha="center")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(ncols=ncols)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{title}.png")
    plt.close()

# --------------------------------------------------------------------
# High‑level plotting routine
# --------------------------------------------------------------------

def plot_figure(folder: str,
                re_expr: str,
                title: str,
                x_label: str,
                save_dir: str = "./figures"):
    """Plot GHZ Actual/Estimated and CHSH bounds in one figure."""

    factors, second_vals, actual, est = process_results(folder, re_expr)
    low, up = process_chsh_results(f"{folder}_chsh", re_expr, factors)

    ys = [actual, est]
    legends = ["Actual Fidelity", "Estimated Fidelity"]
    if not np.all(np.isnan(low)):
        ys += [low, up]
        legends += ["CHSH Lower Bound", "CHSH Upper Bound"]

    annotation = second_vals if second_vals is not None else None
    plot_lines([factors] * len(ys), ys, legends,
               f"{title}_fid", x_label, "Fidelity",
               save_dir=save_dir, annotation=annotation, ncols=2)

# --------------------------------------------------------------------
# Convenience wrapper for two folders
# --------------------------------------------------------------------

def plot_figure_twin(folder1: str, folder2: str,
                      re_expr: str, title: str,
                      x_label: str, save_dir: str = "./figures"):
    plot_figure(folder1, re_expr, title, x_label, save_dir)
    plot_figure(folder2, re_expr, title, x_label, save_dir)

# --------------------------------------------------------------------
# CLI examples (LaTeX strings with single backslash)
# --------------------------------------------------------------------
if __name__ == "__main__":
    plot_figure("./change_sample_size_result_0.1_10000", r"sample_([\d\.]+)_alpha",
                "Change Sample Size", r"$\beta$")

    plot_figure_twin("./change_distance_result_0.1_10000", "./change_distance_result_0.1_10000",
                     r"distance_([\d\.]+)_", "Change Node Distance", r"$L$ (Km)")

    plot_figure_twin("./change_channel_rate_result_0.1_10000", "./change_channel_rate_result_0.1_10000",
                     r"channel_([\d\.]+)_", "Change Depolar Rate", r"$R_c$ (Hz)")

    plot_figure("./sample_rate_change_with_distance_0.1_10000",
                r"distance_([\d\.]+)_sample_([\d\.]+)",
                "Change Distance with Sample Rate", r"$L$ (Km)")

    plot_figure_twin("./change_alpha_result_0.1_10000", "./change_alpha_result_0.1_10000",
                     r"alpha_([\d\.]+).json", "Change Alpha", r"$\alpha$")
