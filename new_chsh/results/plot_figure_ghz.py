# -*- coding: utf-8 -*-
"""
Minimal‑change patch of the original plotting script for GHZ data.

Revision notes (English)
-----------------------
* Preserve the **s_value → p_ghz** remap and the computation
  **estimated_fidelity = 2 * p_ghz - 1**.
* **No data clipping or y‑axis limits**: distance‑tail trimming and explicit
  `ylim` settings have been removed so every data point is shown.
* All variable names and function signatures stay exactly the same to avoid
  breaking code that imports this module.
* Fixed LaTeX math strings in CLI examples (single backslash is required).
"""

import json
import os
import re
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

# ---------------- Global matplotlib style (unchanged) ----------------
width = 7
height = width * (np.sqrt(5) - 1.0) / 2.5
plt.rcParams["figure.figsize"] = (width, height)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.axisbelow"] = True

# ---------------- Core data processing ----------------

def process_results(file_path: str, re_expressions: str) -> Tuple[
    List[float],               # sorted primary factor values for the x‑axis
    Optional[List[float]],     # optional secondary factor values for annotation
    List[float],               # mean actual fidelities
    List[float],               # mean estimated fidelities (2·p_ghz−1)
]:
    """Parse experiment result files and collect data for plotting (single‑run datasets)."""

    all_factor_value, all_factor_dict = [], {}
    actual_fids_dict, estimated_fids_dict = {}, {}
    multi_factor = False

    for root, _, files in os.walk(file_path):
        for file in files:
            match = re.search(re_expressions, file)
            if not match:
                continue
            change_factor = float(match.group(1))
            if len(match.groups()) > 1:
                all_factor_dict[change_factor] = float(match.group(2))
                multi_factor = True
            else:
                all_factor_value.append(change_factor)

            with open(os.path.join(root, file), "r", encoding="utf‑8") as f:
                data = json.load(f)

            # ---- single‑run extraction ----
            actual_fid = np.mean(list(data["actual_fid"].values())[0])
            p_ghz = list(data["p_ghz"].values())[0]
            estimated_fid = 2 * p_ghz - 1

            actual_fids_dict[change_factor] = actual_fid
            estimated_fids_dict[change_factor] = estimated_fid

    if multi_factor:
        all_factor_value = sorted(all_factor_dict.keys())
        second_factor_value = [all_factor_dict[x] for x in all_factor_value]
    else:
        all_factor_value = sorted(all_factor_value)
        second_factor_value = None

    actual_fids = [actual_fids_dict[x] for x in all_factor_value]
    estimated_fids = [estimated_fids_dict[x] for x in all_factor_value]
    return all_factor_value, second_factor_value, actual_fids, estimated_fids

# ---------------- Plot helper (unchanged signature) ----------------

def plot_lines(xs: Sequence[Sequence[float]],
               ys: Sequence[Sequence[float]],
               legends: Sequence[str],
               title: str,
               x_label: str,
               y_label: str,
               save_dir: str = "./figures",
               annotation: Optional[Sequence[float]] = None,
               ncols: int = 1):
    """Generic line‑plot helper used by the high‑level routines."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()
    for x, y, legend in zip(xs, ys, legends):
        ax.plot(x, y, label=legend, marker="o")
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

# ---------------- High‑level figure routine ----------------

def plot_figure(file_path: str,
                re_expressions: str,
                title: str,
                change_factor: str,
                save_dir: str = "./figures"):
    """Plot Actual vs. Estimated Fidelity for a single parameter sweep."""
    (all_factor_value,
     second_factor_value,
     actual_fids,
     estimated_fids) = process_results(file_path, re_expressions)

    annotation = second_factor_value if second_factor_value is not None else None

    # Fidelity comparison curve (full data range)
    plot_lines([all_factor_value] * 2,
               [actual_fids, estimated_fids],
               ["Actual Fidelity", "Estimated Fidelity"],
               f"{title}_fid", change_factor, "Fidelity",
               save_dir=save_dir, annotation=annotation)

# ---------------- Twin wrapper (two directories) ----------------

def plot_figure_twin(file_path_1: str, file_path_2: str,
                      re_expressions: str, title: str,
                      change_factor: str, save_dir: str = "./figures"):
    """Convenience function to plot the same sweep from two folders."""
    plot_figure(file_path_1, re_expressions, title, change_factor, save_dir)
    plot_figure(file_path_2, re_expressions, title, change_factor, save_dir)

# ---------------- CLI examples (unchanged except math strings) --------------
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
