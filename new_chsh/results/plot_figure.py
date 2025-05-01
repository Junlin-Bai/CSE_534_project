import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



width = 7
height = width * (np.sqrt(5) - 1.0) / 2.5
plt.rcParams['figure.figsize'] = (width, height)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.axisbelow'] = True

def process_results(file_path,  re_expressions):
    """
    parse experiment result files and collect data for plotting
    """
    all_factor_value = []
    all_factor_dict = {}
    false_positive_rates = {}
    false_negative_rates = {}
    teleport_fids = {}
    actual_fids = {}
    accept_rate = {}
    thresholds = []
    multi_factor = False
    for root, _, files in os.walk(file_path):
        for file in files:
            match = re.search(re_expressions, file)
            if not match:
                print(f"Error no match found for file {file}")
                continue
            change_factor = float(match.group(1))

            threshold_search = re.search(r"alpha_(\d+\.\d+)", file)
            if not threshold_search:
                print(f"Error no threshold found for file {file}")
                continue
            threshold = 1 - float(threshold_search.group(1))
            thresholds.append(threshold)

            with open(os.path.join(root, file), "r") as f:
                data = json.load(f)

            actual_fid = list(data["actual_fid"].values())
            s_value = list(data["s_value"].values())
            theta = list(data["theta"].values())
            tele_fids = list(data["teleport_fid"].values())

            false_positive_count = 0
            false_negative_count = 0
            accept_count = 0
            # average fid for all run
            actual_fid = [np.mean(x) for x in actual_fid]
            tele_fids = [np.mean(x) for x in tele_fids]

            valid_teleport_fids = []
            valid_actual_fid = []
            for s, t, a_f, t_f in zip(s_value, theta, actual_fid, tele_fids):
                # Prediction: if s_value > theta, then predict fidelity F is "above" (i.e. >0.95)
                predicted = "above" if s > t else "below"
                # Actual judgment based on mean fidelity
                actual = "above" if a_f >= threshold else "below"

                # Count errors:
                if predicted == "above" and actual == "below":
                    false_positive_count += 1
                elif predicted == "below" and actual == "above":
                    false_negative_count += 1

                if predicted == "above":
                    valid_teleport_fids.append(t_f)
                    valid_actual_fid.append(a_f)
                    accept_count += 1

            fp_rate = false_positive_count / len(actual_fid)
            fn_rate = false_negative_count / len(actual_fid)
            print(f"{change_factor}, {fn_rate}, {fp_rate}")
            if len(match.groups()) > 1:
                all_factor_dict[change_factor] = float(match.group(2))
                multi_factor = True
            else:
                all_factor_value.append(change_factor)
            false_positive_rates[change_factor] = fp_rate
            false_negative_rates[change_factor] = fn_rate
            accept_rate[change_factor] = accept_count / len(actual_fid)
            teleport_fids[change_factor] = np.mean(valid_teleport_fids)
            actual_fids[change_factor] = np.mean(valid_actual_fid)
    # sort values for plotting
    second_factor_value = None
    if multi_factor:
        all_factor_value = sorted(all_factor_dict.keys())
        second_factor_value = [all_factor_dict[x] for x in all_factor_value]
    else:
        all_factor_value = sorted(all_factor_value)

    success_rate = [1 - false_negative_rates[x] - false_positive_rates[x] for x in all_factor_value]
    false_positive_rates = [false_positive_rates[x] for x in all_factor_value]
    false_negative_rates = [false_negative_rates[x] for x in all_factor_value]
    accept_rate = [accept_rate[x] for x in all_factor_value]
    actual_fids = [actual_fids[x] for x in all_factor_value]
    teleport_fids = [teleport_fids[x] for x in all_factor_value]

    return (all_factor_value,
            second_factor_value,
            success_rate,
            false_positive_rates,
            false_negative_rates,
            accept_rate,
            actual_fids,
            teleport_fids,
            thresholds)

def plot_lines(xs, ys, legends, title, x_label, y_label, save_dir="./figures", xlim=None,
               ylim=None, ncols=None, annotation=None):

    fig, ax = plt.subplots()

    for x, y, legend in zip(xs, ys, legends):
        ax.plot(x, y, label=legend, marker="o" if "Fidelity" in legend else "s")
        # Add annotation to points if provided
        if annotation is not None:
            for i, (xi, yi) in enumerate(zip(x, y)):
                if i < len(annotation):
                    ax.annotate(f"{annotation[i]}", (xi, yi),
                                textcoords="offset points",
                                xytext=(0, 5),  # 5 points vertical offset
                                ha='center')  # Horizontal alignment
    # ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    if ncols:
        ax.legend(ncols=ncols)
    else:
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, title + ".png"))
    plt.show()

def plot_figure(file_path, re_expressions, title, change_factor, save_dir="./figures"):
    (all_factor_value,
     second_factor_value,
     success_rate,
     false_positive_rates,
     false_negative_rates,
     accept_rate,
     actual_fids,
     teleport_fids,
     thresholds) = process_results(file_path, re_expressions)

    # plot fidelity figure
    if "Sample" in change_factor:
        ylim = (0.965, 0.99)
        ncols = 2
        fid_text = "(α = 0.15)"
    elif "Alpha" in change_factor:
        ylim = (0.965, 0.99)
        ncols = 2
        fid_text = ""
    else:
        ylim = None
        ncols = None
        fid_text = ""
    if r"$L$ (Km)" in change_factor:
        all_factor_value = all_factor_value[:-2]
        actual_fids = actual_fids[:-2]
        teleport_fids = teleport_fids[:-2]
        thresholds = thresholds[:-2]
        success_rate = success_rate[:-2]
        false_positive_rates = false_positive_rates[:-2]
        false_negative_rates = false_negative_rates[:-2]
        accept_rate = accept_rate[:-2]
        
        xlim = None
    else:
        xlim = None
    if second_factor_value is not None:
        annotation = second_factor_value
    else:
        annotation = None
    plot_lines([all_factor_value]*2, [actual_fids, teleport_fids],
               [f"Actual Fidelity {fid_text}", f"Teleport Fidelity {fid_text}"],
               f"{title}_fid", change_factor, "Fidelity", save_dir, ylim=ylim, xlim=xlim,
               ncols=ncols, annotation=annotation)
    # plot success rate
    plot_lines([all_factor_value], [success_rate],
               ["Success Rate"],
               f"{title}_success", change_factor, "Success Rate", save_dir,annotation=annotation, xlim=xlim)
    # plot error
    plot_lines([all_factor_value]*2, [false_positive_rates, false_negative_rates],
               ["False Positive Rate", "False Negative Rate"], f"{title}_error", change_factor
               , "Error Rate", save_dir, annotation=annotation, xlim=xlim)

def plot_figure_twin(file_path_1,file_path_2, re_expressions, title, change_factor, save_dir="./figures"):
    plot_figure(file_path_1, re_expressions, title, change_factor, save_dir="./figures")
    plot_figure(file_path_2, re_expressions, title, change_factor, save_dir="./figures")

if __name__ == "__main__":
    # start plotting
    plot_figure("./change_sample_size_result_0.1_10000", r"sample_([\d\.]+)_alpha",
                "Change Sample Size", r"$\beta$")

    plot_figure_twin("./change_distance_result_0.1_10000", "./change_distance_result_0.1_10000", r"distance_([\d\.]+)_",
                "Change Node Distance", r"$L$ (Km)")

    plot_figure_twin("./change_channel_rate_result_0.1_10000","./change_channel_rate_result_0.1_10000", r"channel_([\d\.]+)_",
                "Change Depolar Rate", r"$R_c$ (Hz)")

    plot_figure("./sample_rate_change_with_distance_0.1_10000", r"distance_([\d\.]+)_sample_([\d\.]+)",
                "Change Distance with Sample Rate", r"$L$ (Km)")

    plot_figure_twin("./change_alpha_result_0.1_10000", "./change_alpha_result_0.1_10000", r"alpha_([\d\.]+).json",
                "Change Alpha", r"$\alpha$")
