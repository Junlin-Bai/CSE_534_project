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


def process_results(file_path, re_expressions):
    """
    Parse experiment result files and collect data for plotting.
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
                # Prediction: if s_value > theta, predict fidelity is "above"
                predicted = "above" if s > t else "below"
                # Actual judgment based on mean fidelity
                actual = "above" if a_f >= threshold else "below"

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
    # Sort values for plotting.
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


def plot_figure_twin(file_path_1, file_path_2, re_expressions, title, change_factor, save_dir="./figures"):
    """
    Create three separate figures (one for Fidelity, one for Success Rate, and one for Error Rate)
    that combine the results from two data sources.
    """
    # Process results for both file paths.
    (x1, second_factor_value1, success_rate1, false_positive_rates1,
     false_negative_rates1, accept_rate1, actual_fids1,
     teleport_fids1, thresholds1) = process_results(file_path_1, re_expressions)

    (x2, second_factor_value2, success_rate2, false_positive_rates2,
     false_negative_rates2, accept_rate2, actual_fids2,
     teleport_fids2, thresholds2) = process_results(file_path_2, re_expressions)

    # For combining, assume x1 and x2 are similar.
    all_factor_value = x1
    annotation = second_factor_value1 if second_factor_value1 is not None else None
    if "Distance" in change_factor:
        all_factor_value = all_factor_value[:-2]
        actual_fids1 = actual_fids1[:-2]
        teleport_fids1 = teleport_fids1[:-2]
        thresholds1 = thresholds1[:-2]
        success_rate1 = success_rate1[:-2]
        false_positive_rates1 = false_positive_rates1[:-2]
        false_negative_rates1 = false_negative_rates1[:-2]
        accept_rate1 = accept_rate1[:-2]
        actual_fids2 = actual_fids2[:-2]
        teleport_fids2 = teleport_fids2[:-2]
        thresholds2 = thresholds2[:-2]
        success_rate2 = success_rate2[:-2]
        false_positive_rates2 = false_positive_rates2[:-2]
        false_negative_rates2 = false_negative_rates2[:-2]
        accept_rate2 = accept_rate2[:-2]

    # Ensure save directory exists.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ------------------ Fidelity Figure ------------------
    fig, ax = plt.subplots()
    ax.plot(all_factor_value, actual_fids1, label=r"Actual F $\beta=0.5$", marker="o")
    ax.plot(all_factor_value, teleport_fids1, label=r"Teleport F $\beta=0.5$", marker="s")
    ax.plot(all_factor_value, actual_fids2, label=r"Actual F $\beta=0.3$", marker="o")
    ax.plot(all_factor_value, teleport_fids2, label=r"Teleport F $\beta=0.3$", marker="s")
    ax.set_xlabel(change_factor)
    ax.set_ylabel("Fidelity")
    if "Sample" in change_factor or "Alpha" in change_factor:
        ax.set_ylim(0.965, 0.99)
    if annotation is not None:
        for i, xi in enumerate(all_factor_value):
            ax.annotate(f"{annotation[i]}", (xi, actual_fids1[i]),
                        textcoords="offset points", xytext=(0, 5), ha='center')
    ax.legend()
    ax.set_title(f"{title} vs Fidelity")
    plt.tight_layout()
    fidelity_save_path = os.path.join(save_dir, title + "_fidelity_twin.png")
    plt.savefig(fidelity_save_path)
    plt.show()

    # ------------------ Success Rate Figure ------------------
    fig, ax = plt.subplots()
    ax.plot(all_factor_value, success_rate1, label=r"Success Rate with $\beta=0.5$", marker="o")
    ax.plot(all_factor_value, success_rate2, label=r"Success Rate with $\beta=0.3$", marker="o")
    ax.set_xlabel(change_factor)
    ax.set_ylabel("Success Rate")
    if annotation is not None:
        for i, xi in enumerate(all_factor_value):
            ax.annotate(f"{annotation[i]}", (xi, success_rate1[i]),
                        textcoords="offset points", xytext=(0, 5), ha='center')
    ax.legend()
    ax.set_title(f"Success Rate vs {title}")
    plt.tight_layout()
    success_save_path = os.path.join(save_dir, title + "_success_twin.png")
    plt.savefig(success_save_path)
    plt.show()

    # ------------------ Error Rate Figure ------------------
    fig, ax = plt.subplots()
    ax.plot(all_factor_value, false_positive_rates1, label=r"False Positive with $\beta=0.5$", marker="s")
    ax.plot(all_factor_value, false_negative_rates1, label=r"False Negative with $\beta=0.5$", marker="s")
    ax.plot(all_factor_value, false_positive_rates2, label=r"False Positive with $\beta=0.3$", marker="s")
    ax.plot(all_factor_value, false_negative_rates2, label=r"False Negative with $\beta=0.3$", marker="s")
    ax.set_xlabel(change_factor)
    ax.set_ylabel("Error Rate")
    if annotation is not None:
        for i, xi in enumerate(all_factor_value):
            ax.annotate(f"{annotation[i]}", (xi, false_positive_rates1[i]),
                        textcoords="offset points", xytext=(0, 5), ha='center')
    ax.legend()
    ax.set_title(f"Error Rate vs {title}")
    plt.tight_layout()
    error_save_path = os.path.join(save_dir, title + "_error_twin.png")
    plt.savefig(error_save_path)
    plt.show()


if __name__ == "__main__":
    # Example calls to plot_figure_twin.
    plot_figure_twin(
        "./change_distance_result_0.10_10000_sample_0.5",
        "./change_distance_result_0.1_10000",
        r"distance_([\d\.]+)_",
        "Change Node Distance",
        "Distance (Km)"
    )

    plot_figure_twin(
        "./change_channel_rate_result_0.1_10000_sample_0.5",
        "./change_channel_rate_result_0.1_10000",
        r"channel_([\d\.]+)_",
        "Change Depolarization Rate",
        r" $R_c$ (Hz)"
    )

    plot_figure_twin(
        "./change_alpha_result_0.1_10000_sample_0.5",
        "./change_alpha_result_0.1_10000",
        r"alpha_([\d\.]+).json",
        r"Change $\alpha$",
        r"$\alpha$"
    )
