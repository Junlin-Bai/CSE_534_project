import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

# Experiment ex5: new parameters
# Fixed total pairs (C) = 10,000
# Fixed channel depolar rate = 8000.0")
# Fixed distance = 1.0 km
# Sample rate is variable (extracted from filename)
# Testing multiple alpha values: from 0.20 down to 0.05


# Fixed parameters for experiment_5
# total_pairs = 10000
# channel_val = "8000.0"  # e.g., "8000.0" in the filename
# distance_val = "1.0"  # in km

# Create file pattern using f-string formatting:
pattern = "./chsh_total_pairs-10000_memory_0_channel_8000.0_distance_1.0_sample_0.3_alpha_*.json"

files = glob.glob(pattern)
if not files:
    print(f"No files found for alpha. Please check the file pattern.")
    sys, exit(1)
else:
    print(f"Found {len(files)} files:")
    for f in files:
        print(f)

# Dictionaries to store metrics keyed by sample value
alpha_valueues = []
thresholds = {}
false_positive_rates = {}
false_negative_rates = {}
teleport_fids_dict = {}
actual_fids_dict = {}
accept_rate = {}

for filename in files:
    # Extract sample value from filename using regex
    match = re.search(r"alpha_([\d\.]+).json", filename)
    if match:
        alpha_value = float(match.group(1))
        threshold = 1 - alpha_value  # Define threshold based on alpha value
    # else:
    #     print(f"Could not extract sample value from filename: {filename}")
    #     continue

    # Load the JSON file
    with open(filename, "r") as f:
        data = json.load(f)

    # Retrieve relevant data fields (assuming the JSON structure as before)
    actual_fid = list(data["actual_fid"].values())
    s_value = list(data["s_value"].values())
    theta = list(data["theta"].values())
    tele_fid_values = list(data["teleport_fid"].values())

    false_positive_count = 0
    false_negative_count = 0
    accept_count = 0

    # Calculate mean fidelities per run
    actual_fid = [np.mean(x) for x in actual_fid]
    tele_fid_mean = [np.mean(x) for x in tele_fid_values]

    valid_teleport_fids = []
    valid_actual_fid = []

    # Evaluate each pair of values
    for s, t, a_f, t_f in zip(s_value, theta, actual_fid, tele_fid_mean):
        # Prediction: if s_value > theta, predict fidelity is “above” threshold
        predicted = "above" if s > t else "below"
        # Actual: determine based on the measured fidelity value
        actual = "above" if a_f >= threshold else "below"

        # Count errors
        if predicted == "above" and actual == "below":
            false_positive_count += 1
        elif predicted == "below" and actual == "above":
            false_negative_count += 1

        # Save fidelity values if predicted as "above"
        if predicted == "above":
            valid_teleport_fids.append(t_f)
            valid_actual_fid.append(a_f)
            accept_count+=1

    # Avoid division by zero by checking length
    n_points = len(actual_fid)
    if n_points > 0:
        fp_rate = false_positive_count / n_points
        fn_rate = false_negative_count / n_points
    else:
        fp_rate = fn_rate = 0

    print(f"Sample {alpha_value}: False Negative Rate = {fn_rate}, False Positive Rate = {fp_rate}")

    alpha_valueues.append(alpha_value)
    false_positive_rates[alpha_value] = fp_rate
    false_negative_rates[alpha_value] = fn_rate
    accept_rate[alpha_value] = accept_count / n_points
    thresholds[alpha_value] = threshold
    if len(valid_teleport_fids) == 0:
        teleport_fids_dict[alpha_value] = 0
        actual_fids_dict[alpha_value] = 0
    else:
        teleport_fids_dict[alpha_value] = np.mean(valid_teleport_fids)
        actual_fids_dict[alpha_value] = np.mean(valid_actual_fid)

    # Check that some data was processed for this alpha
    if not alpha_valueues:
        print("No valid data was processed for this alpha value.")
        continue

# Sort the results by sample value
alpha_valueues = sorted(alpha_valueues)
success_rate = [1 - false_negative_rates[x] - false_positive_rates[x] for x in alpha_valueues]
fp_rates_sorted = [false_positive_rates[x] for x in alpha_valueues]
fn_rates_sorted = [false_negative_rates[x] for x in alpha_valueues]
actual_fids_list = [actual_fids_dict[x] for x in alpha_valueues]
teleport_fids_list = [teleport_fids_dict[x] for x in alpha_valueues]
accept_rate = [accept_rate[x] for x in alpha_valueues]
thresholds = [thresholds[x] for x in alpha_valueues]


# Plot Error Rates vs alpha value
plt.figure(figsize=(8, 6))
plt.plot(alpha_valueues, fp_rates_sorted, marker='o', linestyle='-', label='False Positive Rate')
plt.plot(alpha_valueues, fn_rates_sorted, marker='s', linestyle='-', label='False Negative Rate')
plt.xlabel('alpha value')
plt.ylabel('Error Rate')
plt.title(f'Error Rates vs alpha value')
plt.legend()
plt.grid(True)
error_fig_name = f"./figures/error_ex5_alpha.png"
plt.tight_layout()
plt.savefig(error_fig_name)
print(f"Saved error rates plot to: {error_fig_name}")
plt.show()

# ----------------------------
plt.figure(figsize=(8, 6))
plt.plot(alpha_valueues, actual_fids_list, marker='o', linestyle='-', label='Actual Fidelity')
plt.plot(alpha_valueues, teleport_fids_list, marker='s', linestyle='-', label='Teleport Fidelity')
plt.plot(alpha_valueues,thresholds, linestyle='--', label=f'Thresholds', color='r' )
plt.xlabel('alpha value')
plt.ylabel('Fidelity')
plt.title(f'Fidelity vs Changing Alpha Value')
plt.legend()
plt.grid(True)
fids_fig_name = f"./figures/fids_ex5.png"
plt.tight_layout()
plt.savefig(fids_fig_name)
print(f"Saved fidelity plot to: {fids_fig_name}")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(alpha_valueues, success_rate, marker='o', linestyle='-', label='Success Rate')
plt.plot(alpha_valueues, accept_rate, marker='s', linestyle='-', label='Accept Rate')

plt.xlabel('alpha value')
plt.ylabel('Percentage Rate')
plt.title(f'Success Rate vs Changing Alpha Value')
plt.legend()
plt.grid(True)
fids_fig_name = f"./figures/success_ex5.png"
plt.tight_layout()
plt.savefig(fids_fig_name)
print(f"Saved fidelity plot to: {fids_fig_name}")
plt.show()