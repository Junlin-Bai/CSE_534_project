import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

from glm import distance

# For experiment 2 (modified): fixed C=10000, channel depolar rate = 8xxxHz, α=0.05,
# sample rate = TBD (from experiment 1), and distance varying from 0.5km to 4km.
# We update the pattern to only include files with total_pairs-10000.
pattern = "./chsh_total_pairs-10000_memory_0_channel_8000.0_distance_*_sample_0.3_alpha_0.05.json"
files = glob.glob(pattern)

if not files:
    print("No files found. Please check the file pattern.")
    sys.exit(1)
else:
    print(f"Found {len(files)} files:")
    for f in files:
        print(f)

# Define alpha and the threshold for fidelity judgment.
alpha = 0.05
threshold = 1 - alpha  # 0.95

# Lists to store the distance (km) and computed error rates (error rate normalized over 10 keys: "0"–"9").
distances = []
false_positive_rates = {}
false_negative_rates = {}
teleport_fids = {}
actual_fids = {}
accept_rate = {}

for filename in files:
    # Extract the distance value from the filename. Expected pattern: ..._distance_<value>_...
    match = re.search(r"distance_([\d\.]+)_", filename)
    if match:
        distance_val = float(match.group(1))
    else:
        print(f"Could not extract distance from filename {filename}")
        continue

    with open(filename, "r") as f:
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

    print(f"{distance_val}, {fn_rate}, {fp_rate}")
    distances.append(distance_val)
    false_positive_rates[distance_val] = fp_rate
    false_negative_rates[distance_val] = fn_rate
    accept_rate[distance_val] = accept_count / len(actual_fid)
    if len(valid_teleport_fids) == 0:
        teleport_fids[distance_val] = 0
        actual_fids[distance_val] = 0
    else:
        teleport_fids[distance_val] = np.mean(valid_teleport_fids)
        actual_fids[distance_val] = np.mean(valid_actual_fid)

    # # Extract the relevant fields from the JSON.
    # actual_fid = data.get("actual_fid", {})
    # s_value = data.get("s_value", {})
    # theta = data.get("theta", {})
    #
    # false_positive_count = 0
    # false_negative_count = 0
    #
    # # Expected keys are "0" through "9"
    # keys = sorted(actual_fid.keys(), key=lambda x: int(x))
    # total_keys = len(keys)
    # if total_keys == 0:
    #     continue
    #
    # for key in keys:
    #     values = actual_fid[key]
    #     mean_fid = np.mean(values)
    #
    #     # Retrieve the corresponding s_value and theta.
    #     s_val = s_value.get(key)
    #     theta_val = theta.get(key)
    #     if s_val is None or theta_val is None:
    #         continue
    #
    #     # Prediction: if s_value > theta then predict that fidelity F is above threshold (i.e. "above"),
    #     # else predict "below".
    #     predicted = "above" if s_val > theta_val else "below"
    #     # Actual judgment: if the mean fidelity is at least threshold, consider it "above"; otherwise "below".
    #     actual = "above" if mean_fid >= threshold else "below"
    #
    #     # Count errors:
    #     if predicted == "above" and actual == "below":
    #         false_positive_count += 1
    #     elif predicted == "below" and actual == "above":
    #         false_negative_count += 1
    #
    # # Normalize the error counts over the total number of keys (expected to be 10).
    # fp_rate = false_positive_count / total_keys
    # fn_rate = false_negative_count / total_keys
    #
    # distances.append(distance_val)
    # false_positive_rates.append(fp_rate)
    # false_negative_rates.append(fn_rate)

if not distances:
    print("No valid data was processed. Please check the file contents.")
    sys.exit(1)

# Sort the results by distance.
# distances, false_positive_rates, false_negative_rates = zip(*sorted(
#     zip(distances, false_positive_rates, false_negative_rates)
# ))

distances = sorted(distances)
success_rate= [ 1 - false_negative_rates[x] - false_positive_rates[x] for x in distances]
false_positive_rates = [false_positive_rates[x] for x in distances]
false_negative_rates = [false_negative_rates[x] for x in distances]
accept_rate = [accept_rate[x] for x in distances]



actual_fids = [actual_fids[x] for x in distances]
teleport_fids = [teleport_fids[x] for x in distances]


# Plot the false positive and false negative rates versus distance.
plt.figure(figsize=(8, 6))
plt.plot(distances, false_positive_rates, marker='o', linestyle='-', label='False Positive Rate')
plt.plot(distances, false_negative_rates, marker='s', linestyle='-', label='False Negative Rate')
plt.xlabel('Distance (km)')
plt.ylabel('Error Rate')
plt.title('Ex2 (C=10000): False Positive & False Negative Rates vs Distance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/error_ex2.png")
plt.show()

# Plot the false positive and false negative rates versus distance.
plt.figure(figsize=(8, 6))
plt.plot(distances, actual_fids, marker='o', linestyle='-', label='Actual Fidelity')
plt.plot(distances, teleport_fids, marker='s', linestyle='-', label='Teleport Fidelity')
plt.xlabel('Distance (km)')
plt.ylabel('Fidelity Value')
plt.title('Ex2 (C=10000): Actual & Teleport Fidelity vs Distance')
plt.axhline(y=0.95, color='r', linestyle='--', label=f'Threshold ({threshold})')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/fid_ex2.png")
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(distances, success_rate , marker='o', linestyle='-', label='Success Rate')
plt.plot(distances, accept_rate, marker='s', linestyle='-', label='Accept Rate')
plt.xlabel('Distance (km)')
plt.ylabel('Percentage Rate')
plt.title('Ex2 (C=10000): Success Rate vs Distance')
# plt.axhline(y=0.95, color='r', linestyle='--', label=f'Threshold ({threshold})')

plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/success_ex2.png")
plt.show()