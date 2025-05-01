import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

# File pattern: process files with total_pairs-10000 and variable distance
pattern = "./chsh_total_pairs-10000_memory_0_channel_8000.0_distance_*_sample_*_alpha_0.05.json"
files = glob.glob(pattern)

if not files:
    print("No files found. Please check the file pattern.")
    sys.exit(1)
else:
    print(f"Found {len(files)} files:")
    for f in files:
        print(f)

# Fixed alpha value and thus fidelity threshold 1-α.
alpha = 0.05
threshold = 1 - alpha  # 0.95
sample_rates = {
    0.5:0.1,
    1.0:0.22,
    1.5:0.34,
    2.0:0.46,
    2.5:0.58,
    3.0:0.7,
}
# Lists to store extracted distances and corresponding error rates.
distances = []
teleport_fids = {}
actual_fids = {}
accept_rate = {}
false_positive_rates = {}
false_negative_rates = {}
sample_values = {}  # Store sample values for later annotation

for filename in files:
    # Extract the distance value from the filename.
    # Expected pattern: ..._distance_<value>_...
    match = re.search(r"distance_([\d\.]+)_sample_([\d\.]+)", filename)

    if match:
        distance = float(match.group(1))  # First capture group - distance value
        sample = float(match.group(2))  # Second capture group - sample value
        if distance>3.0 or sample != sample_rates[distance]:
            continue
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
    print(f"{distance}, {fn_rate}, {fp_rate}")
    distances.append(distance)
    false_positive_rates[distance] = fp_rate
    false_negative_rates[distance] = fn_rate
    accept_rate[distance] = accept_count / len(actual_fid)
    sample_values[distance] = sample  # Store sample value for annotation
    if len(valid_teleport_fids) == 0:
        teleport_fids[distance] = 0
        actual_fids[distance] = 0
    else:
        teleport_fids[distance] = np.mean(valid_teleport_fids)
        actual_fids[distance] = np.mean(valid_actual_fid)

if not distances:
    print("No valid data was processed. Please check the file contents.")
    sys.exit(1)

# Sort results by distance.
distances = sorted(distances)
success_rate = [1 - false_negative_rates[x] - false_positive_rates[x] for x in distances]
false_positive_rates = [false_positive_rates[x] for x in distances]
false_negative_rates = [false_negative_rates[x] for x in distances]
accept_rate = [accept_rate[x] for x in distances]
sample_values_list = [sample_values[x] for x in distances]  # Get sample values in sorted order

actual_fids = [actual_fids[x] for x in distances]
teleport_fids = [teleport_fids[x] for x in distances]

# Plot the error rates versus distance.
plt.figure(figsize=(8, 6))
plt.plot(distances, false_positive_rates, marker='o', linestyle='-', label='False Positive Rate')
plt.plot(distances, false_negative_rates, marker='s', linestyle='-', label='False Negative Rate')
plt.xlabel('Distance (km)')
plt.ylabel('Error Rate')
plt.title('Ex4 (C=10000, Channel=8xxxHz, α=0.05, Sample Rate Varies):\nError Rates vs Distance')

# Add sample rate annotations to each point
for i, txt in enumerate(sample_values_list):
    # For false positive points
    plt.annotate(f"s={txt}",
                 (distances[i], false_positive_rates[i]),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8)
    # For false negative points
    plt.annotate(f"s={txt}",
                 (distances[i], false_negative_rates[i]),
                 xytext=(5, -10), textcoords='offset points',
                 fontsize=8)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/error_ex4.png")
plt.show()

# Plot the false positive and false negative rates versus distance.
plt.figure(figsize=(8, 6))
plt.plot(distances, actual_fids, marker='o', linestyle='-', label='Actual Fidelity')
plt.plot(distances, teleport_fids, marker='s', linestyle='-', label='Teleport Fidelity')
plt.xlabel('Distance (km)')
plt.ylabel('Fidelity Value')
plt.title('Ex3 (C=10000): Actual & Teleport Fidelity vs Distance, Sample Rate Varies')
plt.axhline(y=0.95, color='r', linestyle='--', label=f'Threshold ({threshold})')

# Add sample rate annotations to each point
for i, txt in enumerate(sample_values_list):
    # For actual fidelity points
    plt.annotate(f"s={txt}",
                 (distances[i], actual_fids[i]),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8)
    # For teleport fidelity points
    plt.annotate(f"s={txt}",
                 (distances[i], teleport_fids[i]),
                 xytext=(5, -10), textcoords='offset points',
                 fontsize=8)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/fid_ex4.png")
plt.show()

# Plot success rate and accept rate
plt.figure(figsize=(8, 6))
plt.plot(distances, success_rate, marker='o', linestyle='-', label='Success Rate')
plt.plot(distances, accept_rate, marker='s', linestyle='-', label='Accept Rate')
plt.xlabel('Distance (km)')
plt.ylabel('Percentage Rate')
plt.title('Ex2 (C=10000): Success Rate vs Distance, Sample Rate Varies')
# plt.axhline(y=0.95, color='r', linestyle='--', label=f'Threshold ({threshold})')

# Add sample rate annotations to each point
for i, txt in enumerate(sample_values_list):
    # For success rate points
    plt.annotate(f"s={txt}",
                 (distances[i], success_rate[i]),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8)
    # For accept rate points
    plt.annotate(f"s={txt}",
                 (distances[i], accept_rate[i]),
                 xytext=(5, -10), textcoords='offset points',
                 fontsize=8)

plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/success_ex4.png")
plt.show()