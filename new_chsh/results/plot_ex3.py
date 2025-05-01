import glob
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

# File pattern: Only process files with total_pairs-10000, distance=1.0, and a variable channel rate.
pattern = "./chsh_total_pairs-10000_memory_0_channel_*_distance_1.0_sample_0.3_alpha_0.05.json"
files = glob.glob(pattern)

if not files:
    print("No files found. Please check the file pattern.")
    sys.exit(1)
else:
    print(f"Found {len(files)} files:")
    for f in files:
        print(f)

# Set the judgment parameters.
alpha = 0.05
threshold = 1 - alpha  # 0.95

# Lists to collect the channel depolar rates and error rates.
channel_rates = []
false_positive_rates = {}
false_negative_rates = {}
teleport_fids = {}
actual_fids = {}
accept_rate = {}

for filename in files:
    # Extract the channel depolar rate from filename.
    # Expected pattern: "channel_<value>" (e.g., channel_1000.0)
    match = re.search(r"channel_([\d\.]+)_", filename)
    if match:
        channel_rate = float(match.group(1))
    else:
        print(f"Could not extract channel rate from filename {filename}")
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

    print(f"{channel_rate}, {fn_rate}, {fp_rate}")
    channel_rates.append(channel_rate)
    false_positive_rates[channel_rate] = fp_rate
    false_negative_rates[channel_rate] = fn_rate
    accept_rate[channel_rate] = accept_count / len(actual_fid)
    if len(valid_teleport_fids) == 0:
        teleport_fids[channel_rate] = 0
        actual_fids[channel_rate] = 0
    else:
        teleport_fids[channel_rate] = np.mean(valid_teleport_fids)
        actual_fids[channel_rate] = np.mean(valid_actual_fid)

if not channel_rates:
    print("No valid data was processed. Please check the file contents.")
    sys.exit(1)

# Sort the results by channel rate.
channel_rates = sorted(channel_rates)
success_rate= [ 1 - false_negative_rates[x] - false_positive_rates[x] for x in channel_rates]
false_positive_rates = [false_positive_rates[x] for x in channel_rates]
false_negative_rates = [false_negative_rates[x] for x in channel_rates]
accept_rate = [accept_rate[x] for x in channel_rates]



actual_fids = [actual_fids[x] for x in channel_rates]
teleport_fids = [teleport_fids[x] for x in channel_rates]

# Plot the error rates versus the channel depolar rate.
plt.figure(figsize=(8, 6))
plt.plot(channel_rates, false_positive_rates, marker='o', linestyle='-', label='False Positive Rate')
plt.plot(channel_rates, false_negative_rates, marker='s', linestyle='-', label='False Negative Rate')
plt.xlabel('Channel Depolar Rate (Hz)')
plt.ylabel('Error Rate')
plt.title('Ex3 (C=10000, Distance=1km): Error Rates vs Channel Depolar Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/error_ex3.png")
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(channel_rates, actual_fids, marker='o', linestyle='-', label='Actual Fidelity')
plt.plot(channel_rates, teleport_fids, marker='s', linestyle='-', label='Teleport Fidelity')
plt.xlabel('Channel Depolar Rate (Hz)')
plt.ylabel('Fidelity Value')
plt.title('Ex3 (C=10000): Actual & Teleport Fidelity vs Channel Depolar Rate')
plt.axhline(y=0.95, color='r', linestyle='--', label=f'Threshold ({threshold})')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/fid_ex3.png")
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(channel_rates, success_rate , marker='o', linestyle='-', label='Success Rate')
plt.plot(channel_rates, accept_rate, marker='s', linestyle='-', label='Accept Rate')
plt.xlabel('Channel Depolar Rate (Hz)')
plt.ylabel('Percentage Rate')
plt.title('Ex3 (C=10000): Success Rate vs Channel Depolar Rate')
# plt.axhline(y=0.95, color='r', linestyle='--', label=f'Threshold ({threshold})')

plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("./figures/success_ex3.png")
plt.show()




