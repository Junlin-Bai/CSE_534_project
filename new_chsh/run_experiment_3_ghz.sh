#!/bin/bash

# Script to run sim_ghz.py with different channel depolar rates concurrently
# Experiment 3:
#   Fixed parameters: C=10,000, alpha 0.05, sample rate = TBD (from experiment 1), distance = 1km
#   Varying parameter: channel depolar rate from 1000 Hz to 16000 Hz

# Base parameters
TOTAL_PAIR=60000
RUNS=1
SAMPLE_RATE=0.5           # Placeholder: update based on experiment 1 results
DISTANCE=1                # Fixed distance = 1km
ALPHA=0.1
SAVE_DIR="./results/change_channel_rate_result_${ALPHA}_${TOTAL_PAIR}"

# Function to run the simulation with a specific channel depolar rate
run_simulation() {
    local channel_rate=$1
    echo "Starting simulation with channel_rate = $channel_rate Hz, alpha = $ALPHA"
    python sim_ghz.py --sample_rate $SAMPLE_RATE --total_pair $TOTAL_PAIR --runs $RUNS --channel_rate $channel_rate --distance $DISTANCE --alpha $ALPHA --save_dir $SAVE_DIR
    echo "Completed simulation with channel_rate = $channel_rate Hz"
}

# Run simulations concurrently for channel depolar rates from 1000 Hz to 16000 Hz
for channel_rate in 1000 4000 7000 10000 13000 16000; do
    run_simulation $channel_rate &
done

# Wait for all background processes to complete
wait

echo "All simulations completed!"
