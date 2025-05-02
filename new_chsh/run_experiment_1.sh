#!/bin/bash

# Script to run sim_chsh.py with different sample rates concurrently
# Usage: ./run_sim_concurrent.sh

# Base parameters
TOTAL_PAIR=10000
RUNS=1
CHANNEL_RATE=8000
DISTANCE=1
ALPHA=0.1
SAVE_DIR="./results/change_sample_size_result_${ALPHA}_${TOTAL_PAIR}"

# Function to run the simulation with a specific sample rate
run_simulation() {
    local sample_rate=$1
    echo "Starting simulation with sample_rate = $sample_rate, alpha = $ALPHA"
    python sim_chsh.py --sample_rate $sample_rate --total_pair $TOTAL_PAIR --runs $RUNS --channel_rate $CHANNEL_RATE --distance $DISTANCE --alpha $ALPHA --save_dir $SAVE_DIR
    echo "Completed simulation with sample_rate = $sample_rate"
}

# Run simulations concurrently with sample rates from 0.1 to 0.7
for rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
    run_simulation $rate &
done

# Wait for all background processes to complete
wait

echo "All simulations completed!"
