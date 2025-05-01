#!/bin/bash

# Script to run sim_chsh.py with different alpha values concurrently
# Experiment 5:
#   Fixed parameters: C=10,000, channel depolar rate = 8xxxHz, distance = 1km, sample_rate = TBD (from experiment 1)
#   Varying parameter: alpha from 0.20 to 0.05

# Base parameters
TOTAL_PAIR=10000
RUNS=200
SAMPLE_RATE=0.3     # Placeholder: update based on experiment 1 results
CHANNEL_RATE=8000   # Fixed channel depolar rate (8xxxHz)
DISTANCE=1          # Fixed distance = 1km
ALPHA=0.1
SAVE_DIR="./results/change_alpha_result_${ALPHA}_${TOTAL_PAIR}"

# Function to run the simulation for a specific alpha value
run_simulation() {
    local alpha=$1
    echo "Starting simulation with alpha = $alpha"
    python sim_chsh.py --sample_rate $SAMPLE_RATE --total_pair $TOTAL_PAIR --runs $RUNS --channel_rate $CHANNEL_RATE --distance $DISTANCE --alpha $alpha --save_dir $SAVE_DIR
    echo "Completed simulation with alpha = $alpha"
}

# Run simulations concurrently for different alpha values from 0.20 to 0.05
for alpha in 0.20 0.17 0.14 0.11 0.08 0.05; do
    run_simulation $alpha &
done

# Wait for all background processes to complete
wait

echo "All simulations completed!"
