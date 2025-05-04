#!/bin/bash

# Script to run sim_chsh.py with different distances concurrently
# Experiment 2:
#   Fixed parameters: C=10,000, channel depolar rate = 8xxxHz, alpha 0.05,
#                     sample rate = TBD (from experiment 1)
#   Varying parameter: distance from 0.5 km to 4 km

# Base parameters
TOTAL_PAIR=60000
RUNS=1
CHANNEL_RATE=8000
SAMPLE_RATE=0.5        # Placeholder: update based on experiment 1 results
ALPHA=0.1
SAVE_DIR="./results/change_distance_result_${ALPHA}_${TOTAL_PAIR}_chsh"

# Function to run the simulation with a specific distance
run_simulation() {
    local distance=$1
    echo "Starting simulation with distance = $distance, alpha = $ALPHA"
    python sim_chsh.py --sample_rate $SAMPLE_RATE --total_pair $TOTAL_PAIR --runs $RUNS --channel_rate $CHANNEL_RATE --distance $distance --alpha $ALPHA --save_dir $SAVE_DIR
    echo "Completed simulation with distance = $distance"
}

# Run simulations concurrently for distances from 0.5km to 4km
for distance in 0.5 1 1.5 2 2.5 3 3.5 4; do
    run_simulation $distance &
done

# Wait for all background processes to complete
wait

echo "All simulations completed!"
