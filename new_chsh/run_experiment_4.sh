#!/bin/bash

# Script to run sim_chsh.py with different distances concurrently
# Experiment 4:
#   Fixed parameters: C=10,000, channel depolar rate = 8xxxHz, alpha 0.05,
#   sample rate increasing based on distance (computed linearly from 0.1 to 0.7),

#   Here the sample rate is computed as:
#       sample_rate = 0.24 * distance - 0.02
#   which yields 0.1 at 0.5 km and 0.7 at 3 km.

# Base parameters
TOTAL_PAIR=10000
RUNS=200
CHANNEL_RATE=8000
ALPHA=0.1
SAVE_DIR="./results/sample_rate_change_with_distance_${ALPHA}_${TOTAL_PAIR}"

# Function to run the simulation for a specific distance.
# It computes sample_rate based on the distance.
run_simulation() {
    local distance=$1
    # Compute sample_rate using a linear relation:
    # For distance = 0.5 km -> sample_rate = 0.24*0.5 - 0.02 = 0.10
    # For distance = 3 km   -> sample_rate = 0.24*3 - 0.02 = 0.70
    local sample_rate=$(awk "BEGIN {printf \"%.2f\", 0.24 * $distance - 0.02}")
    echo "Starting simulation with distance = $distance km and sample_rate = $sample_rate, alpha = $ALPHA"
    python sim_chsh.py --sample_rate $sample_rate --total_pair $TOTAL_PAIR --runs $RUNS --channel_rate $CHANNEL_RATE --distance $distance --alpha $ALPHA --save_dir $SAVE_DIR
    echo "Completed simulation with distance = $distance km"
}

# Run simulations concurrently for distances from 0.5 km to 3 km
for distance in 0.5 1 1.5 2 2.5 3; do
    run_simulation $distance &
done

# Wait for all background processes to complete
wait

echo "All simulations completed!"
