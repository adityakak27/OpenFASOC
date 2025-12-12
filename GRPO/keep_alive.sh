#!/bin/bash

# Script to keep machine active by continuously writing to an output file
# Run with: nohup ./keep_alive.sh &

OUTPUT_FILE="keep_alive_output.txt"

while true; do
    # Write current timestamp to file (overwrites each time)
    echo "Machine is active - $(date '+%Y-%m-%d %H:%M:%S')" > "$OUTPUT_FILE"
    
    # Sleep for 27 seconds before next write
    sleep 27
done

