#!/usr/bin/env bash

# Checks if the user passed in input
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

INPUT_FILE=$1
NUM_RUNS=10
BEST_FITNESS=9999999999999   # A large initial value
BEST_SEED=-1
SUM_TIME=0

# If you want to store the best overall output, name it:
BEST_OUTPUT="serial_best.txt"

# Clean up any old best file
rm -f "$BEST_OUTPUT"

echo "Running $NUM_RUNS trials on '$INPUT_FILE'..."
echo "==========================================="

for SEED in $(seq 1 $NUM_RUNS); do
  echo "Run #$SEED with SEED=$SEED ..."
  
  # Record start time (in seconds + nanoseconds)
  START_TIME=$(date +%s.%N)
  
  # Run your program
  # We direct stdout/stderr to a temporary file so we can parse it
  ./revROC "$INPUT_FILE" "$SEED" > "tmp_output_$SEED.txt" 2>&1
  
  # Record end time
  END_TIME=$(date +%s.%N)
  
  # Calculate elapsed time
  ELAPSED=$(echo "$END_TIME - $START_TIME" | bc -l)
  
  # Add to the sum of all times
  SUM_TIME=$(echo "$SUM_TIME + $ELAPSED" | bc -l)
  FITNESS_VALUE=$(grep "Fitness:" "tmp_output_$SEED.txt" | awk '{print $2}')
  
  # Print out time and Fitness for this run
  echo "  Time elapsed: $ELAPSED seconds"
  echo "  Fitness:      $FITNESS_VALUE"
  
  if [[ -n "$FITNESS_VALUE" ]] && \
     (( $(echo "$FITNESS_VALUE < $BEST_FITNESS" | bc -l) )); then
    
    BEST_FITNESS=$FITNESS_VALUE
    BEST_SEED=$SEED
    
    # Save this run’s output as the best so far
    cp "tmp_output_$SEED.txt" "$BEST_OUTPUT"
  fi
  
  echo "-------------------------------------------"
done

# Compute average time
AVG_TIME=$(echo "scale=4; $SUM_TIME / $NUM_RUNS" | bc -l)

# Final summary
echo "Done!"
echo "==========================================="
echo "Best Fitness found: $BEST_FITNESS (Seed=$BEST_SEED)"
echo "Average Time over $NUM_RUNS runs: $AVG_TIME seconds"
echo "Best output stored in: $BEST_OUTPUT"
echo "==========================================="

# End of script
