#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <input_file>"
  exit 1
fi

INPUT_FILE=$1
NUM_RUNS=10
BEST_FITNESS=9999999999999   # Very large initial value
BEST_SEED=-1
SUM_TIME=0
BEST_OUTPUT="serial_best.txt"

# Remove any old best file
rm -f "$BEST_OUTPUT"

echo "Running $NUM_RUNS trials on '$INPUT_FILE'..."
echo "==========================================="

for SEED in $(seq 1 $NUM_RUNS); do
  echo "Run #$SEED with SEED=$SEED ..."

  # Record start time
  START_TIME=$(date +%s.%N)

  # Run the program, redirecting all output to a temporary file
  ./revGOL "$INPUT_FILE" "$SEED" > "tmp_output_$SEED.txt" 2>&1

  # Record end time
  END_TIME=$(date +%s.%N)

  # Calculate elapsed time
  ELAPSED=$(echo "$END_TIME - $START_TIME" | bc -l)

  # Add to total time
  SUM_TIME=$(echo "$SUM_TIME + $ELAPSED" | bc -l)

  # Extract fitness. Adjust grep/awk if your output format is different.
  FITNESS_VALUE=$(grep "Fitness:" "tmp_output_$SEED.txt" | awk '{print $2}')

  echo "  Time elapsed: $ELAPSED seconds"
  echo "  Fitness:      $FITNESS_VALUE"

  # Update best fitness if this run is better (lower fitness)
  if [[ -n "$FITNESS_VALUE" ]] && \
     (( $(echo "$FITNESS_VALUE < $BEST_FITNESS" | bc -l) )); then
    BEST_FITNESS=$FITNESS_VALUE
    BEST_SEED=$SEED
    cp "tmp_output_$SEED.txt" "$BEST_OUTPUT"
  fi

  echo "-------------------------------------------"
done

# Compute average time
AVG_TIME=$(echo "scale=4; $SUM_TIME / $NUM_RUNS" | bc -l)

echo "Done!"
echo "==========================================="
echo "Best Fitness found: $BEST_FITNESS (Seed=$BEST_SEED)"
echo "Average Time over $NUM_RUNS runs: $AVG_TIME seconds"
echo "Best output stored in: $BEST_OUTPUT"
echo "==========================================="
