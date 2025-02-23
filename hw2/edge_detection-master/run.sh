#!/bin/bash

runs=10
TIMEFORMAT=%R
sum_cube=0
sum_sparty=0
sum_stadium=0
sum_earth=0

# Ensure the binary exists before running benchmarks
if [[ ! -f "./process" ]]; then
    echo "Error: ./process not found! Compile it first using 'make'."
    exit 1
fi

echo "Benchmarking ./process for $runs runs..."

for i in $(seq 1 $runs); do
    # Measure execution time of ./process
    t_run_cube=$( { time ./process ./images/cube.png test.png > /dev/null 2>&1; } 2>&1 )
    t_run_earth=$( { time ./process ./images/earth.png test.png > /dev/null 2>&1; } 2>&1 )
    t_run_sparty=$( { time ./process ./images/sparty.png test.png > /dev/null 2>&1; } 2>&1 )
    t_run_stadium=$( { time ./process ./images/MSUStadium.png test.png > /dev/null 2>&1; } 2>&1 )

    sum_cube=$(echo "$sum_cube + $t_run_cube" | bc -l)
    sum_earth=$(echo "$sum_earth + $t_run_earth" | bc -l)
    sum_sparty=$(echo "$sum_sparty + $t_run_sparty" | bc -l)
    sum_stadium=$(echo "$sum_stadium + $t_run_stadium" | bc -l)
done

# Calculate average execution time
avg_cube=$(echo "$sum_cube / $runs" | bc -l)
avg_earth=$(echo "$sum_earth / $runs" | bc -l)
avg_sparty=$(echo "$sum_sparty / $runs" | bc -l)
avg_stadium=$(echo "$sum_stadium / $runs" | bc -l)

echo "Average execution time for cube: $avg_cube seconds"
echo "Average execution time for earth: $avg_earth seconds"
echo "Average execution time for sparty: $avg_sparty seconds"
echo "Average execution time for stadium: $avg_stadium seconds"
