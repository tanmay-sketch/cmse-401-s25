# BEST=9999999999999
# BEST_FILE=""
# for f in tmp_output_*.txt; do
#   M=$(grep "fitness=" "$f" | sed -E 's/.*fitness=([0-9]+).*/\1/' | sort -n | head -n 1)
#  [ -z "$M" ] && continue
#   if (( M < BEST )); then
#     BEST=$M
#     BEST_FILE=$f
#   fi
# done
# [ -n "$BEST_FILE" ] && cp "$BEST_FILE" serial_best.txt


#!/usr/bin/env bash
BEST=9999999999999
BEST_FILE=""
for f in part2_outputs/*.txt; do
  M=$(grep "Result Fitness=" "$f" | sed -E 's/.*Fitness=([0-9]+).*/\1/')
  [ -z "$M" ] && continue
  if (( M < BEST )); then
    BEST=$M
    BEST_FILE=$f
  fi
done
[ -n "$BEST_FILE" ] && cp "$BEST_FILE" pp_best.txt
