#!/bin/bash
cd ../
echo "Running Ensemble Analysis for directed graphs, output will be saved to output_ensemble_small.log"
# print start time
echo "Ensemble Analysis started, check output_ensemble_small.log for progress"
date

# run Ensemble Analysis and save the PID
python -u ensemble_analysis.py deg_seq_test_inferred_parameters.json output_ensemble_small 30 > output_ensemble_small.log 2>&1 &
PID=$!

# print the PID
echo "Ensemble Analysis running with PID: $PID"

# wait for background process to complete
wait $PID

# print end time
date
echo "Ensemble Analysis finished, check output_ensemble_small.log for results"
