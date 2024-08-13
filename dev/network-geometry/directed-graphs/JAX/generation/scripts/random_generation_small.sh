cd ../
echo "Running Random Generation for directed graphs, output will be saved to output_generation_small.log"
# print start time
echo "Random Generation started, check output.log for progress"
date

# run Random Generation and save the PID
find . -type f -name '{file_pattern}' -exec rm -rf {} + && python -u random_generation.py deg_seq_test_inferred_parameters.json > output_generation_small.log 2>&1 &
PID=$!

# print the PID
echo "Random Generation running with PID: $PID"

# wait for background process to complete
wait $PID

# print end time
date
echo "Random Generation finished, check output.log for results"
