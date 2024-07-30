echo "Running inference on directed graphs, output will be saved to output.log"
# print start time
echo "Inference started, check output.log for progress"
date

# run inference and save the PID
find . -type f -name '{file_pattern}' -exec rm -rf {} + && python -u infer_params.p degree_sequence.txt > output.log 2>&1 &
PID=$!

# print the PID
echo "Inference running with PID: $PID"

# wait for background process to complete
wait $PID

# print end time
date
echo "Inference finished, check output.log for results"
