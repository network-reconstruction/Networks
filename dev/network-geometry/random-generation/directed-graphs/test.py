# Import the FittingDirectedS1 class
from fitting_directed_s1 import FittingDirectedS1

# Define the input parameters
edgelist_filename = 'degree_sequence.txt'  # Path to your degree sequence file
reciprocity = 0.25  # Example reciprocity value (replace with your actual value)
average_local_clustering = 0.35  # Example average local clustering coefficient (replace with your actual value)

# Create an instance of the model
model = FittingDirectedS1(edgelist_filename, reciprocity, average_local_clustering)

# Fit the model to the data
model.fit()

# Optionally, you can set the ROOTNAME_OUTPUT before fitting if you want to specify the output file name
# model.ROOTNAME_OUTPUT = 'your_output_filename'

# Save the inferred parameters (this is done automatically within the fit method)
# model.save_inferred_parameters()  # This is already called in the fit method

# Print the inferred parameters
print(f"beta: {model.beta}")
print(f"mu: {model.mu}")
print(f"nu: {model.nu}")
print(f"R: {model.R}")
