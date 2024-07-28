import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.polynomial import Polynomial

# Function to generate power-law distributed data with a specified scale
def generate_power_law_data(size, alpha):
    return (np.random.pareto(alpha, size) + 1)

# Function to calculate weighted least squares (WLS) regression with polynomial heteroskedasticity control
def weighted_least_squares(x, y, heteroskedasticity_poly):
    weights = 1 / heteroskedasticity_poly(x)
    wls_fit = np.polyfit(x, y, 1, w=weights)
    return wls_fit

# Parameters for power-law exponents and scaling factor
alpha_in = 2.5  # Power-law exponent for in-degrees
alpha_out = 2.0  # Power-law exponent for out-degrees
scaling_factor_control = 1.5  # Control for scaling factor

# Generating synthetic data for in-degrees (customers) and out-degrees (suppliers)
np.random.seed(42)
size = 1000

# Generate in-degrees and out-degrees
in_degrees = generate_power_law_data(size, alpha_in)
out_degrees = generate_power_law_data(size, alpha_out)

# Ensure the empirical averages are the same using the controllable scaling factor
avg_in_degrees = np.mean(in_degrees)
avg_out_degrees = np.mean(out_degrees)
scaling_factor = (avg_in_degrees / avg_out_degrees) * scaling_factor_control
out_degrees *= scaling_factor

# Apply log transformation for better visualization and regression
log_in_degrees = np.log(in_degrees)
log_out_degrees = np.log(out_degrees)

# Define a polynomial for heteroskedasticity control
heteroskedasticity_coeffs = [1, 0.5, 0.2]  # Example coefficients for a quadratic polynomial
heteroskedasticity_poly = Polynomial(heteroskedasticity_coeffs)

# Perform WLS regression
wls_fit = weighted_least_squares(log_in_degrees, log_out_degrees, heteroskedasticity_poly)
slope, intercept = wls_fit

# Plotting the combined figure
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot for Figure 3 (left: suppliers, right: customers)
suppliers_x, suppliers_y = np.sort(out_degrees), np.linspace(1, 0, size, endpoint=False)
customers_x, customers_y = np.sort(in_degrees), np.linspace(1, 0, size, endpoint=False)
sns.lineplot(x=suppliers_x, y=suppliers_y, ax=axs[0], label="Suppliers (Out-Degrees)")
sns.lineplot(x=customers_x, y=customers_y, ax=axs[0], label="Customers (In-Degrees)")
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_title('Empirical CCDF of Number of Suppliers and Customers')
axs[0].set_xlabel('Number')
axs[0].set_ylabel('CCDF')
axs[0].legend()

# Plot for Figure 4 (correlation between in-degrees and out-degrees)
sns.scatterplot(x=log_in_degrees, y=log_out_degrees, ax=axs[1], alpha=0.5)
axs[1].plot(log_in_degrees, intercept + slope * log_in_degrees, color='red')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_title(f'Correlation between Log In-Degrees and Log Out-Degrees\nSlope: {slope:.2f}')
axs[1].set_xlabel('Log of In-Degrees (Number of Customers)')
axs[1].set_ylabel('Log of Out-Degrees (Number of Suppliers)')

print(f"average in degrees: {np.mean(in_degrees)}")
print(f"average out degrees: {np.mean(out_degrees)}")
plt.tight_layout()
plt.savefig('directed-distribution2.png')
