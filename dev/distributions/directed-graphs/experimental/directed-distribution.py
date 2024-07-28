# Directed graphs distribution
# -----------------------------------------------------------
# OUTPUT FORMAT:
# -----------------------------------------------------------
# vertex  kappa_out  kappa_in
# 0  3.5  2.1
# 1  2.2  1.8
# 2  4.1  3.3
# 3  1.5  1.0
# 4  3.8  2.7
# -----------------------------------------------------------
# This file contains the code for the directed graphs distribution
# Create as a class
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generating synthetic data for the CCDF of number of suppliers and customers
np.random.seed(42)
suppliers = np.random.pareto(a=2, size=1000) + 1
customers = np.random.pareto(a=2, size=1000) + 1

# Empirical CCDF for suppliers and customers
def ecdf(data):
    x = np.sort(data)
    y = 1 - np.arange(1, len(x)+1) / len(x)
    return x, y

suppliers_x, suppliers_y = ecdf(suppliers)
customers_x, customers_y = ecdf(customers)

# Generating synthetic data for the correlation between in-degrees and out-degrees
in_degrees = np.random.pareto(a=2, size=1000) + 1
out_degrees = np.random.pareto(a=2, size=1000) + 1

# Plotting the combined figure
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot for Figure 3 (left: suppliers, right: customers)
sns.lineplot(x=suppliers_x, y=suppliers_y, ax=axs[0], label="Suppliers")
sns.lineplot(x=customers_x, y=customers_y, ax=axs[0], label="Customers")
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_title('Empirical CCDF of Number of Suppliers and Customers')
axs[0].set_xlabel('Number')
axs[0].set_ylabel('CCDF')
axs[0].legend()

# Plot for Figure 4 (correlation between in-degrees and out-degrees)
sns.scatterplot(x=in_degrees, y=out_degrees, ax=axs[1], alpha=0.5)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_title('Correlation between In-Degrees and Out-Degrees')
axs[1].set_xlabel('In-Degrees (Number of Customers)')
axs[1].set_ylabel('Out-Degrees (Number of Suppliers)')

plt.tight_layout()
#save plot
plt.savefig('directed-distribution.png')
