import numpy as np
import matplotlib.pyplot as plt

# Define the TLS line parameters
a_tls = 2.0  # slope of the TLS line
b_tls = 1.0  # intercept of the TLS line
t_tls = np.linspace(0, 10, 100)

def synthetic_tls(slope, intercept, t_values, variance_func=None):
    # Generate corresponding x and y values without noise
    x_tls_true = t_values
    y_tls_true = slope * t_values + intercept
    
    # Define a default variance function if none is provided
    if variance_func is None:
        variance_func = lambda t: 1.0  # constant variance
    
    # Add normally distributed noise to both x and y values with variance depending on t
    noise_variance_tls = variance_func(t_values)
    noise_x_tls = np.random.normal(0, noise_variance_tls, size=x_tls_true.shape)
    noise_y_tls = np.random.normal(0, noise_variance_tls, size=y_tls_true.shape)
    x_tls_noisy = x_tls_true + noise_x_tls
    y_tls_noisy = y_tls_true + noise_y_tls
    
    #save plot of noise x_tls
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, noise_x_tls, label='Noise in x', color='red')
    plt.xlabel('t')
    plt.ylabel('Noise in x')
    plt.title('Noise in x with Variable Variance')
    plt.legend()
    plt.grid(True)
    plt.savefig('noise_x_tls.png')
    
    
    return x_tls_noisy, y_tls_noisy, x_tls_true, y_tls_true

# Example variance function that varies with t
def variance_func(t):
    return 0.1*t  # Example: variance increases linearly with t

# Generate random t values representing parameter along the line
x_tls_noisy, y_tls_noisy, x_tls_true, y_tls_true = synthetic_tls(a_tls, b_tls, t_tls, variance_func)

# Plotting the TLS results
plt.figure(figsize=(10, 6))
plt.plot(x_tls_true, y_tls_true, label='True TLS Line', color='blue')
plt.scatter(x_tls_noisy, y_tls_noisy, label='Noisy Data (TLS)', color='green', alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Synthetic Data with TLS Line and Variable Variance')
plt.legend()
plt.grid(True)
plt.savefig('TLS_variable_variance.png')
plt.show()
