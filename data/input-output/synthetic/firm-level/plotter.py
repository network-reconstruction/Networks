import jax.numpy as jnp
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataPlotter:
    def __init__(self, data: Dict[str, jnp.ndarray]):
        """
        Initialize the DataPlotter with the provided data.
        
        Args:
            data (Dict[str, jnp.ndarray]): The synthetic data to be plotted.
        """
        self.data = data
        sns.set(style="whitegrid")

    def plot(self) -> None:
        """
        Plot the synthetic data using Matplotlib and Seaborn.
        """
        plt.figure(figsize=(14, 10))

        # Plot for In-Degree vs. In-Strength
        plt.subplot(2, 2, 1)
        sns.scatterplot(x=np.array(self.data['kin']), y=np.array(self.data['sin']), 
                        palette='viridis', s=50, alpha=0.7, edgecolor='k')
        plt.title('In-Degree vs. In-Strength', fontsize=14)
        plt.xlabel('In-Degree (Number of Suppliers)', fontsize=12)
        plt.ylabel('In-Strength (Total Intermediate Expenses)', fontsize=12)
        
        # Plot for Out-Degree vs. Out-Strength
        plt.subplot(2, 2, 2)
        sns.scatterplot(x=np.array(self.data['kout']), y=np.array(self.data['sout']), 
                        palette='viridis', s=50, alpha=0.7, edgecolor='k')
        plt.title('Out-Degree vs. Out-Strength', fontsize=14)
        plt.xlabel('Out-Degree (Number of Customers)', fontsize=12)
        plt.ylabel('Out-Strength (Total Intermediate Sales)', fontsize=12)
        
        # Plot for In-Degree vs. Out-Degree
        plt.subplot(2, 2, 3)
        sns.scatterplot(x=np.array(self.data['kin']), y=np.array(self.data['kout']), 
                        palette='viridis', s=50, alpha=0.7, edgecolor='k')
        plt.title('In-Degree vs. Out-Degree', fontsize=14)
        plt.xlabel('In-Degree (Number of Suppliers)', fontsize=12)
        plt.ylabel('Out-Degree (Number of Customers)', fontsize=12)
        
        # Plot for In-Strength vs. Out-Strength
        plt.subplot(2, 2, 4)
        sns.scatterplot(x=np.array(self.data['sin']), y=np.array(self.data['sout']), 
                        palette='viridis', s=50, alpha=0.7, edgecolor='k')
        plt.title('In-Strength vs. Out-Strength', fontsize=14)
        plt.xlabel('In-Strength (Total Intermediate Expenses)', fontsize=12)
        plt.ylabel('Out-Strength (Total Intermediate Sales)', fontsize=12)
        
        plt.tight_layout()

    def show(self) -> None:
        """ Show the plot. """  
        self.plot()
        plt.show()
        
    def save_plot(self, name: str) -> None:
        """ Save the plot to a file. """
        self.plot()  # Ensure the plot is created before saving
        plt.savefig(name)
        plt.close()
