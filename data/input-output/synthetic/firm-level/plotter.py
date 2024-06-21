class DataPlotter:
    def __init__(self, data: Dict[str, jnp.ndarray]):
        """
        Initialize the DataPlotter with the provided data.
        
        Args:
            data (Dict[str, jnp.ndarray]): The synthetic data to be plotted.
        """
        self.data = data
    
    def plot(self) -> None:
        """
        Plot the synthetic data using Matplotlib and Seaborn.
        """
        plt.figure(figsize=(14, 10))
        
        # Plot for In-Degree vs. Network Expenses
        plt.subplot(2, 2, 1)
        sns.scatterplot(x=self.data['In-Degree'], y=self.data['Network Expenses'])
        plt.title('In-Degree vs. Network Expenses')
        plt.xlabel('In-Degree (Number of Suppliers)')
        plt.ylabel('Network Expenses')
        
        # Plot for Out-Degree vs. Network Sales
        plt.subplot(2, 2, 2)
        sns.scatterplot(x=self.data['Out-Degree'], y=self.data['Network Sales'])
        plt.title('Out-Degree vs. Network Sales')
        plt.xlabel('Out-Degree (Number of Customers)')
        plt.ylabel('Network Sales')
        
        # Plot for Suppliers vs. Expenses
        plt.subplot(2, 2, 3)
        sns.scatterplot(x=self.data['Suppliers'], y=self.data['Expenses'])
        plt.title('Suppliers vs. Expenses')
        plt.xlabel('Suppliers')
        plt.ylabel('Expenses')
        
        # Plot for Customers vs. Sales
        plt.subplot(2, 2, 4)
        sns.scatterplot(x=self.data['Customers'], y=self.data['Sales'])
        plt.title('Customers vs. Sales')
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        
        plt.tight_layout()
        plt.show()