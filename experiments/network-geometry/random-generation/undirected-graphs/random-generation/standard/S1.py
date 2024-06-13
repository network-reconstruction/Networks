"""
S1 Model for Network Generation

This implementation of the S1 model is based on the following paper:

Papadopoulos, F., Krioukov, D., Boguñá, M., & Vahdat, A. (2008). Efficient Routing in Complex Networks without Routing Tables. Physical Review Letters, 100(7), 078701. https://doi.org/10.1103/PhysRevLett.100.078701
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from typing import Callable, List

class S1Model:
    def __init__(self, N: int, beta: float, avg_degree: float):
        """
        Initialize the S1Model with given parameters.

        Args:
            N (int): Number of nodes.
            beta (float): Parameter greater than 1.
            avg_degree (float): Target average degree of the network.

        Raises:
            ValueError: If N is non-positive, beta is less than or equal to 1, or avg_degree is non-positive.
        """
        if N <= 0:
            raise ValueError("Number of nodes N must be positive")
        if beta <= 1:
            raise ValueError("Parameter beta must be greater than 1")
        if avg_degree <= 0:
            raise ValueError("Average degree must be positive")

        self.N = N
        self.beta = beta
        self.avg_degree = avg_degree
        self.G = None

    def initialize_hidden_degrees(self, distribution: Callable, **kwargs) -> np.ndarray:
        """
        Initialize hidden degrees for the nodes based on a given distribution.

        Args:
            distribution (Callable): A function from a distribution to generate hidden degrees.
            kwargs: Additional keyword arguments for the distribution function.

        Returns:
            np.ndarray: Array of hidden degrees.
        """
        return distribution(size=self.N, **kwargs)

    def initialize_thetas(self) -> np.ndarray:
        """
        Initialize angular positions for the nodes.

        Returns:
            np.ndarray: Array of angular positions.
        """
        return np.random.uniform(0, 2 * np.pi, size=self.N)

    def generate_network(self, distribution: Callable, **kwargs) -> nx.Graph:
        """
        Generate the network based on the S1 model.

        Args:
            distribution (Callable): A function from a distribution to generate hidden degrees.
            kwargs: Additional keyword arguments for the distribution function.

        Returns:
            nx.Graph: The generated network.
        """
        hidden_degrees = self.initialize_hidden_degrees(distribution, **kwargs)
        thetas = self.initialize_thetas()

        G = nx.Graph()

        for i in range(self.N):
            G.add_node(i, hidden_degree=hidden_degrees[i], theta=thetas[i])

        for i in range(self.N):
            for j in range(i + 1, self.N):
                delta_theta = np.abs(thetas[i] - thetas[j])
                probability = np.exp(-self.beta * np.sin(delta_theta / 2))
                if np.random.rand() < probability:
                    G.add_edge(i, j)

        self.G = G
        return G

    def plot_network(self, notebook: bool = False) -> None:
        """
        Plot the network using PyVis.

        Args:
            notebook (bool): If True, render the plot in a Jupyter Notebook.
        """
        if self.G is None:
            raise ValueError("Network has not been generated yet")

        net = Network(notebook=notebook)
        net.from_nx(self.G)
        net.show("s1_network.html")

    def plot_degree_distribution(self) -> None:
        """
        Plot the degree distribution of the generated network.
        """
        if self.G is None:
            raise ValueError("Network has not been generated yet")

        degrees = [self.G.degree(n) for n in self.G.nodes()]
        plt.hist(degrees, bins=30, edgecolor='black')
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.yscale('log')
        plt.show()

# Example usage
s1_model = S1Model(N=100, beta=2, avg_degree=5)

# Generate network with hidden degrees following a normal distribution
G_s1_normal = s1_model.generate_network(np.random.normal, loc=0, scale=1)
s1_model.plot_network()
s1_model.plot_degree_distribution()

# Generate network with hidden degrees following a power-law distribution
G_s1_powerlaw = s1_model.generate_network(np.random.pareto, a=3)
s1_model.plot_network()
s1_model.plot_degree_distribution()
