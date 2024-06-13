"""
S1 Model for Network Generation

This implementation of the S1 model is based on the following paper:

Papadopoulos, F., Krioukov, D., Boguñá, M., & Vahdat, A. (2008). Efficient Routing in Complex Networks without Routing Tables. Physical Review Letters, 100(7), 078701. https://doi.org/10.1103/PhysRevLett.100.078701
"""

import numpy as np
import networkx as nx
from typing import Callable, Optional
from network_visualizer import NetworkVisualizer

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

        self.N: int = N
        self.beta: float = beta
        self.avg_degree: float = avg_degree
        self.G: Optional[nx.Graph] = None

    def initialize_hidden_degrees(self, distribution: Callable[..., np.ndarray], **kwargs) -> np.ndarray:
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

    def generate_network(self, distribution: Callable[..., np.ndarray], **kwargs) -> nx.Graph:
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

    def get_graph(self) -> nx.Graph:
        """
        Get the generated network graph.

        Returns:
            nx.Graph: The generated network graph.

        Raises:
            ValueError: If the network has not been generated yet.
        """
        if self.G is None:
            raise ValueError("Network has not been generated yet")
        return self.G

if __name__ == "__main__":
    print("Generating network...")
    model = S1Model(N=100, beta=2, avg_degree=5)
    model.generate_network(np.random.pareto, a=4)
    print(f"Generated Number of nodes: {len(model.get_graph().nodes)}")

    print("Plotting network...")
    visualizer = NetworkVisualizer(model.get_graph())
    network_html = visualizer.plot_network("test_network.html")
    print(f"Network plot saved at: {network_html}")

    print("Plotting degree distribution...")
    degree_distribution_file = visualizer.plot_degree_distribution("s1_degree_distribution.png")
    print(f"Degree distribution plot saved at: {degree_distribution_file}")