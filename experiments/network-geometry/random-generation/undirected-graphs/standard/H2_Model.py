"""
H2 Model for Network Generation

This implementation of the H2 model is based on the following paper:

Krioukov, D., Papadopoulos, F., Kitsak, M., Vahdat, A., & Boguñá, M. (2010). Hyperbolic Geometry of Complex Networks. Physical Review E, 82(3), 036106. https://doi.org/10.1103/PhysRevE.82.036106
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from typing import Callable, List

class H2Model:
    def __init__(self, N: int, beta: float, avg_degree: float, gamma: float):
        """
        Initialize the H2Model with given parameters.

        Args:
            N (int): Number of nodes.
            beta (float): Parameter greater than 1.
            avg_degree (float): Target average degree of the network.
            gamma (float): Exponent for the scale-free network.

        Raises:
            ValueError: If N is non-positive, beta is less than or equal to 1, avg_degree is non-positive, or gamma is less than or equal to 2.
        """
        if N <= 0:
            raise ValueError("Number of nodes N must be positive")
        if beta <= 1:
            raise ValueError("Parameter beta must be greater than 1")
        if avg_degree <= 0:
            raise ValueError("Average degree must be positive")
        if gamma <= 2:
            raise ValueError("Gamma must be greater than 2")

        self.N = N
        self.beta = beta
        self.avg_degree = avg_degree
        self.gamma = gamma
        self.R_H2 = 2 * np.log((2 * N) / (beta * np.sin(np.pi / beta) * avg_degree) * ((gamma - 1) / (gamma - 2)) ** 2)
        self.G = None

    def initialize_radial_coords(self) -> np.ndarray:
        """
        Initialize radial coordinates for the nodes.

        Returns:
            np.ndarray: Array of radial coordinates.
        """
        return np.random.uniform(0, self.R_H2, size=self.N)

    def initialize_angular_coords(self) -> np.ndarray:
        """
        Initialize angular positions for the nodes.

        Returns:
            np.ndarray: Array of angular positions.
        """
        return np.random.uniform(0, 2 * np.pi, size=self.N)

    def generate_network(self) -> nx.Graph:
        """
        Generate the network based on the H2 model.

        Returns:
            nx.Graph: The generated network.
        """
        radial_coords = self.initialize_radial_coords()
        angular_coords = self.initialize_angular_coords()

        G = nx.Graph()

        for i in range(self.N):
            G.add_node(i, r=radial_coords[i], theta=angular_coords[i])

        for i in range(self.N):
            for j in range(i + 1, self.N):
                delta_theta = np.pi - abs(np.pi - abs(angular_coords[i] - angular_coords[j]))
                distance = radial_coords[i] + radial_coords[j] + 2 * np.log(delta_theta / 2)
                connection_prob = 1 / (1 + np.exp((distance - self.R_H2) / (2 * np.pi * self.avg_degree)))
                if np.random.rand() < connection_prob:
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
        net.show("h2_network.html")

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
    model = H2Model(N=100, beta=2, avg_degree=5, gamma=2.5)
    print(f"Generated Number of nodes: {len(model.get_graph().nodes)}")

    print("Plotting network...")
    visualizer = NetworkVisualizer(model.get_graph())
    network_html = visualizer.plot_network("test_network.html")
    print(f"Network plot saved at: {network_html}")

    print("Plotting degree distribution...")
    degree_distribution_file = visualizer.plot_degree_distribution("s1_degree_distribution.png")
    print(f"Degree distribution plot saved at: {degree_distribution_file}")