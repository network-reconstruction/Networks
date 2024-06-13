"""
H2 Model for Network Generation using JAX

This implementation of the H2 model is based on the following paper:

Krioukov, D., Papadopoulos, F., Kitsak, M., Vahdat, A., & Boguñá, M. (2010). Hyperbolic Geometry of Complex Networks. Physical Review E, 82(3), 036106. https://doi.org/10.1103/PhysRevE.82.036106
"""

import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from typing import Callable, Tuple

def initialize_radial_coords(N: int, R_H2: float) -> jnp.ndarray:
    """
    Initialize radial coordinates for the nodes.

    Args:
        N (int): Number of nodes.
        R_H2 (float): Radius of the hyperbolic disk.

    Returns:
        jnp.ndarray: Array of radial coordinates.
    """
    return jax.random.uniform(jax.random.PRNGKey(2), shape=(N,), minval=0, maxval=R_H2)

def initialize_angular_coords(N: int) -> jnp.ndarray:
    """
    Initialize angular positions for the nodes.

    Args:
        N (int): Number of nodes.

    Returns:
        jnp.ndarray: Array of angular positions.
    """
    return jax.random.uniform(jax.random.PRNGKey(3), shape=(N,), minval=0, maxval=2 * jnp.pi)

@jax.jit
def generate_h2_network(N: int, beta: float, avg_degree: float, gamma: float) -> nx.Graph:
    """
    Generate the network based on the H2 model.

    Args:
        N (int): Number of nodes.
        beta (float): Parameter greater than 1.
        avg_degree (float): Target average degree of the network.
        gamma (float): Exponent for the scale-free network.

    Returns:
        nx.Graph: The generated network.
    """
    if N <= 0 or beta <= 1 or avg_degree <= 0 or gamma <= 2:
        raise ValueError("N must be positive, beta must be greater than 1, avg_degree must be positive, and gamma must be greater than 2")
    
    R_H2 = 2 * jnp.log((2 * N) / (beta * jnp.sin(jnp.pi / beta) * avg_degree) * ((gamma - 1) / (gamma - 2)) ** 2)
    radial_coords = initialize_radial_coords(N, R_H2)
    angular_coords = initialize_angular_coords(N)
    
    G = nx.Graph()
    
    for i in range(N):
        G.add_node(i, r=radial_coords[i], theta=angular_coords[i])
    
    def connect_nodes(i: int, j: int) -> Tuple[int, int]:
        delta_theta = jnp.pi - jnp.abs(jnp.pi - jnp.abs(angular_coords[i] - angular_coords[j]))
        distance = radial_coords[i] + radial_coords[j] + 2 * jnp.log(delta_theta / 2)
        connection_prob = 1 / (1 + jnp.exp((distance - R_H2) / (2 * jnp.pi * avg_degree)))
        return (i, j) if jax.random.uniform(jax.random.PRNGKey(i * j)) < connection_prob else None
    
    edges = jax.vmap(lambda ij: connect_nodes(ij[0], ij[1]))(jnp.array([(i, j) for i in range(N) for j in range(i + 1, N)]))
    edges = [edge for edge in edges if edge is not None]
    
    G.add_edges_from(edges)
    
    return G

def plot_network(G: nx.Graph, notebook: bool = False) -> None:
    """
    Plot the network using PyVis.

    Args:
        G (nx.Graph): The generated network.
        notebook (bool): If True, render the plot in a Jupyter Notebook.
    """
    net = Network(notebook=notebook)
    net.from_nx(G)
    net.show("h2_network.html")

def plot_degree_distribution(G: nx.Graph) -> None:
    """
    Plot the degree distribution of the generated network.

    Args:
        G (nx.Graph): The generated network.
    """
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=30, edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.show()

# Example usage
G_h2_jax = generate_h2_network(100, 2, 5, 2.5)
plot_network(G_h2_jax)
plot_degree_distribution(G_h2_jax)
