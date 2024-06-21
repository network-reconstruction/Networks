"""
S1 Model for Network Generation using JAX

This implementation of the S1 model is based on the following paper:

Papadopoulos, F., Krioukov, D., Boguñá, M., & Vahdat, A. (2008). Efficient Routing in Complex Networks without Routing Tables. Physical Review Letters, 100(7), 078701. https://doi.org/10.1103/PhysRevLett.100.078701
"""

import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from typing import Callable, Tuple

def initialize_hidden_degrees(N: int, distribution: Callable, **kwargs) -> jnp.ndarray:
    """
    Initialize hidden degrees for the nodes based on a given distribution.

    Args:
        N (int): Number of nodes.
        distribution (Callable): A function from a distribution to generate hidden degrees.
        kwargs: Additional keyword arguments for the distribution function.

    Returns:
        jnp.ndarray: Array of hidden degrees.
    """
    return distribution(jax.random.PRNGKey(0), shape=(N,), **kwargs)

def initialize_thetas(N: int) -> jnp.ndarray:
    """
    Initialize angular positions for the nodes.

    Args:
        N (int): Number of nodes.

    Returns:
        jnp.ndarray: Array of angular positions.
    """
    return jax.random.uniform(jax.random.PRNGKey(1), shape=(N,), minval=0, maxval=2 * jnp.pi)

@jax.jit
def generate_s1_network(N: int, beta: float, avg_degree: float, distribution: Callable, **kwargs) -> nx.Graph:
    """
    Generate the network based on the S1 model.

    Args:
        N (int): Number of nodes.
        beta (float): Parameter greater than 1.
        avg_degree (float): Target average degree of the network.
        distribution (Callable): A function from a distribution to generate hidden degrees.
        kwargs: Additional keyword arguments for the distribution function.

    Returns:
        nx.Graph: The generated network.
    """
    if N <= 0 or beta <= 1 or avg_degree <= 0:
        raise ValueError("N must be positive, beta must be greater than 1, and avg_degree must be positive")
    
    hidden_degrees = initialize_hidden_degrees(N, distribution, **kwargs)
    thetas = initialize_thetas(N)
    
    G = nx.Graph()
    
    for i in range(N):
        G.add_node(i, hidden_degree=hidden_degrees[i], theta=thetas[i])
    
    def connect_nodes(i: int, j: int) -> Tuple[int, int]:
        delta_theta = jnp.abs(thetas[i] - thetas[j])
        probability = jnp.exp(-beta * jnp.sin(delta_theta / 2))
        return (i, j) if jax.random.uniform(jax.random.PRNGKey(i * j)) < probability else None
    
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
    net.show("s1_network.html")

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
G_s1_jax_normal = generate_s1_network(100, 2, 5, jax.random.normal, loc=0, scale=1)
plot_network(G_s1_jax_normal)
plot_degree_distribution(G_s1_jax_normal)

G_s1_jax_powerlaw = generate_s1_network(100, 2, 5, jax.random.pareto, b=3)
plot_network(G_s1_jax_powerlaw)
plot_degree_distribution(G_s1_jax_powerlaw)
