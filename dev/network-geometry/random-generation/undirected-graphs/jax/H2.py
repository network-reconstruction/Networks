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
from typing import Tuple

def initialize_radial_coords(N: int, R_H2: float, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Initialize radial coordinates for the nodes.

    Args:
        N (int): Number of nodes.
        R_H2 (float): Radius of the hyperbolic disk.
        key (jax.random.PRNGKey): PRNG key for randomness.

    Returns:
        jnp.ndarray: Array of radial coordinates.
    """
    return jax.random.uniform(key, shape=(N,), minval=0, maxval=R_H2)

def initialize_angular_coords(N: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Initialize angular positions for the nodes.

    Args:
        N (int): Number of nodes.
        key (jax.random.PRNGKey): PRNG key for randomness.

    Returns:
        jnp.ndarray: Array of angular positions.
    """
    return jax.random.uniform(key, shape=(N,), minval=0, maxval=2 * jnp.pi)

def generate_h2_network(N: int, beta: float, avg_degree: float, gamma: float, key: jax.random.PRNGKey) -> nx.Graph:
    """
    Generate the network based on the H2 model.

    Args:
        N (int): Number of nodes.
        beta (float): Parameter greater than 1.
        avg_degree (float): Target average degree of the network.
        gamma (float): Exponent for the scale-free network.
        key (jax.random.PRNGKey): PRNG key for randomness.

    Returns:
        nx.Graph: The generated network.
    """
    if N <= 0 or beta <= 1 or avg_degree <= 0 or gamma <= 2:
        raise ValueError("N must be positive, beta must be greater than 1, avg_degree must be positive, and gamma must be greater than 2")

    R_H2 = 2 * jnp.log((2 * N) / (beta * jnp.sin(jnp.pi / beta) * avg_degree) * ((gamma - 1) / (gamma - 2)) ** 2)

    key_r, key_theta, key_edge = jax.random.split(key, 3)
    radial_coords = initialize_radial_coords(N, R_H2, key_r)
    angular_coords = initialize_angular_coords(N, key_theta)

    keys = jax.random.split(key_edge, (N * (N - 1)) // 2)

    G = nx.Graph()

    for i in range(N):
        G.add_node(i, r=float(radial_coords[i]), theta=float(angular_coords[i]))

    def connect_nodes(i: int, j: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        delta_theta = jnp.pi - jnp.abs(jnp.pi - jnp.abs(angular_coords[i] - angular_coords[j]))
        distance = radial_coords[i] + radial_coords[j] + 2 * jnp.log(delta_theta / 2)
        connection_prob = 1 / (1 + jnp.exp((distance - R_H2) / (2 * jnp.pi * avg_degree)))
        should_connect = jax.random.uniform(key) < connection_prob
        return jax.lax.cond(should_connect, lambda _: jnp.array([i, j]), lambda _: jnp.array([-1, -1]), operand=None)

    # Create indices mesh for all pairs of nodes but not with themselves
    indices = jnp.tril_indices(N, -1)

    # Over all of the tuples of indices, connect the nodes
    edges = jax.vmap(connect_nodes)(indices[0], indices[1], keys)
    edges = [(int(i), int(j)) for i, j in zip(edges[:, 0], edges[:, 1]) if i != -1]
    
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

def plot_degree_distribution(G: nx.Graph, save: bool = False) -> None:
    """
    Plot the degree distribution of the generated network.

    Args:
        G (nx.Graph): The generated network.
        save (bool): If True, save the plot as an image.
    """
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=30, edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.yscale('log')
    if save:
        plt.savefig("degree_distribution.png")
    plt.show()

# Example usage
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    G_h2_jax = generate_h2_network(100, 2, 5, 2.5, key)
    # plot_network(G_h2_jax)
    plot_degree_distribution(G_h2_jax, save=True)
