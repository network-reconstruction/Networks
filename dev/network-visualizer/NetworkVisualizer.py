import networkx as nx
import numpy as np
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd
from typing import Optional, Union

class NetworkVisualizer:
    """
    A class to visualize network graphs using NetworkX, PyVis, and Seaborn.
    It supports visualization from either a NetworkX graph object, an adjacency matrix (with or without labels), or a CSV file.

    Attributes:
        G (networkx.Graph): The graph to be visualized.
        directed (bool): Indicates if the graph is directed.
        weighted (bool): Indicates if the graph is weighted.
        verbose (bool): Enables verbose output for debugging and informational purposes.
    """

    def __init__(self, 
                 graph: Optional[nx.Graph] = None, 
                 adjacency_matrix: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                 csv_path: Optional[str] = None, 
                 directed: bool = False, 
                 weighted: bool = False, 
                 verbose: bool = False) -> None:
        """
        Initializes the NetworkVisualizer with a graph, an adjacency matrix, or a CSV file.

        Args:
            graph (networkx.Graph, optional): A NetworkX graph object.
            adjacency_matrix (numpy.ndarray or pandas.DataFrame, optional): An adjacency matrix.
            csv_path (str, optional): Path to a CSV file containing edge list or adjacency matrix with labels.
            directed (bool): Whether the graph is directed. Default is False.
            weighted (bool): Whether the graph is weighted. Default is False.
            verbose (bool): Whether to enable verbose output. Default is False.

        Raises:
            ValueError: If neither graph, adjacency matrix, nor CSV path is provided.
        """
        self.verbose = verbose

        if graph:
            self.G = graph
            if self.verbose:
                print("Graph provided directly.")
        elif adjacency_matrix is not None:
            self.G = self._create_graph_from_adj_matrix(adjacency_matrix, directed, weighted)
            if self.verbose:
                print("Graph created from adjacency matrix.")
        elif csv_path is not None:
            self.G = self._load_graph_from_csv(csv_path, directed, weighted)
            if self.verbose:
                print(f"Graph loaded from CSV file: {csv_path}")
        else:
            raise ValueError("Either graph, adjacency matrix, or CSV path must be provided.")
        self.directed = directed
        self.weighted = weighted

    def _create_graph_from_adj_matrix(self, adjacency_matrix: Union[np.ndarray, pd.DataFrame], directed: bool, weighted: bool) -> nx.Graph:
        """
        Creates a graph from an adjacency matrix.

        Args:
            adjacency_matrix (numpy.ndarray or pandas.DataFrame): The adjacency matrix.
            directed (bool): Whether the graph is directed.
            weighted (bool): Whether the graph is weighted.

        Returns:
            networkx.Graph: The graph created from the adjacency matrix.
        """
        if isinstance(adjacency_matrix, pd.DataFrame):
            G = nx.from_pandas_adjacency(adjacency_matrix, create_using=nx.DiGraph() if directed else nx.Graph())
        else:
            G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph() if directed else nx.Graph())

        if weighted:
            for i, j in zip(*np.where(adjacency_matrix != 0)):
                G[i][j]['weight'] = adjacency_matrix[i, j]

        return G

    def _load_graph_from_csv(self, csv_path: str, directed: bool, weighted: bool) -> nx.Graph:
        """
        Loads a graph from a CSV file.

        Args:
            csv_path (str): Path to the CSV file.
            directed (bool): Whether the graph is directed.
            weighted (bool): Whether the graph is weighted.

        Returns:
            networkx.Graph: The graph created from the CSV file.
        """
        df = pd.read_csv(csv_path)
        if 'source' in df.columns and 'target' in df.columns:
            if self.verbose:
                print("CSV file contains edge list.")
                print(df.head())
            if weighted:
                G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph() if directed else nx.Graph())
            else:
                G = nx.from_pandas_edgelist(df, source='source', target='target', create_using=nx.DiGraph() if directed else nx.Graph())
        else:
            if self.verbose:
                print("CSV file contains adjacency matrix with labels.")
            G = self._create_graph_from_adj_matrix(df, directed, weighted)
        return G

    def get_info(self) -> str:
        """
        Returns basic information about the graph.

        Returns:
            str: Information about the graph.
        """
        info = nx.info(self.G)
        if self.verbose:
            print(info)
        return info

    def show_network(self) -> None:
        """
        Displays an interactive visualization of the network using PyVis.
        """
        if self.verbose:
            print("Generating network visualization.")
        net = Network(notebook=True, directed=self.directed)
        for node in self.G.nodes:
            net.add_node(node, label=str(node))
        for edge in self.G.edges:
            if self.weighted:
                net.add_edge(edge[0], edge[1], weight=self.G[edge[0]][edge[1]]['weight'])
            else:
                net.add_edge(edge[0], edge[1])
        net.show('network.html')
        if self.verbose:
            print("Network visualization generated and displayed.")

    def show_heatmap(self) -> None:
        """
        Displays a heatmap of the adjacency matrix using Seaborn.
        """
        if self.verbose:
            print("Generating heatmap of adjacency matrix.")
        adjacency_matrix = nx.to_numpy_array(self.G)
        sns.heatmap(jnp.array(adjacency_matrix), annot=True, fmt='g')
        plt.show()
        if self.verbose:
            print("Heatmap displayed.")

    def show_distributions(self) -> None:
        """
        Displays the degree distributions of the graph based on its type (directed, weighted, etc.)
        using Seaborn.
        """
        if self.verbose:
            print("Generating degree distributions.")
        if not self.weighted and not self.directed:
            degrees = [val for (node, val) in self.G.degree()]
            sns.histplot(degrees, kde=True)
            plt.title('Degree Distribution (Unweighted, Undirected)')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.show()
            sns.scatterplot(x=range(len(degrees)), y=degrees)
            plt.title('Degree Scatter Plot (Unweighted, Undirected)')
            plt.xlabel('Node Index')
            plt.ylabel('Degree')
            plt.show()
        elif self.weighted and not self.directed:
            degree_weights = [val for (node, val) in self.G.degree(weight='weight')]
            sns.histplot(degree_weights, kde=True)
            plt.title('Degree Weight Distribution (Weighted, Undirected)')
            plt.xlabel('Degree Weight')
            plt.ylabel('Frequency')
            plt.show()
            sns.scatterplot(x=range(len(degree_weights)), y=degree_weights)
            plt.title('Degree Weight Scatter Plot (Weighted, Undirected)')
            plt.xlabel('Node Index')
            plt.ylabel('Degree Weight')
            plt.show()
        elif not self.weighted and self.directed:
            in_degrees = [val for (node, val) in self.G.in_degree()]
            out_degrees = [val for (node, val) in self.G.out_degree()]
            sns.histplot(in_degrees, kde=True, color='blue', label='In-Degree')
            sns.histplot(out_degrees, kde=True, color='orange', label='Out-Degree')
            plt.title('In-Degree and Out-Degree Distributions (Unweighted, Directed)')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
            sns.scatterplot(x=range(len(in_degrees)), y=in_degrees, color='blue', label='In-Degree')
            sns.scatterplot(x=range(len(out_degrees)), y=out_degrees, color='orange', label='Out-Degree')
            plt.title('In-Degree and Out-Degree Scatter Plot (Unweighted, Directed)')
            plt.xlabel('Node Index')
            plt.ylabel('Degree')
            plt.legend()
            plt.show()
        elif self.weighted and self.directed:
            in_degrees = [val for (node, val) in self.G.in_degree()]
            out_degrees = [val for (node, val) in self.G.out_degree()]
            in_strengths = [val for (node, val) in self.G.in_degree(weight='weight')]
            out_strengths = [val for (node, val) in self.G.out_degree(weight='weight')]

            plt.figure(figsize=(15, 10))

            # Plot 1: In-Degree and Out-Degree
            plt.subplot(2, 2, 1)
            sns.histplot(in_degrees, kde=True, color='blue', label='In-Degree')
            sns.histplot(out_degrees, kde=True, color='orange', label='Out-Degree')
            plt.title('In-Degree and Out-Degree Distributions')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.legend()

            # Plot 2: In-Degree and In-Strength
            plt.subplot(2, 2, 2)
            sns.histplot(in_degrees, kde=True, color='blue', label='In-Degree')
            sns.histplot(in_strengths, kde=True, color='green', label='In-Strength')
            plt.title('In-Degree and In-Strength Distributions')
            plt.xlabel('In-Degree / In-Strength')
            plt.ylabel('Frequency')
            plt.legend()

            # Plot 3: Out-Degree and Out-Strength
            plt.subplot(2, 2, 3)
            sns.histplot(out_degrees, kde=True, color='orange', label='Out-Degree')
            sns.histplot(out_strengths, kde=True, color='red', label='Out-Strength')
            plt.title('Out-Degree and Out-Strength Distributions')
            plt.xlabel('Out-Degree / Out-Strength')
            plt.ylabel('Frequency')
            plt.legend()

            # Plot 4: In-Strength and Out-Strength
            plt.subplot(2, 2, 4)
            sns.histplot(in_strengths, kde=True, color='green', label='In-Strength')
            sns.histplot(out_strengths, kde=True, color='red', label='Out-Strength')
            plt.title('In-Strength and Out-Strength Distributions')
            plt.xlabel('In-Strength / Out-Strength')
            plt.ylabel('Frequency')
            plt.legend()

            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(15, 10))

            # Scatter Plot 1: In-Degree and Out-Degree
            plt.subplot(2, 2, 1)
            sns.scatterplot(x=range(len(in_degrees)), y=in_degrees, color='blue', label='In-Degree')
            sns.scatterplot(x=range(len(out_degrees)), y=out_degrees, color='orange', label='Out-Degree')
            plt.title('In-Degree and Out-Degree Scatter Plot')
            plt.xlabel('Node Index')
            plt.ylabel('Degree')
            plt.legend()

            # Scatter Plot 2: In-Degree and In-Strength
            plt.subplot(2, 2, 2)
            sns.scatterplot(x=range(len(in_degrees)), y=in_degrees, color='blue', label='In-Degree')
            sns.scatterplot(x=range(len(in_strengths)), y=in_strengths, color='green', label='In-Strength')
            plt.title('In-Degree and In-Strength Scatter Plot')
            plt.xlabel('Node Index')
            plt.ylabel('In-Degree / In-Strength')
            plt.legend()

            # Scatter Plot 3: Out-Degree and Out-Strength
            plt.subplot(2, 2, 3)
            sns.scatterplot(x=range(len(out_degrees)), y=out_degrees, color='orange', label='Out-Degree')
            sns.scatterplot(x=range(len(out_strengths)), y=out_strengths, color='red', label='Out-Strength')
            plt.title('Out-Degree and Out-Strength Scatter Plot')
            plt.xlabel('Node Index')
            plt.ylabel('Out-Degree / Out-Strength')
            plt.legend()

            # Scatter Plot 4: In-Strength and Out-Strength
            plt.subplot(2, 2, 4)
            sns.scatterplot(x=range(len(in_strengths)), y=in_strengths, color='green', label='In-Strength')
            sns.scatterplot(x=range(len(out_strengths)), y=out_strengths, color='red', label='Out-Strength')
            plt.title('In-Strength and Out-Strength Scatter Plot')
            plt.xlabel('Node Index')
            plt.ylabel('In-Strength / Out-Strength')
            plt.legend()

            plt.tight_layout()
            plt.show()
        if self.verbose:
            print("Degree distributions displayed.")

    def save_distributions(self, output_dir: str) -> None:
        """
        Saves the degree distributions of the graph based on its type (directed, weighted, etc.)
        as image files in the specified output directory.

        Args:
            output_dir (str): The directory to save the distribution plots.
        """
        if self.verbose:
            print(f"Saving degree distributions to directory: {output_dir}")
        if not self.weighted and not self.directed:
            degrees = [val for (node, val) in self.G.degree()]
            sns.histplot(degrees, kde=True)
            plt.title('Degree Distribution (Unweighted, Undirected)')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.savefig(f'{output_dir}/degree_distribution_unweighted_undirected.png')
            plt.close()
            sns.scatterplot(x=range(len(degrees)), y=degrees)
            plt.title('Degree Scatter Plot (Unweighted, Undirected)')
            plt.xlabel('Node Index')
            plt.ylabel('Degree')
            plt.savefig(f'{output_dir}/degree_scatter_plot_unweighted_undirected.png')
            plt.close()
        elif self.weighted and not self.directed:
            degree_weights = [val for (node, val) in self.G.degree(weight='weight')]
            sns.histplot(degree_weights, kde=True)
            plt.title('Degree Weight Distribution (Weighted, Undirected)')
            plt.xlabel('Degree Weight')
            plt.ylabel('Frequency')
            plt.savefig(f'{output_dir}/degree_weight_distribution_weighted_undirected.png')
            plt.close()
            sns.scatterplot(x=range(len(degree_weights)), y=degree_weights)
            plt.title('Degree Weight Scatter Plot (Weighted, Undirected)')
            plt.xlabel('Node Index')
            plt.ylabel('Degree Weight')
            plt.savefig(f'{output_dir}/degree_weight_scatter_plot_weighted_undirected.png')
            plt.close()
        elif not self.weighted and self.directed:
            in_degrees = [val for (node, val) in self.G.in_degree()]
            out_degrees = [val for (node, val) in self.G.out_degree()]
            sns.histplot(in_degrees, kde=True, color='blue', label='In-Degree')
            sns.histplot(out_degrees, kde=True, color='orange', label='Out-Degree')
            plt.title('In-Degree and Out-Degree Distributions (Unweighted, Directed)')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(f'{output_dir}/in_out_degree_distribution_unweighted_directed.png')
            plt.close()
            sns.scatterplot(x=range(len(in_degrees)), y=in_degrees, color='blue', label='In-Degree')
            sns.scatterplot(x=range(len(out_degrees)), y=out_degrees, color='orange', label='Out-Degree')
            plt.title('In-Degree and Out-Degree Scatter Plot (Unweighted, Directed)')
            plt.xlabel('Node Index')
            plt.ylabel('Degree')
            plt.legend()
            plt.savefig(f'{output_dir}/in_out_degree_scatter_plot_unweighted_directed.png')
            plt.close()
        elif self.weighted and self.directed:
            in_degrees = [val for (node, val) in self.G.in_degree()]
            out_degrees = [val for (node, val) in self.G.out_degree()]
            in_strengths = [val for (node, val) in self.G.in_degree(weight='weight')]
            out_strengths = [val for (node, val) in self.G.out_degree(weight='weight')]

            plt.figure(figsize=(15, 10))

            # Plot 1: In-Degree and Out-Degree
            plt.subplot(2, 2, 1)
            sns.histplot(in_degrees, kde=True, color='blue', label='In-Degree')
            sns.histplot(out_degrees, kde=True, color='orange', label='Out-Degree')
            plt.title('In-Degree and Out-Degree Distributions')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(f'{output_dir}/in_out_degree_distribution_weighted_directed.png')

            # Plot 2: In-Degree and In-Strength
            plt.subplot(2, 2, 2)
            sns.histplot(in_degrees, kde=True, color='blue', label='In-Degree')
            sns.histplot(in_strengths, kde=True, color='green', label='In-Strength')
            plt.title('In-Degree and In-Strength Distributions')
            plt.xlabel('In-Degree / In-Strength')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(f'{output_dir}/in_degree_in_strength_distribution_weighted_directed.png')

            # Plot 3: Out-Degree and Out-Strength
            plt.subplot(2, 2, 3)
            sns.histplot(out_degrees, kde=True, color='orange', label='Out-Degree')
            sns.histplot(out_strengths, kde=True, color='red', label='Out-Strength')
            plt.title('Out-Degree and Out-Strength Distributions')
            plt.xlabel('Out-Degree / Out-Strength')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(f'{output_dir}/out_degree_out_strength_distribution_weighted_directed.png')

            # Plot 4: In-Strength and Out-Strength
            plt.subplot(2, 2, 4)
            sns.histplot(in_strengths, kde=True, color='green', label='In-Strength')
            sns.histplot(out_strengths, kde=True, color='red', label='Out-Strength')
            plt.title('In-Strength and Out-Strength Distributions')
            plt.xlabel('In-Strength / Out-Strength')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(f'{output_dir}/in_out_strength_distribution_weighted_directed.png')

            plt.tight_layout()
            plt.close()

            plt.figure(figsize=(15, 10))

            # Scatter Plot 1: In-Degree and Out-Degree
            plt.subplot(2, 2, 1)
            sns.scatterplot(x=range(len(in_degrees)), y=in_degrees, color='blue', label='In-Degree')
            sns.scatterplot(x=range(len(out_degrees)), y=out_degrees, color='orange', label='Out-Degree')
            plt.title('In-Degree and Out-Degree Scatter Plot')
            plt.xlabel('Node Index')
            plt.ylabel('Degree')
            plt.legend()
            plt.savefig(f'{output_dir}/in_out_degree_scatter_plot_weighted_directed.png')

            # Scatter Plot 2: In-Degree and In-Strength
            plt.subplot(2, 2, 2)
            sns.scatterplot(x=range(len(in_degrees)), y=in_degrees, color='blue', label='In-Degree')
            sns.scatterplot(x=range(len(in_strengths)), y=in_strengths, color='green', label='In-Strength')
            plt.title('In-Degree and In-Strength Scatter Plot')
            plt.xlabel('Node Index')
            plt.ylabel('In-Degree / In-Strength')
            plt.legend()
            plt.savefig(f'{output_dir}/in_degree_in_strength_scatter_plot_weighted_directed.png')

            # Scatter Plot 3: Out-Degree and Out-Strength
            plt.subplot(2, 2, 3)
            sns.scatterplot(x=range(len(out_degrees)), y=out_degrees, color='orange', label='Out-Degree')
            sns.scatterplot(x=range(len(out_strengths)), y=out_strengths, color='red', label='Out-Strength')
            plt.title('Out-Degree and Out-Strength Scatter Plot')
            plt.xlabel('Node Index')
            plt.ylabel('Out-Degree / Out-Strength')
            plt.legend()
            plt.savefig(f'{output_dir}/out_degree_out_strength_scatter_plot_weighted_directed.png')

            # Scatter Plot 4: In-Strength and Out-Strength
            plt.subplot(2, 2, 4)
            sns.scatterplot(x=range(len(in_strengths)), y=in_strengths, color='green', label='In-Strength')
            sns.scatterplot(x=range(len(out_strengths)), y=out_strengths, color='red', label='Out-Strength')
            plt.title('In-Strength and Out-Strength Scatter Plot')
            plt.xlabel('Node Index')
            plt.ylabel('In-Strength / Out-Strength')
            plt.legend()
            plt.savefig(f'{output_dir}/in_out_strength_scatter_plot_weighted_directed.png')

            plt.tight_layout()
            plt.close()
        if self.verbose:
            print("Degree distributions saved.")

# Example Usage:
# graph = nx.random_graphs.erdos_renyi_graph(100, 0.1)
# adj_matrix = nx.to_numpy_array(graph)
# visualizer = NetworkVisualizer(graph=graph, directed=False, weighted=False, verbose=True)
# visualizer.show_network()
# visualizer.show_heatmap()
# visualizer.show_distributions()
# visualizer.save_distributions(output_dir='output_directory')

# Example usage with CSV:
# visualizer = NetworkVisualizer(csv_path='path_to_your_csv.csv', directed=True, weighted=True, verbose=True)
# visualizer.show_network()
# visualizer.show_heatmap()
# visualizer.show_distributions()
# visualizer.save_distributions(output_dir='output_directory')
