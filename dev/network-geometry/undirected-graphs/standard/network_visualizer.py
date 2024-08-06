import matplotlib.pyplot as plt
from pyvis.network import Network
import networkx as nx

class NetworkVisualizer:
    def __init__(self, graph: nx.Graph):
        self.G: nx.Graph = graph

    def plot_network(self, filename: str = "s1_network.html") -> str:
        net = Network(notebook=False)
        net.from_nx(self.G)
        net.write_html(filename)  # Save the HTML file without opening it in a browser
        return filename

    def plot_degree_distribution(self, filename: str = "degree_distribution.png") -> str:
        degrees = [self.G.degree(n) for n in self.G.nodes()]        
        plt.hist(degrees, bins=30, edgecolor='black')
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        # plt.yscale('log')

        plt.savefig(filename)  # Save the plot as a normal image file
        plt.close()
        return filename
