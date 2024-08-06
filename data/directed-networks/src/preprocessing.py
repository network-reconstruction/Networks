# edges.csv will be in
# source, target, weight
# 0,0,0.8
# 0,1,1.0
# 0,2,1.0
# 0,3,1.0
# 0,4,1.0
# 0,5,0.8
# 0,6,0.6

#format, always with source anf target as the first two columns and more columns, if one of the columns is "layer", 
# then separate each layer into a separate network, if not then save whole file as binary directed network.
import os
import networkx as nx
import pandas as pd
import json

class NetworkProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.network_data = {}

    def load_data(self, file_path):
        return pd.read_csv(file_path)
    
    def process_network(self, edges_df):
        network_data = {}
        if 'layer' in edges_df.columns:
            layers = edges_df['layer'].unique()
            for layer in layers:
                layer_edges = edges_df[edges_df['layer'] == layer]
                G = nx.from_pandas_edgelist(layer_edges, source='source', target='target', edge_attr=True, create_using=nx.DiGraph())
                network_data[layer] = self.extract_network_data(G)
        else:
            G = nx.from_pandas_edgelist(edges_df, source='source', target='target', edge_attr=True, create_using=nx.DiGraph())
            network_data = self.extract_network_data(G)
        return network_data

    def extract_network_data(self, G):
        reciprocity = nx.reciprocity(G)
        avg_clustering = nx.average_clustering(G)
        in_degree_sequence = [d for n, d in G.in_degree()]
        out_degree_sequence = [d for n, d in G.out_degree()]

        network_data = {
            'reciprocity': reciprocity,
            'average_clustering': avg_clustering,
            'in_degree_sequence': in_degree_sequence,
            'out_degree_sequence': out_degree_sequence
        }
        
        return network_data

    def save_data(self, data, output_json):
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=4)

    def recursive_process(self):
        all_network_data = {}
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file == 'edges.csv':
                    file_path = os.path.join(subdir, file)
                    edges_df = self.load_data(file_path)
                    network_data = self.process_network(edges_df)
                    relative_path = os.path.relpath(subdir, self.root_dir)
                    all_network_data[relative_path] = network_data
        
        output_dir = os.path.join(self.root_dir, 'directed_networks/all_data_json')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'network_data.json')
        self.save_data(all_network_data, output_file)

# Example usage
if __name__ == "__main__":
    processor = NetworkProcessor('data/directed-networks')
    processor.recursive_process()
