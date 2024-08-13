import os
import networkx as nx
import pandas as pd
import json

class NetworkProcessor:
    def __init__(self, root_dir, max_edges=500000):
        self.root_dir = root_dir
        self.network_data = {}
        self.max_edges = max_edges
        self.all_network_data = {}

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()  # Strip leading/trailing whitespace from column names
        return df
    
    def process_network(self, edges_df, network_name):
        network_data = {}
        # Strip column names not in a-Z or 0-9
        edges_df.columns = edges_df.columns.str.replace('[^a-zA-Z0-9]', '', regex=True)
        if edges_df.shape[0] > self.max_edges:
            # skip if the number of edges exceeds the limit
            print(f"Skipping {edges_df.shape[0]} edges")
            return network_data
        if 'layer' in edges_df.columns:
            layers = edges_df['layer'].unique()
            for layer in layers:
                layer_edges = edges_df[edges_df['layer'] == layer]
                G = self.create_graph(layer_edges)
                layer_name = f"{network_name}_layer{layer}"
                self.all_network_data[layer_name] = self.extract_network_data(G)
        else:
            G = self.create_graph(edges_df)
            self.all_network_data[network_name] = self.extract_network_data(G)
        return network_data

    def create_graph(self, df):
        if 'weight' in df.columns:
            G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=True, create_using=nx.DiGraph())
        else:
            G = nx.from_pandas_edgelist(df, source='source', target='target', create_using=nx.DiGraph())
        return G

    def extract_network_data(self, G):
        reciprocity = nx.reciprocity(G)
        #compute average undirected clustering coefficient of G
        avg_clustering = nx.average_clustering(G.to_undirected())
        
        in_degree_sequence = [d for n, d in G.in_degree()]
        out_degree_sequence = [d for n, d in G.out_degree()]
        #avg in degree 
        average_in_degree = sum(in_degree_sequence) / len(in_degree_sequence)

        network_data = {
            'reciprocity': reciprocity,
            'average_clustering': avg_clustering,
            'in_degree_sequence': in_degree_sequence,
            'out_degree_sequence': out_degree_sequence,
            'average_in_degree': average_in_degree
        }
        
        return network_data

    def save_data(self, data, output_json):
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=4)

    def recursive_process(self):
        
        print(f'Processing {self.root_dir}')
        print(os.listdir(self.root_dir))    
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file == 'edges.csv':
                    print(f'Processing {file} in {subdir}')
                    file_path = os.path.join(subdir, file)
                    edges_df = self.load_data(file_path)
                    relative_path = os.path.relpath(subdir, self.root_dir)
                    #take the last two / separated values as the network name, unless they are the same string take the last one
                    if relative_path.split('/')[-1] == relative_path.split('/')[-2]:
                        network_name = relative_path.split('/')[-1]
                    else:
                        network_name = relative_path.split('/')[-2] + '_' + relative_path.split('/')[-1]
                    self.process_network(edges_df, network_name)
        output_file = os.path.join(self.root_dir, 'network_data.json')
        self.save_data(self.all_network_data, output_file)
        print(f"Data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    processor = NetworkProcessor('directed_networks')
    processor.recursive_process()
