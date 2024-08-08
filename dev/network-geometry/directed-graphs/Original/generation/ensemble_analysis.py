import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from random_generation import DirectedS1Generator
import os
import logging
from typing import List

class GraphEnsembleAnalysis:
    def __init__(self,
                 gen: DirectedS1Generator,
                 verbose: bool = False,
                 log_file_path: str = "logs/GraphEnsembleAnalysis/output.log"):
        self.verbose = verbose
        # Set up logging
        # -----------------------------------------------------
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        if log_file_path:
            handler = logging.FileHandler(log_file_path, mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # -----------------------------------------------------
        
        self.gen = gen
        self.num_samples = num_samples
        self.reciprocity_list = []
        self.clustering_list = []
        self.in_degree_sequences = []
        self.out_degree_sequences = []
        self.average_reciprocity = 0
        self.average_clustering = 0
        self.variance_reciprocity = 0
        self.variance_clustering = 0
        self.average_in_degree = 0
        self.num_samples = -1
        
    def set_params(self, 
                    hidden_variables_filename,
                    output_rootname: str = "",
                    theta: List[int] = None,
                    save_coordinates: bool = False, 
                    num_samples: int = 10):
        """
        Set the parameters for the underlying generator for ensemble analysis.
        """
        self.num_samples = num_samples
        self.gen.set_params(hidden_variables_filename, output_rootname, theta, save_coordinates)
        
        
    def run_analysis(self):
        """
        Run the ensemble analysis on the generated networks.
        
        The analysis includes:
        - Generating the edgelist
        - Calculating the reciprocity and clustering coefficient
        - Calculating the in-degree and out-degree sequences
        - Saving the results to a json file
        - Plotting the degree distribution
        """
        for _ in range(self.num_samples):
            self.gen.network.clear()
            self.gen.generate()
            reciprocity, clustering, in_degree_sequence, out_degree_sequence = self.gen.calculate_metrics()
            self.reciprocity_list.append(reciprocity)
            self.clustering_list.append(clustering)
            self.in_degree_sequences.append(in_degree_sequence)
            self.out_degree_sequences.append(out_degree_sequence)
        
        self.save_results()
        self.plot_degree_distribution()
    
    def save_results(self):
        self.average_reciprocity = np.mean(self.reciprocity_list)
        self.average_clustering = np.mean(self.clustering_list) 
        self.variance_reciprocity = np.var(self.reciprocity_list)
        self.variance_clustering = np.var(self.clustering_list) 
        self.average_in_degree = np.mean([np.mean(seq) for seq in self.in_degree_sequences])
        
        results = {
            "average_reciprocity": self.average_reciprocity,
            "average_clustering": self.average_clustering,
            "variance_reciprocity": self.variance_reciprocity,
            "variance_clustering": self.variance_clustering,
            "reciprocity_list": self.reciprocity_list,
            "clustering_list": self.clustering_list,
            "average_in_degree": self.average_in_degree,
            "in_degree_sequences": self.in_degree_sequences,
            "out_degree_sequences": self.out_degree_sequences
        }
        # if the folder does not exist, create it
        if not os.path.exists("outputs"):
            os.makedirs("outputs")  
        if not os.path.exists(f"outputs/{self.gen.output_rootname}"):
            os.makedirs(f"outputs/{self.gen.output_rootname}")
        with open(f"outputs/{self.gen.output_rootname}/ensemble_analysis_results.json", 'w') as f:
            json.dump(results, f, indent=4)
    
    def plot_degree_distribution(self):
        """
        Plot the degree distribution of the generated networks.
        
        Plots:
        - Complementary cumulative degree distributions for in-degree and out-degree
        - In-degree vs out-degree distribution
        """
        #create directory for figure outputs
        if not os.path.exists(f"outputs/{self.gen.output_rootname}/figs"):
            os.makedirs(f"outputs/{self.gen.output_rootname}/figs")
        all_in_degrees = np.concatenate(self.in_degree_sequences)
        all_out_degrees = np.concatenate(self.out_degree_sequences)
        
        # Plotting the complementary cumulative degree distributions
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        in_degree_sorted = np.sort(all_in_degrees)
        in_degree_ccdf = 1 - np.arange(1, len(in_degree_sorted)+1) / len(in_degree_sorted)
        plt.plot(in_degree_sorted, in_degree_ccdf, marker='o', linestyle='none', color='blue')
        plt.title("In-Degree CCDF")
        plt.xlabel("In-Degree")
        plt.ylabel("CCDF")
        plt.xscale('log')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        out_degree_sorted = np.sort(all_out_degrees)
        out_degree_ccdf = 1 - np.arange(1, len(out_degree_sorted)+1) / len(out_degree_sorted)
        plt.plot(out_degree_sorted, out_degree_ccdf, marker='o', linestyle='none', color='red')
        plt.title("Out-Degree CCDF")
        plt.xlabel("Out-Degree")
        plt.ylabel("CCDF")
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f"outputs/{self.gen.output_rootname}/figs/complementary_cumulative_degree_distribution.png")

        # Plotting in-degree vs out-degree
        plt.figure(figsize=(8, 6))
        plt.scatter(all_in_degrees, all_out_degrees, alpha=0.5, color='purple')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("In-Degree vs Out-Degree Distribution")
        plt.xlabel("In-Degree")
        plt.ylabel("Out-Degree")
        plt.grid(True)
        plt.savefig(f"outputs/{self.gen.output_rootname}/figs/in_vs_out_degree_distribution.png")
        
        # print reciprocity, clustering, in_degree_sequence, out_degree_sequence averages
        print(f"Average Reciprocity: {self.average_reciprocity}")
        print(f"Variance Reciprocity: {np.var(self.reciprocity_list)}")
        print(f"Average Clustering: {np.mean(self.clustering_list)}")
        print(f"Variance Clustering: {np.var(self.clustering_list)}")
        print(f"Average in-degree: {np.mean([np.mean(seq) for seq in self.in_degree_sequences])}")

    def modify_log_file_path(self, log_file_path: str) -> None:
        """
        Modify the log file path for the logger.
        
        Parameters
        ----------
        log_file_path: str 
            Path to the log file.
        """
        #create directory if doesn't exist
        for i in range(1, len(log_file_path.split("/"))):
            if not os.path.exists("/".join(log_file_path.split("/")[:i])):
                os.mkdir("/".join(log_file_path.split("/")[:i]))
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        handler = logging.StreamHandler(sys.stdout)
        if log_file_path:
            handler = logging.FileHandler(log_file_path, mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)




if __name__ == "__main__":
    gen = DirectedS1Generator()

    
    num_samples = 10    
    
    analysis = GraphEnsembleAnalysis(gen, num_samples)
    analysis.run_analysis()
    print(f"Finished at {gen.get_time()}")
