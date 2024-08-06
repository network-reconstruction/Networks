import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import sys
from random_generation import GeneratingDirectedS1

class GraphEnsembleAnalysis:
    def __init__(self, gen, num_samples):
        self.gen = gen
        self.num_samples = num_samples
        self.reciprocity_list = []
        self.clustering_list = []
        self.in_degree_sequences = []
        self.out_degree_sequences = []
    
    def run_analysis(self):
        for _ in range(self.num_samples):
            self.gen.network.clear()
            self.gen.generate_edgelist()
            reciprocity, clustering, in_degree_sequence, out_degree_sequence = self.gen.calculate_metrics()
            self.reciprocity_list.append(reciprocity)
            self.clustering_list.append(clustering)
            self.in_degree_sequences.append(in_degree_sequence)
            self.out_degree_sequences.append(out_degree_sequence)
        
        self.save_results()
        self.plot_degree_distribution()
    
    def save_results(self):
        avg_reciprocity = np.mean(self.reciprocity_list)
        avg_clustering = np.mean(self.clustering_list)
        
        
        results = {
            "average_reciprocity": avg_reciprocity,
            "average_clustering": avg_clustering,
            "in_degree_sequences": self.in_degree_sequences,
            "out_degree_sequences": self.out_degree_sequences
        }
        
        with open(f"{self.gen.OUTPUT_ROOTNAME}_analysis_results.json", 'w') as f:
            json.dump(results, f, indent=4)
    
    def plot_degree_distribution(self):
        all_in_degrees = np.concatenate(self.in_degree_sequences)
        all_out_degrees = np.concatenate(self.out_degree_sequences)
        
        # Plotting the empirical degree distributions
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(all_in_degrees, bins='auto', alpha=0.7, color='blue', density=True)
        plt.title("In-Degree Distribution")
        plt.xlabel("In-Degree")
        plt.ylabel("Density")
        
        plt.subplot(1, 2, 2)
        plt.hist(all_out_degrees, bins='auto', alpha=0.7, color='red', density=True)
        plt.title("Out-Degree Distribution")
        plt.xlabel("Out-Degree")
        plt.ylabel("Density")
        
        plt.tight_layout()
        plt.savefig(f"{self.gen.OUTPUT_ROOTNAME}_degree_distribution.png")
        plt.show()

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
        plt.savefig(f"{self.gen.OUTPUT_ROOTNAME}_complementary_cumulative_degree_distribution.png")
        plt.show()

        # Plotting in-degree vs out-degree
        plt.figure(figsize=(8, 6))
        plt.scatter(all_in_degrees, all_out_degrees, alpha=0.5, color='purple')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("In-Degree vs Out-Degree Distribution")
        plt.xlabel("In-Degree")
        plt.ylabel("Out-Degree")
        plt.grid(True)
        plt.savefig(f"{self.gen.OUTPUT_ROOTNAME}_in_vs_out_degree_distribution.png")
        plt.show()
        
        # print reciprocity, clustering, in_degree_sequence, out_degree_sequence averages
        print(f"Average Reciprocity: {np.mean(self.reciprocity_list)}")
        print(f"Average Clustering: {np.mean(self.clustering_list)}")

       




if __name__ == "__main__":
    gen = GeneratingDirectedS1()
    gen.CUSTOM_OUTPUT_ROOTNAME = False
    gen.NAME_PROVIDED = False
    gen.NATIVE_INPUT_FILE = True
    gen.THETA_PROVIDED = False
    gen.SAVE_COORDINATES = False
    gen.HIDDEN_VARIABLES_FILENAME = sys.argv[1]
    gen.OUTPUT_ROOTNAME = sys.argv[2]
    
    gen.load_hidden_variables()
    
    num_samples = int(sys.argv[3])
    
    analysis = GraphEnsembleAnalysis(gen, num_samples)
    analysis.run_analysis()
    print(f"Finished at {gen.get_time()}")
