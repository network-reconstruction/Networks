import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from random_generation import DirectedS1Generator
import os
import logging
from typing import List

#TODO perhaps save multiple coordinates for everything generated?
#TODO plots with customizable directories and features
class DirectedS1EnsembleAnalyser:
    """
    Class to analyze an ensemble of directed networks generated using the S1 model.

    Attributes:
        gen (DirectedS1Generator): The generator used to create the networks.
        verbose (bool): Whether to print log messages.
        reciprocity_list (List[float]): List of reciprocity values for each generated network.
        clustering_list (List[float]): List of clustering coefficients for each generated network.
        in_degree_sequences (List[List[int]]): List of in-degree sequences for each generated network.
        out_degree_sequences (List[List[int]]): List of out-degree sequences for each generated network.
        average_reciprocity (float): Average reciprocity across the ensemble.
        average_clustering (float): Average clustering coefficient across the ensemble.
        variance_reciprocity (float): Variance of reciprocity across the ensemble.
        variance_clustering (float): Variance of clustering coefficient across the ensemble.
        average_in_degree (float): Average in-degree across the ensemble.

    """
    def __init__(self,
                 gen: DirectedS1Generator,
                 verbose: bool = False,
                 log_file_path: str = "logs/DirectedS1EnsembleAnalyser/output.log"):
        """
        Initializes the analyzer with a generator and default attributes.

        Args:
            gen (DirectedS1Generator): The generator used to create the networks.
            verbose (bool): Whether to print log messages.
            log_file_path (str): The path to the log file.
        """
        self.verbose = verbose
        # TODO this line is kinda redundant
        self.logger = self._setup_logging(log_file_path)
        self.log_file_path = log_file_path
        self.gen = gen
        
        self.reciprocity_list = []
        self.clustering_list = []
        self.in_degree_sequences = []
        self.out_degree_sequences = []
        self.average_reciprocity = 0
        self.average_clustering = 0
        self.variance_reciprocity = 0
        self.variance_clustering = 0
        self.average_in_degree = 0

        
    def _setup_logging(self, log_file_path: str) -> logging.Logger:
        """
        Sets up logging with the specified log file path.

        Args:
            log_file_path (str): Path to the log file.

        Returns:
            logging.Logger: Configured logger.
        """
        for i in range(1, len(log_file_path.split("/"))):
            if not os.path.exists("/".join(log_file_path.split("/")[:i])):
                os.mkdir("/".join(log_file_path.split("/")[:i]))
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        if log_file_path:
            handler = logging.FileHandler(log_file_path, mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler) 
        return logger
        
    def set_params(self, **kwargs) -> None:
        """
        Sets parameters for the DirectedS1Generator.

        Args:
            **kwargs (dict): The parameters to set.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def run_analysis(self, 
                     num_samples: int = 10,
                     hidden_variables_data: dict = None,
                     hidden_variables_filename: str = "",
                     output_rootname: str = "",
                     save_results: bool = False, 
                     plot_degree_distribution: bool = False) -> None:
        """
        Runs the ensemble analysis on the generated networks.

        Args:
            num_samples (int): Number of samples to generate.
            hidden_variables_data (dict): Dictionary of hidden variables. Defaults to None.
            hidden_variables_filename (str): Path to the hidden variables file.
            output_rootname (str): Rootname for the output files. Defaults to "".
            save_results (bool): Whether to save the results to a JSON file. Defaults to False.
            plot_degree_distribution (bool): Whether to plot the degree distribution. Defaults to False.

        The analysis includes:
            - Generating the edgelist
            - Calculating the reciprocity and clustering coefficient
            - Calculating the in-degree and out-degree sequences
            - Saving the results to a JSON file
            - Plotting the degree distribution
        """
        #TODO this is kinda bad way to modify log file path
        self.gen.modify_log_file_path(self.log_file_path)  
        if self.verbose:
            self.logger.info("Running ensemble analysis")
            if hidden_variables_data is not None:
                self.logger.info("Generating networks from inference data")
            else:
                self.logger.info("Generating networks from file")
                  
        for _ in range(num_samples):
            self.gen.network.clear()
            if hidden_variables_data is not None:
                self.gen.generate_from_inference_data(hidden_variables_data, output_rootname)
            else:
                self.gen.generate_from_file(hidden_variables_filename)

                
            reciprocity, clustering, in_degree_sequence, out_degree_sequence = self.gen.calculate_metrics()
            self.reciprocity_list.append(reciprocity)
            self.clustering_list.append(clustering)
            self.in_degree_sequences.append(in_degree_sequence)
            self.out_degree_sequences.append(out_degree_sequence)
        
        if not os.path.exists("outputs"):
            os.makedirs("outputs")  
        if not os.path.exists(f"outputs/{self.gen.output_rootname}"):
            os.makedirs(f"outputs/{self.gen.output_rootname}")
        if not os.path.exists(f"outputs/{self.gen.output_rootname}/ensemble_analysis"):
            os.makedirs(f"outputs/{self.gen.output_rootname}/ensemble_analysis")
        if save_results:
            self._save_data()
        if plot_degree_distribution:
            self._plot_degree_distribution()
            
    def get_output(self)-> dict:
        """   
        Obtains the results of the ensemble analysis such as average reciprocity, average clustering,
        variance reciprocity, variance clustering, reciprocity list, clustering list, average in-degree, in-degree sequences,
        and out-degree sequences.
        
        Returns:
            dict: Dictionary containing the results.
        
        Output contains the following:
            average_reciprocity (float): Average reciprocity across the ensemble.
            average_clustering (float): Average clustering coefficient across the ensemble.
            average_in_degree (float): Average in-degree across the ensemble.
            variance_reciprocity (float): Variance of reciprocity across the ensemble.
            variance_clustering (float): Variance of clustering coefficient across the ensemble.
            variance_in_degree (float): Variance of in-degree across the ensemble.
            reciprocity_list (List[float]): List of reciprocity values for each generated network.
            clustering_list (List[float]): List of clustering coefficients for each generated network.
            in_degree_sequences (List[List[int]]): List of in-degree sequences for each generated network.
            out_degree_sequences (List[List[int]]): List of out-degree sequences for each generated network.
            
        """
        self.average_reciprocity = np.mean(self.reciprocity_list)
        self.average_clustering = np.mean(self.clustering_list) 
        self.variance_reciprocity = np.var(self.reciprocity_list)
        self.variance_clustering = np.var(self.clustering_list) 
        self.average_in_degree = np.mean([np.mean(seq) for seq in self.in_degree_sequences])
        self.variance_in_degree = np.var([np.mean(seq) for seq in self.in_degree_sequences])
        
        output = {
            "average_reciprocity": self.average_reciprocity,
            "average_clustering": self.average_clustering,
            "average_in_degree": self.average_in_degree,
            "variance_reciprocity": self.variance_reciprocity,
            "variance_clustering": self.variance_clustering,
            "variance_in_degree": self.variance_in_degree,
            "reciprocity_list": self.reciprocity_list,
            "clustering_list": self.clustering_list,
            "in_degree_sequences": self.in_degree_sequences,
            "out_degree_sequences": self.out_degree_sequences
            
        }
        
        return output
    
    def _save_data(self) -> None:
        """
        Saves the analysis output data to a JSON file.

        Output contains the following:
            average_reciprocity (float): Average reciprocity across the ensemble.
            average_clustering (float): Average clustering coefficient across the ensemble.
            average_in_degree (float): Average in-degree across the ensemble.
            variance_reciprocity (float): Variance of reciprocity across the ensemble.
            variance_clustering (float): Variance of clustering coefficient across the ensemble.
            variance_in_degree (float): Variance of in-degree across the ensemble.
            reciprocity_list (List[float]): List of reciprocity values for each generated network.
            clustering_list (List[float]): List of clustering coefficients for each generated network.
            in_degree_sequences (List[List[int]]): List of in-degree sequences for each generated network.
            out_degree_sequences (List[List[int]]): List of out-degree sequences for each generated network.
            
        """
        
        output = self.get_output()
        # if the folder does not exist, create it
        
        with open(f"outputs/{self.gen.output_rootname}/ensemble_analysis/results.json", 'w') as f:
            json.dump(output, f, indent=4)
    
    def _plot_degree_distribution(self) -> None:
        """
        Plots the degree distribution of the generated networks.

        Plots:
        - Complementary cumulative degree distributions for in-degree and out-degree
        - In-degree vs out-degree distribution
        """
        #create directory for figure outputs
        if not os.path.exists(f"outputs/{self.gen.output_rootname}/ensemble_analysis/figs"):
            os.makedirs(f"outputs/{self.gen.output_rootname}/ensemble_analysis/figs")
            
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
        plt.savefig(f"outputs/{self.gen.output_rootname}/ensemble_analysis/figs/complementary_cumulative_degree_distribution.png")

        # Plotting in-degree vs out-degree
        plt.figure(figsize=(8, 6))
        plt.scatter(all_in_degrees, all_out_degrees, alpha=0.5, color='purple')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("In-Degree vs Out-Degree Distribution")
        plt.xlabel("In-Degree")
        plt.ylabel("Out-Degree")
        plt.grid(True)
        plt.savefig(f"outputs/{self.gen.output_rootname}/ensemble_analysis/figs/in_vs_out_degree_distribution.png")
        
       
if __name__ == "__main__":
    functional_random_generation_model = DirectedS1Generator(verbose=True,
                               log_file_path = "logs/DirectedS1Generator/test_generation_functional_generate_from_file.log")
    functional_ensemble_analysis_model = DirectedS1EnsembleAnalyser(gen = functional_random_generation_model,
                                    verbose = True,
                                    log_file_path = "logs/DirectedS1EnsembleAnalyser/test_ensemble_analysis_functional.log")
    # functional_random_generation_model.modify_log_file_path("logs/DirectedS1EnsembleAnalyser/test_ensemble_analysis_functional.log")
    functional_ensemble_analysis_model.run_analysis(hidden_variables_filename = "outputs/TestNetwork/inferred_params.json",
                                                    num_samples = 10,
                                                    save_results = True,
                                                    plot_degree_distribution = True)