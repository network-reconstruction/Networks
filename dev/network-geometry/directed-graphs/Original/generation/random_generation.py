import json
import math
import random
import time
from datetime import datetime
import sys
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple
import logging
import os
class DirectedS1Generator:
    """
    Class to generate a directed network using the S1 model.
    
    Params:
    -------
    seed: int
        The seed for the random number generator
    verbose: bool
        Whether to print log messages
    log_file_path: str
        The path to the log file
    """
    def __init__(self,
                 seed: int = 0,
                 verbose: bool = False,
                 log_file_path: str = "logs/DirectedS1Generator/output.log"):
        
        self.verbose = verbose
        # Set up logging
        # -----------------------------------------------------
        for i in range(1, len(log_file_path.split("/"))):
            if not os.path.exists("/".join(log_file_path.split("/")[:i])):
                os.mkdir("/".join(log_file_path.split("/")[:i]))
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
        
        self.save_coordinates = False
        
        self.SEED = seed
        random.seed(self.SEED)
        
        self.BETA = -10
        self.MU = -10
        self.NU = -10
        self.R = -10
        self.hidden_variables_filename = ""
        self.output_rootname = ""
        self.PI = math.pi
        self.NUMERICAL_ZERO = 1e-5
        
        self.Num2Name = []
        
        self.nb_vertices = 0
        self.in_Kappa = []
        self.outKappa = []
        self.theta = []
        self.network = nx.DiGraph()
        
        self.reciprocity = 0    
        self.clustering = 0
        self.in_degree_sequence = []
        self.out_degree_sequence = []
        
        
    def _load_hidden_variables(self) -> None:
        """
        Load the hidden variables from the hidden variables file.
        """
        if self.verbose:
            self.logger.info(f"Loading hidden variables from {self.hidden_variables_filename}")
        with open(self.hidden_variables_filename, 'r') as file:
            data = json.load(file)
        
        self.BETA = data.get("beta", self.BETA)
        self.MU = data.get("mu", self.MU)
        self.NU = data.get("nu", self.NU)
        self.R = data.get("R", self.R)
        
        inferred_kappas = data.get("inferred_kappas", [])
        self.nb_vertices = len(inferred_kappas)
        
        self.in_Kappa = [kappa["kappa_in"] for kappa in inferred_kappas]
        self.outKappa = [kappa["kappa_out"] for kappa in inferred_kappas]
        self.Num2Name = [f"node_{i}" for i in range(self.nb_vertices)]
        
        if self.theta:
            self.theta = [kappa["theta"] for kappa in inferred_kappas]
        else:
            self.theta = [2 * self.PI * random.random() for _ in range(self.nb_vertices)]
        
    def _generate_edgelist(self) -> None:
        """
        Generate the edgelist of the network using the hidden variables.
        """
        if self.verbose:
            self.logger.info(f"Generating edgelist of length {self.nb_vertices}")
        if self.BETA == -10:
            raise ValueError("The value of parameter beta must be provided.")
        
        if self.MU == -10:
            average_kappa = sum((in_k + out_k) / 2.0 for in_k, out_k in zip(self.in_Kappa, self.outKappa)) / self.nb_vertices
            self.MU = self.BETA * math.sin(self.PI / self.BETA) / (2.0 * self.PI * average_kappa)
        
        if self.NU == -10:
            raise ValueError("The value of parameter nu must be provided.")
        
        prefactor = self.nb_vertices / (2 * self.PI * self.MU)
        for v1 in range(self.nb_vertices):
            kout1 = self.outKappa[v1]
            k_in1 = self.in_Kappa[v1]
            theta1 = self.theta[v1]
            for v2 in range(v1 + 1, self.nb_vertices):
                dtheta = self.PI - abs(self.PI - abs(theta1 - self.theta[v2]))
                koutkin = kout1 * self.in_Kappa[v2]
                p12 = 1 / (1 + ((prefactor * dtheta) / koutkin) ** self.BETA) if koutkin > self.NUMERICAL_ZERO else 0
                koutkin = self.outKappa[v2] * k_in1
                p21 = 1 / (1 + ((prefactor * dtheta) / koutkin) ** self.BETA) if koutkin > self.NUMERICAL_ZERO else 0
                
                if self.NU > 0:
                    p11 = ((1 - self.NU) * max(p12, p21) + self.NU) * min(p12, p21)
                else:
                    p11 = (1 + self.NU) * p12 * p21 + self.NU * (1 - p12 - p21) if p12 + p21 >= 1 else (1 + self.NU) * p12 * p21
                
                r = random.random()
                if r < p11:
                    self.network.add_edge(self.Num2Name[v1], self.Num2Name[v2])
                    self.network.add_edge(self.Num2Name[v2], self.Num2Name[v1])
                elif r < p21:
                    self.network.add_edge(self.Num2Name[v2], self.Num2Name[v1])
                elif r < (p21 + p12 - p11):
                    self.network.add_edge(self.Num2Name[v1], self.Num2Name[v2])
        
    def calculate_metrics(self) -> Tuple[float, float, List[int], List[int]]:
        """
        Calculate the reciprocity, clustering coefficient, in-degree sequence, and out-degree sequence of the network.
        """
        if self.verbose:
            self.logger.info("Calculating metrics")
        self.reciprocity = nx.reciprocity(self.network)
        self.clustering = nx.average_clustering(self.network.to_undirected())
        self.in_degree_sequence = [d for n, d in self.network.in_degree()]
        self.out_degree_sequence = [d for n, d in self.network.out_degree()]
        return self.reciprocity, self.clustering, self.in_degree_sequence, self.out_degree_sequence
    
    def _save_data(self) -> None:
        """
        Saves data to outputs/<output_rootname>/generation_data.json
        """
        if self.verbose:
            self.logger.info(f"Saving data to outputs/{self.output_rootname}/generation_data.json")
        adjacency_list = [list(self.network.neighbors(node)) for node in self.network.nodes()]
        self.calculate_metrics()
        data = {
            "parameters": {
                "beta": self.BETA,
                "mu": self.MU,
                "nu": self.NU,
                "N": self.nb_vertices,
                "R": self.R,
                "seed": self.SEED,
                "hidden_variables_file": self.hidden_variables_filename
            },
            "adjacency_list": adjacency_list,
            "network_data": {
                "reciprocity": self.reciprocity,
                "clustering": self.clustering,
                "in_degree_sequence": self.in_degree_sequence,
                "out_degree_sequence": self.out_degree_sequence
            }
        }
        
        if not os.path.exists("outputs"):
            os.makedirs("outputs")  
        if not os.path.exists(f"outputs/{self.output_rootname}"):
            os.makedirs(f"outputs/{self.output_rootname}")
        with open(f"outputs/{self.output_rootname}/generation_data.json", 'w') as f:
            json.dump(data, f, indent=4)
            
    def _save_coordinates(self) -> None:
        """
        Save the coordinates of the nodes to outputs/<output_rootname>/coordinates.csv
        """
        if self.verbose:
            self.logger.info(f"Current working directory: {os.getcwd()}")
            self.logger.info(f"Saving coordinates to outputs/{self.output_rootname}/coordinates.csv")
        #node in_kappa, out_kappa, theta save as dataframe
        if not os.path.exists("outputs"):
            os.makedirs("outputs")  
        if not os.path.exists(f"outputs/{self.output_rootname}"):
            os.makedirs(f"outputs/{self.output_rootname}")
        with open(f"outputs/{self.output_rootname}/coordinates.csv", 'w') as f:
            f.write("node,in_kappa,out_kappa,theta\n")
            for i in range(self.nb_vertices):
                f.write(f"{self.Num2Name[i]},{self.in_Kappa[i]},{self.outKappa[i]},{self.theta[i]}\n")
                
    def set_params(self,
                    hidden_variables_filename,
                    output_rootname: str = "",
                    theta: List[int] = None,
                    save_coordinates: bool = False) -> None:  
        """
        Set the parameters for the generator.
        
        Params:
        -------
        hidden_variables_filename: str
            The filename of the hidden variables json file
        output_rootname: str
            The rootname of the output files, if not assigned it will be the same as the hidden variables subdirectory without file name and parent directories.
        theta: List[int]
            The theta values for the network
        save_coordinates: bool
            Whether to save the coordinates of the nodes
        
        """
        if self.verbose:
            self.logger.info(f"Setting parameters: hidden_variables_filename: {hidden_variables_filename}, output_rootname: {output_rootname}, theta: {theta}, save_coordinates: {save_coordinates}")
        self.save_coordinates = save_coordinates
        self.theta = theta
        self.hidden_variables_filename = hidden_variables_filename
        if output_rootname == "":
            self.output_rootname = hidden_variables_filename.split(".")[0].split("/")[-2]
            if self.verbose:
                self.logger.info(f"Output rootname set to {self.output_rootname}")
           
    def generate(self, 
                 hidden_variables_filename,
                 output_rootname: str = "",
                 theta: List[int] = None,
                 save_coordinates: bool = False) -> None:
        """
        Generate a directed network using the hidden variables provided in the hidden_variables_filename, and save the data.
        
        Params:
        -------
        hidden_variables_filename: str
            The filename of the hidden variables json file
        output_rootname: str
            The rootname of the output files
        theta: List[int]
            The theta values for the network
        save_coordinates: bool
            Whether to save the coordinates of the nodes
        """
        if self.verbose:
            self.logger.info(f"Generating network with hidden variables from {hidden_variables_filename}")
        self.set_params(hidden_variables_filename, output_rootname, theta, save_coordinates)
        self._load_hidden_variables()
        self._generate_edgelist()
        self._save_data()
        if self.save_coordinates:
            self._save_coordinates()
        if self.verbose:
            self.logger.info(f"Finished generating network at {self.get_time()}")
    
    def modify_log_file_path(self, log_file_path: str) -> None:
        """
        Modify the log file path for the logger.
        
        Params:
        -------
        log_file_path: str
            The new log file path
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
    def get_time(self) -> str:
        return datetime.utcnow().strftime("%Y/%m/%d %H:%M UTC")

    def plot_degree_sequence(self, degree_sequence) -> None:
        plt.figure()
        plt.hist(degree_sequence, bins='auto')
        plt.title("Degree Sequence Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.savefig(f"{self.output_rootname}_degree_sequence.png")


if __name__ == "__main__":
    gen = DirectedS1Generator()
    gen.generate(hidden_variables_filename = "outputs/TestNetwork/inferred_params.json",
                 save_coordinates = True)
    
    print(f"Finished at {gen.get_time()}")
