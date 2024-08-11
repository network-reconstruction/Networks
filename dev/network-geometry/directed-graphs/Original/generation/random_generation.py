
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

#TODO unify Parameters and Parameters
#TODO fix set_params log_file_path to be more rebust.
#TODO save coordinates and data should it be in generate or take in parameter for where to save to?
#TODO Set theta always uniform, how to fix?
class DirectedS1Generator:
    """
    Class to generate a directed network using the S1 model.

    Attributes:
        PI (float): The value of Pi.
        SEED (int): The random seed.
        BETA (float): Parameter for the model.
        MU (float): Parameter for the model.
        NU (float): Parameter for the model.
        R (float): Parameter for the model.
        output_rootname (str): The rootname for output files.
        NUMERICAL_ZERO (float): A small numerical value considered as zero.
        Num2Name (List[str]): List of node names.
        nb_vertices (int): Number of vertices.
        in_Kappa (List[float]): In-degree distribution.
        out_Kappa (List[float]): Out-degree distribution.
        theta (List[float]): Angular positions of nodes.
        network (nx.DiGraph): The generated network.
        reciprocity (float): Network reciprocity.
        clustering (float): Network clustering coefficient.
        in_degree_sequence (List[int]): In-degree sequence.
        out_degree_sequence (List[int]): Out-degree sequence.

    """
    def __init__(self,
                 seed: int = 0,
                 verbose: bool = False,
                 log_file_path: str = "logs/DirectedS1Generator/output.log"):
        """
        Initializes the model with default parameters and attributes.

        Args:
            seed (int): The seed for the random number generator.
            verbose (bool): Whether to print log messages.
            log_file_path (str): The path to the log file.
        """
        self.verbose = verbose
        # Set up logging
        # -----------------------------------------------------
        self.logger = self._setup_logging(log_file_path)
        # -----------------------------------------------------
        
        self.SEED = seed
        random.seed(self.SEED)
        
        self.BETA = -10
        self.MU = -10
        self.NU = -10
        self.R = -10
        self.output_rootname = ""
        self.PI = math.pi
        self.NUMERICAL_ZERO = 1e-5
        

        
        self.nb_vertices = 0
        self.in_Kappa = []
        self.out_Kappa = []
        self.theta = None
        self.network = nx.DiGraph()
        
        self.reciprocity = 0    
        self.clustering = 0
        self.in_degree_sequence = []
        self.out_degree_sequence = []
        
    def _setup_logging(self, log_file_path: str) -> logging.Logger:
        """
        Setup logging with the given log file path.

        Args:
            log_file_path (str): Path to the log file.

        Returns:
            logging.Logger: Logger
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
        
    def set_hidden_variables(self, data: dict) -> None:
        """
        Set the hidden variables from the data.
        
        Args:
            data (dict): The data containing the hidden variables.
                Data contains the following:
                    beta (float): The beta parameter.
                    mu (float): The mu parameter.
                    nu (float): The nu parameter.
                    R (float): The R parameter.
                    inferred_kappas (List[Dict[str, float]]): The inferred kappas.
                    flags (List[Dict[str, Any]]): The flags (not used).
                
        """
        
        #RESET everytime new data is loaded
        self.network = nx.DiGraph()
        
        self.BETA = data.get("beta", self.BETA)
        self.MU = data.get("mu", self.MU)
        self.NU = data.get("nu", self.NU)
        self.R = data.get("R", self.R)
        
        inferred_kappas = data.get("inferred_kappas", [])
        self.nb_vertices = len(inferred_kappas)
        
        self.in_Kappa = [kappa["kappa_in"] for kappa in inferred_kappas]
        self.out_Kappa = [kappa["kappa_out"] for kappa in inferred_kappas]
        
        self.theta = [2 * self.PI * random.random() for _ in range(self.nb_vertices)]
        
    def _load_hidden_variables(self, hidden_variables_filename: str) -> None:
        """
        Load the hidden variables from the hidden variables file.
        
        Args:
            hidden_variables_filename (str): The filename with full
            path of the hidden variables json file.
        """
        if self.verbose:
            self.logger.info(f"Loading hidden variables from {hidden_variables_filename}")
        with open(hidden_variables_filename, 'r') as file:
            data = json.load(file)
        
        self.set_hidden_variables(data)
        
    def _generate_edgelist(self) -> None:
        """
        Generate the edgelist of the network using the hidden variables.
        """
        if self.verbose:
            self.logger.info(f"Generating edgelist of length {self.nb_vertices}")
            self.logger.info(f"theta: {self.theta}")
        if self.BETA == -10:
            raise ValueError("The value of parameter beta must be provided.")
        
        if self.MU == -10:
            average_kappa = sum((in_k + out_k) / 2.0 for in_k, out_k in zip(self.in_Kappa, self.out_Kappa)) / self.nb_vertices
            self.MU = self.BETA * math.sin(self.PI / self.BETA) / (2.0 * self.PI * average_kappa)
        
        if self.NU == -10:
            raise ValueError("The value of parameter nu must be provided.")
        
        prefactor = self.nb_vertices / (2 * self.PI * self.MU)
        for v1 in range(self.nb_vertices):
            kout1 = self.out_Kappa[v1]
            k_in1 = self.in_Kappa[v1]
            theta1 = self.theta[v1]
            for v2 in range(v1 + 1, self.nb_vertices):
                dtheta = self.PI - abs(self.PI - abs(theta1 - self.theta[v2]))
                koutkin = kout1 * self.in_Kappa[v2]
                p12 = 1 / (1 + ((prefactor * dtheta) / koutkin) ** self.BETA) if koutkin > self.NUMERICAL_ZERO else 0
                koutkin = self.out_Kappa[v2] * k_in1
                p21 = 1 / (1 + ((prefactor * dtheta) / koutkin) ** self.BETA) if koutkin > self.NUMERICAL_ZERO else 0
                
                if self.NU > 0:
                    p11 = ((1 - self.NU) * max(p12, p21) + self.NU) * min(p12, p21)
                else:
                    p11 = (1 + self.NU) * p12 * p21 + self.NU * (1 - p12 - p21) if p12 + p21 >= 1 else (1 + self.NU) * p12 * p21
                
                r = random.random()
                if r < p11:
                    self.network.add_edge(v1, v2)
                    self.network.add_edge(v2, v1)
                elif r < p21:
                    self.network.add_edge(v2, v1)
                elif r < (p21 + p12 - p11):
                    self.network.add_edge(v1, v2)
        
    def calculate_metrics(self) -> Tuple[float, float, List[int], List[int]]:
        """
        Calculate the reciprocity, clustering coefficient, in-degree sequence, and out-degree sequence of the network.

        Returns:
            Tuple[float, float, List[int], List[int]]: The reciprocity, clustering coefficient, in-degree sequence, and out-degree sequence.
        """
        if self.verbose:
            self.logger.info("Calculating metrics")
        self.reciprocity = nx.reciprocity(self.network)
        self.clustering = nx.average_clustering(self.network.to_undirected())
        self.in_degree_sequence = [d for n, d in self.network.in_degree()]
        self.out_degree_sequence = [d for n, d in self.network.out_degree()]
        return self.reciprocity, self.clustering, self.in_degree_sequence, self.out_degree_sequence
    
    def get_output(self) -> dict:
        """
        Get the output of the generated network.
        
        Returns:
            dict: The output data of the generated network.
        
        Output contains the following:
            parameters (dict): The parameters of the model.
                beta (float): The beta parameter.
                mu (float): The mu parameter.
                nu (float): The nu parameter.
                N (int): The number of vertices.
                R (float): The R parameter.
                seed (int): The random seed.
                output_rootname (str): The network name of the hidden variables file.
            adjacency_list (List[List[int]]): The adjacency list of the network.
            network_data (dict): The data of the network.
                reciprocity (float): The network reciprocity.
                clustering (float): The network clustering coefficient.
                in_degree_sequence (List[int]): The in-degree sequence.
                out_degree_sequence (List[int]): The out-degree sequence.
        """
        if self.verbose:
            self.logger.info("Getting data")  
        adjacency_list = [list(self.network.neighbors(node)) for node in self.network.nodes()]
        self.calculate_metrics()
        output = {
            "parameters": {
                "beta": self.BETA,
                "mu": self.MU,
                "nu": self.NU,
                "N": self.nb_vertices,
                "R": self.R,
                "seed": self.SEED,
                "output_rootname": self.output_rootname
            },
            "adjacency_list": adjacency_list,
            "network_data": {
                "reciprocity": self.reciprocity,
                "clustering": self.clustering,
                "in_degree_sequence": self.in_degree_sequence,
                "out_degree_sequence": self.out_degree_sequence
            }
        }
        return output
    
    def _save_data(self) -> None:
        """
        Saves output data to outputs/<output_rootname>/generation_data.json.
        
        Output contains the following:
            parameters (dict): The parameters of the model.
                beta (float): The beta parameter.
                mu (float): The mu parameter.
                nu (float): The nu parameter.
                N (int): The number of vertices.
                R (float): The R parameter.
                seed (int): The random seed.
                output_rootname (str): The network of the hidden variables file.
            adjacency_list (List[List[int]]): The adjacency list of the network.
            network_data (dict): The data of the network.
                reciprocity (float): The network reciprocity.
                clustering (float): The network clustering coefficient.
                in_degree_sequence (List[int]): The in-degree sequence.
                out_degree_sequence (List[int]): The out-degree sequence.
                
        """
        if self.verbose:
            self.logger.info(f"Saving data to outputs/{self.output_rootname}/generation_data.json")
        output = self.get_output()
        
        if not os.path.exists("outputs"):
            os.makedirs("outputs")  
        if not os.path.exists(f"outputs/{self.output_rootname}"):
            os.makedirs(f"outputs/{self.output_rootname}")
        with open(f"outputs/{self.output_rootname}/generation_data.json", 'w') as f:
            json.dump(output, f, indent=4)
            
    def _save_coordinates(self) -> None:
        """
        Save the coordinates of the nodes to outputs/<output_rootname>/coordinates.csv.
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
                f.write(f"{i},{self.in_Kappa[i]},{self.out_Kappa[i]},{self.theta[i]}\n")
                
    def set_params(self, **kwargs) -> None:
        """
        Set the DirectedS1Generator parameters.
        
        Args:
            **kwargs (dict): The parameters to set.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def generate_from_file(self,
                           hidden_variables_filename: str = "", 
                           output_rootname: str = "",
                           theta: List[int] = None,
                           save_data: bool = True,
                           save_coordinates: bool = True) -> None:
        if output_rootname == "": 
            #if output_rootname is not provided, use the name of subdirectory containing the hidden variables file
            self.output_rootname = hidden_variables_filename.split("/")[-2].split(".")[0]
        else:
            self.output_rootname = output_rootname
        self._load_hidden_variables(hidden_variables_filename = hidden_variables_filename)
        self._generate(theta, save_data, save_coordinates)
        
    def generate_from_inference_data(self,
                                   hidden_variables_data: dict = None,
                                   output_rootname: str = "",
                                   theta: List[int] = None,
                                   save_data: bool = True,
                                   save_coordinates: bool = True) -> None:
        if output_rootname == "":
            raise ValueError("Output rootname must be provided.")
        self.output_rootname = output_rootname
        self.set_hidden_variables(hidden_variables_data)
        self._generate(theta, save_data, save_coordinates)
        
    def _generate(self, 
                  theta: List[int] = None,
                  save_data: bool = True,
                  save_coordinates: bool = True) -> None:
        """
        Generate a directed network using the hidden variables provided in the hidden_variables_filename, and save the data.
        
        Args:
            hidden_variables_data (dict): The hidden variables
            hidden_variables_filename (str): The filename of the hidden variables json file.
            output_rootname (str): The rootname of the output files. If not provided, the name of the subdirectory containing the hidden variables file is used.
            theta (List[int], optional): The theta values for the network. Defaults to None.
            save_data (bool, optional): Whether to save the data. Defaults to True.
            save_coordinates (bool, optional): Whether to save the coordinates of the nodes. Defaults to True.
        """
            
        self._generate_edgelist()
        if save_data:
            self._save_data()
        if save_coordinates:
            self._save_coordinates()
        if self.verbose:
            self.logger.info(f"Finished generating network at {self.get_time()}")
    
    def modify_log_file_path(self, log_file_path: str) -> None:
        """
        Modify the log file path for the logger.

        Args:
            log_file_path (str): The new log file path.
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
        """
        Get the current UTC time as a string.

        Returns:
            str: The current UTC time.
        """
        return datetime.utcnow().strftime("%Y/%m/%d %H:%M UTC")

    def plot_degree_sequence(self, degree_sequence: List[int]) -> None:
        """
        Plot the degree sequence distribution and save it as a PNG file.

        Args:
            degree_sequence (List[int]): The degree sequence to plot.
        """
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
