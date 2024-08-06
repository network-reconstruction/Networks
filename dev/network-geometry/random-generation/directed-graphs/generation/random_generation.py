import json
import math
import random
import time
from datetime import datetime
import sys
import networkx as nx
import matplotlib.pyplot as plt

class GeneratingDirectedS1:
    def __init__(self):
        self.CUSTOM_OUTPUT_ROOTNAME = False
        self.NAME_PROVIDED = False
        self.NATIVE_INPUT_FILE = False
        self.THETA_PROVIDED = False
        self.SAVE_COORDINATES = False
        
        self.SEED = int(time.time())
        random.seed(self.SEED)
        
        self.BETA = -10
        self.MU = -10
        self.NU = -10
        self.R = -10
        self.OUTPUT_ROOTNAME = "default_output_rootname"
        self.HIDDEN_VARIABLES_FILENAME = ""
        
        self.PI = math.pi
        self.NUMERICAL_ZERO = 1e-5
        
        self.Num2Name = []
        
        self.nb_vertices = 0
        self.in_Kappa = []
        self.outKappa = []
        self.theta = []
        self.network = nx.DiGraph()
        
    def load_hidden_variables(self):
        with open(self.HIDDEN_VARIABLES_FILENAME, 'r') as file:
            data = json.load(file)
        
        if self.NATIVE_INPUT_FILE:
            self.BETA = data.get("beta", self.BETA)
            self.MU = data.get("mu", self.MU)
            self.NU = data.get("nu", self.NU)
            self.R = data.get("R", self.R)
        
        inferred_kappas = data.get("inferred_kappas", [])
        self.nb_vertices = len(inferred_kappas)
        
        self.in_Kappa = [kappa["kappa_in"] for kappa in inferred_kappas]
        self.outKappa = [kappa["kappa_out"] for kappa in inferred_kappas]
        self.Num2Name = [f"node_{i}" for i in range(self.nb_vertices)]
        
        if self.THETA_PROVIDED:
            self.theta = [kappa["theta"] for kappa in inferred_kappas]
        else:
            self.theta = [2 * self.PI * random.random() for _ in range(self.nb_vertices)]
        
    def generate_edgelist(self, width=15):
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
        
    def calculate_metrics(self):
        reciprocity = nx.reciprocity(self.network)
        clustering = nx.average_clustering(self.network.to_undirected())
        in_degree_sequence = [d for n, d in self.network.in_degree()]
        out_degree_sequence = [d for n, d in self.network.out_degree()]
        return reciprocity, clustering, in_degree_sequence, out_degree_sequence
    
    def save_adjacency_list(self):
        adjacency_list = [list(self.network.neighbors(node)) for node in self.network.nodes()]
        
        data = {
            "parameters": {
                "beta": self.BETA,
                "mu": self.MU,
                "nu": self.NU,
                "N": self.nb_vertices,
                "R": self.R,
                "seed": self.SEED,
                "hidden_variables_file": self.HIDDEN_VARIABLES_FILENAME
            },
            "adjacency_list": adjacency_list
        }
        
        with open(f"{self.OUTPUT_ROOTNAME}_adjacency_list.json", 'w') as f:
            json.dump(data, f, indent=4)
    
    def get_time(self):
        return datetime.utcnow().strftime("%Y/%m/%d %H:%M UTC")

    def plot_degree_sequence(self, degree_sequence):
        plt.figure()
        plt.hist(degree_sequence, bins='auto')
        plt.title("Degree Sequence Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()


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
    gen.generate_edgelist()
    gen.save_adjacency_list()
    print(f"Finished at {gen.get_time()}")
