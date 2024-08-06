import jax
import jax.numpy as jnp
import json
import time
from datetime import datetime

class GeneratingDirectedS1JAX:
    def __init__(self):
        self.CUSTOM_OUTPUT_ROOTNAME = False
        self.NAME_PROVIDED = False
        self.NATIVE_INPUT_FILE = False
        self.THETA_PROVIDED = False
        self.SAVE_COORDINATES = False
        
        self.SEED = int(time.time())
        self.key = jax.random.PRNGKey(self.SEED)
        
        self.BETA = -10
        self.MU = -10
        self.NU = -10
        self.OUTPUT_ROOTNAME = "default_output_rootname"
        self.HIDDEN_VARIABLES_FILENAME = ""
        
        self.PI = jnp.pi
        self.NUMERICAL_ZERO = 1e-5
        
        self.Num2Name = []
        
        self.nb_vertices = 0
        self.in_Kappa = None
        self.outKappa = None
        self.theta = None
        self.adjacency_list = {}
        
    def load_hidden_variables(self):
        with open(self.HIDDEN_VARIABLES_FILENAME, 'r') as file:
            data = json.load(file)
        
        if self.NATIVE_INPUT_FILE:
            self.BETA = data.get("beta", self.BETA)
            self.MU = data.get("mu", self.MU)
            self.NU = data.get("nu", self.NU)
        
        self.nb_vertices = len(data.get("vertices", []))
        self.in_Kappa = jnp.array([vertex["inKappa"] for vertex in data["vertices"]])
        self.outKappa = jnp.array([vertex["outKappa"] for vertex in data["vertices"]])
        self.Num2Name = [vertex["name"] for vertex in data["vertices"]]
        
        if self.THETA_PROVIDED:
            self.theta = jnp.array([vertex["theta"] for vertex in data["vertices"]])
        else:
            self.key, subkey = jax.random.split(self.key)
            self.theta = 2 * self.PI * jax.random.uniform(subkey, (self.nb_vertices,))
        
    def generate_edgelist(self, width=15):
        if self.BETA == -10:
            raise ValueError("The value of parameter beta must be provided.")
        
        if self.MU == -10:
            average_kappa = jnp.mean((self.in_Kappa + self.outKappa) / 2.0)
            self.MU = self.BETA * jnp.sin(self.PI / self.BETA) / (2.0 * self.PI * average_kappa)
        
        if self.NU == -10:
            raise ValueError("The value of parameter nu must be provided.")
        
        prefactor = self.nb_vertices / (2 * self.PI * self.MU)
        indices = jnp.arange(self.nb_vertices)
        v1, v2 = jnp.meshgrid(indices, indices, indexing='ij')
        
        dtheta = self.PI - jnp.abs(self.PI - jnp.abs(self.theta[v1] - self.theta[v2]))
        koutkin1 = self.outKappa[v1] * self.in_Kappa[v2]
        p12 = jnp.where(koutkin1 > self.NUMERICAL_ZERO, 
                        1 / (1 + (prefactor * dtheta / koutkin1) ** self.BETA), 0)
        koutkin2 = self.outKappa[v2] * self.in_Kappa[v1]
        p21 = jnp.where(koutkin2 > self.NUMERICAL_ZERO, 
                        1 / (1 + (prefactor * dtheta / koutkin2) ** self.BETA), 0)
        
        if self.NU > 0:
            p11 = jnp.where(p12 < p21, ((1 - self.NU) * p21 + self.NU) * p12, ((1 - self.NU) * p12 + self.NU) * p21)
        else:
            p11 = jnp.where(p12 + p21 < 1, (1 + self.NU) * p12 * p21, (1 + self.NU) * p12 * p21 + self.NU * (1 - p12 - p21))
        
        self.key, subkey = jax.random.split(self.key)
        r = jax.random.uniform(subkey, p11.shape)
        
        for i in range(self.nb_vertices):
            for j in range(i + 1, self.nb_vertices):
                if r[i, j] < p11[i, j]:
                    self.add_edge(i, j)
                    self.add_edge(j, i)
                elif r[i, j] < p21[i, j]:
                    self.add_edge(j, i)
                elif r[i, j] < (p21[i, j] + p12[i, j] - p11[i, j]):
                    self.add_edge(i, j)
        
        self.save_adjacency_list()
    
    def add_edge(self, v1, v2):
        if self.Num2Name[v1] not in self.adjacency_list:
            self.adjacency_list[self.Num2Name[v1]] = []
        self.adjacency_list[self.Num2Name[v1]].append(self.Num2Name[v2])
    
    def save_adjacency_list(self):
        data = {
            "parameters": {
                "beta": self.BETA,
                "mu": self.MU,
                "nu": self.NU,
                "N": self.nb_vertices,
                "R": self.nb_vertices / (2 * self.PI),
                "seed": self.SEED,
                "hidden_variables_file": self.HIDDEN_VARIABLES_FILENAME
            },
            "adjacency_list": self.adjacency_list
        }
        
        with open(f"{self.OUTPUT_ROOTNAME}_adjacency_list.json", 'w') as f:
            json.dump(data, f, indent=4)
    
    def get_time(self):
        return datetime.utcnow().strftime("%Y/%m/%d %H:%M UTC")
