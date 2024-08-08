"""
Model from Supplementary Information of the paper: Geometric description of clustering in directed networks: Antoine Allard et al.
"""

import jax.numpy as jnp
import jax.random as random
from scipy.special import hyp2f1
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import time
import sys
from sortedcontainers import SortedDict
import json
import logging
import os


def lower_bound(d: SortedDict, key: Any) -> Optional[Tuple]:
    """
    Compute the lower bound of a key in a sorted dictionary.

    Parameters:
        d (SortedDict): Sorted dictionary.
        key (Any): Key to search.

    Returns:
        Optional[Tuple]: Lower bound of the key, or None if the key is not found.
    """
    index = d.bisect_left(key)
    if index < len(d):
        return d.peekitem(index)
    else:
        return None


def hyp2f1a(beta: float, z: float) -> float:
    """Compute hypergeometric function for parameters beta and z."""
    return hyp2f1(1.0, 1.0 / beta, 1.0 + (1.0 / beta), z).real


def hyp2f1b(beta: float, z: float) -> float:
    """Compute hypergeometric function for parameters beta and z."""
    return hyp2f1(1.0, 2.0 / beta, 1.0 + (2.0 / beta), z).real


def hyp2f1c(beta: float, z: float) -> float:
    """Compute hypergeometric function for parameters beta and z (S75b)."""
    return hyp2f1(2.0, 1.0 / beta, 1.0 + (1.0 / beta), z).real


class DirectedS1Fitter:
    """
    A class used to represent the parameter inference model for directed hyperbolic networks.

    Attributes:
        PI (float): The value of Pi.
        CUSTOM_BETA (bool): Flag indicating if beta is custom-set.
        CUSTOM_MU (bool): Flag indicating if mu is custom-set.
        CUSTOM_NU (bool): Flag indicating if nu is custom-set.
        BETA_ABS_MAX (float): Maximum absolute value of beta.
        BETA_ABS_MIN (float): Minimum absolute value of beta.
        deg_seq_filename (str): Filename for degree sequence.
        EXP_CLUST_NB_INTEGRATION_MC_STEPS (int): Number of Monte Carlo steps for clustering integration.
        KAPPA_MAX_NB_ITER_CONV (int): Maximum iterations for kappa convergence.
        NUMERICAL_CONVERGENCE_THRESHOLD_1 (float): Numerical convergence threshold 1.
        NUMERICAL_CONVERGENCE_THRESHOLD_2 (float): Numerical convergence threshold 2.
        NUMERICAL_ZERO (float): Small value considered as zero.
        output_rootname (str): Rootname for output files.
        SEED (int): Random seed.
        nb_vertices (int): Number of vertices.
        nb_vertices_undir_degree_gt_one (int): Number of vertices with undirected degree greater than one.
        nb_edges (int): Number of edges.
        nb_reciprocal_edges (int): Number of reciprocal edges.
        nb_triangles (int): Number of triangles.
        reciprocity (float): Reciprocity of the network.
        average_undir_clustering (float): Average undirected clustering.
        beta (float): Parameter beta.
        mu (float): Parameter mu.
        nu (float): Parameter nu.
        R (float): Radius of the hyperbolic space.
        random_ensemble_average_degree (float): Average degree of the random ensemble.
        random_ensemble_average_clustering (float): Average clustering of the random ensemble.
        random_ensemble_reciprocity (float): Reciprocity of the random ensemble.
        random_ensemble_expected_degree_per_degree_class (List[float]): Expected degree per degree class.
        random_ensemble_kappa_per_degree_class (List[float]): Kappa values per degree class.
        cumul_prob_kgkp (Dict): Cumulative probability distribution for Monte Carlo integration.
        engine (random.PRNGKey): Random number generator key.
        out_degree_sequence (List[float]): Sequence of out-degrees.
        in_degree_sequence (List[float]): Sequence of in-degrees.
        degree (Tuple[List[float], List[float]]): Tuple of in-degree and out-degree sequences.
        degree_histogram (List[Dict[int, int]]): Histograms of in-degrees and out-degrees.
        degree_class (Dict[int, Dict[int, int]]): Classification of nodes by their in-degree and out-degree.

    """

    PI = jnp.pi

    def __init__(self, 
                 seed: int = 0, 
                 verbose: bool = False, 
                 KAPPA_MAX_NB_ITER_CONV: int = 100, 
                 EXP_CLUST_NB_INTEGRATION_MC_STEPS: int = 50, 
                 NUMERICAL_CONVERGENCE_THRESHOLD_1: float = 1e-2, 
                 NUMERICAL_CONVERGENCE_THRESHOLD_2: float = 1e-2,
                 log_file_path: str = "logs/DirectedS1Fitter/output.log"): 
        """
        Initialize the parameter inference model for directed hyperbolic networks.

        Parameters:
            seed (int): Seed for random number generation (default = 0).
            verbose (bool): Whether to print verbose output (default = False).
            KAPPA_MAX_NB_ITER_CONV (int): Maximum number of iterations for convergence of kappa (default = 100).
            EXP_CLUST_NB_INTEGRATION_MC_STEPS (int): Number of Monte Carlo steps for integration of expected clustering (default = 50).
            NUMERICAL_CONVERGENCE_THRESHOLD_1 (float): Threshold for numerical convergence of kappa (default = 1e-2).
            NUMERICAL_CONVERGENCE_THRESHOLD_2 (float): Threshold for numerical convergence of clustering (default = 1e-2).
            log_file_path (str): Path to the log file (default = "logs/DirectedS1Fitter/output.log").
        """
        
        # for loop create folders in log file path if they don't exist
        self.verbose = verbose
        # Setup logging
        # -----------------------------------------------------
        self.logger = self._setup_logging(log_file_path)
        # -----------------------------------------------------

        self.CUSTOM_BETA = False
        self.CUSTOM_MU = False
        self.CUSTOM_NU = False
        self.BETA_ABS_MAX = 25
        self.BETA_ABS_MIN = 1.01
        self.deg_seq_filename = ""
        self.EXP_CLUST_NB_INTEGRATION_MC_STEPS = EXP_CLUST_NB_INTEGRATION_MC_STEPS
        self.KAPPA_MAX_NB_ITER_CONV = KAPPA_MAX_NB_ITER_CONV
        self.NUMERICAL_CONVERGENCE_THRESHOLD_1 = NUMERICAL_CONVERGENCE_THRESHOLD_1
        self.NUMERICAL_CONVERGENCE_THRESHOLD_2 = NUMERICAL_CONVERGENCE_THRESHOLD_2
        self.NUMERICAL_ZERO = 1e-5
        self.output_rootname = ""
        self.SEED = seed

        self.nb_vertices = 0
        self.nb_vertices_undir_degree_gt_one = 0
        self.nb_edges = 0

        self.nb_reciprocal_edges = 0
        self.nb_triangles = 0
        self.reciprocity = 0
        self.average_undir_clustering = 0

        self.beta = 0
        self.mu = 0
        self.nu = 0
        self.R = 0

        self.random_ensemble_average_degree = 0
        self.random_ensemble_average_clustering = 0
        self.random_ensemble_reciprocity = 0
        self.random_ensemble_expected_degree_per_degree_class = []
        self.random_ensemble_kappa_per_degree_class = []

        self.cumul_prob_kgkp = {}
        self.engine = random.PRNGKey(self.SEED)

        self.out_degree_sequence, self.in_degree_sequence = [], []
        self.verbose = verbose
        
    def _setup_logging(self, log_file_path: str) -> logging.Logger:
        """
        Setup logging with the given log file path.

        Parameters:
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
    
    def set_params(self, **kwargs) -> None:
        """
        Set the DirectedS1Fitter parameters.

        Parameters:
            **kwargs (dict): The parameters to set.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _compute_degree_histogram(self, degrees: List[float]) -> Dict[int, int]:
        """
        Compute the degree histogram.

        Parameters:
            degrees (List[float]): List of degrees.

        Returns:
            Dict[int, int]: Degree histogram.
        """
        histogram = {}
        for degree in degrees:
            degree = int(degree)
            if degree not in histogram:
                histogram[degree] = 0
            histogram[degree] += 1
        return histogram

    def _classify_degrees(self) -> Dict[int, Dict[int, int]]:
        """
        Classify nodes based on their in-degrees and out-degrees.

        Returns:
            Dict[int, Dict[int, int]]: Degree class, outer key is in-degree, inner key is out-degree, inner value is count.
        """
        degree_class = {}
        for k_in, k_out in zip(self.in_degree_sequence, self.out_degree_sequence):
            k_in, k_out = int(k_in), int(k_out)
            if k_in not in degree_class:
                degree_class[k_in] = {}
            if k_out not in degree_class[k_in]:
                degree_class[k_in][k_out] = 0
            degree_class[k_in][k_out] += 1
        return degree_class

    def directed_connection_probability(self, z: float, koutkin: float) -> float:
        """
        Compute the directed connection probability using hypergeometric function.

        Parameters:
            z (float): Angle between two nodes.
            koutkin (float): Product of out-degree and in-degree.

        Returns:
            float: Directed connection probability.
        """
        return (z / self.PI) * hyp2f1a(self.beta, -((self.R * z) / (self.mu * koutkin)) ** self.beta)

    def undirected_connection_probability(self, z: float, kout1kin2: float, kout2kin1: float) -> float:
        """
        Compute the undirected connection probability.

        Parameters:
            z (float): Angle between two nodes.
            kout1kin2 (float): Product of out-degree of node 1 and in-degree of node 2.
            kout2kin1 (float): Product of out-degree of node 2 and in-degree of node 1.

        Returns:
            float: Undirected connection probability.
        """
        p12 = self.directed_connection_probability(z, kout1kin2)
        p21 = self.directed_connection_probability(z, kout2kin1)

        conn_prob = 0
        if jnp.abs(kout1kin2 - kout2kin1) < self.NUMERICAL_CONVERGENCE_THRESHOLD_2:
            conn_prob -= (1 - jnp.abs(self.nu)) * hyp2f1c(self.beta, -((self.R * self.PI) / (self.mu * kout1kin2)) ** self.beta)
        else:
            conn_prob -= ((1 - jnp.abs(self.nu)) / (1 - (kout1kin2 / kout2kin1) ** self.beta)) * (p12 - (kout1kin2 / kout2kin1) ** self.beta * p21)
        conn_prob += p12
        conn_prob += p21
        if self.nu > 0:
            if kout1kin2 < kout2kin1:
                conn_prob -= self.nu * p12
            else:
                conn_prob -= self.nu * p21
        elif self.nu < 0:
            z_max = self.find_minimal_angle_by_bisection(kout1kin2, kout2kin1)
            if z_max < z:
                p12 = self.directed_connection_probability(z_max, kout1kin2)
                p21 = self.directed_connection_probability(z_max, kout2kin1)
                conn_prob -= self.nu * ((z_max / self.PI) - p12 - p21)
            else:
                conn_prob -= self.nu * ((z / self.PI) - p12 - p21)

        return conn_prob

    def find_minimal_angle_by_bisection(self, kout1kin2: float, kout2kin1: float) -> float:
        """
        Find the minimal angle by bisection.

        Parameters:
            kout1kin2 (float): Product of out-degree of node 1 and in-degree of node 2.
            kout2kin1 (float): Product of out-degree of node 2 and in-degree of node 1.

        Returns:
            float: Minimal angle by bisection.
        """
        z_min = 0
        z_max = self.PI
        while (z_max - z_min) > self.NUMERICAL_CONVERGENCE_THRESHOLD_2:
            z_mid = (z_min + z_max) / 2
            pz_min = self.directed_connection_probability(z_min, kout1kin2) + self.directed_connection_probability(z_min, kout2kin1)
            pz_mid = self.directed_connection_probability(z_mid, kout1kin2) + self.directed_connection_probability(z_mid, kout2kin1)
            pz_max = self.directed_connection_probability(z_max, kout1kin2) + self.directed_connection_probability(z_max, kout2kin1)

            if (pz_min * pz_mid) > 0:
                z_min = z_mid
            else:
                z_max = z_mid
        return (z_min + z_max) / 2

    def build_cumul_dist_for_mc_integration(self) -> None:
        """
        Build cumulative distribution for Monte Carlo integration.
        """
        if self.verbose:
            self.logger.info("Building cumulative distribution for Monte Carlo integration ...")
            self.logger.info(f"self.degree_class: {self.degree_class}")
            start_time = time.time()
        for in_deg, out_degs in self.degree_class.items(): 
            self.cumul_prob_kgkp[in_deg] = {}
            for out_deg in out_degs:
                nkkp = {(i, j): 0 for i in self.degree_class for j in self.degree_class[i]}
                tmp_cumul = 0
                k_in1 = self.random_ensemble_kappa_per_degree_class[0][in_deg] 
                kout1 = self.random_ensemble_kappa_per_degree_class[1][out_deg]
                for in_deg2, out_degs2 in self.degree_class.items():
                    for out_deg2 in out_degs2:
                        k_in2 = self.random_ensemble_kappa_per_degree_class[0][in_deg2]
                        kout2 = self.random_ensemble_kappa_per_degree_class[1][out_deg2]
                        tmp_val = self.undirected_connection_probability(self.PI, kout1 * k_in2, kout2 * k_in1)
                        
                        if in_deg == in_deg2 and out_deg == out_deg2:
                            nkkp[(in_deg2, out_deg2)] = (out_degs2[out_deg2] - 1) * tmp_val 
                        else:
                            nkkp[(in_deg2, out_deg2)] = out_degs2[out_deg2] * tmp_val
                        tmp_cumul += nkkp[(in_deg2, out_deg2)]
                
                for key in nkkp:
                    nkkp[key] /= tmp_cumul 
                    
                tmp_cumul = 0
                for key in nkkp:
                    tmp_val = nkkp[key]
                    if tmp_val > self.NUMERICAL_ZERO:
                        tmp_cumul += tmp_val 
                        if out_deg not in self.cumul_prob_kgkp[in_deg]:
                            self.cumul_prob_kgkp[in_deg][out_deg] = SortedDict()
                        self.cumul_prob_kgkp[in_deg][out_deg][int(tmp_cumul)] = key
        if self.verbose:
            self.logger.info("Finished building cumulative distribution for Monte Carlo integration ...")
            self.logger.info(f"Runtime: {time.time() - start_time}")
            
    def compute_random_ensemble_average_degree(self) -> None:
        """
        Compute the random ensemble average degree.
        """
        if self.verbose:
            self.logger.info("Computing random ensemble average degree ...")
        directions = [0, 1]
        random_ensemble_average_degree = 0
        for direction in directions:
            for el in self.degree_histogram[direction].items():
                random_ensemble_average_degree += el[1] * self.random_ensemble_expected_degree_per_degree_class[direction][el[0]]
        self.random_ensemble_average_degree = random_ensemble_average_degree / (self.nb_vertices * 2)

    def compute_random_ensemble_clustering(self) -> None:
        """
        Compute the random ensemble average clustering.
        """
        if self.verbose:
            self.logger.info("Computing random ensemble clustering ...")
        random_ensemble_average_clustering = 0
        for in_deg, out_degs in self.degree_class.items():
            for out_deg in out_degs:
                if in_deg + out_deg > 1:
                    tmp_val = self.compute_random_ensemble_clustering_for_degree_class(in_deg, out_deg)
                    random_ensemble_average_clustering += tmp_val * out_degs[out_deg]
        self.random_ensemble_average_clustering = random_ensemble_average_clustering / self.nb_vertices_undir_degree_gt_one

    def compute_random_ensemble_clustering_for_degree_class(self, in_deg: int, out_deg: int) -> float:
        """
        Compute the random ensemble clustering for a given degree class.

        Parameters:
            in_deg (int): In-degree.
            out_deg (int): Out-degree.

        Returns:
            float: Random ensemble clustering for the given degree class.
        """
        if self.verbose:
            self.logger.info(f"Computing random ensemble clustering for degree class: {in_deg}, {out_deg} ...")
            self.logger.info(f"Cumul_prob_kgkp: {self.cumul_prob_kgkp}")
            start_time = time.time()

        tmp_cumul = 0
        k_in1 = self.random_ensemble_kappa_per_degree_class[0][in_deg]
        kout1 = self.random_ensemble_kappa_per_degree_class[1][out_deg]

        M = self.EXP_CLUST_NB_INTEGRATION_MC_STEPS

        for i in range(M):
            cumul_prob_dict = self.cumul_prob_kgkp[int(in_deg)].get(int(out_deg), SortedDict())
            random_val = random.uniform(self.engine)
            lower_bound_key = cumul_prob_dict.bisect_left(random_val)
            lower_bound_key = min(lower_bound_key, len(cumul_prob_dict) - 1)
            p2_key = list(cumul_prob_dict.keys())[lower_bound_key]
            p2 = cumul_prob_dict[p2_key]

            k_in2 = self.random_ensemble_kappa_per_degree_class[0][p2[0]]
            kout2 = self.random_ensemble_kappa_per_degree_class[1][p2[1]]
            p12 = self.undirected_connection_probability(self.PI, kout1 * k_in2, kout2 * k_in1)

            pc = random.uniform(self.engine)
            zmin = 0
            zmax = self.PI
            while (zmax - zmin) > self.NUMERICAL_CONVERGENCE_THRESHOLD_2:
                z = (zmax + zmin) / 2
                pz = self.undirected_connection_probability(z, kout1 * k_in2, kout2 * k_in1) / p12
                if pz > pc:
                    zmax = z
                else:
                    zmin = z
            z12 = (zmax + zmin) / 2

            lower_bound_key = cumul_prob_dict.bisect_left(random_val)
            lower_bound_key = min(lower_bound_key, len(cumul_prob_dict) - 1)
            p3_key = list(cumul_prob_dict.keys())[lower_bound_key]
            p3 = cumul_prob_dict[p3_key]

            k_in3 = self.random_ensemble_kappa_per_degree_class[0][p3[0]]
            kout3 = self.random_ensemble_kappa_per_degree_class[1][p3[1]]
            p13 = self.undirected_connection_probability(self.PI, kout1 * k_in3, kout3 * k_in1)

            pc = random.uniform(self.engine)
            zmin = 0
            zmax = self.PI
            while (zmax - zmin) > self.NUMERICAL_CONVERGENCE_THRESHOLD_2:
                z = (zmax + zmin) / 2
                pz = self.undirected_connection_probability(z, kout1 * k_in3, kout3 * k_in1) / p13
                if pz > pc:
                    zmax = z
                else:
                    zmin = z
            z13 = (zmax + zmin) / 2

            if random.uniform(self.engine) < 0.5:
                da = jnp.abs(z12 + z13)
            else:
                da = jnp.abs(z12 - z13)
            da = min(da, (2.0 * self.PI) - da)

            tmp_val = 0
            if da < self.NUMERICAL_ZERO:
                tmp_val += 1
            else:
                p23 = 1.0 / (1.0 + ((da * self.R) / (self.mu * kout2 * k_in3)) ** self.beta)
                p32 = 1.0 / (1.0 + ((da * self.R) / (self.mu * kout3 * k_in2)) ** self.beta)
                tmp_val += p23 + p32
                tmp_val -= (1 - jnp.abs(self.nu)) * p23 * p32
                if self.nu > 0:
                    if p23 < p32:
                        tmp_val -= self.nu * p23
                    else:
                        tmp_val -= self.nu * p32
                if self.nu < 0:
                    if (p23 + p32) > 1:
                        tmp_val -= self.nu * (1 - p23 - p32)
            tmp_cumul += tmp_val

        if self.verbose:
            self.logger.info(f"Time: {time.time() - start_time}")
        return tmp_cumul / M

    def infer_kappas(self) -> None:
        """
        Infer kappa values.
        """
        if self.verbose:
            self.logger.info("Inferring kappas ...")
            self.logger.info(f"degree_histogram: {self.degree_histogram}")
        self.random_ensemble_kappa_per_degree_class = [{} for _ in range(2)]
        self.random_ensemble_expected_degree_per_degree_class = [{} for _ in range(2)]
        directions = [0, 1]
        for direction in directions:
            for el in self.degree_histogram[direction].items():
                self.random_ensemble_kappa_per_degree_class[direction][el[0]] = el[0] + 0.001

        keep_going = True
        cnt = 0
        start_time = time.time()
        if self.verbose:
            self.logger.info(f"KAPPA_MAX_NB_ITER_CONV: {self.KAPPA_MAX_NB_ITER_CONV}")

        while keep_going and cnt < self.KAPPA_MAX_NB_ITER_CONV:
            if self.verbose:
                self.logger.info(f"Iteration: {cnt}, Previous Iteration time: {time.time() - start_time}")
                start_time = time.time()
            for direction in directions:
                for el in self.degree_histogram[direction].items():
                    self.random_ensemble_expected_degree_per_degree_class[direction][el[0]] = 0

            for i, (in_deg, count_in) in enumerate(self.degree_histogram[0].items()):
                for j, (out_deg, count_out) in enumerate(self.degree_histogram[1].items()):
                    prob_conn = self.directed_connection_probability(
                        self.PI,
                        self.random_ensemble_kappa_per_degree_class[1][out_deg] * self.random_ensemble_kappa_per_degree_class[0][in_deg]
                    )
                    self.random_ensemble_expected_degree_per_degree_class[0][in_deg] += prob_conn * count_in
                    self.random_ensemble_expected_degree_per_degree_class[1][out_deg] += prob_conn * count_out
            error = jnp.inf
            keep_going = False
            for direction in directions:
                for el in self.degree_histogram[direction].items():
                    error = jnp.abs(self.random_ensemble_expected_degree_per_degree_class[direction][el[0]] - el[0])
                    if error > self.NUMERICAL_CONVERGENCE_THRESHOLD_1:
                        keep_going = True
                        break
            if self.verbose:
                self.logger.info(f"Error: {error}, NUMERICAL_CONVERGENCE_THRESHOLD_1: {self.NUMERICAL_CONVERGENCE_THRESHOLD_1}")
                
            if keep_going:
                for direction in directions:
                    for el in self.degree_histogram[direction].items():
                        self.random_ensemble_kappa_per_degree_class[direction][el[0]] += (el[0] - self.random_ensemble_expected_degree_per_degree_class[direction][el[0]]) * random.uniform(self.engine)
                        if self.random_ensemble_kappa_per_degree_class[direction][el[0]] < 0:
                            self.random_ensemble_kappa_per_degree_class[direction][el[0]] = jnp.abs(self.random_ensemble_kappa_per_degree_class[direction][el[0]])
                cnt += 1

            if cnt >= self.KAPPA_MAX_NB_ITER_CONV:
                self.logger.info("WARNING: Maximum number of iterations reached before convergence. This limit can be adjusted through the parameter KAPPA_MAX_NB_ITER_CONV.")

    def infer_nu(self) -> None:
        """
        Infer the parameter nu.
        """
        if self.verbose:
            self.logger.info("Inferring nu ...")
            self.logger.info(f"self.degree: {self.degree}")
            self.logger.info(f"self.random_ensemble_kappa_per_degree_class: {self.random_ensemble_kappa_per_degree_class}")
            start_time = time.time()
        xi_m1, xi_00, xi_p1 = 0, 0, 0
        for v1 in range(self.nb_vertices):
            for v2 in range(v1 + 1, self.nb_vertices):
                kout1kin2 = self.random_ensemble_kappa_per_degree_class[1][int(self.degree[1][v1])] * self.random_ensemble_kappa_per_degree_class[0][int(self.degree[0][v2])]
                kout2kin1 = self.random_ensemble_kappa_per_degree_class[1][int(self.degree[1][v2])] * self.random_ensemble_kappa_per_degree_class[0][int(self.degree[0][v1])]

                p12 = self.directed_connection_probability(self.PI, kout1kin2)
                p21 = self.directed_connection_probability(self.PI, kout2kin1)

                if jnp.abs(kout1kin2 - kout2kin1) < self.NUMERICAL_CONVERGENCE_THRESHOLD_2:
                    xi_00 += hyp2f1c(self.beta, -((self.R * self.PI) / (self.mu * kout1kin2)) ** self.beta)
                else:
                    xi_00 += (1 / (1 - (kout1kin2 / kout2kin1) ** self.beta)) * (p12 - (kout1kin2 / kout2kin1) ** self.beta * p21)

                if kout1kin2 < kout2kin1:
                    xi_p1 += p12
                else:
                    xi_p1 += p21

                z_max = self.find_minimal_angle_by_bisection(kout1kin2, kout2kin1)
                if z_max < self.PI:
                    p12 = self.directed_connection_probability(z_max, kout1kin2)
                    p21 = self.directed_connection_probability(z_max, kout2kin1)
                    xi_m1 += (z_max / self.PI) - p12 - p21
                else:
                    xi_m1 += 1 - p12 - p21

        xi_m1 /= self.random_ensemble_average_degree * self.nb_vertices / 2
        xi_00 /= self.random_ensemble_average_degree * self.nb_vertices / 2
        xi_p1 /= self.random_ensemble_average_degree * self.nb_vertices / 2

        if self.reciprocity > xi_00:
            self.nu = (self.reciprocity - xi_00) / (xi_p1 - xi_00)
        else:
            self.nu = (self.reciprocity - xi_00) / (xi_m1 + xi_00)

        if self.nu > 1:
            self.nu = 1
        if self.nu < -1:
            self.nu = -1

        if self.nu > 0:
            self.random_ensemble_reciprocity = (xi_p1 - xi_00) * self.nu + xi_00
        else:
            self.random_ensemble_reciprocity = (xi_m1 + xi_00) * self.nu + xi_00
        if self.verbose:
            self.logger.info(f"nu: {self.nu}, random_ensemble_reciprocity: {self.random_ensemble_reciprocity}, time: {time.time() - start_time}")

    def infer_parameters(self) -> None:
        """
        Infer the parameters beta, mu, nu, and R.

        The function performs the following steps:
        1. Infer beta.
        2. Infer mu given beta.
        3. Infer R.
        4. Infer kappa given beta and mu.
        5. Compute random ensemble average degree.
        6. Infer nu.
        7. Build cumulative distribution for Monte Carlo integration for clustering.
        8. Compute random ensemble clustering.
        9. Refine beta (back to step 1) if not converged.
        """
        if not self.CUSTOM_BETA:
            self.beta = 1 + random.uniform(self.engine)
            if not self.CUSTOM_MU:
                self.mu = self.beta * jnp.sin(self.PI / self.beta) / (2.0 * self.PI * self.average_degree)
            self.R = self.nb_vertices / (2 * self.PI)

            beta_max = -1
            beta_min = 1
            random_ensemble_average_clustering_min = 0
            random_ensemble_average_clustering_max = 0
            iter = 0
            while True:
                if self.verbose:
                    self.logger.info(f"Beta: {self.beta}, iteration count: {iter}")
                self.infer_kappas()
                self.compute_random_ensemble_average_degree()
                if not self.CUSTOM_NU:
                    self.infer_nu()
                self.build_cumul_dist_for_mc_integration()
                self.compute_random_ensemble_clustering()

                if jnp.abs(self.random_ensemble_average_clustering - self.average_undir_clustering) < self.NUMERICAL_CONVERGENCE_THRESHOLD_1:
                    break

                if self.random_ensemble_average_clustering > self.average_undir_clustering:
                    beta_max = self.beta
                    random_ensemble_average_clustering_max = self.random_ensemble_average_clustering

                    if beta_min == 1:
                        self.beta = (beta_max + beta_min) / 2
                    else:
                        self.beta = beta_min + (beta_max - beta_min) * (self.average_undir_clustering - random_ensemble_average_clustering_min) / (random_ensemble_average_clustering_max - random_ensemble_average_clustering_min)
                    if self.beta < self.BETA_ABS_MIN:
                        break
                else:
                    beta_min = self.beta
                    random_ensemble_average_clustering_min = self.random_ensemble_average_clustering
                    if beta_max == -1:
                        self.beta *= 1.5
                    else:
                        self.beta = beta_min + (beta_max - beta_min) * (self.average_undir_clustering - random_ensemble_average_clustering_min) / (random_ensemble_average_clustering_max - random_ensemble_average_clustering_min)

                if self.beta > self.BETA_ABS_MAX:
                    break
        else:
            if not self.CUSTOM_MU:
                self.mu = self.beta * jnp.sin(self.PI / self.beta) / (2.0 * self.PI * self.average_degree)
            self.R = self.nb_vertices / (2 * self.PI)
            self.infer_kappas()
            self.compute_random_ensemble_average_degree()
            if not self.CUSTOM_NU:
                self.infer_nu()
            self.build_cumul_dist_for_mc_integration()
            self.compute_random_ensemble_clustering()

    def initialize_degrees(self) -> None:
        """
        Initialize the degrees compatible to computation by classes.

        The function performs the following steps:
        - Initialize degree histograms.
        - Compute average degree.
        - Classify nodes by their degrees.
        """
        self.degree_histogram = [self._compute_degree_histogram(self.in_degree_sequence),
                                 self._compute_degree_histogram(self.out_degree_sequence)]
        self.average_degree = sum(self.in_degree_sequence) / self.nb_vertices
        self.degree_class = self._classify_degrees()

    def _fit(self, reciprocity: float, average_local_clustering: float) -> None:
        """
        Fit the model to the network data.

        Parameters:
            reciprocity (float): Reciprocity of the network.
            average_local_clustering (float): Average local clustering coefficient of the network.
        """
        self.reciprocity = reciprocity
        self.average_undir_clustering = average_local_clustering
        self.initialize_degrees()
        self.infer_parameters()
        self.save_inferred_parameters()

    def fit_from_deg_seq(self, deg_seq: Tuple[List[float], List[float]], reciprocity: float, average_local_clustering: float, network_name: str = "", verbose: bool = False) -> None:
        """
        Fit the model to the network data from a degree sequence, reciprocity, and average local clustering coefficient.

        Parameters:
            deg_seq (Tuple[List[float], List[float]]): Tuple of two lists of integers/float representing in-degree and out-degree sequences.
            reciprocity (float): Reciprocity of the network.
            average_local_clustering (float): Average local clustering coefficient of the network.
            network_name (str): Name of the network (default = "").
            verbose (bool): Verbosity (default = False).
        """
        self.verbose = verbose
        self.output_rootname = network_name
        try:
            self.in_degree_sequence, self.out_degree_sequence = deg_seq
            assert len(self.in_degree_sequence) == len(self.out_degree_sequence)
        except:
            raise ValueError("Degree is a tuple of two lists of integers/float and sequences must be of equal length.")

        self.nb_vertices = len(self.in_degree_sequence)
        self.degree = deg_seq
        if self.verbose:
            self.logger.info(f"Using degree sequence of length: {self.nb_vertices}")

        self._fit(reciprocity, average_local_clustering)

    def fit_from_file(self, filename: str, reciprocity: float, average_local_clustering: float, network_name: str = "", verbose: bool = False) -> None:
        """
        Fit the model to the network data from a file.

        Parameters:
            filename (str): Filename containing the degree sequence.
            reciprocity (float): Reciprocity of the network.
            average_local_clustering (float): Average local clustering coefficient of the network.
            network_name (str): Name of the network (default = "").
            verbose (bool): Verbosity (default = False).
        """
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Reading degree sequence from file: {filename}")

        if network_name:
            self.output_rootname = network_name
        else:
            self.output_rootname = self.deg_seq_filename.split(".")[0]

        self.deg_seq_filename = filename
        with open(self.deg_seq_filename, 'r') as f:
            for line in f:
                try:
                    k_out, k_in = map(float, line.strip().split())
                    self.out_degree_sequence.append(k_out)
                    self.in_degree_sequence.append(k_in)
                except:
                    if self.verbose:
                        self.logger.info(f"Skipping: {line}")
                    pass

        self.nb_vertices = len(self.out_degree_sequence)
        if self.verbose:
            self.logger.info(f"Number of vertices: {self.nb_vertices}")

        self.degree = [self.in_degree_sequence, self.out_degree_sequence]

        self._fit(reciprocity, average_local_clustering)

    def save_inferred_parameters(self) -> None:
        """
        Save the inferred parameters to a JSON file.
        """

        data = {
            "beta": float(self.beta),
            "mu": float(self.mu),
            "nu": float(self.nu),
            "R": float(self.R),
            "inferred_kappas": []
        }
        if self.verbose:
            self.logger.info("Data types:")
            self.logger.info(f"beta: {type(self.beta)}, mu: {type(self.mu)}, nu: {type(self.nu)}, R: {type(self.R)}")

        for in_deg, out_degs in self.degree_class.items():
            for out_deg, count in out_degs.items():
                kappa_in = self.random_ensemble_kappa_per_degree_class[0][in_deg]
                kappa_out = self.random_ensemble_kappa_per_degree_class[1][out_deg]
                data["inferred_kappas"].append({
                    "in_deg": in_deg,
                    "out_deg": out_deg,
                    "kappa_in": kappa_in.tolist() if hasattr(kappa_in, 'tolist') else kappa_in,
                    "kappa_out": kappa_out.tolist() if hasattr(kappa_out, 'tolist') else kappa_out
                })
        if not os.path.exists("outputs"):
            os.makedirs("outputs")  
        if not os.path.exists(f"outputs/{self.output_rootname}"):
            os.makedirs(f"outputs/{self.output_rootname}")
        with open(f"outputs/{self.output_rootname}/inferred_params.json", 'w') as f:
            json.dump(data, f, indent=4)
            
    def modify_log_file_path(self, log_file_path: str) -> None:
        """
        Modify the log file path for the logger.

        Parameters:
            log_file_path (str): Path to the log file.
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

def main():
    """
    Main function to initialize the model and fit it to the network data.
    """
    start_time = time.time()
    model = DirectedS1Fitter(seed=0, verbose=True, log_file_path="output_small.log")
    deg_seq_filename = sys.argv[1]
    reciprocity = 0.05  # Example value, set appropriately
    average_local_clustering = 0.25  # Example value, set appropriately
    print(f"Inferring params with inputs: {deg_seq_filename, reciprocity, average_local_clustering}")
    print("Fitting ...")
    model.fit_from_file(deg_seq_filename, reciprocity, average_local_clustering, verbose=True)
    print("Fitted! Inferring parameters")
    model.save_inferred_parameters()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
