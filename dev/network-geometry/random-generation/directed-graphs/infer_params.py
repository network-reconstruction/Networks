"""
Model from Supplmentary Information of the paper: Geometric description of clustering in directed networks: Antoine Allard et al.
"""
import jax.numpy as jnp
import jax.random as random
from scipy.special import hyp2f1
import numpy as np
from typing import Dict, List, Tuple
from jax import vmap, pmap
import time
import sys
from sortedcontainers import SortedDict #cum

def lower_bound(d, key):
    index = d.bisect_left(key)
    if index < len(d):
        return d.peekitem(index)
    else:
        return None
    
    
def hyp2f1a(beta, z):
    return hyp2f1(1.0, 1.0 / beta, 1.0 + (1.0 / beta), z).real

def hyp2f1b(beta, z):
    return hyp2f1(1.0, 2.0 / beta, 1.0 + (2.0 / beta), z).real

def hyp2f1c(beta, z): #S75b
    return hyp2f1(2.0, 1.0 / beta, 1.0 + (1.0 / beta), z).real

class FittingDirectedS1:
    PI = jnp.pi

    def __init__(self, seed: int = 0, verbose: bool = False, KAPPA_MAX_NB_ITER_CONV: int = 3):
        self.ALLOW_LARGE_INITIAL_ANGULAR_GAPS = True
        self.CHARACTERIZATION_MODE = False
        self.CLEAN_RAW_OUTPUT_MODE = False
        self.CUSTOM_BETA = False
        self.CUSTOM_MU = False
        self.CUSTOM_NU = False
        self.CUSTOM_CHARACTERIZATION_NB_GRAPHS = False
        self.CUSTOM_INFERRED_COORDINATES = False
        self.CUSTOM_OUTPUT_ROOTNAME_MODE = False
        self.CUSTOM_SEED = False
        self.KAPPA_POST_INFERENCE_MODE = True
        self.MAXIMIZATION_MODE = True
        self.QUIET_MODE = False
        self.REFINE_MODE = False
        self.VALIDATION_MODE = False
        self.VERBOSE_MODE = False

        self.ALREADY_INFERRED_PARAMETERS_FILENAME = ""
        self.BETA_ABS_MAX = 25
        self.BETA_ABS_MIN = 1.01
        self.CHARACTERIZATION_NB_GRAPHS = 100
        self.MIN_NB_ANGLES_TO_TRY = 100
        self.EDGELIST_FILENAME = ""
        self.EXP_CLUST_NB_INTEGRATION_MC_STEPS = 10 #200
        self.KAPPA_MAX_NB_ITER_CONV = KAPPA_MAX_NB_ITER_CONV
        self.NUMERICAL_CONVERGENCE_THRESHOLD_1 = 1e-2
        self.NUMERICAL_CONVERGENCE_THRESHOLD_2 = 1e-2
        self.NUMERICAL_ZERO = 1e-5
        self.ROOTNAME_OUTPUT = ""
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

        self.kappas_out, self.kappas_in = [], []
        self.verbose = False
        

    def _compute_degree_histogram(self, degrees: List[float]) -> Dict[int, int]:
        """
        Compute the degree histogram.
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
        """
        degree_class = {}
        for k_in, k_out in zip(self.kappas_in, self.kappas_out):
            k_in, k_out = int(k_in), int(k_out)
            if k_in not in degree_class:
                degree_class[k_in] = {}
            if k_out not in degree_class[k_in]:
                degree_class[k_in][k_out] = 0
            degree_class[k_in][k_out] += 1
        return degree_class

    def directed_connection_probability(self, z: float, koutkin: float) -> float:
        """
        Compute the directed connection probability.
        """
        return (z / self.PI) * hyp2f1a(self.beta, -((self.R * z) / (self.mu * koutkin)) ** self.beta)

    def undirected_connection_probability(self, z: float, kout1kin2: float, kout2kin1: float) -> float:
        """
        Compute the undirected connection probability.
        This whole function is S79 until S82
        """
        # Potentially bellow between S81 line and S82 line
        p12 = self.directed_connection_probability(z, kout1kin2) #S23a 
        p21 = self.directed_connection_probability(z, kout2kin1) #S23a 

        # S81
        # -------------------------------------
        conn_prob = 0
        if jnp.abs(kout1kin2 - kout2kin1) < self.NUMERICAL_CONVERGENCE_THRESHOLD_2:
            conn_prob -= (1 - jnp.abs(self.nu)) * hyp2f1c(self.beta, -((self.R * self.PI) / (self.mu * kout1kin2)) ** self.beta) #S75b term in S82 with 0<= nu <=1
            #why minus? We are subtracting from the equation in S80
        else:
            conn_prob -= ((1 - jnp.abs(self.nu)) / (1 - (kout1kin2 / kout2kin1) ** self.beta)) #I've 
            conn_prob *= (p12 - (kout1kin2 / kout2kin1) ** self.beta * p21)
        conn_prob += p12
        conn_prob += p21
        # -------------------------------------
        # TODO: Figure this equation out
        # -------------------------------------
        if self.nu > 0:
            if kout1kin2 < kout2kin1:
                conn_prob -= self.nu * p12
            else:
                conn_prob -= self.nu * p21 # -------------------------------------
        elif self.nu < 0:
            z_max = self.find_minimal_angle_by_bisection(kout1kin2, kout2kin1) #S86
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
        Equation (S86)
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
        From S80 to S85 of the paper, calculation for point 4 in S86
        """
        if self.verbose:
            print(f"Building cumulative distribution for Monte Carlo integration ...")
            print(f"self.degree_class: {self.degree_class}")
            start_time = time.time()
        for in_deg, out_degs in self.degree_class.items(): 
            #Iterating over all nodes,basically but via degree classes
            #This iteration and inner iteration form point 1 above S86
            self.cumul_prob_kgkp[in_deg] = {}
            for out_deg in out_degs:
                
                #for each pair of in and out observed degree classes, (k_1in, k_1out)
                    
                nkkp = {(i, j): 0 for i in self.degree_class for j in self.degree_class[i]}
                #nkkp is a dictionary of tuples of in and out degrees, with values as 0
                tmp_cumul = 0
                k_in1 = self.random_ensemble_kappa_per_degree_class[0][in_deg] 
                kout1 = self.random_ensemble_kappa_per_degree_class[1][out_deg]
                #We take kappa values from random ensemble
                for in_deg2, out_degs2 in self.degree_class.items(): 
                    #iterating over all nodes,basically but via degree classes
                    for out_deg2 in out_degs2:
                        k_in2 = self.random_ensemble_kappa_per_degree_class[0][in_deg2]
                        kout2 = self.random_ensemble_kappa_per_degree_class[1][out_deg2]
                        tmp_val = self.undirected_connection_probability(self.PI, kout1 * k_in2, kout2 * k_in1) #S86
                        
                        if in_deg == in_deg2 and out_deg == out_deg2:
                            nkkp[(in_deg2, out_deg2)] = (out_degs2[out_deg2] - 1) * tmp_val 
                        else:
                            nkkp[(in_deg2, out_deg2)] = out_degs2[out_deg2] * tmp_val
                        tmp_cumul += nkkp[(in_deg2, out_deg2)]
                        
                #noralizing as we multiplied with counts per degree class, so now we have probabilities in nkkp[key]
                for key in nkkp:
                    nkkp[key] /= tmp_cumul 
                    
                # For each degree class (in, out), we have nkkp which gives cumulative distribution of probability of connection.
                tmp_cumul = 0
                for key in nkkp:
                    tmp_val = nkkp[key]
                    if tmp_val > self.NUMERICAL_ZERO:
                        tmp_cumul += tmp_val 
                        if out_deg not in self.cumul_prob_kgkp[in_deg]:
                            self.cumul_prob_kgkp[in_deg][out_deg] = SortedDict()
                        self.cumul_prob_kgkp[in_deg][out_deg][int(tmp_cumul)] = key
                        #tmp_cumul is single digit array in jax so we need to convert to hashable type by dictionary
        if self.verbose:
            print(f"finished building cumulative distribution for Monte Carlo integration ...")
            print(f"runtime: {time.time() - start_time}")
    def compute_random_ensemble_average_degree(self) -> None:
        """
        Compute the random ensemble average degree.
        """
        if self.verbose:
            print(f"Computing random ensemble average degree ...")
        directions = [0, 1]
        random_ensemble_average_degree = 0
        for direction in directions:
            for el in self.degree_histogram[direction].items():
                random_ensemble_average_degree += el[1] * self.random_ensemble_expected_degree_per_degree_class[direction][el[0]]
        self.random_ensemble_average_degree = random_ensemble_average_degree / (self.nb_vertices * 2)

    def compute_random_ensemble_clustering(self) -> None:
        """
        Compute the random ensemble average clustering.
        S85 of the paper.
        """
        if self.verbose:
            print(f"Computing random ensemble clustering ...")  
        random_ensemble_average_clustering = 0
        for in_deg, out_degs in self.degree_class.items():
            for out_deg in out_degs:
                if in_deg + out_deg > 1: # only consider degree classes with in and out degrees greater than 1
                    # basically over all nodes, done in degree class outer summation of S85
                    tmp_val = self.compute_random_ensemble_clustering_for_degree_class(in_deg, out_deg) 
                    random_ensemble_average_clustering += tmp_val * out_degs[out_deg] #this is per class
        self.random_ensemble_average_clustering = random_ensemble_average_clustering / self.nb_vertices_undir_degree_gt_one # 1 / MN

    def compute_random_ensemble_clustering_for_degree_class(self, in_deg: int, out_deg: int) -> float:
        """
        Compute the random ensemble clustering for a given degree class.
        Point 1-4 after Equation S86 of the paper, inner summation of S85
        """
        if self.verbose:
            print(f"Computing random ensemble clustering for degree class: {in_deg}, {out_deg} ...")
            print(f"Cumul_prob_kgkp: {self.cumul_prob_kgkp}")
            start_time = time.time()

        tmp_cumul = 0
        k_in1 = self.random_ensemble_kappa_per_degree_class[0][in_deg]
        kout1 = self.random_ensemble_kappa_per_degree_class[1][out_deg]

        M = self.EXP_CLUST_NB_INTEGRATION_MC_STEPS #this is the value of M in S85
        if self.verbose:
            #print all of the first two keys of self.cumul_prob_kgkp
            print(f"In and out pairs for cumul_prob_kgkp:")
            for key1 in self.cumul_prob_kgkp.keys():
                for key2 in self.cumul_prob_kgkp[key1].keys():
                    print(f"({key1}, {key2})")
            print(f"just for debugging: { self.cumul_prob_kgkp[2][3]}")
            
        for i in range(M): #this is the value of M in S85
            cumul_prob_dict = self.cumul_prob_kgkp[int(in_deg)].get(int(out_deg), SortedDict())
            random_val = random.uniform(self.engine)
            lower_bound_key = cumul_prob_dict.bisect_left(random_val)
            lower_bound_key = min(lower_bound_key, len(cumul_prob_dict) - 1)
            if self.verbose:
                print(f"sample: {i}")
            #     print(f"in_deg: {in_deg}, out_deg: {out_deg}")
            #     print(f"cumul_prob_dict: {cumul_prob_dict}")
            #     print(f"lower_bound_key: {lower_bound_key}")
            p2_key = cumul_prob_dict.iloc[lower_bound_key]
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
            p3_key = cumul_prob_dict.iloc[lower_bound_key]
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
            print(f"Time: {time.time() - start_time}")
        return tmp_cumul / M

    def infer_kappas(self) -> None:
        if self.verbose:
            print(f"Infering kappas ...")
            print(f"degree_histogram: {self.degree_histogram}")
        self.random_ensemble_kappa_per_degree_class = [{} for _ in range(2)]
        self.random_ensemble_expected_degree_per_degree_class = [{} for _ in range(2)]
        directions = [0, 1] # in and out degrees [in, out]
        for direction in directions:
            for el in self.degree_histogram[direction].items():
                self.random_ensemble_kappa_per_degree_class[direction][el[0]] = el[0] + 0.001

        keep_going = True
        cnt = 0
        start_time = time.time()
        if self.verbose:
            print(f"KAPPA_MAX_NB_ITER_CONV: {self.KAPPA_MAX_NB_ITER_CONV}")

        while keep_going and cnt < self.KAPPA_MAX_NB_ITER_CONV:
            if self.verbose:
                print(f"Iteration: {cnt}, Previous Iteration time: {time.time() - start_time}")
                start_time = time.time()
            for direction in directions:
                for el in self.degree_histogram[direction].items():
                    self.random_ensemble_expected_degree_per_degree_class[direction][el[0]] = 0

            
            if self.verbose:
                
                prob_conn_mat = np.zeros((len(self.degree_histogram[0]), len(self.degree_histogram[1])))
                kappa_prod_mat = np.zeros((len(self.degree_histogram[0]), len(self.degree_histogram[1])))
                print(f"kappa_prod_mat: {kappa_prod_mat}")
            for i, (in_deg, count_in) in enumerate(self.degree_histogram[0].items()):
                for j, (out_deg, count_out) in enumerate(self.degree_histogram[1].items()):
                    prob_conn = self.directed_connection_probability(
                        self.PI,
                        self.random_ensemble_kappa_per_degree_class[1][out_deg] * self.random_ensemble_kappa_per_degree_class[0][in_deg]
                    )
                    #indexing with degree histogram index
                    if self.verbose:
                        prob_conn_mat[i,j] = prob_conn 
                        kappa_prod_mat[i,j] = self.random_ensemble_kappa_per_degree_class[1][out_deg] * self.random_ensemble_kappa_per_degree_class[0][in_deg]
                    self.random_ensemble_expected_degree_per_degree_class[0][in_deg] += prob_conn * count_in
                    self.random_ensemble_expected_degree_per_degree_class[1][out_deg] += prob_conn * count_out
            if self.verbose:
                print(f"prob_conn_mat: {prob_conn_mat}")
                print(f"kappa_prod_mat: {kappa_prod_mat}")
            error = jnp.inf
            keep_going = False
            for direction in directions:
                for el in self.degree_histogram[direction].items():
                    error = jnp.abs(self.random_ensemble_expected_degree_per_degree_class[direction][el[0]] - el[0])
                    if error > self.NUMERICAL_CONVERGENCE_THRESHOLD_1:
                        keep_going = True
                        break
            if self.verbose:
                print(f"Error: {error}, NUMERICAL_CONVERGENCE_THRESHOLD_1: {self.NUMERICAL_CONVERGENCE_THRESHOLD_1}")
                
            if keep_going:
                for direction in directions:
                    for el in self.degree_histogram[direction].items():
                        self.random_ensemble_kappa_per_degree_class[direction][el[0]] += (el[0] - self.random_ensemble_expected_degree_per_degree_class[direction][el[0]]) * random.uniform(self.engine)
                        if self.random_ensemble_kappa_per_degree_class[direction][el[0]] < 0:
                            self.random_ensemble_kappa_per_degree_class[direction][el[0]] = jnp.abs(self.random_ensemble_kappa_per_degree_class[direction][el[0]])
                cnt += 1

            if cnt >= self.KAPPA_MAX_NB_ITER_CONV:
                print("WARNING: Maximum number of iterations reached before convergence. This limit can be adjusted through the parameter KAPPA_MAX_NB_ITER_CONV.")


    def infer_nu(self) -> None:
        """
        Infer the parameter nu.
        """
        if self.verbose:
            print(f"Infering nu ...")
            print(f"self.random_ensemble_kappa_per_degree_class: {self.random_ensemble_kappa_per_degree_class}")
            start_time = time.time()    
            #random_ensemble_kappa_per_degree_class
        xi_m1, xi_00, xi_p1 = 0, 0, 0
        for v1 in range(self.nb_vertices):
            for v2 in range(v1 + 1, self.nb_vertices):
                #convert self.degree to int buckets, as data generated originally needs degree sequence but I random generated without being degree seq, rather decimals.
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
            print(f"nu: {self.nu}, random_ensemble_reciprocity: {self.random_ensemble_reciprocity}, time: {time.time() - start_time}")
    def infer_parameters(self) -> None:
        """
        Infer the parameters beta, mu, nu, and R.
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
                    print(f"Beta: {self.beta}, iteration count: {iter}")      
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

    def initialize(self) -> None:
        if self.verbose:
            print(f"Initializing ...")
        with open(self.EDGELIST_FILENAME, 'r') as f:
            for line in f:
                try:
                    k_out, k_in = map(float, line.strip().split())
                    self.kappas_out.append(k_out)
                    self.kappas_in.append(k_in)
                except:
                    if self.verbose:
                        print(f"Skipping: {line}")
                    pass

        self.nb_vertices = len(self.kappas_out)
        if self.verbose:
            print(f"Number of vertices: {self.nb_vertices}")

        self.degree = [self.kappas_in, self.kappas_out]

        # Initialize degree histograms
        self.degree_histogram = [self._compute_degree_histogram(self.kappas_in),
                                self._compute_degree_histogram(self.kappas_out)]

        # Compute average degree
        self.average_degree = sum(self.kappas_in) / self.nb_vertices

        # Classify nodes by their degrees
        self.degree_class = self._classify_degrees()


    def fit(self, edgelist_filename: str, reciprocity: float, average_local_clustering: float, verbose: bool = False) -> None:
        """
        Fit the model to the network data.
        """
        self.verbose = verbose
        self.EDGELIST_FILENAME = edgelist_filename
        self.reciprocity = reciprocity
        self.average_undir_clustering = average_local_clustering
        self.initialize()
        self.infer_parameters()
        self.save_inferred_parameters()
        self.finalize()

    def save_inferred_parameters(self) -> None:
        """
        Save the inferred parameters to a file.
        """
        output_filename = self.ROOTNAME_OUTPUT + "_inferred_parameters.txt"
        with open(output_filename, 'w') as f:
            f.write(f"beta: {self.beta}\n")
            f.write(f"mu: {self.mu}\n")
            f.write(f"nu: {self.nu}\n")
            f.write(f"R: {self.R}\n")
            f.write("Inferred kappas:\n")
            for in_deg, out_degs in self.degree_class.items():
                for out_deg, count in out_degs.items():
                    kappa_in = self.random_ensemble_kappa_per_degree_class[0][in_deg]
                    kappa_out = self.random_ensemble_kappa_per_degree_class[1][out_deg]
                    f.write(f"{in_deg} {out_deg} {kappa_in} {kappa_out}\n")

    def finalize(self) -> None:
        """
        Finalize the fitting process.
        """
        # Placeholder for any final steps or cleanup
        pass


def main():
    # Initialize the model
    # Ecuador 2015
    start_time = time.time()
    model = FittingDirectedS1(seed=0, verbose=True)
    #take edge list file name from arguments:
    # Set parameters
    edgelist_filename = sys.argv[1]
    reciprocity = 0.046  # Example value, set appropriately
    average_local_clustering = 0.28  # Example value, set appropriately
    print(f"Infering params with inputs: {edgelist_filename}, reciprocity: {reciprocity}, average_local_clustering: {average_local_clustering}")
    # Fit the model
    print(f"Fitting ...")
    model.fit(edgelist_filename, reciprocity, average_local_clustering, verbose=True)

    print(f"Fitted! inferring parameters")
    # Output inferred parameters (this will be saved to a file)
    model.save_inferred_parameters()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
