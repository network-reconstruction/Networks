import jax.numpy as jnp
import jax.random as random
from scipy.special import poch, factorial
import numpy as np
from typing import Dict, List, Tuple, Union
import time
import jax 
from jax import lax, jit, vmap
from functools import partial
import sys

def pochhammer(a, n):
    print(f" poch a: {a}, n: {n}")
    res = poch(a, n)
    print(f"poch res: {res}")
    return res

def hyp2f1_approx(a: float, b: float, c: float, z: float, n_terms: int =10) -> jnp.ndarray:
    """
    Compute the Taylor series expansion for _2F_1(a, b; c; z) using JAX.
    a, b, c, z: Scalar parameters for the hypergeometric function.
    n_terms: Number of terms in the series.
    """
    print(f"a: {a}, b: {b}, c: {c}, z: {z}")
    def series_term(n):
        term = (pochhammer(a, n) * pochhammer(b, n ) / pochhammer(c, n )) * (z**n) / factorial(n)
        return term
    
    terms = []
    for n in range(n_terms):
        term = series_term(n)
        #if a term has -inf in the array print it and the values inputed into the equation
        print(f"term: {term}, n: {n}")
        terms.append(term)
    terms = jnp.array(terms)
    # print(f"terms: {terms}")
    result = jnp.sum(terms, axis=0)
    # print(f"result: {result}")
    return result
  

def hyp2f1a(beta, z):
    return hyp2f1_approx(1.0, 1.0 / beta, 1.0 + (1.0 / beta), z).real

def hyp2f1b(beta, z):
    return hyp2f1_approx(1.0, 2.0 / beta, 1.0 + (2.0 / beta), z).real

def hyp2f1c(beta, z):
    return hyp2f1_approx(2.0, 1.0 / beta, 1.0 + (1.0 / beta), z).real

class FittingDirectedS1:
    PI = jnp.pi

    def __init__(self, seed: int = 0, verbose: bool = False):
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
        self.EXP_CLUST_NB_INTEGRATION_MC_STEPS = 200
        self.KAPPA_MAX_NB_ITER_CONV = 1000
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
        print(f"")
        return (z / self.PI) * hyp2f1a(self.beta, -((self.R * z) / (self.mu * koutkin)) ** self.beta)
    
    def directed_connection_probability_vmap(self, z: Union[float, jnp.ndarray], koutkin: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the directed connection probability vmapped
        z: float or jnp.ndarray
        koutkin: jnp.ndarray
        """
        if isinstance(z, float):
            #vectorize over koutkin vmap
            return jax.vmap(lambda k: self.directed_connection_probability(z, k))(koutkin)
        elif isinstance(z, jnp.ndarray):
            #vectorize over both z and koutkin
            return jax.vmap(lambda z, k: self.directed_connection_probability(z, k))(z, koutkin)
        else:
            raise ValueError("z must be a float or jnp.ndarray")

    def undirected_connection_probability(self, z: float, kout1kin2: float, kout2kin1: float) -> float:
        """
        Compute the undirected connection probability.
        """
        p12 = self.directed_connection_probability(z, kout1kin2)
        p21 = self.directed_connection_probability(z, kout2kin1)

        conn_prob = 0
        if jnp.abs(kout1kin2 - kout2kin1) < self.NUMERICAL_CONVERGENCE_THRESHOLD_2:
            conn_prob -= (1 - jnp.abs(self.nu)) * hyp2f1c(self.beta, -((self.R * self.PI) / (self.mu * kout1kin2)) ** self.beta)
        else:
            conn_prob -= ((1 - jnp.abs(self.nu)) / (1 - (kout1kin2 / kout2kin1) ** self.beta))
            conn_prob *= (p12 - (kout1kin2 / kout2kin1) ** self.beta * p21)
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
            print(f"Building cumulative distribution for Monte Carlo integration ...")
        cumul_prob_kgkp = {}
        for in_deg, out_degs in self.degree_class.items():
            for out_deg in out_degs:
                cumul_prob_kgkp[in_deg] = {}
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
                        cumul_prob_kgkp[in_deg][out_deg] = cumul_prob_kgkp[in_deg].get(out_deg, {})
                        cumul_prob_kgkp[in_deg][out_deg][tmp_cumul] = key
        self.cumul_prob_kgkp = cumul_prob_kgkp
        

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
        """
        if self.verbose:
            print(f"Computing random ensemble clustering ...")  
        random_ensemble_average_clustering = 0
        for in_deg, out_degs in self.degree_class.items():
            for out_deg in out_degs:
                if in_deg + out_deg > 1:
                    tmp_val = self.compute_random_ensemble_clustering_for_degree_class((in_deg, out_deg))
                    random_ensemble_average_clustering += tmp_val * out_degs[out_deg]
        self.random_ensemble_average_clustering = random_ensemble_average_clustering / self.nb_vertices_undir_degree_gt_one

    def compute_random_ensemble_clustering_for_degree_class(self, in_deg: int, out_deg: int) -> float:
        """
        Compute the random ensemble clustering for a specific degree class.
        """
        if self.verbose:
            print(f"Computing random ensemble clustering for degree class: {in_deg}, {out_deg} ...")
            start_time = time.time()

        p2, p3 = None, None
        p12, p13, p23, p32, pc, pz, zmin, zmax, z, z12, z13, da, k_in2, kout2, k_in3, kout3 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        tmp_val, tmp_cumul = 0, 0
        k_in1 = self.random_ensemble_kappa_per_degree_class[0][in_deg]
        kout1 = self.random_ensemble_kappa_per_degree_class[1][out_deg]

        nb_points = self.EXP_CLUST_NB_INTEGRATION_MC_STEPS

        for i in range(nb_points):
            p2 = self.cumul_prob_kgkp[in_deg][out_deg].lower_bound(random.uniform(self.engine)).second
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

            p3 = self.cumul_prob_kgkp[in_deg][out_deg].lower_bound(random.uniform(self.engine)).second
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
            da = jnp.min(da, (2.0 * self.PI) - da)

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
        return tmp_cumul / nb_points

    def infer_kappas(self) -> None:
        if self.verbose:
            print(f"Infering kappas ...")
        self.random_ensemble_kappa_per_degree_class = [{} for _ in range(2)]
        self.random_ensemble_expected_degree_per_degree_class = [{} for _ in range(2)]
        directions = [0, 1]
        
        # Vectorized 1
        for direction in directions:
            for el in self.degree_histogram[direction].items():
                self.random_ensemble_kappa_per_degree_class[direction][el[0]] = el[0] + 0.001
        # ---------------------

        keep_going = True
        cnt = 0
        start_time = time.time()
        while keep_going and cnt < self.KAPPA_MAX_NB_ITER_CONV:
            if self.verbose:
                print(f"Iteration: {cnt}, Previous Iteration time: {time.time() - start_time}")
                start_time = time.time()
            for direction in directions:
                for el in self.degree_histogram[direction].items():
                    self.random_ensemble_expected_degree_per_degree_class[direction][el[0]] = 0

            for in_deg, count_in in self.degree_histogram[0].items():
                for out_deg, count_out in self.degree_histogram[1].items():
                    prob_conn = self.directed_connection_probability(
                        self.PI,
                        self.random_ensemble_kappa_per_degree_class[1][out_deg] * self.random_ensemble_kappa_per_degree_class[0][in_deg]
                    )
                    self.random_ensemble_expected_degree_per_degree_class[0][in_deg] += prob_conn * count_in
                    self.random_ensemble_expected_degree_per_degree_class[1][out_deg] += prob_conn * count_out

            keep_going = False
            for direction in directions:
                for el in self.degree_histogram[direction].items():
                    if jnp.abs(self.random_ensemble_expected_degree_per_degree_class[direction][el[0]] - el[0]) > self.NUMERICAL_CONVERGENCE_THRESHOLD_1:
                        keep_going = True

            if keep_going:
                for direction in directions:
                    for el in self.degree_histogram[direction].items():
                        self.random_ensemble_kappa_per_degree_class[direction][el[0]] += (el[0] - self.random_ensemble_expected_degree_per_degree_class[direction][el[0]]) * random.uniform(self.engine)
                        if self.random_ensemble_kappa_per_degree_class[direction][el[0]] < 0:
                            self.random_ensemble_kappa_per_degree_class[direction][el[0]] = jnp.abs(self.random_ensemble_kappa_per_degree_class[direction][el[0]])
                cnt += 1

            if cnt >= self.KAPPA_MAX_NB_ITER_CONV:
                print("WARNING: Maximum number of iterations reached before convergence. This limit can be adjusted through the parameter KAPPA_MAX_NB_ITER_CONV.")
    def infer_kappas_vmap(self) -> None:
        """
        infer kappas using vmap for parallelization
        """
        if self.verbose:
            print(f"Infering kappas ...")
        self.random_ensemble_kappa_per_degree_class = [{} for _ in range(2)] # [in-degree, out-degree]
        self.random_ensemble_expected_degree_per_degree_class = [{} for _ in range(2)] # [in-degree, out-degree]

        if self.verbose:
            print(f"degree histogram: {self.degree_histogram}")
        
        for direction, direction_dict in enumerate(self.degree_histogram):
            for degree in direction_dict.keys():
                self.random_ensemble_kappa_per_degree_class[direction][degree] = degree + 0.001

        keep_going = True
        cnt = 0
        start_time = time.time()
        while keep_going and cnt < self.KAPPA_MAX_NB_ITER_CONV:
            if self.verbose:
                print(f"Iteration: {cnt}, Previous Iteration time: {time.time() - start_time}")
                start_time = time.time()
            for direction, direction_dict in enumerate(self.degree_histogram):
                for degree in direction_dict.keys():
                    self.random_ensemble_expected_degree_per_degree_class[direction][degree] = degree

            # Vectorize 1:
            in_degrees = jnp.array(list(self.degree_histogram[0].keys()))
            count_ins = jnp.array(list(self.degree_histogram[0].values()))
            out_degrees = jnp.array(list(self.degree_histogram[1].keys()))
            count_outs = jnp.array(list(self.degree_histogram[1].values()))

            # Convert JAX arrays to native Python integers for dictionary lookups
            in_degrees_list = [int(deg) for deg in in_degrees]
            out_degrees_list = [int(deg) for deg in out_degrees]

            # Extract kappa values for the in-degrees and out-degrees
            in_kappas = jnp.array([self.random_ensemble_kappa_per_degree_class[0][deg] for deg in in_degrees_list])
            out_kappas = jnp.array([self.random_ensemble_kappa_per_degree_class[1][deg] for deg in out_degrees_list])
            
            # Create grids for the degrees and kappas
            in_deg_grid, out_deg_grid = jnp.meshgrid(in_degrees, out_degrees, indexing='ij')
            in_kappa_grid, out_kappa_grid = jnp.meshgrid(in_kappas, out_kappas, indexing='ij')
            count_in_grid, count_out_grid = jnp.meshgrid(count_ins, count_outs, indexing='ij')
            if self.verbose:
                print(f"count_ins: {count_ins}\n shape: {count_ins.shape}")
                print(f"count_outs: {count_outs}\n shape: {count_outs.shape}")
                print(f"count_in_grid: {count_in_grid}\n shape: {count_in_grid.shape}")
                print(f"count_out_grid: {count_out_grid}\n shape: {count_out_grid.shape}")
            # Compute the product of kappas
            kappa_product = in_kappa_grid * out_kappa_grid

            # Compute the connection probabilities using vmap
            if self.verbose:
                print(f"kappa_product: {kappa_product}\n shape: {kappa_product.shape}")
            kappa_product_flattened = kappa_product.flatten()
            prob_conn = self.directed_connection_probability_vmap(self.PI, kappa_product_flattened)
            prob_conn = prob_conn.reshape(kappa_product.shape) #shape is (in_degrees, out_degrees) by class
            
            # Compute the expected degrees probability of connection * count per class, and then sum over all classes in opposite direction
            expected_in_degrees = jnp.sum(prob_conn * count_out_grid, axis=1)
            # prob_conn: [[prob kappa_inclass1 kappa_outclass1, prob kappa_inclass1 kappa_outclass2, ...], [prob kappa_inclass2 kappa_outclass1, prob kappa_inclass2 kappa_outclass2, ...] ... ] * count_out_grid: [[out_class1, out_class2, ...], [out_class1, out_class2, ...] ...] 
            # --> [[outclass1 * prob kappa_inclass1 kappa_outclass1, outclass2 * prob kappa_inclass1 kappa_outclass2, ...], [outclass1 * prob kappa_inclass2 kappa_outclass1, outclass2 * prob kappa_inclass2 kappa_outclass2, ...] ...]
            # --> sum over axis 0 to get expected in degrees
            # --> [expected in degree class 1, expected in degree class 2, ...]
            expected_out_degrees = jnp.sum(prob_conn * count_in_grid, axis=0) 
            # Update the dictionaries for expected degrees
            updated_in_degrees = {deg: expected_in_degrees[i] for i, deg in enumerate(in_degrees_list)}
            updated_out_degrees = {deg: expected_out_degrees[i] for i, deg in enumerate(out_degrees_list)}

            # Assign the updated values back to the original dictionaries
            self.random_ensemble_expected_degree_per_degree_class[0].update(updated_in_degrees)
            self.random_ensemble_expected_degree_per_degree_class[1].update(updated_out_degrees)

            # Calculate error and check for convergence
            error = jnp.inf
            keep_going = False
            for direction, direction_dict in enumerate(self.degree_histogram):
                for degree in direction_dict.keys():
                    error = jnp.abs(self.random_ensemble_expected_degree_per_degree_class[direction][degree] - degree)
                    if error > self.NUMERICAL_CONVERGENCE_THRESHOLD_1:
                        keep_going = True
                        break
            if self.verbose:
                print(f"Error: {error} NUMERICAL CONVERGENCE THRESHOLD: {self.NUMERICAL_CONVERGENCE_THRESHOLD_1}")

            if keep_going:
                for direction, direction_dict in enumerate(self.degree_histogram):
                    for degree in direction_dict.keys():
                        self.random_ensemble_kappa_per_degree_class[direction][degree] += (degree - self.random_ensemble_expected_degree_per_degree_class[direction][degree]) * random.uniform(self.engine)
                        if self.random_ensemble_kappa_per_degree_class[direction][degree] < 0:
                            self.random_ensemble_kappa_per_degree_class[direction][degree] = jnp.abs(self.random_ensemble_kappa_per_degree_class[direction][degree])
                cnt += 1

            if cnt >= self.KAPPA_MAX_NB_ITER_CONV:
                print("WARNING: Maximum number of iterations reached before convergence. This limit can be adjusted through the parameter KAPPA_MAX_NB_ITER_CONV.")

    def infer_nu(self) -> None:
        """
        Infer the parameter nu.
        """
        if self.verbose:
            print(f"Infering nu ...")
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
            print(f"Time: {time.time() - start_time}")
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
                self.infer_kappas_vmap()
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
                    #append rounded nearest integer k_out and k_in
                    self.kappas_out.append(round(k_out))
                    self.kappas_in.append(round(k_in))
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

    # Set parameters
    edgelist_filename = sys.argv[1]  # Example value, set appropriately
    reciprocity = 0.046  # Example value, set appropriately
    average_local_clustering = 0.28  # Example value, set appropriately

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
