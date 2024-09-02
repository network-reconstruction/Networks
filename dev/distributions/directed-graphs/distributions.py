import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from typing import Tuple, List

def bivariate_lognormal(key: jnp.ndarray, n: int, exp_mu: jnp.ndarray, Sigma: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate bivariate lognormal samples.

    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        exp_mu (jnp.ndarray): Expected mean values.
        Sigma (jnp.ndarray): Covariance matrix.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Generated samples for in-degrees and out-degrees.
    """
    mu = jnp.log(exp_mu) - 0.5 * jnp.diag(Sigma)
    L = jnp.linalg.cholesky(Sigma)
    samples = jnp.exp(mu + jnp.dot(random.normal(key, (n, 2)), L.T))
    return samples[:, 1], samples[:, 0]

def pareto_samples(key: jnp.ndarray, n: int, tail: float, scale: float = 1.0) -> jnp.ndarray:
    """
    Generate Pareto samples.

    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        tail (float): Tail index of the Pareto distribution.
        scale (float): Scale parameter of the Pareto distribution.

    Returns:
        jnp.ndarray: Generated Pareto samples.
    """
    return scale * (random.uniform(key, (n,)) ** (-1 / tail))


def bivariate_lognormal_pareto2(key: jnp.ndarray, n: int, exp_mu: jnp.ndarray, Sigma: jnp.ndarray, tail_indices: List[float], transition_point: float, beta: float = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate bivariate samples using a mixture of lognormal and Pareto distributions with a smooth transition.

    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        exp_mu (jnp.ndarray): Expected mean values.
        Sigma (jnp.ndarray): Covariance matrix.
        tail_indices (List[float]): Tail indices for the Pareto distributions.
        transition_point (float): Point at which the transition occurs.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Generated samples for in-degrees and out-degrees.
    """
    # generate normal samples then exponential them to transform them to lognormal
    
    mu = jnp.log(exp_mu) - 0.5 * jnp.diag(Sigma)
    #generate normal samples then exponential them to transform them to lognormal
    L = jnp.linalg.cholesky(Sigma)
    lognormal_samples = jnp.exp(mu + jnp.dot(random.normal(key, (n, 2)), L.T))
    
    #verify lognormal_samples: mathematically:
    print(f"exp_mu: {exp_mu}")  
    key1, key2 = random.split(key)
    scale_1 = exp_mu[0] * (tail_indices[0] - 2) / (tail_indices[0] - 1)
    # scale_2 = exp_mu[1] * (tail_indices[1] - 2) / (tail_indices[1] - 1)
    # print(f"scales: {scale_1}, {scale_2}")
    pareto_samples_1 = pareto_samples(key1, n, tail_indices[0], scale=1)
    pareto_samples_2 = pareto_samples(key2, n, tail_indices[1], scale=1)
    pareto_samples_combined = jnp.column_stack((pareto_samples_1, pareto_samples_2)) 



    def weighting_function(x: jnp.ndarray, transition_point: float, beta: float = 10) -> jnp.ndarray:
        """
        Define a weighting function that smoothly transitions from lognormal to Pareto (logistic).

        Args:
            x (jnp.ndarray): Input samples.
            transition_point (float): Point at which the transition occurs.
            beta (float): Slope of the transition.

        Returns:
            jnp.ndarray: Weighting values.
        """
        return 1 / (1 + jnp.exp(beta *(x - transition_point)))

    weights_log = weighting_function(lognormal_samples, transition_point, beta=10)
    weights_pareto = 1 - weights_log

    #  (exp_mu / jnp.mean(pareto_samples_combined, axis=0))

    mixed_samples = (weights_log * lognormal_samples + weights_pareto * pareto_samples_combined)
    scaled_mix = mixed_samples * (exp_mu / jnp.mean(mixed_samples, axis=0))
    #keep values of where at least one value along axis= 1 is greater than 1
    mixed_samples = jnp.where(jnp.any(mixed_samples > 1, axis=1)[:, None], mixed_samples, 1)
    return scaled_mix[:, 1], scaled_mix[:, 0]


