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

def bivariate_lognormal_pareto(key: jnp.ndarray, n: int, exp_mu: jnp.ndarray, Sigma: jnp.ndarray, tail_indices: List[float], weights: List[float]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate bivariate samples using a mixture of lognormal and Pareto distributions.

    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        exp_mu (jnp.ndarray): Expected mean values.
        Sigma (jnp.ndarray): Covariance matrix.
        tail_indices (List[float]): Tail indices for the Pareto distributions.
        weights (List[float]): Weights for the mixture of lognormal and Pareto samples.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Generated samples for in-degrees and out-degrees.
    """
    mu = jnp.log(exp_mu) - 0.5 * jnp.diag(Sigma)
    L = jnp.linalg.cholesky(Sigma)
    lognormal_samples = jnp.exp(mu + jnp.dot(random.normal(key, (n, 2)), L.T))

    key1, key2 = random.split(key)
    
    pareto_samples_1 = pareto_samples(key1, n, tail_indices[0])
    pareto_samples_2 = pareto_samples(key2, n, tail_indices[1])
    pareto_samples_combined = jnp.column_stack((pareto_samples_1, pareto_samples_2))

    mixed_samples = (weights[0] * lognormal_samples + weights[1] * pareto_samples_combined)

    return mixed_samples[:, 1], mixed_samples[:, 0]

def bivariate_lognormal_pareto2(key: jnp.ndarray, n: int, exp_mu: jnp.ndarray, Sigma: jnp.ndarray, tail_indices: List[float], transition_point: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    mu = jnp.log(exp_mu) - 0.5 * jnp.diag(Sigma)
    L = jnp.linalg.cholesky(Sigma)
    lognormal_samples = jnp.exp(mu + jnp.dot(random.normal(key, (n, 2)), L.T))

    key1, key2 = random.split(key)
    
    pareto_samples_1 = pareto_samples(key1, n, tail_indices[0])
    pareto_samples_2 = pareto_samples(key2, n, tail_indices[1])
    pareto_samples_combined = jnp.column_stack((pareto_samples_1, pareto_samples_2))

    def weighting_function(x: jnp.ndarray, transition_point: float) -> jnp.ndarray:
        """
        Define a weighting function that smoothly transitions from lognormal to Pareto.

        Args:
            x (jnp.ndarray): Input samples.
            transition_point (float): Point at which the transition occurs.

        Returns:
            jnp.ndarray: Weighting values.
        """
        return 1 / (1 + jnp.exp((x - transition_point) / transition_point))

    weights_log = weighting_function(lognormal_samples, transition_point)
    weights_pareto = 1 - weights_log

    scaled_pareto_samples = pareto_samples_combined * (exp_mu / jnp.mean(pareto_samples_combined, axis=0))

    mixed_samples = (weights_log * lognormal_samples + weights_pareto * scaled_pareto_samples)

    return mixed_samples[:, 1], mixed_samples[:, 0]

def frechet_samples(key: jnp.ndarray, n: int, alpha: float, scale: float = 1.0) -> jnp.ndarray:
    """
    Generate Frechet samples.

    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        alpha (float): Shape parameter of the Frechet distribution.
        scale (float): Scale parameter of the Frechet distribution.

    Returns:
        jnp.ndarray: Generated Frechet samples.
    """
    return scale / (random.uniform(key, (n,)) ** (1 / alpha))

def gumbel_samples(key: jnp.ndarray, n: int, loc: float = 0.0, scale: float = 1.0) -> jnp.ndarray:
    """
    Generate Gumbel samples.

    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        loc (float): Location parameter of the Gumbel distribution.
        scale (float): Scale parameter of the Gumbel distribution.

    Returns:
        jnp.ndarray: Generated Gumbel samples.
    """
    return loc - scale * jnp.log(-jnp.log(random.uniform(key, (n,))))

def bivariate_frechet(key: jnp.ndarray, n: int, alpha: List[float], scale: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate bivariate Frechet samples.

    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        alpha (List[float]): Shape parameters for the Frechet distributions.
        scale (float): Scale parameter of the Frechet distribution.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Generated samples for in-degrees and out-degrees.
    """
    key1, key2 = random.split(key)
    frechet_samples_1 = frechet_samples(key1, n, alpha[0], scale)
    frechet_samples_2 = frechet_samples(key2, n, alpha[1], scale)
    return frechet_samples_1, frechet_samples_2

def bivariate_gumbel(key: jnp.ndarray, n: int, loc: List[float], scale: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate bivariate Gumbel samples.

    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        loc (List[float]): Location parameters for the Gumbel distributions.
        scale (float): Scale parameter of the Gumbel distribution.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Generated samples for in-degrees and out-degrees.
    """
    key1, key2 = random.split(key)
    gumbel_samples_1 = gumbel_samples(key1, n, loc[0], scale)
    gumbel_samples_2 = gumbel_samples(key2, n, loc[1], scale)
    return gumbel_samples_1, gumbel_samples_2
