import jax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Tuple, List
from numpy import random as np_random

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
    # print(f"exp_mu: {exp_mu}")
    mu = jnp.log(exp_mu) - 0.5 * jnp.diag(Sigma)
    # print(f"mu: {mu}")
    L = jnp.linalg.cholesky(Sigma)
    positive_samples = []
    batch_size = n
    while len(positive_samples) < n:
        key, subkey = jax.random.split(key)
        samples = jnp.exp(mu + jnp.dot(random.normal(subkey, (batch_size, 2)), L.T))
        positive_samples_batch = samples[jnp.all(samples >= 1, axis=1)]
        positive_samples.extend(positive_samples_batch)
        batch_size = n - len(positive_samples)
        # print(f"len(positive_samples_batch): {len(positive_samples_batch)}, batch_size: {batch_size}")
    positive_samples = jnp.array(positive_samples)
    print(f"lognormal mean: {jnp.mean(positive_samples, axis=0)}")
    return positive_samples

def pareto_inverse_cdf(u: jnp.ndarray, scale: float, shape: float) -> jnp.ndarray:
    """
    Inverse CDF (quantile function) of the Pareto distribution.
    
    Args:
        u (jnp.ndarray): Uniform random variable
        scale (float): Scale parameter
        shape (float): Shape parameter
    
    Returns:
        jnp.ndarray: Inverse CDF of the Pareto distribution
    """
    return scale * (u)** (-(1 / shape))

def independent_copula_sample(key: jnp.ndarray, n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate samples from an independent copula.
    
    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
    
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Samples from the bivariate independent copula
    """
    u1 = random.uniform(key, (n,))
    u2 = random.uniform(key, (n,))
    return u1, u2

def comonotonicity_copula_sample(key: jnp.ndarray, n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate samples from a comonotonicity copula.
    
    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
    
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Samples from the bivariate comonotonicity copula
    """
    u = random.uniform(key, (n,))
    return u, u



def gaussian_copula_sample(key: jnp.ndarray, n: int, mean: jnp.ndarray = jnp.array([1,1]), cov_matrix: jnp.ndarray = jnp.array([[1,0],[0,1]])) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate samples from a Gaussian copula.
    
    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        float: Mean of the multivariate normal distribution.
    
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Samples from the bivariate Gaussian cop
    """
    mean = mean
    # Generate multivariate normal samples
    normal_samples = random.multivariate_normal(key, mean, cov_matrix, (n,))
    # Convert to uniform samples using the standard normal CDF
    u1 = norm.cdf(normal_samples[:, 0])
    u2 = norm.cdf(normal_samples[:, 1])
    print(f"u1: {u1}, u2: {u2}")
    return u1, u2

def lognormal_capula_sample(key: jnp.ndarray, n: int, mean: float = jnp.array([0,0]), cov_matrix: jnp.ndarray = jnp.array([[1,0],[0,1]])) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate samples from a lognormal copula.
    
    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        mean (float): Mean of the lognormal distribution.
        cov_matrix (jnp.ndarray): Covariance matrix.
    
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Samples from the bivariate lognormal cop
    """
    mean = mean
    # Generate multivariate normal samples
    lognormal_samples = bivariate_lognormal(key, n, mean, cov_matrix)
    # Convert to uniform samples using the standard normal CDF
    u1 = norm.cdf(lognormal_samples[:, 0])
    u2 = norm.cdf(lognormal_samples[:, 1])
    print(f"creating lognormal capula!")
    plt.figure()
    #plot lognormal samples
    plt.plot(lognormal_samples[:, 0], lognormal_samples[:, 1], 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('lognormal_samples.png')
    return u1, u2
def sample_bivariate_pareto(key: jnp.ndarray, n: int, x_m: float, y_m: float, gamma_1: float, gamma_2: float, cov_matrix: jnp.ndarray = jnp.array([[1,0],[0,1]]), capula: str= "gaussian") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample from a bivariate Pareto distribution with different tails.
    
    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        x_m (float): Scale parameter for the x-axis.
        y_m (float): Scale parameter for the y-axis.
        gamma_1 (float): Shape parameter for the x-axis.
        gamma_2 (float): Shape parameter for the y-axis.
        cov_matrix (jnp.ndarray): Covariance matrix.
        type (str): Type of copula to use.
    
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Samples from the bivariate Pareto distribution
    """
    # Step 1: Generate dependent uniform random variables using Gaussian copula
    if capula == "gaussian":
        print(f"copula is gaussian!")
        u1, u2 = gaussian_copula_sample(key, n, mean=jnp.array([0,0]), cov_matrix=cov_matrix)
    elif capula == "lognormal": 
        print(f"copula is lognormal!")
        u1, u2 = lognormal_capula_sample(key, n, mean=jnp.array([100,100]), cov_matrix=cov_matrix)
    elif capula == "independent":
        print(f"copula is independent!")
        u1, u2 = independent_copula_sample(key, n)
    elif capula == "comonotonicity":
        print(f"copula is comonotonicity!")
        u1, u2 = comonotonicity_copula_sample(key, n)
    print(f"gamma_1: {gamma_1}, gamma_2: {gamma_2}, x_m: {x_m}, y_m: {y_m}")
    # Step 2: Transform uniform variables to Pareto-distributed variables using inverse CDF
    x_samples = pareto_inverse_cdf(u1, x_m, gamma_1)
    y_samples = pareto_inverse_cdf(u2, y_m, gamma_2)
    pareto_samples = jnp.stack([x_samples, y_samples], axis=1)
    
    # print(f"first 100 pareto samples: {pareto_samples[:100]}")
    #loglog plot pareto samples:
    plt.figure()
    plt.plot(x_samples, y_samples, 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('pareto_samples.png')
    return pareto_samples


def bivariate_lognormal_pareto2(key: jnp.ndarray,
                                n: int,
                                exp_mu: jnp.ndarray,
                                Sigma: jnp.ndarray,
                                tail_indices: List[float],
                                transition_point: float,
                                scale_1: float = 1.0,
                                scale_2: float = 1.0,
                                capula: str = "gaussian",
                                capula_cov_matrix: jnp.ndarray = jnp.array([[1,0],[0,1]])) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate bivariate samples using a mixture of lognormal and Pareto distributions with a smooth transition.

    Args:
        key (jnp.ndarray): Random key for reproducibility.
        n (int): Number of samples to generate.
        exp_mu (jnp.ndarray): Expected mean values.
        Sigma (jnp.ndarray): Covariance matrix.
        tail_indices (List[float]): Tail indices for the Pareto distributions.
        transition_point (float): Transition point between the two distributions.
        scale_1 (float): Scale parameter for the first Pareto distribution.
        scale_2 (float): Scale parameter for the second Pareto distribution.
        capula (str): Type of copula to use.
        capula_cov_matrix (jnp.ndarray): Covariance matrix for the copula.


    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Generated samples for in-degrees and out-degrees.
    """
    # generate normal samples then exponential them to transform them to lognormal
    
    lognormal_samples = bivariate_lognormal(key, n, exp_mu, Sigma)
    #verify lognormal_samples: mathematically:
    print(f"exp_mu: {exp_mu}")  
    print(f"scale_1: {scale_1}, scale_2: {scale_2}")
    bivariate_pareto = sample_bivariate_pareto(key, len(lognormal_samples), scale_1, scale_2, tail_indices[0], tail_indices[1], cov_matrix=capula_cov_matrix, capula=capula)
    print(f"shape of bivariate_pareto: {bivariate_pareto.shape}")
    #generate mixed samples by drawing samples from lognormal_samples and pareto_samples_combined
    # mixed_samples = lognormal_samples
    mixed_samples = jnp.where(random.uniform(key, (len(lognormal_samples),1)) < transition_point, lognormal_samples, bivariate_pareto)
    #get all values where value along axis 1 <10, and along axis 2 is greater than 100
    print(f"weird: {mixed_samples[jnp.all(mixed_samples < 10, axis=1) & jnp.all(mixed_samples > 100, axis=1)]}")
    

    mixed_samples = mixed_samples * (exp_mu / jnp.mean(mixed_samples, axis=0))
    #keep values of where at least one value along axis= 1 is greater than 1
    mixed_samples = jnp.floor(mixed_samples)
    return mixed_samples[:, 0], mixed_samples[:, 1]


