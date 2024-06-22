import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict
from stats import Stats

class SyntheticDataGenerator:
    def __init__(self, country: str, year: str):
        """
        Initialize the SyntheticDataGenerator with specified country and year.
        
        Args:
            country (str): The country for which to generate data ('Ecuador' or 'Hungary').
            year (str): The year for which to generate data ('2015' or '2021').
        """
        self.stats = Stats(country, year)
        self.cov_matrix = self.stats.get_cov_matrix()
        self.means = self.stats.get_means()

    def generate_data(self, key: jax.random.PRNGKey, n_samples: int) -> Dict[str, jnp.ndarray]:
        """
        Generate synthetic data based on specified country, year, and number of samples.
        
        Args:
            key (jax.random.PRNGKey): JAX random key for generating random numbers.
            n_samples (int): Number of samples to generate.
        
        Returns:
            Dict[str, jnp.ndarray]: Generated synthetic data.
        """
        cov_matrix = jnp.array([
            self.cov_matrix['kout'],
            self.cov_matrix['kin'],
            self.cov_matrix['sout'],
            self.cov_matrix['sin']
        ])

        mean_vector = jnp.array(self.means)

        # Ensure n_samples is a Python integer
        n_samples = int(n_samples)

        # Generate log-normal synthetic data using a normal function
        log_data = self.generate_multivariate_normal(key, mean_vector, cov_matrix, n_samples)
        data = jnp.exp(log_data)  # Convert log-normal data back to original scale

        synthetic_data = {
            'kout': data[:, 0],
            'kin': data[:, 1],
            'sout': data[:, 2],
            'sin': data[:, 3]
        }

        return synthetic_data

    @staticmethod
    def generate_multivariate_normal(key, mean, cov, size: int):
        """
        Function to generate multivariate normal samples.
        
        Args:
            key: JAX random key for generating random numbers.
            mean: Mean vector for the multivariate normal distribution.
            cov: Covariance matrix for the multivariate normal distribution.
            size: Number of samples to generate.
        
        Returns:
            jnp.ndarray: Generated multivariate normal samples.
        """
        return jax.random.multivariate_normal(key, mean, cov, (size,))
