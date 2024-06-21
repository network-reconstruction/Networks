"""
Synthetic Data Generator and Plotter for Firm-Level Production Networks
Using JAX and Matplotlib

Author: Yi Yao Tan
Date: 21-06-2024

Description:
This module contains classes for generating synthetic data and plotting
firm-level production networks for specified countries and years using JAX.

Based on:
Bacilieri, A., Borsos, A., Astudillo-Estevez, P., & Lafond, F. (2023). 
Firm-level production networks: what do we (really) know? INET Oxford 
Working Paper No. 2023-08.
"""

__author__ = "Yi Yao Tan"
__date__ = "21-06-2024"


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Tuple

class SyntheticDataGenerator:
    def __init__(self, country: str, year: str):
        """
        Initialize the SyntheticDataGenerator with specified country and year.
        
        Args:
            country (str): The country for which to generate data ('Ecuador' or 'Hungary').
            year (str): The year for which to generate data ('2015' or '2021').
        """
        if country not in ['Ecuador', 'Hungary'] or year not in ['2015', '2021']:
            raise ValueError("Invalid country or year. Country should be 'Ecuador' or 'Hungary' and year should be '2015' or '2021'.")
        
        self.country = country
        self.year = year
        
        self.stats = {
            'Ecuador': {
                '2015': {
                    'in_degree': {'mean': 12.7, 'std_dev': 20.3, 'min': 1, 'max': 370},
                    'out_degree': {'mean': 13.2, 'std_dev': 21.4, 'min': 1, 'max': 400},
                    'network_sales': {'mean': 2.4e6, 'std_dev': 7.3e6, 'min': 1e3, 'max': 80e6},
                    'network_expenses': {'mean': 2.2e6, 'std_dev': 6.9e6, 'min': 1e3, 'max': 75e6},
                    'correlations': {
                        'suppliers_expenses': {'correlation': 0.76, 'slope': 1.54},
                        'customers_sales': {'correlation': 0.72, 'slope': 1.05}
                    }
                }
            },
            'Hungary': {
                '2021': {
                    'in_degree': {'mean': 14.3, 'std_dev': 22.8, 'min': 1, 'max': 420},
                    'out_degree': {'mean': 15.1, 'std_dev': 24.3, 'min': 1, 'max': 450},
                    'network_sales': {'mean': 2.5e6, 'std_dev': 7.5e6, 'min': 1e3, 'max': 85e6},
                    'network_expenses': {'mean': 2.3e6, 'std_dev': 7.1e6, 'min': 1e3, 'max': 78e6},
                    'correlations': {
                        'suppliers_expenses': {'correlation': 0.78, 'slope': 1.35},
                        'customers_sales': {'correlation': 0.74, 'slope': 1.05}
                    }
                }
            }
        }
    
    def generate_correlated_data(self, key: jax.random.PRNGKey, x_mean: float, x_std: float, y_mean: float, y_std: float, correlation: float, size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate correlated data based on specified means, standard deviations, and correlation.
        
        Args:
            key (jax.random.PRNGKey): JAX random key for generating random numbers.
            x_mean (float): Mean of the x variable.
            x_std (float): Standard deviation of the x variable.
            y_mean (float): Mean of the y variable.
            y_std (float): Standard deviation of the y variable.
            correlation (float): Correlation coefficient between x and y variables.
            size (int): Number of samples to generate.
        
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: Generated x and y data.
        """
        x = jax.random.normal(key, (size,)) * x_std + x_mean
        y = jax.random.normal(key, (size,)) * y_std + y_mean
        y = correlation * x + jnp.sqrt(1 - correlation**2) * y
        return x, y
    
    def generate(self, key: jax.random.PRNGKey, n_samples: int) -> Dict[str, jnp.ndarray]:
        """
        Generate synthetic data based on specified country, year, and number of samples.
        
        Args:
            key (jax.random.PRNGKey): JAX random key for generating random numbers.
            n_samples (int): Number of samples to generate.
        
        Returns:
            Dict[str, jnp.ndarray]: Generated synthetic data.
        """
        stats = self.stats[self.country][self.year]
        
        in_degree = jnp.clip(jax.random.normal(key, (n_samples,)) * stats['in_degree']['std_dev'] + stats['in_degree']['mean'],
                             stats['in_degree']['min'], stats['in_degree']['max'])
        out_degree = jnp.clip(jax.random.normal(key, (n_samples,)) * stats['out_degree']['std_dev'] + stats['out_degree']['mean'],
                              stats['out_degree']['min'], stats['out_degree']['max'])
        network_sales = jnp.clip(jax.random.normal(key, (n_samples,)) * stats['network_sales']['std_dev'] + stats['network_sales']['mean'],
                                 stats['network_sales']['min'], stats['network_sales']['max'])
        network_expenses = jnp.clip(jax.random.normal(key, (n_samples,)) * stats['network_expenses']['std_dev'] + stats['network_expenses']['mean'],
                                    stats['network_expenses']['min'], stats['network_expenses']['max'])
        
        suppliers, expenses = self.generate_correlated_data(key, stats['in_degree']['mean'], stats['in_degree']['std_dev'],
                                                            stats['network_expenses']['mean'], stats['network_expenses']['std_dev'],
                                                            stats['correlations']['suppliers_expenses']['correlation'], n_samples)
        customers, sales = self.generate_correlated_data(key, stats['out_degree']['mean'], stats['out_degree']['std_dev'],
                                                         stats['network_sales']['mean'], stats['network_sales']['std_dev'],
                                                         stats['correlations']['customers_sales']['correlation'], n_samples)
        
        synthetic_data = {
            'In-Degree': in_degree,
            'Out-Degree': out_degree,
            'Network Sales': network_sales,
            'Network Expenses': network_expenses,
            'Suppliers': suppliers,
            'Expenses': expenses,
            'Customers': customers,
            'Sales': sales
        }
        
        return synthetic_data



# Example usage

