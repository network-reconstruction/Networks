"""
Statistics, Synthetic Data Generator and Plotter for Firm-Level Production Networks
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

from typing import Dict


class Stats:
    def __init__(self, country: str, year: str):
        """
        Initialize the Stats with specified country and year.
        
        Args:
            country (str): The country for which to provide stats ('Ecuador' or 'Hungary').
            year (str): The year for which to provide stats ('2015' or '2021').
        """
        if country not in ['Ecuador', 'Hungary'] or year not in ['2015', '2021']:
            raise ValueError("Invalid country or year. Country should be 'Ecuador' or 'Hungary' and year should be '2015' or '2021'.")
        
        self.country = country
        self.year = year
        
        self.stats = self._load_stats()

    def _load_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Load the statistical properties for the specified country and year.
        
        Returns:
            Dict[str, Dict[str, float]]: A dictionary containing the statistical properties.
        """
        stats = {
            'Ecuador': {
                '2015': {
                    'cov_matrix': {
                        'kout': [2.89, 1.37, 2.12, 2.39],
                        'kin': [1.37, 2.06, 1.98, 3.07],
                        'sout': [2.12, 1.98, 6.97, 4.34],
                        'sin': [2.39, 3.07, 4.34, 7.63],
                    },
                    'means': [2.07, 3.19, 11.08, 10.53]
                }
            },
            'Hungary': {
                '2021': {
                    'cov_matrix': {
                        'kout': [2.75, 1.14, 2.19, 1.92],
                        'kin': [1.14, 1.65, 1.20, 2.19],
                        'sout': [2.19, 1.20, 5.90, 2.98],
                        'sin': [1.92, 2.19, 2.98, 5.22],
                    },
                    'means': [2.04, 3.30, 9.38, 9.64]
                }
            }
        }
        return stats[self.country][self.year]

    def get_cov_matrix(self) -> Dict[str, float]:
        """
        Get the covariance matrix.
        
        Returns:
            Dict[str, float]: The covariance matrix of the specified country and year.
        """
        return self.stats['cov_matrix']

    def get_means(self) -> Dict[str, float]:
        """
        Get the means.
        
        Returns:
            Dict[str, float]: The means of the specified country and year.
        """
        return self.stats['means']
