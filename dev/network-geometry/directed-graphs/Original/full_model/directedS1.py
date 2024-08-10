""".
├── docs
│   ├── _build
│   │   ├── doctrees
│   │   │   ├── environment.pickle
│   │   │   └── index.doctree
│   │   └── html
│   │       ├── genindex.html
│   │       ├── index.html
│   │       ├── objects.inv
│   │       ├── py-modindex.html
│   │       ├── search.html
│   │       ├── searchindex.js
│   │       ├── _sources
│   │       │   └── index.rst.txt
│   │       └── _static
│   │           ├── alabaster.css
│   │           ├── basic.css
│   │           ├── basic.css.jinja
│   │           ├── custom.css
│   │           ├── doctools.js
│   │           ├── documentation_options.js
│   │           ├── documentation_options.js.jinja
│   │           ├── file.png
│   │           ├── github-banner.svg
│   │           ├── language_data.js
│   │           ├── language_data.js.jinja
│   │           ├── minus.png
│   │           ├── plus.png
│   │           ├── pygments.css
│   │           ├── searchtools.js
│   │           └── sphinx_highlight.js
│   ├── conf.py
│   ├── index.rst
│   ├── make.bat
│   ├── Makefile
│   ├── _static
│   └── _templates
├── general
│   └── model.py
├── __init__.py
├── JAX
│   ├── generation
│   │   ├── deg_seq_test_inferred_parameters.json
│   │   ├── __pycache__
│   │   │   └── random_generation_JAX.cpython-310.pyc
│   │   └── random_generation_JAX.py
│   └── infer_parameters
│       ├── degree_sequence.txt
│       ├── deg_seq_test.txt
│       ├── infer_params_JAX.py
│       ├── output_JAX.log
│       ├── output_JAX_small.log
│       ├── __pycache__
│       │   └── infer_params_JAX.cpython-310.pyc
│       ├── run_infer_params_JAX.sh
│       └── run_infer_params_JAX_small.sh
├── Original
│   ├── full_model
│   │   └── directedS1.py
│   ├── generation
│   │   ├── deg_seq_test_inferred_parameters.json
│   │   ├── ensemble_analysis.py
│   │   ├── __init__.py
│   │   ├── logs
│   │   │   ├── output_ensemble_small.log
│   │   │   └── output_generation_small.log
│   │   ├── outputs
│   │   │   ├── default_output_rootname_adjacency_list.json
│   │   │   └── output_ensemble_small_analysis_results.json
│   │   ├── plots
│   │   │   ├── output_ensemble_small_complementary_cumulative_degree_distribution.png
│   │   │   └── output_ensemble_small_in_vs_out_degree_distribution.png
│   │   ├── __pycache__
│   │   │   ├── ensemble_analysis.cpython-310.pyc
│   │   │   ├── ensemble_analysis.cpython-311.pyc
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── random_generation.cpython-310.pyc
│   │   │   └── random_generation.cpython-311.pyc
│   │   ├── random_generation.py
│   │   └── scripts
│   │       ├── ensemble_analysis_small.sh
│   │       └── random_generation_small.sh
│   └── infer_parameters
│       ├── data
│       │   ├── degree_sequence.txt
│       │   └── deg_seq_test.txt
│       ├── deg_seq_test.txt_results.json
│       ├── infer_params_annotated.py
│       ├── infer_params.py
│       ├── __init__.py
│       ├── logs
│       │   ├── infer_params_cumul_prob.log
│       │   ├── output.log
│       │   └── output_small.log
│       ├── plots
│       │   ├── deg_seq_test.txt_ccdf.png
│       │   ├── deg_seq_test.txt_degree_distribution.png
│       │   └── deg_seq_test.txt_in_vs_out_degree.png
│       ├── __pycache__
│       │   ├── infer_params.cpython-310.pyc
│       │   ├── infer_params.cpython-311.pyc
│       │   └── __init__.cpython-310.pyc
│       ├── run_infer_params.sh
│       └── run_infer_params_small.sh
├── __pycache__
│   └── verification.cpython-310.pyc
├── test
│   ├── data
│   │   ├── 7th_graders_layer1_inferred_parameters.json
│   │   ├── deg_seq_test.txt
│   │   ├── network_data.json
│   │   └── TestNetwork_inferred_parameters.json
│   └── src
│       ├── logs
│       │   ├── DirectedS1EnsembleAnalyser
│       │   │   └── output_test_ensemble_analysis_functional.log
│       │   ├── DirectedS1Fitter
│       │   │   ├── output_test_infer_params_functional_fit_from_deg_seq.log
│       │   │   └── test_infer_params_functional_fit_from_file.log
│       │   └── DirectedS1Generator
│       │       └── output_test_generation_fast.log
│       ├── outputs
│       │   ├── 7th_graders_layer1
│       │   │   └── inferred_params.json
│       │   ├── coordinates.csv
│       │   ├── ensemble_analysis_results.json
│       │   ├── generation_data.json
│       │   └── TestNetwork
│       │       ├── coordinates.csv
│       │       ├── ensemble_analysis
│       │       │   ├── figs
│       │       │   │   ├── complementary_cumulative_degree_distribution.png
│       │       │   │   └── in_vs_out_degree_distribution.png
│       │       │   └── results.json
│       │       ├── ensemble_analysis_results.json
│       │       ├── generation_data.json
│       │       └── inferred_params.json
│       ├── __pycache__
│       │   ├── test_JAX.cpython-310-pytest-7.4.4.pyc
│       │   ├── test_JAX.cpython-311-pytest-7.4.0.pyc
│       │   ├── test_original.cpython-310-pytest-7.4.4.pyc
│       │   └── test_original.cpython-311-pytest-7.4.0.pyc
│       ├── test_JAX.py
│       └── test_original.py
├── utils
│   ├── DegSeqAnalysis.py
│   └── __pycache__
│       └── DegSeqAnalysis.cpython-310.pyc
└── verification.py"""

#from general.model import Model as base model in this class
import os
import sys
import logging
from typing import List, Dict, Any, Tuple
from general.model import Model
import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
#Imports for the generation and infer_parameters modules
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'Original', 'generation'))
sys.path.append(os.path.join(parent_dir, 'Original', 'infer_parameters'))
from random_generation import DirectedS1Generator
from infer_params import DirectedS1Fitter
from ensemble_analysis import DirectedS1EnsembleAnalyser

#TODO - Multi modify log file path.
class DirectedS1(Model):
    def __init__(self, seed: int = 0,
                 log_file_path: str = None,
                 verbose: bool = False):
        """
        Initialize the DirectedS1 Model.
        """
        super().__init__(log_file_path, verbose)
        if self.verbose:
            self.logger.info("DirectedS1 Model initialized.")
            
        if self.verbose:
            self.logger.info("Initializing DirectedS1 Generator.")
        self.fitter = DirectedS1Fitter(seed, log_file_path, verbose)
        
        if self.verbose:
            self.logger.info("Initializing DirectedS1 Fitter.")
        self.generator = DirectedS1Generator(seed, log_file_path, verbose)
        
        if self.verbose:
            self.logger.info("Initializing DirectedS1 Ensemble Analyser.")
        self.ensemble_analyser = DirectedS1EnsembleAnalyser(gen = self.generator, verbose = self.verbose, log_file_path = log_file_path)
        
        self.reciprocity = -10
        self.average_local_clustering = -10
        self.network_name = ""
        self.in_degree_sequence = []
        self.out_degree_sequence = []
        self.num_samples = 10
    
    def set_params_generator(self, **kwargs):
        """Modify generator parameters."""
        self.generator.set_params(**kwargs)
        
    def set_params_fitter(self, **kwargs):
        """Modify fitter parameters."""
        self.fitter.set_params(**kwargs)
        
    def set_params_ensemble_analyser(self, **kwargs):
        """Modify ensemble analyser parameters."""
        self.ensemble_analyser.set_params(**kwargs)
    
    def set_original_network_data(self, in_degree_sequence: List[int], out_degree_sequence: List[int], reciprocity: float, average_local_clustering: float):
        """
        Set the network data.
        
        Args:
            in_degree_sequence (List[int]): In-degree sequence.
            out_degree_sequence (List[int]): Out-degree sequence.
            reciprocity (float): Reciprocity of the network.
            average_local_clustering (float): Average local clustering coefficient of the network.
        """
        self.in_degree_sequence = in_degree_sequence
        self.out_degree_sequence = out_degree_sequence
        self.reciprocity = reciprocity
        self.average_local_clustering = average_local_clustering
        
    def fit_generate_analyse_from_file(self, filename: str, reciprocity: float, average_local_clustering: float, network_name: str = "", num_samples: int = 10):
        """
        Fit, generate and analyse the DirectedS1 Model.
        
        Args:
            filename (str): Filename containing the network data.
            reciprocity (float): Reciprocity of the network.
            average_local_clustering (float): Average local clustering coefficient of the network.
            network_name (str): Name of the network (default = "").
            num_samples (int): Number of samples (default = 10).
            
        """

        self.fitter.fit_from_file(filename = filename, reciprocity = reciprocity, average_local_clustering = average_local_clustering, network_name = network_name)
        inferred_params = self.fitter.get_output()
        self.ensemble_analyser.run_analysis(num_samples = num_samples,
                                            hidden_variables_data = inferred_params,
                                            output_rootname = network_name,
                                            plot_degree_distribution = True)
        ensemble_analysis = self.ensemble_analyser.get_output()
        self.evaluate(ensemble_analysis = ensemble_analysis)
        
    def fit_generate_analyse_from_network_data(self, deg_seq: Tuple[List[float], List[float]], reciprocity: float, average_local_clustering: float, network_name: str = "", num_samples: int = 10):
        """
        Fit, generate and analyse the DirectedS1 Model.
        
        Args:
            deg_seq (Tuple[List[float], List[float]]): Tuple of two lists of integers/float representing in-degree and out-degree sequences.
            reciprocity (float): Reciprocity of the network.
            average_local_clustering (float): Average local clustering coefficient of the network.
            network_name (str): Name of the network (default = "").
            num_samples (int): Number of samples (default = 10).
            
        """
        self.set_original_network_data(in_degree_sequence = deg_seq[0], out_degree_sequence = deg_seq[1], reciprocity = reciprocity, average_local_clustering = average_local_clustering)
        self.fitter.fit_from_network_data(deg_seq = deg_seq, reciprocity = reciprocity, average_local_clustering = average_local_clustering, network_name = network_name)
        inferred_params = self.fitter.get_output()
        self.ensemble_analyser.run_analysis(num_samples = num_samples,
                                            hidden_variables_data = inferred_params,
                                            output_rootname = network_name,
                                            save_results = True,
                                            plot_degree_distribution = True)
        ensemble_analysis = self.ensemble_analyser.get_output()
        self.evaluate(ensemble_analysis = ensemble_analysis)
        
    def generate(self, 
                 hidden_variables_filename: str = "",
                 hidden_variables_data: dict = None,    
                 output_rootname: str = "",
                 theta: List[int] = None,
                 save_data: bool = True,
                 save_coordinates: bool = True) -> None:
        """
        Generate a directed network using the hidden variables provided in the hidden_variables_filename, and save the data.
        
        Args:
            hidden_variables_filename (str): The filename of the hidden variables json file.
            output_rootname (str): The rootname of the output files. If not provided, the name of the subdirectory containing the hidden variables file is used.
            theta (List[int], optional): The theta values for the network. Defaults to None.
            save_data (bool, optional): Whether to save the data. Defaults to True.
            save_coordinates (bool, optional): Whether to save the coordinates of the nodes. Defaults to True.
        """
        self.generator.generate(hidden_variables_filename, hidden_variables_data, output_rootname, theta, save_data, save_coordinates)

    
        
    def ensemble_analysis(self, **kwargs):
        """
        Ensemble analysis of the DirectedS1 Model.
        
        Args:
            **kwargs (dict): Parameters
        """
        self.logger.info("Fitting the DirectedS1 Model.")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger.info("DirectedS1 Model fitted.")

        
    def fit(self, **kwargs):
        """
        Fit the DirectedS1 Model.
        
        Args:
            **kwargs (dict): Parameters
        """
        self.logger.info("Fitting the DirectedS1 Model.")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger.info("DirectedS1 Model fitted.")
        
        
    def evaluate(self, ensemble_analysis: Dict[str, Any], save= False) -> None:
        """
        Evaluate the ensemble of networks generated using the DirectedS1 Model.
        
        Args:
            ensemble_analysis (Dict[str, Any]): The ensemble analysis data.
                average_reciprocity (float): Average reciprocity across the ensemble.
                average_clustering (float): Average clustering coefficient across the ensemble.
                variance_reciprocity (float): Variance of reciprocity across the ensemble.
                variance_clustering (float): Variance of clustering coefficient across the ensemble.
                reciprocity_list (List[float]): List of reciprocity values for each generated network.
                clustering_list (List[float]): List of clustering coefficients for each generated network.
                average_in_degree (float): Average in-degree across the ensemble.
                in_degree_sequences (List[List[int]]): List of in-degree sequences for each generated network.
                out_degree_sequences (List[List[int]]): List of out-degree sequences for each generated network.
                
            save (bool, optional): Whether to save the evaluation results. Defaults to False.
        """
        #compare with original input data
        if self.verbose:
            self.logger.info("Comparing the ensemble analysis with the original data.")
            
        #Deviation average reciprocity from self.reciprocity
        average_reciprocity_error = jnp.abs(ensemble_analysis['average_reciprocity'] - self.reciprocity)
        average_clustering_error = jnp.abs(ensemble_analysis['average_clustering'] - self.average_local_clustering)
        average_in_degree_error = jnp.abs(ensemble_analysis['average_in_degree'] - jnp.mean(self.in_degree_sequence))

        evaluation = {
            'average_reciprocity_error': average_reciprocity_error,
            'average_clustering_error': average_clustering_error,
            'average_in_degree_error': average_in_degree_error
        }
        if self.verbose:
            self.logger.info("Evaluation complete.")
            self.logger.info(f"Average reciprocity error: {average_reciprocity_error}")
            self.logger.info(f"Average clustering error: {average_clustering_error}")
            self.logger.info(f"Average in-degree error: {average_in_degree_error}")
        if save:
            self._save_evaluation(evaluation)
    def _save_evaluation(self, evaluation: Dict[str, Any]) -> None:
        """
        Save the evaluation results.
        
        Args:
            evaluation (Dict[str, Any]): The evaluation results.
        """
        pass
if __name__ == "__main__":
    pass

        