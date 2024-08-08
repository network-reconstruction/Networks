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
from typing import List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from general.model import Model

class DirectedS1(Model):
    def __init__(self, log_file_path: str = None, verbose: bool = False, **kwargs):
        """
        Initialize the DirectedS1 Model.
        """
        super().__init__(log_file_path, verbose, **kwargs)
        self.logger.info("DirectedS1 Model initialized.")
        
    def generate(self, **kwargs):
        """
        Generate the DirectedS1 Model.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters.
        """
        self.logger.info("Generating DirectedS1 Model.")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger.info("DirectedS1 Model generated.")
        
    def infer_parameters(self, **kwargs):
        """
        Infer the parameters of the DirectedS1 Model.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters.
        """
        self.logger.info("Inferring parameters of the DirectedS1 Model.")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger.info("Parameters of the DirectedS1 Model inferred.")
        
    def ensemble_analysis(self, **kwargs):
        """
        Analyse the ensemble of the DirectedS1 Model.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters.
        """
        self.logger.info("Analysing the ensemble of the DirectedS1 Model.")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger.info("Ensemble of the DirectedS1 Model analysed.")
        
    def fit(self, **kwargs):
        """
        Fit the DirectedS1 Model.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters.
        """
        self.logger.info("Fitting the DirectedS1 Model.")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger.info("DirectedS1 Model fitted.")
        
    def test(self, **kwargs):
        """
        Test the DirectedS1 Model.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters.
        """
        self.logger.info("Testing the DirectedS1 Model.")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger.info("DirectedS1 Model tested.")
        
    def evaluate(self, **kwargs):
        """
        Evaluate the DirectedS1 Model.
        Parameters
        ----------
        **kwargs : dict
            Parameters
        """
        self.logger.info("Evaluating the DirectedS1 Model.")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger.info("DirectedS1 Model evaluated.")