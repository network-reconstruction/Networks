# file: test_original.py

import json
import os
import sys
import pytest
import logging
# Organization:
"""
12 directories, 43 files
tree
.
├── __init__.py
├── JAX
│   ├── generation
│   │   ├── deg_seq_test_inferred_parameters.json
│   │   └── random_generation_JAX.py
│   └── infer_parameters
│       ├── degree_sequence.txt
│       ├── deg_seq_test.txt
│       ├── infer_params_JAX.py
│       ├── output_JAX.log
│       ├── output_JAX_small.log
│       ├── run_infer_params_JAX.sh
│       └── run_infer_params_JAX_small.sh
├── Original
│   ├── generation
│   │   ├── default_output_rootname_adjacency_list.json
│   │   ├── deg_seq_test_inferred_parameters.json
│   │   ├── ensemble_analysis.py
│   │   ├── ensemble_analysis_small.sh
│   │   ├── __init__.py
│   │   ├── output_ensemble_small_analysis_results.json
│   │   ├── output_ensemble_small_complementary_cumulative_degree_distribution.png
│   │   ├── output_ensemble_small_in_vs_out_degree_distribution.png
│   │   ├── output_ensemble_small.log
│   │   ├── output_generation_small.log
│   │   ├── __pycache__
│   │   │   └── random_generation.cpython-310.pyc
│   │   ├── random_generation.py
│   │   └── random_generation_small.sh
│   └── infer_parameters
│       ├── degree_sequence.txt
│       ├── deg_seq_test.txt
│       ├── deg_seq_test.txt_ccdf.png
│       ├── deg_seq_test.txt_degree_distribution.png
│       ├── deg_seq_test.txt_in_vs_out_degree.png
│       ├── deg_seq_test.txt_results.json
│       ├── infer_params_cumul_prob.log
│       ├── infer_params.py
│       ├── _inferred_parameters.txt
│       ├── __init__.py
│       ├── output.log
│       ├── output_small.log
│       ├── read.ipynb
│       ├── run_infer_params.sh
│       ├── run_infer_params_small.sh
│       └── test.ipynb
├── test
│   ├── data
│   │   └── network_data.json
│   └── src
│       ├── test_JAX.py
│       └── test_original.py
├── utils
│   └── DegSeqAnalysis.py
└── verification.py
"""

@pytest.fixture(scope='module')
def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    parent_dir = os.path.dirname(parent_dir)

    # Append the necessary paths to sys.path
    sys.path.append(parent_dir)
    sys.path.append(os.path.join(parent_dir, 'Original', 'generation'))
    sys.path.append(os.path.join(parent_dir, 'Original', 'infer_parameters'))

@pytest.fixture(scope='module')
def imported_modules(setup_paths):
    
    import random_generation
    import ensemble_analysis
    import infer_params
    
    return {
        'random_generation': random_generation,
        'ensemble_analysis': ensemble_analysis,
        'infer_params': infer_params
    }
    
def test_imports(imported_modules):
    """
    Simply check that the imports were successful
    """
    assert 'random_generation' in imported_modules
    assert 'ensemble_analysis' in imported_modules
    assert 'infer_params' in imported_modules
        
    assert True, "All modules imported successfully"
    
    # Test random_generation functionality
    random_generation = imported_modules['random_generation']
    assert hasattr(random_generation, 'GeneratingDirectedS1'), "GeneratingDirectedS1 not found in random_generation"
    logging.info("GeneratingDirectedS1 is available in random_generation")


    # Test ensemble_analysis functionality
    ensemble_analysis = imported_modules['ensemble_analysis']
    assert hasattr(ensemble_analysis, 'GraphEnsembleAnalysis'), "GraphEnsembleAnalysis not found in ensemble_analysis"
    logging.info("GraphEnsembleAnalysis is available in ensemble_analysis")
    
    # Test infer_params functionality
    infer_params = imported_modules['infer_params']
    assert hasattr(infer_params, 'FittingDirectedS1'), "FittingDirectedS1 not found in infer_params"
    logging.info("FittingDirectedS1 is available in infer_params")
    
    
@pytest.fixture(scope='module')
def network_data():
    """
    Fixture to import and load network data from JSON file.
    """
    logging.info("Importing network data from ../data/network_data.json")
    with open('../data/network_data.json', 'r') as file:
        data = json.load(file)

    return data

@pytest.fixture(scope='module')
def deg_seq_filename():
    """
    Fixture to return the filename of the degree sequence file.
    """
    return '../data/deg_seq_test.txt'

@pytest.fixture(scope='module')
def fast_fit_model():
    """
    Fixture to instantiate the FittingDirectedS1 class converging fast just to test the functionality runs.
    """
    from infer_params import FittingDirectedS1
    # def __init__(self, 
    #              seed: int = 0, 
    #              verbose: bool = False, 
    #              KAPPA_MAX_NB_ITER_CONV: int = 10, 
    #              EXP_CLUST_NB_INTEGRATION_MC_STEPS: int = 10, 
    #              NUMERICAL_CONVERGENCE_THRESHOLD_1: float = 1e-2, 
    #              NUMERICAL_CONVERGENCE_THRESHOLD_2: float = 1e-2,
    #              log_file: str = "output.log")
    return FittingDirectedS1(verbose = True,
                             KAPPA_MAX_NB_ITER_CONV = 10,
                             EXP_CLUST_NB_INTEGRATION_MC_STEPS = 10,
                             log_file = "output_test_fast.log")

def test_import_data(network_data):
    """
    Test that the data file for testing can be imported successfully.
    """
    for network in network_data:
        assert 'reciprocity' in network_data[network]
        assert 'average_clustering' in network_data[network]
        assert 'in_degree_sequence' in network_data[network]
        assert 'out_degree_sequence' in network_data[network]
        assert 'average_in_degree' in network_data[network]

    assert True, "Data imported successfully"

def test_infer_params_fit_from_file(setup_paths, fast_fit_model, deg_seq_filename):
    """
    Test that the FittingDirectedS1 class can be instantiated and fit works as expected from deg_seq_filename
    """
    #TODO: TEST (Works on script)
    # def fit_from_file(self, 
    #               filename: str,
    #               reciprocity: float,
    #               average_local_clustering: float,
    #               network_name: str = "", 
    #               verbose: bool = False
    #               ) -> None:
    
    fast_fit_model.fit_from_file(filename = deg_seq_filename,
                        reciprocity = 0.5,
                        average_local_clustering = 0.5,
                        network_name = "TestNetwork",
                        verbose = True)
    assert os.path.exists("TestNetwork_inferred_parameters.json"), "Inferred parameters file not found"

        
def test_infer_params_fit_from_deg_seq(setup_paths, fast_fit_model, network_data):
    """
    Test that the FittingDirectedS1 class can be instantiated and fit works as expected.
    """
    #TODO: TEST (Works on script)
    # take the first network in network data
    # def fit_from_deg_seq(self,
    #                  deg_seq: Tuple[List[float], List[float]],
    #                  reciprocity: float,
    #                  average_local_clustering: float,
    #                  network_name: str = "",
    #                  verbose: bool = False
    #                  ) -> None:
    network = list(network_data.keys())[0]
    deg_seq = (network_data[network]['in_degree_sequence'], network_data[network]['out_degree_sequence'])
    
    #shouldn't happen theoretically
    assert len(deg_seq[0]) == len(deg_seq[1]), "TEST DATA ERROR: In and out degree sequences are not of the same length" 
        
    fast_fit_model.fit_from_deg_seq(deg_seq = deg_seq,
                                reciprocity = network_data[network]['reciprocity'],
                                average_local_clustering = network_data[network]['average_clustering'],
                                network_name = network,
                                verbose = True)
    
    #assert that the output name network name is correct
    assert os.path.exists(f"{network}_inferred_parameters.json"), "Inferred parameters file not found"
    assert True, "FittingDirectedS1 class instantiated successfully"

pytest.fixture(scope='module')
def random_generation_model():
    """
    Fixture to instantiate the GeneratingDirectedS1 class converging fast just to test the functionality runs.
    """
    from random_generation import GeneratingDirectedS1
    # def __init__(self, 
    #              seed: int = 0, 
    #              verbose: bool = False, 
    #              KAPPA_MAX_NB_ITER_CONV: int = 10, 
    #              EXP_CLUST_NB_INTEGRATION_MC_STEPS: int = 10, 
    #              NUMERICAL_CONVERGENCE_THRESHOLD_1: float = 1e-2, 
    #              NUMERICAL_CONVERGENCE_THRESHOLD_2: float = 1e-2,
    #              log_file: str = "output.log")
    return GeneratingDirectedS1(verbose = True,
                             KAPPA_MAX_NB_ITER_CONV = 10,
                             EXP_CLUST_NB_INTEGRATION_MC_STEPS = 10,
                             log_file = "output_test_fast.log")

if __name__ == '__main__':
    # Test all
    exit_code = pytest.main(['-v', '-s'])
    # test just test_infer_params_fit_from_deg_seq
    # exit_code = pytest.main(['-v', '-s', 'test_original.py::test_infer_params_fit_from_deg_seq'])
    sys.exit(exit_code)