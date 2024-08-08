# file: test_original.py

import json
import os
import sys
import pytest
# import logging

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
    # print("GeneratingDirectedS1 is available in random_generation")


    # Test ensemble_analysis functionality
    ensemble_analysis = imported_modules['ensemble_analysis']
    assert hasattr(ensemble_analysis, 'GraphEnsembleAnalysis'), "GraphEnsembleAnalysis not found in ensemble_analysis"
    # print("GraphEnsembleAnalysis is available in ensemble_analysis")
    
    # Test infer_params functionality
    infer_params = imported_modules['infer_params']
    assert hasattr(infer_params, 'FittingDirectedS1'), "FittingDirectedS1 not found in infer_params"
    # print("FittingDirectedS1 is available in infer_params")
    
    
@pytest.fixture(scope='module')
def network_data():
    """
    Fixture to import and load network data from JSON file.
    """
    print("Importing network data from ../data/network_data.json")
    with open('../data/network_data.json', 'r') as file:
        data = json.load(file)

    return data

@pytest.fixture(scope='module')
def deg_seq_filename():
    """
    Fixture to return the filename of the degree sequence file.
    """
    return '../data/deg_seq_test.txt'

#quick tests are tests for functionality only, not accuracy
@pytest.fixture(scope='module')
def quick_fit_model():
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
    #              log_file_path: str = "logs/output.log")
    return FittingDirectedS1(verbose = True,
                             KAPPA_MAX_NB_ITER_CONV = 10,
                             EXP_CLUST_NB_INTEGRATION_MC_STEPS = 10,
                             log_file_path = "logs/FittingDirectedS1/test_infer_params_quick_fit_from_file.log")

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

    # print("Data imported successfully")

def test_infer_params_quick_fit_from_file(setup_paths, quick_fit_model, deg_seq_filename):
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
    
    quick_fit_model.fit_from_file(filename = deg_seq_filename,
                        reciprocity = 0.5,
                        average_local_clustering = 0.5,
                        network_name = "TestNetwork",
                        verbose = True)
    assert os.path.exists("outputs/TestNetwork/inferred_params.json"), "Inferred parameters file not found"
    # print("FittingDirectedS1 class fits from file successfully")

        
def test_infer_params_quick_fit_from_deg_seq(setup_paths, quick_fit_model, network_data):
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
    
    quick_fit_model.modify_log_file_path("logs/FittingDirectedS1/output_test_infer_params_quick_fit_from_deg_seq.log")
    quick_fit_model.fit_from_deg_seq(deg_seq = deg_seq,
                                reciprocity = network_data[network]['reciprocity'],
                                average_local_clustering = network_data[network]['average_clustering'],
                                network_name = network,
                                verbose = True)
    
    #assert that the output name network name is correct
    assert os.path.exists(f"outputs/{network}/inferred_params.json"), "Inferred parameters file not found"
    # print("FittingDirectedS1 class fits from degree sequence successfully")

@pytest.fixture(scope='module')
def quick_random_generation_model():
    """
    Fixture to instantiate the GeneratingDirectedS1 class converging fast just to test the functionality runs.
    """
    from random_generation import GeneratingDirectedS1
    # def __init__(self,
    #              seed: int = 0,
    #              verbose: bool = False,
    #              log_file_path: str = "logs/GeneratingDirectedS1/output.log"):
    return GeneratingDirectedS1(verbose=True,
                             log_file_path = "logs/GeneratingDirectedS1/output_test_generation_fast.log")

def test_random_generation_quick_generate(setup_paths, quick_random_generation_model):
    """
    Test that the GeneratingDirectedS1 class can be instantiated and generate works as expected.
    """
    # def generate(self, 
    #              hidden_variables_filename,
    #              output_rootname: str = "",
    #              theta: List[int] = None,
    #              save_coordinates: bool = False):
    quick_random_generation_model.generate(hidden_variables_filename = "outputs/TestNetwork/inferred_params.json",
                                           save_coordinates = True)
    assert os.path.exists("outputs/TestNetwork/generation_data.json"), "Generation data file not found"

#TODO test the ensemble analysis
#TODO test whole pipeline with accuracy
#TODO large test on whole pipeline

if __name__ == '__main__':
    # Test all
    exit_code = pytest.main(['-v', '-s'])
    # exit_code = pytest.main(['-v', '-s', 'test_original.py::test_infer_params_fit_from_deg_seq'])
    sys.exit(exit_code)