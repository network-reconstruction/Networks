# file: test_JAX.py
"""
Test the JAX code for the Directed S1 model, converted to Python.
"""
import json
import os
import sys
import pytest
# import logging

#TODO make the tests more rigorous checking the data.
#TODO Make testing into class.
@pytest.fixture(scope='module')
def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    parent_dir = os.path.dirname(parent_dir)
    # Append the necessary paths to sys.path
    sys.path.append(parent_dir)
    sys.path.append(os.path.join(parent_dir, 'JAX', 'generation'))
    sys.path.append(os.path.join(parent_dir, 'JAX', 'infer_parameters'))
    sys.path.append(os.path.join(parent_dir, 'JAX', 'full_model'))
    
@pytest.fixture(scope='module')
def imported_modules(setup_paths):
    
    import random_generation
    import ensemble_analysis
    import infer_params
    import directedS1
    
    return {
        'random_generation': random_generation,
        'ensemble_analysis': ensemble_analysis,
        'infer_params': infer_params,
        'directedS1': directedS1
    }
    
def test_imports(imported_modules):
    """
    Simply check that the imports were successful
    """
    assert 'random_generation' in imported_modules
    assert 'ensemble_analysis' in imported_modules
    assert 'infer_params' in imported_modules
    assert 'directedS1' in imported_modules
        
    assert True, "All modules imported successfully"
    
    # Test random_generation functionality
    random_generation = imported_modules['random_generation']
    assert hasattr(random_generation, 'DirectedS1Generator'), "DirectedS1Generator not found in random_generation"
    # print("DirectedS1Generator is available in random_generation")


    # Test ensemble_analysis functionality
    ensemble_analysis = imported_modules['ensemble_analysis']
    assert hasattr(ensemble_analysis, 'DirectedS1EnsembleAnalyser'), "DirectedS1EnsembleAnalyser not found in ensemble_analysis"
    # print("DirectedS1EnsembleAnalyser is available in ensemble_analysis")
    
    # Test infer_params functionality
    infer_params = imported_modules['infer_params']
    assert hasattr(infer_params, 'DirectedS1Fitter'), "DirectedS1Fitter not found in infer_params"
    # print("DirectedS1Fitter is available in infer_params")
    
    directedS1 = imported_modules['directedS1']
    assert hasattr(directedS1, 'DirectedS1'), "DirectedS1 not found in directedS1"
    # print("DirectedS1 is available in directedS1")
    
    
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

#functional tests are tests for functionality only, not accuracy
@pytest.fixture(scope='module')
def functional_fit_model():
    """
    Fixture to instantiate the DirectedS1Fitter class converging fast just to test the functionality runs.
    """
    from infer_params import DirectedS1Fitter
    """
    def __init__(self, 
                 seed: int = 0, 
                 verbose: bool = False, 
                 KAPPA_MAX_NB_ITER_CONV: int = 10, 
                 EXP_CLUST_NB_INTEGRATION_MC_STEPS: int = 10, 
                 NUMERICAL_CONVERGENCE_THRESHOLD_1: float = 1e-2, 
                 NUMERICAL_CONVERGENCE_THRESHOLD_2: float = 1e-2,
                 log_file_path: str = "logs/output.log")
    """
    return DirectedS1Fitter(verbose = True,
                             KAPPA_MAX_NB_ITER_CONV = 10,
                             EXP_CLUST_NB_INTEGRATION_MC_STEPS = 10,
                             log_file_path = "logs/DirectedS1Fitter/test_infer_params_functional_fit_from_file.log")

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

def test_infer_params_functional_fit_from_file(setup_paths, functional_fit_model, deg_seq_filename):
    """
    Test that the DirectedS1Fitter class can be instantiated and fit works as expected from deg_seq_filename
    """
    #TODO: TEST (Works on script)
    """
    def fit_from_file(self, 
                  filename: str,
                  reciprocity: float,
                  average_local_clustering: float,
                  network_name: str = "", 
                  ) -> None:
    """
    
    functional_fit_model.fit_from_file(filename = deg_seq_filename,
                                       reciprocity = 0.5,
                                       average_local_clustering = 0.5,
                                       network_name = "TestNetwork")
    assert os.path.exists("outputs/TestNetwork/inferred_params.json"), "Inferred parameters file not found"

    """Check if data is saved as expected
     data = {
                "beta": float,
                "mu": float,
                "nu": float,
                "R": float,
                "inferred_kappas": [
                    {
                        "in_deg": int,
                        "out_deg": int,
                        "kappa_in": float,
                        "kappa_out": float
                    },
                    ...
                ]
                "flags":
                {
                    "clustering_cvg": bool,
                    "kappa_cvg": List,
                    "beta_min_hit": bool, 
                    "beta_max_hit": bool
                }
                "seed": int
            }
    """
    with open(f"outputs/TestNetwork/inferred_params.json", 'r') as file:
        data = json.load(file)
    assert 'beta' in data and isinstance(data['beta'], float)
    assert 'mu' in data and isinstance(data['mu'], float)
    assert 'nu' in data and isinstance(data['nu'], float)
    assert 'R' in data and isinstance(data['R'], float)
    assert 'inferred_kappas' in data and isinstance(data['inferred_kappas'], list)
    assert 'flags' in data and isinstance(data['flags'], dict)
    
    

        
def test_infer_params_functional_fit_from_network_data(setup_paths, functional_fit_model, network_data):
    """
    Test that the DirectedS1Fitter class can be instantiated and fit works as expected.
    """
    #TODO: TEST (Works on script)
    # take the first network in network data
    """
    def fit_from_network_data(self,
                        deg_seq: Tuple[List[float], List[float]],
                        reciprocity: float,
                        average_local_clustering: float,
                        network_name: str = ""
                        ) -> None:
    """
    network = list(network_data.keys())[0]
    deg_seq = (network_data[network]['in_degree_sequence'], network_data[network]['out_degree_sequence'])
    
    #shouldn't happen theoretically
    assert len(deg_seq[0]) == len(deg_seq[1]), "TEST DATA ERROR: In and out degree sequences are not of the same length" 
    
    functional_fit_model.modify_log_file_path("logs/DirectedS1Fitter/test_infer_params_functional_fit_from_network_data.log")
    functional_fit_model.set_params(debug = True)
    functional_fit_model.fit_from_network_data(deg_seq = deg_seq,
                                                reciprocity = network_data[network]['reciprocity'],
                                                average_local_clustering = network_data[network]['average_clustering'],
                                                network_name = network)
    
    assert os.path.exists(f"outputs/{network}/inferred_params.json"), "Inferred parameters file not found"
    
    """
    Check if data is saved as expected, data is also hidden_variables for generation.
    data = {
                "beta": float,
                "mu": float,
                "nu": float,
                "R": float,
                "inferred_kappas": [
                    {
                        "in_deg": int,
                        "out_deg": int,
                        "kappa_in": float,
                        "kappa_out": float
                    },
                    ...
                ],
                
                "flags":
                {
                    "clustering_cvg": bool,
                    "kappa_cvg": List,
                    "beta_min_hit": bool, 
                    "beta_max_hit": bool
                },
                
                "model_params":
                {
                    "seed": seed,
                    "verbose": verbose,
                    "KAPPA_MAX_NB_ITER_CONV": KAPPA_MAX_NB_ITER_CONV, 
                    "EXP_CLUST_NB_INTEGRATION_MC_STEPS": EXP_CLUST_NB_INTEGRATION_MC_STEPS, 
                    "NUMERICAL_CONVERGENCE_THRESHOLD_1": NUMERICAL_CONVERGENCE_THRESHOLD_1, 
                    "NUMERICAL_CONVERGENCE_THRESHOLD_2": NUMERICAL_CONVERGENCE_THRESHOLD_2, 
                    "log_file_path": log_file_path
                },
                
                "inference_time": float
            } 
    """
    with open(f"outputs/{network}/inferred_params.json", 'r') as file:
        data = json.load(file)
    assert 'beta' in data and isinstance(data['beta'], float)
    assert 'mu' in data and isinstance(data['mu'], float)
    assert 'nu' in data and isinstance(data['nu'], float)
    assert 'R' in data and isinstance(data['R'], float)
    assert 'inferred_kappas' in data and isinstance(data['inferred_kappas'], list)
    assert 'flags' in data and isinstance(data['flags'], dict)
    assert 'model_params' in data and isinstance(data['model_params'], dict)
    
@pytest.fixture(scope='module')
def functional_random_generation_model():
    """
    Fixture to instantiate the DirectedS1Generator class converging fast just to test the functionality runs.
    """
    from random_generation import DirectedS1Generator
    """
    def __init__(self,
                 seed: int = 0,
                 verbose: bool = False,
                 log_file_path: str = "logs/DirectedS1Generator/output.log"):
    """
    return DirectedS1Generator(verbose=True,
                               log_file_path = "logs/DirectedS1Generator/test_generation_functional_generate_from_file.log")

def test_random_generation_functional_generate_from_file(setup_paths, functional_random_generation_model):
    """
    Test that the DirectedS1Generator class can be instantiated and generate works as expected.
    """
    """
    def generate_from_file(self,
                           hidden_variables_filename: str = "", 
                           output_rootname: str = "",
                           theta: List[int] = None,
                           save_data: bool = True,
                           save_coordinates: bool = True) -> None:
    """
    with open("outputs/TestNetwork/inferred_params.json", 'r') as file:
        hidden_variables = json.load(file)
    
    functional_random_generation_model.generate_from_file(hidden_variables_filename = "outputs/TestNetwork/inferred_params.json")
    assert os.path.exists("outputs/TestNetwork/generation_data.json"), "Generation data file not found"
    
    """Check if data is saved as expected
    data = {
            "parameters": {
                "beta": self.BETA,
                "mu": self.MU,
                "nu": self.NU,
                "N": self.nb_vertices,
                "R": self.R,
                "seed": self.SEED,
                "hidden_variables_file": self.hidden_variables_filename
            },
            "adjacency_list": adjacency_list,
            "network_data": {
                "reciprocity": self.reciprocity,
                "clustering": self.clustering,
                "in_degree_sequence": self.in_degree_sequence,
                "out_degree_sequence": self.out_degree_sequence
            }
        }
    """
    with open("outputs/TestNetwork/generation_data.json", 'r') as file:
        data = json.load(file)
        
    #TODO enforce types
    assert 'parameters' in data and isinstance(data['parameters'], dict)
    assert 'adjacency_list' in data and isinstance(data['adjacency_list'], list)
    assert 'network_data' in data and isinstance(data['network_data'], dict)
    assert 'reciprocity' in data['network_data'] and isinstance(data['network_data']['reciprocity'], float)
    assert 'clustering' in data['network_data'] and isinstance(data['network_data']['clustering'], float)
    assert 'in_degree_sequence' in data['network_data'] and isinstance(data['network_data']['in_degree_sequence'], list)
    assert 'out_degree_sequence' in data['network_data'] and isinstance(data['network_data']['out_degree_sequence'], list)

def test_random_generation_functional_generate_from_inference_data(setup_paths, functional_random_generation_model, network_data: dict):
    """
    Test that the DirectedS1Generator class can be instantiated and generate works as expected.
    """
    """
    def generate_from_inference_data(self,
                                   hidden_variables_data: dict = None,
                                   output_rootname: str = "",
                                   theta: List[int] = None,
                                   save_data: bool = True,
                                   save_coordinates: bool = True) -> None:
    """
    network = list(network_data.keys())[0]
    functional_random_generation_model.modify_log_file_path("logs/DirectedS1Generator/test_generation_functional_generate_from_inference_data.log")
    # open hidden variables file
    with open(f"outputs/{network}/inferred_params.json", 'r') as file:
        hidden_variables = json.load(file)

    functional_random_generation_model.generate_from_inference_data(hidden_variables_data = hidden_variables, output_rootname = network)
    assert os.path.exists(f"outputs/{network}/generation_data.json"), f"Generation data file not found in outputs/{network}/generation_data.json"
    
    """Check if data is saved as expected
    data = {
            "parameters": {
                "beta": self.BETA,
                "mu": self.MU,
                "nu": self.NU,
                "N": self.nb_vertices,
                "R": self.R,
                "seed": self.SEED,
                "hidden_variables_file": self.hidden_variables_filename
            },
            "adjacency_list": adjacency_list,
            "network_data": {
                "reciprocity": self.reciprocity,
                "clustering": self.clustering,
                "in_degree_sequence": self.in_degree_sequence,
                "out_degree_sequence": self.out_degree_sequence
            }
        }
    """
    with open(f"outputs/{network}/generation_data.json", 'r') as file:
        data = json.load(file)
        
    #TODO enforce types
    assert 'parameters' in data and isinstance(data['parameters'], dict)
    assert 'adjacency_list' in data and isinstance(data['adjacency_list'], list)
    assert 'network_data' in data and isinstance(data['network_data'], dict)
    assert 'reciprocity' in data['network_data'] and isinstance(data['network_data']['reciprocity'], float)
    assert 'clustering' in data['network_data'] and isinstance(data['network_data']['clustering'], float)
    assert 'in_degree_sequence' in data['network_data'] and isinstance(data['network_data']['in_degree_sequence'], list)
    assert 'out_degree_sequence' in data['network_data']

#TODO test the ensemble analysis
@pytest.fixture(scope='module')
def functional_ensemble_analysis_model(functional_random_generation_model):
    """
    Fixture to instantiate the DirectedS1EnsembleAnalyser class converging fast just to test the functionality runs.
    """
    from ensemble_analysis import DirectedS1EnsembleAnalyser
    # def __init__(self, 
    #              gen: DirectedS1Generator,
    #              num_samples: int,
    #              verbose: bool = False,
    #              log_file_path: str = "logs/DirectedS1EnsembleAnalyser/output.log"):
    return DirectedS1EnsembleAnalyser(gen = functional_random_generation_model,
                                    verbose = True,
                                    log_file_path = "logs/DirectedS1EnsembleAnalyser/test_ensemble_analysis_functional.log")

def test_ensemble_analysis_functional_analysis(setup_paths, functional_ensemble_analysis_model):
    """
    Test that the DirectedS1EnsembleAnalyser class can be instantiated and run_analysis works as expected from file.
    """
    # def run_analysis(self):

    functional_ensemble_analysis_model.run_analysis(hidden_variables_filename = "outputs/TestNetwork/inferred_params.json",
                                                    num_samples = 10,
                                                    save_results = True,
                                                    plot_degree_distribution = True)
    assert os.path.exists("outputs/TestNetwork/ensemble_analysis/figs/complementary_cumulative_degree_distribution.png"), "CCDF plot not found"
    assert os.path.exists("outputs/TestNetwork/ensemble_analysis/figs/in_vs_out_degree_distribution.png"), "In vs Out degree distribution plot not found"
    assert os.path.exists("outputs/TestNetwork/ensemble_analysis/results.json"), "Ensemble analysis results file not found"

    """Check if data is saved as expected
    results = {
            "average_reciprocity": self.average_reciprocity,
            "average_clustering": self.average_clustering,
            "variance_reciprocity": self.variance_reciprocity,
            "variance_clustering": self.variance_clustering,
            "reciprocity_list": self.reciprocity_list,
            "clustering_list": self.clustering_list,
            "average_in_degree": self.average_in_degree,
            "in_degree_sequences": self.in_degree_sequences,
            "out_degree_sequences": self.out_degree_sequences
            "generation_time_list": self.generation_time_list
            
        }
    """
    with open("outputs/TestNetwork/ensemble_analysis/results.json", 'r') as file:
        data = json.load(file)
    assert 'average_reciprocity' in data and isinstance(data['average_reciprocity'], float)
    assert 'average_clustering' in data and isinstance(data['average_clustering'], float)
    assert 'variance_reciprocity' in data and isinstance(data['variance_reciprocity'], float)
    assert 'variance_clustering' in data and isinstance(data['variance_clustering'], float)
    assert 'reciprocity_list' in data and isinstance(data['reciprocity_list'], list)
    assert 'clustering_list' in data and isinstance(data['clustering_list'], list)
    assert 'average_in_degree' in data and isinstance(data['average_in_degree'], float)
    assert 'in_degree_sequences' in data and isinstance(data['in_degree_sequences'], list)
    assert 'out_degree_sequences' in data and isinstance(data['out_degree_sequences'], list)
    assert 'generation_time_list' in data and isinstance(data['generation_time_list'], list)

#TODO test whole pipeline with accuracy
@pytest.fixture(scope='module')
def functional_directedS1_model():
    """
    Fixture to instantiate the DirectedS1 class converging fast just to test the functionality runs.
    """
    from directedS1 import DirectedS1
    model =  DirectedS1(log_file_path = "logs/full/DirectedS1/test_directedS1_functional.log",
                        verbose = True)
    model.set_params_fitter(seed = 0,
                            verbose = True,
                            KAPPA_MAX_NB_ITER_CONV = 10,
                            EXP_CLUST_NB_INTEGRATION_MC_STEPS = 10,
                            log_file_path = "logs/full/DirectedS1Fitter/test_infer_params_functional_fit_from_file.log")
    model.set_params_generator(seed = 0, verbose = True, log_file_path = "logs/full/DirectedS1Generator/test_generation_functional_generate_from_file.log")
    model.set_params_ensemble_analyser(verbose = True, log_file_path = "logs/full/DirectedS1EnsembleAnalyser/test_ensemble_analysis_functional.log")
    return model

def test_functional_directedS1_model(imported_modules, functional_directedS1_model, network_data, deg_seq_filename):
    """
    Test that the DirectedS1 class can be instantiated and run works as expected.
    """
    # fit_generate_analyse_from_network_data(self,
                                        #    deg_seq: Tuple[List[float], List[float]],
                                        #    reciprocity: float,
                                        #    average_local_clustering: float,
                                        #    network_name: str = "",
                                        #    num_samples: int = 10,
                                        #    save_inferred_params: bool = False,
                                        #    save_evaluation: bool = False) -> None:
    network = list(network_data.keys())[0]
    deg_seq = (network_data[network]['in_degree_sequence'], network_data[network]['out_degree_sequence'])
    functional_directedS1_model.fit_generate_analyse_from_network_data(deg_seq = deg_seq,
                                                                       reciprocity = network_data[network]['reciprocity'],
                                                                       average_local_clustering = network_data[network]['average_clustering'],
                                                                       network_name = network,
                                                                       num_samples = 10,
                                                                       save_inferred_params = True,
                                                                       save_evaluation = True)
    assert os.path.exists("outputs/TestNetwork/inferred_params.json"), "Inferred parameters file not found"
    assert os.path.exists("outputs/TestNetwork/generation_data.json"), "Generation data file not found"
    assert os.path.exists("outputs/TestNetwork/ensemble_analysis/figs/complementary_cumulative_degree_distribution.png"), "CCDF plot not found"
    assert os.path.exists("outputs/TestNetwork/ensemble_analysis/figs/in_vs_out_degree_distribution.png"), "In vs Out degree distribution plot not found"
    assert os.path.exists("outputs/TestNetwork/ensemble_analysis/results.json"), "Ensemble analysis results file not found"


#TODO large test on whole pipeline multiple datasets
#TODO add Beta convergence flags in model inference and save it to evaluation.
# @pytest.fixture(scope='module')
def test_directedS1_functional_model_multi_datasets(imported_modules, network_data):
    """
    Test the DirectedS1 class on works on multinetwork in network_data, testing on first 3 of list"""
    from directedS1 import DirectedS1

    for network in list(network_data.keys())[2:]:
        model =  DirectedS1(log_file_path = f"logs/full_multi_functional/{network}/DirectedS1/test_directedS1_functional.log",
                            verbose = True)
        #Model is going serious mode.
        model.set_params_fitter(seed = 0,
                                verbose = True,
                                log_file_path = f"logs/full_multi_functional/{network}/DirectedS1Fitter/test_infer_params_functional_fit_from_file.log",
                                KAPPA_MAX_NB_ITER_CONV = 10,
                                EXP_CLUST_NB_INTEGRATION_MC_STEPS = 10,
                                NUMERICAL_CONVERGENCE_THRESHOLD_1 = 1e-1,
                                NUMERICAL_CONVERGENCE_THRESHOLD_2 = 1e-1)
        model.set_params_generator(seed = 0, verbose = True, log_file_path = f"logs/full_multi_functional/{network}/DirectedS1Generator/test_generation_functional_generate_from_file.log")
        model.set_params_ensemble_analyser(verbose = True, log_file_path = f"logs/full_multi_functional/{network}/DirectedS1EnsembleAnalyser/test_ensemble_analysis_functional.log")
        deg_seq = (network_data[network]['in_degree_sequence'], network_data[network]['out_degree_sequence'])
        model.fit_generate_analyse_from_network_data(deg_seq = deg_seq,
                                                    reciprocity = network_data[network]['reciprocity'],
                                                    average_local_clustering = network_data[network]['average_clustering'],
                                                    network_name = network,
                                                    parent_dir = "outputs/full_multi_functional",
                                                    num_samples = 2,
                                                    save_ensemble_results = True,
                                                    save_inferred_params = True,
                                                    plot_ensemble_distributions = True,
                                                    save_evaluation = True)

#Current run saved inferred params is wrong because saving set_params_fitter forgets to update model params before saving. 
def test_directedS1_REAL_multi_datasets(imported_modules, network_data):
    """
    Test and evaluate the DirectedS1 class on every network in network_data, needed for research results.
    """
    from directedS1 import DirectedS1
    proccessed = []
    for network in list(network_data.keys())[3:]:
        if len(network_data[network]['in_degree_sequence']) > 1000:
            print(f" {network} for later")
            continue
        print(f"Testing on {network}")
        model =  DirectedS1(log_file_path = f"logs/full_multi/{network}/DirectedS1/test_directedS1.log",
                            verbose = True, 
                            debug =True)
        #Model is going serious mode.
        model.set_params_fitter(seed = 0,
                                verbose = True,
                                log_file_path = f"logs/full_multi/{network}/DirectedS1Fitter/test_infer_params_fit_from_file.log",
                                KAPPA_MAX_NB_ITER_CONV = 30,
                                EXP_CLUST_NB_INTEGRATION_MC_STEPS = 500,
                                NUMERICAL_CONVERGENCE_THRESHOLD_1 = 1.5e-1,
                                NUMERICAL_CONVERGENCE_THRESHOLD_2 = 1.5e-1,
                                debug = False)
        model.set_params_generator(seed = 0, verbose = True, log_file_path = f"logs/full_multi/{network}/DirectedS1Generator/test_generation_generate_from_file.log")
        model.set_params_ensemble_analyser(verbose = True, log_file_path = f"logs/full_multi/{network}/DirectedS1EnsembleAnalyser/test_ensemble_analysis.log")
        deg_seq = (network_data[network]['in_degree_sequence'], network_data[network]['out_degree_sequence'])
        try:
            model.fit_generate_analyse_from_network_data(deg_seq = deg_seq,
                                                        reciprocity = network_data[network]['reciprocity'],
                                                        average_local_clustering = network_data[network]['average_clustering'],
                                                        network_name = network,
                                                        parent_dir = "outputs/full_multi",
                                                        num_samples = 30,
                                                        save_ensemble_results = True,
                                                        save_inferred_params = True,
                                                        plot_ensemble_distributions = True,
                                                        save_evaluation = True)
            proccessed.append(network)
            print(f"Proccessed {proccessed}")
        except Exception as e:
            print(f"Error on {network}, can found in logs/full_multi/{network}/DirectedS1/ERROR.txt")
            #print all of the error message and trace to a file
            with open(f"logs/full_multi/{network}/DirectedS1/ERROR.txt", 'w') as file:
                file.write(str(e))
            continue 
                                  
def test_infer_nu_vmap_functional(setup_paths, functional_fit_model, network_data):
    """
    Test that the DirectedS1Fitter class can be instantiated and fit works as expected.
    """
    #TODO: TEST (Works on script)
    # take the first network in network data
    """
    def fit_from_network_data(self,
                        deg_seq: Tuple[List[float], List[float]],
                        reciprocity: float,
                        average_local_clustering: float,
                        network_name: str = ""
                        ) -> None:
    """
    network = list(network_data.keys())[0]
    deg_seq = (network_data[network]['in_degree_sequence'], network_data[network]['out_degree_sequence'])
    
    #shouldn't happen theoretically
    assert len(deg_seq[0]) == len(deg_seq[1]), "TEST DATA ERROR: In and out degree sequences are not of the same length" 
    functional_fit_model.clean_params()
    functional_fit_model.modify_log_file_path("logs/test_infer_nus_JAX/test_infer_nus_JAX.log")
    
    with open("outputs/test_infer_nus_JAX/exogenous_variables_infer_nu_iter_0.json") as file:
        exogenous_variables_infer_nu = json.load(file)
    # exogenous_variables_infer_nu is dictionary of parameters to set
    #set params from exogenous_variables_infer_nu
    functional_fit_model.set_params(**exogenous_variables_infer_nu)
    
    print(f"Running normal infer nu")
    functional_fit_model.infer_nu()
    
    normal_nu = functional_fit_model.nu
    
    print(f"Running vmap infer nu")
    functional_fit_model.infer_nu_vmap()
    vmap_nu = functional_fit_model.nu
    
    print(f"Normal nu: {normal_nu}, Vmap nu: {vmap_nu}")
    assert normal_nu == vmap_nu, "Normal and vmap nu are not equal"
    
    
if __name__ == '__main__':
    # Test all
    # exit_code = pytest.main(['-v', '-s'])
    exit_code = pytest.main(['-v', '-s', 'test_JAX.py::test_infer_params_functional_fit_from_network_data'])
    # exit_code = pytest.main(['-v', '-s', 'test_JAX.py::test_infer_nu_vmap_functional'])
    sys.exit(exit_code)