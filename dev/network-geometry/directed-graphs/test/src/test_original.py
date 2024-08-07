# file: test_original.py

import json
import os
import sys
import pytest

# Organization:
"""
12 directories, 43 files
(network_reconstruction) tr02[~/Thesis/networks/dev/network-geometry/directed-graphs]$ tree
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

    print(f"sys.path: {sys.path}")

@pytest.fixture(scope='module')
def imported_modules(setup_paths):
    try:
        import random_generation
        import ensemble_analysis
        import infer_params
    except ModuleNotFoundError as e:
        pytest.fail(f"Error importing modules: {e}")
    
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
    try:
        assert hasattr(random_generation, 'GeneratingDirectedS1'), "GeneratingDirectedS1 not found in random_generation"
        print("GeneratingDirectedS1 is available in random_generation")
    except AssertionError as e:
        pytest.fail(f"random_generation test failed: {e}")

    # Test ensemble_analysis functionality
    ensemble_analysis = imported_modules['ensemble_analysis']
    try:
        assert hasattr(ensemble_analysis, 'GraphEnsembleAnalysis'), "GraphEnsembleAnalysis not found in ensemble_analysis"
        print("GraphEnsembleAnalysis is available in ensemble_analysis")
    except AssertionError as e:
        pytest.fail(f"ensemble_analysis test failed: {e}")

    # Test infer_params functionality
    infer_params = imported_modules['infer_params']
    try:
        assert hasattr(infer_params, 'FittingDirectedS1'), "FittingDirectedS1 not found in infer_params"
        print("FittingDirectedS1 is available in infer_params")
    except AssertionError as e:
        pytest.fail(f"infer_params test failed: {e}")
    
def test_import_data(setup_paths):
    """
    Test that the data file for testing can be imported successfully.
    """
    try:
        with open('../data/network_data.json', 'r') as file:
            data = json.load(file)
        # check that it is correct format for the test
        # network_data = {
        #     'reciprocity': reciprocity,
        #     'average_clustering': avg_clustering,
        #     'in_degree_sequence': in_degree_sequence,
        #     'out_degree_sequence': out_degree_sequence,
        #     'average_in_degree': average_in_degree
        # }
        for network in data:
            assert 'reciprocity' in data[network]
            assert 'average_clustering' in data[network]
            assert 'in_degree_sequence' in data[network]
            assert 'out_degree_sequence' in data[network]
            assert 'average_in_degree' in data[network]
            
    except ModuleNotFoundError as e:
        pytest.fail(f"Error importing data: {e}")
        
    assert True, "Data imported successfully"
    
def test_infer_params(setup_paths):
    """
    Test that the FittingDirectedS1 class can be instantiated.
    """
    try:
        from infer_params import FittingDirectedS1
        model = FittingDirectedS1()
    except Exception as e:
        pytest.fail(f"Error instantiating FittingDirectedS1: {e}")
        
    assert True, "FittingDirectedS1 class instantiated successfully"

if __name__ == '__main__':
    exit_code = pytest.main(['-v'])
    sys.exit(exit_code)