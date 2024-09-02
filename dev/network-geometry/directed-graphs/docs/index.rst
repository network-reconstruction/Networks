Welcome to Directed-Network-Geometry's documentation!
====================================

This module was developped for my thesis (Yi Yao Tan), MFoCs 2024, University of Oxford.

Work based on the paper "Geometric description of clustering in directed networks" by Antoine Allard et al. in the network geometry framework. Hopefully soon to a python module released on PyPi.

The module is divided into 2 main parts:

1. Original: The original code provided by Antoine Allard et al., modified for python and as a full pipeline, 
including the generation of random directed networks, inference of parameters from a given network, and analysis of the ensemble of networks generated.

2. JAX: The code that has been converted to JAX for vectorized inference, generation, and analysis of networks.

The pipeline consists of 3 main parts:

1. Generation of random directed networks

2. Inference of parameters from a given network

3. Analysis of the ensemble of networks generated

github: https://github.com/network-reconstruction/Networks.git

Feel free to reach out and contribute: yytanwork@gmail.com



====================================
Contents:

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Module Documentation
====================

.. automodule:: JAX.generation.random_generation_JAX
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: JAX.infer_parameters.infer_params_JAX
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: Original.generation.ensemble_analysis
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: Original.generation.random_generation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: Original.infer_parameters.infer_params
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: Original.full_model.directedS1
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: general.model
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: general.metrics
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: utils.DegSeqAnalysis
    :members:
    :undoc-members:
    :show-inheritance:


