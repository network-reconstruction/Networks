from setuptools import setup, find_packages

setup(
    name='network-analysis',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'jax',
    ],
)
