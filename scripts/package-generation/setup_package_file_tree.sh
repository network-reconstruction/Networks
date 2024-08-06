#!/bin/bash

# Function to create directories and initialize files
create_file_structure() {
    local PACKAGE_NAME=$1

    echo "Creating directory structure for $PACKAGE_NAME..."

    mkdir -p $PACKAGE_NAME
    mkdir -p $PACKAGE_NAME/$PACKAGE_NAME
    mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/standard
    mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/jax
    mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/utils
    mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/data
    mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/examples
    mkdir -p $PACKAGE_NAME/docs/source
    mkdir -p $PACKAGE_NAME/tests
    mkdir -p $PACKAGE_NAME/scripts

    # Create __init__.py files with basic initialization
    touch $PACKAGE_NAME/$PACKAGE_NAME/__init__.py
    echo "# $PACKAGE_NAME package initialization" > $PACKAGE_NAME/$PACKAGE_NAME/__init__.py

    # Create requirements.txt, readme.md in root
    touch $PACKAGE_NAME/requirements.txt
    


    touch $PACKAGE_NAME/$PACKAGE_NAME/standard/__init__.py
    echo "# standard module initialization" > $PACKAGE_NAME/$PACKAGE_NAME/standard/__init__.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/jax/__init__.py
    echo "# jax module initialization" > $PACKAGE_NAME/$PACKAGE_NAME/jax/__init__.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/utils/__init__.py
    echo "# utils module initialization" > $PACKAGE_NAME/$PACKAGE_NAME/utils/__init__.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/data/__init__.py
    echo "# data module initialization" > $PACKAGE_NAME/$PACKAGE_NAME/data/__init__.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/examples/__init__.py
    echo "# examples module initialization" > $PACKAGE_NAME/$PACKAGE_NAME/examples/__init__.py

    touch $PACKAGE_NAME/tests/__init__.py
    echo "# tests module initialization" > $PACKAGE_NAME/tests/__init__.py

    # Create example module files
    touch $PACKAGE_NAME/$PACKAGE_NAME/standard/module1.py
    echo "# module1 for standard operations" > $PACKAGE_NAME/$PACKAGE_NAME/standard/module1.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/standard/module2.py
    echo "# module2 for standard operations" > $PACKAGE_NAME/$PACKAGE_NAME/standard/module2.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/jax/module1.py
    echo "# module1 for jax operations" > $PACKAGE_NAME/$PACKAGE_NAME/jax/module1.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/jax/module2.py
    echo "# module2 for jax operations" > $PACKAGE_NAME/$PACKAGE_NAME/jax/module2.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/utils/helpers.py
    echo "# helper functions" > $PACKAGE_NAME/$PACKAGE_NAME/utils/helpers.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/utils/math_ops.py
    echo "# additional mathematical operations" > $PACKAGE_NAME/$PACKAGE_NAME/utils/math_ops.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/data/data_loader.py
    echo "# data loading and processing" > $PACKAGE_NAME/$PACKAGE_NAME/data/data_loader.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/examples/example1.py
    echo "# example usage of the package" > $PACKAGE_NAME/$PACKAGE_NAME/examples/example1.py

    touch $PACKAGE_NAME/$PACKAGE_NAME/examples/example2.py
    echo "# another example usage of the package" > $PACKAGE_NAME/$PACKAGE_NAME/examples/example2.py

    # Create documentation files
    touch $PACKAGE_NAME/docs/conf.py
    echo "# Configuration file for the Sphinx documentation builder." > $PACKAGE_NAME/docs/conf.py
    echo "import os" >> $PACKAGE_NAME/docs/conf.py
    echo "import sys" >> $PACKAGE_NAME/docs/conf.py
    echo "sys.path.insert(0, os.path.abspath('../../$PACKAGE_NAME'))" >> $PACKAGE_NAME/docs/conf.py

    touch $PACKAGE_NAME/docs/index.rst
    echo "$PACKAGE_NAME Documentation" > $PACKAGE_NAME/docs/index.rst
    echo "=========================" >> $PACKAGE_NAME/docs/index.rst
    echo ".. toctree::" >> $PACKAGE_NAME/docs/index.rst
    echo "   :maxdepth: 2" >> $PACKAGE_NAME/docs/index.rst
    echo "   :caption: Contents:" >> $PACKAGE_NAME/docs/index.rst
    echo "" >> $PACKAGE_NAME/docs/index.rst
    echo "modules" >> $PACKAGE_NAME/docs/index.rst

    touch $PACKAGE_NAME/docs/make.bat
    echo "@echo off" > $PACKAGE_NAME/docs/make.bat
    echo "REM Command file for Sphinx documentation" >> $PACKAGE_NAME/docs/make.bat
    echo "make html" >> $PACKAGE_NAME/docs/make.bat

    touch $PACKAGE_NAME/docs/Makefile
    echo "# Makefile for Sphinx documentation" > $PACKAGE_NAME/docs/Makefile
    echo "html:" >> $PACKAGE_NAME/docs/Makefile
    echo "\tsphinx-build -b html source build" >> $PACKAGE_NAME/docs/Makefile

    touch $PACKAGE_NAME/docs/source/modules.rst
    echo "Modules" > $PACKAGE_NAME/docs/source/modules.rst
    echo "=======" >> $PACKAGE_NAME/docs/source/modules.rst
    echo ".. automodule:: $PACKAGE_NAME" >> $PACKAGE_NAME/docs/source/modules.rst
    echo "   :members:" >> $PACKAGE_NAME/docs/source/modules.rst

    touch $PACKAGE_NAME/docs/source/standard.rst
    echo "standard Module" > $PACKAGE_NAME/docs/source/standard.rst
    echo "=================" >> $PACKAGE_NAME/docs/source/standard.rst

    touch $PACKAGE_NAME/docs/source/jax.rst
    echo "jax Module" > $PACKAGE_NAME/docs/source/jax.rst
    echo "=================" >> $PACKAGE_NAME/docs/source/jax.rst

    # Create test files
    touch $PACKAGE_NAME/tests/test_standard.py
    echo "# Tests for standard module" > $PACKAGE_NAME/tests/test_standard.py

    touch $PACKAGE_NAME/tests/test_jax.py
    echo "# Tests for jax module" > $PACKAGE_NAME/tests/test_jax.py

    # Create setup files
    touch $PACKAGE_NAME/scripts/setup.py
    echo "from setuptools import setup, find_packages" > $PACKAGE_NAME/scripts/setup.py
    echo "" >> $PACKAGE_NAME/scripts/setup.py
    echo "setup(" >> $PACKAGE_NAME/scripts/setup.py
    echo "    name='$PACKAGE_NAME'," >> $PACKAGE_NAME/scripts/setup.py
    echo "    version='0.1'," >> $PACKAGE_NAME/scripts/setup.py
    echo "    packages=find_packages()," >> $PACKAGE_NAME/scripts/setup.py
    echo "    install_requires=[" >> $PACKAGE_NAME/scripts/setup.py
    echo "        'numpy'," >> $PACKAGE_NAME/scripts/setup.py
    echo "        'scipy'," >> $PACKAGE_NAME/scripts/setup.py
    echo "        'jax'," >> $PACKAGE_NAME/scripts/setup.py
    echo "    ]," >> $PACKAGE_NAME/scripts/setup.py
    echo ")" >> $PACKAGE_NAME/scripts/setup.py

    touch $PACKAGE_NAME/scripts/requirements.txt
    echo "numpy" > $PACKAGE_NAME/scripts/requirements.txt
    echo "scipy" >> $PACKAGE_NAME/scripts/requirements.txt
    echo "jax" >> $PACKAGE_NAME/scripts/requirements.txt

    touch $PACKAGE_NAME/scripts/README.md
    echo "# $PACKAGE_NAME" > $PACKAGE_NAME/scripts/README.md
    echo "This is a Python package for high-performance computing simulations." >> $PACKAGE_NAME/scripts/README.md

    touch $PACKAGE_NAME/scripts/LICENSE
    echo "MIT License" > $PACKAGE_NAME/scripts/LICENSE
    echo "" >> $PACKAGE_NAME/scripts/LICENSE
    echo "Permission is hereby granted, free of charge, to any person obtaining a copy" >> $PACKAGE_NAME/scripts/LICENSE
    echo "of this software and associated documentation files (the \"Software\"), to deal" >> $PACKAGE_NAME/scripts/LICENSE
    echo "in the Software without restriction, including without limitation the rights" >> $PACKAGE_NAME/scripts/LICENSE
    echo "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell" >> $PACKAGE_NAME/scripts/LICENSE
    echo "copies of the Software, and to permit persons to whom the Software is" >> $PACKAGE_NAME/scripts/LICENSE
    echo "furnished to do so, subject to the following conditions:" >> $PACKAGE_NAME/scripts/LICENSE
    echo "" >> $PACKAGE_NAME/scripts/LICENSE
    echo "The above copyright notice and this permission notice shall be included in all" >> $PACKAGE_NAME/scripts/LICENSE
    echo "copies or substantial portions of the Software." >> $PACKAGE_NAME/scripts/LICENSE
    echo "" >> $PACKAGE_NAME/scripts/LICENSE
    echo "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR" >> $PACKAGE_NAME/scripts/LICENSE
    echo "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY," >> $PACKAGE_NAME/scripts/LICENSE
    echo "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE" >> $PACKAGE_NAME/scripts/LICENSE
    echo "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER" >> $PACKAGE_NAME/scripts/LICENSE
    echo "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM," >> $PACKAGE_NAME/scripts/LICENSE
    echo "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE" >> $PACKAGE_NAME/scripts/LICENSE
    echo "SOFTWARE." >> $PACKAGE_NAME/scripts/LICENSE

    echo "File structure and initialization for $PACKAGE_NAME completed successfully."
}

# Main script starts here

# Check if a package name is provided
if [ -z "$1" ]; then
    echo "Error: Package name is required."
    echo "Usage: $0 <package_name>"
    exit 1
fi

PACKAGE_NAME="$1"

# Call function to create directory structure and initialize files
create_file_structure "$PACKAGE_NAME"
