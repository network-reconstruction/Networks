#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 <package_name>"
    exit 1
}

# Check if a package name was provided as an argument
if [ -z "$1" ]; then
    echo "Error: No package name provided."
    usage
fi

PACKAGE_NAME=$1

# Create the root directory
mkdir -p $PACKAGE_NAME

# Create main package directory and subdirectories
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/linearized
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/tensorized
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/utils
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/data
mkdir -p $PACKAGE_NAME/$PACKAGE_NAME/examples
mkdir -p $PACKAGE_NAME/docs/source
mkdir -p $PACKAGE_NAME/tests
mkdir -p $PACKAGE_NAME/scripts

# Create __init__.py files
touch $PACKAGE_NAME/$PACKAGE_NAME/__init__.py
touch $PACKAGE_NAME/$PACKAGE_NAME/linearized/__init__.py
touch $PACKAGE_NAME/$PACKAGE_NAME/tensorized/__init__.py
touch $PACKAGE_NAME/$PACKAGE_NAME/utils/__init__.py
touch $PACKAGE_NAME/$PACKAGE_NAME/data/__init__.py
touch $PACKAGE_NAME/$PACKAGE_NAME/examples/__init__.py
touch $PACKAGE_NAME/tests/__init__.py

# Create module files (example placeholders)
touch $PACKAGE_NAME/$PACKAGE_NAME/linearized/module1.py
touch $PACKAGE_NAME/$PACKAGE_NAME/linearized/module2.py
touch $PACKAGE_NAME/$PACKAGE_NAME/tensorized/module1.py
touch $PACKAGE_NAME/$PACKAGE_NAME/tensorized/module2.py
touch $PACKAGE_NAME/$PACKAGE_NAME/utils/helpers.py
touch $PACKAGE_NAME/$PACKAGE_NAME/utils/math_ops.py
touch $PACKAGE_NAME/$PACKAGE_NAME/data/data_loader.py
touch $PACKAGE_NAME/$PACKAGE_NAME/examples/example1.py
touch $PACKAGE_NAME/$PACKAGE_NAME/examples/example2.py

# Create docs files
touch $PACKAGE_NAME/docs/conf.py
touch $PACKAGE_NAME/docs/index.rst
touch $PACKAGE_NAME/docs/make.bat
touch $PACKAGE_NAME/docs/Makefile
touch $PACKAGE_NAME/docs/source/modules.rst
touch $PACKAGE_NAME/docs/source/linearized.rst
touch $PACKAGE_NAME/docs/source/tensorized.rst

# Create test files
touch $PACKAGE_NAME/tests/test_linearized.py
touch $PACKAGE_NAME/tests/test_tensorized.py

# Create script files
touch $PACKAGE_NAME/scripts/setup.py
touch $PACKAGE_NAME/scripts/requirements.txt
touch $PACKAGE_NAME/scripts/README.md
touch $PACKAGE_NAME/scripts/LICENSE

# Create .gitignore file
touch $PACKAGE_NAME/.gitignore

echo "File structure for $PACKAGE_NAME created successfully."
