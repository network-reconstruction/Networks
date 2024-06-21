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

# Check if the directory exists
if [ ! -d "$PACKAGE_NAME" ]; then
    echo "Error: Directory $PACKAGE_NAME does not exist. Please run setup_file_tree.sh first."
    exit 1
fi

# Create the setup.py file
cat <<EOL > $PACKAGE_NAME/setup.py
from setuptools import setup, find_packages

setup(
    name='$PACKAGE_NAME',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your project's dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Add CLI scripts here
        ],
    },
)
EOL

echo "Python package $PACKAGE_NAME is ready for distribution."
