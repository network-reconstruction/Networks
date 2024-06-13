#!/bin/bash

# Prompt for user inputs
read -p "Do you want to create a new Conda environment? (yes/no): " CREATE_ENV

if [ "$CREATE_ENV" == "yes" ]; then
  read -p "Enter the new Conda environment name: " ENV_NAME
  read -p "Enter the Python version (e.g., 3.10): " PYTHON_VERSION

  # Create a conda environment
  echo "Creating conda environment named $ENV_NAME with Python $PYTHON_VERSION..."
  conda create -y -n $ENV_NAME python=$PYTHON_VERSION

elif [ "$CREATE_ENV" == "no" ]; then
  read -p "Enter the name of the existing Conda environment: " ENV_NAME

  # Check if the environment exists
  if ! conda env list | grep -q "$ENV_NAME"; then
    echo "The specified environment does not exist."
    exit 1
  fi

else
  echo "Invalid choice. Please enter 'yes' or 'no'."
  exit 1
fi

# Activate the conda environment
echo "Activating conda environment $ENV_NAME..."
source activate $ENV_NAME


# Run setup.py to install the package with arguments
echo "Running setup.py with arguments..."
python scripts/setup/find_requirements.py .

# Install the requirements
if [ -f "scripts/setup/requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    while IFS= read -r package || [[ -n "$package" ]]; do
        echo "Installing $package..."
        if pip install "$package"; then
            echo "$package installed successfully."
        else
            echo "Failed to install $package. Skipping..."
        fi
    done < scripts/setup/requirements.txt
else
    echo "scripts/setup/requirements.txt not found!"
fi

echo "Setup complete!"
