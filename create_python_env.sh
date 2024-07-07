#!/bin/bash

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide an environment name as an argument."
    echo "Usage: $0 <environment_name>"
    exit 1
fi

# Get the environment name from the command line argument
ENV_NAME=$1

# Check if the environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists."
    read -p "Do you want to remove it? (y/n): " answer
    if [[ $answer == [Yy]* ]]; then
        echo "Removing existing environment $ENV_NAME..."
        conda env remove -n "$ENV_NAME"
        echo "Existing environment $ENV_NAME has been removed."
    else
        echo "Keeping existing environment $ENV_NAME."
        exit 0
    fi
fi

# Create a new environment
echo "Creating new environment $ENV_NAME..."
conda create -n "$ENV_NAME" python=3.11 -y

if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment. Exiting."
    exit 1
fi

conda activate "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment. Exiting."
    exit 1
fi

poetry install --no-interaction

# Check if CUDA is available
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA is available: {torch.cuda.is_available()}')"
if [ $? -eq 0 ]; then
    echo "CUDA check completed successfully."
else
    echo "Error: Failed to check CUDA availability. Please ensure PyTorch is installed correctly."
fi
python -m xformers.info
python -m bitsandbytes