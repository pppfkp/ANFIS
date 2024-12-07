#!/bin/bash

# Get the directory of the script and set it as the project root
PROJECT_ROOT="$(dirname "$(realpath "$0")")"

# Ensure the output directory exists
mkdir -p "$PROJECT_ROOT/training_runs"

# Export PYTHONPATH dynamically
export PYTHONPATH="$PROJECT_ROOT"
echo "PYTHONPATH is set to: $PYTHONPATH"

# Run commands
python -u "$PROJECT_ROOT/training_scripts/anfis_abalone_training.py" | tee "$PROJECT_ROOT/training_runs/anfis_abalone_training_output.txt"
python -u "$PROJECT_ROOT/training_scripts/anfis_power_plant_training.py" | tee "$PROJECT_ROOT/training_runs/anfis_power_plant_training_output.txt"
python -u "$PROJECT_ROOT/training_scripts/anfis_iris_training.py" | tee "$PROJECT_ROOT/training_runs/anfis_iris_training_output.txt"
python -u "$PROJECT_ROOT/training_scripts/mlp_abalone_training.py" | tee "$PROJECT_ROOT/training_runs/mlp_abalone_training_output.txt"
python -u "$PROJECT_ROOT/training_scripts/mlp_power_plant_training.py" | tee "$PROJECT_ROOT/training_runs/mlp_power_plant_training_output.txt"
python -u "$PROJECT_ROOT/training_scripts/mlp_iris_training.py" | tee "$PROJECT_ROOT/training_runs/mlp_iris_training_output.txt"
