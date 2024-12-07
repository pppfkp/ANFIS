#!/bin/bash

# Get the directory of the script and set it as the project root
PROJECT_ROOT="$(dirname "$(realpath "$0")")"

# Ensure the output directory exists
mkdir -p "$PROJECT_ROOT/validation_runs"

# Export PYTHONPATH dynamically
export PYTHONPATH="$PROJECT_ROOT"
echo "PYTHONPATH is set to: $PYTHONPATH"

# Run commands
python -u "$PROJECT_ROOT/cross_validation_scripts/anfis_abalone_cross_validation.py" | tee "$PROJECT_ROOT/validation_runs/anfis_abalone_cv_output.txt"
python -u "$PROJECT_ROOT/cross_validation_scripts/anfis_power_plant_cross_validation.py" | tee "$PROJECT_ROOT/validation_runs/anfis_power_plant_cv_output.txt"
python -u "$PROJECT_ROOT/cross_validation_scripts/anfis_iris_cross_validation.py" | tee "$PROJECT_ROOT/validation_runs/anfis_iris_cv_output.txt"
python -u "$PROJECT_ROOT/cross_validation_scripts/mlp_abalone_cross_validation.py" | tee "$PROJECT_ROOT/validation_runs/mlp_abalone_cv_output.txt"
python -u "$PROJECT_ROOT/cross_validation_scripts/mlp_power_plant_cross_validation.py" | tee "$PROJECT_ROOT/validation_runs/mlp_power_plant_cv_output.txt"
python -u "$PROJECT_ROOT/cross_validation_scripts/mlp_iris_cross_validation.py" | tee "$PROJECT_ROOT/validation_runs/mlp_iris_cv_output.txt"
