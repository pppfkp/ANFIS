# Get the directory of the script and set it as the project root
$PROJECT_ROOT = Split-Path -Parent (Resolve-Path $MyInvocation.MyCommand.Path)

# Ensure the output directory exists
$OUTPUT_DIR = Join-Path $PROJECT_ROOT "validation_runs"
if (-Not (Test-Path -Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
}

# Export PYTHONPATH dynamically
$env:PYTHONPATH = $PROJECT_ROOT
Write-Host "PYTHONPATH is set to: $env:PYTHONPATH"

# Define script paths and output files
$scripts = @(
    "anfis_abalone_cross_validation.py",
    "anfis_power_plant_cross_validation.py",
    "anfis_iris_cross_validation.py",
    "mlp_abalone_cross_validation.py",
    "mlp_power_plant_cross_validation.py",
    "mlp_iris_cross_validation.py"
)

# Run commands
foreach ($script in $scripts) {
    $scriptPath = Join-Path "$PROJECT_ROOT/cross_validation_scripts" $script
    $outputFile = Join-Path $OUTPUT_DIR ("$($script -replace '\.py$', '_output.txt')")
    python -u $scriptPath | Tee-Object -FilePath $outputFile
}
