@echo off
REM Quick start script for using Azure Functions dataset

echo.
echo ============================================================
echo Azure Load Balancing Simulation - Quick Start
echo ============================================================
echo.

echo Checking for Azure dataset...
if exist "data\invocations_per_function_md5_90min.csv" (
    echo Found: invocations_per_function_md5_90min.csv
    python train_with_azure_data.py --timesteps 100000 --episodes 10
) else if exist "data\azurefunctions_trace.csv" (
    echo Found: azurefunctions_trace.csv
    python train_with_azure_data.py --dataset data\azurefunctions_trace.csv
) else (
    echo.
    echo WARNING: Azure dataset not found!
    echo.
    echo To use real Azure data:
    echo   1. Download from: https://github.com/Azure/AzurePublicDataset
    echo   2. Place CSV files in: data\
    echo   3. Re-run this script
    echo.
    echo Running with synthetic data instead...
    python train_with_azure_data.py --timesteps 50000
)

echo.
echo Results saved to: results\
pause
