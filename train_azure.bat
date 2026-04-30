@echo off
cd /d "%~dp0"
echo.
echo ========================================
echo Adaptive Load Balancing - Azure Training
echo ========================================
echo.
python train_on_real_azure.py
pause
