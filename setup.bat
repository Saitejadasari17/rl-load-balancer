@echo off
REM Setup script for Windows

echo ========================================
echo RL Load Balancer - Development Setup
echo ========================================
echo.

REM Create environment files
echo [1/3] Creating environment files...
(
echo REACT_APP_API_URL=http://localhost:8000
) > frontend\.env.local

(
echo PYTHONUNBUFFERED=1
) > backend\.env

echo [2/3] Creating data directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "results" mkdir results

echo [3/3] Installing dependencies...
echo Skipping npm install - run manually when needed

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To run locally:
echo.
echo Terminal 1 - Backend:
echo   cd backend
echo   pip install -r requirements.txt
echo   python -m uvicorn main:app --reload
echo.
echo Terminal 2 - Frontend:
echo   cd frontend
echo   npm install
echo   npm start
echo.
echo Then visit: http://localhost:3000
echo.
echo ========================================
pause
