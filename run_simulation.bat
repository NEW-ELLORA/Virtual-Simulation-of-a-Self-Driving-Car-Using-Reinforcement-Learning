@echo off
title Autonomous City Driving Simulator
cd /d "%~dp0"

echo =======================================================
echo    AUTONOMOUS SELF-DRIVING CAR SIMULATOR LAUNCHER
echo =======================================================
echo.

:: Detect Python
set PYTHON_CMD=python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    if exist ".venv\Scripts\python.exe" (
        set PYTHON_CMD=".venv\Scripts\python.exe"
    ) else if exist "venv\Scripts\python.exe" (
        set PYTHON_CMD="venv\Scripts\python.exe"
    ) else (
        echo [ERROR] Python was not found in your system PATH!
        echo Please install Python 3.10+ from python.org and check "Add Python to PATH".
        echo.
        pause
        exit /b 1
    )
)

echo [OK] Using Python: %PYTHON_CMD%
echo.
echo Launching City Driving Simulator with Reinforcement Learning Autopilot...
echo Press TAB in-game to Pause/Resume, O for Auto Drive, V/B for Cameras.
echo.

%PYTHON_CMD% driving_sim.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================
    echo [ERROR] Simulation exited with error code %ERRORLEVEL%.
    echo =======================================================
    pause
)
