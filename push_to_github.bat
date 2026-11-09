@echo off
title Push to GitHub
cd /d "%~dp0"
set "PATH=%PATH%;C:\Users\Ayush C S\.mingit\cmd"

echo =======================================================
echo          PUSHING SIMULATION REPO TO GITHUB
echo =======================================================
echo.
echo Remote: https://github.com/NEW-ELLORA/Virtual-Simulation-of-a-Self-Driving-Car-Using-Reinforcement-Learning.git
echo Branch: main
echo.
echo Connecting to GitHub...
echo.

git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] Successfully pushed all code to GitHub!
    echo =======================================================
) else (
    echo.
    echo =======================================================
    echo [NOTICE] If prompted for credentials:
    echo Username: Your GitHub Username
    echo Password: Use a GitHub Personal Access Token (PAT)
    echo           (GitHub does not accept account passwords for git push)
    echo =======================================================
)
echo.
pause
