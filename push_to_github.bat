@echo off
title GitHub Automatic Login & Push
cd /d "%~dp0"
set "PATH=%PATH%;C:\Users\Ayush C S\.mingit\cmd;C:\Users\Ayush C S\.gh\bin"

echo =======================================================
echo      AUTOMATIC GITHUB 1-CLICK LOGIN & PUSH
echo =======================================================
echo.
echo Target Repository:
echo https://github.com/NEW-ELLORA/Virtual-Simulation-of-a-Self-Driving-Car-Using-Reinforcement-Learning.git
echo.
echo Step 1: Connecting your GitHub account via browser...
echo.

:: Authenticate with GitHub via web browser
gh auth status >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo A browser window will open shortly.
    echo Press ENTER when prompted to open GitHub and authorize.
    echo.
    gh auth login --web -h github.com -p https -s repo,workflow
) else (
    echo [OK] Already authenticated with GitHub!
)

echo.
echo Step 2: Configuring Git Credential Helper...
gh auth setup-git

echo.
echo Step 3: Pushing code to main branch...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =======================================================
    echo [SUCCESS] PUSH COMPLETED SUCCESSFULLY!
    echo.
    echo Your project is now live at:
    echo https://github.com/NEW-ELLORA/Virtual-Simulation-of-a-Self-Driving-Car-Using-Reinforcement-Learning
    echo =======================================================
) else (
    echo.
    echo =======================================================
    echo [NOTICE] If your account does not have write access to
    echo NEW-ELLORA, please ensure you are logged into the right account.
    echo =======================================================
)

echo.
pause
