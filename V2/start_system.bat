@echo off
REM start_system.bat - Windows startup script for Dual Brain AI System

echo.
echo ==========================================
echo    Dual Brain AI System - Windows Launch
echo ==========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Set working directory to script location
cd /d "%~dp0"

REM Check if we have arguments
if "%1"=="" (
    echo Starting in autonomous mode by default...
    python run_system.py start --mode autonomous
) else (
    REM Pass all arguments to the Python script
    python run_system.py %*
)

echo.
echo System has stopped.
if "%1"=="" pause