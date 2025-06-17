#!/bin/bash
# start_system.sh - Unix/Linux startup script for Dual Brain AI System

echo
echo "=========================================="
echo "   Dual Brain AI System - Unix Launch"
echo "=========================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Error: Python is not installed or not in PATH"
        echo "Please install Python 3.8+ and make sure it's accessible"
        exit 1
    else
        PYTHON=python
    fi
else
    PYTHON=python3
fi

# Set working directory to script location
cd "$(dirname "$0")"

# Check if we have arguments
if [ $# -eq 0 ]; then
    echo "Starting in autonomous mode by default..."
    $PYTHON run_system.py start --mode autonomous
else
    # Pass all arguments to the Python script
    $PYTHON run_system.py "$@"
fi

echo
echo "System has stopped."