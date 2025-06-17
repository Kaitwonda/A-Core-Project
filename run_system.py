#!/usr/bin/env python3
# run_system.py - Simple startup script for Dual Brain AI System

"""
Dual Brain AI System Startup Script

This script provides a simple way to start and manage the dual brain AI system.
It serves as the main entry point and provides quick access to common operations.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point - delegate to CLI"""
    try:
        from cli import main as cli_main
        return cli_main()
    except ImportError as e:
        print(f"❌ Error importing CLI: {e}")
        print("   Make sure all dependencies are installed")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    # Show banner
    print("🧠 Dual Brain AI System")
    print("=" * 50)
    
    # If no arguments, show quick help
    if len(sys.argv) == 1:
        print("\nQuick Start Commands:")
        print("  python run_system.py start --mode autonomous    # Start autonomous mode")
        print("  python run_system.py start --mode interactive   # Start interactive mode")
        print("  python run_system.py chat                       # Start chat session")
        print("  python run_system.py status                     # Show system status")
        print("  python run_system.py health                     # Check system health")
        print("  python run_system.py --help                     # Show full help")
        print("\nFor full command list, use: python run_system.py --help")
        sys.exit(0)
    
    # Run the CLI
    exit_code = main()
    sys.exit(exit_code)