#!/bin/bash

# Setup script to fix all dependencies and configuration issues
echo "🔧 Setting up Dual Brain AI System..."

# Step 1: Install missing Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Step 2: Download spaCy model
echo "🧠 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Step 3: Create data directories if they don't exist
echo "📁 Creating data directories..."
mkdir -p data/logs
mkdir -p data/backups
mkdir -p data/quarantine
mkdir -p data/user_vault

# Step 4: Initialize empty JSON files if they don't exist
echo "📄 Initializing data files..."
# Only create if files don't exist
[ ! -f data/logic_memory.json ] && echo "[]" > data/logic_memory.json
[ ! -f data/symbolic_memory.json ] && echo "[]" > data/symbolic_memory.json
[ ! -f data/bridge_memory.json ] && echo "[]" > data/bridge_memory.json
[ ! -f data/adaptive_config.json ] && echo '{"link_score_weight_static": 0.6, "link_score_weight_dynamic": 0.4}' > data/adaptive_config.json
[ ! -f data/unified_weights.json ] && echo '{}' > data/unified_weights.json

# Step 5: Test the system
echo "🧪 Testing system components..."
python -c "
import sys
try:
    from bs4 import BeautifulSoup
    print('✅ BeautifulSoup4 installed successfully')
except ImportError:
    print('❌ BeautifulSoup4 not installed')
    sys.exit(1)

try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print('✅ spaCy and language model loaded successfully')
except:
    print('❌ spaCy or language model not loaded')
    sys.exit(1)

try:
    import pandas
    print('✅ Pandas installed successfully')
except ImportError:
    print('❌ Pandas not installed')
    sys.exit(1)

print('\n✅ All dependencies installed successfully!')
"

echo ""
echo "✅ Setup complete! You can now run the system with:"
echo "   python run_system.py --help"
echo "   python run_system.py chat"
echo "   python run_system.py interact"
echo ""