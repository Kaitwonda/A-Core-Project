#!/usr/bin/env python3
"""
View Learning Data - Inspect what the AI has learned
"""

import json
from pathlib import Path
from datetime import datetime

def view_discoveries():
    """View all symbol discoveries"""
    discoveries_file = Path("data/symbol_discoveries.json")
    
    if not discoveries_file.exists():
        print("❌ No discoveries file found. Run some learning first!")
        return
        
    try:
        with open(discoveries_file, 'r') as f:
            discoveries = json.load(f)
            
        print(f"🔍 DISCOVERED SYMBOLS ({len(discoveries)} total)")
        print("=" * 60)
        
        for discovery in discoveries:
            print(f"\n📍 {discovery['symbol']} ({discovery['name']})")
            print(f"   Confidence: {discovery['confidence']:.2f}")
            print(f"   Context: {discovery['context_snippet'][:100]}...")
            print(f"   Math concepts: {discovery['mathematical_concepts']}")
            print(f"   Metaphor concepts: {discovery['metaphorical_concepts']}")
            print(f"   Discovered: {discovery['discovery_timestamp'][:19]}")
            
    except Exception as e:
        print(f"❌ Error reading discoveries: {e}")

def view_vector_learning():
    """View vector symbol learning progress"""
    learning_file = Path("data/vector_symbol_learning.json")
    
    if not learning_file.exists():
        print("❌ No vector learning file found.")
        return
        
    try:
        with open(learning_file, 'r') as f:
            learning_data = json.load(f)
            
        print(f"\n🧠 VECTOR SYMBOL LEARNING")
        print("=" * 60)
        print(f"Current threshold: {learning_data.get('system_threshold', 'Unknown')}")
        
        symbols = {k: v for k, v in learning_data.items() if k != 'system_threshold'}
        
        if symbols:
            print(f"\nLearned symbols: {len(symbols)}")
            for symbol, data in symbols.items():
                usage = data.get('usage_count', 0)
                success = data.get('successful_matches', 0)
                failed = data.get('failed_matches', 0)
                success_rate = (success / usage * 100) if usage > 0 else 0
                
                print(f"   {symbol}: {usage} uses, {success_rate:.1f}% success rate")
                
                if data.get('context_adaptations'):
                    print(f"      Recent contexts: {len(data['context_adaptations'])}")
        else:
            print("No symbols have been used in learning yet.")
            
    except Exception as e:
        print(f"❌ Error reading vector learning: {e}")

def view_memory_files():
    """View all learning-related files"""
    data_dir = Path("data")
    
    print(f"\n📁 LEARNING FILES STATUS")
    print("=" * 60)
    
    learning_files = [
        "symbol_discoveries.json",
        "vector_symbol_learning.json", 
        "logic_memory.json",
        "symbolic_memory.json",
        "bridge_memory.json"
    ]
    
    for filename in learning_files:
        file_path = data_dir / filename
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    count = len(data)
                else:
                    count = "Unknown format"
                    
                # Get file modification time
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                print(f"✅ {filename}: {count} items (modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
                
            except Exception as e:
                print(f"⚠️ {filename}: Error reading ({e})")
        else:
            print(f"❌ {filename}: Not found")

def main():
    print("🔍 Learning Data Viewer")
    print("See what your AI has discovered and learned!")
    
    while True:
        print("\n" + "="*50)
        print("📊 What would you like to view?")
        print("1. Symbol discoveries")
        print("2. Vector learning progress") 
        print("3. All learning files status")
        print("4. Exit")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == '1':
            view_discoveries()
        elif choice == '2':
            view_vector_learning()
        elif choice == '3':
            view_memory_files()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("Please choose 1-4")

if __name__ == "__main__":
    main()