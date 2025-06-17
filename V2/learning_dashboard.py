#!/usr/bin/env python3
"""
Real-time Learning Dashboard - Watch the AI discover and learn symbols autonomously
"""

import time
import json
from pathlib import Path
from processing_nodes import SymbolicNode
from unified_symbol_system import VectorSymbolSystem, SymbolDiscoveryEngine

class LearningDashboard:
    """Interactive dashboard to monitor autonomous learning"""
    
    def __init__(self):
        print("🚀 Initializing Learning Dashboard...")
        self.symbolic_node = SymbolicNode()
        self.data_dir = Path("data")
        
    def show_current_status(self):
        """Display current learning status"""
        print("\n" + "="*60)
        print("🧠 AUTONOMOUS LEARNING STATUS")
        print("="*60)
        
        # Vector symbol system status
        vector_insights = self.symbolic_node.vector_symbols.get_symbol_insights()
        print(f"📊 Vector Symbol System:")
        print(f"   Total symbols: {vector_insights['total_symbols']}")
        print(f"   Active symbols: {vector_insights['active_symbols']}")
        print(f"   Current threshold: {vector_insights['current_threshold']:.3f}")
        
        # Discovery system status
        discovery_insights = self.symbolic_node.symbol_discovery.get_discovery_insights()
        print(f"\n🔍 Discovery System:")
        print(f"   Total discoveries: {discovery_insights['total_discoveries']}")
        print(f"   High confidence: {discovery_insights.get('high_confidence_discoveries', 0)}")
        print(f"   Average confidence: {discovery_insights.get('average_confidence', 0):.3f}")
        
        # Recent discoveries
        if discovery_insights['total_discoveries'] > 0:
            print(f"\n🎯 Recent Discoveries:")
            recent = discovery_insights['symbols_by_confidence'][-5:]  # Last 5
            for discovery in recent:
                print(f"   {discovery['symbol']} ({discovery.get('name', 'Unknown')}): {discovery['confidence']:.2f}")
        
        # Symbol performance
        if vector_insights.get('symbol_performance'):
            print(f"\n📈 Top Performing Symbols:")
            top_symbols = sorted(vector_insights['symbol_performance'], 
                               key=lambda x: x['usage_count'], reverse=True)[:5]
            for symbol in top_symbols:
                print(f"   {symbol['glyph']} {symbol['name']}: {symbol['usage_count']} uses, "
                      f"{symbol['success_rate']:.1%} success")
    
    def feed_learning_text(self, text, source="manual_input"):
        """Feed text to the system and watch it learn"""
        print(f"\n📝 Processing: '{text[:60]}...'")
        
        # Get initial state
        initial_symbols = len(self.symbolic_node.vector_symbols.symbols)
        initial_discoveries = self.symbolic_node.symbol_discovery.get_discovery_insights()['total_discoveries']
        
        # Process the text (this triggers discovery!)
        score = self.symbolic_node.evaluate_chunk_symbolically(
            text, {'symbolic_node_access_max_phase': 1}
        )
        
        # Check what happened
        new_symbols = len(self.symbolic_node.vector_symbols.symbols)
        new_discoveries = self.symbolic_node.symbol_discovery.get_discovery_insights()['total_discoveries']
        
        print(f"   Symbolic score: {score:.3f}")
        
        if new_symbols > initial_symbols:
            print(f"   🎉 Learned {new_symbols - initial_symbols} new symbols!")
            
        if new_discoveries > initial_discoveries:
            print(f"   🔍 Made {new_discoveries - initial_discoveries} new discoveries!")
            
        # Show matched symbols
        vector_matches = self.symbolic_node.symbolic_state.get('vector_symbols_matched', [])
        if vector_matches:
            matched = [f"{m['glyph']} ({m['score']:.2f})" for m in vector_matches]
            print(f"   Matched symbols: {', '.join(matched)}")
    
    def run_interactive_session(self):
        """Interactive session where you can feed text and watch learning"""
        print("\n🎮 Interactive Learning Session")
        print("Enter text containing symbol explanations, or 'status' to see progress, 'quit' to exit")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n📝 Enter text (or command): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'status':
                    self.show_current_status()
                elif user_input.lower() == 'help':
                    self.show_help()
                elif len(user_input) > 10:
                    self.feed_learning_text(user_input)
                else:
                    print("   Please enter longer text or a command")
                    
            except KeyboardInterrupt:
                break
                
        print("\n👋 Learning session ended. Progress saved!")
    
    def run_auto_discovery_test(self):
        """Run automated discovery with various symbol explanations"""
        print("\n🤖 Running Automated Discovery Test...")
        
        test_texts = [
            "The integral symbol ∫ represents integration in calculus and accumulation of quantities over intervals.",
            "The partial derivative ∂ is used in multivariable calculus to show change with respect to one variable.",
            "The not equal symbol ≠ indicates that two values or expressions are not identical in mathematics.",
            "The approximately equal symbol ≈ shows that two values are very close but not exactly the same.",
            "The much greater than symbol ≫ indicates that one quantity is significantly larger than another.",
            "The element of symbol ∈ shows that an object belongs to a particular set in set theory.",
            "The subset symbol ⊂ indicates that one set is entirely contained within another set.",
            "The intersection symbol ∩ represents the common elements between two or more sets.",
            "The union symbol ∪ combines all elements from two or more sets without duplication.",
            "The empty set symbol ∅ represents a set that contains no elements whatsoever."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n🧪 Test {i}/10:")
            self.feed_learning_text(text, f"auto_test_{i}")
            time.sleep(0.5)  # Brief pause to see progress
            
        print("\n📊 Auto-discovery test complete!")
        self.show_current_status()
    
    def show_help(self):
        """Show available commands"""
        print("\n📖 Available Commands:")
        print("   'status' - Show current learning status")
        print("   'help' - Show this help")
        print("   'quit' or 'q' - Exit the session")
        print("   Or enter text like:")
        print("   'The theta symbol (θ) represents angles in trigonometry'")
        print("   'Using epsilon (ε) to denote small values in analysis'")
    
    def monitor_learning_files(self):
        """Monitor the learning files to see persistent progress"""
        print("\n📁 Learning Files Monitor:")
        
        files_to_monitor = [
            "data/vector_symbol_learning.json",
            "data/symbol_discoveries.json"
        ]
        
        for file_path in files_to_monitor:
            path = Path(file_path)
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    print(f"   {file_path}: {len(data) if isinstance(data, list) else 'OK'} entries")
                except:
                    print(f"   {file_path}: Error reading")
            else:
                print(f"   {file_path}: Not found")

def main():
    dashboard = LearningDashboard()
    
    print("\n🔮 Welcome to the Autonomous Learning Dashboard!")
    print("This is where you can watch your AI discover and learn symbols in real-time.")
    
    while True:
        print("\n🎯 What would you like to do?")
        print("1. Show current learning status")
        print("2. Interactive learning session (feed your own text)")
        print("3. Run automated discovery test")
        print("4. Monitor learning files")
        print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '1':
            dashboard.show_current_status()
        elif choice == '2':
            dashboard.run_interactive_session()
        elif choice == '3':
            dashboard.run_auto_discovery_test()
        elif choice == '4':
            dashboard.monitor_learning_files()
        elif choice == '5':
            print("👋 Goodbye! Your AI continues learning...")
            break
        else:
            print("Please choose 1-5")

if __name__ == "__main__":
    main()