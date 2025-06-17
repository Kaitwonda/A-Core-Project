#!/usr/bin/env python3
"""
Autonomous Learning Runner - Let the AI continuously learn from various sources
"""

import time
import random
from processing_nodes import SymbolicNode

class AutonomousLearner:
    """Continuously feeds the AI new content to learn from"""
    
    def __init__(self):
        print("🤖 Initializing Autonomous Learner...")
        self.symbolic_node = SymbolicNode()
        self.learning_sessions = 0
        
    def generate_symbol_explanations(self):
        """Generate various symbol explanations for learning"""
        
        # Mathematical symbols with explanations
        math_symbols = [
            "The factorial symbol (!) represents the product of all positive integers up to a given number.",
            "The floor function ⌊x⌋ gives the greatest integer less than or equal to x in mathematics.",
            "The ceiling function ⌈x⌉ returns the smallest integer greater than or equal to x.",
            "The modulo operator (%) gives the remainder after division of one number by another.",
            "The absolute value bars |x| represent the distance of x from zero on the number line.",
            "The square root symbol √ indicates the principal square root of a number.",
            "The cube root symbol ∛ represents the number that when cubed gives the original value.",
            "The proportional symbol ∝ indicates that one quantity varies directly with another.",
            "The congruent symbol ≡ shows that two expressions are equivalent in modular arithmetic."
        ]
        
        # Greek letters in context
        greek_symbols = [
            "The alpha symbol (α) represents the first in a series and angular acceleration in physics.",
            "The beta symbol (β) denotes the second position and represents velocity ratios in relativity.",
            "The gamma symbol (γ) indicates the third element and represents the Lorentz factor in physics.",
            "The epsilon symbol (ε) represents arbitrarily small positive quantities in mathematical analysis.",
            "The theta symbol (θ) commonly represents angles in trigonometry and polar coordinates.",
            "The mu symbol (μ) represents micro units and the coefficient of friction in physics.",
            "The rho symbol (ρ) denotes density in physics and correlation coefficients in statistics.",
            "The tau symbol (τ) represents the circle constant 2π and proper time in relativity.",
            "The chi symbol (χ) is used in the chi-squared test and represents characteristics in statistics.",
            "The eta symbol (η) represents efficiency ratios and the Dedekind eta function in mathematics."
        ]
        
        return math_symbols + greek_symbols
    
    def run_learning_session(self, duration_minutes=5):
        """Run a continuous learning session"""
        print(f"\n🚀 Starting {duration_minutes}-minute autonomous learning session...")
        
        explanations = self.generate_symbol_explanations()
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        learned_count = 0
        processed_count = 0
        
        while time.time() < end_time:
            # Pick a random explanation
            explanation = random.choice(explanations)
            
            print(f"\n📚 Learning from: '{explanation[:60]}...'")
            
            # Get initial state
            initial_symbols = len(self.symbolic_node.vector_symbols.symbols)
            
            # Process the explanation (this triggers discovery and learning!)
            score = self.symbolic_node.evaluate_chunk_symbolically(
                explanation, {'symbolic_node_access_max_phase': 1}
            )
            
            # Check what was learned
            new_symbols = len(self.symbolic_node.vector_symbols.symbols)
            if new_symbols > initial_symbols:
                learned_count += new_symbols - initial_symbols
                print(f"   🎉 Learned {new_symbols - initial_symbols} new symbols!")
            
            print(f"   Score: {score:.3f}")
            processed_count += 1
            
            # Brief pause between learning
            time.sleep(1)
            
        self.learning_sessions += 1
        print(f"\n✅ Learning session {self.learning_sessions} complete!")
        print(f"   Processed: {processed_count} explanations")
        print(f"   Learned: {learned_count} new symbols")
        
        return learned_count, processed_count
    
    def show_learning_progress(self):
        """Show current learning state"""
        print(f"\n📈 LEARNING PROGRESS")
        print("=" * 50)
        
        vector_insights = self.symbolic_node.vector_symbols.get_symbol_insights()
        discovery_insights = self.symbolic_node.symbol_discovery.get_discovery_insights()
        
        print(f"Sessions completed: {self.learning_sessions}")
        print(f"Total symbols: {vector_insights['total_symbols']}")
        print(f"Active symbols: {vector_insights['active_symbols']}")  
        print(f"Total discoveries: {discovery_insights['total_discoveries']}")
        print(f"Learning threshold: {vector_insights['current_threshold']:.3f}")
        
        if discovery_insights['total_discoveries'] > 0:
            print(f"\n🎯 Recent discoveries:")
            recent = discovery_insights['symbols_by_confidence'][-3:]
            for disc in recent:
                print(f"   {disc['symbol']} (confidence: {disc['confidence']:.2f})")

def main():
    learner = AutonomousLearner()
    
    print("🚀 Autonomous Learning System")
    print("Let your AI learn continuously from symbol explanations!")
    
    while True:
        print("\n" + "="*50)
        learner.show_learning_progress()
        
        print("\n🎯 Learning Options:")
        print("1. Quick learning session (1 minute)")
        print("2. Extended learning session (3 minutes)")
        print("3. Show detailed progress")
        print("4. Exit")
        
        choice = input("\nChoice (1-4): ").strip()
        
        if choice == '1':
            learner.run_learning_session(1)
        elif choice == '2':
            learner.run_learning_session(3)
        elif choice == '3':
            # Show detailed insights
            vector_insights = learner.symbolic_node.vector_symbols.get_symbol_insights()
            if vector_insights.get('symbol_performance'):
                print("\n📊 Symbol Performance:")
                for symbol in vector_insights['symbol_performance'][:10]:
                    print(f"   {symbol['glyph']} {symbol['name']}: "
                          f"{symbol['usage_count']} uses, {symbol['success_rate']:.1%} success")
        elif choice == '4':
            print("👋 Learning continues in the background...")
            break
        else:
            print("Please choose 1-4")

if __name__ == "__main__":
    main()