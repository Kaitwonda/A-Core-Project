# run_pipeline.py - Complete Learning and Evolution Pipeline

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Import the learning and evolution components
from autonomous_learner import autonomous_learning_cycle
from memory_evolution_engine import run_memory_evolution
from memory_architecture import TripartiteMemory

def run_learning_pipeline(data_dir="data", 
                         learning_config=None, 
                         evolution_config=None,
                         cycles=1):
    """
    Run complete learning pipeline:
    1. Autonomous learning (crawl and store in bridge)
    2. Memory evolution (migrate and optimize)
    
    Args:
        data_dir: Directory for data storage
        learning_config: Config for autonomous learner
        evolution_config: Config for memory evolution
        cycles: Number of learn-evolve cycles to run
    """
    
    # Default configurations
    if learning_config is None:
        learning_config = {
            'focus_only_on_phase_1': True,
            'max_urls_per_session': 5,
            'store_to_bridge': True  # New flag to force bridge storage
        }
    
    if evolution_config is None:
        evolution_config = {
            'reverse_audit_confidence_threshold': 0.3,
            'enable_reverse_migration': True,
            'enable_weight_evolution': True,
            'save_detailed_logs': True
        }
    
    print("="*60)
    print("🚀 COMPLETE LEARNING & EVOLUTION PIPELINE")
    print("="*60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Data directory: {data_dir}")
    print(f"🔄 Cycles to run: {cycles}")
    print()
    
    for cycle_num in range(1, cycles + 1):
        print(f"\n{'='*60}")
        print(f"📊 CYCLE {cycle_num}/{cycles}")
        print(f"{'='*60}")
        
        # Phase 1: Autonomous Learning
        print(f"\n🔍 PHASE 1: Autonomous Learning")
        print("-"*40)
        
        try:
            # Get initial memory state
            memory = TripartiteMemory(data_dir=data_dir)
            initial_counts = memory.get_counts()
            print(f"📊 Initial memory state:")
            print(f"   Total: {initial_counts['total']} items")
            print(f"   Bridge: {initial_counts['bridge']} items")
            
            # Run autonomous learning
            start_time = time.time()
            autonomous_learning_cycle(
                focus_only_on_phase_1=learning_config.get('focus_only_on_phase_1', True)
            )
            learning_time = time.time() - start_time
            
            # Check new memory state
            memory = TripartiteMemory(data_dir=data_dir)
            post_learning_counts = memory.get_counts()
            new_items = post_learning_counts['total'] - initial_counts['total']
            
            print(f"\n✅ Learning complete in {learning_time:.1f}s")
            print(f"   New items added: {new_items}")
            print(f"   Total items now: {post_learning_counts['total']}")
            
        except Exception as e:
            print(f"❌ Learning phase failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Phase 2: Memory Evolution
        print(f"\n🧬 PHASE 2: Memory Evolution")
        print("-"*40)
        
        try:
            start_time = time.time()
            evolution_results = run_memory_evolution(
                data_dir=data_dir,
                config=evolution_config
            )
            evolution_time = time.time() - start_time
            
            print(f"\n✅ Evolution complete in {evolution_time:.1f}s")
            print(f"   Items migrated: {evolution_results['migrated']}")
            print(f"   Items reversed: {evolution_results['reversed']}")
            print(f"   Health status: {evolution_results['health_status']}")
            
            # Show distribution changes
            final_dist = evolution_results['final_distribution']
            print(f"\n📊 Memory distribution after evolution:")
            print(f"   Logic: {final_dist['logic']['count']} ({final_dist['logic']['percentage']:.1f}%)")
            print(f"   Symbolic: {final_dist['symbolic']['count']} ({final_dist['symbolic']['percentage']:.1f}%)")
            print(f"   Bridge: {final_dist['bridge']['count']} ({final_dist['bridge']['percentage']:.1f}%)")
            
        except Exception as e:
            print(f"❌ Evolution phase failed: {e}")
            import traceback
            traceback.print_exc()
            
        # Brief pause between cycles
        if cycle_num < cycles:
            print(f"\n⏳ Waiting 5 seconds before next cycle...")
            time.sleep(5)
    
    print(f"\n{'='*60}")
    print("✅ PIPELINE COMPLETE")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Final summary
    try:
        memory = TripartiteMemory(data_dir=data_dir)
        final_counts = memory.get_counts()
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Total items: {final_counts['total']}")
        print(f"   Logic memory: {final_counts['logic']}")
        print(f"   Symbolic memory: {final_counts['symbolic']}")
        print(f"   Bridge memory: {final_counts['bridge']}")
        
        if initial_counts['total'] > 0:
            bridge_reduction = ((initial_counts['bridge'] - final_counts['bridge']) / 
                              initial_counts['bridge'] * 100)
            print(f"   Bridge reduction: {bridge_reduction:.1f}%")
            
    except Exception as e:
        print(f"Could not generate final summary: {e}")


def main():
    """Main entry point with CLI arguments"""
    parser = argparse.ArgumentParser(
        description="Run the complete learning and evolution pipeline"
    )
    parser.add_argument(
        "--cycles", 
        type=int, 
        default=1,
        help="Number of learn-evolve cycles to run (default: 1)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for data storage (default: data)"
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=5,
        help="Maximum URLs to process per learning session (default: 5)"
    )
    parser.add_argument(
        "--skip-evolution",
        action="store_true",
        help="Skip evolution phase (learning only)"
    )
    parser.add_argument(
        "--evolution-only",
        action="store_true",
        help="Skip learning phase (evolution only)"
    )
    
    args = parser.parse_args()
    
    # Configure based on arguments
    learning_config = {
        'focus_only_on_phase_1': True,
        'max_urls_per_session': args.max_urls,
        'store_to_bridge': True
    }
    
    evolution_config = {
        'reverse_audit_confidence_threshold': 0.3,
        'enable_reverse_migration': True,
        'enable_weight_evolution': True,
        'save_detailed_logs': True
    }
    
    # Handle special modes
    if args.evolution_only:
        print("🧬 Running evolution only...")
        evolution_results = run_memory_evolution(
            data_dir=args.data_dir,
            config=evolution_config
        )
        print(f"✅ Evolution complete: {evolution_results['migrated']} migrated, "
              f"{evolution_results['reversed']} reversed")
        return
    
    if args.skip_evolution:
        print("🔍 Running learning only...")
        autonomous_learning_cycle(
            focus_only_on_phase_1=learning_config['focus_only_on_phase_1']
        )
        print("✅ Learning complete")
        return
    
    # Run full pipeline
    run_learning_pipeline(
        data_dir=args.data_dir,
        learning_config=learning_config,
        evolution_config=evolution_config,
        cycles=args.cycles
    )


if __name__ == "__main__":
    # Ensure we're in the right directory
    if not Path("autonomous_learner.py").exists():
        print("❌ Error: Must run from project root directory")
        print("   Current directory:", Path.cwd())
        sys.exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)