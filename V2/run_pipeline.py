#!/usr/bin/env python3
# run_pipeline.py - Unified Learning Pipeline for Both Autonomous and Single-Link Modes
"""
Unified pipeline that merges the best of both learning approaches:

Mode 1: Massive Autonomous Learning
- Processes 500+ URLs autonomously
- Context-aware link following
- Full brain integration with evolution cycles
- Continuous cognitive monitoring

Mode 2: Smart Single-Link Processing  
- User provides one URL
- AI discovers 5 related high-similarity links
- Comprehensive analysis and response
- Same brain integration as autonomous mode

Mode 3: Enhanced Interactive
- Upgraded memory_optimizer.py experience
- Smart URL processing during chat
- Evolution and analytics integration
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Import all the enhanced systems
from enhanced_autonomous_learner import EnhancedAutonomousLearner, start_massive_web_learning
from smart_link_processor import SmartLinkProcessor, process_user_url_smart
from memory_optimizer import main as run_memory_optimizer
from memory_evolution_engine import run_memory_evolution
from unified_memory import UnifiedMemory, generate_self_diagnostic_voice
from memory_analytics import MemoryAnalyzer
from evolution_anchor import EvolutionAnchor
from linguistic_warfare import check_url_safety

class UnifiedLearningPipeline:
    """
    Unified learning pipeline that can handle massive autonomous learning,
    smart single-link processing, and enhanced interactive sessions.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core systems
        self.unified_memory = UnifiedMemory(data_dir)
        self.analyzer = MemoryAnalyzer(self.unified_memory, data_dir)
        self.evolution_anchor = EvolutionAnchor(data_dir)
        
        print(f"🧠 Unified Learning Pipeline initialized")
        
    def run_autonomous_learning(self, seed_urls: List[str], target_urls: int = 500, 
                               learning_focus: str = "general", enable_evolution: bool = True):
        """
        Run massive autonomous learning mode with 500+ URL processing.
        """
        print(f"\n🚀 AUTONOMOUS LEARNING MODE")
        print(f"🎯 Target: {target_urls} URLs")
        print(f"📚 Focus: {learning_focus}")
        print("="*50)
        
        # Pre-learning cognitive state
        pre_stats = self.analyzer.get_memory_stats()
        pre_report = generate_self_diagnostic_voice()
        
        print(f"💭 Pre-learning AI state: \"{pre_report}\"")
        print(f"📊 Starting memory: {pre_stats['total_items']} items")
        
        # Start massive learning
        learner = EnhancedAutonomousLearner(self.data_dir)
        learner.start_massive_learning_session(seed_urls, target_urls, learning_focus)
        
        # Post-learning analysis
        post_stats = self.analyzer.get_memory_stats()
        post_report = generate_self_diagnostic_voice()
        
        print(f"\n📈 AUTONOMOUS LEARNING COMPLETE")
        print(f"💭 Post-learning AI state: \"{post_report}\"")
        print(f"📊 Final memory: {post_stats['total_items']} items (+{post_stats['total_items'] - pre_stats['total_items']})")
        
        # Final evolution cycle if enabled
        if enable_evolution:
            print(f"\n🧬 Running final evolution cycle...")
            evolution_result = run_memory_evolution(data_dir=self.data_dir)
            if evolution_result and evolution_result.get('success'):
                print(f"   ✅ Evolution complete: {evolution_result.get('migrated', 0)} migrated")
        
        return {
            'mode': 'autonomous',
            'urls_processed': learner.session_stats.get('urls_processed', 0),
            'chunks_learned': learner.session_stats.get('chunks_learned', 0),
            'symbols_discovered': learner.session_stats.get('symbols_discovered', 0),
            'memory_growth': post_stats['total_items'] - pre_stats['total_items'],
            'final_health': post_stats['health_indicators']['status']
        }
    
    def run_smart_single_link(self, user_url: str, max_related: int = 5, 
                            context: str = "general", interactive: bool = True):
        """
        Run smart single-link processing with related link discovery.
        """
        print(f"\n🔗 SMART SINGLE-LINK MODE")
        print(f"📄 Processing: {user_url}")
        print(f"🔍 Max related links: {max_related}")
        print("="*50)
        
        # Security check first
        is_safe, security_analysis = check_url_safety(user_url, context=context)
        
        if not is_safe:
            error_response = f"🛡️ Security Warning: This URL appears unsafe\\n"
            error_response += f"Risk Score: {security_analysis['risk_score']:.1%}\\n"
            error_response += f"Risk Factors: {', '.join(security_analysis['risk_factors'])}"
            
            if interactive:
                print(error_response)
            return {
                'mode': 'single_link',
                'status': 'blocked',
                'response': error_response,
                'security_analysis': security_analysis
            }
        
        # Process the URL and related links
        processor = SmartLinkProcessor(self.data_dir)
        result = processor.process_user_link_with_discovery(user_url, max_related, context)
        
        if interactive and result.get('comprehensive_response'):
            print(f"\n📋 COMPREHENSIVE ANALYSIS:")
            print(result['comprehensive_response'])
        
        # Learning summary
        if result.get('learning_summary'):
            summary = result['learning_summary']
            print(f"\n📚 LEARNING SUMMARY:")
            print(f"   Links processed: {summary['total_links_processed']}")
            print(f"   Content analyzed: {summary['total_content_length']:,} characters")
            print(f"   Concepts discovered: {summary['unique_concepts_discovered']}")
            print(f"   Educational value: {summary['average_educational_value']:.0%}")
        
        return {
            'mode': 'single_link',
            'status': result.get('status', 'unknown'),
            'response': result.get('comprehensive_response', ''),
            'learning_summary': result.get('learning_summary', {}),
            'related_links_found': len(result.get('related_links', []))
        }
    
    def run_enhanced_interactive(self):
        """
        Run enhanced interactive mode (upgraded memory_optimizer.py).
        """
        print(f"\n💬 ENHANCED INTERACTIVE MODE")
        print("Launching enhanced memory optimizer with smart link processing...")
        print("="*50)
        
        # Enhance memory_optimizer to use smart link processing
        self._patch_memory_optimizer_for_smart_links()
        
        # Run the enhanced memory optimizer
        run_memory_optimizer()
    
    def _patch_memory_optimizer_for_smart_links(self):
        """
        Dynamically enhance memory_optimizer to use smart link processing.
        """
        # This would modify memory_optimizer.py to use our smart link processor
        # For now, we'll rely on the integration already built into memory_optimizer
        pass
    
    def show_system_status(self):
        """Show comprehensive system status."""
        print(f"\n📊 UNIFIED LEARNING SYSTEM STATUS")
        print("="*50)
        
        # Memory statistics
        stats = self.analyzer.get_memory_stats()
        print(f"🧠 Memory Status:")
        print(f"   Total items: {stats['total_items']:,}")
        print(f"   Health: {stats['health_indicators']['status']}")
        print(f"   Distribution: Logic({stats['distribution']['logic']['count']:,}) " +
              f"Symbolic({stats['distribution']['symbolic']['count']:,}) " +
              f"Bridge({stats['distribution']['bridge']['count']:,})")
        
        # AI self-assessment
        self_report = generate_self_diagnostic_voice()
        print(f"\\n💭 AI Self-Assessment: \"{self_report}\"")
        
        # Evolution anchor status
        anchor_status = self.evolution_anchor.get_anchor_status()
        print(f"\\n🌟 Evolution Anchors: {anchor_status['total_snapshots']} cognitive snapshots available")
        
        # System capabilities
        print(f"\\n🎯 Available Learning Modes:")
        print(f"   🚀 Autonomous: Process 500+ URLs with context-aware discovery")
        print(f"   🔗 Smart Link: Single URL + related link discovery")
        print(f"   💬 Interactive: Enhanced chat with smart URL processing")
        
    def run_evolution_cycle(self):
        """Run a memory evolution cycle."""
        print(f"\\n🧬 Running Memory Evolution Cycle...")
        
        result = run_memory_evolution(data_dir=self.data_dir)
        
        if result and result.get('success'):
            print(f"✅ Evolution complete!")
            print(f"   Migrated: {result.get('migrated', 0)} items")
            print(f"   Reversed: {result.get('reversed', 0)} items") 
            print(f"   Health: {result.get('health_status', 'unknown')}")
            return result
        else:
            print(f"⚠️ Evolution had issues")
            return None

def main():
    """Main entry point for the unified pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified Learning Pipeline - Autonomous, Single-Link, or Interactive Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Massive autonomous learning (500 URLs)
  python run_pipeline.py --mode autonomous --urls "https://en.wikipedia.org/wiki/AI,https://arxiv.org/abs/..." --target 500

  # Smart single-link processing
  python run_pipeline.py --mode single-link --url "https://example.com/article"

  # Enhanced interactive chat
  python run_pipeline.py --mode interactive

  # System status check
  python run_pipeline.py --status

  # Run evolution cycle
  python run_pipeline.py --evolve
        """
    )
    
    parser.add_argument('--mode', choices=['autonomous', 'single-link', 'interactive'], 
                       help='Learning mode to use')
    parser.add_argument('--urls', type=str, help='Comma-separated seed URLs for autonomous mode')
    parser.add_argument('--url', type=str, help='Single URL for smart processing')
    parser.add_argument('--target', type=int, default=500, help='Target URLs for autonomous mode')
    parser.add_argument('--related', type=int, default=5, help='Max related links for single-link mode')
    parser.add_argument('--focus', type=str, default='general', 
                       choices=['general', 'ai_consciousness', 'science', 'philosophy', 'technology'],
                       help='Learning focus area')
    parser.add_argument('--context', type=str, default='general', help='Context for single-link mode')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--evolve', action='store_true', help='Run evolution cycle')
    parser.add_argument('--no-evolution', action='store_true', help='Disable evolution in autonomous mode')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = UnifiedLearningPipeline(args.data_dir)
    
    # Handle status check
    if args.status:
        pipeline.show_system_status()
        return
    
    # Handle evolution cycle
    if args.evolve:
        pipeline.run_evolution_cycle()
        return
    
    # Handle different modes
    if args.mode == 'autonomous':
        if not args.urls:
            print("❌ Error: --urls required for autonomous mode")
            parser.print_help()
            sys.exit(1)
        
        seed_urls = [url.strip() for url in args.urls.split(',')]
        enable_evolution = not args.no_evolution
        
        result = pipeline.run_autonomous_learning(
            seed_urls=seed_urls,
            target_urls=args.target,
            learning_focus=args.focus,
            enable_evolution=enable_evolution
        )
        
        print(f"\\n🎯 Autonomous learning result: {result}")
        
    elif args.mode == 'single-link':
        if not args.url:
            print("❌ Error: --url required for single-link mode")
            parser.print_help()
            sys.exit(1)
        
        result = pipeline.run_smart_single_link(
            user_url=args.url,
            max_related=args.related,
            context=args.context,
            interactive=True
        )
        
    elif args.mode == 'interactive':
        pipeline.run_enhanced_interactive()
        
    else:
        # No mode specified, show help and system status
        parser.print_help()
        print("\\n" + "="*50)
        pipeline.show_system_status()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n\\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"\\n\\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()