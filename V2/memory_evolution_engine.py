# memory_evolution_engine.py - Complete Memory Evolution Integration

import json
from pathlib import Path
from datetime import datetime

# Import all our components
from unified_memory import UnifiedMemory
from adaptive_migration import AdaptiveThresholds, MigrationEngine
from reverse_migration import ReverseMigrationAuditor
from weight_evolution import WeightEvolver
from memory_analytics import MemoryAnalyzer
from evolution_anchor import EvolutionAnchor

class MemoryEvolutionEngine:
    """
    Main orchestrator for the complete memory evolution system.
    Integrates all components for autonomous memory organization.
    """
    
    def __init__(self, data_dir="data", config=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or {
            'reverse_audit_confidence_threshold': 0.3,
            'enable_reverse_migration': True,
            'enable_weight_evolution': True,
            'save_detailed_logs': True
        }
        
        # Initialize all components
        self.memory = UnifiedMemory(data_dir=data_dir)
        self.thresholds = AdaptiveThresholds(data_dir=data_dir)
        self.migration_engine = MigrationEngine(self.memory.tripartite, self.thresholds)
        self.auditor = ReverseMigrationAuditor(
            self.memory.tripartite, 
            confidence_threshold=self.config['reverse_audit_confidence_threshold']
        )
        self.weight_evolver = WeightEvolver(data_dir=data_dir)
        self.analyzer = MemoryAnalyzer(self.memory, data_dir=data_dir)
        self.evolution_anchor = EvolutionAnchor(data_dir=data_dir)
        
        # Session tracking
        self.session_start = datetime.utcnow()
        self.session_log = []
        
    def run_evolution_cycle(self):
        """
        Run a complete evolution cycle with safety anchors:
        0. Create cognitive safety anchor
        1. Get initial stats and assess baseline
        2. Reverse audit (catch misclassifications)
        3. Forward migration
        4. Weight evolution
        5. Analytics and reporting
        6. Distress monitoring and safety checks
        """
        print("\n" + "="*60)
        print("🧬 MEMORY EVOLUTION CYCLE STARTING")
        print("="*60)
        
        # Step 0: Create safety anchor before any changes
        print("\n🌟 Creating cognitive safety anchor...")
        snapshot_id = self.evolution_anchor.create_cognitive_snapshot("Before evolution cycle")
        
        if snapshot_id:
            print(f"   ✅ Safety anchor established: {snapshot_id}")
        else:
            print("   ⚠️ Could not create safety anchor - proceeding with caution")
        
        # Step 1: Initial state
        print("\n📊 Initial State:")
        initial_stats = self.analyzer.get_memory_stats()
        self._print_distribution(initial_stats)
        
        # Step 2: Reverse audit
        reversed_count = 0
        if self.config['enable_reverse_migration']:
            print("\n🔍 Running reverse migration audit...")
            reversed_count = self.auditor.audit_all()
            if reversed_count > 0:
                print(f"  ← Moved {reversed_count} misclassified items back to bridge")
                audit_summary = self.auditor.get_audit_summary()
                for reason, count in audit_summary['reasons'].items():
                    print(f"    - {reason}: {count}")
            else:
                print("  ✓ No misclassifications detected")
        
        # Step 3: Forward migration
        print("\n🔄 Running forward migration...")
        self.thresholds.bump()  # Increment age
        threshold = self.thresholds.get_migration_threshold()
        print(f"  Migration threshold: {threshold:.2f}")
        
        migrated_count = self.migration_engine.migrate_from_bridge()
        if migrated_count > 0:
            migration_summary = self.migration_engine.get_migration_summary()
            print(f"  → Migrated {migrated_count} items from bridge")
            print(f"    - To logic: {migration_summary['to_logic']}")
            print(f"    - To symbolic: {migration_summary['to_symbolic']}")
        else:
            print("  ✓ No items ready for migration")
            
        # Save memory state
        print("\n💾 Saving memory state...")
        save_results = self.memory.save_all()
        successful_saves = sum(1 for v in save_results.values() if v)
        print(f"  Saved {successful_saves}/3 memory stores")
        
        # Step 4: Weight evolution
        if self.config['enable_weight_evolution']:
            print("\n⚡ Evolving weights...")
            current_stats = self.analyzer.get_memory_stats()
            
            # Get performance stats from brain metrics if available
            performance_stats = self._get_performance_stats()
            
            evolved = self.weight_evolver.evolve_weights(
                run_count=self.thresholds.age,
                memory_stats=current_stats,
                performance_stats=performance_stats
            )
            
            if not evolved:
                print("  ✓ Weights already at target specialization")
                
        # Step 5: Final analytics
        print("\n📈 Generating evolution report...")
        final_stats = self.analyzer.get_memory_stats()
        
        # Create summaries
        migration_summary = self.migration_engine.get_migration_summary() if migrated_count > 0 else None
        audit_summary = self.auditor.get_audit_summary() if reversed_count > 0 else None
        
        report = self.analyzer.generate_evolution_report(
            migration_summary=migration_summary,
            audit_summary=audit_summary
        )
        
        # Print report
        self.analyzer.print_report(report)
        
        # Show evolution summary
        weight_summary = self.weight_evolver.get_evolution_summary()
        print(f"\n⚖️  Weight Evolution Summary:")
        print(f"  Current weights: static={weight_summary['current_weights']['static']:.3f}, "
              f"dynamic={weight_summary['current_weights']['dynamic']:.3f}")
        print(f"  Specialization: {weight_summary['current_specialization']:.3f}")
        print(f"  Total evolutions: {weight_summary['total_evolutions']}")
        
        # Log session
        self._log_session({
            'timestamp': datetime.utcnow().isoformat(),
            'initial_stats': initial_stats,
            'final_stats': final_stats,
            'reversed': reversed_count,
            'migrated': migrated_count,
            'threshold': threshold,
            'weights': self.weight_evolver.weights,
            'report': report
        })
        
        # Step 6: Post-evolution distress monitoring
        print("\n🔍 Post-evolution safety check...")
        distress_assessment = self.evolution_anchor.detect_evolution_distress()
        
        print(f"   🎯 Distress level: {distress_assessment['distress_level']:.2f}")
        print(f"   📊 Status: {distress_assessment['status']}")
        
        if distress_assessment['distress_level'] > 0.3:
            print(f"   ⚠️ Recommendation: {distress_assessment.get('recommendation', 'Monitor closely')}")
            if distress_assessment['signals']:
                print("   🔔 Distress signals:")
                for signal in distress_assessment['signals']:
                    print(f"     • {signal}")
        else:
            print("   ✅ Evolution proceeding healthily")
        
        # Check if we should recommend rollback
        recommend_rollback = distress_assessment['distress_level'] > 0.7
        
        print("\n✅ Evolution cycle complete!")
        
        return {
            'success': True,
            'reversed': reversed_count,
            'migrated': migrated_count,
            'final_distribution': final_stats['distribution'],
            'health_status': final_stats['health_indicators']['status'],
            'safety_anchor': snapshot_id,
            'distress_assessment': distress_assessment,
            'recommend_rollback': recommend_rollback
        }
        
    def _print_distribution(self, stats):
        """Pretty print memory distribution"""
        dist = stats['distribution']
        total = stats['total_items']
        
        print(f"  Total items: {total}")
        print(f"  Logic:    {dist['logic']['count']:4d} ({dist['logic']['percentage']:5.1f}%)")
        print(f"  Symbolic: {dist['symbolic']['count']:4d} ({dist['symbolic']['percentage']:5.1f}%)")
        print(f"  Bridge:   {dist['bridge']['count']:4d} ({dist['bridge']['percentage']:5.1f}%)")
        
    def _get_performance_stats(self):
        """Get performance stats from brain metrics if available"""
        try:
            # This would integrate with your brain_metrics module
            # For now, return None to use default evolution
            return None
        except:
            return None
            
    def _log_session(self, session_data):
        """Log session data"""
        if not self.config['save_detailed_logs']:
            return
            
        log_file = self.data_dir / "evolution_sessions.json"
        
        sessions = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    sessions = json.load(f)
            except:
                pass
                
        sessions.append(session_data)
        sessions = sessions[-50:]  # Keep last 50 sessions
        
        with open(log_file, 'w') as f:
            json.dump(sessions, f, indent=2)
            
    def add_test_data(self):
        """Add test data for demonstration"""
        print("\n🔧 Adding test data...")
        
        # Add logic items
        for i in range(15):
            item = {
                'id': f'logic_{i}',
                'text': f'Clear algorithmic explanation of process {i}',
                'logic_score': 8 + (i % 2),
                'symbolic_score': 1 + (i % 2)
            }
            self.memory.store(item, 'FOLLOW_LOGIC')
            
        # Add symbolic items
        for i in range(10):
            item = {
                'id': f'symbolic_{i}',
                'text': f'Emotional narrative about journey {i}',
                'logic_score': 2,
                'symbolic_score': 8
            }
            self.memory.store(item, 'FOLLOW_SYMBOLIC')
            
        # Add bridge items (various patterns)
        # Ready to migrate
        for i in range(5):
            item = {
                'id': f'bridge_ready_{i}',
                'text': f'Technical content {i}',
                'logic_score': 8.5,
                'symbolic_score': 1.5
            }
            # Build stable history
            for _ in range(3):
                self.memory.store(item.copy(), 'FOLLOW_HYBRID')
                
        # Volatile items
        for i in range(3):
            item = {
                'id': f'bridge_volatile_{i}',
                'text': f'Mixed content {i}',
                'logic_score': 5,
                'symbolic_score': 5
            }
            # Build unstable history
            for dec in ['FOLLOW_LOGIC', 'FOLLOW_SYMBOLIC', 'FOLLOW_HYBRID']:
                self.memory.store(item.copy(), dec)
                
        # Misclassified items
        for i in range(2):
            item = {
                'id': f'misclassified_{i}',
                'text': f'Actually symbolic content {i}',
                'logic_score': 3,
                'symbolic_score': 7
            }
            self.memory.store(item, 'FOLLOW_LOGIC')  # Wrong classification
            
        print(f"  Added {self.memory.get_counts()['total']} test items")
        

# Integration function for memory_optimizer.py
def run_memory_evolution(data_dir="data", config=None):
    """
    Main entry point for memory evolution.
    Call this from memory_optimizer.py at the end of main().
    """
    engine = MemoryEvolutionEngine(data_dir=data_dir, config=config)
    return engine.run_evolution_cycle()


# Standalone testing
if __name__ == "__main__":
    import tempfile
    import sys
    
    print("🧪 Testing Complete Memory Evolution System...")
    
    # Use temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n📁 Using temporary directory: {tmpdir}")
        
        # Create engine
        engine = MemoryEvolutionEngine(data_dir=tmpdir)
        
        # Add test data
        engine.add_test_data()
        
        # Run evolution cycle
        print("\n🚀 Running first evolution cycle...")
        result1 = engine.run_evolution_cycle()
        
        print(f"\n✅ First cycle results:")
        print(f"  - Reversed: {result1['reversed']}")
        print(f"  - Migrated: {result1['migrated']}")
        print(f"  - Health: {result1['health_status']}")
        
        # Run second cycle to see evolution
        print("\n🚀 Running second evolution cycle...")
        result2 = engine.run_evolution_cycle()
        
        print(f"\n✅ Second cycle results:")
        print(f"  - Reversed: {result2['reversed']}")
        print(f"  - Migrated: {result2['migrated']}")
        print(f"  - Health: {result2['health_status']}")
        
        # Check that bridge is shrinking
        initial_bridge = 13  # From test data
        final_bridge = result2['final_distribution']['bridge']['count']
        
        print(f"\n📊 Bridge evolution: {initial_bridge} → {final_bridge}")
        assert final_bridge < initial_bridge, "Bridge should shrink over time"
        
    print("\n✅ All integration tests passed!")
    print("\n💡 To integrate with memory_optimizer.py, add this at the end of main():")
    print("    from memory_evolution_engine import run_memory_evolution")
    print("    run_memory_evolution(data_dir='data')")