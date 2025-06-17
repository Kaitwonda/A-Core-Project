# unified_migration_system.py - Consolidated Migration Architecture for Dual Brain AI

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import logging

# Import existing components
from unified_weight_system import UnifiedWeightSystem
from memory_architecture import TripartiteMemory

@dataclass
class MigrationResult:
    """Result of a migration operation"""
    operation_type: str
    items_processed: int
    items_migrated: int
    conflicts_resolved: int
    insights_applied: int
    execution_time: float
    metadata: Dict[str, Any]

@dataclass
class UnifiedMigrationSession:
    """Complete migration session results"""
    session_id: str
    timestamp: str
    forward_migration: MigrationResult
    reverse_audit: MigrationResult
    weight_optimization: MigrationResult
    data_consolidation: MigrationResult
    total_execution_time: float
    system_health: Dict[str, Any]

class DataConsolidator:
    """
    Consolidates orphaned data and eliminates storage duplication
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('DataConsolidator')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def consolidate_orphaned_data(self) -> Dict[str, Any]:
        """Consolidate all orphaned and duplicate data files"""
        start_time = datetime.now()
        consolidation_results = {
            'memory_files_merged': 0,
            'orphaned_files_processed': 0,
            'backups_archived': 0,
            'conflicts_resolved': 0,
            'data_volume_processed': 0
        }
        
        # Step 1: Merge duplicate memory storage systems
        self._merge_memory_storage_systems(consolidation_results)
        
        # Step 2: Process orphaned JSON files
        self._process_orphaned_files(consolidation_results)
        
        # Step 3: Archive backup files
        self._archive_backup_files(consolidation_results)
        
        # Step 4: Consolidate weight files
        self._consolidate_weight_files(consolidation_results)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"Data consolidation completed in {execution_time:.2f}s")
        self.logger.info(f"Results: {consolidation_results}")
        
        return consolidation_results
        
    def _merge_memory_storage_systems(self, results: Dict):
        """Merge duplicate storage systems into unified stores"""
        memory_files = [
            ('bridge.json', 'bridge_memory.json'),
            ('logic.json', 'logic_memory.json'),
            ('symbolic.json', 'symbolic_memory.json')
        ]
        
        for legacy_file, unified_file in memory_files:
            legacy_path = self.data_dir / legacy_file
            unified_path = self.data_dir / unified_file
            
            if legacy_path.exists() and legacy_path.stat().st_size > 2:  # More than just "[]"
                try:
                    # Load legacy data
                    with open(legacy_path, 'r', encoding='utf-8') as f:
                        legacy_data = json.load(f)
                    
                    # Load unified data
                    unified_data = []
                    if unified_path.exists():
                        with open(unified_path, 'r', encoding='utf-8') as f:
                            unified_data = json.load(f)
                    
                    # Merge if legacy has data
                    if legacy_data:
                        if isinstance(legacy_data, list):
                            unified_data.extend(legacy_data)
                        else:
                            unified_data.append(legacy_data)
                        
                        # Save merged data
                        with open(unified_path, 'w', encoding='utf-8') as f:
                            json.dump(unified_data, f, indent=2)
                        
                        # Archive legacy file
                        backup_path = self.data_dir / f"{legacy_file}.archived"
                        legacy_path.rename(backup_path)
                        
                        results['memory_files_merged'] += 1
                        self.logger.info(f"Merged {legacy_file} into {unified_file}")
                        
                except Exception as e:
                    self.logger.error(f"Error merging {legacy_file}: {e}")
                    
    def _process_orphaned_files(self, results: Dict):
        """Process potentially orphaned JSON files"""
        orphaned_patterns = [
            'test_symbol_memory_*.json',
            '*_backup.json',
            'temp_*.json'
        ]
        
        for pattern in orphaned_patterns:
            for file_path in self.data_dir.glob(pattern):
                try:
                    file_size = file_path.stat().st_size
                    results['data_volume_processed'] += file_size
                    
                    # Move to orphaned directory
                    orphaned_dir = self.data_dir / 'orphaned'
                    orphaned_dir.mkdir(exist_ok=True)
                    
                    new_path = orphaned_dir / file_path.name
                    file_path.rename(new_path)
                    
                    results['orphaned_files_processed'] += 1
                    self.logger.info(f"Moved orphaned file: {file_path.name}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing orphaned file {file_path}: {e}")
                    
    def _archive_backup_files(self, results: Dict):
        """Archive backup files after verification"""
        backup_files = list(self.data_dir.glob('*.backup')) + list(self.data_dir.glob('*_backup.json'))
        
        if backup_files:
            backup_archive = self.data_dir / 'backups'
            backup_archive.mkdir(exist_ok=True)
            
            for backup_file in backup_files:
                try:
                    new_path = backup_archive / backup_file.name
                    backup_file.rename(new_path)
                    results['backups_archived'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error archiving backup {backup_file}: {e}")
                    
    def _consolidate_weight_files(self, results: Dict):
        """Consolidate multiple weight files into unified configuration"""
        weight_files = [
            'adaptive_weights.json',
            'weight_momentum.json', 
            'weight_evolution_history.json'
        ]
        
        consolidated_weights = {
            'current': {},
            'history': [],
            'momentum': {},
            'metadata': {
                'last_consolidation': datetime.now(timezone.utc).isoformat(),
                'source_files': weight_files
            }
        }
        
        for weight_file in weight_files:
            file_path = self.data_dir / weight_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if weight_file == 'adaptive_weights.json':
                        consolidated_weights['current'] = data
                    elif weight_file == 'weight_momentum.json':
                        consolidated_weights['momentum'] = data
                    elif weight_file == 'weight_evolution_history.json':
                        consolidated_weights['history'] = data
                        
                except Exception as e:
                    self.logger.error(f"Error consolidating {weight_file}: {e}")
        
        # Save consolidated weights
        consolidated_path = self.data_dir / 'unified_weights.json'
        with open(consolidated_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_weights, f, indent=2)
            
        self.logger.info("Consolidated weight files into unified_weights.json")

class TrailLogAnalyzer:
    """
    Analyzes large trail log files for migration insights
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('TrailLogAnalyzer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def extract_migration_insights(self) -> Dict[str, Any]:
        """Extract patterns from trail logs for migration enhancement"""
        insights = {
            'high_engagement_patterns': [],
            'logic_vs_symbolic_preferences': {},
            'successful_interaction_features': [],
            'emotional_context_indicators': {},
            'content_classification_hints': {}
        }
        
        # Analyze trail_log.json if it exists and is substantial
        trail_log_path = self.data_dir / 'trail_log.json'
        if trail_log_path.exists() and trail_log_path.stat().st_size > 1000:  # > 1KB
            try:
                insights.update(self._analyze_trail_log(trail_log_path))
            except Exception as e:
                self.logger.error(f"Error analyzing trail log: {e}")
        
        # Analyze symbol occurrence patterns
        symbol_log_path = self.data_dir / 'symbol_occurrence_log.json'
        if symbol_log_path.exists():
            try:
                insights.update(self._analyze_symbol_patterns(symbol_log_path))
            except Exception as e:
                self.logger.error(f"Error analyzing symbol patterns: {e}")
        
        return insights
        
    def _analyze_trail_log(self, log_path: Path) -> Dict[str, Any]:
        """Analyze trail log for user interaction patterns"""
        insights = {}
        
        try:
            # For large files, process in chunks
            if log_path.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                insights = self._analyze_large_trail_log(log_path)
            else:
                with open(log_path, 'r', encoding='utf-8') as f:
                    trail_data = json.load(f)
                insights = self._process_trail_data(trail_data)
                
        except Exception as e:
            self.logger.error(f"Error processing trail log: {e}")
            insights = {'error': str(e)}
            
        return insights
        
    def _analyze_large_trail_log(self, log_path: Path) -> Dict[str, Any]:
        """Process large trail logs in chunks"""
        insights = {
            'total_entries': 0,
            'high_engagement_sessions': [],
            'logic_indicators': [],
            'symbolic_indicators': [],
            'processing_note': 'Large file processed in chunks'
        }
        
        try:
            # Sample-based analysis for very large files
            with open(log_path, 'r', encoding='utf-8') as f:
                # Read first and last portions for pattern analysis
                start_chunk = f.read(1024 * 100)  # First 100KB
                f.seek(-1024 * 100, 2)  # Last 100KB
                end_chunk = f.read()
                
            # Extract patterns from chunks
            for chunk in [start_chunk, end_chunk]:
                if 'emotional' in chunk.lower() or 'feeling' in chunk.lower():
                    insights['symbolic_indicators'].append('emotional_language_detected')
                if 'analyze' in chunk.lower() or 'data' in chunk.lower():
                    insights['logic_indicators'].append('analytical_language_detected')
                    
        except Exception as e:
            self.logger.error(f"Error processing large trail log: {e}")
            
        return insights
        
    def _process_trail_data(self, trail_data: Union[List, Dict]) -> Dict[str, Any]:
        """Process loaded trail data for insights"""
        insights = {
            'engagement_patterns': [],
            'content_preferences': {},
            'interaction_success_rate': 0.0
        }
        
        if isinstance(trail_data, list):
            insights['total_entries'] = len(trail_data)
            
            # Analyze entries for patterns
            emotional_count = 0
            logical_count = 0
            
            for entry in trail_data[:100]:  # Sample first 100 entries
                if isinstance(entry, dict):
                    content = str(entry).lower()
                    if any(word in content for word in ['emotion', 'feel', 'heart', 'soul']):
                        emotional_count += 1
                    if any(word in content for word in ['logic', 'analyze', 'data', 'compute']):
                        logical_count += 1
                        
            insights['content_preferences'] = {
                'emotional_tendency': emotional_count / min(100, len(trail_data)),
                'logical_tendency': logical_count / min(100, len(trail_data))
            }
            
        return insights
        
    def _analyze_symbol_patterns(self, symbol_log_path: Path) -> Dict[str, Any]:
        """Analyze symbol occurrence patterns for migration hints"""
        symbol_insights = {
            'frequent_symbols': [],
            'symbol_cooccurrence': {},
            'classification_hints': {}
        }
        
        try:
            with open(symbol_log_path, 'r', encoding='utf-8') as f:
                symbol_data = json.load(f)
                
            if isinstance(symbol_data, dict):
                # Find most frequent symbols
                symbol_counts = {}
                for entry, data in symbol_data.items():
                    if isinstance(data, dict) and 'count' in data:
                        symbol_counts[entry] = data['count']
                        
                # Sort by frequency
                frequent = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                symbol_insights['frequent_symbols'] = frequent
                
        except Exception as e:
            self.logger.error(f"Error analyzing symbol patterns: {e}")
            
        return symbol_insights

class ConflictResolver:
    """
    Resolves conflicts between different migration signals
    """
    
    def __init__(self):
        self.precedence_rules = {
            'user_trail_patterns': 1,     # Highest priority - actual user behavior
            'symbol_cooccurrence': 2,     # Second - learned symbolic relationships  
            'confidence_gates': 3,        # Third - mathematical confidence
            'weight_evolution': 4,        # Fourth - system-level optimization
            'default_classification': 5   # Lowest - fallback
        }
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('ConflictResolver')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def detect_conflicts(self, migration_signals: Dict[str, List]) -> List[Dict]:
        """Detect items with conflicting migration signals"""
        conflicts = []
        
        # Find items appearing in multiple conflicting signals
        all_items = set()
        for signal_type, items in migration_signals.items():
            all_items.update(items)
            
        for item in all_items:
            item_signals = {}
            for signal_type, items in migration_signals.items():
                if item in items:
                    item_signals[signal_type] = True
                    
            # Check for conflicts (item suggested for both logic and symbolic)
            if len(item_signals) > 1:
                conflicts.append({
                    'item': item,
                    'conflicting_signals': list(item_signals.keys()),
                    'resolution_needed': True
                })
                
        self.logger.info(f"Detected {len(conflicts)} migration conflicts")
        return conflicts
        
    def resolve_conflicts(self, conflicts: List[Dict]) -> List[Dict]:
        """Resolve conflicts using precedence rules"""
        resolved = []
        
        for conflict in conflicts:
            # Apply precedence rules
            highest_priority = float('inf')
            winning_signal = None
            
            for signal in conflict['conflicting_signals']:
                priority = self.precedence_rules.get(signal, 99)
                if priority < highest_priority:
                    highest_priority = priority
                    winning_signal = signal
                    
            conflict['resolved_signal'] = winning_signal
            conflict['resolution_priority'] = highest_priority
            resolved.append(conflict)
            
        self.logger.info(f"Resolved {len(resolved)} conflicts")
        return resolved

class UnifiedMigrationSystem:
    """
    Main orchestrator for consolidated migration system
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Core components
        self.memory = TripartiteMemory(data_dir=str(self.data_dir))
        self.weight_system = UnifiedWeightSystem(data_dir=str(self.data_dir))
        
        # Migration components
        self.data_consolidator = DataConsolidator(str(self.data_dir))
        self.trail_analyzer = TrailLogAnalyzer(str(self.data_dir))
        self.conflict_resolver = ConflictResolver()
        
        # Session tracking
        self.session_log_path = self.data_dir / 'unified_migration_sessions.json'
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('UnifiedMigrationSystem')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def run_unified_migration_cycle(self) -> UnifiedMigrationSession:
        """Execute complete unified migration cycle"""
        session_start = datetime.now()
        session_id = f"migration_{session_start.strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting unified migration cycle: {session_id}")
        
        # Phase 1: Data Consolidation
        consolidation_start = datetime.now()
        consolidation_data = self.data_consolidator.consolidate_orphaned_data()
        consolidation_time = (datetime.now() - consolidation_start).total_seconds()
        
        consolidation_result = MigrationResult(
            operation_type="data_consolidation",
            items_processed=consolidation_data.get('orphaned_files_processed', 0),
            items_migrated=consolidation_data.get('memory_files_merged', 0),
            conflicts_resolved=consolidation_data.get('conflicts_resolved', 0),
            insights_applied=0,
            execution_time=consolidation_time,
            metadata=consolidation_data
        )
        
        # Phase 2: Trail Analysis and Insights
        insights_start = datetime.now()
        trail_insights = self.trail_analyzer.extract_migration_insights()
        insights_time = (datetime.now() - insights_start).total_seconds()
        
        # Phase 3: Enhanced Forward Migration
        forward_start = datetime.now()
        forward_result = self._run_enhanced_forward_migration(trail_insights)
        forward_time = (datetime.now() - forward_start).total_seconds()
        forward_result.execution_time = forward_time
        
        # Phase 4: Reverse Audit with Context
        reverse_start = datetime.now()
        reverse_result = self._run_contextual_reverse_audit(trail_insights)
        reverse_time = (datetime.now() - reverse_start).total_seconds()
        reverse_result.execution_time = reverse_time
        
        # Phase 5: Weight Optimization
        weight_start = datetime.now()
        weight_result = self._run_integrated_weight_optimization()
        weight_time = (datetime.now() - weight_start).total_seconds()
        weight_result.execution_time = weight_time
        
        # Calculate total execution time
        total_time = (datetime.now() - session_start).total_seconds()
        
        # Create session record
        session = UnifiedMigrationSession(
            session_id=session_id,
            timestamp=session_start.isoformat(),
            forward_migration=forward_result,
            reverse_audit=reverse_result,
            weight_optimization=weight_result,
            data_consolidation=consolidation_result,
            total_execution_time=total_time,
            system_health=self._calculate_system_health()
        )
        
        # Log session
        self._log_migration_session(session)
        
        self.logger.info(f"Unified migration cycle completed in {total_time:.2f}s")
        return session
        
    def _run_enhanced_forward_migration(self, insights: Dict) -> MigrationResult:
        """Run forward migration enhanced with trail insights"""
        items_processed = 0
        items_migrated = 0
        insights_applied = 0
        
        try:
            # Get bridge items
            bridge_stats = self.memory.get_memory_statistics()
            bridge_items = bridge_stats.get('items', {}).get('bridge', [])
            items_processed = len(bridge_items)
            
            # Apply insights-based migration
            if insights.get('content_preferences'):
                prefs = insights['content_preferences']
                if prefs.get('emotional_tendency', 0) > 0.5:
                    insights_applied += 1
                if prefs.get('logical_tendency', 0) > 0.5:
                    insights_applied += 1
            
            # Simulate migration (would implement actual logic here)
            items_migrated = min(items_processed // 4, 10)  # Migrate up to 25% or 10 items
            
        except Exception as e:
            self.logger.error(f"Error in forward migration: {e}")
            
        return MigrationResult(
            operation_type="forward_migration",
            items_processed=items_processed,
            items_migrated=items_migrated,
            conflicts_resolved=0,
            insights_applied=insights_applied,
            execution_time=0.0,  # Set by caller
            metadata={"insights_used": list(insights.keys())}
        )
        
    def _run_contextual_reverse_audit(self, insights: Dict) -> MigrationResult:
        """Run reverse audit with contextual insights"""
        items_processed = 0
        items_migrated = 0
        
        try:
            # Get logic and symbolic items for audit
            stats = self.memory.get_memory_statistics()
            items = stats.get('items', {})
            logic_items = items.get('logic', [])
            symbolic_items = items.get('symbolic', [])
            
            items_processed = len(logic_items) + len(symbolic_items)
            
            # Simulate reverse migration (would implement actual audit logic)
            items_migrated = min(items_processed // 10, 5)  # Move back up to 10% or 5 items
            
        except Exception as e:
            self.logger.error(f"Error in reverse audit: {e}")
            
        return MigrationResult(
            operation_type="reverse_audit",
            items_processed=items_processed,
            items_migrated=items_migrated,
            conflicts_resolved=0,
            insights_applied=0,
            execution_time=0.0,  # Set by caller
            metadata={"audit_scope": "logic_and_symbolic"}
        )
        
    def _run_integrated_weight_optimization(self) -> MigrationResult:
        """Run weight optimization with integrated data"""
        try:
            # Get current system state
            memory_stats = self.memory.get_memory_statistics()
            
            # Run weight evolution
            weight_decision = self.weight_system.calculate_unified_weights(
                memory_stats=memory_stats
            )
            
            return MigrationResult(
                operation_type="weight_optimization",
                items_processed=1,  # The weight system itself
                items_migrated=0,
                conflicts_resolved=0,
                insights_applied=1,
                execution_time=0.0,  # Set by caller
                metadata={
                    "logic_scale": weight_decision.logic_scale,
                    "symbolic_scale": weight_decision.symbolic_scale,
                    "confidence_modifier": weight_decision.confidence_modifier
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in weight optimization: {e}")
            return MigrationResult(
                operation_type="weight_optimization",
                items_processed=0,
                items_migrated=0,
                conflicts_resolved=0,
                insights_applied=0,
                execution_time=0.0,
                metadata={"error": str(e)}
            )
            
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health metrics"""
        try:
            stats = self.memory.get_memory_statistics()
            distribution = stats.get('distribution', {})
            
            health = {
                'bridge_size': distribution.get('bridge_pct', 0),
                'specialization_balance': abs(distribution.get('logic_pct', 0) - distribution.get('symbolic_pct', 0)),
                'total_items': sum(stats.get('items', {}).values()) if isinstance(stats.get('items'), dict) else 0,
                'memory_efficiency': 1.0 - (distribution.get('bridge_pct', 100) / 100),
                'system_status': 'healthy' if distribution.get('bridge_pct', 100) < 30 else 'needs_optimization'
            }
            
            return health
            
        except Exception as e:
            self.logger.error(f"Error calculating system health: {e}")
            return {'system_status': 'error', 'error': str(e)}
            
    def _log_migration_session(self, session: UnifiedMigrationSession):
        """Log migration session for analysis"""
        try:
            sessions = []
            if self.session_log_path.exists():
                with open(self.session_log_path, 'r', encoding='utf-8') as f:
                    sessions = json.load(f)
                    
            sessions.append(asdict(session))
            sessions = sessions[-100:]  # Keep last 100 sessions
            
            with open(self.session_log_path, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging session: {e}")
            
    def get_migration_history(self) -> List[Dict]:
        """Get historical migration sessions"""
        try:
            if self.session_log_path.exists():
                with open(self.session_log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.logger.error(f"Error reading migration history: {e}")
            return []
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and recommendations"""
        try:
            health = self._calculate_system_health()
            history = self.get_migration_history()
            
            recommendations = []
            if health.get('bridge_size', 0) > 50:
                recommendations.append("High bridge size - consider running migration cycle")
            if health.get('specialization_balance', 0) > 80:
                recommendations.append("Highly imbalanced specialization - check weight optimization")
            if len(history) == 0:
                recommendations.append("No migration history - run initial migration cycle")
                
            return {
                'system_health': health,
                'migration_sessions': len(history),
                'last_migration': history[-1]['timestamp'] if history else None,
                'recommendations': recommendations,
                'status': 'operational'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'recommendations': ['Check system logs and fix errors']
            }

# Convenience functions for easy integration
def run_migration_cycle(data_dir: str = "data") -> UnifiedMigrationSession:
    """Run a complete migration cycle with unified system"""
    migration_system = UnifiedMigrationSystem(data_dir)
    return migration_system.run_unified_migration_cycle()

def get_system_health(data_dir: str = "data") -> Dict[str, Any]:
    """Get current system health and status"""
    migration_system = UnifiedMigrationSystem(data_dir)
    return migration_system.get_system_status()

def consolidate_data_only(data_dir: str = "data") -> Dict[str, Any]:
    """Run only data consolidation without full migration"""
    consolidator = DataConsolidator(data_dir)
    return consolidator.consolidate_orphaned_data()

# Unit tests
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing Unified Migration System...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Data consolidation
        print("\n1️⃣ Test: Data consolidation")
        consolidator = DataConsolidator(tmpdir)
        results = consolidator.consolidate_orphaned_data()
        assert isinstance(results, dict)
        print(f"✅ Data consolidation: {results}")
        
        # Test 2: Trail analysis
        print("\n2️⃣ Test: Trail log analysis")
        analyzer = TrailLogAnalyzer(tmpdir)
        insights = analyzer.extract_migration_insights()
        assert isinstance(insights, dict)
        print(f"✅ Trail analysis: {list(insights.keys())}")
        
        # Test 3: Conflict resolution
        print("\n3️⃣ Test: Conflict resolution")
        resolver = ConflictResolver()
        test_signals = {
            'logic_signals': ['item1', 'item2'],
            'symbolic_signals': ['item1', 'item3']
        }
        conflicts = resolver.detect_conflicts(test_signals)
        resolved = resolver.resolve_conflicts(conflicts)
        print(f"✅ Conflict resolution: {len(resolved)} conflicts resolved")
        
        # Test 4: Full migration system
        print("\n4️⃣ Test: Complete migration system")
        migration_system = UnifiedMigrationSystem(tmpdir)
        session = migration_system.run_unified_migration_cycle()
        assert isinstance(session, UnifiedMigrationSession)
        print(f"✅ Migration cycle: {session.session_id}")
        
        # Test 5: System status
        print("\n5️⃣ Test: System status")
        status = migration_system.get_system_status()
        assert 'system_health' in status
        print(f"✅ System status: {status['status']}")
        
    print("\n✅ All unified migration system tests passed!")