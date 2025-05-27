# adaptive_migration.py - Confidence-Based Migration System

import json
from pathlib import Path
from datetime import datetime
from decision_history import HistoryAwareMemory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Option 2: Updated local version with correct scales matching autonomous_learner
def evaluate_link_with_confidence_gates(logic_score, symbolic_score, logic_scale=2.0, sym_scale=1.0):
    """
    Local implementation of confidence gates - matching autonomous_learner scales.
    
    Returns (decision_type, final_score).
    decision_type: "FOLLOW_LOGIC", "FOLLOW_SYMBOLIC", or "FOLLOW_HYBRID"
    final_score: the score to use for threshold comparison.
    """
    # Normalize into [0,1]
    logic_conf = min(1.0, logic_score / logic_scale if logic_scale > 0 else (1.0 if logic_score > 0 else 0.0))
    sym_conf = min(1.0, symbolic_score / sym_scale if sym_scale > 0 else (1.0 if symbolic_score > 0 else 0.0))
    
    # High-confidence overrides
    if logic_conf > 0.8 and sym_conf < 0.3:
        return "FOLLOW_LOGIC", logic_score
    elif sym_conf > 0.8 and logic_conf < 0.3:
        return "FOLLOW_SYMBOLIC", symbolic_score
    else:
        # Hybrid blend (60% logic, 40% symbolic)
        combined = (logic_score * 0.6) + (symbolic_score * 0.4)
        return "FOLLOW_HYBRID", combined


class AdaptiveThresholds:
    """Manages time-varying migration thresholds"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.age_file = self.data_dir / "migration_age.json"
        self.age = self._load_age()
        
    def _load_age(self):
        """Load the migration age (number of runs)"""
        try:
            if self.age_file.exists():
                with open(self.age_file, 'r') as f:
                    data = json.load(f)
                    return data.get('age', 0)
        except Exception:
            pass
        return 0
        
    def _save_age(self):
        """Save the current age"""
        with open(self.age_file, 'w') as f:
            json.dump({'age': self.age, 'last_updated': datetime.utcnow().isoformat()}, f)
            
    def bump(self):
        """Increment age and save"""
        self.age += 1
        self._save_age()
        
    def get_migration_threshold(self):
        """
        Get current migration threshold.
        Starts high (0.9), decreases by 0.05 per run, floor at 0.3 (lowered from 0.6)
        """
        return max(0.3, 0.9 - (self.age * 0.05))
        

class MigrationEngine:
    """Handles migration of items between memory stores"""
    
    def __init__(self, memory: HistoryAwareMemory, thresholds: AdaptiveThresholds):
        self.memory = memory
        self.thresholds = thresholds
        self.migration_log = []
        
    def should_migrate(self, item, new_decision, new_score, threshold):
        """
        Determine if an item should be migrated based on:
        1. Score exceeds threshold
        2. Decision history shows stability
        3. No recent ping-ponging
        """
        # Must exceed threshold
        if new_score < threshold:
            return False, "Score below threshold"
            
        # Get stability info
        stability = self.memory.get_item_stability(item)
        
        # Need sufficient history
        if stability['history_length'] < 3:
            return False, "Insufficient history"
            
        # Check recent consistency
        if not stability['recent_consistency']:
            return False, "Recent decisions inconsistent"
            
        # Check for ping-ponging (too many different decisions)
        if len(stability['decision_counts']) > 2:
            return False, "Too much ping-ponging"
            
        # If dominant decision matches new decision, good to migrate
        if stability['dominant_decision'] == new_decision:
            return True, "Stable and consistent"
            
        # If new decision is different but very confident, allow it
        if new_score > threshold + 0.1:  # Extra margin
            return True, "High confidence override"
            
        return False, "Decision mismatch"
    
    def find_similar_items(self, item, all_items, similarity_threshold=0.8):
        """
        Find items similar to the given item based on text similarity.
        Uses cosine similarity of logic/symbolic scores as a simple approach.
        """
        similar_items = []
        
        # Create feature vector for the item (logic_score, symbolic_score)
        item_vector = np.array([item.get('logic_score', 0), item.get('symbolic_score', 0)]).reshape(1, -1)
        
        for other_item in all_items:
            if other_item.get('id') == item.get('id'):
                continue  # Skip self
                
            # Create feature vector for other item
            other_vector = np.array([other_item.get('logic_score', 0), other_item.get('symbolic_score', 0)]).reshape(1, -1)
            
            # Calculate cosine similarity
            if np.any(item_vector) and np.any(other_vector):  # Avoid zero vectors
                similarity = cosine_similarity(item_vector, other_vector)[0][0]
                
                if similarity >= similarity_threshold:
                    similar_items.append(other_item)
                    
        return similar_items
    
    def should_migrate_by_overlap(self, item, bridge_items):
        """
        Simple 5-overlap rule: If 5+ similar items in bridge have same classification,
        they can all migrate together.
        
        This leverages Wikipedia's redundancy - if the same fact appears 5+ times
        with consistent classification, it's probably correct.
        """
        # Find similar items (high similarity in scores)
        similar_items = self.find_similar_items(item, bridge_items, similarity_threshold=0.8)
        
        # Include the item itself in the count
        similar_items_including_self = similar_items + [item]
        
        if len(similar_items_including_self) >= 5:
            # Check if they all lean the same way
            logic_leaning = sum(1 for i in similar_items_including_self 
                              if i.get('logic_score', 0) > i.get('symbolic_score', 0))
            symbolic_leaning = len(similar_items_including_self) - logic_leaning
            
            # Strong consensus for logic
            if logic_leaning >= 5:
                return "FOLLOW_LOGIC", "5+ similar items agree on logic classification"
            # Strong consensus for symbolic
            elif symbolic_leaning >= 5:
                return "FOLLOW_SYMBOLIC", "5+ similar items agree on symbolic classification"
                
        return None, "Not enough overlap consensus"
        
    def migrate_from_bridge(self):
        """
        Migrate items from bridge memory to logic/symbolic based on confidence.
        Now includes both traditional migration AND the 5-overlap rule.
        Returns number of items migrated.
        """
        threshold = self.thresholds.get_migration_threshold()
        migrated = 0
        new_bridge = []
        items_to_migrate = []  # Collect items to migrate
        
        print(f"\n🔄 Migration attempt with threshold: {threshold:.2f}")
        print(f"  Checking {len(self.memory.bridge_memory)} bridge items...")
        
        # First pass: identify items to migrate
        for item in self.memory.bridge_memory:
            # Check the 5-overlap rule first
            overlap_decision, overlap_reason = self.should_migrate_by_overlap(item, self.memory.bridge_memory)
            
            if overlap_decision:
                # 5-overlap rule triggered!
                items_to_migrate.append({
                    'item': item,
                    'decision': overlap_decision,
                    'reason': overlap_reason,
                    'method': 'overlap'
                })
                print(f"  ✓ Overlap rule: {item.get('text', '')[:50]}... → {overlap_decision}")
            else:
                # Try traditional migration
                logic_score = item.get('logic_score', 0)
                symbolic_score = item.get('symbolic_score', 0)
                
                new_decision, new_score = evaluate_link_with_confidence_gates(
                    logic_score, symbolic_score
                )
                
                # Check if should migrate traditionally
                should_move, reason = self.should_migrate(item, new_decision, new_score, threshold)
                
                if should_move and new_decision != "FOLLOW_HYBRID":
                    items_to_migrate.append({
                        'item': item,
                        'decision': new_decision,
                        'reason': reason,
                        'method': 'traditional'
                    })
                    print(f"  → Traditional: {item.get('text', '')[:50]}... → {new_decision}")
                else:
                    # Keep in bridge
                    new_bridge.append(item)
                    if new_decision != "FOLLOW_HYBRID" and len(new_bridge) <= 5:  # Limit logging
                        print(f"  ✗ Kept in bridge ({reason}): {item.get('text', '')[:50]}...")
        
        # Second pass: actually migrate items
        for migration in items_to_migrate:
            item = migration['item']
            decision = migration['decision']
            
            # Log the migration
            self.migration_log.append({
                'item_id': item.get('id', 'unknown'),
                'text_preview': item.get('text', '')[:50],
                'from': 'bridge',
                'to': decision.replace('FOLLOW_', '').lower(),
                'threshold': threshold,
                'reason': migration['reason'],
                'method': migration['method'],
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Add migration metadata
            item['migrated_from'] = 'bridge'
            item['migration_threshold'] = threshold
            item['migration_date'] = datetime.utcnow().isoformat()
            item['migration_method'] = migration['method']
            
            # Store in new location
            self.memory.store(item, decision)
            migrated += 1
                    
        # Update bridge memory
        self.memory.bridge_memory = new_bridge
        
        print(f"\n  Summary: Migrated {migrated} items ({len(items_to_migrate)} by overlap rule)")
        
        return migrated
        
    def get_migration_summary(self):
        """Get summary of migrations in this session"""
        if not self.migration_log:
            return {
                'total_migrated': 0,
                'to_logic': 0,
                'to_symbolic': 0,
                'by_overlap': 0,
                'by_traditional': 0
            }
            
        summary = {
            'total_migrated': len(self.migration_log),
            'to_logic': len([m for m in self.migration_log if m['to'] == 'logic']),
            'to_symbolic': len([m for m in self.migration_log if m['to'] == 'symbolic']),
            'by_overlap': len([m for m in self.migration_log if m['method'] == 'overlap']),
            'by_traditional': len([m for m in self.migration_log if m['method'] == 'traditional'])
        }
        
        return summary
        

# Unit tests
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing Adaptive Migration System with 5-Overlap Rule...")
    
    # Test 1: Threshold decay
    print("\n1️⃣ Test: Adaptive threshold decay")
    with tempfile.TemporaryDirectory() as tmpdir:
        thresh = AdaptiveThresholds(data_dir=tmpdir)
        
        # Initial threshold
        assert thresh.get_migration_threshold() == 0.9, "Initial threshold should be 0.9"
        
        # After some runs
        for _ in range(3):
            thresh.bump()
        assert thresh.get_migration_threshold() == 0.75, "Threshold should decay"
        
        # Test new lower floor
        for _ in range(10):
            thresh.bump()
        assert thresh.get_migration_threshold() == 0.3, "Threshold should not go below 0.3"
        
        print("✅ Threshold decay works correctly with new floor")
        
    # Test 2: Confidence gates with correct scales
    print("\n2️⃣ Test: Confidence gates with updated scales")
    
    # Test with new scales (2.0/1.0)
    decision, score = evaluate_link_with_confidence_gates(1.8, 0.2)
    assert decision == "FOLLOW_LOGIC", f"Expected FOLLOW_LOGIC, got {decision}"
    
    decision, score = evaluate_link_with_confidence_gates(0.3, 0.9)
    assert decision == "FOLLOW_SYMBOLIC", f"Expected FOLLOW_SYMBOLIC, got {decision}"
    
    decision, score = evaluate_link_with_confidence_gates(1.0, 0.5)
    assert decision == "FOLLOW_HYBRID", f"Expected FOLLOW_HYBRID, got {decision}"
    
    print("✅ Confidence gates work with correct scales")
    
    # Test 3: 5-Overlap Rule
    print("\n3️⃣ Test: 5-Overlap Rule")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        thresh = AdaptiveThresholds(data_dir=tmpdir)
        engine = MigrationEngine(memory, thresh)
        
        # Add 6 similar items to bridge (all logic-leaning)
        for i in range(6):
            item = {
                'id': f'similar_{i}',
                'text': f'Computer algorithm explanation variant {i}',
                'logic_score': 8.0 + (i * 0.1),  # Slight variation
                'symbolic_score': 2.0
            }
            memory.store(item, 'FOLLOW_HYBRID')
            
        # Add 4 similar symbolic items (not enough for migration)
        for i in range(4):
            item = {
                'id': f'symbolic_{i}',
                'text': f'Emotional narrative variant {i}',
                'logic_score': 2.0,
                'symbolic_score': 8.0 + (i * 0.1)
            }
            memory.store(item, 'FOLLOW_HYBRID')
            
        # Run migration
        initial_bridge = len(memory.bridge_memory)
        migrated = engine.migrate_from_bridge()
        
        # Check results
        assert migrated >= 6, f"Should migrate at least 6 logic items by overlap, but migrated {migrated}"
        assert len(memory.bridge_memory) == 4, f"Should have 4 items left in bridge, but has {len(memory.bridge_memory)}"
        
        # Check migration summary
        summary = engine.get_migration_summary()
        assert summary['by_overlap'] >= 6, "Should have migrated 6 by overlap rule"
        assert summary['to_logic'] >= 6, "Should have migrated 6 to logic"
        
        print(f"✅ 5-Overlap rule works: {summary}")
        
    # Test 4: Mixed migration (overlap + traditional)
    print("\n4️⃣ Test: Mixed migration methods")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        thresh = AdaptiveThresholds(data_dir=tmpdir)
        engine = MigrationEngine(memory, thresh)
        
        # Add items for overlap rule
        for i in range(5):
            item = {
                'id': f'overlap_{i}',
                'text': f'Clear factual content {i}',
                'logic_score': 9.0,
                'symbolic_score': 1.0
            }
            memory.store(item, 'FOLLOW_HYBRID')
            
        # Add item for traditional migration (with history)
        traditional_item = {
            'id': 'traditional_1',
            'text': 'Stable content with history',
            'logic_score': 8.0,
            'symbolic_score': 2.0
        }
        # Build stable history
        for _ in range(3):
            memory.store(traditional_item.copy(), 'FOLLOW_HYBRID')
            
        # Run migration
        migrated = engine.migrate_from_bridge()
        summary = engine.get_migration_summary()
        
        print(f"✅ Mixed migration: {summary}")
        assert summary['by_overlap'] >= 5, "Should migrate by overlap"
        assert summary['total_migrated'] >= 5, "Should migrate multiple items"
        
    print("\n✅ All adaptive migration tests passed with 5-overlap rule!")