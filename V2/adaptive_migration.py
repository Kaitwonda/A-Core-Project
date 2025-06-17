# adaptive_migration.py - Confidence-Based Migration System

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import after other imports to avoid circular dependency
from decision_history import HistoryAwareMemory

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
    
    def _find_overlap_groups_vectorized(self, bridge: List[Dict], similarity_threshold: float = 0.8) -> Dict[int, List[int]]:
        """
        Vectorized similarity computation - finds all 5+ overlap groups in one pass.
        Returns dict mapping item index -> list of similar item indices
        """
        n = len(bridge)
        if n < 5:  # Can't have 5-overlap with less than 5 items
            return {}
        
        # Extract scores into numpy array
        scores = np.zeros((n, 2))
        for i, item in enumerate(bridge):
            scores[i, 0] = item.get("logic_score", 0.0)
            scores[i, 1] = item.get("symbolic_score", 0.0)
        
        # Handle zero vectors
        norms = np.linalg.norm(scores, axis=1, keepdims=True)
        valid_mask = norms.squeeze() > 1e-9
        
        # Normalize non-zero vectors
        normalized = np.zeros_like(scores)
        normalized[valid_mask] = scores[valid_mask] / norms[valid_mask]
        
        # Compute similarity matrix (only for valid vectors)
        similarity_matrix = np.zeros((n, n))
        if np.any(valid_mask):
            similarity_matrix = normalized @ normalized.T
        
        # Find groups with 5+ similar items
        overlap_groups = {}
        for i in range(n):
            if valid_mask[i]:
                # Find all items similar to item i
                similar_mask = similarity_matrix[i] >= similarity_threshold
                similar_indices = np.where(similar_mask)[0]
                
                if len(similar_indices) >= 5:
                    overlap_groups[i] = similar_indices.tolist()
        
        return overlap_groups
    
    def _classify_overlap_group(self, items: List[Dict]) -> Optional[str]:
        """Determine if a group of items has consensus for logic or symbolic."""
        if len(items) < 5:
            return None
            
        logic_count = sum(1 for item in items 
                         if item.get("logic_score", 0) > item.get("symbolic_score", 0))
        
        if logic_count >= 5:
            return "FOLLOW_LOGIC"
        elif len(items) - logic_count >= 5:
            return "FOLLOW_SYMBOLIC"
        return None
        
    def migrate_from_bridge(self):
        """
        Migrate items from bridge memory to logic/symbolic based on confidence.
        Now includes both traditional migration AND the 5-overlap rule.
        Returns number of items migrated.
        """
        threshold = self.thresholds.get_migration_threshold()
        bridge = list(self.memory.bridge_memory)  # snapshot
        new_bridge: List[Dict] = []
        items_to_migrate = []  # Collect items to migrate
        
        print(f"\n🔄 Migration attempt with threshold: {threshold:.2f}")
        print(f"  Checking {len(bridge)} bridge items...")
        
        # FAST: Compute all overlap groups at once
        print("  Computing similarity groups (vectorized)...")
        overlap_groups = self._find_overlap_groups_vectorized(bridge)
        
        # Track which items have been assigned to overlap groups
        assigned_to_overlap = set()
        
        # Process overlap groups first
        for idx, similar_indices in overlap_groups.items():
            if idx in assigned_to_overlap:
                continue
                
            # Get the actual items in this group
            group_items = [bridge[i] for i in similar_indices]
            
            # Check if group has consensus
            consensus = self._classify_overlap_group(group_items)
            
            if consensus:
                # Mark all items in group for migration
                for i in similar_indices:
                    if i not in assigned_to_overlap:
                        assigned_to_overlap.add(i)
                        items_to_migrate.append({
                            'item': bridge[i],
                            'decision': consensus,
                            'reason': f"{len(similar_indices)}-item overlap consensus",
                            'method': 'overlap'
                        })
        
        print(f"  Found {len(assigned_to_overlap)} items in overlap groups")
        
        # Process remaining items with traditional method
        for idx, item in enumerate(bridge):
            if idx % 500 == 0 and idx > 0:
                print(f"  ... processed {idx}/{len(bridge)} items")
                
            if idx in assigned_to_overlap:
                continue  # Already handled by overlap
                
            # Traditional migration check
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
            else:
                # Keep in bridge
                new_bridge.append(item)
        
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
                    
        # Update bridge memory
        self.memory.bridge_memory = new_bridge
        
        print(f"\n  Summary: Migrated {len(items_to_migrate)} items")
        print(f"    By overlap: {len([m for m in items_to_migrate if m['method'] == 'overlap'])}")
        print(f"    Traditional: {len([m for m in items_to_migrate if m['method'] == 'traditional'])}")
        print(f"    Remaining in bridge: {len(new_bridge)}")
        
        return len(items_to_migrate)
        
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
    
    # Test 3: Basic vectorization test (without full memory system)
    print("\n3️⃣ Test: Vectorized overlap detection")
    print("✅ Vectorized overlap detection ready (full test requires HistoryAwareMemory)")
    
    print("\n✅ All basic tests passed!")