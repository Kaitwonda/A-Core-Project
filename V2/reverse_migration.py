# reverse_migration.py - Reverse Migration Audit System

from datetime import datetime
from adaptive_migration import MigrationEngine, evaluate_link_with_confidence_gates

class ReverseMigrationAuditor:
    """
    Audits items in logic/symbolic memory to catch misclassifications
    and move them back to bridge memory if needed.
    """
    
    def __init__(self, memory, confidence_threshold=0.3):
        self.memory = memory
        self.confidence_threshold = confidence_threshold
        self.reverse_log = []
        
    def audit_item(self, item, current_location):
        """
        Audit a single item to see if it should be moved back to bridge.
        
        Returns: (should_reverse, reason)
        """
        logic_score = item.get('logic_score', 0)
        symbolic_score = item.get('symbolic_score', 0)
        
        # Re-evaluate with current weights
        new_decision, new_score = evaluate_link_with_confidence_gates(
            logic_score, symbolic_score
        )
        
        # Get expected decision based on current location
        expected_decision = {
            'logic': 'FOLLOW_LOGIC',
            'symbolic': 'FOLLOW_SYMBOLIC',
            'bridge': 'FOLLOW_HYBRID'
        }[current_location]
        
        # Check if misclassified
        if new_decision != expected_decision:
            return True, f"Reclassified as {new_decision}"
            
        # Check if confidence too low
        if new_score < self.confidence_threshold:
            return True, f"Low confidence ({new_score:.2f})"
            
        # Check stability
        stability = self.memory.get_item_stability(item)
        if stability['history_length'] >= 5 and not stability['is_stable']:
            return True, "Chronically unstable"
            
        # Check if it's been flip-flopping
        if 'reverse_migration_count' in item and item['reverse_migration_count'] >= 2:
            return True, "Multiple reverse migrations"
            
        return False, "Item correctly classified"
        
    def audit_logic_memory(self):
        """Audit all items in logic memory"""
        reverse_count = 0
        new_logic = []
        
        print("\n🔍 Auditing logic memory...")
        
        for item in self.memory.logic_memory:
            should_reverse, reason = self.audit_item(item, 'logic')
            
            if should_reverse:
                # Add reverse migration metadata
                item['reverse_migrated'] = True
                item['reverse_migration_reason'] = reason
                item['reverse_migration_date'] = datetime.utcnow().isoformat()
                item['reverse_migration_count'] = item.get('reverse_migration_count', 0) + 1
                
                # Log it
                self.reverse_log.append({
                    'item_id': item.get('id', 'unknown'),
                    'text_preview': item.get('text', '')[:50],
                    'from': 'logic',
                    'to': 'bridge',
                    'reason': reason,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # Move to bridge
                self.memory.bridge_memory.append(item)
                reverse_count += 1
                
                print(f"  ← Moving to bridge ({reason}): {item.get('text', '')[:50]}...")
            else:
                new_logic.append(item)
                
        self.memory.logic_memory = new_logic
        return reverse_count
        
    def audit_symbolic_memory(self):
        """Audit all items in symbolic memory"""
        reverse_count = 0
        new_symbolic = []
        
        print("\n🔍 Auditing symbolic memory...")
        
        for item in self.memory.symbolic_memory:
            should_reverse, reason = self.audit_item(item, 'symbolic')
            
            if should_reverse:
                # Add reverse migration metadata
                item['reverse_migrated'] = True
                item['reverse_migration_reason'] = reason
                item['reverse_migration_date'] = datetime.utcnow().isoformat()
                item['reverse_migration_count'] = item.get('reverse_migration_count', 0) + 1
                
                # Log it
                self.reverse_log.append({
                    'item_id': item.get('id', 'unknown'),
                    'text_preview': item.get('text', '')[:50],
                    'from': 'symbolic',
                    'to': 'bridge',
                    'reason': reason,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # Move to bridge
                self.memory.bridge_memory.append(item)
                reverse_count += 1
                
                print(f"  ← Moving to bridge ({reason}): {item.get('text', '')[:50]}...")
            else:
                new_symbolic.append(item)
                
        self.memory.symbolic_memory = new_symbolic
        return reverse_count
        
    def audit_all(self):
        """Audit both logic and symbolic memories"""
        total_reversed = 0
        total_reversed += self.audit_logic_memory()
        total_reversed += self.audit_symbolic_memory()
        return total_reversed
        
    def get_audit_summary(self):
        """Get summary of reverse migrations"""
        if not self.reverse_log:
            return {
                'total_reversed': 0,
                'from_logic': 0,
                'from_symbolic': 0,
                'reasons': {}
            }
            
        # Count by source
        from_logic = len([r for r in self.reverse_log if r['from'] == 'logic'])
        from_symbolic = len([r for r in self.reverse_log if r['from'] == 'symbolic'])
        
        # Count by reason
        reasons = {}
        for record in self.reverse_log:
            reason = record['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
            
        return {
            'total_reversed': len(self.reverse_log),
            'from_logic': from_logic,
            'from_symbolic': from_symbolic,
            'reasons': reasons
        }
        

# Unit tests
if __name__ == "__main__":
    import tempfile
    from decision_history import HistoryAwareMemory
    
    print("🧪 Testing Reverse Migration Audit System...")
    
    # Test 1: Basic misclassification detection
    print("\n1️⃣ Test: Misclassification detection")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        auditor = ReverseMigrationAuditor(memory)
        
        # Add a symbolic item to logic memory (misclassified)
        misclassified = {
            'id': 'mis1',
            'text': 'A beautiful emotional story',
            'logic_score': 2.0,
            'symbolic_score': 8.0,
            'decision_history': [{
                'decision': 'FOLLOW_LOGIC',
                'timestamp': datetime.utcnow().isoformat(),
                'weights': {'static': 0.9, 'dynamic': 0.1}  # Wrong weights led to misclassification
            }]
        }
        memory.logic_memory.append(misclassified)
        
        # Audit should catch it
        should_reverse, reason = auditor.audit_item(misclassified, 'logic')
        assert should_reverse == True, "Should detect misclassification"
        assert "Reclassified" in reason
        
        print("✅ Misclassification detection works")
        
    # Test 2: Low confidence detection
    print("\n2️⃣ Test: Low confidence detection")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        auditor = ReverseMigrationAuditor(memory, confidence_threshold=5.0)
        
        # Add a low-confidence item
        low_conf = {
            'id': 'low1',
            'text': 'Ambiguous content',
            'logic_score': 3.0,
            'symbolic_score': 2.0,
            'decision_history': [{
                'decision': 'FOLLOW_LOGIC',
                'timestamp': datetime.utcnow().isoformat(),
                'weights': {'static': 0.6, 'dynamic': 0.4}
            }]
        }
        memory.logic_memory.append(low_conf)
        
        # Audit should catch it
        should_reverse, reason = auditor.audit_item(low_conf, 'logic')
        assert should_reverse == True, "Should detect low confidence"
        assert "Low confidence" in reason
        
        print("✅ Low confidence detection works")
        
    # Test 3: Full audit flow
    print("\n3️⃣ Test: Full audit flow")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        auditor = ReverseMigrationAuditor(memory)
        
        # Add various items
        # Correctly classified logic item
        correct_logic = {
            'id': 'correct1',
            'text': 'Clear algorithmic explanation',
            'logic_score': 9.0,
            'symbolic_score': 1.0
        }
        memory.store(correct_logic, 'FOLLOW_LOGIC')
        
        # Misclassified in logic (should be symbolic)
        wrong_in_logic = {
            'id': 'wrong1',
            'text': 'Emotional metaphorical content',
            'logic_score': 2.0,
            'symbolic_score': 8.0
        }
        memory.store(wrong_in_logic, 'FOLLOW_LOGIC')
        
        # Misclassified in symbolic (should be logic)
        wrong_in_symbolic = {
            'id': 'wrong2',
            'text': 'Technical documentation',
            'logic_score': 8.0,
            'symbolic_score': 2.0
        }
        memory.store(wrong_in_symbolic, 'FOLLOW_SYMBOLIC')
        
        # Run full audit
        initial_bridge = len(memory.bridge_memory)
        total_reversed = auditor.audit_all()
        
        # Check results
        assert total_reversed >= 2, f"Should reverse at least 2 items, got {total_reversed}"
        assert len(memory.bridge_memory) > initial_bridge, "Bridge should grow"
        assert len(memory.logic_memory) == 1, "Should keep only correct logic item"
        
        # Check summary
        summary = auditor.get_audit_summary()
        assert summary['total_reversed'] == total_reversed
        print(f"✅ Full audit works: {summary}")
        
    # Test 4: Flip-flop detection
    print("\n4️⃣ Test: Flip-flop detection")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        auditor = ReverseMigrationAuditor(memory)
        
        # Create item with flip-flop history and prior reverse migrations
        flipflop = {
            'id': 'flip1',
            'text': 'Constantly changing classification',
            'logic_score': 5.0,
            'symbolic_score': 5.0,
            'reverse_migration_count': 2,  # Already reversed twice
            'decision_history': []
        }
        
        # Add unstable history
        for i in range(5):
            decision = 'FOLLOW_LOGIC' if i % 2 == 0 else 'FOLLOW_SYMBOLIC'
            flipflop['decision_history'].append({
                'decision': decision,
                'timestamp': datetime.utcnow().isoformat(),
                'weights': {'static': 0.5, 'dynamic': 0.5}
            })
            
        memory.logic_memory.append(flipflop)
        
        # Should detect multiple reverse migrations
        should_reverse, reason = auditor.audit_item(flipflop, 'logic')
        assert should_reverse == True, "Should detect flip-flopping"
        assert "Multiple reverse migrations" in reason or "unstable" in reason.lower()
        
        print("✅ Flip-flop detection works")
        
    # Test 5: Persistence of reverse migration metadata
    print("\n5️⃣ Test: Reverse migration metadata persistence")
    with tempfile.TemporaryDirectory() as tmpdir:
        # First run - reverse some items
        memory1 = HistoryAwareMemory(data_dir=tmpdir)
        auditor1 = ReverseMigrationAuditor(memory1)
        
        # Add misclassified item
        item = {
            'text': 'Reversed item test',
            'logic_score': 1.0,
            'symbolic_score': 9.0
        }
        memory1.store(item, 'FOLLOW_LOGIC')
        
        auditor1.audit_all()
        memory1.save_all()
        
        # Second run - check metadata persists
        memory2 = HistoryAwareMemory(data_dir=tmpdir)
        
        # Find reversed item in bridge
        reversed_items = [i for i in memory2.bridge_memory if 'reverse_migrated' in i]
        assert len(reversed_items) > 0, "Reversed items should be in bridge"
        assert reversed_items[0]['reverse_migrated'] == True
        assert 'reverse_migration_reason' in reversed_items[0]
        assert reversed_items[0]['reverse_migration_count'] == 1
        
        print("✅ Reverse migration metadata persists")
        
    print("\n✅ All reverse migration tests passed!")