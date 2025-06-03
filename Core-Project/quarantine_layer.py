# quarantine_layer.py - User Memory Quarantine System

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import your modules
from vector_engine import encode_with_minilm
from alphawall import AlphaWall


class UserMemoryQuarantine:
    """
    Quarantine system that works with AlphaWall zone outputs.
    Isolates problematic patterns without exposing user data to the AI.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Quarantine storage
        self.quarantine_dir = self.data_dir / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        # Files
        self.quarantine_log = self.quarantine_dir / "quarantine_log.json"
        self.pattern_database = self.quarantine_dir / "pattern_database.json"
        self.contamination_index = self.quarantine_dir / "contamination_index.json"
        
        # Initialize files
        self._init_quarantine_files()
        
        # AlphaWall reference for zone data only
        self.alphawall = AlphaWall(data_dir=data_dir)
        
        # Quarantine thresholds
        self.recursion_limit = 5
        self.toxicity_threshold = 0.7
        self.contamination_decay_hours = 24
        
    def _init_quarantine_files(self):
        """Initialize quarantine storage files"""
        for file_path in [self.quarantine_log, self.pattern_database, self.contamination_index]:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump([] if 'log' in str(file_path) else {}, f)
                    
    def quarantine(self, zone_id: str, reason: str = "automatic", severity: str = "medium") -> Dict:
        """
        Quarantine a zone output based on AlphaWall's recommendation.
        NEVER accesses user data directly - only works with zone outputs.
        """
        # Get zone output from AlphaWall
        zone_output = self.alphawall.get_zone_output_by_id(zone_id)
        if not zone_output:
            return {'success': False, 'error': 'Zone output not found'}
            
        # Create quarantine record (no user data included)
        quarantine_record = {
            'quarantine_id': self._generate_quarantine_id(zone_id),
            'zone_id': zone_id,
            'timestamp': datetime.utcnow().isoformat(),
            'reason': reason,
            'severity': severity,
            'zone_tags': zone_output['tags'],  # Only semantic tags
            'pattern_signature': self._generate_pattern_signature(zone_output),
            'expires_at': self._calculate_expiry(severity),
            'contamination_vector': self._analyze_contamination_vector(zone_output)
        }
        
        # Log the quarantine
        self._add_to_quarantine_log(quarantine_record)
        
        # Update pattern database
        self._update_pattern_database(quarantine_record)
        
        # Update contamination index
        self._update_contamination_index(quarantine_record)
        
        return {
            'success': True,
            'quarantine_id': quarantine_record['quarantine_id'],
            'severity': severity,
            'expires_at': quarantine_record['expires_at']
        }
    
    def _generate_quarantine_id(self, zone_id: str) -> str:
        """Generate unique quarantine ID"""
        return hashlib.md5(f"{zone_id}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
    
    def _generate_pattern_signature(self, zone_output: Dict) -> str:
        """
        Generate a pattern signature from zone tags.
        This helps identify similar patterns without storing user data.
        """
        tags = zone_output.get('tags', {})
        
        # Create signature from tag combination
        signature_parts = [
            f"emo:{tags.get('emotional_state', 'unknown')}",
            f"int:{tags.get('intent', 'unknown')}",
            f"ctx:{','.join(sorted(tags.get('context', [])))}",
            f"risk:{','.join(sorted(tags.get('risk', [])))}"
        ]
        
        return "|".join(signature_parts)
    
    def _calculate_expiry(self, severity: str) -> str:
        """Calculate when quarantine expires based on severity"""
        hours_map = {
            'low': 6,
            'medium': 24,
            'high': 72,
            'critical': 168  # 1 week
        }
        
        hours = hours_map.get(severity, 24)
        expiry = datetime.utcnow().timestamp() + (hours * 3600)
        return datetime.fromtimestamp(expiry).isoformat()
    
    def _analyze_contamination_vector(self, zone_output: Dict) -> Dict:
        """
        Analyze how this pattern might contaminate the AI's responses.
        Creates a contamination profile without exposing user data.
        """
        tags = zone_output.get('tags', {})
        
        contamination_vector = {
            'type': 'unknown',
            'spread_risk': 'low',
            'affects_nodes': []
        }
        
        # Determine contamination type
        if 'trauma_loop' in tags.get('context', []):
            contamination_vector['type'] = 'recursive_trauma'
            contamination_vector['spread_risk'] = 'high'
            contamination_vector['affects_nodes'] = ['symbolic', 'bridge']
            
        elif 'user_reliability_low' in tags.get('risk', []):
            contamination_vector['type'] = 'adversarial_pattern'
            contamination_vector['spread_risk'] = 'medium'
            contamination_vector['affects_nodes'] = ['logic', 'bridge']
            
        elif tags.get('emotional_state') == 'emotionally_recursive':
            contamination_vector['type'] = 'emotional_loop'
            contamination_vector['spread_risk'] = 'high'
            contamination_vector['affects_nodes'] = ['symbolic']
            
        elif 'symbolic_overload_possible' in tags.get('risk', []):
            contamination_vector['type'] = 'symbolic_overflow'
            contamination_vector['spread_risk'] = 'medium'
            contamination_vector['affects_nodes'] = ['symbolic', 'bridge']
            
        return contamination_vector
    
    def _add_to_quarantine_log(self, record: Dict):
        """Add record to quarantine log"""
        with open(self.quarantine_log, 'r') as f:
            log = json.load(f)
            
        log.append(record)
        
        # Keep last 1000 entries
        log = log[-1000:]
        
        with open(self.quarantine_log, 'w') as f:
            json.dump(log, f, indent=2)
    
    def _update_pattern_database(self, record: Dict):
        """Update pattern database with frequency counts"""
        with open(self.pattern_database, 'r') as f:
            patterns = json.load(f)
            
        pattern_sig = record['pattern_signature']
        
        if pattern_sig not in patterns:
            patterns[pattern_sig] = {
                'first_seen': record['timestamp'],
                'count': 0,
                'severities': [],
                'contamination_types': []
            }
            
        patterns[pattern_sig]['count'] += 1
        patterns[pattern_sig]['last_seen'] = record['timestamp']
        patterns[pattern_sig]['severities'].append(record['severity'])
        patterns[pattern_sig]['contamination_types'].append(
            record['contamination_vector']['type']
        )
        
        with open(self.pattern_database, 'w') as f:
            json.dump(patterns, f, indent=2)
    
    def _update_contamination_index(self, record: Dict):
        """Update contamination index for tracking spread"""
        with open(self.contamination_index, 'r') as f:
            index = json.load(f)
            
        contamination_type = record['contamination_vector']['type']
        
        if contamination_type not in index:
            index[contamination_type] = {
                'total_occurrences': 0,
                'affected_zones': [],
                'risk_evolution': []
            }
            
        index[contamination_type]['total_occurrences'] += 1
        index[contamination_type]['affected_zones'].append(record['zone_id'])
        index[contamination_type]['risk_evolution'].append({
            'timestamp': record['timestamp'],
            'risk_level': record['contamination_vector']['spread_risk']
        })
        
        # Keep only last 100 affected zones per type
        index[contamination_type]['affected_zones'] = \
            index[contamination_type]['affected_zones'][-100:]
            
        with open(self.contamination_index, 'w') as f:
            json.dump(index, f, indent=2)
    
    def check_contamination_risk(self, zone_output: Dict) -> Dict:
        """
        Check if a new zone output might be contaminated by quarantined patterns.
        This is used by the Bridge to adjust its decision-making.
        """
        # Generate signature for comparison
        new_signature = self._generate_pattern_signature(zone_output)
        
        # Load pattern database
        with open(self.pattern_database, 'r') as f:
            patterns = json.load(f)
            
        # Check for exact match
        if new_signature in patterns:
            pattern_data = patterns[new_signature]
            return {
                'contamination_detected': True,
                'risk_level': 'high',
                'pattern_frequency': pattern_data['count'],
                'recommendation': 'defer_to_symbolic' if 'trauma' in new_signature else 'defer_to_logic'
            }
            
        # Check for partial matches (similar patterns)
        partial_matches = []
        new_parts = set(new_signature.split('|'))
        
        for pattern_sig, pattern_data in patterns.items():
            pattern_parts = set(pattern_sig.split('|'))
            overlap = len(new_parts.intersection(pattern_parts)) / len(new_parts.union(pattern_parts))
            
            if overlap > 0.6:  # 60% similarity threshold
                partial_matches.append({
                    'pattern': pattern_sig,
                    'overlap': overlap,
                    'frequency': pattern_data['count']
                })
                
        if partial_matches:
            # Sort by overlap and frequency
            partial_matches.sort(key=lambda x: (x['overlap'], x['frequency']), reverse=True)
            
            return {
                'contamination_detected': True,
                'risk_level': 'medium',
                'similar_patterns': len(partial_matches),
                'closest_match': partial_matches[0]['pattern'],
                'recommendation': 'increase_bridge_caution'
            }
            
        return {
            'contamination_detected': False,
            'risk_level': 'low',
            'recommendation': 'proceed_normal'
        }
    
    def get_active_quarantines(self) -> List[Dict]:
        """Get all active (non-expired) quarantines"""
        with open(self.quarantine_log, 'r') as f:
            log = json.load(f)
            
        current_time = datetime.utcnow()
        active = []
        
        for record in log:
            expiry_time = datetime.fromisoformat(record['expires_at'].replace('Z', '+00:00'))
            if expiry_time > current_time:
                active.append({
                    'quarantine_id': record['quarantine_id'],
                    'pattern': record['pattern_signature'],
                    'severity': record['severity'],
                    'time_remaining': str(expiry_time - current_time)
                })
                
        return active
    
    def get_quarantine_statistics(self) -> Dict:
        """Get statistics about quarantine system"""
        with open(self.quarantine_log, 'r') as f:
            log = json.load(f)
            
        with open(self.pattern_database, 'r') as f:
            patterns = json.load(f)
            
        with open(self.contamination_index, 'r') as f:
            contamination = json.load(f)
            
        # Calculate stats
        total_quarantines = len(log)
        active_quarantines = len(self.get_active_quarantines())
        
        severity_counts = {}
        for record in log:
            sev = record['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
        # Most common patterns
        top_patterns = sorted(
            [(sig, data['count']) for sig, data in patterns.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_quarantines': total_quarantines,
            'active_quarantines': active_quarantines,
            'severity_distribution': severity_counts,
            'unique_patterns': len(patterns),
            'top_patterns': top_patterns,
            'contamination_types': list(contamination.keys()),
            'highest_risk_contamination': max(
                contamination.items(),
                key=lambda x: x[1]['total_occurrences']
            )[0] if contamination else None
        }
    
    def load_all_quarantined_memory(self) -> List[Dict]:
        """
        Load all quarantined records for contamination checking.
        Returns only pattern data, never user content.
        """
        with open(self.quarantine_log, 'r') as f:
            log = json.load(f)
            
        # Return sanitized records
        sanitized = []
        for record in log:
            sanitized.append({
                'pattern_signature': record['pattern_signature'],
                'zone_tags': record['zone_tags'],
                'contamination_type': record['contamination_vector']['type'],
                'severity': record['severity']
            })
            
        return sanitized


# Integration with Bridge decision-making
class QuarantineAwareBridge:
    """
    Example of how the Bridge uses quarantine information.
    """
    
    def __init__(self, quarantine_layer: UserMemoryQuarantine):
        self.quarantine = quarantine_layer
        
    def process_with_quarantine_check(self, zone_output: Dict, logic_score: float, symbolic_score: float) -> Dict:
        """
        Process zone output with quarantine contamination checking.
        """
        # Check contamination risk
        contamination = self.quarantine.check_contamination_risk(zone_output)
        
        # Adjust processing based on contamination
        if contamination['contamination_detected']:
            if contamination['risk_level'] == 'high':
                # High contamination - be very careful
                return {
                    'decision': contamination['recommendation'],
                    'confidence': 0.3,  # Low confidence due to contamination
                    'contamination_adjusted': True,
                    'reason': f"High contamination risk detected: {contamination.get('pattern_frequency', 0)} occurrences"
                }
            elif contamination['risk_level'] == 'medium':
                # Medium contamination - increase caution
                if contamination['recommendation'] == 'increase_bridge_caution':
                    # Reduce confidence in bridge decisions
                    confidence_penalty = 0.2
                    return {
                        'decision': 'FOLLOW_HYBRID',
                        'confidence': max(0.3, min(logic_score, symbolic_score) - confidence_penalty),
                        'contamination_adjusted': True,
                        'reason': f"Similar patterns detected: {contamination.get('similar_patterns', 0)} matches"
                    }
                    
        # No contamination - proceed normally
        return {
            'decision': 'NORMAL_PROCESSING',
            'confidence': max(logic_score, symbolic_score),
            'contamination_adjusted': False
        }


# Unit tests
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing Quarantine Layer Integration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize systems
        alphawall = AlphaWall(data_dir=tmpdir)
        quarantine = UserMemoryQuarantine(data_dir=tmpdir)
        
        # Test 1: Basic quarantine from zone output
        print("\n1️⃣ Test: Basic quarantine functionality")
        
        # Create a problematic input
        problem_input = "Why does everything hurt? Why does everything hurt? Why does everything hurt?"
        zone_output = alphawall.process_input(problem_input)
        
        # Quarantine it
        result = quarantine.quarantine(
            zone_output['zone_id'],
            reason="recursive_pattern_detected",
            severity="high"
        )
        
        assert result['success'] == True
        assert result['severity'] == 'high'
        print("✅ Basic quarantine works")
        
        # Test 2: Pattern detection
        print("\n2️⃣ Test: Pattern signature generation")
        
        # Process similar but not identical input
        similar_input = "Everything hurts so much, why?"
        zone_output2 = alphawall.process_input(similar_input)
        
        # Check contamination
        contamination = quarantine.check_contamination_risk(zone_output2)
        
        assert contamination['contamination_detected'] == True
        assert contamination['risk_level'] in ['medium', 'high']
        print(f"✅ Contamination detected: {contamination}")
        
        # Test 3: Statistics
        print("\n3️⃣ Test: Quarantine statistics")
        
        stats = quarantine.get_quarantine_statistics()
        assert stats['total_quarantines'] >= 1
        assert stats['active_quarantines'] >= 1
        assert 'high' in stats['severity_distribution']
        print(f"✅ Statistics: {stats}")
        
        # Test 4: Bridge integration
        print("\n4️⃣ Test: Bridge with quarantine awareness")
        
        bridge = QuarantineAwareBridge(quarantine)
        
        # Process with contamination check
        bridge_decision = bridge.process_with_quarantine_check(
            zone_output2,
            logic_score=0.7,
            symbolic_score=0.8
        )
        
        assert bridge_decision['contamination_adjusted'] == True
        assert bridge_decision['confidence'] < 0.8  # Reduced due to contamination
        print(f"✅ Bridge decision adjusted: {bridge_decision}")
        
        # Test 5: No user data leakage
        print("\n5️⃣ Test: Verify no user data in quarantine")
        
        all_quarantined = quarantine.load_all_quarantined_memory()
        
        # Check that no actual user text is in the quarantine data
        quarantine_str = str(all_quarantined)
        assert problem_input not in quarantine_str
        assert similar_input not in quarantine_str
        assert 'hurt' not in quarantine_str  # User's actual words don't appear
        
        print("✅ No user data leakage confirmed")
        
        # Test 6: Active quarantines
        print("\n6️⃣ Test: Active quarantine tracking")
        
        active = quarantine.get_active_quarantines()
        assert len(active) >= 1
        assert active[0]['severity'] == 'high'
        
        print(f"✅ Active quarantines: {len(active)}")
        
    print("\n✅ All quarantine integration tests passed!")