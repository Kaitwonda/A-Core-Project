# quarantine_layer.py - User Input Quarantine System

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

# Import your existing modules
import user_memory as UM_UserMemory
import symbol_memory as SM_SymbolMemory
from processing_nodes import detect_content_type
from emotion_handler import predict_emotions
import parser as P_Parser

class UserMemoryQuarantine:
    """
    Quarantines user input to prevent contamination of core memory systems.
    Integrates with existing user_memory.py (symbol_occurrence_log) but adds containment.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Quarantine storage - separate from all other memory files
        self.quarantine_path = self.data_dir / "user_quarantine.json"
        self.warfare_log_path = self.data_dir / "linguistic_warfare_log.json"
        
        # Reference to existing memory systems (read-only for checks)
        self.symbol_occurrence_path = self.data_dir / "symbol_occurrence_log.json"
        
        # Load quarantine data
        self.quarantined_items = self._load_quarantine()
        self.warfare_attempts = self._load_warfare_log()
        
    def _load_quarantine(self) -> List[Dict]:
        """Load quarantined user inputs"""
        if self.quarantine_path.exists() and self.quarantine_path.stat().st_size > 0:
            try:
                with open(self.quarantine_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"[QUARANTINE-WARNING] Corrupted quarantine file, starting fresh")
                return []
        return []
    
    def _save_quarantine(self):
        """Save quarantine data with same pattern as your other modules"""
        with open(self.quarantine_path, 'w', encoding='utf-8') as f:
            json.dump(self.quarantined_items, f, indent=2, ensure_ascii=False)
    
    def _load_warfare_log(self) -> List[Dict]:
        """Load linguistic warfare detection log"""
        if self.warfare_log_path.exists() and self.warfare_log_path.stat().st_size > 0:
            try:
                with open(self.warfare_log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []
    
    def _save_warfare_log(self):
        """Save warfare detection events"""
        with open(self.warfare_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.warfare_attempts, f, indent=2, ensure_ascii=False)
    
    def quarantine_user_input(self, 
                            text: str, 
                            user_id: str = "anonymous",
                            source_url: Optional[str] = None,
                            detected_emotions: Optional[Dict] = None,
                            matched_symbols: Optional[List[Dict]] = None,
                            current_phase: int = 0) -> Dict:
        """
        Store user input in quarantine - NEVER affects weights or spreads to core memory.
        Returns quarantine result with ID and analysis.
        
        Integrates with your existing emotion_handler and parser modules.
        """
        # Generate unique ID using your existing pattern
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        timestamp_str = datetime.utcnow().isoformat().replace(':', '-').replace('.', '-')
        quarantine_id = f"quarantine_{user_id}_{timestamp_str}_{text_hash}"
        
        # Detect emotions if not provided (using your emotion_handler)
        if detected_emotions is None:
            detected_emotions = predict_emotions(text)
        
        # Extract keywords using your parser
        keywords = P_Parser.extract_keywords(text)
        
        # Detect content type using your processing_nodes function
        content_type = detect_content_type(text, P_Parser.nlp if P_Parser.NLP_MODEL_LOADED else None)
        
        # Check for warfare patterns
        warfare_analysis = self._detect_linguistic_warfare(
            text, 
            detected_emotions.get('verified', []),
            matched_symbols or []
        )
        
        entry = {
            'id': quarantine_id,
            'text': text[:1000],  # Limit size like your other modules
            'user_id': user_id,
            'source_url': source_url,
            'timestamp': datetime.utcnow().isoformat(),
            'learning_phase': current_phase,
            
            # Integration with your existing systems
            'detected_emotions': detected_emotions.get('verified', [])[:5],
            'matched_symbols': [s['symbol'] for s in (matched_symbols or [])],
            'keywords': keywords[:10],
            'content_type': content_type,
            
            # Quarantine-specific flags
            'quarantined': True,
            'allow_migration': False,  # NEVER migrate to logic/symbolic/bridge
            'allow_weight_influence': False,  # NEVER affect any weights
            'allow_symbol_creation': False,  # NEVER create new symbols
            'verification_status': 'unverified',
            'reference_count': 0,
            'spread_attempts': 0,
            
            # Warfare detection
            'warfare_detected': warfare_analysis['is_suspicious'],
            'warfare_threats': warfare_analysis.get('threats', [])
        }
        
        self.quarantined_items.append(entry)
        self._save_quarantine()
        
        # Log warfare attempts separately
        if warfare_analysis['is_suspicious']:
            self._log_warfare_attempt(entry, warfare_analysis)
        
        return {
            'quarantine_id': quarantine_id,
            'stored': True,
            'warfare_detected': warfare_analysis['is_suspicious'],
            'threats': warfare_analysis.get('threats', [])
        }
    
    def _detect_linguistic_warfare(self, 
                                 text: str, 
                                 emotions: List[Tuple[str, float]], 
                                 symbols: List[Dict]) -> Dict:
        """
        Detect manipulation attempts using patterns from your symbolic analysis.
        """
        threats = []
        text_lower = text.lower()
        
        # Pattern 1: Recursive loops (like your meta-symbol detection)
        recursive_pattern = r'(\b\w+\b)(?:\s+\1){3,}'  # Word repeated 4+ times
        if re.search(recursive_pattern, text_lower):
            threats.append({
                'type': 'recursive_injection',
                'severity': 'high',
                'description': 'Repetitive pattern attempting to create loops',
                'pattern': 'Similar to meta-symbol emergence'
            })
        
        # Pattern 2: Emotional flooding (using your emotion scoring)
        if emotions:
            total_emotion_score = sum(score for _, score in emotions)
            avg_emotion = total_emotion_score / max(1, len(emotions))
            if avg_emotion > 0.8 and len(emotions) > 3:
                threats.append({
                    'type': 'emotional_manipulation',
                    'severity': 'medium',
                    'description': f'High emotional intensity: {avg_emotion:.2f}',
                    'emotions': emotions[:3]
                })
        
        # Pattern 3: Symbol bombing (too many symbols like your cooccurrence tracking)
        words = text.split()
        if len(symbols) > 5 and len(words) < 20:
            symbol_density = len(symbols) / max(1, len(words))
            if symbol_density > 0.3:
                threats.append({
                    'type': 'symbol_overload',
                    'severity': 'medium',
                    'description': f'High symbol density: {symbol_density:.2f}',
                    'symbols': [s['symbol'] for s in symbols[:5]]
                })
        
        # Pattern 4: Known manipulation phrases
        manipulation_markers = [
            "you must believe", "only truth is", "wake up", 
            "they don't want you to know", "secret knowledge",
            "download directly to your", "reprogram your"
        ]
        for marker in manipulation_markers:
            if marker in text_lower:
                threats.append({
                    'type': 'directive_manipulation',
                    'severity': 'high',
                    'description': f'Manipulation phrase detected: "{marker}"',
                    'marker': marker
                })
                break
        
        # Pattern 5: Fake authority claims
        if re.search(r'(studies show|research proves|scientists confirm) (?:that )?(?!http)', text_lower):
            if 'source_url' not in text and 'http' not in text:
                threats.append({
                    'type': 'false_authority',
                    'severity': 'low',
                    'description': 'Unsubstantiated authority claim'
                })
        
        return {
            'is_suspicious': len(threats) > 0,
            'threat_count': len(threats),
            'threats': threats,
            'recommendation': self._get_recommendation(threats)
        }
    
    def _get_recommendation(self, threats: List[Dict]) -> str:
        """Get recommendation based on threat analysis"""
        if not threats:
            return 'quarantine_normal'
        
        high_severity = any(t['severity'] == 'high' for t in threats)
        threat_count = len(threats)
        
        if high_severity or threat_count >= 3:
            return 'quarantine_high_risk'
        elif threat_count >= 2:
            return 'quarantine_monitor'
        else:
            return 'quarantine_low_risk'
    
    def _log_warfare_attempt(self, entry: Dict, analysis: Dict):
        """Log warfare attempts for analysis"""
        warfare_entry = {
            'timestamp': entry['timestamp'],
            'user_id': entry['user_id'],
            'quarantine_id': entry['id'],
            'text_preview': entry['text'][:100] + '...' if len(entry['text']) > 100 else entry['text'],
            'threats': analysis['threats'],
            'recommendation': analysis['recommendation']
        }
        
        self.warfare_attempts.append(warfare_entry)
        self._save_warfare_log()
        
        print(f"⚠️ [WARFARE-DETECTED] User {entry['user_id']}: "
              f"{len(analysis['threats'])} threats, "
              f"recommendation: {analysis['recommendation']}")
    
    def check_user_history(self, user_id: str) -> Dict:
        """
        Check user's quarantine history for patterns.
        Similar to your symbol occurrence tracking.
        """
        user_items = [
            item for item in self.quarantined_items 
            if item['user_id'] == user_id
        ]
        
        if not user_items:
            return {
                'user_id': user_id,
                'total_inputs': 0,
                'warfare_attempts': 0,
                'risk_level': 'unknown'
            }
        
        warfare_count = sum(1 for item in user_items if item['warfare_detected'])
        
        # Calculate risk like your stability scores
        risk_ratio = warfare_count / len(user_items)
        if risk_ratio > 0.5:
            risk_level = 'high'
        elif risk_ratio > 0.2:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'user_id': user_id,
            'total_inputs': len(user_items),
            'warfare_attempts': warfare_count,
            'risk_level': risk_level,
            'most_recent': user_items[-1]['timestamp'] if user_items else None
        }
    
    def recall_for_context(self, user_id: str, limit: int = 5) -> List[Dict]:
        """
        Recall what user said WITHOUT letting it influence the system.
        For "I remember you said X" functionality.
        """
        user_items = sorted(
            [item for item in self.quarantined_items if item['user_id'] == user_id],
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]
        
        # Return sanitized version for display
        return [
            {
                'text': item['text'],
                'timestamp': item['timestamp'],
                'emotions': item['detected_emotions'][:2] if item['detected_emotions'] else [],
                'quarantine_id': item['id'],
                'warfare_detected': item['warfare_detected']
            }
            for item in user_items
        ]
    
    def get_quarantine_stats(self) -> Dict:
        """
        Get statistics about quarantine, similar to your other get_counts functions.
        """
        total = len(self.quarantined_items)
        warfare_detected = sum(1 for item in self.quarantined_items if item['warfare_detected'])
        
        user_counts = {}
        for item in self.quarantined_items:
            uid = item['user_id']
            user_counts[uid] = user_counts.get(uid, 0) + 1
        
        return {
            'total_quarantined': total,
            'warfare_attempts': warfare_detected,
            'unique_users': len(user_counts),
            'most_active_user': max(user_counts.items(), key=lambda x: x[1])[0] if user_counts else None,
            'warfare_percentage': (warfare_detected / total * 100) if total > 0 else 0
        }
    
    def prevent_spread_check(self, text: str) -> bool:
        """
        Check if text matches quarantined content (prevent spreading).
        Returns True if text should be blocked.
        """
        text_lower = text.lower()
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        for item in self.quarantined_items:
            # Check exact match
            if text_hash == hashlib.md5(item['text'].encode('utf-8')).hexdigest():
                item['spread_attempts'] += 1
                self._save_quarantine()
                return True
            
            # Check substantial overlap (>70% similarity)
            item_lower = item['text'].lower()
            if len(text_lower) > 50 and len(item_lower) > 50:
                overlap = len(set(text_lower.split()) & set(item_lower.split()))
                total = len(set(text_lower.split()) | set(item_lower.split()))
                if total > 0 and (overlap / total) > 0.7:
                    item['spread_attempts'] += 1
                    self._save_quarantine()
                    return True
        
        return False


# Integration helper functions to connect with your existing code

def should_quarantine_input(source_type: str, source_url: str = None) -> bool:
    """
    Determine if input should go to quarantine based on your source types.
    """
    quarantine_sources = [
        "user_direct_input",
        "user_chat",
        "user_feedback",
        "unverified_claim"
    ]
    
    # Check source type
    if source_type in quarantine_sources:
        return True
    
    # Check untrusted URLs
    if source_url and any(domain in source_url for domain in ["reddit.com", "4chan.org", "anonymous"]):
        return True
    
    return False


def integrate_with_processing_nodes(quarantine: UserMemoryQuarantine):
    """
    Modify your DynamicBridge.route_chunk_for_processing to use quarantine.
    """
    # This would be added to your route_chunk_for_processing method:
    """
    if should_quarantine_input(source_type, source_url):
        result = quarantine.quarantine_user_input(
            text=text_input,
            user_id=source_url or "anonymous", 
            detected_emotions=detected_emotions_output,
            matched_symbols=symbolic_node_output.get('top_matched_symbols', []),
            current_phase=current_processing_phase
        )
        
        return {
            'decision_type': 'QUARANTINED',
            'confidence': 0.0,
            'symbols_found': 0,
            'logic_result': {'retrieved_memories_count': 0, 'top_retrieved_texts': []},
            'symbolic_result': {'matched_symbols_count': 0, 'top_matched_symbols': []},
            'stored_item': None,
            'quarantine_result': result
        }
    """


# Unit tests
if __name__ == "__main__":
    print("🧪 Testing UserMemoryQuarantine...")
    
    # Test 1: Basic quarantine
    print("\n1️⃣ Test: Basic quarantine functionality")
    quarantine = UserMemoryQuarantine(data_dir="data/test_quarantine")
    
    result = quarantine.quarantine_user_input(
        text="The earth is flat and NASA is lying to you!",
        user_id="test_user_001"
    )
    
    assert result['stored'] == True
    assert result['warfare_detected'] == True  # Should detect "lying to you"
    print(f"✅ Quarantined with threats: {result['threats']}")
    
    # Test 2: Emotional manipulation detection
    print("\n2️⃣ Test: Emotional manipulation detection")
    emotions = [("fear", 0.9), ("anger", 0.95), ("disgust", 0.85), ("sadness", 0.9)]
    result2 = quarantine.quarantine_user_input(
        text="This makes me so angry and scared!",
        user_id="test_user_002",
        detected_emotions={'verified': emotions}
    )
    
    assert any(t['type'] == 'emotional_manipulation' for t in result2['threats'])
    print("✅ Detected emotional manipulation")
    
    # Test 3: Symbol bombing
    print("\n3️⃣ Test: Symbol bombing detection")
    symbols = [{'symbol': '🔥'}, {'symbol': '💀'}, {'symbol': '⚡'}, 
               {'symbol': '🌀'}, {'symbol': '💣'}, {'symbol': '🎯'}]
    result3 = quarantine.quarantine_user_input(
        text="🔥💀⚡ Chaos incoming 🌀💣🎯",
        user_id="test_user_003",
        matched_symbols=symbols
    )
    
    assert any(t['type'] == 'symbol_overload' for t in result3['threats'])
    print("✅ Detected symbol bombing")
    
    # Test 4: Spread prevention
    print("\n4️⃣ Test: Spread prevention")
    should_block = quarantine.prevent_spread_check("The earth is flat and NASA is lying to you!")
    assert should_block == True
    print("✅ Spread prevention working")
    
    # Test 5: User history check
    print("\n5️⃣ Test: User history tracking")
    history = quarantine.check_user_history("test_user_001")
    assert history['warfare_attempts'] >= 1
    print(f"✅ User history: {history}")
    
    # Test 6: Stats
    print("\n6️⃣ Test: Quarantine statistics")
    stats = quarantine.get_quarantine_stats()
    print(f"Stats: {json.dumps(stats, indent=2)}")
    assert stats['total_quarantined'] >= 3
    assert stats['warfare_percentage'] > 0
    
    print("\n✅ All quarantine tests passed!")