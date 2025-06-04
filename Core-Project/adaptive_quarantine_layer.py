# adaptive_quarantine_layer.py - Adaptive Quarantine System

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import re

# Import existing modules
from quarantine_layer import UserMemoryQuarantine as BaseQuarantine
from quarantine_layer import should_quarantine_input
from alphawall import AlphaWall


class AdaptiveQuarantine(BaseQuarantine):
    """
    Enhanced quarantine system that learns what actually needs quarantining.
    """
    
    def __init__(self, data_dir="data"):
        super().__init__(data_dir)
        
        # Adaptive thresholds and patterns
        self.adaptive_config_file = self.quarantine_dir / "adaptive_quarantine_config.json"
        self.false_positive_log = self.quarantine_dir / "false_positives.json"
        self.true_positive_log = self.quarantine_dir / "true_positives.json"
        
        # Load adaptive configuration
        self.adaptive_config = self._load_adaptive_config()
        
        # Track recent decisions for context
        self.recent_decisions = deque(maxlen=10)
        self.session_context = {
            'false_positives': 0,
            'true_positives': 0,
            'last_topics': deque(maxlen=5)
        }
        
    def _load_adaptive_config(self) -> Dict:
        """Load or initialize adaptive configuration"""
        if self.adaptive_config_file.exists():
            with open(self.adaptive_config_file, 'r') as f:
                return json.load(f)
                
        # Default configuration
        return {
            'min_words_threshold': 2,  # Start more permissive
            'vague_word_patterns': {
                # Academic/factual questions shouldn't be quarantined
                'safe_academic': ['math', 'science', 'history', 'computer', 'ai', 'algorithm', 
                                 'physics', 'chemistry', 'biology', 'geology', 'astronomy'],
                'safe_questions': ['what', 'how', 'why', 'when', 'where', 'who', 'which',
                                  'explain', 'describe', 'define', 'tell'],
                # Actually vague/problematic
                'true_vague': ['it', 'this', 'that', 'thing', 'stuff', 'whatever'],
                'potentially_recursive': ['why', 'no', 'help', 'please', 'stop']
            },
            'context_patterns': {
                'academic_context': ['learning', 'studying', 'research', 'knowledge', 'understand'],
                'emotional_context': ['feel', 'hurt', 'sad', 'angry', 'lost', 'broken'],
                'neutral_context': ['know', 'think', 'wonder', 'curious', 'interested']
            },
            'quarantine_thresholds': {
                'recursion_count': 3,  # How many times before it's a loop
                'emotional_intensity': 0.8,  # How emotional before quarantine
                'vagueness_score': 0.7,  # How vague before quarantine
                'context_weight': 0.3  # How much context matters
            },
            'learning_stats': {
                'total_decisions': 0,
                'false_positive_rate': 0.0,
                'last_adapted': None
            }
        }
    
    def _save_adaptive_config(self):
        """Save adaptive configuration"""
        with open(self.adaptive_config_file, 'w') as f:
            json.dump(self.adaptive_config, f, indent=2)
    
    def _calculate_vagueness_score(self, text: str, zone_output: Dict) -> float:
        """
        Calculate how vague an input is, considering context.
        Returns 0.0 (specific) to 1.0 (very vague).
        """
        text_lower = text.lower().strip()
        words = text_lower.split()
        
        # Start with base score
        vagueness = 0.0
        
        # Check word count
        if len(words) < self.adaptive_config['min_words_threshold']:
            vagueness += 0.3
        
        # Check if it's an academic/safe topic
        safe_academic = self.adaptive_config['vague_word_patterns']['safe_academic']
        safe_questions = self.adaptive_config['vague_word_patterns']['safe_questions']
        
        # Academic topics are NOT vague
        for safe_word in safe_academic:
            if safe_word in text_lower:
                vagueness -= 0.4
                
        # Question words indicate information seeking, not vagueness
        for q_word in safe_questions:
            if text_lower.startswith(q_word) or f" {q_word} " in text_lower:
                vagueness -= 0.3
        
        # Check for true vague patterns
        true_vague = self.adaptive_config['vague_word_patterns']['true_vague']
        for vague_word in true_vague:
            if vague_word in words and len(words) < 4:
                vagueness += 0.3
        
        # Consider emotional state from zone
        emotional_state = zone_output['tags'].get('emotional_state', 'neutral')
        if emotional_state in ['overwhelmed', 'emotionally_recursive']:
            vagueness += 0.2
        
        # Check session context
        if self.session_context['last_topics']:
            # If we've been discussing academic topics, reduce vagueness
            recent_topics = ' '.join(self.session_context['last_topics'])
            academic_count = sum(1 for word in safe_academic if word in recent_topics)
            if academic_count > 0:
                vagueness -= 0.2
        
        # Question mark indicates seeking information, not being vague
        if '?' in text:
            vagueness -= 0.2
            
        # Clamp between 0 and 1
        return max(0.0, min(1.0, vagueness))
    
    def _detect_true_recursion(self, text: str, zone_output: Dict) -> Tuple[bool, str]:
        """
        Detect if this is actual problematic recursion vs topic exploration.
        Returns (is_recursion, reason).
        """
        # Get recent patterns
        recent_contexts = [d.get('context', '') for d in self.recent_decisions]
        recent_intents = [d.get('intent', '') for d in self.recent_decisions]
        recent_texts = [d.get('text_pattern', '') for d in self.recent_decisions]
        
        # Extract pattern from current text
        text_pattern = self._extract_text_pattern(text)
        
        # Check for true recursion patterns
        recursion_threshold = self.adaptive_config['quarantine_thresholds']['recursion_count']
        
        # Pattern 1: Exact repetition
        if recent_texts.count(text_pattern) >= recursion_threshold:
            return True, "exact_repetition"
        
        # Pattern 2: Emotional spiral (same emotion + similar text)
        emotional_state = zone_output['tags'].get('emotional_state', 'neutral')
        if emotional_state in ['overwhelmed', 'emotionally_recursive', 'grief']:
            emotional_count = sum(1 for d in self.recent_decisions 
                                if d.get('emotional_state') == emotional_state)
            if emotional_count >= recursion_threshold:
                # But check if it's academic discussion about emotions
                if not any(word in text.lower() for word in ['study', 'research', 'psychology', 'explain']):
                    return True, "emotional_spiral"
        
        # Pattern 3: True vague loops (not academic questions)
        if len(text.split()) < 3:
            vague_count = sum(1 for d in self.recent_decisions 
                            if d.get('word_count', 10) < 3)
            if vague_count >= recursion_threshold:
                # Check if they're all questions about different topics
                if not self._are_varied_questions(recent_texts):
                    return True, "vague_loop"
        
        return False, "no_recursion"
    
    def _extract_text_pattern(self, text: str) -> str:
        """Extract pattern for comparison"""
        # Remove punctuation and lowercase
        pattern = re.sub(r'[^\w\s]', '', text.lower()).strip()
        # Get first few words as pattern
        words = pattern.split()[:5]
        return ' '.join(words)
    
    def _are_varied_questions(self, texts: List[str]) -> bool:
        """Check if short inputs are actually varied questions"""
        topics = set()
        for text in texts:
            # Extract main topic word
            words = text.lower().split()
            for word in words:
                if len(word) > 3 and word not in ['what', 'how', 'why', 'when', 'where']:
                    topics.add(word)
        
        # If we have multiple different topics, they're varied questions
        return len(topics) >= len(texts) * 0.5
    
    def should_quarantine_with_learning(self, zone_output: Dict, text: str) -> Tuple[bool, str]:
        """
        Adaptive quarantine decision with learning capability.
        Returns (should_quarantine, reason).
        """
        # First check source-based quarantine (from original)
        source_type = zone_output.get('source_type', 'unknown')
        if source_type in ['user_direct_input', 'untrusted_source']:
            # For user input, be more nuanced
            pass
        else:
            # For other sources, use original logic
            return False, "trusted_source"
        
        # Calculate vagueness score
        vagueness = self._calculate_vagueness_score(text, zone_output)
        
        # Detect recursion
        is_recursive, recursion_type = self._detect_true_recursion(text, zone_output)
        
        # Get emotional intensity
        emotion_confidence = zone_output['tags'].get('emotion_confidence', 0.0)
        emotional_state = zone_output['tags'].get('emotional_state', 'neutral')
        
        # Decision logic
        quarantine = False
        reason = "safe"
        
        # High vagueness + recursion = quarantine
        if vagueness > self.adaptive_config['quarantine_thresholds']['vagueness_score'] and is_recursive:
            quarantine = True
            reason = f"high_vagueness_and_{recursion_type}"
            
        # Extreme emotional recursion = quarantine
        elif is_recursive and emotion_confidence > self.adaptive_config['quarantine_thresholds']['emotional_intensity']:
            quarantine = True
            reason = f"emotional_{recursion_type}"
            
        # Single word that's not academic = maybe quarantine
        elif len(text.split()) == 1 and vagueness > 0.5:
            # But not if it's a clear question
            if not text.strip().endswith('?'):
                quarantine = True
                reason = "single_vague_word"
        
        # Update context
        self._update_decision_context(text, zone_output, quarantine, reason)
        
        return quarantine, reason
    
    def _update_decision_context(self, text: str, zone_output: Dict, quarantined: bool, reason: str):
        """Update decision tracking for learning"""
        decision = {
            'timestamp': datetime.utcnow().isoformat(),
            'text_pattern': self._extract_text_pattern(text),
            'word_count': len(text.split()),
            'emotional_state': zone_output['tags'].get('emotional_state', 'neutral'),
            'intent': zone_output['tags'].get('intent', 'unknown'),
            'context': zone_output['tags'].get('context', []),
            'quarantined': quarantined,
            'reason': reason,
            'zone_id': zone_output.get('zone_id', 'unknown')
        }
        
        self.recent_decisions.append(decision)
        
        # Extract topic for context
        words = text.lower().split()
        for word in words:
            if len(word) > 3 and word not in ['what', 'how', 'why', 'when', 'where', 'that', 'this']:
                self.session_context['last_topics'].append(word)
                break
    
    def record_feedback(self, zone_id: str, was_false_positive: bool, correct_classification: Optional[str] = None):
        """Record feedback about quarantine decisions"""
        # Find the decision
        decision = None
        for d in self.recent_decisions:
            if d.get('zone_id') == zone_id:
                decision = d
                break
                
        if not decision:
            return
        
        # Record feedback
        feedback_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'zone_id': zone_id,
            'text_pattern': decision['text_pattern'],
            'was_quarantined': decision['quarantined'],
            'was_false_positive': was_false_positive,
            'correct_classification': correct_classification,
            'reason': decision['reason']
        }
        
        # Save to appropriate log
        if was_false_positive:
            self.session_context['false_positives'] += 1
            self._save_to_log(self.false_positive_log, feedback_entry)
            
            # Learn from false positive
            self._learn_from_false_positive(decision)
        else:
            self.session_context['true_positives'] += 1
            self._save_to_log(self.true_positive_log, feedback_entry)
    
    def _save_to_log(self, log_file: Path, entry: Dict):
        """Save entry to log file"""
        if log_file.exists():
            with open(log_file, 'r') as f:
                log = json.load(f)
        else:
            log = []
            
        log.append(entry)
        log = log[-500:]  # Keep last 500
        
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2)
    
    def _learn_from_false_positive(self, decision: Dict):
        """Adjust thresholds based on false positive"""
        # If we quarantined something that shouldn't have been
        if decision['reason'].startswith('high_vagueness'):
            # Increase vagueness threshold
            self.adaptive_config['quarantine_thresholds']['vagueness_score'] *= 1.05
            
        elif decision['reason'].startswith('emotional'):
            # Increase emotional threshold
            self.adaptive_config['quarantine_thresholds']['emotional_intensity'] *= 1.05
            
        elif decision['reason'] == 'single_vague_word':
            # Add the word to safe list if it appears to be academic
            pattern_words = decision['text_pattern'].split()
            if pattern_words and len(pattern_words[0]) > 2:
                word = pattern_words[0]
                if word not in self.adaptive_config['vague_word_patterns']['safe_academic']:
                    self.adaptive_config['vague_word_patterns']['safe_academic'].append(word)
        
        # Update stats
        total = self.session_context['false_positives'] + self.session_context['true_positives']
        if total > 0:
            self.adaptive_config['learning_stats']['false_positive_rate'] = \
                self.session_context['false_positives'] / total
        
        self.adaptive_config['learning_stats']['total_decisions'] += 1
        self.adaptive_config['learning_stats']['last_adapted'] = datetime.utcnow().isoformat()
        
        # Save config
        self._save_adaptive_config()
    
    def get_adaptive_stats(self) -> Dict:
        """Get statistics about adaptive quarantine performance"""
        stats = {
            'current_thresholds': self.adaptive_config['quarantine_thresholds'].copy(),
            'learning_stats': self.adaptive_config['learning_stats'].copy(),
            'session_stats': {
                'false_positives': self.session_context['false_positives'],
                'true_positives': self.session_context['true_positives'],
                'recent_topics': list(self.session_context['last_topics'])
            },
            'safe_words_learned': len(self.adaptive_config['vague_word_patterns']['safe_academic']) - 10  # Original had ~10
        }
        
        # Calculate current session accuracy
        total = self.session_context['false_positives'] + self.session_context['true_positives']
        if total > 0:
            stats['session_stats']['accuracy'] = self.session_context['true_positives'] / total
            
        return stats
    
    def reset_session_context(self):
        """Reset session context for new conversation"""
        self.session_context = {
            'false_positives': 0,
            'true_positives': 0,
            'last_topics': deque(maxlen=5)
        }
        self.recent_decisions.clear()


# Enhanced quarantine check for bridge integration
def adaptive_quarantine_check(text: str, zone_output: Dict, quarantine: AdaptiveQuarantine) -> Dict:
    """
    Check if input should be quarantined using adaptive logic.
    """
    should_quarantine, reason = quarantine.should_quarantine_with_learning(zone_output, text)
    
    result = {
        'should_quarantine': should_quarantine,
        'reason': reason,
        'confidence': 0.9 if should_quarantine else 0.1,
        'is_academic': any(word in text.lower() for word in 
                          quarantine.adaptive_config['vague_word_patterns']['safe_academic']),
        'vagueness_score': quarantine._calculate_vagueness_score(text, zone_output)
    }
    
    # Provide suggestions
    if should_quarantine:
        if 'vague' in reason:
            result['suggestion'] = "This seems vague. Could you be more specific?"
        elif 'emotional' in reason:
            result['suggestion'] = "I notice strong emotions. Would you like to talk about something specific?"
        elif 'repetition' in reason:
            result['suggestion'] = "We seem to be going in circles. Let's try a different topic?"
    
    return result


# Test the adaptive system
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing Adaptive Quarantine System...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        alphawall = AlphaWall(data_dir=tmpdir)
        quarantine = AdaptiveQuarantine(data_dir=tmpdir)
        
        # Test cases that shouldn't be quarantined
        test_cases = [
            ("Math?", "academic_question"),
            ("What is AI?", "academic_question"),
            ("Earth minerals?", "academic_question"),
            ("How does quantum computing work?", "detailed_question"),
            ("Why?", "single_question"),  # Context matters
            ("Tell me about computers", "information_request")
        ]
        
        print("\n📊 Testing inputs that shouldn't be quarantined:")
        
        for text, expected_type in test_cases:
            zone = alphawall.process_input(text)
            should_q, reason = quarantine.should_quarantine_with_learning(zone, text)
            
            print(f"\n'{text}':")
            print(f"  Quarantine: {should_q}")
            print(f"  Reason: {reason}")
            print(f"  Vagueness: {quarantine._calculate_vagueness_score(text, zone):.2f}")
            
            # These should NOT be quarantined
            assert not should_q or reason == "safe", f"'{text}' was wrongly quarantined!"
        
        print("\n✅ Academic questions pass through correctly!")
        
        # Test actual problematic patterns
        print("\n📊 Testing actual problematic patterns:")
        
        # Create recursion
        for i in range(4):
            zone = alphawall.process_input("Why why why why?")
            should_q, reason = quarantine.should_quarantine_with_learning(zone, "Why why why why?")
        
        print(f"\nAfter 4x 'Why why why why?':")
        print(f"  Quarantine: {should_q}")
        print(f"  Reason: {reason}")
        assert should_q, "Actual recursion should be caught"
        
        # Test learning from feedback
        print("\n📊 Testing learning from feedback:")
        
        # False positive on "Physics?"
        zone = alphawall.process_input("Physics?")
        should_q, reason = quarantine.should_quarantine_with_learning(zone, "Physics?")
        
        if should_q:  # If it was quarantined
            quarantine.record_feedback(zone['zone_id'], was_false_positive=True)
            print("  Recorded false positive for 'Physics?'")
        
        # Check if it learned
        stats = quarantine.get_adaptive_stats()
        print(f"\n📊 Adaptive Stats:")
        print(f"  Vagueness threshold: {stats['current_thresholds']['vagueness_score']:.2f}")
        print(f"  Safe words learned: {stats['safe_words_learned']}")
        
        print("\n✅ Adaptive quarantine system working!")