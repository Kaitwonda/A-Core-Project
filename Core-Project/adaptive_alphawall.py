# adaptive_alphawall.py - Adaptive AlphaWall with learning emotion thresholds

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional

# Import the original AlphaWall
from alphawall import AlphaWall as BaseAlphaWall
from emotion_handler import predict_emotions


class AdaptiveAlphaWall(BaseAlphaWall):
    """
    Enhanced AlphaWall that adapts its emotion detection thresholds based on feedback.
    """
    
    def __init__(self, data_dir="data", max_recursion_window=10):
        super().__init__(data_dir, max_recursion_window)
        
        # Adaptive threshold storage
        self.threshold_file = self.data_dir / "adaptive_emotion_thresholds.json"
        self.feedback_file = self.data_dir / "alphawall_feedback.json"
        self.calibration_file = self.data_dir / "emotion_calibration.json"
        
        # Load or initialize adaptive thresholds
        self.emotion_thresholds = self._load_thresholds()
        self.feedback_history = self._load_feedback()
        self.calibration_data = self._load_calibration()
        
        # Track recent classifications for pattern detection
        self.recent_classifications = deque(maxlen=20)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.threshold_momentum = 0.9  # How much to weight historical vs new data
        
    def _load_thresholds(self) -> Dict:
        """Load adaptive thresholds or initialize with defaults"""
        if self.threshold_file.exists():
            with open(self.threshold_file, 'r') as f:
                return json.load(f)
        
        # Default thresholds (will adapt over time)
        return {
            'emotion_confidence_threshold': 0.6,  # Min confidence to consider emotion significant
            'emotion_specific_thresholds': {
                'fear': {'to_overwhelmed': 0.8, 'to_neutral': 0.4},
                'surprise': {'to_overwhelmed': 0.8, 'to_neutral': 0.4},
                'sadness': {'to_grief': 0.7, 'to_neutral': 0.4},
                'disgust': {'to_angry': 0.7, 'to_neutral': 0.4},
                'anger': {'to_angry': 0.7, 'to_neutral': 0.4},
                'joy': {'to_calm': 0.3, 'to_neutral': 0.7},
                'trust': {'to_calm': 0.3, 'to_neutral': 0.7},
                'anticipation': {'to_calm': 0.3, 'to_neutral': 0.7}
            },
            'recursion_score_threshold': 0.8,
            'question_override_threshold': 0.85,  # How emotional before question becomes expressive
            'adaptation_stats': {
                'total_adaptations': 0,
                'last_adapted': None,
                'performance_score': 0.5
            }
        }
    
    def _save_thresholds(self):
        """Save current thresholds"""
        self.emotion_thresholds['adaptation_stats']['last_adapted'] = datetime.utcnow().isoformat()
        with open(self.threshold_file, 'w') as f:
            json.dump(self.emotion_thresholds, f, indent=2)
    
    def _load_feedback(self) -> list:
        """Load feedback history"""
        if self.feedback_file.exists():
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_feedback(self):
        """Save feedback history"""
        # Keep last 500 feedback entries
        self.feedback_history = self.feedback_history[-500:]
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
    
    def _load_calibration(self) -> Dict:
        """Load calibration data for emotion detection"""
        if self.calibration_file.exists():
            with open(self.calibration_file, 'r') as f:
                return json.load(f)
        
        return {
            'question_patterns': {
                'strong_questions': ['what is', 'how does', 'can you explain', 'tell me about'],
                'weak_questions': ['why', 'what if'],  # Can be emotional
                'learned_patterns': defaultdict(lambda: {'info_count': 0, 'expr_count': 0})
            },
            'false_positive_phrases': [],
            'context_weights': {
                'has_question_mark': 0.3,
                'starts_with_question_word': 0.4,
                'contains_emotional_words': -0.3,
                'all_caps': -0.2,
                'multiple_punctuation': -0.2
            }
        }
    
    def _detect_emotional_state(self, text: str) -> Tuple[str, float]:
        """
        Adaptive emotion detection that learns from feedback.
        """
        emotions = predict_emotions(text)
        
        if not emotions.get('verified'):
            return "neutral", 0.0
            
        primary_emotion, score = emotions['verified'][0]
        
        # Get adaptive thresholds
        base_threshold = self.emotion_thresholds['emotion_confidence_threshold']
        specific_thresholds = self.emotion_thresholds['emotion_specific_thresholds'].get(
            primary_emotion, {'to_neutral': 0.5}
        )
        
        # Calculate context modifiers
        context_score = self._calculate_context_score(text)
        
        # Adjust emotion score based on context
        adjusted_score = score * (1 - context_score * 0.3)  # Context can reduce emotion score by up to 30%
        
        # Determine emotional state using adaptive thresholds
        if adjusted_score < base_threshold:
            emotional_state = "neutral"
        else:
            # Check specific emotion mappings
            if primary_emotion in ['fear', 'surprise']:
                if adjusted_score >= specific_thresholds.get('to_overwhelmed', 0.8):
                    emotional_state = 'overwhelmed'
                else:
                    emotional_state = 'neutral'
            elif primary_emotion in ['sadness']:
                if adjusted_score >= specific_thresholds.get('to_grief', 0.7):
                    emotional_state = 'grief'
                else:
                    emotional_state = 'neutral'
            elif primary_emotion in ['disgust', 'anger']:
                if adjusted_score >= specific_thresholds.get('to_angry', 0.7):
                    emotional_state = 'angry'
                else:
                    emotional_state = 'neutral'
            elif primary_emotion in ['joy', 'trust', 'anticipation']:
                if adjusted_score >= specific_thresholds.get('to_calm', 0.3):
                    emotional_state = 'calm'
                else:
                    emotional_state = 'neutral'
            else:
                emotional_state = 'neutral'
        
        # Check for emotional recursion with adaptive threshold
        recursion_threshold = self.emotion_thresholds.get('recursion_score_threshold', 0.8)
        if adjusted_score > recursion_threshold:
            self.recent_patterns.append(f"emotion:{emotional_state}")
            if self.recent_patterns.count(f"emotion:{emotional_state}") >= self.recursion_threshold:
                emotional_state = "emotionally_recursive"
        
        # Track classification for learning
        self.recent_classifications.append({
            'text_length': len(text),
            'has_question': '?' in text,
            'emotion': primary_emotion,
            'score': score,
            'adjusted_score': adjusted_score,
            'state': emotional_state,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return emotional_state, adjusted_score
    
    def _calculate_context_score(self, text: str) -> float:
        """
        Calculate context score to adjust emotion detection.
        Higher score = more likely to be informational.
        """
        score = 0.0
        text_lower = text.lower()
        weights = self.calibration_data['context_weights']
        
        # Question indicators
        if '?' in text:
            score += weights['has_question_mark']
            
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(text_lower.startswith(qw) for qw in question_words):
            score += weights['starts_with_question_word']
        
        # Check learned patterns
        for pattern, stats in self.calibration_data['question_patterns']['learned_patterns'].items():
            if pattern in text_lower:
                total = stats['info_count'] + stats['expr_count']
                if total > 5:  # Enough data to be meaningful
                    info_ratio = stats['info_count'] / total
                    score += info_ratio * 0.3
        
        # Emotional indicators (reduce score)
        emotional_words = ['feel', 'felt', 'feeling', 'hurt', 'sad', 'angry', 'upset', 'broken', 'lost']
        if any(ew in text_lower for ew in emotional_words):
            score += weights['contains_emotional_words']
        
        # Style indicators
        if text.isupper() and len(text) > 3:
            score += weights['all_caps']
            
        if any(text.count(p) > 2 for p in ['!', '?', '.']):
            score += weights['multiple_punctuation']
        
        # Check false positive phrases
        for phrase in self.calibration_data['false_positive_phrases']:
            if phrase in text_lower:
                score += 0.5  # Strong indicator of information request
        
        return max(0, min(1, score))  # Clamp between 0 and 1
    
    def _detect_intent(self, text: str, emotional_state: str) -> str:
        """
        Enhanced intent detection with adaptive thresholds.
        """
        text_lower = text.lower().strip()
        
        # Get emotion score from recent classification
        recent = self.recent_classifications[-1] if self.recent_classifications else {}
        emotion_score = recent.get('adjusted_score', 0.5)
        
        # Question detection with adaptive threshold
        question_markers = ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'can', 'could', 'would', 'should']
        if text_lower.endswith('?') or any(text_lower.startswith(qm) for qm in question_markers):
            # Use adaptive threshold for question override
            override_threshold = self.emotion_thresholds.get('question_override_threshold', 0.85)
            
            if emotional_state in ['overwhelmed', 'angry', 'emotionally_recursive'] and emotion_score > override_threshold:
                return 'expressive'
            return 'information_request'
        
        # Self-reference detection
        self_markers = ['i ', 'me ', 'my ', 'myself', "i'm", "i've", "i'll"]
        if any(marker in text_lower for marker in self_markers):
            if emotional_state in ['grief', 'overwhelmed'] and emotion_score > 0.6:
                return 'self_reference'
        
        # Other intent detection logic remains the same...
        return super()._detect_intent(text, emotional_state)
    
    def record_feedback(self, zone_id: str, was_correct: bool, correct_intent: Optional[str] = None, 
                       correct_emotion: Optional[str] = None):
        """
        Record feedback about a classification to improve future performance.
        """
        feedback_entry = {
            'zone_id': zone_id,
            'timestamp': datetime.utcnow().isoformat(),
            'was_correct': was_correct,
            'correct_intent': correct_intent,
            'correct_emotion': correct_emotion
        }
        
        self.feedback_history.append(feedback_entry)
        self._save_feedback()
        
        # Adapt thresholds if we have enough feedback
        if len(self.feedback_history) % 10 == 0:
            self._adapt_thresholds()
    
    def _adapt_thresholds(self):
        """
        Adapt thresholds based on recent feedback.
        """
        recent_feedback = self.feedback_history[-50:]  # Last 50 entries
        if len(recent_feedback) < 10:
            return
        
        # Calculate performance metrics
        correct_count = sum(1 for f in recent_feedback if f['was_correct'])
        accuracy = correct_count / len(recent_feedback)
        
        # Update performance score
        old_score = self.emotion_thresholds['adaptation_stats']['performance_score']
        new_score = old_score * self.threshold_momentum + accuracy * (1 - self.threshold_momentum)
        self.emotion_thresholds['adaptation_stats']['performance_score'] = new_score
        
        # Adapt thresholds based on performance
        if accuracy < 0.7:  # Poor performance, adjust thresholds
            # Analyze false positives/negatives
            false_emotional = sum(1 for f in recent_feedback 
                                if not f['was_correct'] and f.get('correct_intent') == 'information_request')
            false_neutral = sum(1 for f in recent_feedback 
                              if not f['was_correct'] and f.get('correct_emotion') == 'neutral')
            
            # Adjust base threshold
            if false_emotional > false_neutral:
                # Too many false emotional detections, increase threshold
                self.emotion_thresholds['emotion_confidence_threshold'] *= 1.05
            elif false_neutral > false_emotional:
                # Missing real emotions, decrease threshold
                self.emotion_thresholds['emotion_confidence_threshold'] *= 0.95
            
            # Clamp threshold
            self.emotion_thresholds['emotion_confidence_threshold'] = max(0.4, min(0.8, 
                self.emotion_thresholds['emotion_confidence_threshold']))
        
        # Update adaptation stats
        self.emotion_thresholds['adaptation_stats']['total_adaptations'] += 1
        self._save_thresholds()
        
        print(f"🔧 AlphaWall adapted: accuracy={accuracy:.2f}, new threshold={self.emotion_thresholds['emotion_confidence_threshold']:.2f}")
    
    def learn_pattern(self, text: str, actual_intent: str):
        """
        Learn from a specific pattern for future classification.
        """
        text_lower = text.lower()
        
        # Extract 2-3 word phrases as patterns
        words = text_lower.split()
        for i in range(len(words) - 1):
            pattern = ' '.join(words[i:i+2])
            if len(pattern) > 3:  # Meaningful pattern
                stats = self.calibration_data['question_patterns']['learned_patterns'][pattern]
                if actual_intent == 'information_request':
                    stats['info_count'] += 1
                else:
                    stats['expr_count'] += 1
        
        # Save calibration data periodically
        if sum(stats['info_count'] + stats['expr_count'] 
               for stats in self.calibration_data['question_patterns']['learned_patterns'].values()) % 20 == 0:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
    
    def add_false_positive(self, phrase: str):
        """
        Add a phrase that was incorrectly classified as emotional.
        """
        phrase_lower = phrase.lower()
        if phrase_lower not in self.calibration_data['false_positive_phrases']:
            self.calibration_data['false_positive_phrases'].append(phrase_lower)
            # Keep list manageable
            self.calibration_data['false_positive_phrases'] = self.calibration_data['false_positive_phrases'][-100:]
            
            with open(self.calibration_file, 'w') as f:
                json.dump(self.calibration_data, f, indent=2)
    
    def get_adaptation_stats(self) -> Dict:
        """
        Get statistics about the adaptation process.
        """
        stats = self.emotion_thresholds['adaptation_stats'].copy()
        stats['current_thresholds'] = {
            'base': self.emotion_thresholds['emotion_confidence_threshold'],
            'recursion': self.emotion_thresholds['recursion_score_threshold'],
            'question_override': self.emotion_thresholds['question_override_threshold']
        }
        stats['feedback_count'] = len(self.feedback_history)
        stats['recent_accuracy'] = None
        
        if len(self.feedback_history) >= 10:
            recent = self.feedback_history[-10:]
            correct = sum(1 for f in recent if f['was_correct'])
            stats['recent_accuracy'] = correct / len(recent)
        
        return stats


# Integration helper
def upgrade_to_adaptive_alphawall(existing_alphawall: Optional[BaseAlphaWall] = None) -> AdaptiveAlphaWall:
    """
    Upgrade existing AlphaWall to adaptive version.
    """
    if existing_alphawall:
        data_dir = existing_alphawall.data_dir
    else:
        data_dir = "data"
    
    return AdaptiveAlphaWall(data_dir=data_dir)


# Feedback integration for talk_to_ai.py
def create_feedback_handler(adaptive_wall: AdaptiveAlphaWall):
    """
    Create a feedback handler for the main conversation loop.
    """
    def handle_user_feedback(zone_id: str, user_response: str):
        """
        Process user feedback about classification accuracy.
        """
        response_lower = user_response.lower()
        
        # Simple feedback parsing
        if any(word in response_lower for word in ['wrong', 'incorrect', 'no', 'bad']):
            adaptive_wall.record_feedback(zone_id, was_correct=False)
            
            # Try to extract what it should have been
            if 'question' in response_lower or 'asking' in response_lower:
                adaptive_wall.record_feedback(zone_id, False, correct_intent='information_request')
            elif 'neutral' in response_lower or 'not emotional' in response_lower:
                adaptive_wall.record_feedback(zone_id, False, correct_emotion='neutral')
                
        elif any(word in response_lower for word in ['correct', 'right', 'yes', 'good']):
            adaptive_wall.record_feedback(zone_id, was_correct=True)
    
    return handle_user_feedback


# Test the adaptive system
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing Adaptive AlphaWall...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wall = AdaptiveAlphaWall(data_dir=tmpdir)
        
        # Test adaptive emotion detection
        test_cases = [
            ("What is machine learning?", "information_request", "neutral"),
            ("HOW DOES THIS WORK???!!!", "information_request", "neutral"),  # Should adapt
            ("I feel so lost", "expressive", "overwhelmed"),
            ("Math?", "information_request", "neutral"),  # Should not quarantine
            ("Why why why why?", "expressive", "emotionally_recursive")  # After repetition
        ]
        
        print("\n📊 Initial thresholds:")
        print(f"   Base emotion threshold: {wall.emotion_thresholds['emotion_confidence_threshold']}")
        
        for text, expected_intent, expected_emotion in test_cases:
            result = wall.process_input(text)
            actual_intent = result['tags']['intent']
            actual_emotion = result['tags']['emotional_state']
            
            correct = (actual_intent == expected_intent and 
                      (actual_emotion == expected_emotion or expected_emotion == "any"))
            
            print(f"\n📝 '{text}'")
            print(f"   Expected: {expected_intent}/{expected_emotion}")
            print(f"   Got: {actual_intent}/{actual_emotion}")
            print(f"   ✅ Correct" if correct else "   ❌ Wrong")
            
            # Record feedback
            wall.record_feedback(result['zone_id'], correct, expected_intent, expected_emotion)
            
            # Learn pattern
            wall.learn_pattern(text, expected_intent)
        
        # Force adaptation
        wall._adapt_thresholds()
        
        print("\n📊 After adaptation:")
        stats = wall.get_adaptation_stats()
        print(f"   Base emotion threshold: {stats['current_thresholds']['base']}")
        print(f"   Performance score: {stats['performance_score']:.2f}")
        print(f"   Total adaptations: {stats['total_adaptations']}")
        
    print("\n✅ Adaptive AlphaWall test complete!")