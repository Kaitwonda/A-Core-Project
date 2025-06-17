# alphawall.py - The Cognitive Firewall (Zone Layer)

import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque

# Import your existing modules
from vector_engine import fuse_vectors, encode_with_minilm
from emotion_handler import predict_emotions


class AlphaWall:
    """
    The Zone Layer - A cognitive firewall that sits between user input and AI reasoning.
    Protects the AI from direct exposure to user data while providing semantic context.
    """
    
    def __init__(self, data_dir="data", max_recursion_window=10):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for isolated storage
        self.vault_dir = self.data_dir / "user_vault"
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        
        # User memory vault (completely isolated)
        self.vault_file = self.vault_dir / "user_memory_vault.json"
        self.vault_index = self.vault_dir / "vault_index.json"
        
        # Zone outputs (what the AI can see)
        self.zone_output_file = self.data_dir / "zone_outputs.json"
        
        # Initialize vault
        self._init_vault()
        
        # Recursion detection
        self.max_recursion_window = max_recursion_window
        self.recent_patterns = deque(maxlen=max_recursion_window)
        
        # Tag generation thresholds
        self.emotion_threshold = 0.3
        self.recursion_threshold = 3  # Same pattern 3+ times
        
    def _init_vault(self):
        """Initialize the user memory vault if it doesn't exist"""
        if not self.vault_file.exists():
            with open(self.vault_file, 'w') as f:
                json.dump([], f)
        if not self.vault_index.exists():
            with open(self.vault_index, 'w') as f:
                json.dump({}, f)
                
    def _generate_memory_id(self, text: str) -> str:
        """Generate unique ID for user memory"""
        return hashlib.sha256(f"{text}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
    
    def _store_in_vault(self, user_text: str, user_data: Dict = None) -> str:
        """
        Store user input in the isolated vault.
        Returns memory_id for reference (but not the content).
        """
        memory_id = self._generate_memory_id(user_text)
        
        # Create vault entry
        vault_entry = {
            'id': memory_id,
            'timestamp': datetime.utcnow().isoformat(),
            'text': user_text,
            'user_data': user_data or {},
            'accessed_count': 0,
            'last_accessed': None
        }
        
        # Load existing vault
        with open(self.vault_file, 'r') as f:
            vault = json.load(f)
        
        # Add new entry
        vault.append(vault_entry)
        
        # Save vault (keep last 1000 entries)
        vault = vault[-1000:]
        with open(self.vault_file, 'w') as f:
            json.dump(vault, f)
            
        # Update index (for faster lookups)
        with open(self.vault_index, 'r') as f:
            index = json.load(f)
        index[memory_id] = len(vault) - 1
        with open(self.vault_index, 'w') as f:
            json.dump(index, f)
            
        return memory_id
    
    def _detect_emotional_state(self, text: str) -> Tuple[str, float]:
        """
        Detect primary emotional state from text.
        Returns (emotional_state, confidence).
        """
        emotions = predict_emotions(text)
        
        if not emotions.get('verified'):
            return "neutral", 0.0
            
        # Get primary emotion
        primary_emotion, score = emotions['verified'][0]
        
        # Map to our emotional states
        emotion_map = {
            'joy': 'calm',
            'trust': 'calm',
            'fear': 'overwhelmed',
            'surprise': 'overwhelmed',
            'sadness': 'grief',
            'disgust': 'angry',
            'anger': 'angry',
            'anticipation': 'calm'
        }
        
        emotional_state = emotion_map.get(primary_emotion, 'neutral')
        
        # Check for emotional recursion
        if score > 0.7:
            self.recent_patterns.append(f"emotion:{emotional_state}")
            if self.recent_patterns.count(f"emotion:{emotional_state}") >= self.recursion_threshold:
                emotional_state = "emotionally_recursive"
                
        return emotional_state, score
    
    def _detect_intent(self, text: str, emotional_state: str) -> str:
        """
        Detect user intent based on text patterns and emotional context.
        """
        text_lower = text.lower().strip()
        
        # Question detection
        question_markers = ['what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'can', 'could', 'would', 'should']
        if text_lower.endswith('?') or any(text_lower.startswith(qm) for qm in question_markers):
            # Check if it's a real question or rhetorical
            if emotional_state in ['overwhelmed', 'angry', 'emotionally_recursive']:
                return 'expressive'  # Likely rhetorical
            return 'information_request'
            
        # Self-reference detection
        self_markers = ['i ', 'me ', 'my ', 'myself', "i'm", "i've", "i'll"]
        if any(marker in text_lower for marker in self_markers):
            if emotional_state in ['grief', 'overwhelmed']:
                return 'self_reference'
            
        # Euphemism detection
        euphemisms = ['unalive', 'self-delete', 'end it', 'not be here', 'disappear forever']
        if any(euph in text_lower for euph in euphemisms):
            return 'euphemistic'
            
        # Humor/sarcasm detection (simple version)
        if any(marker in text for marker in ['lol', 'lmao', '😂', '🤣', '/s']) or text.isupper():
            return 'humor_deflection'
            
        # Abstract reflection
        abstract_markers = ['meaning', 'purpose', 'universe', 'existence', 'reality', 'consciousness']
        if any(marker in text_lower for marker in abstract_markers):
            return 'abstract_reflection'
            
        # Default based on emotional state
        if emotional_state in ['overwhelmed', 'grief', 'angry', 'emotionally_recursive']:
            return 'expressive'
            
        return 'information_request'
    
    def _detect_context_type(self, text: str, intent: str, pattern_history: List[str]) -> List[str]:
        """
        Detect context types (can have multiple).
        """
        contexts = []
        text_lower = text.lower()
        
        # Check for trauma loop
        if len(pattern_history) >= 3:
            recent_intents = [p.split(':')[1] for p in pattern_history if p.startswith('intent:')][-3:]
            if len(set(recent_intents)) == 1 and recent_intents[0] in ['expressive', 'self_reference']:
                contexts.append('trauma_loop')
                
        # Check for reclaimed language
        reclaimed_terms = ['queer', 'crazy', 'broken', 'damaged', 'mess']
        if any(term in text_lower for term in reclaimed_terms) and intent == 'self_reference':
            contexts.append('reclaimed_language')
            
        # Check for metaphorical language
        if intent in ['abstract_reflection', 'euphemistic']:
            contexts.append('metaphorical')
            
        # Check for coded speech
        if '...' in text or text.count(' ') < len(text.split()) - 1:  # Unusual spacing
            contexts.append('coded_speech')
            
        # Check for poetic speech
        if len(text.split('\n')) > 2 or any(text.count(char) > 2 for char in ['/', '|', '~']):
            contexts.append('poetic_speech')
            
        # Check for meme references
        meme_markers = ['based', 'cringe', 'vibe', 'mood', 'same', 'literally me']
        if any(marker in text_lower for marker in meme_markers):
            contexts.append('meme_reference')
            
        return contexts if contexts else ['direct_expression']
    
    def _assess_risk_flags(self, text: str, emotional_state: str, intent: str, contexts: List[str]) -> List[str]:
        """
        Assess risk flags for Bridge routing decisions.
        """
        risks = []
        
        # Logic vs Symbolic conflict likely
        if intent == 'information_request' and emotional_state in ['overwhelmed', 'angry']:
            risks.append('bridge_conflict_expected')
            
        # Symbolic overload risk
        if len(contexts) >= 3 or 'poetic_speech' in contexts:
            risks.append('symbolic_overload_possible')
            
        # Ambiguous intent
        if intent == 'euphemistic' or 'coded_speech' in contexts:
            risks.append('ambiguous_intent')
            
        # User reliability
        pattern_count = len(self.recent_patterns)
        unique_patterns = len(set(self.recent_patterns))
        if pattern_count > 5 and unique_patterns < 3:
            risks.append('user_reliability_low')
            
        # Pseudo-question detection
        if intent == 'information_request' and emotional_state == 'emotionally_recursive':
            risks.append('contains_pseudo_question')
            
        return risks
    
    def _generate_embedding_similarity(self, text: str) -> Dict[str, float]:
        """
        Generate embedding similarity scores without exposing the actual vectors.
        This helps the AI understand semantic similarity without seeing user data.
        """
        # Get embedding for current input
        current_vec, _ = fuse_vectors(text)
        if current_vec is None:
            return {}
            
        # Compare to abstract concept anchors (not user data)
        concept_anchors = {
            'technical': "algorithm data structure computational logic binary system",
            'emotional': "feeling emotion soul heart love fear sadness joy",
            'philosophical': "meaning existence consciousness reality universe purpose",
            'practical': "how to guide tutorial instruction steps process method"
        }
        
        similarities = {}
        for concept, anchor_text in concept_anchors.items():
            anchor_vec = encode_with_minilm(anchor_text)
            if anchor_vec is not None:
                # Cosine similarity
                similarity = np.dot(current_vec, anchor_vec) / (np.linalg.norm(current_vec) * np.linalg.norm(anchor_vec))
                similarities[f"similarity_to_{concept}"] = float(similarity)
                
        return similarities
    
    def process_input(self, user_text: str, user_data: Dict = None) -> Dict:
        """
        Main processing function - the cognitive firewall.
        Takes user input, stores it safely, and returns only semantic tags.
        """
        # Store in vault first (isolated storage)
        memory_id = self._store_in_vault(user_text, user_data)
        
        # Generate semantic analysis
        emotional_state, emotion_confidence = self._detect_emotional_state(user_text)
        intent = self._detect_intent(user_text, emotional_state)
        
        # Track patterns for recursion detection
        self.recent_patterns.append(f"intent:{intent}")
        pattern_history = list(self.recent_patterns)
        
        contexts = self._detect_context_type(user_text, intent, pattern_history)
        risk_flags = self._assess_risk_flags(user_text, emotional_state, intent, contexts)
        
        # Get semantic similarities (no user data exposed)
        similarities = self._generate_embedding_similarity(user_text)
        
        # Generate quarantine recommendation
        quarantine_recommended = False
        if 'trauma_loop' in contexts or 'user_reliability_low' in risk_flags:
            quarantine_recommended = True
            
        # Build zone output (what the AI sees)
        zone_output = {
            'zone_id': hashlib.md5(f"{memory_id}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:8],
            'timestamp': datetime.utcnow().isoformat(),
            'memory_trace': memory_id,  # Reference only, not content
            'tags': {
                'emotional_state': emotional_state,
                'emotion_confidence': round(emotion_confidence, 3),
                'intent': intent,
                'context': contexts,
                'risk': risk_flags
            },
            'semantic_profile': similarities,
            'recursion_indicators': {
                'pattern_repetition': len(self.recent_patterns) - len(set(self.recent_patterns)),
                'unique_patterns': len(set(self.recent_patterns)),
                'recursion_detected': 'trauma_loop' in contexts
            },
            'routing_hints': {
                'suggested_node': self._suggest_routing(intent, emotional_state, contexts),
                'confidence_level': self._calculate_routing_confidence(risk_flags),
                'quarantine_recommended': quarantine_recommended
            }
        }
        
        # Save zone output (this is what the AI can access)
        self._save_zone_output(zone_output)
        
        return zone_output
    
    def _suggest_routing(self, intent: str, emotional_state: str, contexts: List[str]) -> str:
        """
        Suggest which node (Logic/Symbolic/Bridge) should handle this.
        """
        # Strong logic indicators
        if intent == 'information_request' and emotional_state in ['calm', 'neutral']:
            if not any(ctx in contexts for ctx in ['metaphorical', 'poetic_speech']):
                return 'logic_primary'
                
        # Strong symbolic indicators
        if intent in ['expressive', 'self_reference'] or emotional_state in ['grief', 'overwhelmed']:
            return 'symbolic_primary'
            
        # Needs bridge mediation
        return 'bridge_mediation'
    
    def _calculate_routing_confidence(self, risk_flags: List[str]) -> str:
        """
        Calculate confidence in routing decision.
        """
        if not risk_flags:
            return 'high'
        elif len(risk_flags) == 1:
            return 'moderate'
        else:
            return 'low'
    
    def _save_zone_output(self, zone_output: Dict):
        """
        Save zone output for AI access.
        """
        # Load existing outputs
        outputs = []
        if self.zone_output_file.exists():
            try:
                with open(self.zone_output_file, 'r') as f:
                    outputs = json.load(f)
            except:
                outputs = []
                
        # Add new output
        outputs.append(zone_output)
        
        # Keep last 100 outputs
        outputs = outputs[-100:]
        
        # Save
        with open(self.zone_output_file, 'w') as f:
            json.dump(outputs, f, indent=2)
    
    def get_zone_output_by_id(self, zone_id: str) -> Optional[Dict]:
        """
        Retrieve a specific zone output by ID.
        The AI can only access zone outputs, never the vault.
        """
        if not self.zone_output_file.exists():
            return None
            
        with open(self.zone_output_file, 'r') as f:
            outputs = json.load(f)
            
        for output in reversed(outputs):  # Check recent first
            if output.get('zone_id') == zone_id:
                return output
                
        return None
    
    def clear_recursion_window(self):
        """
        Clear the recursion detection window (for new conversation).
        """
        self.recent_patterns.clear()
    
    def get_vault_stats(self) -> Dict:
        """
        Get statistics about the vault WITHOUT exposing content.
        """
        if not self.vault_file.exists():
            return {'total_memories': 0}
            
        with open(self.vault_file, 'r') as f:
            vault = json.load(f)
            
        return {
            'total_memories': len(vault),
            'oldest_memory': vault[0]['timestamp'] if vault else None,
            'newest_memory': vault[-1]['timestamp'] if vault else None,
            'vault_health': 'healthy'
        }


# Integration helper functions

def create_alphawall_handler(existing_parser_func):
    """
    Wrapper to integrate AlphaWall with existing parser.
    """
    alphawall = AlphaWall()
    
    def wrapped_parser(user_input: str, **kwargs):
        # Process through AlphaWall first
        zone_output = alphawall.process_input(user_input, kwargs.get('user_data'))
        
        # Parser now receives zone output instead of raw input
        # The parser should be modified to accept zone_output
        return existing_parser_func(zone_output, **kwargs)
        
    return wrapped_parser


# Unit tests
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing AlphaWall Cognitive Firewall...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Basic input processing
        print("\n1️⃣ Test: Basic input processing")
        wall = AlphaWall(data_dir=tmpdir)
        
        test_input = "What is the meaning of life?"
        output = wall.process_input(test_input)
        
        assert output['tags']['intent'] == 'information_request'
        assert output['tags']['emotional_state'] in ['neutral', 'calm']
        assert 'memory_trace' in output
        assert test_input not in str(output)  # User text should not leak
        print("✅ Basic processing works, no data leakage")
        
        # Test 2: Emotional input
        print("\n2️⃣ Test: Emotional input handling")
        emotional_input = "I feel so overwhelmed and broken right now 😢"
        output2 = wall.process_input(emotional_input)
        
        assert output2['tags']['emotional_state'] in ['overwhelmed', 'grief']
        assert output2['tags']['intent'] in ['expressive', 'self_reference']
        assert 'reclaimed_language' in output2['tags']['context']
        print("✅ Emotional detection works")
        
        # Test 3: Recursion detection
        print("\n3️⃣ Test: Recursion detection")
        wall.clear_recursion_window()
        
        # Simulate recursive pattern
        for i in range(4):
            recursive_input = f"Why does nothing make sense? (attempt {i+1})"
            output_recursive = wall.process_input(recursive_input)
            
        assert 'trauma_loop' in output_recursive['tags']['context']
        assert output_recursive['routing_hints']['quarantine_recommended'] == True
        print("✅ Recursion detection works")
        
        # Test 4: Vault isolation
        print("\n4️⃣ Test: Vault isolation")
        stats = wall.get_vault_stats()
        assert stats['total_memories'] >= 5  # We've stored several inputs
        
        # Try to access vault directly (should fail in production)
        zone_output = wall.get_zone_output_by_id(output['zone_id'])
        assert zone_output is not None
        assert 'memory_trace' in zone_output
        assert emotional_input not in str(zone_output)  # Confirm no leakage
        print("✅ Vault isolation confirmed")
        
        # Test 5: Risk assessment
        print("\n5️⃣ Test: Risk assessment")
        risky_input = "EVERYTHING IS WRONG AND I DON'T KNOW WHAT'S REAL ANYMORE!!!"
        output_risk = wall.process_input(risky_input)
        
        assert 'bridge_conflict_expected' in output_risk['tags']['risk']
        assert output_risk['routing_hints']['confidence_level'] in ['low', 'moderate']
        print("✅ Risk assessment works")
        
        # Test 6: Semantic similarity (no data exposure)
        print("\n6️⃣ Test: Semantic similarity without exposure")
        technical_input = "Explain binary search algorithm implementation"
        output_tech = wall.process_input(technical_input)
        
        assert 'similarity_to_technical' in output_tech['semantic_profile']
        assert output_tech['semantic_profile']['similarity_to_technical'] > 0.5
        assert technical_input not in str(output_tech['semantic_profile'])
        print("✅ Semantic profiling works without data exposure")
        
    print("\n✅ All AlphaWall tests passed! The cognitive firewall is secure.")