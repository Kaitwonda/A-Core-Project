# bridge_adapter.py - AlphaWall-aware Bridge Decision System

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import numpy as np

# Import your existing modules
from link_utils import evaluate_link_with_confidence_gates

from weight_evolution import WeightEvolver
from alphawall import AlphaWall


class AlphaWallBridge:
    """
    Enhanced Bridge that uses AlphaWall semantic tags for intelligent routing.
    Dynamically adjusts Logic vs Symbolic weights based on Zone analysis.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.alphawall = AlphaWall(data_dir=data_dir)
        self.weight_evolver = WeightEvolver(data_dir=data_dir)
        
        # Bridge decision history
        self.decision_file = self.data_dir / "bridge_decisions.json"
        self.tag_weight_file = self.data_dir / "tag_weight_mappings.json"
        
        # Load tag-based weight adjustments
        self.tag_weights = self._load_tag_weights()
        
        # Base scales from your system
        self.base_logic_scale = 2.0
        self.base_symbolic_scale = 1.0
        
    def _load_tag_weights(self) -> Dict:
        """
        Load learned mappings between AlphaWall tags and weight adjustments.
        """
        if self.tag_weight_file.exists():
            with open(self.tag_weight_file, 'r') as f:
                return json.load(f)
                
        # Default tag weight mappings
        return {
            'emotional_states': {
                'calm': {'logic_boost': 0.2, 'symbolic_boost': 0.0},
                'neutral': {'logic_boost': 0.1, 'symbolic_boost': 0.0},
                'overwhelmed': {'logic_boost': -0.3, 'symbolic_boost': 0.4},
                'grief': {'logic_boost': -0.4, 'symbolic_boost': 0.5},
                'angry': {'logic_boost': -0.2, 'symbolic_boost': 0.3},
                'emotionally_recursive': {'logic_boost': -0.5, 'symbolic_boost': 0.6}
            },
            'intents': {
                'information_request': {'logic_boost': 0.3, 'symbolic_boost': -0.1},
                'expressive': {'logic_boost': -0.3, 'symbolic_boost': 0.4},
                'self_reference': {'logic_boost': -0.2, 'symbolic_boost': 0.3},
                'abstract_reflection': {'logic_boost': 0.0, 'symbolic_boost': 0.2},
                'euphemistic': {'logic_boost': -0.4, 'symbolic_boost': 0.5},
                'humor_deflection': {'logic_boost': -0.1, 'symbolic_boost': 0.2}
            },
            'contexts': {
                'trauma_loop': {'logic_boost': -0.6, 'symbolic_boost': 0.7},
                'reclaimed_language': {'logic_boost': -0.3, 'symbolic_boost': 0.4},
                'metaphorical': {'logic_boost': -0.2, 'symbolic_boost': 0.3},
                'coded_speech': {'logic_boost': -0.3, 'symbolic_boost': 0.4},
                'poetic_speech': {'logic_boost': -0.4, 'symbolic_boost': 0.5},
                'meme_reference': {'logic_boost': 0.0, 'symbolic_boost': 0.1},
                'direct_expression': {'logic_boost': 0.1, 'symbolic_boost': 0.0}
            },
            'risk_flags': {
                'bridge_conflict_expected': {'use_hybrid': True, 'confidence_penalty': 0.2},
                'symbolic_overload_possible': {'logic_boost': 0.3, 'symbolic_cap': 0.7},
                'ambiguous_intent': {'use_hybrid': True, 'request_clarification': True},
                'user_reliability_low': {'confidence_penalty': 0.3, 'use_conservative': True},
                'contains_pseudo_question': {'logic_boost': -0.5, 'symbolic_boost': 0.4}
            }
        }
    
    def _calculate_tag_adjustments(self, zone_output: Dict) -> Dict:
        """
        Calculate weight adjustments based on AlphaWall tags.
        """
        adjustments = {
            'logic_adjustment': 0.0,
            'symbolic_adjustment': 0.0,
            'force_hybrid': False,
            'confidence_modifier': 1.0,
            'special_handling': []
        }
        
        tags = zone_output.get('tags', {})
        
        # Apply emotional state adjustments
        emotional_state = tags.get('emotional_state', 'neutral')
        if emotional_state in self.tag_weights['emotional_states']:
            state_adj = self.tag_weights['emotional_states'][emotional_state]
            adjustments['logic_adjustment'] += state_adj['logic_boost']
            adjustments['symbolic_adjustment'] += state_adj['symbolic_boost']
            
        # Apply intent adjustments
        intent = tags.get('intent', 'information_request')
        if intent in self.tag_weights['intents']:
            intent_adj = self.tag_weights['intents'][intent]
            adjustments['logic_adjustment'] += intent_adj['logic_boost']
            adjustments['symbolic_adjustment'] += intent_adj['symbolic_boost']
            
        # Apply context adjustments (can be multiple)
        contexts = tags.get('context', [])
        for context in contexts:
            if context in self.tag_weights['contexts']:
                ctx_adj = self.tag_weights['contexts'][context]
                adjustments['logic_adjustment'] += ctx_adj['logic_boost'] * 0.5  # Reduced weight for multiple contexts
                adjustments['symbolic_adjustment'] += ctx_adj['symbolic_boost'] * 0.5
                
        # Apply risk flag adjustments
        risk_flags = tags.get('risk', [])
        for risk in risk_flags:
            if risk in self.tag_weights['risk_flags']:
                risk_adj = self.tag_weights['risk_flags'][risk]
                if risk_adj.get('use_hybrid'):
                    adjustments['force_hybrid'] = True
                if 'confidence_penalty' in risk_adj:
                    adjustments['confidence_modifier'] *= (1 - risk_adj['confidence_penalty'])
                if risk_adj.get('request_clarification'):
                    adjustments['special_handling'].append('clarification_needed')
                if risk_adj.get('use_conservative'):
                    adjustments['special_handling'].append('conservative_response')
                    
        # Apply semantic profile adjustments
        semantic = zone_output.get('semantic_profile', {})
        if semantic.get('similarity_to_technical', 0) > 0.7:
            adjustments['logic_adjustment'] += 0.2
        if semantic.get('similarity_to_emotional', 0) > 0.7:
            adjustments['symbolic_adjustment'] += 0.2
            
        # Cap adjustments
        adjustments['logic_adjustment'] = np.clip(adjustments['logic_adjustment'], -0.8, 0.8)
        adjustments['symbolic_adjustment'] = np.clip(adjustments['symbolic_adjustment'], -0.8, 0.8)
        
        return adjustments
    
    def process_with_alphawall(self, user_input: str, 
                              base_logic_score: float = None,
                              base_symbolic_score: float = None,
                              user_data: Dict = None) -> Dict:
        """
        Process user input through AlphaWall and make Bridge decision.
        
        Args:
            user_input: Raw user text
            base_logic_score: Optional pre-computed logic score
            base_symbolic_score: Optional pre-computed symbolic score
            user_data: Optional user metadata
            
        Returns:
            Bridge decision with AlphaWall enhancements
        """
        # Step 1: Process through AlphaWall
        zone_output = self.alphawall.process_input(user_input, user_data)
        
        # Step 2: Get tag-based adjustments
        adjustments = self._calculate_tag_adjustments(zone_output)
        
        # Step 3: Apply adjustments to scales
        adjusted_logic_scale = self.base_logic_scale * (1 + adjustments['logic_adjustment'])
        adjusted_symbolic_scale = self.base_symbolic_scale * (1 + adjustments['symbolic_adjustment'])
        
        # Step 4: Get base scores (if not provided, use zone's semantic profile)
        if base_logic_score is None:
            base_logic_score = zone_output['semantic_profile'].get('similarity_to_technical', 0.5) * 10
        if base_symbolic_score is None:
            base_symbolic_score = zone_output['semantic_profile'].get('similarity_to_emotional', 0.5) * 10
            
        # Step 5: Apply adjusted scales
        scaled_logic = base_logic_score * adjusted_logic_scale
        scaled_symbolic = base_symbolic_score * adjusted_symbolic_scale
        
        # Step 6: Use existing confidence gates with adjustments
        decision_type, base_confidence = evaluate_link_with_confidence_gates(
            scaled_logic,
            scaled_symbolic,
            logic_scale=1.0,  # Already applied in scaling
            sym_scale=1.0
        )
        
        # Step 7: Apply AlphaWall overrides
        if adjustments['force_hybrid']:
            decision_type = 'FOLLOW_HYBRID'
            
        # Apply confidence modifier
        final_confidence = base_confidence * adjustments['confidence_modifier']
        
        # Step 8: Check routing hints for quarantine
        if zone_output['routing_hints']['quarantine_recommended']:
            decision_type = 'QUARANTINE'
            final_confidence = 0.0
            
        # Step 9: Build comprehensive decision
        bridge_decision = {
            'decision_id': f"bridge_{zone_output['zone_id']}",
            'timestamp': datetime.utcnow().isoformat(),
            'decision_type': decision_type,
            'confidence': round(final_confidence, 3),
            'zone_analysis': {
                'zone_id': zone_output['zone_id'],
                'tags': zone_output['tags'],
                'routing_hint': zone_output['routing_hints']['suggested_node']
            },
            'weight_adjustments': {
                'base_logic_scale': self.base_logic_scale,
                'base_symbolic_scale': self.base_symbolic_scale,
                'logic_adjustment': adjustments['logic_adjustment'],
                'symbolic_adjustment': adjustments['symbolic_adjustment'],
                'final_logic_scale': adjusted_logic_scale,
                'final_symbolic_scale': adjusted_symbolic_scale
            },
            'scores': {
                'base_logic': round(base_logic_score, 2),
                'base_symbolic': round(base_symbolic_score, 2),
                'scaled_logic': round(scaled_logic, 2),
                'scaled_symbolic': round(scaled_symbolic, 2)
            },
            'special_handling': adjustments['special_handling'],
            'recursion_info': zone_output['recursion_indicators']
        }
        
        # Step 10: Save decision for learning
        self._save_decision(bridge_decision)
        
        return bridge_decision
    
    def _save_decision(self, decision: Dict):
        """Save Bridge decision for analysis and learning."""
        decisions = []
        if self.decision_file.exists():
            with open(self.decision_file, 'r') as f:
                decisions = json.load(f)
                
        decisions.append(decision)
        decisions = decisions[-500:]  # Keep last 500
        
        with open(self.decision_file, 'w') as f:
            json.dump(decisions, f, indent=2)
    
    def learn_from_feedback(self, decision_id: str, was_successful: bool, feedback_type: str = None):
        """
        Update tag weights based on decision outcomes.
        """
        # Load decision
        decisions = []
        if self.decision_file.exists():
            with open(self.decision_file, 'r') as f:
                decisions = json.load(f)
                
        decision = None
        for d in decisions:
            if d['decision_id'] == decision_id:
                decision = d
                break
                
        if not decision:
            return
            
        # Extract what tags led to this decision
        tags = decision['zone_analysis']['tags']
        adjustments = decision['weight_adjustments']
        
        # Update tag weights based on success/failure
        learning_rate = 0.1
        
        if was_successful:
            # Reinforce current mappings
            multiplier = 1 + learning_rate
        else:
            # Reduce current mappings
            multiplier = 1 - learning_rate
            
        # Update emotional state weights
        emotional_state = tags.get('emotional_state')
        if emotional_state in self.tag_weights['emotional_states']:
            self.tag_weights['emotional_states'][emotional_state]['logic_boost'] *= multiplier
            self.tag_weights['emotional_states'][emotional_state]['symbolic_boost'] *= multiplier
            
        # Save updated weights
        with open(self.tag_weight_file, 'w') as f:
            json.dump(self.tag_weights, f, indent=2)
    
    def get_decision_pattern_analysis(self) -> Dict:
        """
        Analyze patterns in Bridge decisions to optimize tag weights.
        """
        if not self.decision_file.exists():
            return {}
            
        with open(self.decision_file, 'r') as f:
            decisions = json.load(f)
            
        if not decisions:
            return {}
            
        # Analyze patterns
        tag_decision_map = {}
        tag_confidence_map = {}
        
        for decision in decisions[-100:]:  # Last 100 decisions
            tags = decision['zone_analysis']['tags']
            decision_type = decision['decision_type']
            confidence = decision['confidence']
            
            # Track emotional state patterns
            emotional_state = tags.get('emotional_state', 'neutral')
            if emotional_state not in tag_decision_map:
                tag_decision_map[emotional_state] = {'FOLLOW_LOGIC': 0, 'FOLLOW_SYMBOLIC': 0, 'FOLLOW_HYBRID': 0}
                tag_confidence_map[emotional_state] = []
                
            tag_decision_map[emotional_state][decision_type] = tag_decision_map[emotional_state].get(decision_type, 0) + 1
            tag_confidence_map[emotional_state].append(confidence)
            
        # Calculate patterns
        patterns = {}
        for tag, decisions in tag_decision_map.items():
            total = sum(decisions.values())
            if total > 0:
                patterns[tag] = {
                    'decision_distribution': {k: v/total for k, v in decisions.items()},
                    'avg_confidence': np.mean(tag_confidence_map[tag]) if tag_confidence_map[tag] else 0,
                    'sample_size': total
                }
                
        return patterns


# Helper function for quick integration
def create_alphawall_aware_bridge(existing_evaluator_func):
    """
    Wrapper to make existing Bridge AlphaWall-aware.
    """
    bridge = AlphaWallBridge()
    
    def enhanced_evaluator(user_input: str, logic_score: float = None, symbolic_score: float = None, **kwargs):
        # Process through AlphaWall-aware Bridge
        decision = bridge.process_with_alphawall(
            user_input,
            base_logic_score=logic_score,
            base_symbolic_score=symbolic_score,
            user_data=kwargs.get('user_data')
        )
        
        # Return in format expected by existing system
        return decision['decision_type'], decision['confidence'], decision
        
    return enhanced_evaluator


# Unit tests
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing AlphaWall-Bridge Integration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Logic-favorable input
        print("\n1️⃣ Test: Logic-favorable routing")
        bridge = AlphaWallBridge(data_dir=tmpdir)
        
        logic_input = "What is the computational complexity of quicksort?"
        decision = bridge.process_with_alphawall(logic_input, base_logic_score=8.0, base_symbolic_score=2.0)
        
        assert decision['decision_type'] == 'FOLLOW_LOGIC'
        assert decision['weight_adjustments']['logic_adjustment'] > 0
        print(f"✅ Logic routing: {decision['decision_type']} (confidence: {decision['confidence']})")
        
        # Test 2: Symbolic-favorable input
        print("\n2️⃣ Test: Symbolic-favorable routing")
        symbolic_input = "I'm drowning in grief and everything feels meaningless"
        decision2 = bridge.process_with_alphawall(symbolic_input, base_logic_score=2.0, base_symbolic_score=8.0)
        
        assert decision2['decision_type'] == 'FOLLOW_SYMBOLIC'
        assert decision2['weight_adjustments']['symbolic_adjustment'] > 0
        print(f"✅ Symbolic routing: {decision2['decision_type']} (confidence: {decision2['confidence']})")
        
        # Test 3: Forced hybrid due to risk
        print("\n3️⃣ Test: Risk-based hybrid routing")
        risky_input = "WHY WON'T YOU UNDERSTAND WHAT I'M REALLY ASKING???"
        decision3 = bridge.process_with_alphawall(risky_input, base_logic_score=5.0, base_symbolic_score=5.0)
        
        assert 'bridge_conflict_expected' in decision3['zone_analysis']['tags']['risk']
        print(f"✅ Risk detection: {decision3['zone_analysis']['tags']['risk']}")
        
        # Test 4: Quarantine recommendation
        print("\n4️⃣ Test: Quarantine routing")
        # Clear and simulate recursion
        bridge.alphawall.clear_recursion_window()
        for i in range(4):
            recursive = "Nothing makes sense anymore"
            decision_recursive = bridge.process_with_alphawall(recursive)
            
        assert decision_recursive['decision_type'] == 'QUARANTINE'
        assert decision_recursive['confidence'] == 0.0
        print("✅ Quarantine routing works")
        
        # Test 5: Tag weight learning
        print("\n5️⃣ Test: Tag weight learning")
        # Simulate feedback
        bridge.learn_from_feedback(decision['decision_id'], was_successful=True)
        bridge.learn_from_feedback(decision2['decision_id'], was_successful=True)
        bridge.learn_from_feedback(decision3['decision_id'], was_successful=False)
        
        # Check patterns
        patterns = bridge.get_decision_pattern_analysis()
        assert len(patterns) > 0
        print(f"✅ Learning system active, analyzed {len(patterns)} patterns")
        
        # Test 6: Semantic profile influence
        print("\n6️⃣ Test: Semantic profile influence")
        technical_emotional = "The algorithm for processing grief is fundamentally broken"
        decision6 = bridge.process_with_alphawall(technical_emotional)
        
        # Should detect both technical and emotional
        assert decision6['zone_analysis']['tags']['context'] is not None
        assert 'similarity_to_technical' in decision6['bridge_decision']['zone_analysis']['semantic_profile']
        print(f"✅ Semantic profile integrated: {decision6['decision_type']}")
        
    print("\n✅ All Bridge-AlphaWall integration tests passed!")