# unified_weight_system.py - Unified Weight Management for Dual Brain AI

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class WeightDecision:
    """Structured result from unified weight calculation"""
    logic_scale: float
    symbolic_scale: float
    confidence_modifier: float
    decision_type: str
    reasoning: Dict[str, Any]
    metadata: Dict[str, Any]

class UnifiedWeightSystem:
    """
    Unified Weight System that combines:
    1. Autonomous learning from Weight Evolution
    2. Semantic context awareness from AlphaWall
    3. Reliable confidence-based routing
    
    Single source of truth for all weight decisions in the dual brain system.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load components
        self._load_autonomous_weights()
        self._load_semantic_adjustments()
        self._load_confidence_gates()
        
        # Decision history for learning
        self.decision_history_file = self.data_dir / "unified_weight_decisions.json"
        self.learning_stats_file = self.data_dir / "weight_learning_stats.json"
        
        # Performance tracking
        self.performance_stats = self._load_performance_stats()
        
    def _load_autonomous_weights(self):
        """Load autonomous learning weights from weight evolution system"""
        weights_file = self.data_dir / "adaptive_weights.json"
        
        if weights_file.exists():
            try:
                with open(weights_file, 'r') as f:
                    data = json.load(f)
                    self.base_static_weight = data.get('link_score_weight_static', 0.6)
                    self.base_dynamic_weight = data.get('link_score_weight_dynamic', 0.4)
                    self.weights_last_updated = data.get('last_updated')
            except Exception:
                self._set_default_autonomous_weights()
        else:
            self._set_default_autonomous_weights()
            
        # Convert to standardized logic/symbolic scales
        # Higher static weight = prefer logic (established patterns)
        # Higher dynamic weight = prefer symbolic (new/emotional content)
        total = self.base_static_weight + self.base_dynamic_weight
        static_ratio = self.base_static_weight / total
        
        # Map to 2.0/1.0 standard scale
        if static_ratio > 0.5:
            # Logic-favoring system
            self.base_logic_scale = 2.0
            self.base_symbolic_scale = 1.0 * (1 - (static_ratio - 0.5))
        else:
            # Symbolic-favoring system  
            self.base_symbolic_scale = 2.0
            self.base_logic_scale = 1.0 * (1 - (0.5 - static_ratio))
            
    def _set_default_autonomous_weights(self):
        """Set default autonomous weights"""
        self.base_static_weight = 0.6
        self.base_dynamic_weight = 0.4
        self.weights_last_updated = None
        
    def _load_semantic_adjustments(self):
        """Load semantic context adjustment mappings"""
        tag_weights_file = self.data_dir / "tag_weight_mappings.json"
        
        if tag_weights_file.exists():
            try:
                with open(tag_weights_file, 'r') as f:
                    self.semantic_adjustments = json.load(f)
            except Exception:
                self._set_default_semantic_adjustments()
        else:
            self._set_default_semantic_adjustments()
            
    def _set_default_semantic_adjustments(self):
        """Set default semantic adjustment mappings"""
        self.semantic_adjustments = {
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
                'direct_expression': {'logic_boost': 0.1, 'symbolic_boost': 0.0}
            }
        }
        
    def _load_confidence_gates(self):
        """Load confidence gate thresholds"""
        self.confidence_thresholds = {
            'high_confidence_logic': 6.0,    # logic_score * scale > this = high confidence logic
            'high_confidence_symbolic': 3.0,  # symbolic_score * scale > this = high confidence symbolic  
            'force_hybrid_threshold': 0.8,   # If scores within this ratio, force hybrid
            'quarantine_confidence': 0.3,    # Below this confidence = quarantine
            'min_decision_confidence': 0.5   # Minimum confidence for any decision
        }
        
    def _load_performance_stats(self):
        """Load learning performance statistics"""
        if self.learning_stats_file.exists():
            try:
                with open(self.learning_stats_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'successful_logic_routes': 0,
            'successful_symbolic_routes': 0,
            'successful_hybrid_routes': 0,
            'failed_routes': 0,
            'total_decisions': 0,
            'confidence_accuracy': []
        }
        
    def calculate_unified_weights(self, 
                                user_input: str = None,
                                semantic_tags: Dict = None,
                                memory_stats: Dict = None,
                                force_context: str = None) -> WeightDecision:
        """
        Calculate unified weights combining all systems.
        
        Args:
            user_input: Raw user text (for semantic analysis)
            semantic_tags: Pre-computed AlphaWall tags
            memory_stats: Current memory distribution statistics
            force_context: Override context for testing
            
        Returns:
            WeightDecision with unified scales and reasoning
        """
        
        reasoning = {
            'base_autonomous': {
                'static_weight': self.base_static_weight,
                'dynamic_weight': self.base_dynamic_weight,
                'base_logic_scale': self.base_logic_scale,
                'base_symbolic_scale': self.base_symbolic_scale
            },
            'semantic_adjustments': {},
            'confidence_factors': {},
            'final_calculation': {}
        }
        
        # Step 1: Start with autonomous base weights
        current_logic_scale = self.base_logic_scale
        current_symbolic_scale = self.base_symbolic_scale
        confidence_modifier = 1.0
        
        # Step 2: Apply semantic context adjustments
        if semantic_tags or force_context:
            tags = semantic_tags or {'emotional_state': force_context}
            adjustments = self._calculate_semantic_adjustments(tags)
            
            current_logic_scale *= (1 + adjustments['logic_adjustment'])
            current_symbolic_scale *= (1 + adjustments['symbolic_adjustment'])
            confidence_modifier = adjustments['confidence_modifier']
            
            reasoning['semantic_adjustments'] = {
                'tags_used': tags,
                'logic_adjustment': adjustments['logic_adjustment'],
                'symbolic_adjustment': adjustments['symbolic_adjustment'],
                'confidence_modifier': confidence_modifier,
                'special_handling': adjustments['special_handling']
            }
            
        # Step 3: Apply memory-based learning adjustments
        if memory_stats:
            memory_adjustments = self._calculate_memory_adjustments(memory_stats)
            current_logic_scale *= memory_adjustments['logic_multiplier']
            current_symbolic_scale *= memory_adjustments['symbolic_multiplier']
            
            reasoning['memory_adjustments'] = memory_adjustments
            
        # Step 4: Normalize and apply bounds
        current_logic_scale = np.clip(current_logic_scale, 0.1, 4.0)
        current_symbolic_scale = np.clip(current_symbolic_scale, 0.1, 4.0)
        confidence_modifier = np.clip(confidence_modifier, 0.1, 1.5)
        
        # Step 5: Determine decision type
        decision_type = self._determine_decision_type(
            current_logic_scale, 
            current_symbolic_scale,
            confidence_modifier
        )
        
        reasoning['final_calculation'] = {
            'final_logic_scale': current_logic_scale,
            'final_symbolic_scale': current_symbolic_scale,
            'scale_ratio': current_logic_scale / current_symbolic_scale,
            'decision_logic': f"Logic scale {current_logic_scale:.3f} vs Symbolic scale {current_symbolic_scale:.3f}"
        }
        
        # Step 6: Create decision object
        decision = WeightDecision(
            logic_scale=round(current_logic_scale, 3),
            symbolic_scale=round(current_symbolic_scale, 3),
            confidence_modifier=round(confidence_modifier, 3),
            decision_type=decision_type,
            reasoning=reasoning,
            metadata={
                'timestamp': datetime.utcnow().isoformat(),
                'autonomous_weights_age': self.weights_last_updated,
                'decision_id': f"unified_{hash(str(reasoning))}"[:12]
            }
        )
        
        # Step 7: Log decision for learning
        self._log_decision(decision)
        
        return decision
        
    def _calculate_semantic_adjustments(self, tags: Dict) -> Dict:
        """Calculate adjustments based on semantic tags"""
        adjustments = {
            'logic_adjustment': 0.0,
            'symbolic_adjustment': 0.0,
            'confidence_modifier': 1.0,
            'special_handling': []
        }
        
        # Apply emotional state adjustments
        emotional_state = tags.get('emotional_state', 'neutral')
        if emotional_state in self.semantic_adjustments['emotional_states']:
            state_adj = self.semantic_adjustments['emotional_states'][emotional_state]
            adjustments['logic_adjustment'] += state_adj['logic_boost']
            adjustments['symbolic_adjustment'] += state_adj['symbolic_boost']
            
        # Apply intent adjustments
        intent = tags.get('intent', 'information_request')
        if intent in self.semantic_adjustments['intents']:
            intent_adj = self.semantic_adjustments['intents'][intent]
            adjustments['logic_adjustment'] += intent_adj['logic_boost']
            adjustments['symbolic_adjustment'] += intent_adj['symbolic_boost']
            
        # Apply context adjustments
        contexts = tags.get('context', [])
        if isinstance(contexts, str):
            contexts = [contexts]
        for context in contexts:
            if context in self.semantic_adjustments['contexts']:
                ctx_adj = self.semantic_adjustments['contexts'][context]
                adjustments['logic_adjustment'] += ctx_adj['logic_boost'] * 0.5
                adjustments['symbolic_adjustment'] += ctx_adj['symbolic_boost'] * 0.5
                
        # Cap adjustments
        adjustments['logic_adjustment'] = np.clip(adjustments['logic_adjustment'], -0.8, 0.8)
        adjustments['symbolic_adjustment'] = np.clip(adjustments['symbolic_adjustment'], -0.8, 0.8)
        
        return adjustments
        
    def _calculate_memory_adjustments(self, memory_stats: Dict) -> Dict:
        """Calculate adjustments based on memory distribution"""
        adjustments = {
            'logic_multiplier': 1.0,
            'symbolic_multiplier': 1.0,
            'reasoning': ''
        }
        
        if 'distribution' not in memory_stats:
            return adjustments
            
        dist = memory_stats['distribution']
        bridge_pct = dist.get('bridge_pct', 0)
        logic_pct = dist.get('logic_pct', 0)
        symbolic_pct = dist.get('symbolic_pct', 0)
        
        # If bridge is large, adjust to help classification
        if bridge_pct > 30:
            if logic_pct > symbolic_pct * 2:
                # Too much logic bias, balance it
                adjustments['logic_multiplier'] = 0.9
                adjustments['symbolic_multiplier'] = 1.1
                adjustments['reasoning'] = 'Reducing logic bias due to large bridge'
            elif symbolic_pct > logic_pct * 2:
                # Too much symbolic bias, balance it
                adjustments['logic_multiplier'] = 1.1
                adjustments['symbolic_multiplier'] = 0.9
                adjustments['reasoning'] = 'Reducing symbolic bias due to large bridge'
                
        return adjustments
        
    def _determine_decision_type(self, logic_scale: float, symbolic_scale: float, confidence_modifier: float) -> str:
        """Determine routing decision type based on scales"""
        ratio = logic_scale / symbolic_scale
        
        if confidence_modifier < 0.5:
            return 'QUARANTINE'
        elif ratio > 1.5:
            return 'FOLLOW_LOGIC'
        elif ratio < 0.67:
            return 'FOLLOW_SYMBOLIC' 
        else:
            return 'FOLLOW_HYBRID'
            
    def route_with_unified_weights(self, 
                                 logic_score: float,
                                 symbolic_score: float,
                                 user_input: str = None,
                                 semantic_tags: Dict = None,
                                 memory_stats: Dict = None) -> Tuple[str, float, WeightDecision]:
        """
        Complete routing decision using unified weight system.
        
        Returns:
            (decision_type, confidence, weight_decision)
        """
        
        # Get unified weights
        weight_decision = self.calculate_unified_weights(
            user_input=user_input,
            semantic_tags=semantic_tags,
            memory_stats=memory_stats
        )
        
        # Apply weights to scores
        scaled_logic = logic_score * weight_decision.logic_scale
        scaled_symbolic = symbolic_score * weight_decision.symbolic_scale
        
        # Calculate final confidence
        max_score = max(scaled_logic, scaled_symbolic)
        score_difference = abs(scaled_logic - scaled_symbolic)
        
        # Base confidence from score strength and difference
        base_confidence = min(1.0, max_score / 10.0) * min(1.0, score_difference / 3.0)
        final_confidence = base_confidence * weight_decision.confidence_modifier
        
        # Apply confidence gates
        if final_confidence < self.confidence_thresholds['quarantine_confidence']:
            decision_type = 'QUARANTINE'
            final_confidence = 0.0
        elif final_confidence < self.confidence_thresholds['min_decision_confidence']:
            decision_type = 'FOLLOW_HYBRID'
        else:
            decision_type = weight_decision.decision_type
            
        # Update decision with routing results
        weight_decision.metadata.update({
            'input_scores': {'logic': logic_score, 'symbolic': symbolic_score},
            'scaled_scores': {'logic': scaled_logic, 'symbolic': scaled_symbolic},
            'final_confidence': round(final_confidence, 3),
            'routing_decision': decision_type
        })
        
        return decision_type, final_confidence, weight_decision
        
    def _log_decision(self, decision: WeightDecision):
        """Log decision for learning and analysis"""
        decisions = []
        if self.decision_history_file.exists():
            try:
                with open(self.decision_history_file, 'r') as f:
                    decisions = json.load(f)
            except Exception:
                pass
                
        decisions.append(decision.__dict__)
        decisions = decisions[-1000:]  # Keep last 1000
        
        with open(self.decision_history_file, 'w') as f:
            json.dump(decisions, f, indent=2)
            
    def learn_from_feedback(self, decision_id: str, was_successful: bool, feedback_data: Dict = None):
        """Update system based on routing decision feedback"""
        self.performance_stats['total_decisions'] += 1
        
        if was_successful:
            # Find which route was taken and increment success counter
            # This would integrate with the autonomous weight evolution
            pass
        else:
            self.performance_stats['failed_routes'] += 1
            
        # Save updated stats
        with open(self.learning_stats_file, 'w') as f:
            json.dump(self.performance_stats, f, indent=2)
            
    def get_system_status(self) -> Dict:
        """Get current status of unified weight system"""
        return {
            'autonomous_weights': {
                'static': self.base_static_weight,
                'dynamic': self.base_dynamic_weight,
                'last_updated': self.weights_last_updated
            },
            'base_scales': {
                'logic': self.base_logic_scale,
                'symbolic': self.base_symbolic_scale
            },
            'performance': self.performance_stats,
            'confidence_thresholds': self.confidence_thresholds,
            'semantic_adjustment_categories': list(self.semantic_adjustments.keys())
        }


# Convenience function for easy integration
def create_unified_router(data_dir="data"):
    """Create a unified weight-aware routing function"""
    weight_system = UnifiedWeightSystem(data_dir=data_dir)
    
    def route(logic_score: float, symbolic_score: float, user_input: str = None, **kwargs):
        return weight_system.route_with_unified_weights(
            logic_score=logic_score,
            symbolic_score=symbolic_score,
            user_input=user_input,
            semantic_tags=kwargs.get('semantic_tags'),
            memory_stats=kwargs.get('memory_stats')
        )
    
    return route


# Unit tests
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing Unified Weight System...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Basic weight calculation
        print("\n1️⃣ Test: Basic unified weight calculation")
        system = UnifiedWeightSystem(data_dir=tmpdir)
        
        decision = system.calculate_unified_weights()
        assert decision.logic_scale > 0
        assert decision.symbolic_scale > 0
        assert decision.decision_type in ['FOLLOW_LOGIC', 'FOLLOW_SYMBOLIC', 'FOLLOW_HYBRID']
        print(f"✅ Basic calculation: Logic={decision.logic_scale}, Symbolic={decision.symbolic_scale}")
        
        # Test 2: Semantic context awareness
        print("\n2️⃣ Test: Semantic context adjustments")
        emotional_decision = system.calculate_unified_weights(
            semantic_tags={'emotional_state': 'grief', 'intent': 'expressive'}
        )
        
        # Should favor symbolic for grief/expressive content
        assert emotional_decision.symbolic_scale >= decision.symbolic_scale
        print(f"✅ Emotional context: Symbolic scale increased to {emotional_decision.symbolic_scale}")
        
        # Test 3: Complete routing
        print("\n3️⃣ Test: Complete routing with unified weights")
        decision_type, confidence, weight_decision = system.route_with_unified_weights(
            logic_score=8.0,
            symbolic_score=3.0
        )
        
        assert decision_type in ['FOLLOW_LOGIC', 'FOLLOW_SYMBOLIC', 'FOLLOW_HYBRID', 'QUARANTINE']
        assert 0 <= confidence <= 1
        print(f"✅ Complete routing: {decision_type} (confidence: {confidence:.3f})")
        
        # Test 4: Memory-based adjustments
        print("\n4️⃣ Test: Memory distribution adjustments")
        memory_stats = {
            'distribution': {
                'logic_pct': 70,
                'symbolic_pct': 10,
                'bridge_pct': 20
            }
        }
        
        memory_decision = system.calculate_unified_weights(memory_stats=memory_stats)
        print(f"✅ Memory-aware: Logic={memory_decision.logic_scale}, Symbolic={memory_decision.symbolic_scale}")
        
        # Test 5: System status
        print("\n5️⃣ Test: System status reporting")
        status = system.get_system_status()
        assert 'autonomous_weights' in status
        assert 'base_scales' in status
        assert 'performance' in status
        print("✅ System status complete")
        
        # Test 6: Convenience router
        print("\n6️⃣ Test: Convenience router function")
        router = create_unified_router(data_dir=tmpdir)
        route_result = router(logic_score=5.0, symbolic_score=7.0)
        assert len(route_result) == 3
        print(f"✅ Convenience router: {route_result[0]} (confidence: {route_result[1]:.3f})")
        
    print("\n✅ All unified weight system tests passed!")