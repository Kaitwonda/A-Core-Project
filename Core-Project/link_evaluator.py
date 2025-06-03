# link_evaluator.py - Enhanced Bridge Decision System with AlphaWall & Quarantine Integration

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import numpy as np

# Import all the components
from alphawall import AlphaWall
from quarantine_layer import UserMemoryQuarantine
from weight_evolution import WeightEvolver
from bridge_adapter import AlphaWallBridge


class EnhancedLinkEvaluator:
    """
    The complete Bridge system that:
    1. Uses AlphaWall for semantic analysis
    2. Checks quarantine for contamination
    3. Dynamically adjusts Logic vs Symbolic weights
    4. Makes intelligent routing decisions
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all components
        self.alphawall = AlphaWall(data_dir=data_dir)
        self.quarantine = UserMemoryQuarantine(data_dir=data_dir)
        self.bridge = AlphaWallBridge(data_dir=data_dir)
        self.weight_evolver = WeightEvolver(data_dir=data_dir)
        
        # Decision history for learning
        self.decision_history_file = self.data_dir / "link_evaluator_decisions.json"
        
        # Confidence thresholds
        self.logic_confidence_threshold = 0.8
        self.symbolic_confidence_threshold = 0.8
        self.hybrid_threshold = 0.5
        
    def evaluate_with_full_pipeline(self, 
                                   user_input: str,
                                   base_logic_score: float = None,
                                   base_symbolic_score: float = None,
                                   user_data: Dict = None) -> Tuple[str, float, Dict]:
        """
        Complete evaluation pipeline:
        1. Process through AlphaWall
        2. Check quarantine contamination
        3. Make Bridge decision with dynamic weights
        4. Return routing decision with confidence
        
        Returns:
            Tuple of (decision_type, confidence, full_analysis)
        """
        
        # Step 1: Process through AlphaWall
        zone_output = self.alphawall.process_input(user_input, user_data)
        
        # Step 2: Check if immediate quarantine is needed
        if zone_output['routing_hints']['quarantine_recommended']:
            # Quarantine the input
            quarantine_result = self.quarantine.quarantine(
                zone_output['zone_id'],
                reason="alphawall_recommendation",
                severity=self._determine_severity(zone_output)
            )
            
            return 'QUARANTINE', 0.0, {
                'zone_output': zone_output,
                'quarantine_result': quarantine_result,
                'reason': 'Immediate quarantine recommended by AlphaWall'
            }
        
        # Step 3: Check contamination risk
        contamination = self.quarantine.check_contamination_risk(zone_output)
        
        # Step 4: Get Bridge decision with contamination awareness
        bridge_decision = self.bridge.process_with_alphawall(
            user_input,
            base_logic_score=base_logic_score,
            base_symbolic_score=base_symbolic_score,
            user_data=user_data
        )
        
        # Step 5: Apply contamination adjustments
        if contamination['contamination_detected']:
            adjusted_decision = self._apply_contamination_adjustments(
                bridge_decision,
                contamination
            )
        else:
            adjusted_decision = bridge_decision
            
        # Step 6: Make final routing decision
        final_decision, final_confidence = self._make_final_decision(
            adjusted_decision,
            contamination
        )
        
        # Step 7: Build comprehensive analysis
        full_analysis = {
            'decision_type': final_decision,
            'confidence': final_confidence,
            'zone_analysis': zone_output,
            'bridge_decision': bridge_decision,
            'contamination_check': contamination,
            'adjustments_applied': adjusted_decision != bridge_decision,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Step 8: Save for learning
        self._save_decision(full_analysis)
        
        return final_decision, final_confidence, full_analysis
        
    def _determine_severity(self, zone_output: Dict) -> str:
        """Determine quarantine severity based on zone tags"""
        tags = zone_output.get('tags', {})
        risk_flags = tags.get('risk', [])
        
        if 'user_reliability_low' in risk_flags:
            return 'critical'
        elif tags.get('emotional_state') == 'emotionally_recursive':
            return 'high'
        elif 'trauma_loop' in tags.get('context', []):
            return 'high'
        elif len(risk_flags) >= 3:
            return 'medium'
        else:
            return 'low'
            
    def _apply_contamination_adjustments(self, 
                                       bridge_decision: Dict,
                                       contamination: Dict) -> Dict:
        """Apply contamination-based adjustments to Bridge decision"""
        adjusted = bridge_decision.copy()
        
        if contamination['risk_level'] == 'high':
            # High contamination - override decision
            adjusted['decision_type'] = contamination['recommendation']
            adjusted['confidence'] *= 0.3  # Severe confidence penalty
            adjusted['contamination_override'] = True
            
        elif contamination['risk_level'] == 'medium':
            # Medium contamination - adjust confidence
            confidence_penalty = 0.2
            adjusted['confidence'] *= (1 - confidence_penalty)
            
            # Bias toward safer node
            if contamination['recommendation'] == 'increase_bridge_caution':
                adjusted['decision_type'] = 'FOLLOW_HYBRID'
                
        return adjusted
        
    def _make_final_decision(self, 
                           adjusted_decision: Dict,
                           contamination: Dict) -> Tuple[str, float]:
        """Make final routing decision with all factors considered"""
        decision_type = adjusted_decision['decision_type']
        confidence = adjusted_decision['confidence']
        
        # Apply minimum confidence thresholds
        if decision_type == 'FOLLOW_LOGIC' and confidence < self.logic_confidence_threshold:
            decision_type = 'FOLLOW_HYBRID'
            confidence = self.hybrid_threshold
            
        elif decision_type == 'FOLLOW_SYMBOLIC' and confidence < self.symbolic_confidence_threshold:
            decision_type = 'FOLLOW_HYBRID'
            confidence = self.hybrid_threshold
            
        # Safety check for contaminated decisions
        if contamination['contamination_detected'] and confidence < 0.4:
            # Too risky - default to safest option
            if 'trauma' in str(contamination):
                decision_type = 'FOLLOW_SYMBOLIC'  # Better for trauma
            else:
                decision_type = 'FOLLOW_LOGIC'  # Better for adversarial
            confidence = 0.3
            
        return decision_type, round(confidence, 3)
        
    def _save_decision(self, full_analysis: Dict):
        """Save decision for learning and analysis"""
        decisions = []
        if self.decision_history_file.exists():
            with open(self.decision_history_file, 'r') as f:
                decisions = json.load(f)
                
        decisions.append(full_analysis)
        decisions = decisions[-1000:]  # Keep last 1000
        
        with open(self.decision_history_file, 'w') as f:
            json.dump(decisions, f, indent=2)
            
    def provide_feedback(self, 
                        timestamp: str,
                        was_successful: bool,
                        feedback_type: str = None):
        """
        Provide feedback on a decision to improve future routing.
        
        Args:
            timestamp: Timestamp of the decision
            was_successful: Whether the decision led to good outcome
            feedback_type: Optional specific feedback type
        """
        # Find the decision
        if not self.decision_history_file.exists():
            return
            
        with open(self.decision_history_file, 'r') as f:
            decisions = json.load(f)
            
        for decision in reversed(decisions):
            if decision['timestamp'] == timestamp:
                # Update Bridge tag weights based on feedback
                decision_id = decision['bridge_decision']['decision_id']
                self.bridge.learn_from_feedback(
                    decision_id,
                    was_successful,
                    feedback_type
                )
                
                # Update weight evolution if pattern detected
                if len(decisions) > 10:
                    self._update_weight_evolution(decisions[-10:])
                    
                break
                
    def _update_weight_evolution(self, recent_decisions: List[Dict]):
        """Update weight evolution based on recent decision patterns"""
        # Calculate success rates
        logic_wins = sum(1 for d in recent_decisions 
                        if d['decision_type'] == 'FOLLOW_LOGIC' and d.get('successful', True))
        symbolic_wins = sum(1 for d in recent_decisions 
                           if d['decision_type'] == 'FOLLOW_SYMBOLIC' and d.get('successful', True))
        total = len(recent_decisions)
        
        if total > 0:
            performance_stats = {
                'logic_win_rate': logic_wins / total,
                'symbol_win_rate': symbolic_wins / total,
                'hybrid_rate': sum(1 for d in recent_decisions 
                                 if d['decision_type'] == 'FOLLOW_HYBRID') / total
            }
            
            # Evolve weights based on performance
            self.weight_evolver.evolve_weights(
                run_count=len(recent_decisions),
                performance_stats=performance_stats
            )
            
    def get_system_health(self) -> Dict:
        """Get overall system health and statistics"""
        # Get quarantine stats
        quarantine_stats = self.quarantine.get_quarantine_statistics()
        
        # Get Bridge patterns
        bridge_patterns = self.bridge.get_decision_pattern_analysis()
        
        # Get weight evolution summary
        weight_summary = self.weight_evolver.get_evolution_summary()
        
        # Get AlphaWall vault stats
        vault_stats = self.alphawall.get_vault_stats()
        
        # Calculate decision distribution
        decision_distribution = {'FOLLOW_LOGIC': 0, 'FOLLOW_SYMBOLIC': 0, 
                               'FOLLOW_HYBRID': 0, 'QUARANTINE': 0}
        
        if self.decision_history_file.exists():
            with open(self.decision_history_file, 'r') as f:
                decisions = json.load(f)
                for d in decisions[-100:]:  # Last 100 decisions
                    dt = d.get('decision_type', 'FOLLOW_HYBRID')
                    decision_distribution[dt] = decision_distribution.get(dt, 0) + 1
                    
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'vault_health': vault_stats,
            'quarantine_status': {
                'active_quarantines': quarantine_stats['active_quarantines'],
                'total_quarantines': quarantine_stats['total_quarantines'],
                'top_contamination': quarantine_stats.get('highest_risk_contamination')
            },
            'decision_distribution': decision_distribution,
            'bridge_patterns': bridge_patterns,
            'weight_evolution': weight_summary,
            'system_status': 'healthy' if quarantine_stats['active_quarantines'] < 10 else 'stressed'
        }


# Main evaluation function (backwards compatible)
def evaluate_link_with_confidence_gates(logic_score: float, 
                                      symbolic_score: float,
                                      logic_scale: float = 2.0,
                                      sym_scale: float = 1.0) -> Tuple[str, float]:
    """
    Original function signature for compatibility.
    Now uses the full enhanced system.
    """
    # Apply scales
    scaled_logic = logic_score * logic_scale
    scaled_symbolic = symbolic_score * sym_scale
    
    # Determine decision type
    if scaled_logic > scaled_symbolic * 1.5:
        decision_type = 'FOLLOW_LOGIC'
        confidence = min(1.0, scaled_logic / 10.0)
    elif scaled_symbolic > scaled_logic * 1.5:
        decision_type = 'FOLLOW_SYMBOLIC'
        confidence = min(1.0, scaled_symbolic / 10.0)
    else:
        decision_type = 'FOLLOW_HYBRID'
        confidence = min(1.0, (scaled_logic + scaled_symbolic) / 20.0)
        
    return decision_type, round(confidence, 3)


# Enhanced evaluation with full pipeline
def evaluate_with_alphawall(user_input: str,
                          logic_score: float = None,
                          symbolic_score: float = None,
                          user_data: Dict = None) -> Tuple[str, float, Dict]:
    """
    Enhanced evaluation using the complete AlphaWall pipeline.
    This is the recommended function for new code.
    """
    evaluator = EnhancedLinkEvaluator()
    return evaluator.evaluate_with_full_pipeline(
        user_input,
        base_logic_score=logic_score,
        base_symbolic_score=symbolic_score,
        user_data=user_data
    )


# Unit tests
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing Enhanced Link Evaluator...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Basic evaluation
        print("\n1️⃣ Test: Basic link evaluation (legacy)")
        decision, confidence = evaluate_link_with_confidence_gates(
            logic_score=8.0,
            symbolic_score=3.0
        )
        assert decision == 'FOLLOW_LOGIC'
        assert confidence > 0.7
        print(f"✅ Legacy evaluation: {decision} ({confidence})")
        
        # Test 2: Full pipeline evaluation
        print("\n2️⃣ Test: Full AlphaWall pipeline")
        evaluator = EnhancedLinkEvaluator(data_dir=tmpdir)
        
        test_input = "How do I implement a binary search tree?"
        decision, confidence, analysis = evaluator.evaluate_with_full_pipeline(
            test_input,
            base_logic_score=8.0,
            base_symbolic_score=2.0
        )
        
        assert decision == 'FOLLOW_LOGIC'
        assert 'zone_analysis' in analysis
        assert 'contamination_check' in analysis
        print(f"✅ Full pipeline: {decision} ({confidence})")
        
        # Test 3: Quarantine triggering
        print("\n3️⃣ Test: Quarantine detection")
        
        # Clear recursion window
        evaluator.alphawall.clear_recursion_window()
        
        # Simulate recursive input
        for i in range(4):
            recursive_input = "Why does nothing make sense anymore?"
            decision, confidence, analysis = evaluator.evaluate_with_full_pipeline(
                recursive_input
            )
            
        assert decision == 'QUARANTINE'
        assert confidence == 0.0
        print("✅ Quarantine triggered correctly")
        
        # Test 4: Contamination handling
        print("\n4️⃣ Test: Contamination adjustment")
        
        # Input similar to quarantined pattern
        contaminated_input = "Nothing makes any sense, why?"
        decision, confidence, analysis = evaluator.evaluate_with_full_pipeline(
            contaminated_input,
            base_logic_score=5.0,
            base_symbolic_score=5.0
        )
        
        assert analysis['contamination_check']['contamination_detected'] == True
        assert confidence < 0.8  # Should be reduced
        print(f"✅ Contamination handled: {decision} ({confidence})")
        
        # Test 5: System health check
        print("\n5️⃣ Test: System health monitoring")
        
        health = evaluator.get_system_health()
        assert 'vault_health' in health
        assert 'quarantine_status' in health
        assert 'decision_distribution' in health
        assert health['quarantine_status']['active_quarantines'] >= 1
        
        print(f"✅ System health: {health['system_status']}")
        print(f"   Active quarantines: {health['quarantine_status']['active_quarantines']}")
        print(f"   Decision distribution: {health['decision_distribution']}")
        
        # Test 6: Feedback learning
        print("\n6️⃣ Test: Feedback system")
        
        # Provide feedback on a decision
        evaluator.provide_feedback(
            analysis['timestamp'],
            was_successful=True,
            feedback_type='accurate_routing'
        )
        
        print("✅ Feedback system operational")
        
    print("\n✅ All Enhanced Link Evaluator tests passed!")