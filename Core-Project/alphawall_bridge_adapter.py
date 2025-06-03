# alphawall_bridge_adapter.py - Integration layer between AlphaWall and existing AI nodes

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import AlphaWall and your existing modules
from alphawall import AlphaWall
from vector_engine import fuse_vectors
from link_evaluator import evaluate_link_with_confidence_gates
import parser as P_Parser
import symbol_memory as SM_SymbolMemory


class AlphaWallBridgeAdapter:
    """
    Bridges AlphaWall's semantic tags with your existing parser/link_evaluator system.
    Ensures the AI only sees tags, never raw user data.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.alphawall = AlphaWall(data_dir=data_dir)
        
        # Cache for tag-to-action mappings
        self.tag_mappings = self._load_tag_mappings()
        
        # Bridge decision history
        self.decision_history = []
        
    def _load_tag_mappings(self) -> Dict:
        """
        Load predefined mappings for how different tag combinations 
        should influence parser behavior.
        """
        return {
            # Intent + Emotion → Parser Action
            ('information_request', 'calm'): {
                'parser_mode': 'standard',
                'weight_adjustment': {'logic': 1.2, 'symbolic': 0.8}
            },
            ('information_request', 'overwhelmed'): {
                'parser_mode': 'supportive',
                'weight_adjustment': {'logic': 0.8, 'symbolic': 1.2}
            },
            ('expressive', 'grief'): {
                'parser_mode': 'empathetic',
                'weight_adjustment': {'logic': 0.3, 'symbolic': 1.7}
            },
            ('self_reference', 'emotionally_recursive'): {
                'parser_mode': 'redirect',
                'weight_adjustment': {'logic': 0.5, 'symbolic': 0.5},
                'special_handling': 'break_loop'
            },
            ('humor_deflection', '*'): {
                'parser_mode': 'acknowledge_humor',
                'weight_adjustment': {'logic': 0.7, 'symbolic': 1.3}
            },
            ('abstract_reflection', '*'): {
                'parser_mode': 'philosophical',
                'weight_adjustment': {'logic': 1.0, 'symbolic': 1.0}
            }
        }
    
    def process_user_input(self, user_text: str, user_data: Dict = None) -> Dict:
        """
        Main processing pipeline with AlphaWall integration.
        Replaces direct text parsing with tag-based routing.
        """
        # Step 1: Process through AlphaWall
        zone_output = self.alphawall.process_input(user_text, user_data)
        
        # Step 2: Convert tags to parser instructions
        parser_config = self._tags_to_parser_config(zone_output)
        
        # Step 3: Generate synthetic input for parser (tags only, no user data)
        synthetic_input = self._generate_synthetic_input(zone_output)
        
        # Step 4: Parse with modified weights
        parser_output = self._parse_with_alphawall_context(
            synthetic_input, 
            parser_config,
            zone_output
        )
        
        # Step 5: Evaluate through link evaluator with tag context
        final_decision = self._evaluate_with_tags(parser_output, zone_output)
        
        # Step 6: Record decision for learning
        self._record_decision(zone_output, parser_output, final_decision)
        
        return {
            'zone_id': zone_output['zone_id'],
            'parser_output': parser_output,
            'final_decision': final_decision,
            'alphawall_tags': zone_output['tags'],
            'routing_used': parser_config['parser_mode']
        }
    
    def _tags_to_parser_config(self, zone_output: Dict) -> Dict:
        """
        Convert AlphaWall tags into parser configuration.
        """
        tags = zone_output['tags']
        intent = tags['intent']
        emotion = tags['emotional_state']
        
        # Look up mapping
        key = (intent, emotion)
        if key in self.tag_mappings:
            config = self.tag_mappings[key].copy()
        else:
            # Check wildcard mappings
            wildcard_key = (intent, '*')
            if wildcard_key in self.tag_mappings:
                config = self.tag_mappings[wildcard_key].copy()
            else:
                # Default configuration
                config = {
                    'parser_mode': 'standard',
                    'weight_adjustment': {'logic': 1.0, 'symbolic': 1.0}
                }
        
        # Apply risk adjustments
        if 'bridge_conflict_expected' in tags['risk']:
            config['needs_bridge_mediation'] = True
            
        if 'symbolic_overload_possible' in tags['risk']:
            # Reduce symbolic weight to prevent overload
            config['weight_adjustment']['symbolic'] *= 0.7
            
        if 'user_reliability_low' in tags['risk']:
            config['apply_skepticism'] = True
            
        return config
    
    def _generate_synthetic_input(self, zone_output: Dict) -> str:
        """
        Generate synthetic text that represents the semantic content
        without containing any actual user data.
        """
        tags = zone_output['tags']
        intent = tags['intent']
        emotion = tags['emotional_state']
        contexts = tags['context']
        
        # Map to abstract representations
        synthetic_templates = {
            'information_request': "REQUEST_FOR_INFORMATION TYPE_QUERY",
            'expressive': "EMOTIONAL_EXPRESSION SHARING_STATE",
            'self_reference': "SELF_REFERENTIAL_STATEMENT PERSONAL_CONTEXT",
            'abstract_reflection': "PHILOSOPHICAL_INQUIRY ABSTRACT_CONCEPT",
            'euphemistic': "INDIRECT_REFERENCE CODED_MEANING",
            'humor_deflection': "HUMOR_MECHANISM DEFLECTION_PATTERN"
        }
        
        # Build synthetic input
        base = synthetic_templates.get(intent, "GENERAL_INPUT")
        
        # Add emotion markers
        emotion_markers = {
            'calm': "EMOTION_STABLE",
            'overwhelmed': "EMOTION_INTENSE",
            'grief': "EMOTION_LOSS",
            'angry': "EMOTION_FRUSTRATION",
            'emotionally_recursive': "EMOTION_LOOP"
        }
        
        synthetic = f"{base} {emotion_markers.get(emotion, 'EMOTION_NEUTRAL')}"
        
        # Add context flags
        for context in contexts:
            synthetic += f" CONTEXT_{context.upper()}"
            
        # Add semantic similarity hints
        if 'semantic_profile' in zone_output:
            profile = zone_output['semantic_profile']
            dominant = max(profile.items(), key=lambda x: x[1]) if profile else None
            if dominant and dominant[1] > 0.6:
                synthetic += f" SEMANTIC_{dominant[0].upper()}"
                
        return synthetic
    
    def _parse_with_alphawall_context(self, 
                                    synthetic_input: str, 
                                    parser_config: Dict,
                                    zone_output: Dict) -> Dict:
        """
        Modified parser that works with synthetic input and AlphaWall context.
        """
        # Extract symbols based on context tags, not user text
        active_symbols = []
        
        if 'metaphorical' in zone_output['tags']['context']:
            active_symbols.extend(['🌀', '💭', '🔮'])  # Metaphor symbols
            
        if 'emotional_recursive' in zone_output['tags']['emotional_state']:
            active_symbols.extend(['🔄', '♾️', '🔁'])  # Recursion symbols
            
        if 'coded_speech' in zone_output['tags']['context']:
            active_symbols.extend(['🔐', '🗝️', '📝'])  # Coded meaning symbols
            
        # Calculate scores based on tags, not content
        logic_score = self._calculate_logic_score_from_tags(zone_output)
        symbolic_score = self._calculate_symbolic_score_from_tags(zone_output)
        
        # Apply weight adjustments from config
        logic_score *= parser_config['weight_adjustment']['logic']
        symbolic_score *= parser_config['weight_adjustment']['symbolic']
        
        # Build parser output
        parser_output = {
            'synthetic_input': synthetic_input,
            'parser_mode': parser_config['parser_mode'],
            'logic_score': round(logic_score, 2),
            'symbolic_score': round(symbolic_score, 2),
            'extracted_symbols': active_symbols,
            'processing_hints': {
                'needs_bridge': parser_config.get('needs_bridge_mediation', False),
                'apply_skepticism': parser_config.get('apply_skepticism', False),
                'special_handling': parser_config.get('special_handling')
            }
        }
        
        return parser_output
    
    def _calculate_logic_score_from_tags(self, zone_output: Dict) -> float:
        """
        Calculate logic score purely from tags, no user content.
        """
        score = 0.0
        tags = zone_output['tags']
        
        # Intent-based scoring
        logic_intents = {
            'information_request': 3.0,
            'abstract_reflection': 2.0,
            'humor_deflection': 1.0
        }
        score += logic_intents.get(tags['intent'], 0.5)
        
        # Emotion modifiers
        if tags['emotional_state'] in ['calm', 'neutral']:
            score *= 1.2
        elif tags['emotional_state'] in ['overwhelmed', 'emotionally_recursive']:
            score *= 0.7
            
        # Semantic profile boost
        if 'semantic_profile' in zone_output:
            tech_similarity = zone_output['semantic_profile'].get('similarity_to_technical', 0)
            score += tech_similarity * 2.0
            
        return min(score, 10.0)
    
    def _calculate_symbolic_score_from_tags(self, zone_output: Dict) -> float:
        """
        Calculate symbolic score purely from tags, no user content.
        """
        score = 0.0
        tags = zone_output['tags']
        
        # Intent-based scoring
        symbolic_intents = {
            'expressive': 3.0,
            'self_reference': 2.5,
            'euphemistic': 2.0,
            'abstract_reflection': 1.5
        }
        score += symbolic_intents.get(tags['intent'], 0.5)
        
        # Emotion boost
        emotion_weights = {
            'overwhelmed': 2.0,
            'grief': 2.5,
            'angry': 1.5,
            'emotionally_recursive': 3.0
        }
        score += emotion_weights.get(tags['emotional_state'], 0.5)
        
        # Context modifiers
        symbolic_contexts = {
            'metaphorical': 1.5,
            'poetic_speech': 2.0,
            'reclaimed_language': 1.0,
            'trauma_loop': 2.5
        }
        for context in tags['context']:
            score += symbolic_contexts.get(context, 0)
            
        # Semantic profile boost
        if 'semantic_profile' in zone_output:
            emotional_similarity = zone_output['semantic_profile'].get('similarity_to_emotional', 0)
            score += emotional_similarity * 2.0
            
        return min(score, 10.0)
    
    def _evaluate_with_tags(self, parser_output: Dict, zone_output: Dict) -> Dict:
        """
        Use link evaluator with AlphaWall context for final decision.
        """
        # Get scores
        logic_score = parser_output['logic_score']
        symbolic_score = parser_output['symbolic_score']
        
        # Check if quarantine recommended
        if zone_output['routing_hints']['quarantine_recommended']:
            return {
                'decision_type': 'QUARANTINED',
                'confidence': 1.0,
                'reason': 'AlphaWall quarantine recommendation',
                'safe_response': 'I notice you might be going through something difficult. Would you like to talk about something else?'
            }
        
        # Use existing link evaluator
        decision_type, confidence = evaluate_link_with_confidence_gates(
            logic_score,
            symbolic_score,
            logic_scale=2.0,
            sym_scale=1.0
        )
        
        # Generate response strategy based on decision and tags
        response_strategy = self._determine_response_strategy(
            decision_type,
            zone_output['tags'],
            parser_output['processing_hints']
        )
        
        return {
            'decision_type': decision_type,
            'confidence': confidence,
            'response_strategy': response_strategy,
            'tag_context': {
                'primary_emotion': zone_output['tags']['emotional_state'],
                'primary_intent': zone_output['tags']['intent'],
                'active_risks': zone_output['tags']['risk']
            }
        }
    
    def _determine_response_strategy(self, 
                                   decision_type: str, 
                                   tags: Dict,
                                   hints: Dict) -> Dict:
        """
        Determine how to respond based on decision and tag context.
        """
        strategies = {
            ('FOLLOW_LOGIC', 'information_request'): {
                'tone': 'informative',
                'structure': 'clear_explanation',
                'elements': ['facts', 'examples', 'logic_flow']
            },
            ('FOLLOW_SYMBOLIC', 'expressive'): {
                'tone': 'empathetic',
                'structure': 'supportive_response',
                'elements': ['validation', 'understanding', 'gentle_guidance']
            },
            ('FOLLOW_SYMBOLIC', 'self_reference'): {
                'tone': 'reflective',
                'structure': 'mirror_and_support',
                'elements': ['acknowledgment', 'reframe', 'hope']
            },
            ('FOLLOW_HYBRID', 'abstract_reflection'): {
                'tone': 'philosophical',
                'structure': 'balanced_exploration',
                'elements': ['concepts', 'perspectives', 'synthesis']
            }
        }
        
        key = (decision_type, tags['intent'])
        strategy = strategies.get(key, {
            'tone': 'neutral',
            'structure': 'standard_response',
            'elements': ['acknowledgment', 'content', 'closing']
        })
        
        # Apply special handling
        if hints.get('special_handling') == 'break_loop':
            strategy['special'] = 'redirect_from_recursion'
            
        if hints.get('apply_skepticism'):
            strategy['verification'] = 'high'
            
        return strategy
    
    def _record_decision(self, zone_output: Dict, parser_output: Dict, final_decision: Dict):
        """
        Record decision for analysis and learning.
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'zone_id': zone_output['zone_id'],
            'tags': zone_output['tags'],
            'scores': {
                'logic': parser_output['logic_score'],
                'symbolic': parser_output['symbolic_score']
            },
            'decision': final_decision['decision_type'],
            'confidence': final_decision['confidence']
        }
        
        self.decision_history.append(record)
        
        # Keep last 100 decisions
        self.decision_history = self.decision_history[-100:]
        
        # Save to file for analysis
        history_file = self.data_dir / "alphawall_bridge_decisions.json"
        with open(history_file, 'w') as f:
            json.dump(self.decision_history, f, indent=2)
    
    def get_routing_stats(self) -> Dict:
        """
        Get statistics on routing decisions.
        """
        if not self.decision_history:
            return {'total_decisions': 0}
            
        total = len(self.decision_history)
        
        # Count decision types
        decisions = [d['decision'] for d in self.decision_history]
        decision_counts = {
            'FOLLOW_LOGIC': decisions.count('FOLLOW_LOGIC'),
            'FOLLOW_SYMBOLIC': decisions.count('FOLLOW_SYMBOLIC'),
            'FOLLOW_HYBRID': decisions.count('FOLLOW_HYBRID'),
            'QUARANTINED': decisions.count('QUARANTINED')
        }
        
        # Average confidence by decision type
        confidence_by_type = {}
        for decision_type in decision_counts.keys():
            matching = [d['confidence'] for d in self.decision_history if d['decision'] == decision_type]
            if matching:
                confidence_by_type[decision_type] = sum(matching) / len(matching)
                
        # Tag frequency
        all_intents = [d['tags']['intent'] for d in self.decision_history]
        all_emotions = [d['tags']['emotional_state'] for d in self.decision_history]
        
        return {
            'total_decisions': total,
            'decision_distribution': decision_counts,
            'average_confidence': confidence_by_type,
            'common_intents': self._get_top_n(all_intents, 3),
            'common_emotions': self._get_top_n(all_emotions, 3)
        }
    
    def _get_top_n(self, items: List[str], n: int) -> List[Tuple[str, int]]:
        """Get top N most common items with counts."""
        from collections import Counter
        counter = Counter(items)
        return counter.most_common(n)


# Convenience functions for integration

def create_alphawall_pipeline(data_dir="data"):
    """
    Create a complete AlphaWall-integrated pipeline.
    """
    bridge = AlphaWallBridgeAdapter(data_dir=data_dir)
    
    def process_input(user_text: str, **kwargs) -> Dict:
        """
        Drop-in replacement for your current input processor.
        """
        return bridge.process_user_input(user_text, kwargs)
        
    return process_input


# Example integration with your existing code
def integrate_with_existing_parser():
    """
    Shows how to modify your existing parser.py to work with AlphaWall.
    """
    bridge = AlphaWallBridgeAdapter()
    
    # Replace in parser.py:
    # OLD: def parse_input(user_text: str) -> Dict:
    # NEW:
    def parse_input_with_alphawall(user_text: str) -> Dict:
        # Process through AlphaWall bridge
        result = bridge.process_user_input(user_text)
        
        # Extract what the parser needs
        parser_output = result['parser_output']
        decision = result['final_decision']
        
        # Return in your expected format
        return {
            'content_type': decision['response_strategy']['structure'],
            'classification': decision['decision_type'],
            'confidence': decision['confidence'],
            'logic_score': parser_output['logic_score'],
            'symbolic_score': parser_output['symbolic_score'],
            'symbols': parser_output['extracted_symbols']
        }
    
    return parse_input_with_alphawall


# Unit tests
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing AlphaWall Bridge Adapter...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Basic integration
        print("\n1️⃣ Test: Basic pipeline integration")
        bridge = AlphaWallBridgeAdapter(data_dir=tmpdir)
        
        result = bridge.process_user_input("How does photosynthesis work?")
        
        assert 'zone_id' in result
        assert 'parser_output' in result
        assert 'final_decision' in result
        assert result['alphawall_tags']['intent'] == 'information_request'
        assert result['final_decision']['decision_type'] in ['FOLLOW_LOGIC', 'FOLLOW_HYBRID']
        print("✅ Basic integration works")
        
        # Test 2: Emotional routing
        print("\n2️⃣ Test: Emotional content routing")
        result2 = bridge.process_user_input("I'm feeling so lost and broken 😢")
        
        assert result2['alphawall_tags']['emotional_state'] in ['overwhelmed', 'grief']
        assert result2['final_decision']['decision_type'] in ['FOLLOW_SYMBOLIC', 'FOLLOW_HYBRID']
        assert result2['final_decision']['response_strategy']['tone'] == 'empathetic'
        print("✅ Emotional routing works")
        
        # Test 3: Recursion handling
        print("\n3️⃣ Test: Recursion detection and handling")
        bridge.alphawall.clear_recursion_window()
        
        # Simulate recursive input
        for i in range(5):
            result_recursive = bridge.process_user_input("Why why why why why???")
            
        assert 'trauma_loop' in result_recursive['alphawall_tags']['context']
        assert result_recursive['final_decision']['decision_type'] == 'QUARANTINED'
        print("✅ Recursion handling works")
        
        # Test 4: Synthetic input generation
        print("\n4️⃣ Test: Synthetic input generation (no data leakage)")
        test_input = "My secret password is ABC123!"
        result4 = bridge.process_user_input(test_input)
        
        # Check no user data in processing pipeline
        synthetic = result4['parser_output']['synthetic_input']
        assert "ABC123" not in synthetic
        assert "password" not in synthetic
        assert "REQUEST_FOR_INFORMATION" in synthetic or "SELF_REFERENTIAL_STATEMENT" in synthetic
        print("✅ No data leakage confirmed")
        
        # Test 5: Tag-based scoring
        print("\n5️⃣ Test: Tag-based scoring without content")
        technical_input = "Explain quantum computing algorithms"
        result5 = bridge.process_user_input(technical_input)
        
        logic_score = result5['parser_output']['logic_score']
        symbolic_score = result5['parser_output']['symbolic_score']
        
        assert logic_score > symbolic_score  # Technical query should favor logic
        assert result5['final_decision']['decision_type'] in ['FOLLOW_LOGIC', 'FOLLOW_HYBRID']
        print("✅ Tag-based scoring works")
        
        # Test 6: Routing statistics
        print("\n6️⃣ Test: Routing statistics")
        stats = bridge.get_routing_stats()
        
        assert stats['total_decisions'] >= 5
        assert 'FOLLOW_LOGIC' in stats['decision_distribution']
        assert len(stats['common_intents']) > 0
        print("✅ Statistics tracking works")
        
    print("\n✅ All AlphaWall Bridge Adapter tests passed!")