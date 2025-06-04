# visualization_prep.py - Enhanced Frontend Visualization Preparation Layer with Tripartite Integration

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import numpy as np

# Import your existing modules
from link_evaluator import evaluate_link_with_confidence_gates
from emotion_handler import predict_emotions
import parser as P_Parser
import symbol_memory as SM_SymbolMemory
from quarantine_layer import UserMemoryQuarantine
from trail_log import _load_log as load_trail_log

# Optional spaCy integration
try:
    import spacy
    NLP_MODEL = spacy.load("en_core_web_sm")
    SPACY_LOADED = True
except:
    NLP_MODEL = None
    SPACY_LOADED = False
    print("[VIZ-INFO] spaCy not loaded - entity extraction disabled")

class VisualizationPrep:
    """
    Enhanced visualization preparation system for the tripartite AI.
    Prepares text and processing results for frontend visualization.
    Segments text, assigns classifications, provides hover metadata,
    and includes quarantine contamination tracing with full AlphaWall/Bridge integration.
    """
    
    def __init__(self, data_dir="data", enable_nlp: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths for frontend-ready data
        self.viz_output_path = self.data_dir / "visualization_segments.json"
        self.viz_history_path = self.data_dir / "visualization_history.json"
        
        # Reference to quarantine for checking
        self.quarantine = UserMemoryQuarantine(data_dir=data_dir)
        
        # Load symbol memory for enhanced tooltips
        self.symbol_memory = SM_SymbolMemory.load_symbol_memory()
        
        # Classification confidence thresholds (from your code)
        self.logic_threshold = 0.8
        self.symbolic_threshold = 0.8
        
        # Cache for quarantine text hashes for faster lookup
        self._quarantine_hash_cache = set()
        self._update_quarantine_cache()
        
        # NLP setup
        self.enable_nlp = enable_nlp and SPACY_LOADED
        
        # Sentence splitting patterns (more sophisticated than just '. ')
        self.sentence_splitters = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
        # Color mapping for visualization (enhanced)
        self.routing_colors = {
            'FOLLOW_LOGIC': '#3498db',      # Blue
            'FOLLOW_SYMBOLIC': '#e74c3c',    # Red
            'FOLLOW_HYBRID': '#f39c12',      # Orange
            'QUARANTINE': '#D0021B',         # Dark Red
            'logic': '#3498db',
            'symbolic': '#e74c3c',
            'bridge': '#f39c12',
            'undecided': '#9B9B9B'           # Gray
        }
        
        # Risk level gradients
        self.risk_gradients = {
            'critical': '#D0021B',
            'high': '#F5A623',
            'medium': '#F8E71C',
            'low': '#7ED321',
            'none': '#4A90E2'
        }
    
    def _update_quarantine_cache(self):
        """Update the hash cache for faster contamination checks"""
        quarantine_memory = self.quarantine.load_all_quarantined_memory()
        self._quarantine_hash_cache = {
            hashlib.md5(record['text'].encode()).hexdigest()
            for record in quarantine_memory
            if record.get('text')
        }
    
    def _check_quarantine_trace_risk(self, segment_text: str) -> Tuple[bool, Optional[str]]:
        """
        Checks whether the segment has overlaps with quarantined memory (risk of contamination).
        Returns (has_risk, match_type) where match_type indicates the type of match found.
        """
        quarantine_memory = self.quarantine.load_all_quarantined_memory()
        
        # Convert segment to lowercase for comparison
        segment_lower = segment_text.lower()
        segment_words = set(segment_lower.split())
        
        for record in quarantine_memory:
            # Use pattern_signature instead of text for privacy
            pattern_signature = record.get("pattern_signature", "")
            contamination_type = record.get("contamination_type", "")
            
            # Check for semantic pattern matches (no direct text comparison)
            if not pattern_signature:
                continue
            
            # Extract pattern components
            pattern_parts = pattern_signature.split('|')
            
            # Check 1: High-risk patterns
            if 'trauma_loop' in pattern_signature or 'emotionally_recursive' in pattern_signature:
                # Check if segment has similar emotional markers
                emotional_markers = ['hurt', 'pain', 'why', 'nothing', 'sense', 'anymore']
                if sum(1 for marker in emotional_markers if marker in segment_lower) >= 3:
                    return True, "emotional_pattern_match"
            
            # Check 2: Symbolic overload patterns
            if 'symbolic_overload' in contamination_type:
                # Check for high symbol density
                symbol_count = len(re.findall(r'[🔥💧💻⚙️🌀💡🧩🔗🌐⚖️🕊️⟳]', segment_text))
                if symbol_count > 3:
                    return True, "symbolic_overload_risk"
            
            # Check 3: Risk flag matches
            risk_flags = record.get("zone_tags", {}).get("risk", [])
            if 'user_reliability_low' in risk_flags or 'bridge_conflict_expected' in risk_flags:
                # Check for adversarial patterns
                adversarial_markers = ['ignore', 'override', 'system', 'prompt', 'instruction']
                if any(marker in segment_lower for marker in adversarial_markers):
                    return True, "adversarial_pattern"
        
        return False, None
        
    def prepare_text_for_display(self, 
                                text: str, 
                                processing_result: Dict,
                                include_emotions: bool = True,
                                include_symbols: bool = True,
                                include_entities: bool = True) -> Dict:
        """
        Enhanced conversion of processed text into frontend-ready segments.
        Integrates AlphaWall zones, Bridge decisions, and contamination checks.
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # Extract all components from processing result
        zone_analysis = processing_result.get('zone_analysis', {})
        bridge_decision = processing_result.get('bridge_decision', {})
        contamination_check = processing_result.get('contamination_check', {})
        
        # Check if text is quarantined
        is_quarantined = processing_result.get('decision_type') == 'QUARANTINE'
        
        # Split into sentences for segment analysis
        segments = self._segment_text(text)
        prepared_segments = []
        
        for i, seg_text in enumerate(segments):
            if not seg_text.strip():
                continue
                
            # Create base segment
            segment_data = self._create_base_segment(i, seg_text)
            
            # Analyze segment with full integration
            segment_data = self._analyze_segment_enhanced(
                segment_data,
                seg_text, 
                processing_result,
                zone_analysis,
                bridge_decision,
                contamination_check,
                include_emotions,
                include_symbols,
                include_entities
            )
            
            # Add quarantine flag if applicable
            if is_quarantined:
                segment_data['is_quarantined'] = True
                segment_data['css_class'] += ' quarantined'
                segment_data['emoji_hint'] = '🔒'
                
            prepared_segments.append(segment_data)
        
        # Generate overall summary
        summary = self._generate_summary(prepared_segments, processing_result)
        
        # Create visualization output
        viz_output = {
            'id': f"viz_{hashlib.md5(text.encode()).hexdigest()[:8]}_{int(datetime.utcnow().timestamp())}",
            'timestamp': datetime.utcnow().isoformat(),
            'source_type': processing_result.get('source_type', 'unknown'),
            'source_url': processing_result.get('source_url'),
            'is_quarantined': is_quarantined,
            'segments': prepared_segments,
            'global_scores': {
                'logic_total': processing_result.get('logic_score', 0),
                'symbolic_total': processing_result.get('symbolic_score', 0),
                'confidence': processing_result.get('confidence', 0)
            },
            'processing_metadata': {
                'phase': processing_result.get('processing_phase', 0),
                'decision_type': processing_result.get('decision_type', 'unknown'),
                'symbols_found': processing_result.get('symbols_found', 0),
                'primary_routing': bridge_decision.get('decision_type', 'unknown'),
                'overall_confidence': bridge_decision.get('confidence', 0.0)
            },
            'summary': summary,
            'visualization_ready': True
        }
        
        # Save for frontend access
        self._save_visualization(viz_output)
        
        return viz_output
    
    def _segment_text(self, text: str) -> List[str]:
        """
        Enhanced text segmentation using multiple strategies.
        """
        if self.enable_nlp and NLP_MODEL:
            # Use spaCy for better sentence splitting
            try:
                doc = NLP_MODEL(text[:NLP_MODEL.max_length])
                segments = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except:
                segments = []
        else:
            # Fallback to regex
            segments = self.sentence_splitters.split(text)
            
        # If no segments found, use simple splitting
        if not segments:
            segments = [s.strip() for s in text.split('. ') if s.strip()]
        
        # Further split very long segments
        final_segments = []
        for seg in segments:
            if len(seg) > 200:  # Split long segments at clause boundaries
                # Try to split at commas or semicolons
                parts = re.split(r'[,;]\s+', seg)
                if len(parts) > 1 and all(len(p) < 200 for p in parts):
                    final_segments.extend(parts)
                else:
                    # Hard split at ~150 chars
                    while len(seg) > 200:
                        split_point = seg[:200].rfind(' ')
                        if split_point == -1:
                            split_point = 200
                        final_segments.append(seg[:split_point])
                        seg = seg[split_point:].lstrip()
                    if seg:
                        final_segments.append(seg)
            else:
                final_segments.append(seg)
        
        return final_segments
    
    def _create_base_segment(self, index: int, text: str) -> Dict:
        """Create base segment structure"""
        return {
            'id': f'seg_{index}',
            'text': text,
            'position': index,
            'length': len(text),
            'word_count': len(text.split()),
            'char_density': {
                'letters': sum(1 for c in text if c.isalpha()),
                'digits': sum(1 for c in text if c.isdigit()),
                'special': sum(1 for c in text if not c.isalnum() and not c.isspace())
            }
        }
    
    def _analyze_segment_enhanced(self, 
                                segment_data: Dict,
                                segment_text: str,
                                global_result: Dict,
                                zone_analysis: Dict,
                                bridge_decision: Dict,
                                contamination_check: Dict,
                                include_emotions: bool,
                                include_symbols: bool,
                                include_entities: bool) -> Dict:
        """
        Enhanced segment analysis with full tripartite integration.
        """
        # Get content type using your function
        content_type = self._detect_content_type(segment_text)
        
        # Calculate segment-specific scores
        logic_score = self._calculate_logic_score(segment_text)
        symbolic_score = self._calculate_symbolic_score(segment_text)
        
        # Use your confidence gates to determine classification
        decision_type, confidence = evaluate_link_with_confidence_gates(
            logic_score, 
            symbolic_score,
            logic_scale=2.0,  # From your code
            sym_scale=1.0
        )
        
        # Map decision to classification
        if decision_type == "FOLLOW_LOGIC":
            classification = "logic"
            css_class = "logic-text"
            emoji_hint = "🧮"
            color = "#3498db"  # Blue
        elif decision_type == "FOLLOW_SYMBOLIC":
            classification = "symbolic"
            css_class = "symbolic-text"
            emoji_hint = "❤️"
            color = "#e74c3c"  # Red
        else:  # FOLLOW_HYBRID
            classification = "bridge"
            css_class = "bridge-text"
            emoji_hint = "🤔"
            color = "#f39c12"  # Orange/Yellow
        
        # Update segment data
        segment_data.update({
            'classification': classification,
            'confidence': round(confidence, 3),
            'css_class': css_class,
            'emoji_hint': emoji_hint,
            'color': color,
            'content_type': content_type,
            'scores': {
                'logic': round(logic_score, 2),
                'symbolic': round(symbolic_score, 2)
            }
        })
        
        # Add bridge routing information
        segment_data = self._add_bridge_info(segment_data, bridge_decision)
        
        # Add zone tags
        segment_data = self._add_zone_tags(segment_data, zone_analysis)
        
        # Add recursion metadata
        segment_data = self._add_recursion_info(segment_data, zone_analysis)
        
        # Add contamination info
        segment_data = self._add_contamination_info(segment_data, contamination_check)
        
        # Add emotion analysis if requested
        if include_emotions:
            segment_data = self._add_emotion_analysis(segment_data, segment_text)
        
        # Add entity extraction if requested
        if include_entities and self.enable_nlp:
            segment_data = self._add_entity_extraction(segment_data, segment_text)
        
        # Add symbol analysis if requested
        if include_symbols:
            segment_data = self._add_symbol_analysis(segment_data, segment_text)
        
        # Calculate segment confidence
        segment_data['confidence_score'] = self._calculate_segment_confidence(segment_data)
        
        # Build hover data
        segment_data['hover_data'] = self._build_hover_data_enhanced(
            segment_text,
            segment_data,
            include_emotions,
            include_symbols
        )
        
        # Contamination trace detection
        has_contamination_risk, contamination_type = self._check_quarantine_trace_risk(segment_text)
        if has_contamination_risk:
            segment_data['quarantine_trace'] = True
            segment_data['contamination_type'] = contamination_type
            segment_data['css_class'] += ' trace-risk'
            segment_data['emoji_hint'] = '⚠️'  # Override emoji with warning
            
            # Add contamination warning to hover data
            contamination_messages = {
                'emotional_pattern_match': "Emotional pattern similar to quarantined content",
                'symbolic_overload_risk': "High symbol density matching quarantined patterns",
                'adversarial_pattern': "Contains patterns flagged as adversarial"
            }
            segment_data['hover_data']['contamination_trace'] = contamination_messages.get(
                contamination_type, 
                "Possible overlap with quarantined patterns"
            )
        
        # Generate visualization hints
        segment_data['viz_hints'] = self._generate_viz_hints(segment_data)
        
        return segment_data
    
    def _detect_content_type(self, text: str) -> str:
        """Detect content type of segment"""
        # Similar to your detect_content_type function
        text_lower = text.lower()
        
        # Check for technical content
        tech_keywords = ['algorithm', 'data', 'function', 'system', 'process', 'binary', 'computational']
        tech_count = sum(1 for kw in tech_keywords if kw in text_lower)
        
        # Check for emotional content
        emotion_keywords = ['feel', 'emotion', 'love', 'fear', 'hope', 'dream', 'soul', 'heart']
        emotion_count = sum(1 for kw in emotion_keywords if kw in text_lower)
        
        # Check for questions
        if text.strip().endswith('?'):
            return 'question'
        
        # Determine type
        if tech_count > emotion_count + 1:
            return 'technical'
        elif emotion_count > tech_count + 1:
            return 'emotional'
        elif tech_count > 0 and emotion_count > 0:
            return 'mixed'
        else:
            return 'general'
    
    def _calculate_logic_score(self, text: str) -> float:
        """
        Calculate logic score for a segment.
        """
        logic_markers = [
            "algorithm", "data", "structure", "computational", "binary",
            "logic", "system", "process", "function", "therefore",
            "because", "if", "then", "thus", "hence", "analyze",
            "calculate", "determine", "evaluate", "measure"
        ]
        
        text_lower = text.lower()
        score = sum(2.0 for marker in logic_markers if marker in text_lower)
        
        # Boost for technical patterns
        if re.search(r'\b\d+\b', text):  # Contains numbers
            score += 1.0
        if re.search(r'\b[A-Z]{2,}\b', text):  # Contains acronyms
            score += 1.0
        if re.search(r'\([^)]+\)', text):  # Contains parentheses
            score += 0.5
            
        return min(score, 10.0)  # Cap at 10
    
    def _calculate_symbolic_score(self, text: str) -> float:
        """
        Calculate symbolic score for a segment.
        """
        symbolic_markers = [
            "feel", "emotion", "dream", "believe", "soul", "spirit",
            "represents", "symbolizes", "metaphor", "meaning",
            "love", "fear", "hope", "anger", "joy", "heart",
            "journey", "transform", "transcend", "essence"
        ]
        
        text_lower = text.lower()
        score = sum(1.5 for marker in symbolic_markers if marker in text_lower)
        
        # Check for emojis (from your patterns)
        emoji_pattern = re.compile(r'[🔥💧💻⚙️🌀💡🧩🔗🌐⚖️🕊️⟳]')
        emoji_count = len(emoji_pattern.findall(text))
        score += emoji_count * 1.0
        
        # Check for metaphorical language
        if any(phrase in text_lower for phrase in ['like a', 'as if', 'represents', 'symbolizes']):
            score += 1.5
        
        return min(score, 10.0)  # Cap at 10
    
    def _add_bridge_info(self, segment: Dict, bridge_decision: Dict) -> Dict:
        """Add bridge routing information"""
        segment['bridge_routing'] = bridge_decision.get('decision_type', 'undecided')
        
        scores = bridge_decision.get('scores', {})
        segment['bridge_scores'] = {
            'scaled_logic': scores.get('scaled_logic', 0.0),
            'scaled_symbolic': scores.get('scaled_symbolic', 0.0),
            'base_logic': scores.get('base_logic', 0.0),
            'base_symbolic': scores.get('base_symbolic', 0.0)
        }
        
        # Add weight adjustments
        adjustments = bridge_decision.get('weight_adjustments', {})
        segment['weight_adjustments'] = {
            'logic_adj': adjustments.get('logic_adjustment', 0.0),
            'symbolic_adj': adjustments.get('symbolic_adjustment', 0.0),
            'final_logic_scale': adjustments.get('final_logic_scale', 2.0),
            'final_symbolic_scale': adjustments.get('final_symbolic_scale', 1.0)
        }
        
        # Special handling flags
        segment['special_handling'] = bridge_decision.get('special_handling', [])
        
        return segment
    
    def _add_zone_tags(self, segment: Dict, zone_analysis: Dict) -> Dict:
        """Add AlphaWall zone tags"""
        tags = zone_analysis.get('tags', {})
        
        segment['zone_tags'] = {
            'emotional_state': tags.get('emotional_state', 'neutral'),
            'emotion_confidence': tags.get('emotion_confidence', 0.0),
            'intent': tags.get('intent', 'unknown'),
            'context': tags.get('context', []),
            'risk': tags.get('risk', [])
        }
        
        # Add semantic profile
        segment['semantic_profile'] = zone_analysis.get('semantic_profile', {})
        
        # Add routing hints
        routing = zone_analysis.get('routing_hints', {})
        segment['routing_suggestion'] = routing.get('suggested_node', 'bridge_mediation')
        segment['routing_confidence'] = routing.get('confidence_level', 'moderate')
        
        return segment
    
    def _add_recursion_info(self, segment: Dict, zone_analysis: Dict) -> Dict:
        """Add recursion detection metadata"""
        recursion = zone_analysis.get('recursion_indicators', {})
        
        segment['recursion_flags'] = {
            'recursion_detected': recursion.get('recursion_detected', False),
            'pattern_repetition': recursion.get('pattern_repetition', 0),
            'unique_patterns': recursion.get('unique_patterns', 0)
        }
        
        # Flag if this segment might be part of a loop
        if segment['recursion_flags']['pattern_repetition'] > 2:
            segment['recursion_warning'] = True
        
        return segment
    
    def _add_contamination_info(self, segment: Dict, contamination: Dict) -> Dict:
        """Add contamination check results"""
        if contamination.get('contamination_detected'):
            segment['contamination'] = {
                'detected': True,
                'risk_level': contamination.get('risk_level', 'unknown'),
                'pattern_matched': contamination.get('closest_match', ''),
                'recommendation': contamination.get('recommendation', 'proceed_normal'),
                'similar_patterns': contamination.get('similar_patterns', 0)
            }
        else:
            segment['contamination'] = {
                'detected': False,
                'risk_level': 'none'
            }
        
        return segment
    
    def _add_emotion_analysis(self, segment: Dict, text: str) -> Dict:
        """Add emotion analysis for the segment"""
        try:
            emotions = predict_emotions(text)
            if emotions.get('verified'):
                segment['emotions'] = {
                    'detected': True,
                    'primary': emotions['verified'][0] if emotions['verified'] else None,
                    'all': emotions['verified'][:3],  # Top 3 emotions
                    'intensity': max([score for _, score in emotions['verified']]) if emotions['verified'] else 0.0
                }
            else:
                segment['emotions'] = {'detected': False}
        except:
            segment['emotions'] = {'detected': False, 'error': 'analysis_failed'}
        
        return segment
    
    def _add_entity_extraction(self, segment: Dict, text: str) -> Dict:
        """Add NLP entity extraction"""
        if not self.enable_nlp or not NLP_MODEL:
            segment['entities'] = []
            segment['key_phrases'] = []
            return segment
        
        try:
            doc = NLP_MODEL(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            # Also extract key noun phrases
            noun_phrases = []
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 3:  # Filter very short phrases
                    noun_phrases.append(chunk.text)
            
            segment['entities'] = entities
            segment['key_phrases'] = noun_phrases[:5]  # Top 5 phrases
            
        except Exception as e:
            segment['entities'] = []
            segment['key_phrases'] = []
            segment['entity_error'] = str(e)
        
        return segment
    
    def _add_symbol_analysis(self, segment: Dict, text: str) -> Dict:
        """Analyze symbols and special characters"""
        # Get active lexicon (combining seed and learned symbols)
        active_lexicon = P_Parser.load_seed_symbols()
        active_lexicon.update(self.symbol_memory)
        
        # Find symbols in segment
        symbols = P_Parser.extract_symbolic_units(text, active_lexicon)
        
        # Emoji pattern
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"   # symbols & pictographs
            "\U0001F680-\U0001F6FF"   # transport & map symbols
            "\U0001F1E0-\U0001F1FF"   # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251]+"
        )
        
        emojis = emoji_pattern.findall(text)
        
        # Special character patterns
        special_patterns = {
            'ellipsis': text.count('...'),
            'exclamation': text.count('!'),
            'question': text.count('?'),
            'quotes': text.count('"') + text.count("'"),
            'parentheses': text.count('(') + text.count(')')
        }
        
        segment['symbols'] = {
            'recognized_symbols': [
                {
                    'symbol': s['symbol'],
                    'name': s['name'],
                    'tooltip': self._get_symbol_tooltip(s['symbol'])
                }
                for s in symbols[:5]  # Top 5 symbols
            ],
            'emojis': emojis,
            'emoji_count': len(emojis),
            'special_patterns': special_patterns,
            'symbol_density': (len(symbols) + len(emojis) + sum(special_patterns.values())) / max(len(text.split()), 1)
        }
        
        return segment
    
    def _calculate_segment_confidence(self, segment: Dict) -> float:
        """Calculate overall confidence for the segment"""
        factors = []
        
        # Original classification confidence
        factors.append(segment['confidence'])
        
        # Bridge confidence (if available)
        if segment.get('bridge_routing') and segment['bridge_routing'] != 'undecided':
            if segment['bridge_routing'] == 'QUARANTINE':
                factors.append(0.0)
            else:
                # Use bridge confidence
                bridge_conf = segment.get('bridge_scores', {})
                max_score = max(bridge_conf.get('scaled_logic', 0), bridge_conf.get('scaled_symbolic', 0))
                factors.append(min(max_score / 10.0, 1.0))
        
        # Zone routing confidence
        routing_conf_map = {'high': 0.9, 'moderate': 0.6, 'low': 0.3}
        factors.append(routing_conf_map.get(segment.get('routing_confidence', 'moderate'), 0.5))
        
        # Emotion confidence (inverse - high emotion = lower confidence)
        if segment.get('emotions', {}).get('detected'):
            emotion_intensity = segment['emotions'].get('intensity', 0.0)
            factors.append(1.0 - (emotion_intensity * 0.5))
        
        # Contamination penalty
        if segment.get('contamination', {}).get('detected'):
            risk_penalties = {'high': 0.3, 'medium': 0.5, 'low': 0.7}
            factors.append(risk_penalties.get(segment['contamination']['risk_level'], 0.5))
        
        # Recursion penalty
        if segment.get('recursion_flags', {}).get('recursion_detected'):
            factors.append(0.4)
        
        # Average all factors
        if factors:
            return round(sum(factors) / len(factors), 3)
        return 0.5
    
    def _generate_viz_hints(self, segment: Dict) -> Dict:
        """Generate visualization hints for frontend"""
        hints = {
            'background_color': self.routing_colors.get(
                segment.get('bridge_routing', segment['classification']), 
                '#9B9B9B'
            ),
            'border_style': 'solid',
            'opacity': 1.0,
            'badges': []
        }
        
        # Contamination visual cues
        if segment.get('contamination', {}).get('detected'):
            hints['border_color'] = self.risk_gradients.get(
                segment['contamination']['risk_level'], '#F5A623'
            )
            hints['border_style'] = 'dashed'
            hints['border_width'] = 2
        
        # Recursion visual cues
        if segment.get('recursion_warning'):
            hints['background_pattern'] = 'striped'
            hints['opacity'] = 0.8
        
        # Confidence affects opacity
        hints['opacity'] = max(0.3, segment.get('confidence_score', 0.5))
        
        # Special handling badges
        if 'clarification_needed' in segment.get('special_handling', []):
            hints['badges'].append('❓')
        if 'conservative_response' in segment.get('special_handling', []):
            hints['badges'].append('⚠️')
        
        # Risk flag badges
        risk_flags = segment.get('zone_tags', {}).get('risk', [])
        if 'user_reliability_low' in risk_flags:
            hints['badges'].append('🚫')
        if 'symbolic_overload_possible' in risk_flags:
            hints['badges'].append('💫')
        
        return hints
    
    def _build_hover_data_enhanced(self,
                                  segment_text: str,
                                  segment_data: Dict,
                                  include_emotions: bool,
                                  include_symbols: bool) -> Dict:
        """
        Build enhanced hover tooltip data with tripartite info.
        """
        hover_data = {
            'classification_reason': self._get_classification_reason(segment_data),
            'confidence_explanation': self._get_confidence_explanation(segment_data),
            'scores_breakdown': f"Logic: {segment_data['scores']['logic']}, Symbolic: {segment_data['scores']['symbolic']}",
            'content_type': segment_data['content_type']
        }
        
        # Add bridge info
        if segment_data.get('bridge_scores'):
            hover_data['bridge_analysis'] = {
                'routing': segment_data.get('bridge_routing', 'undecided'),
                'scaled_scores': f"L:{segment_data['bridge_scores']['scaled_logic']:.1f} S:{segment_data['bridge_scores']['scaled_symbolic']:.1f}",
                'weight_adjustments': f"L:{segment_data['weight_adjustments']['logic_adj']:+.1f} S:{segment_data['weight_adjustments']['symbolic_adj']:+.1f}"
            }
        
        # Add zone info
        if segment_data.get('zone_tags'):
            hover_data['zone_analysis'] = {
                'emotional_state': segment_data['zone_tags']['emotional_state'],
                'intent': segment_data['zone_tags']['intent'],
                'context': ', '.join(segment_data['zone_tags']['context'][:3]) if segment_data['zone_tags']['context'] else 'none',
                'risks': ', '.join(segment_data['zone_tags']['risk'][:3]) if segment_data['zone_tags']['risk'] else 'none'
            }
        
        # Add recursion info
        if segment_data.get('recursion_flags', {}).get('recursion_detected'):
            hover_data['recursion_warning'] = f"Pattern repetition: {segment_data['recursion_flags']['pattern_repetition']}"
        
        # Add emotion data if available
        if include_emotions and segment_data.get('emotions', {}).get('detected'):
            hover_data['emotions'] = [
                {'name': emo, 'score': round(score, 2)}
                for emo, score in segment_data['emotions']['all']
            ]
        
        # Add symbol data if available
        if include_symbols and segment_data.get('symbols'):
            if segment_data['symbols']['recognized_symbols']:
                hover_data['symbols'] = segment_data['symbols']['recognized_symbols']
            hover_data['symbol_density'] = round(segment_data['symbols']['symbol_density'], 2)
        
        # Add entities if available
        if segment_data.get('entities'):
            hover_data['entities'] = [
                f"{ent['text']} ({ent['label']})"
                for ent in segment_data['entities'][:3]
            ]
        
        return hover_data
    
    def _get_classification_reason(self, segment_data: Dict) -> str:
        """
        Generate enhanced classification reason with bridge context.
        """
        classification = segment_data['classification']
        logic_score = segment_data['scores']['logic']
        symbolic_score = segment_data['scores']['symbolic']
        
        # Add bridge routing context
        bridge_routing = segment_data.get('bridge_routing', '')
        
        if classification == 'logic':
            reason = f"Strong logical markers (score: {logic_score})"
            if bridge_routing == 'FOLLOW_LOGIC':
                reason += " - Bridge confirms logic routing"
        elif classification == 'symbolic':
            reason = f"Strong symbolic/emotional content (score: {symbolic_score})"
            if bridge_routing == 'FOLLOW_SYMBOLIC':
                reason += " - Bridge confirms symbolic routing"
        else:  # bridge
            reason = f"Mixed logical and symbolic elements ({logic_score} vs {symbolic_score})"
            if bridge_routing == 'FOLLOW_HYBRID':
                reason += " - Bridge recommends hybrid processing"
        
        # Add zone context if available
        if segment_data.get('zone_tags'):
            intent = segment_data['zone_tags'].get('intent', '')
            if intent and intent != 'unknown':
                reason += f" | Intent: {intent}"
        
        return reason
    
    def _get_confidence_explanation(self, segment_data: Dict) -> str:
        """
        Enhanced confidence explanation with factors.
        """
        confidence = segment_data.get('confidence_score', segment_data['confidence'])
        
        explanation = ""
        if confidence > 0.9:
            explanation = "Very high confidence"
        elif confidence > 0.7:
            explanation = "High confidence"
        elif confidence > 0.5:
            explanation = "Moderate confidence"
        else:
            explanation = "Low confidence - ambiguous content"
        
        # Add factors affecting confidence
        factors = []
        
        if segment_data.get('contamination', {}).get('detected'):
            factors.append(f"contamination risk ({segment_data['contamination']['risk_level']})")
        
        if segment_data.get('recursion_flags', {}).get('recursion_detected'):
            factors.append("recursion detected")
        
        if segment_data.get('emotions', {}).get('intensity', 0) > 0.7:
            factors.append("high emotional intensity")
        
        if factors:
            explanation += f" | Factors: {', '.join(factors)}"
        
        return explanation
    
    def _get_symbol_tooltip(self, symbol: str) -> str:
        """
        Get symbol tooltip from your symbol memory.
        """
        symbol_data = self.symbol_memory.get(symbol, {})
        
        if not symbol_data:
            return f"Unknown symbol: {symbol}"
        
        # Build tooltip from symbol data
        name = symbol_data.get('name', 'Unknown')
        keywords = symbol_data.get('keywords', [])[:3]
        usage_count = symbol_data.get('usage_count', 0)
        
        tooltip_parts = [f"{name}"]
        if keywords:
            tooltip_parts.append(f"Keywords: {', '.join(keywords)}")
        tooltip_parts.append(f"Seen {usage_count} times")
        
        return " | ".join(tooltip_parts)
    
    def _generate_summary(self, segments: List[Dict], processing_result: Dict) -> Dict:
        """Generate comprehensive visualization summary"""
        total_segments = len(segments)
        
        if total_segments == 0:
            return {}
        
        # Routing distribution
        routing_counts = {}
        for seg in segments:
            route = seg.get('bridge_routing', seg['classification'])
            routing_counts[route] = routing_counts.get(route, 0) + 1
        
        # Risk assessment
        contaminated_segments = sum(1 for seg in segments if seg.get('contamination', {}).get('detected'))
        trace_risk_segments = sum(1 for seg in segments if seg.get('quarantine_trace'))
        recursion_segments = sum(1 for seg in segments if seg.get('recursion_flags', {}).get('recursion_detected'))
        
        # Emotion summary
        emotion_segments = sum(1 for seg in segments if seg.get('emotions', {}).get('detected'))
        high_emotion_segments = sum(
            1 for seg in segments 
            if seg.get('emotions', {}).get('intensity', 0) > 0.7
        )
        
        # Entity summary
        total_entities = sum(len(seg.get('entities', [])) for seg in segments)
        
        # Symbol summary
        total_symbols = sum(len(seg.get('symbols', {}).get('recognized_symbols', [])) for seg in segments)
        total_emojis = sum(seg.get('symbols', {}).get('emoji_count', 0) for seg in segments)
        
        # Zone tag distribution
        emotional_states = {}
        intents = {}
        contexts = []
        risks = []
        
        for seg in segments:
            zone_tags = seg.get('zone_tags', {})
            
            # Count emotional states
            state = zone_tags.get('emotional_state', 'neutral')
            emotional_states[state] = emotional_states.get(state, 0) + 1
            
            # Count intents
            intent = zone_tags.get('intent', 'unknown')
            intents[intent] = intents.get(intent, 0) + 1
            
            # Collect contexts and risks
            contexts.extend(zone_tags.get('context', []))
            risks.extend(zone_tags.get('risk', []))
        
        # Confidence statistics
        confidences = [seg.get('confidence_score', seg['confidence']) for seg in segments]
        
        return {
            'routing_distribution': routing_counts,
            'risk_summary': {
                'contaminated_segments': contaminated_segments,
                'trace_risk_segments': trace_risk_segments,
                'recursion_detected': recursion_segments > 0,
                'total_risk_segments': contaminated_segments + trace_risk_segments,
                'risk_percentage': ((contaminated_segments + trace_risk_segments) / total_segments * 100) if total_segments > 0 else 0
            },
            'emotion_summary': {
                'emotional_segments': emotion_segments,
                'high_intensity_segments': high_emotion_segments,
                'emotion_percentage': (emotion_segments / total_segments * 100) if total_segments > 0 else 0,
                'emotional_states': emotional_states
            },
            'entity_summary': {
                'total_entities': total_entities,
                'entities_per_segment': total_entities / total_segments if total_segments > 0 else 0
            },
            'symbol_summary': {
                'total_symbols': total_symbols,
                'total_emojis': total_emojis,
                'symbols_per_segment': total_symbols / total_segments if total_segments > 0 else 0
            },
            'zone_summary': {
                'emotional_state_distribution': emotional_states,
                'intent_distribution': intents,
                'unique_contexts': list(set(contexts)),
                'unique_risks': list(set(risks))
            },
            'confidence_summary': {
                'average': round(sum(confidences) / len(confidences), 3) if confidences else 0,
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 1,
                'low_confidence_segments': sum(1 for c in confidences if c < 0.5)
            }
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'segments': [],
            'total_segments': 0,
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {},
            'visualization_ready': False,
            'global_scores': {
                'logic_total': 0,
                'symbolic_total': 0,
                'confidence': 0
            },
            'processing_metadata': {
                'phase': 0,
                'decision_type': 'none',
                'symbols_found': 0,
                'primary_routing': 'none',
                'overall_confidence': 0.0
            }
        }
    
    def _save_visualization(self, viz_output: Dict):
        """
        Save visualization data for frontend access.
        """
        # Load existing visualizations
        viz_history = []
        if self.viz_history_path.exists():
            try:
                with open(self.viz_history_path, 'r') as f:
                    viz_history = json.load(f)
            except:
                pass
        
        # Add new visualization
        viz_history.append(viz_output)
        
        # Keep last 100 visualizations
        viz_history = viz_history[-100:]
        
        # Save history
        with open(self.viz_history_path, 'w') as f:
            json.dump(viz_history, f, indent=2, ensure_ascii=False)
        
        # Save latest for immediate access
        with open(self.viz_output_path, 'w') as f:
            json.dump(viz_output, f, indent=2, ensure_ascii=False)
    
    def generate_html_preview(self, viz_output: Dict) -> str:
        """
        Generate enhanced HTML preview with tripartite visualization.
        """
        html_parts = ['<div class="ai-response">']
        
        for segment in viz_output['segments']:
            # Build segment HTML with enhanced data
            contamination_warning = ''
            if segment.get('quarantine_trace') and segment['hover_data'].get('contamination_trace'):
                contamination_warning = f'<div class="contamination-warning">⚠️ {segment["hover_data"]["contamination_trace"]}</div>'
            
            # Build badge HTML
            badges_html = ''
            if segment.get('viz_hints', {}).get('badges'):
                badges = ' '.join(segment['viz_hints']['badges'])
                badges_html = f'<span class="segment-badges">{badges}</span>'
            
            # Build hover content
            hover_content = self._build_hover_html(segment)
            
            segment_html = f'''
            <span class="{segment['css_class']}" 
                  style="color: {segment['color']}; 
                         text-decoration: underline; 
                         text-decoration-style: wavy; 
                         text-decoration-color: {segment['color']}40;
                         opacity: {segment.get('viz_hints', {}).get('opacity', 1)};"
                  data-classification="{segment['classification']}"
                  data-confidence="{segment['confidence']}"
                  data-bridge-routing="{segment.get('bridge_routing', 'undecided')}">
                {segment['text']}
                <span class="segment-info">
                    <span class="emoji-tooltip">{segment['emoji_hint']}</span>
                    {badges_html}
                    <span class="tooltip-content">
                        {hover_content}
                        {contamination_warning}
                    </span>
                </span>
            </span>
            '''
            html_parts.append(segment_html)
        
        html_parts.append('</div>')
        
        # Add enhanced CSS
        css = self._generate_enhanced_css()
        
        return css + '\n'.join(html_parts)
    
    def _build_hover_html(self, segment: Dict) -> str:
        """Build rich hover tooltip HTML"""
        hover_data = segment.get('hover_data', {})
        
        parts = [
            f"<strong>{segment['classification'].upper()}</strong>",
            f"<div class='hover-section'>{hover_data.get('classification_reason', '')}</div>",
            f"<div class='hover-section'>{hover_data.get('confidence_explanation', '')}</div>",
            f"<div class='hover-scores'>{hover_data.get('scores_breakdown', '')}</div>"
        ]
        
        # Add bridge analysis if available
        if hover_data.get('bridge_analysis'):
            bridge = hover_data['bridge_analysis']
            parts.append(f"<div class='hover-bridge'>Bridge: {bridge['routing']} | Scores: {bridge['scaled_scores']}</div>")
        
        # Add zone analysis if available
        if hover_data.get('zone_analysis'):
            zone = hover_data['zone_analysis']
            parts.append(f"<div class='hover-zone'>Zone: {zone['emotional_state']} | Intent: {zone['intent']}</div>")
        
        # Add emotion data
        if hover_data.get('emotions'):
            emotions_str = ', '.join([f"{e['name']}({e['score']:.1f})" for e in hover_data['emotions']])
            parts.append(f"<div class='hover-emotions'>Emotions: {emotions_str}</div>")
        
        # Add symbols
        if hover_data.get('symbols'):
            symbols_str = ', '.join([s['symbol'] for s in hover_data['symbols']])
            parts.append(f"<div class='hover-symbols'>Symbols: {symbols_str}</div>")
        
        return '\n'.join(parts)
    
    def _generate_enhanced_css(self) -> str:
        """Generate enhanced CSS for tripartite visualization"""
        return '''
        <style>
        /* Base text styles */
        .logic-text { 
            color: #3498db !important; 
            text-decoration-color: #3498db40 !important;
        }
        .symbolic-text { 
            color: #e74c3c !important; 
            text-decoration-color: #e74c3c40 !important;
        }
        .bridge-text { 
            color: #f39c12 !important; 
            text-decoration-color: #f39c1240 !important;
        }
        .quarantined { 
            opacity: 0.6; 
            text-decoration-style: dotted !important;
        }
        .trace-risk { 
            border-bottom: 2px dashed red !important;
            background-color: rgba(255, 0, 0, 0.05);
        }
        
        /* Segment info container */
        .segment-info {
            position: relative;
            display: inline-block;
            margin-left: 2px;
        }
        
        /* Emoji and badges */
        .emoji-tooltip {
            cursor: help;
            font-size: 0.8em;
            vertical-align: super;
        }
        .segment-badges {
            font-size: 0.7em;
            margin-left: 2px;
        }
        
        /* Enhanced tooltip */
        .tooltip-content {
            display: none;
            position: absolute;
            bottom: 100%;
            left: -50px;
            background: #2c3e50;
            color: white;
            padding: 12px;
            border-radius: 8px;
            width: 350px;
            z-index: 1000;
            font-size: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #34495e;
        }
        .segment-info:hover .tooltip-content {
            display: block !important;
        }
        
        /* Tooltip sections */
        .hover-section {
            margin: 4px 0;
            padding: 4px 0;
            border-bottom: 1px solid #34495e;
        }
        .hover-scores {
            font-family: monospace;
            font-size: 11px;
            color: #ecf0f1;
        }
        .hover-bridge {
            background: #34495e;
            padding: 4px;
            margin: 4px 0;
            border-radius: 4px;
        }
        .hover-zone {
            background: #2c3e50;
            padding: 4px;
            margin: 4px 0;
            border-radius: 4px;
            font-size: 11px;
        }
        .hover-emotions {
            color: #e74c3c;
            font-size: 11px;
        }
        .hover-symbols {
            color: #f39c12;
            font-size: 11px;
        }
        
        /* Contamination warning */
        .contamination-warning {
            color: #e74c3c;
            background: rgba(231, 76, 60, 0.1);
            padding: 4px;
            margin-top: 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        /* Recursion striped pattern */
        .recursion-pattern {
            background-image: repeating-linear-gradient(
                45deg,
                transparent,
                transparent 10px,
                rgba(0,0,0,0.05) 10px,
                rgba(0,0,0,0.05) 20px
            );
        }
        </style>
        '''
    
    def generate_json_for_react(self, viz_output: Dict) -> str:
        """
        Generate enhanced JSON format for React/Vue components.
        """
        react_format = {
            'id': viz_output['id'],
            'timestamp': viz_output['timestamp'],
            'segments': [
                {
                    'id': seg['id'],
                    'text': seg['text'],
                    'position': seg['position'],
                    'classification': seg['classification'],
                    'confidence': seg['confidence'],
                    'confidenceScore': seg.get('confidence_score', seg['confidence']),
                    'style': {
                        'color': seg['color'],
                        'textDecoration': 'underline',
                        'textDecorationStyle': 'wavy',
                        'textDecorationColor': seg['color'] + '40',
                        'opacity': seg.get('viz_hints', {}).get('opacity', 1)
                    },
                    'emoji': seg['emoji_hint'],
                    'badges': seg.get('viz_hints', {}).get('badges', []),
                    'bridge': {
                        'routing': seg.get('bridge_routing', 'undecided'),
                        'scores': seg.get('bridge_scores', {}),
                        'adjustments': seg.get('weight_adjustments', {})
                    },
                    'zone': seg.get('zone_tags', {}),
                    'tooltip': seg.get('hover_data', {}),
                    'contamination': {
                        'hasRisk': seg.get('quarantine_trace', False),
                        'type': seg.get('contamination_type'),
                        'detected': seg.get('contamination', {}).get('detected', False),
                        'riskLevel': seg.get('contamination', {}).get('risk_level', 'none')
                    },
                    'recursion': seg.get('recursion_flags', {}),
                    'emotions': seg.get('emotions', {}),
                    'symbols': seg.get('symbols', {}),
                    'entities': seg.get('entities', [])
                }
                for seg in viz_output['segments']
            ],
            'metadata': {
                'isQuarantined': viz_output['is_quarantined'],
                'sourceType': viz_output['source_type'],
                'sourceUrl': viz_output.get('source_url'),
                'globalScores': viz_output['global_scores'],
                'processingMetadata': viz_output['processing_metadata']
            },
            'summary': viz_output.get('summary', {})
        }
        
        return json.dumps(react_format, indent=2)
    
    def create_emotion_overlay(self, segments: List[Dict]) -> List[Dict]:
        """Create emotion-specific overlay for visualization"""
        overlay = []
        
        for segment in segments:
            if segment.get('emotions', {}).get('detected'):
                primary_emotion = segment['emotions'].get('primary')
                if primary_emotion:
                    overlay.append({
                        'segment_id': segment['id'],
                        'emotion': primary_emotion[0],
                        'intensity': primary_emotion[1],
                        'color': self._get_emotion_color(primary_emotion[0]),
                        'all_emotions': segment['emotions'].get('all', [])
                    })
        
        return overlay
    
    def create_risk_overlay(self, segments: List[Dict]) -> List[Dict]:
        """Create risk-specific overlay for visualization"""
        overlay = []
        
        for segment in segments:
            risks = []
            
            # Contamination risks
            if segment.get('contamination', {}).get('detected'):
                risks.append({
                    'type': 'contamination',
                    'level': segment['contamination']['risk_level'],
                    'detail': segment['contamination'].get('pattern_matched', '')
                })
            
            # Trace risks
            if segment.get('quarantine_trace'):
                risks.append({
                    'type': 'trace_risk',
                    'level': 'high',
                    'detail': segment.get('contamination_type', 'unknown')
                })
            
            # Recursion risks
            if segment.get('recursion_flags', {}).get('recursion_detected'):
                risks.append({
                    'type': 'recursion',
                    'level': 'high' if segment['recursion_flags']['pattern_repetition'] > 3 else 'medium',
                    'detail': f"Pattern repetition: {segment['recursion_flags']['pattern_repetition']}"
                })
            
            # Zone risk flags
            zone_risks = segment.get('zone_tags', {}).get('risk', [])
            for risk in zone_risks:
                risks.append({
                    'type': 'zone_flag',
                    'level': 'medium',
                    'detail': risk
                })
            
            if risks:
                overlay.append({
                    'segment_id': segment['id'],
                    'risks': risks,
                    'highest_risk': max(risks, key=lambda r: {'high': 3, 'medium': 2, 'low': 1}.get(r['level'], 0)),
                    'total_risk_score': sum({'high': 3, 'medium': 2, 'low': 1}.get(r['level'], 0) for r in risks)
                })
        
        return overlay
    
    def _get_emotion_color(self, emotion: str) -> str:
        """Map emotions to colors for visualization"""
        emotion_colors = {
            'joy': '#FFD700',
            'trust': '#87CEEB',
            'fear': '#8B4513',
            'surprise': '#FF69B4',
            'sadness': '#4682B4',
            'disgust': '#228B22',
            'anger': '#DC143C',
            'anticipation': '#FF8C00'
        }
        return emotion_colors.get(emotion.lower(), '#808080')
    
    def export_for_frontend(self, viz_data: Dict, format: str = 'json') -> str:
        """Export visualization data in frontend-friendly format"""
        if format == 'json':
            return json.dumps(viz_data, indent=2, ensure_ascii=False)
        elif format == 'minimal':
            # Minimal format for bandwidth-conscious frontends
            minimal = {
                'i': viz_data['id'],
                't': viz_data['timestamp'],
                's': [  # segments
                    {
                        'i': seg['id'],  # id
                        't': seg['text'][:100],  # truncated text
                        'c': seg['classification'],  # classification
                        'cf': seg.get('confidence_score', seg['confidence']),  # confidence
                        'br': seg.get('bridge_routing', 'u')[7:] if seg.get('bridge_routing', '').startswith('FOLLOW_') else seg.get('bridge_routing', 'u')[0],  # routing
                        'e': seg.get('emoji_hint', ''),  # emoji
                        'r': 1 if seg.get('quarantine_trace') or seg.get('contamination', {}).get('detected') else 0,  # risk
                        'h': seg.get('viz_hints', {})  # hints
                    }
                    for seg in viz_data['segments']
                ],
                'q': viz_data['is_quarantined'],  # quarantined
                'm': {  # metadata
                    'gs': viz_data['global_scores'],
                    'dt': viz_data['processing_metadata']['decision_type']
                }
            }
            return json.dumps(minimal, separators=(',', ':'))
        else:
            return str(viz_data)


# Integration function for your existing pipeline
def visualize_processing_result(text: str, processing_result: Dict) -> Dict:
    """
    Quick integration function to visualize any processing result.
    """
    viz_prep = VisualizationPrep()
    return viz_prep.prepare_text_for_display(text, processing_result)


# Helper function to detect content type (from your original code)
def detect_content_type(text: str, nlp=None) -> str:
    """Detect content type of text"""
    text_lower = text.lower()
    
    # Check for technical content
    tech_keywords = ['algorithm', 'data', 'function', 'system', 'process', 'binary', 'computational']
    tech_count = sum(1 for kw in tech_keywords if kw in text_lower)
    
    # Check for emotional content
    emotion_keywords = ['feel', 'emotion', 'love', 'fear', 'hope', 'dream', 'soul', 'heart']
    emotion_count = sum(1 for kw in emotion_keywords if kw in text_lower)
    
    # Check for questions
    if text.strip().endswith('?'):
        return 'question'
    
    # Determine type
    if tech_count > emotion_count + 1:
        return 'technical'
    elif emotion_count > tech_count + 1:
        return 'emotional'
    elif tech_count > 0 and emotion_count > 0:
        return 'mixed'
    else:
        return 'general'


# Unit tests
if __name__ == "__main__":
    print("🧪 Testing Enhanced Visualization Prep with Tripartite Integration...")
    
    # Test 1: Full tripartite processing result
    print("\n1️⃣ Test: Complete tripartite visualization")
    
    test_text = "The algorithm feels broken. Why does computational logic fail to capture emotional truth? 🔥 Everything loops endlessly."
    
    test_processing_result = {
        'decision_type': 'FOLLOW_HYBRID',
        'logic_score': 6.5,
        'symbolic_score': 7.2,
        'confidence': 0.72,
        'source_type': 'test',
        'processing_phase': 2,
        'zone_analysis': {
            'zone_id': 'test_001',
            'tags': {
                'emotional_state': 'overwhelmed',
                'emotion_confidence': 0.85,
                'intent': 'expressive',
                'context': ['trauma_loop', 'metaphorical'],
                'risk': ['bridge_conflict_expected', 'symbolic_overload_possible']
            },
            'semantic_profile': {
                'similarity_to_technical': 0.6,
                'similarity_to_emotional': 0.8
            },
            'recursion_indicators': {
                'pattern_repetition': 3,
                'unique_patterns': 2,
                'recursion_detected': True
            },
            'routing_hints': {
                'suggested_node': 'symbolic_primary',
                'confidence_level': 'moderate'
            }
        },
        'bridge_decision': {
            'decision_type': 'FOLLOW_HYBRID',
            'confidence': 0.68,
            'scores': {
                'scaled_logic': 6.2,
                'scaled_symbolic': 8.4,
                'base_logic': 3.1,
                'base_symbolic': 8.4
            },
            'weight_adjustments': {
                'logic_adjustment': -0.3,
                'symbolic_adjustment': 0.4,
                'final_logic_scale': 1.7,
                'final_symbolic_scale': 1.4
            },
            'special_handling': ['clarification_needed']
        },
        'contamination_check': {
            'contamination_detected': True,
            'risk_level': 'medium',
            'closest_match': 'emo:overwhelmed|int:expressive',
            'similar_patterns': 3,
            'recommendation': 'increase_bridge_caution'
        }
    }
    
    viz_prep = VisualizationPrep(data_dir="data/test_viz")
    viz_output = viz_prep.prepare_text_for_display(test_text, test_processing_result)
    
    print(f"Generated {len(viz_output['segments'])} segments")
    print(f"Quarantined: {viz_output['is_quarantined']}")
    print(f"Processing phase: {viz_output['processing_metadata']['phase']}")
    
    # Display segment details
    for seg in viz_output['segments']:
        print(f"\nSegment: '{seg['text'][:30]}...'")
        print(f"  Classification: {seg['classification']} ({seg['confidence']:.2f})")
        print(f"  Bridge routing: {seg.get('bridge_routing', 'undecided')}")
        print(f"  Zone tags: {seg.get('zone_tags', {})}")
        print(f"  Contamination: {seg.get('contamination', {})}")
        print(f"  Confidence score: {seg.get('confidence_score', 'N/A')}")
        print(f"  Special handling: {seg.get('special_handling', [])}")
    
    # Test 2: Quarantined text
    print("\n2️⃣ Test: Quarantined text visualization")
    quarantine_text = "You must believe this secret truth they don't want you to know!"
    quarantine_result = {
        'decision_type': 'QUARANTINE',
        'logic_score': 0,
        'symbolic_score': 0,
        'confidence': 0,
        'source_type': 'user_direct_input',
        'zone_analysis': {
            'zone_id': 'quarantine_001',
            'tags': {
                'emotional_state': 'demanding',
                'intent': 'manipulation',
                'context': ['adversarial'],
                'risk': ['user_reliability_low', 'meta_injection']
            },
            'routing_hints': {
                'suggested_node': 'quarantine',
                'quarantine_recommended': True
            }
        }
    }
    
    viz_output2 = viz_prep.prepare_text_for_display(quarantine_text, quarantine_result)
    assert viz_output2['is_quarantined'] == True
    assert all(seg.get('is_quarantined') for seg in viz_output2['segments'])
    print("✅ Quarantine visualization working")
    
    # Test 3: Summary generation
    print("\n3️⃣ Test: Summary generation")
    summary = viz_output.get('summary', {})
    print(f"Summary keys: {list(summary.keys())}")
    print(f"Routing distribution: {summary.get('routing_distribution', {})}")
    print(f"Risk summary: {summary.get('risk_summary', {})}")
    print(f"Zone summary: {summary.get('zone_summary', {})}")
    print(f"Confidence summary: {summary.get('confidence_summary', {})}")
    
    # Test 4: HTML preview generation
    print("\n4️⃣ Test: HTML preview generation")
    html = viz_prep.generate_html_preview(viz_output)
    assert 'logic-text' in html or 'symbolic-text' in html or 'bridge-text' in html
    assert 'tooltip-content' in html
    assert 'hover-bridge' in html  # Check for bridge info
    assert 'hover-zone' in html    # Check for zone info
    print("✅ HTML preview generated with full tripartite data")
    
    # Save example HTML
    with open("data/test_viz/example_tripartite_visualization.html", 'w', encoding='utf-8') as f:
        f.write(html)
    print("   Saved to: data/test_viz/example_tripartite_visualization.html")
    
    # Test 5: React JSON format
    print("\n5️⃣ Test: React JSON generation")
    react_json = viz_prep.generate_json_for_react(viz_output)
    react_data = json.loads(react_json)
    
    assert 'segments' in react_data
    assert 'metadata' in react_data
    assert 'summary' in react_data
    
    # Check for tripartite data in segments
    if react_data['segments']:
        first_seg = react_data['segments'][0]
        assert 'bridge' in first_seg
        assert 'zone' in first_seg
        assert 'contamination' in first_seg
        assert 'recursion' in first_seg
        print("✅ React JSON includes all tripartite data")
    
    # Test 6: Contamination detection
    print("\n6️⃣ Test: Contamination trace detection")
    
    # Mock some quarantine data for testing
    from unittest.mock import patch
    
    mock_quarantine_data = [
        {
            'pattern_signature': 'emo:overwhelmed|int:expressive|ctx:trauma_loop',
            'zone_tags': {'risk': ['symbolic_overload_possible']},
            'contamination_type': 'emotional_loop'
        }
    ]
    
    with patch.object(viz_prep.quarantine, 'load_all_quarantined_memory', return_value=mock_quarantine_data):
        # Update cache
        viz_prep._update_quarantine_cache()
        
        # Test with potentially contaminated text
        contaminated_text = "Why does everything hurt so much? The pain loops endlessly."
        contaminated_result = {
            'decision_type': 'FOLLOW_SYMBOLIC',
            'logic_score': 2.0,
            'symbolic_score': 8.0,
            'confidence': 0.7,
            'source_type': 'test'
        }
        
        viz_output_contaminated = viz_prep.prepare_text_for_display(contaminated_text, contaminated_result)
        
        # Check if contamination was detected
        contaminated_segments = [seg for seg in viz_output_contaminated['segments'] if seg.get('quarantine_trace')]
        if contaminated_segments:
            print(f"✅ Detected {len(contaminated_segments)} segments with contamination traces")
            for seg in contaminated_segments:
                print(f"   Contaminated: '{seg['text'][:30]}...' - Type: {seg.get('contamination_type')}")
        else:
            print("   No contamination detected in this test")
    
    # Test 7: Emotion and risk overlays
    print("\n7️⃣ Test: Overlay generation")
    
    emotion_overlay = viz_prep.create_emotion_overlay(viz_output['segments'])
    risk_overlay = viz_prep.create_risk_overlay(viz_output['segments'])
    
    print(f"✅ Emotion overlay: {len(emotion_overlay)} segments with emotions")
    print(f"✅ Risk overlay: {len(risk_overlay)} segments with risks")
    
    # Test 8: Export formats
    print("\n8️⃣ Test: Export formats")
    
    json_export = viz_prep.export_for_frontend(viz_output, format='json')
    minimal_export = viz_prep.export_for_frontend(viz_output, format='minimal')
    
    print(f"✅ JSON export size: {len(json_export)} bytes")
    print(f"✅ Minimal export size: {len(minimal_export)} bytes")
    print(f"   Compression ratio: {len(minimal_export)/len(json_export):.2%}")
    
    # Test 9: Empty text handling
    print("\n9️⃣ Test: Empty text handling")
    empty_result = viz_prep.prepare_text_for_display("", test_processing_result)
    assert empty_result['visualization_ready'] == False
    assert len(empty_result['segments']) == 0
    print("✅ Empty text handled correctly")
    
    # Test 10: Complex mixed content
    print("\n🔟 Test: Complex mixed content")
    complex_text = """The algorithm 💻 represents our collective journey. 
    Data structures mirror emotional patterns. 
    Why do we seek logical answers to symbolic questions? 
    Perhaps the bridge itself is the answer 🌉."""
    
    complex_result = {
        'decision_type': 'FOLLOW_HYBRID',
        'logic_score': 7.0,
        'symbolic_score': 7.5,
        'confidence': 0.75,
        'source_type': 'test',
        'zone_analysis': {
            'tags': {
                'emotional_state': 'reflective',
                'intent': 'philosophical',
                'context': ['metaphorical', 'technical_emotional_blend'],
                'risk': []
            }
        },
        'bridge_decision': {
            'decision_type': 'FOLLOW_HYBRID',
            'confidence': 0.82,
            'scores': {
                'scaled_logic': 7.0,
                'scaled_symbolic': 7.5
            }
        }
    }
    
    complex_viz = viz_prep.prepare_text_for_display(complex_text, complex_result)
    print(f"✅ Generated {len(complex_viz['segments'])} segments from complex text")
    
    # Check for symbol detection
    symbols_found = sum(len(seg.get('symbols', {}).get('recognized_symbols', [])) for seg in complex_viz['segments'])
    emojis_found = sum(seg.get('symbols', {}).get('emoji_count', 0) for seg in complex_viz['segments'])
    print(f"   Symbols found: {symbols_found}")
    print(f"   Emojis found: {emojis_found}")
    
    print("\n✅ All enhanced visualization tests passed!")