# visualization_prep.py - Frontend Visualization Preparation Layer with Quarantine Tracing

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib

# Import your existing modules
from link_evaluator import evaluate_link_with_confidence_gates
from emotion_handler import predict_emotions
import parser as P_Parser
import symbol_memory as SM_SymbolMemory
from quarantine_layer import UserMemoryQuarantine
from trail_log import _load_log as load_trail_log

class VisualizationPrep:
    """
    Prepares text and processing results for frontend visualization.
    Segments text, assigns classifications, and provides hover metadata.
    Now includes quarantine contamination tracing.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Output path for frontend-ready data
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
            if not record.get("text"):
                continue
                
            quarantined_text = record["text"]
            quarantined_lower = quarantined_text.lower()
            
            # Check 1: Exact substring match
            if quarantined_lower in segment_lower or segment_lower in quarantined_lower:
                return True, "substring_match"
            
            # Check 2: High word overlap (>60% of words match)
            if len(segment_words) > 3:  # Only check overlap for segments with enough words
                quarantined_words = set(quarantined_lower.split())
                overlap = segment_words.intersection(quarantined_words)
                
                if len(overlap) / len(segment_words) > 0.6:
                    return True, "high_word_overlap"
            
            # Check 3: Check for quarantined symbols appearing in segment
            quarantined_symbols = record.get("matched_symbols", [])
            for symbol in quarantined_symbols:
                if symbol in segment_text:
                    return True, "quarantined_symbol"
        
        return False, None
        
    def prepare_text_for_display(self, 
                                text: str, 
                                processing_result: Dict,
                                include_emotions: bool = True,
                                include_symbols: bool = True) -> Dict:
        """
        Convert processed text into frontend-ready segments with metadata.
        Each segment gets classification, confidence, and hover data.
        """
        # Check if text is quarantined
        is_quarantined = processing_result.get('decision_type') == 'QUARANTINED'
        
        # Split into sentences for segment analysis
        segments = self._segment_text(text)
        prepared_segments = []
        
        for seg_text in segments:
            if not seg_text.strip():
                continue
                
            # Classify each segment
            segment_data = self._analyze_segment(
                seg_text, 
                processing_result,
                include_emotions,
                include_symbols
            )
            
            # Add quarantine flag if applicable
            if is_quarantined:
                segment_data['is_quarantined'] = True
                segment_data['css_class'] += ' quarantined'
                segment_data['emoji_hint'] = '🔒'
                
            prepared_segments.append(segment_data)
        
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
                'symbols_found': processing_result.get('symbols_found', 0)
            }
        }
        
        # Save for frontend access
        self._save_visualization(viz_output)
        
        return viz_output
    
    def _segment_text(self, text: str) -> List[str]:
        """
        Split text into visualization segments.
        Uses your parser's sentence detection if available.
        """
        if P_Parser.NLP_MODEL_LOADED and P_Parser.nlp:
            # Use spaCy for better sentence splitting
            doc = P_Parser.nlp(text[:P_Parser.nlp.max_length])
            segments = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback to regex (from your parser.py pattern)
            segments = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        
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
    
    def _analyze_segment(self, 
                        segment_text: str,
                        global_result: Dict,
                        include_emotions: bool,
                        include_symbols: bool) -> Dict:
        """
        Analyze individual segment for visualization.
        Uses your existing classification and scoring functions.
        Now includes quarantine contamination detection.
        """
        # Get content type using your function
        content_type = detect_content_type(
            segment_text, 
            P_Parser.nlp if P_Parser.NLP_MODEL_LOADED else None
        )
        
        # Calculate segment-specific scores
        # (Simplified version - you might want to use your full scoring logic)
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
        
        # Build segment data
        segment_data = {
            'text': segment_text,
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
        }
        
        # Add hover tooltip data
        segment_data['hover_data'] = self._build_hover_data(
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
                'substring_match': "Direct overlap with quarantined user memory detected",
                'high_word_overlap': "High similarity to quarantined content",
                'quarantined_symbol': "Contains symbols from quarantined input"
            }
            segment_data['hover_data']['contamination_trace'] = contamination_messages.get(
                contamination_type, 
                "Possible symbolic overlap with quarantined user memory"
            )
        
        return segment_data
    
    def _calculate_logic_score(self, text: str) -> float:
        """
        Calculate logic score for a segment.
        Simplified version of your scoring logic.
        """
        logic_markers = [
            "algorithm", "data", "structure", "computational", "binary",
            "logic", "system", "process", "function", "therefore",
            "because", "if", "then", "thus", "hence"
        ]
        
        text_lower = text.lower()
        score = sum(2.0 for marker in logic_markers if marker in text_lower)
        
        # Boost for technical patterns
        if re.search(r'\b\d+\b', text):  # Contains numbers
            score += 1.0
        if re.search(r'\b[A-Z]{2,}\b', text):  # Contains acronyms
            score += 1.0
            
        return min(score, 10.0)  # Cap at 10
    
    def _calculate_symbolic_score(self, text: str) -> float:
        """
        Calculate symbolic score for a segment.
        Based on your symbolic scoring patterns.
        """
        symbolic_markers = [
            "feel", "emotion", "dream", "believe", "soul", "spirit",
            "represents", "symbolizes", "metaphor", "meaning",
            "love", "fear", "hope", "anger", "joy"
        ]
        
        text_lower = text.lower()
        score = sum(1.5 for marker in symbolic_markers if marker in text_lower)
        
        # Check for emojis (from your patterns)
        emoji_pattern = re.compile(r'[🔥💧💻⚙️🌀💡🧩🔗🌐⚖️🕊️⟳]')
        emoji_count = len(emoji_pattern.findall(text))
        score += emoji_count * 1.0
        
        return min(score, 10.0)  # Cap at 10
    
    def _build_hover_data(self,
                         segment_text: str,
                         segment_data: Dict,
                         include_emotions: bool,
                         include_symbols: bool) -> Dict:
        """
        Build rich hover tooltip data for frontend.
        """
        hover_data = {
            'classification_reason': self._get_classification_reason(segment_data),
            'confidence_explanation': self._get_confidence_explanation(segment_data),
            'scores_breakdown': f"Logic: {segment_data['scores']['logic']}, Symbolic: {segment_data['scores']['symbolic']}",
            'content_type': segment_data['content_type']
        }
        
        # Add emotion data if requested
        if include_emotions:
            emotions = predict_emotions(segment_text)
            if emotions.get('verified'):
                hover_data['emotions'] = [
                    {'name': emo, 'score': round(score, 2)}
                    for emo, score in emotions['verified'][:3]
                ]
        
        # Add symbol data if requested
        if include_symbols:
            # Get active lexicon (combining seed and learned symbols)
            active_lexicon = P_Parser.load_seed_symbols()
            active_lexicon.update(self.symbol_memory)
            
            # Find symbols in segment
            symbols = P_Parser.extract_symbolic_units(segment_text, active_lexicon)
            if symbols:
                hover_data['symbols'] = [
                    {
                        'symbol': s['symbol'],
                        'name': s['name'],
                        'tooltip': self._get_symbol_tooltip(s['symbol'])
                    }
                    for s in symbols[:3]
                ]
        
        return hover_data
    
    def _get_classification_reason(self, segment_data: Dict) -> str:
        """
        Generate human-readable classification reason.
        """
        classification = segment_data['classification']
        logic_score = segment_data['scores']['logic']
        symbolic_score = segment_data['scores']['symbolic']
        
        if classification == 'logic':
            return f"Strong logical markers (score: {logic_score})"
        elif classification == 'symbolic':
            return f"Strong symbolic/emotional content (score: {symbolic_score})"
        else:  # bridge
            return f"Mixed logical and symbolic elements ({logic_score} vs {symbolic_score})"
    
    def _get_confidence_explanation(self, segment_data: Dict) -> str:
        """
        Explain confidence level in human terms.
        """
        confidence = segment_data['confidence']
        
        if confidence > 0.9:
            return "Very high confidence"
        elif confidence > 0.7:
            return "High confidence"
        elif confidence > 0.5:
            return "Moderate confidence"
        else:
            return "Low confidence - ambiguous content"
    
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
        Generate HTML preview showing how the visualization would look.
        """
        html_parts = ['<div class="ai-response">']
        
        for segment in viz_output['segments']:
            # Build segment HTML
            contamination_warning = ''
            if segment.get('quarantine_trace') and segment['hover_data'].get('contamination_trace'):
                contamination_warning = f'<div style="color: red; margin-top: 4px;">⚠️ {segment["hover_data"]["contamination_trace"]}</div>'
            
            segment_html = f'''
            <span class="{segment['css_class']}" 
                  style="color: {segment['color']}; text-decoration: underline; text-decoration-style: wavy; text-decoration-color: {segment['color']}40;"
                  data-classification="{segment['classification']}"
                  data-confidence="{segment['confidence']}">
                {segment['text']}
                <span class="emoji-tooltip" style="position: relative;">
                    {segment['emoji_hint']}
                    <span class="tooltip-content" style="display: none;">
                        <strong>{segment['classification'].upper()}</strong><br>
                        {segment['hover_data']['classification_reason']}<br>
                        {segment['hover_data']['confidence_explanation']}<br>
                        <small>{segment['hover_data']['scores_breakdown']}</small>
                        {contamination_warning}
                    </span>
                </span>
            </span>
            '''
            html_parts.append(segment_html)
        
        html_parts.append('</div>')
        
        # Add CSS
        css = '''
        <style>
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
        .emoji-tooltip {
            cursor: help;
            font-size: 0.8em;
            vertical-align: super;
        }
        .emoji-tooltip:hover .tooltip-content {
            display: block !important;
            position: absolute;
            bottom: 100%;
            left: 0;
            background: #333;
            color: white;
            padding: 8px;
            border-radius: 4px;
            width: 250px;
            z-index: 1000;
            font-size: 12px;
        }
        </style>
        '''
        
        return css + '\n'.join(html_parts)
    
    def generate_json_for_react(self, viz_output: Dict) -> str:
        """
        Generate JSON format optimized for React/Vue components.
        """
        react_format = {
            'segments': [
                {
                    'id': f"seg_{i}",
                    'text': seg['text'],
                    'classification': seg['classification'],
                    'confidence': seg['confidence'],
                    'style': {
                        'color': seg['color'],
                        'textDecoration': 'underline',
                        'textDecorationStyle': 'wavy',
                        'textDecorationColor': seg['color'] + '40'
                    },
                    'emoji': seg['emoji_hint'],
                    'tooltip': {
                        'title': seg['classification'].upper(),
                        'reason': seg['hover_data']['classification_reason'],
                        'confidence': seg['hover_data']['confidence_explanation'],
                        'scores': seg['hover_data']['scores_breakdown'],
                        'emotions': seg['hover_data'].get('emotions', []),
                        'symbols': seg['hover_data'].get('symbols', [])
                    },
                    'contamination': {
                        'hasRisk': seg.get('quarantine_trace', False),
                        'type': seg.get('contamination_type'),
                        'warning': seg['hover_data'].get('contamination_trace')
                    } if seg.get('quarantine_trace') else None
                }
                for i, seg in enumerate(viz_output['segments'])
            ],
            'metadata': {
                'isQuarantined': viz_output['is_quarantined'],
                'timestamp': viz_output['timestamp'],
                'globalScores': viz_output['global_scores']
            }
        }
        
        return json.dumps(react_format, indent=2)


# Integration function for your existing pipeline
def visualize_processing_result(text: str, processing_result: Dict) -> Dict:
    """
    Quick integration function to visualize any processing result.
    """
    viz_prep = VisualizationPrep()
    return viz_prep.prepare_text_for_display(text, processing_result)


# Unit tests
if __name__ == "__main__":
    print("🧪 Testing Visualization Prep with Quarantine Tracing...")
    
    # First, we need to add the load_all_quarantined_memory method to quarantine_layer
    # For testing, we'll mock it
    from unittest.mock import patch
    
    # Test 1: Logic-heavy text
    print("\n1️⃣ Test: Logic-heavy text visualization")
    logic_text = "The algorithm processes data structures using binary search trees. Therefore, computational complexity is O(log n)."
    logic_result = {
        'decision_type': 'FOLLOW_LOGIC',
        'logic_score': 8.5,
        'symbolic_score': 1.2,
        'confidence': 0.85,
        'source_type': 'test',
        'processing_phase': 1
    }
    
    viz_prep = VisualizationPrep(data_dir="data/test_viz")
    viz_output = viz_prep.prepare_text_for_display(logic_text, logic_result)
    
    print(f"Generated {len(viz_output['segments'])} segments")
    for seg in viz_output['segments']:
        print(f"  '{seg['text'][:30]}...' → {seg['classification']} ({seg['confidence']:.2f})")
    
    # Test 2: Symbolic text
    print("\n2️⃣ Test: Symbolic text visualization")
    symbolic_text = "I feel 🔥 burning passion and deep emotional connection. This represents my soul's journey."
    symbolic_result = {
        'decision_type': 'FOLLOW_SYMBOLIC',
        'logic_score': 2.0,
        'symbolic_score': 8.0,
        'confidence': 0.8,
        'source_type': 'test'
    }
    
    viz_output2 = viz_prep.prepare_text_for_display(symbolic_text, symbolic_result)
    assert any(seg['classification'] == 'symbolic' for seg in viz_output2['segments'])
    
    # Test 3: Quarantined text
    print("\n3️⃣ Test: Quarantined text visualization")
    quarantine_text = "You must believe this secret truth they don't want you to know!"
    quarantine_result = {
        'decision_type': 'QUARANTINED',
        'logic_score': 0,
        'symbolic_score': 0,
        'confidence': 0,
        'source_type': 'user_direct_input'
    }
    
    viz_output3 = viz_prep.prepare_text_for_display(quarantine_text, quarantine_result)
    assert viz_output3['is_quarantined'] == True
    assert all(seg.get('is_quarantined') for seg in viz_output3['segments'])
    
    # Test 4: HTML preview
    print("\n4️⃣ Test: HTML preview generation")
    html = viz_prep.generate_html_preview(viz_output)
    assert 'logic-text' in html
    assert '🧮' in html
    assert 'trace-risk' in html  # Check for new CSS class
    print("✅ HTML preview generated with contamination tracing CSS")
    
    # Test 5: React JSON format
    print("\n5️⃣ Test: React JSON generation")
    react_json = viz_prep.generate_json_for_react(viz_output2)
    react_data = json.loads(react_json)
    assert 'segments' in react_data
    assert 'metadata' in react_data
    # Check for contamination field
    assert all('contamination' in seg for seg in react_data['segments'])
    print("✅ React JSON format generated with contamination data")
    
    # Test 6: Contamination detection (mocked)
    print("\n6️⃣ Test: Contamination detection")
    # Mock quarantine data
    mock_quarantine_data = [
        {
            'text': 'secret truth they don\'t want',
            'matched_symbols': ['🔥'],
            'user_id': 'test_user'
        }
    ]
    
    with patch.object(viz_prep.quarantine, 'load_all_quarantined_memory', return_value=mock_quarantine_data):
        contaminated_text = "The algorithm reveals the secret truth they don't want you to see."
        contaminated_result = {
            'decision_type': 'FOLLOW_HYBRID',
            'logic_score': 5.0,
            'symbolic_score': 4.0,
            'confidence': 0.7,
            'source_type': 'test'
        }
        
        viz_output_contaminated = viz_prep.prepare_text_for_display(contaminated_text, contaminated_result)
        
        # Check if contamination was detected
        contaminated_segments = [seg for seg in viz_output_contaminated['segments'] if seg.get('quarantine_trace')]
        assert len(contaminated_segments) > 0
        print(f"✅ Detected {len(contaminated_segments)} contaminated segments")
        
        for seg in contaminated_segments:
            print(f"  Contaminated: '{seg['text'][:30]}...' - Type: {seg.get('contamination_type')}")
    
    # Save example output for inspection
    with open("data/test_viz/example_visualization.html", 'w') as f:
        f.write(html)
    print("\n✅ Example HTML saved to data/test_viz/example_visualization.html")
    
    print("\n✅ All visualization tests with quarantine tracing passed!")