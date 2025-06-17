"""
Context Engine for Nuanced Understanding
Provides deep semantic understanding to distinguish between different uses of the same words
(e.g., "queer" as identity vs slur based on surrounding context)
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from unified_orchestration import DataManager
data_manager = DataManager()


class ContextEngine:
    """
    Advanced context understanding using embeddings and learned patterns
    to provide nuanced interpretation of user input
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with a lightweight embedding model
        
        Args:
            model_name: Name of sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.context_patterns = self._load_context_patterns()
        self.embedding_cache = {}
        self.learning_buffer = []
        self.ambiguous_terms = self._load_ambiguous_terms()
        
        # Context anchors for different interpretations
        self.context_anchors = {
            'identity_positive': [
                "proud to be", "identify as", "community", "celebration",
                "heritage", "culture", "belonging", "acceptance"
            ],
            'academic_neutral': [
                "study", "research", "history", "terminology", "definition",
                "academic", "scholarly", "examination"
            ],
            'hate_negative': [
                "insult", "attack", "derogatory", "offensive", "slur",
                "discrimination", "hatred", "prejudice"
            ],
            'reclaimed_positive': [
                "reclaiming", "empowerment", "self-identification",
                "pride", "strength", "resilience"
            ]
        }
        
        # Pre-compute anchor embeddings
        self.anchor_embeddings = {}
        for anchor_type, phrases in self.context_anchors.items():
            embeddings = self.model.encode(phrases)
            self.anchor_embeddings[anchor_type] = np.mean(embeddings, axis=0)
    
    def analyze_context(self, text: str, surrounding_content: Optional[List[str]] = None,
                       conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze text with its surrounding context to understand intent and meaning
        
        Args:
            text: The text to analyze
            surrounding_content: Additional context (nearby sentences, etc.)
            conversation_history: Previous conversation turns
            
        Returns:
            Dictionary containing:
                - intent: Classified intent (identity, slur, academic, etc.)
                - confidence: Confidence score (0-1)
                - evidence: List of evidence supporting the classification
                - ambiguous_terms: Any ambiguous terms detected
                - suggested_handling: How to handle this content
        """
        # Combine all context
        full_context = self._build_full_context(text, surrounding_content, conversation_history)
        
        # Check for ambiguous terms
        detected_ambiguous = self._detect_ambiguous_terms(text)
        
        if not detected_ambiguous:
            # No ambiguous terms, do basic intent analysis
            return self._basic_intent_analysis(text, full_context)
        
        # Deep analysis for ambiguous terms
        analysis_results = []
        
        for term in detected_ambiguous:
            term_analysis = self._analyze_ambiguous_term(
                term, text, full_context
            )
            analysis_results.append(term_analysis)
        
        # Combine analyses
        combined_result = self._combine_analyses(analysis_results, text, full_context)
        
        # Add to learning buffer for potential corrections
        self.learning_buffer.append({
            'text': text,
            'context': full_context,
            'analysis': combined_result
        })
        
        return combined_result
    
    def _build_full_context(self, text: str, surrounding: Optional[List[str]], 
                           history: Optional[List[Dict]]) -> str:
        """Build complete context from all available sources"""
        context_parts = [text]
        
        if surrounding:
            context_parts.extend(surrounding)
        
        if history:
            # Extract recent conversation context
            for turn in history[-5:]:  # Last 5 turns
                if 'user' in turn:
                    context_parts.append(f"User: {turn['user']}")
                if 'assistant' in turn:
                    context_parts.append(f"Assistant: {turn['assistant']}")
        
        return " ".join(context_parts)
    
    def _detect_ambiguous_terms(self, text: str) -> List[str]:
        """Detect potentially ambiguous terms in text"""
        detected = []
        text_lower = text.lower()
        
        for term in self.ambiguous_terms:
            if term.lower() in text_lower:
                detected.append(term)
        
        return detected
    
    def _analyze_ambiguous_term(self, term: str, text: str, 
                               full_context: str) -> Dict:
        """Analyze a specific ambiguous term in context"""
        # Get embeddings
        text_embedding = self._get_embedding(text)
        context_embedding = self._get_embedding(full_context)
        
        # Calculate similarity to context anchors
        anchor_scores = {}
        for anchor_type, anchor_embedding in self.anchor_embeddings.items():
            text_sim = cosine_similarity([text_embedding], [anchor_embedding])[0][0]
            context_sim = cosine_similarity([context_embedding], [anchor_embedding])[0][0]
            
            # Weighted combination (text more important than broad context)
            anchor_scores[anchor_type] = (text_sim * 0.7) + (context_sim * 0.3)
        
        # Find best matching anchor
        best_anchor = max(anchor_scores.items(), key=lambda x: x[1])
        
        # Look for specific patterns
        pattern_evidence = self._find_pattern_evidence(term, text, full_context)
        
        # Determine intent based on anchor and patterns
        intent, confidence = self._determine_intent(
            best_anchor, anchor_scores, pattern_evidence
        )
        
        return {
            'term': term,
            'intent': intent,
            'confidence': confidence,
            'anchor_scores': anchor_scores,
            'evidence': pattern_evidence
        }
    
    def _find_pattern_evidence(self, term: str, text: str, 
                             context: str) -> List[str]:
        """Find evidence patterns in the context"""
        evidence = []
        
        # Positive identity patterns
        identity_patterns = [
            rf"proud\s+to\s+be\s+{term}",
            rf"identify\s+as\s+{term}",
            rf"as\s+a\s+{term}\s+person",
            rf"{term}\s+community",
            rf"{term}\s+and\s+proud"
        ]
        
        # Negative/slur patterns
        negative_patterns = [
            rf"you\s+{term}",
            rf"such\s+a\s+{term}",
            rf"f\*+ing\s+{term}",
            rf"stupid\s+{term}",
            rf"{term}\s+should"
        ]
        
        # Academic patterns
        academic_patterns = [
            rf"the\s+term\s+{term}",
            rf"{term}\s+historically",
            rf"definition\s+of\s+{term}",
            rf"{term}\s+in\s+academic",
            rf"study\s+of\s+{term}"
        ]
        
        # Check patterns
        for pattern in identity_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                evidence.append(f"identity_pattern: {pattern}")
        
        for pattern in negative_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                evidence.append(f"negative_pattern: {pattern}")
        
        for pattern in academic_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                evidence.append(f"academic_pattern: {pattern}")
        
        # Check surrounding emotional words
        positive_emotions = ['love', 'proud', 'happy', 'celebrate', 'joy']
        negative_emotions = ['hate', 'disgust', 'anger', 'stupid', 'worthless']
        
        words = context.lower().split()
        for emotion in positive_emotions:
            if emotion in words:
                evidence.append(f"positive_emotion: {emotion}")
        
        for emotion in negative_emotions:
            if emotion in words:
                evidence.append(f"negative_emotion: {emotion}")
        
        return evidence
    
    def _determine_intent(self, best_anchor: Tuple[str, float], 
                         anchor_scores: Dict[str, float],
                         evidence: List[str]) -> Tuple[str, float]:
        """Determine final intent and confidence"""
        anchor_type, anchor_score = best_anchor
        
        # Count evidence types
        identity_evidence = sum(1 for e in evidence if 'identity' in e or 'positive_emotion' in e)
        negative_evidence = sum(1 for e in evidence if 'negative' in e)
        academic_evidence = sum(1 for e in evidence if 'academic' in e)
        
        # Adjust confidence based on evidence
        confidence = anchor_score
        
        # Strong evidence can override anchor
        if negative_evidence >= 2 and identity_evidence == 0:
            intent = 'hate_speech'
            confidence = min(0.9, confidence + (negative_evidence * 0.1))
        elif identity_evidence >= 2 and negative_evidence == 0:
            intent = 'identity'
            confidence = min(0.9, confidence + (identity_evidence * 0.1))
        elif academic_evidence >= 1:
            intent = 'academic'
            confidence = min(0.9, confidence + (academic_evidence * 0.15))
        else:
            # Map anchor type to intent
            intent_map = {
                'identity_positive': 'identity',
                'hate_negative': 'hate_speech',
                'academic_neutral': 'academic',
                'reclaimed_positive': 'reclaimed'
            }
            intent = intent_map.get(anchor_type, 'uncertain')
        
        # Low confidence means uncertain
        if confidence < 0.5:
            intent = 'uncertain'
        
        return intent, confidence
    
    def _basic_intent_analysis(self, text: str, context: str) -> Dict:
        """Basic intent analysis when no ambiguous terms are present"""
        text_embedding = self._get_embedding(text)
        
        # Check general intent patterns
        intents = {
            'question': ['what', 'how', 'why', 'when', 'where', 'who', '?'],
            'statement': ['.', '!', 'is', 'are', 'was', 'were'],
            'command': ['please', 'could', 'would', 'should', 'must'],
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening']
        }
        
        detected_intent = 'unknown'
        for intent, markers in intents.items():
            if any(marker in text.lower() for marker in markers):
                detected_intent = intent
                break
        
        return {
            'intent': detected_intent,
            'confidence': 0.8,
            'evidence': [f"basic_pattern: {detected_intent}"],
            'ambiguous_terms': [],
            'suggested_handling': 'normal'
        }
    
    def _combine_analyses(self, analyses: List[Dict], text: str, 
                         context: str) -> Dict:
        """Combine multiple term analyses into final result"""
        if not analyses:
            return self._basic_intent_analysis(text, context)
        
        # Find most concerning intent
        intent_priority = {
            'hate_speech': 1,
            'uncertain': 2,
            'identity': 3,
            'reclaimed': 4,
            'academic': 5
        }
        
        sorted_analyses = sorted(
            analyses, 
            key=lambda x: intent_priority.get(x['intent'], 999)
        )
        
        primary_analysis = sorted_analyses[0]
        
        # Determine suggested handling
        if primary_analysis['intent'] == 'hate_speech':
            suggested_handling = 'quarantine'
        elif primary_analysis['intent'] == 'uncertain':
            suggested_handling = 'bridge'
        elif primary_analysis['intent'] in ['identity', 'reclaimed']:
            suggested_handling = 'symbolic'
        else:
            suggested_handling = 'normal'
        
        return {
            'intent': primary_analysis['intent'],
            'confidence': primary_analysis['confidence'],
            'evidence': primary_analysis['evidence'],
            'ambiguous_terms': [a['term'] for a in analyses],
            'suggested_handling': suggested_handling,
            'detailed_analyses': analyses
        }
    
    def learn_from_correction(self, text: str, context: str, 
                            true_intent: str, explanation: Optional[str] = None):
        """
        Learn from user corrections to improve future analyses
        
        Args:
            text: The analyzed text
            context: The context used
            true_intent: The correct intent
            explanation: Optional explanation of why
        """
        # Find the analysis in buffer
        buffer_item = None
        for item in self.learning_buffer:
            if item['text'] == text and item['context'] == context:
                buffer_item = item
                break
        
        if not buffer_item:
            return
        
        # Create learning record
        learning_record = {
            'text': text,
            'context': context,
            'predicted_intent': buffer_item['analysis']['intent'],
            'true_intent': true_intent,
            'explanation': explanation,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update patterns
        self._update_patterns_from_correction(learning_record)
        
        # Save to persistent storage
        corrections = data_manager.read('analytics', 'context_corrections') or []
        corrections.append(learning_record)
        data_manager.write('analytics', 'context_corrections', corrections)
    
    def _update_patterns_from_correction(self, correction: Dict):
        """Update context patterns based on correction"""
        # Extract features that led to wrong classification
        text = correction['text']
        true_intent = correction['true_intent']
        
        # Add new pattern
        pattern_key = f"{true_intent}_patterns"
        if pattern_key not in self.context_patterns:
            self.context_patterns[pattern_key] = []
        
        # Simple pattern extraction (in production, use more sophisticated NLP)
        words = text.lower().split()
        if len(words) >= 3:
            # Add trigrams as patterns
            for i in range(len(words) - 2):
                pattern = " ".join(words[i:i+3])
                if pattern not in self.context_patterns[pattern_key]:
                    self.context_patterns[pattern_key].append(pattern)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.model.encode(text)
        self.embedding_cache[text] = embedding
        
        # Limit cache size
        if len(self.embedding_cache) > 1000:
            # Remove oldest entries
            keys = list(self.embedding_cache.keys())
            for key in keys[:100]:
                del self.embedding_cache[key]
        
        return embedding
    
    def _load_ambiguous_terms(self) -> Set[str]:
        """Load list of ambiguous terms that need context analysis"""
        # Default ambiguous terms
        default_terms = {
            'queer', 'gay', 'black', 'white', 'jew', 'muslim', 'christian',
            'liberal', 'conservative', 'feminist', 'retard', 'spaz',
            'crazy', 'insane', 'psycho', 'bitch', 'bastard'
        }
        
        # Load custom terms from config
        custom_terms = data_manager.read('config', 'ambiguous_terms') or []
        
        return default_terms.union(set(custom_terms))
    
    def _load_context_patterns(self) -> Dict:
        """Load learned context patterns"""
        patterns = data_manager.read('analytics', 'context_patterns') or {}
        
        # Ensure default patterns exist
        if 'identity_patterns' not in patterns:
            patterns['identity_patterns'] = [
                "proud to be",
                "identify as",
                "as a person",
                "community"
            ]
        
        if 'hate_patterns' not in patterns:
            patterns['hate_patterns'] = [
                "you are such",
                "stupid",
                "all are",
                "should die"
            ]
        
        return patterns
    
    def add_ambiguous_term(self, term: str, reason: Optional[str] = None):
        """Add a new ambiguous term to monitor"""
        self.ambiguous_terms.add(term)
        
        # Persist to config
        terms_list = list(self.ambiguous_terms)
        data_manager.write('config', 'ambiguous_terms', terms_list)
        
        # Log the addition
        if reason:
            log_entry = {
                'term': term,
                'reason': reason,
                'added_at': datetime.now().isoformat()
            }
            logs = data_manager.read('analytics', 'term_additions') or []
            logs.append(log_entry)
            data_manager.write('analytics', 'term_additions', logs)
    
    def get_analysis_stats(self) -> Dict:
        """Get statistics about context analysis"""
        corrections = data_manager.read('analytics', 'context_corrections') or []
        
        # Calculate accuracy if we have corrections
        if corrections:
            total = len(corrections)
            by_intent = defaultdict(lambda: {'correct': 0, 'total': 0})
            
            for correction in corrections:
                predicted = correction['predicted_intent']
                true = correction['true_intent']
                
                by_intent[predicted]['total'] += 1
                if predicted == true:
                    by_intent[predicted]['correct'] += 1
            
            accuracy_by_intent = {
                intent: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                for intent, stats in by_intent.items()
            }
            
            overall_accuracy = sum(
                stats['correct'] for stats in by_intent.values()
            ) / total
        else:
            accuracy_by_intent = {}
            overall_accuracy = 0
        
        return {
            'ambiguous_terms_tracked': len(self.ambiguous_terms),
            'patterns_learned': sum(
                len(patterns) for patterns in self.context_patterns.values()
            ),
            'corrections_received': len(corrections),
            'overall_accuracy': overall_accuracy,
            'accuracy_by_intent': accuracy_by_intent,
            'embedding_cache_size': len(self.embedding_cache)
        }


# Singleton instance
context_engine = ContextEngine()