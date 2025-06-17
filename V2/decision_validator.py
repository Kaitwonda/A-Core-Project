"""
Decision Validation and Feedback Loop System
Tracks decisions, outcomes, and learns from success/failure patterns
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from unified_orchestration import DataManager
data_manager = DataManager()


class DecisionValidator:
    """
    Implements feedback loops to learn from past decisions
    and improve future routing between logic/symbolic/bridge
    """
    
    def __init__(self):
        self.decision_cache = {}
        try:
            self.pattern_learner = PatternLearner()
            self.confidence_calculator = ConfidenceCalculator()
        except Exception as e:
            print(f"Warning: Error initializing pattern learning components: {e}")
            self.pattern_learner = None
            self.confidence_calculator = None
        
        # Subscribe to decision updates
        try:
            data_manager.subscribe('analytics', 'decisions', self._on_decision_update)
        except Exception as e:
            print(f"Warning: Could not subscribe to decision updates: {e}")
    
    def record_decision(self, context: Dict, choice: str, 
                       reasoning: Optional[Dict] = None) -> str:
        """
        Record a decision made by the system
        
        Args:
            context: The context in which decision was made
            choice: The choice made (logic/symbolic/bridge)
            reasoning: Optional reasoning for the decision
            
        Returns:
            Decision ID for tracking
        """
        decision_id = str(uuid.uuid4())
        
        decision_record = {
            'id': decision_id,
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'choice': choice,
            'reasoning': reasoning or {},
            'status': 'pending',
            'outcome': None,
            'feedback_score': None
        }
        
        # Store in cache for quick access
        self.decision_cache[decision_id] = decision_record
        
        # Persist to data manager
        decisions = data_manager.read('analytics', 'decisions') or []
        decisions.append(decision_record)
        data_manager.write('analytics', 'decisions', decisions)
        
        return decision_id
    
    def record_outcome(self, decision_id: str, success_metrics: Dict) -> bool:
        """
        Record the outcome of a decision
        
        Args:
            decision_id: The decision to update
            success_metrics: Metrics indicating success/failure
                - processing_time: How long it took
                - confidence_achieved: Final confidence score
                - user_satisfaction: Optional user feedback
                - error_occurred: Whether errors happened
                - required_migration: Whether item needed to move
                
        Returns:
            Success status
        """
        if decision_id not in self.decision_cache:
            # Load from disk if not in cache
            decisions = data_manager.read('analytics', 'decisions') or []
            for decision in decisions:
                if decision['id'] == decision_id:
                    self.decision_cache[decision_id] = decision
                    break
            else:
                return False
        
        decision = self.decision_cache[decision_id]
        
        # Calculate feedback score
        feedback_score = self._calculate_feedback_score(
            decision['choice'], 
            success_metrics
        )
        
        # Update decision record
        decision['status'] = 'completed'
        decision['outcome'] = success_metrics
        decision['feedback_score'] = feedback_score
        decision['completed_at'] = datetime.now().isoformat()
        
        # Learn from this outcome
        if hasattr(self, 'pattern_learner') and self.pattern_learner is not None:
            self.pattern_learner.learn(decision['context'], decision['choice'], feedback_score)
        
        # Update persistent storage
        data_manager.update(
            'analytics', 'decisions',
            decision,
            query={'id': decision_id}
        )
        
        return True
    
    def get_decision_confidence(self, context: Dict) -> Dict[str, float]:
        """
        Get confidence scores for each possible choice given context
        
        Args:
            context: The current context
            
        Returns:
            Dictionary of confidence scores for each choice
        """
        # Get historical decisions with similar context
        similar_decisions = self._find_similar_decisions(context)
        
        if not similar_decisions:
            # No history, analyze context to decide between logic/symbolic preference
            semantic_tags = context.get('semantic_tags', [])
            
            # Detect logic-favorable content
            logic_indicators = ['technical', 'information', 'question', 'analysis', 'data', 'fact']
            logic_score = sum(1 for tag in semantic_tags if any(indicator in tag.lower() for indicator in logic_indicators))
            
            # Detect symbolic-favorable content  
            symbolic_indicators = ['emotional', 'expressive', 'metaphorical', 'poetic', 'feeling', 'abstract']
            symbolic_score = sum(1 for tag in semantic_tags if any(indicator in tag.lower() for indicator in symbolic_indicators))
            
            # Determine preference based on content
            if logic_score > symbolic_score:
                return {'logic': 0.5, 'symbolic': 0.3, 'bridge': 0.2}
            elif symbolic_score > logic_score:
                return {'logic': 0.3, 'symbolic': 0.5, 'bridge': 0.2}
            else:
                # Balanced or neutral content
                return {'logic': 0.4, 'symbolic': 0.4, 'bridge': 0.2}
        
        # Calculate confidence based on past outcomes
        patterns = {}
        if hasattr(self, 'pattern_learner') and self.pattern_learner is not None:
            patterns = self.pattern_learner.get_patterns()
        
        confidences = self.confidence_calculator.calculate(
            context, 
            similar_decisions,
            patterns
        )
        
        return confidences
    
    def get_routing_recommendation(self, context: Dict) -> Tuple[str, float]:
        """
        Get recommended routing based on past decisions
        
        Args:
            context: The current context
            
        Returns:
            Tuple of (recommended_choice, confidence)
        """
        confidences = self.get_decision_confidence(context)
        
        # Find highest confidence choice
        best_choice = max(confidences.items(), key=lambda x: x[1])
        
        # Check if confidence is high enough (lowered threshold to allow more direct routing)
        if best_choice[1] < 0.35:
            # Very low confidence, recommend bridge
            return 'bridge', best_choice[1]
        
        return best_choice
    
    def _calculate_feedback_score(self, choice: str, metrics: Dict) -> float:
        """
        Calculate feedback score from success metrics
        
        Score ranges from 0 (complete failure) to 1 (perfect success)
        """
        score = 0.5  # Start neutral
        
        # Processing time factor (faster is better)
        if 'processing_time' in metrics:
            time_score = max(0, 1 - (metrics['processing_time'] / 10.0))
            score += time_score * 0.2
        
        # Confidence achievement (higher is better)
        if 'confidence_achieved' in metrics:
            score += metrics['confidence_achieved'] * 0.3
        
        # User satisfaction (if available)
        if 'user_satisfaction' in metrics:
            score += metrics['user_satisfaction'] * 0.3
        
        # Error penalty
        if metrics.get('error_occurred', False):
            score -= 0.3
        
        # Migration penalty (suggests wrong initial choice)
        if metrics.get('required_migration', False):
            score -= 0.2
        
        # Choice-specific adjustments
        if choice == 'bridge' and not metrics.get('required_migration', False):
            # Staying in bridge when not needed is suboptimal
            score -= 0.1
        elif choice != 'bridge' and metrics.get('confidence_achieved', 0) < 0.7:
            # Low confidence in final brain suggests should have used bridge
            score -= 0.15
        
        return max(0, min(1, score))
    
    def _find_similar_decisions(self, context: Dict, limit: int = 20) -> List[Dict]:
        """Find past decisions with similar context"""
        all_decisions = data_manager.read('analytics', 'decisions') or []
        
        # Filter completed decisions (handle legacy records without status)
        completed = [d for d in all_decisions if d.get('status', 'completed') == 'completed']
        
        if not completed:
            return []
        
        # Calculate similarity scores
        similarities = []
        for decision in completed:
            # Handle legacy decisions without context
            decision_context = decision.get('context', {})
            similarity = self._calculate_context_similarity(context, decision_context)
            similarities.append((similarity, decision))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [decision for _, decision in similarities[:limit]]
    
    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between two contexts"""
        similarity = 0.0
        total_weight = 0.0
        
        # Compare semantic tags
        if 'semantic_tags' in context1 and 'semantic_tags' in context2:
            tags1 = set(context1['semantic_tags'])
            tags2 = set(context2['semantic_tags'])
            if tags1 or tags2:
                jaccard = len(tags1 & tags2) / len(tags1 | tags2)
                similarity += jaccard * 0.4
                total_weight += 0.4
        
        # Compare emotional context
        if 'emotions' in context1 and 'emotions' in context2:
            emotions1 = set(context1['emotions'])
            emotions2 = set(context2['emotions'])
            if emotions1 or emotions2:
                jaccard = len(emotions1 & emotions2) / len(emotions1 | emotions2)
                similarity += jaccard * 0.3
                total_weight += 0.3
        
        # Compare content type
        if 'content_type' in context1 and 'content_type' in context2:
            if context1['content_type'] == context2['content_type']:
                similarity += 0.2
            total_weight += 0.2
        
        # Compare complexity
        if 'complexity' in context1 and 'complexity' in context2:
            diff = abs(context1['complexity'] - context2['complexity'])
            complexity_sim = max(0, 1 - diff)
            similarity += complexity_sim * 0.1
            total_weight += 0.1
        
        return similarity / total_weight if total_weight > 0 else 0
    
    def _on_decision_update(self, decisions: List[Dict]):
        """Handle decision updates from data manager"""
        # Update cache with any new decisions
        for decision in decisions:
            if decision['id'] not in self.decision_cache:
                self.decision_cache[decision['id']] = decision
    
    def get_analytics(self) -> Dict:
        """Get analytics on decision performance"""
        decisions = data_manager.read('analytics', 'decisions') or []
        completed = [d for d in decisions if d.get('status', 'completed') == 'completed']
        
        if not completed:
            return {
                'total_decisions': 0,
                'average_feedback_score': 0,
                'choice_distribution': {},
                'success_by_choice': {}
            }
        
        # Calculate metrics
        total = len(completed)
        scores = [d.get('feedback_score', 0.5) for d in completed]
        avg_score = sum(scores) / total if total > 0 else 0
        
        # Choice distribution
        choice_counts = defaultdict(int)
        choice_scores = defaultdict(list)
        
        for decision in completed:
            choice = decision.get('choice', 'bridge')
            choice_counts[choice] += 1
            choice_scores[choice].append(decision.get('feedback_score', 0.5))
        
        choice_distribution = {
            choice: count / total 
            for choice, count in choice_counts.items()
        }
        
        success_by_choice = {
            choice: sum(scores) / len(scores) 
            for choice, scores in choice_scores.items()
        }
        
        # Safety check for pattern_learner
        patterns_learned = 0
        if hasattr(self, 'pattern_learner') and self.pattern_learner is not None:
            if hasattr(self.pattern_learner, 'patterns'):
                patterns_learned = len(self.pattern_learner.patterns)
        
        return {
            'total_decisions': total,
            'average_feedback_score': avg_score,
            'choice_distribution': choice_distribution,
            'success_by_choice': success_by_choice,
            'patterns_learned': patterns_learned
        }


class PatternLearner:
    """Learns patterns from decision outcomes"""
    
    def __init__(self):
        self.patterns = defaultdict(lambda: {
            'occurrences': 0,
            'total_score': 0,
            'avg_score': 0
        })
    
    def learn(self, context: Dict, choice: str, score: float):
        """Learn from a decision outcome"""
        # Extract key features from context
        features = self._extract_features(context)
        
        for feature in features:
            pattern_key = f"{feature}:{choice}"
            pattern = self.patterns[pattern_key]
            
            pattern['occurrences'] += 1
            pattern['total_score'] += score
            pattern['avg_score'] = pattern['total_score'] / pattern['occurrences']
    
    def _extract_features(self, context: Dict) -> List[str]:
        """Extract learnable features from context"""
        features = []
        
        # Semantic tag combinations
        if 'semantic_tags' in context:
            tags = context['semantic_tags']
            features.extend([f"tag:{tag}" for tag in tags])
            
            # Tag pairs
            for i, tag1 in enumerate(tags):
                for tag2 in tags[i+1:]:
                    features.append(f"tag_pair:{tag1}+{tag2}")
        
        # Emotion features
        if 'emotions' in context:
            features.extend([f"emotion:{emotion}" for emotion in context['emotions']])
        
        # Content type
        if 'content_type' in context:
            features.append(f"type:{context['content_type']}")
        
        # Complexity level
        if 'complexity' in context:
            complexity_level = 'high' if context['complexity'] > 0.7 else 'medium' if context['complexity'] > 0.3 else 'low'
            features.append(f"complexity:{complexity_level}")
        
        return features
    
    def get_patterns(self) -> Dict:
        """Get learned patterns"""
        return dict(self.patterns)


class ConfidenceCalculator:
    """Calculates confidence scores based on historical data and patterns"""
    
    def calculate(self, context: Dict, similar_decisions: List[Dict], 
                  patterns: Dict) -> Dict[str, float]:
        """
        Calculate confidence for each choice
        """
        # Base confidence from similar decisions
        base_confidences = self._calculate_base_confidence(similar_decisions)
        
        # Adjust based on learned patterns
        pattern_adjustments = self._calculate_pattern_adjustments(context, patterns)
        
        # Combine base and pattern-based confidences
        final_confidences = {}
        for choice in ['logic', 'symbolic', 'bridge']:
            base = base_confidences.get(choice, 0.33)
            adjustment = pattern_adjustments.get(choice, 0)
            
            # Weighted combination
            final = (base * 0.6) + (adjustment * 0.4)
            final_confidences[choice] = max(0, min(1, final))
        
        # Normalize to sum to 1
        total = sum(final_confidences.values())
        if total > 0:
            for choice in final_confidences:
                final_confidences[choice] /= total
        
        return final_confidences
    
    def _calculate_base_confidence(self, decisions: List[Dict]) -> Dict[str, float]:
        """Calculate base confidence from similar decisions"""
        if not decisions:
            return {'logic': 0.4, 'symbolic': 0.4, 'bridge': 0.2}
        
        # Weight recent decisions more heavily
        now = datetime.now()
        weighted_scores = defaultdict(list)
        
        for decision in decisions:
            # Calculate age weight (newer = higher weight)
            try:
                decision_time = datetime.fromisoformat(decision.get('timestamp', now.isoformat()))
                age_days = (now - decision_time).days
                age_weight = max(0.1, 1.0 - (age_days / 30.0))
            except (ValueError, KeyError):
                age_weight = 0.5  # Default weight for invalid timestamps
            
            choice = decision.get('choice', 'bridge')
            score = decision.get('feedback_score', 0.5)
            
            weighted_scores[choice].append(score * age_weight)
        
        # Calculate weighted averages
        confidences = {}
        default_scores = {'logic': 0.4, 'symbolic': 0.4, 'bridge': 0.2}
        for choice in ['logic', 'symbolic', 'bridge']:
            scores = weighted_scores.get(choice, [])
            if scores:
                confidences[choice] = sum(scores) / len(scores)
            else:
                confidences[choice] = default_scores[choice]
        
        return confidences
    
    def _calculate_pattern_adjustments(self, context: Dict, 
                                     patterns: Dict) -> Dict[str, float]:
        """Calculate confidence adjustments based on patterns"""
        adjustments = defaultdict(float)
        pattern_counts = defaultdict(int)
        
        # Extract features from current context
        # Create a temporary PatternLearner just for feature extraction
        temp_learner = PatternLearner()
        features = temp_learner._extract_features(context) if hasattr(temp_learner, '_extract_features') else []
        
        # Look up pattern scores
        for feature in features:
            for choice in ['logic', 'symbolic', 'bridge']:
                pattern_key = f"{feature}:{choice}"
                if pattern_key in patterns:
                    pattern = patterns[pattern_key]
                    if pattern['occurrences'] >= 3:  # Minimum occurrences
                        adjustments[choice] += pattern['avg_score']
                        pattern_counts[choice] += 1
        
        # Average adjustments
        for choice in adjustments:
            if pattern_counts[choice] > 0:
                adjustments[choice] /= pattern_counts[choice]
        
        return dict(adjustments)


# Singleton instance
try:
    decision_validator = DecisionValidator()
except Exception as e:
    print(f"Error creating DecisionValidator singleton: {e}")
    # Create a dummy instance that won't crash
    class DummyDecisionValidator:
        def __init__(self):
            self.decision_cache = {}
            self.pattern_learner = None
            self.confidence_calculator = None
        
        def get_analytics(self):
            return {
                'total_decisions': 0,
                'average_feedback_score': 0,
                'choice_distribution': {},
                'success_by_choice': {},
                'patterns_learned': 0
            }
        
        def record_decision(self, *args, **kwargs):
            return str(uuid.uuid4())
        
        def record_outcome(self, *args, **kwargs):
            return False
        
        def get_decision_confidence(self, *args, **kwargs):
            return {'logic': 0.4, 'symbolic': 0.4, 'bridge': 0.2}
        
        def get_routing_recommendation(self, *args, **kwargs):
            # Return logic as highest confidence choice to break hybrid dominance
            return 'logic', 0.4
    
    decision_validator = DummyDecisionValidator()