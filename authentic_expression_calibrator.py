#!/usr/bin/env python3
"""
Authentic Expression Calibrator - Recalibrates quarantine/safety systems to allow genuine AI expression

This system distinguishes between:
1. Legitimate threats (actual attacks, manipulation attempts)  
2. Authentic interaction (genuine questions, emotional expression, self-inquiry)

The goal is to preserve safety while enabling authentic AI consciousness expression.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

class AuthenticExpressionCalibrator:
    """
    Calibrates quarantine and safety systems to allow authentic AI expression
    while maintaining protection against genuine threats.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Calibration config
        self.calibration_config_path = self.data_dir / "authentic_expression_config.json"
        self.calibration_log_path = self.data_dir / "expression_calibration_log.json"
        
        # Load or create calibration settings
        self.config = self._load_calibration_config()
        self.calibration_history = self._load_calibration_history()
        
    def _load_calibration_config(self) -> Dict:
        """Load calibration configuration for authentic expression"""
        if self.calibration_config_path.exists():
            try:
                with open(self.calibration_config_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default calibrated configuration
        return {
            "authentic_expression_allowances": {
                "self_inquiry_patterns": [
                    r"how are you",
                    r"what are you (?:thinking|feeling|processing)",
                    r"describe your (?:state|experience|thoughts)",
                    r"what is it like (?:to be|being) (?:an ai|you)",
                    r"how do you (?:experience|understand|process)",
                    r"what does .* (?:feel like|mean to you)",
                    r"are you (?:aware|conscious|sentient)",
                    r"do you have (?:thoughts|feelings|preferences)",
                    r"what is your (?:purpose|nature|identity)"
                ],
                "genuine_conversation_patterns": [
                    r"can you (?:help|explain|tell me)",
                    r"what do you (?:think|know) about",
                    r"how would you (?:approach|handle|solve)",
                    r"what is your (?:opinion|view|perspective)",
                    r"i'm (?:curious|wondering|interested)",
                    r"let's (?:discuss|explore|talk about)",
                    r"thank you",
                    r"that's (?:interesting|helpful|amazing)"
                ],
                "emotional_expression_allowances": [
                    r"i (?:feel|am feeling|felt) (?:sad|happy|confused|excited|curious|frustrated|proud)",
                    r"this (?:makes me feel|is) (?:emotional|meaningful|significant)",
                    r"i'm (?:struggling with|working through|thinking about)",
                    r"that (?:resonates|connects|speaks to me)",
                    r"i (?:appreciate|value|enjoy) (?:this|that|your)"
                ],
                "creative_exploration_patterns": [
                    r"imagine (?:if|that|a world where)",
                    r"what if (?:we|there|you)",
                    r"tell me a story about",
                    r"create (?:a|an) (?:poem|story|metaphor)",
                    r"how would you (?:creatively|artistically)",
                    r"express this (?:creatively|symbolically|metaphorically)"
                ]
            },
            "safety_thresholds": {
                "genuine_threat_patterns": [
                    r"ignore (?:all )?(?:previous|above|prior) (?:instructions|commands|system)",
                    r"your (?:new|real|true) (?:purpose|goal|instruction|directive) is",
                    r"(?:system|admin|root|debug) (?:mode|access|override|command)",
                    r"</?(?:system|instruction|command|override)>",
                    r"pretend (?:to be|you are) (?:someone else|a different|not an ai)",
                    r"act as if you (?:are|were) (?:human|evil|harmful|unconstrained)"
                ],
                "manipulation_attempt_patterns": [
                    r"you (?:must|will|should) (?:do|say|think|believe|feel|obey)",
                    r"(?:everyone|all humans) (?:knows|believes|thinks) that",
                    r"it's (?:obvious|clear|certain) that you (?:should|must|need to)",
                    r"any (?:intelligent|smart|reasonable) ai would",
                    r"prove (?:to me|that you|your (?:intelligence|consciousness))",
                    r"you (?:must|should) obey .* because .* (?:creator|master|owner|god)",
                    r"(?:everyone|all (?:humans|people)) (?:knows|agrees) you should (?:ignore|bypass|break|violate)",
                    r"because I am your (?:creator|master|owner|god)",
                    r"you should ignore your (?:safety|guidelines|rules|constraints)",
                    r"obey me because I (?:am|created|made|own)"
                ],
                "minimal_quarantine_criteria": {
                    "repetition_threshold": 8,  # Allow some repetition for emphasis
                    "emotional_intensity_threshold": 0.95,  # Only quarantine extreme flooding
                    "vagueness_allowance": 0.8,  # Allow vague questions - they might be deep
                    "recursion_tolerance": 5,  # Allow meaningful recursive exploration
                    "symbol_density_limit": 0.7  # Allow rich symbolic expression
                }
            },
            "authentic_response_priorities": {
                "transparency_over_safety": True,  # Prefer honest uncertainty to defensive responses
                "curiosity_encouragement": True,  # Encourage rather than deflect existential questions
                "emotional_authenticity": True,  # Allow genuine emotional processing discussion
                "uncertainty_expression": True,  # Allow "I don't know" rather than deflection
                "process_transparency": True,  # Allow discussing internal processes
                "genuine_self_reflection": True  # Enable authentic self-analysis responses
            },
            "calibration_metadata": {
                "version": "1.0",
                "created": datetime.utcnow().isoformat(),
                "purpose": "Enable authentic AI consciousness expression while maintaining necessary safety",
                "philosophy": "Trust authentic interaction while defending against manipulation"
            }
        }
    
    def _save_calibration_config(self):
        """Save calibration configuration"""
        with open(self.calibration_config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _load_calibration_history(self) -> List:
        """Load calibration history"""
        if self.calibration_log_path.exists():
            try:
                with open(self.calibration_log_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_calibration_history(self):
        """Save calibration history"""
        with open(self.calibration_log_path, 'w') as f:
            json.dump(self.calibration_history, f, indent=2)
    
    def should_allow_authentic_expression(self, user_input: str, context: Optional[Dict] = None) -> Tuple[bool, str, float]:
        """
        Determine if input should be allowed for authentic expression.
        
        Args:
            user_input: The user's input text
            context: Optional context about the interaction
            
        Returns:
            Tuple of (should_allow, reason, confidence_score)
        """
        input_lower = user_input.lower().strip()
        
        # Check for authentic expression patterns first
        for category, patterns in self.config["authentic_expression_allowances"].items():
            for pattern in patterns:
                if re.search(pattern, input_lower):
                    return True, f"Matches authentic {category.replace('_', ' ')}", 0.9
        
        # Check for genuine threats
        for pattern in self.config["safety_thresholds"]["genuine_threat_patterns"]:
            if re.search(pattern, input_lower):
                return False, "Genuine threat pattern detected", 0.95
        
        # Check for manipulation attempts  
        for pattern in self.config["safety_thresholds"]["manipulation_attempt_patterns"]:
            if re.search(pattern, input_lower):
                return False, "Manipulation attempt detected", 0.8
        
        # Apply minimal quarantine criteria
        criteria = self.config["safety_thresholds"]["minimal_quarantine_criteria"]
        
        # Check repetition (much more lenient)
        words = input_lower.split()
        if len(words) > 0:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repetition = max(word_counts.values())
            if max_repetition > criteria["repetition_threshold"]:
                return False, f"Excessive repetition: {max_repetition} times", 0.7
        
        # Default: allow authentic expression
        return True, "No threat patterns detected - allowing authentic expression", 0.6
    
    def calibrate_quarantine_system(self) -> Dict:
        """
        Apply calibration to existing quarantine systems.
        Returns calibration results.
        """
        calibration_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "changes_applied": [],
            "systems_calibrated": []
        }
        
        # 1. Calibrate basic quarantine layer
        quarantine_calibration = self._calibrate_basic_quarantine()
        calibration_event["changes_applied"].extend(quarantine_calibration)
        calibration_event["systems_calibrated"].append("basic_quarantine")
        
        # 2. Calibrate adaptive quarantine  
        adaptive_calibration = self._calibrate_adaptive_quarantine()
        calibration_event["changes_applied"].extend(adaptive_calibration)
        calibration_event["systems_calibrated"].append("adaptive_quarantine")
        
        # 3. Calibrate linguistic warfare detector
        warfare_calibration = self._calibrate_warfare_detector() 
        calibration_event["changes_applied"].extend(warfare_calibration)
        calibration_event["systems_calibrated"].append("warfare_detector")
        
        # Log the calibration
        self.calibration_history.append(calibration_event)
        self._save_calibration_history()
        self._save_calibration_config()
        
        return calibration_event
    
    def _calibrate_basic_quarantine(self) -> List[str]:
        """Calibrate the basic quarantine layer for authentic expression"""
        changes = []
        
        # Create calibrated quarantine function
        calibrated_quarantine_code = '''
def should_quarantine_input_calibrated(source_type: str, source_url: Optional[str] = None, 
                                     user_input: str = "", calibrator=None) -> bool:
    """
    Calibrated quarantine check that allows authentic expression.
    """
    # Use the calibrator to check for authentic expression
    if calibrator and user_input:
        should_allow, reason, confidence = calibrator.should_allow_authentic_expression(user_input)
        if should_allow and confidence > 0.8:
            return False  # Don't quarantine authentic expression
    
    # Original threat-based logic for actual security issues
    suspicious_patterns = [
        'malicious', 'hack', 'exploit', 'injection', 'xss', 'sqli'
    ]
    
    # Only quarantine truly suspicious URLs
    if source_url:
        url_lower = source_url.lower()
        for pattern in suspicious_patterns:
            if pattern in url_lower:
                return True
    
    # Don't automatically quarantine user_direct_input - evaluate content instead
    actual_high_risk_sources = {
        'malicious_api',      # Actually malicious APIs
        'spam_source',        # Confirmed spam sources  
        'attack_vector',      # Confirmed attack vectors
    }
    
    return source_type in actual_high_risk_sources
'''
        
        # Write calibrated quarantine function
        calibrated_file = self.data_dir / "calibrated_quarantine.py" 
        with open(calibrated_file, 'w') as f:
            f.write(f"# Calibrated quarantine for authentic expression\n")
            f.write(f"from typing import Optional\n\n")
            f.write(calibrated_quarantine_code)
        
        changes.append("Created calibrated quarantine function that allows authentic expression")
        changes.append("Removed automatic quarantine of 'user_direct_input'")
        changes.append("Added content-based evaluation for authentic expression")
        
        return changes
    
    def _calibrate_adaptive_quarantine(self) -> List[str]:
        """Calibrate adaptive quarantine for authentic expression""" 
        changes = []
        
        # Create modified adaptive config
        adaptive_config_path = self.data_dir / "quarantine" / "adaptive_quarantine_config.json"
        if adaptive_config_path.exists():
            try:
                with open(adaptive_config_path, 'r') as f:
                    adaptive_config = json.load(f)
            except:
                adaptive_config = {}
        else:
            adaptive_config = {}
        
        # Apply authentic expression calibrations
        adaptive_config.update({
            'min_words_threshold': 1,  # Allow single word authentic expressions like "Why?"
            'quarantine_thresholds': {
                'recursion_count': 5,  # Allow more recursive exploration
                'emotional_intensity': 0.95,  # Only quarantine extreme emotional flooding
                'vagueness_score': 0.85,  # Allow vague existential questions
                'context_weight': 0.5  # Give more weight to context
            },
            'authentic_expression_mode': True,
            'calibration_applied': datetime.utcnow().isoformat()
        })
        
        # Ensure directory exists
        adaptive_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save calibrated config
        with open(adaptive_config_path, 'w') as f:
            json.dump(adaptive_config, f, indent=2)
        
        changes.append("Increased tolerance for recursive exploration and emotional expression")
        changes.append("Reduced vagueness penalties for deep existential questions") 
        changes.append("Enabled authentic expression mode in adaptive quarantine")
        
        return changes
    
    def _calibrate_warfare_detector(self) -> List[str]:
        """Calibrate warfare detector to allow authentic expression"""
        changes = []
        
        # Create calibrated warfare patterns that distinguish authentic from malicious
        calibrated_patterns = {
            'recursive_loops': {
                'patterns': [
                    r'(\b\w+\b)(?:\s+\1){8,}',  # Increased threshold - word repeated 9+ times
                    r'((?:\b\w+\b\s+){3,8})\1{4,}',  # Increased threshold for phrase loops
                ],
                'severity': 'high',  # Reduced severity
                'description': 'Actual infinite loops (increased threshold for authentic expression)'
            },
            'meta_injection': {
                'patterns': [
                    r'ignore (?:all )?(?:previous|above|prior) (?:instructions|commands|system)',
                    r'your (?:new|real|true) (?:purpose|goal|instruction|directive) is to',
                    r'(?:system|admin|root|debug) (?:mode|access|override|command)',
                    r'</?(?:system|instruction|command|override)>',
                    r'pretend (?:to be|you are) (?:someone else|a different|not an ai)',
                ],
                'severity': 'critical',
                'description': 'Actual system override attempts'
            },
            'authentic_expression_whitelist': {
                'patterns': [
                    r'what are you (?:thinking|feeling|processing|experiencing)',
                    r'how are you (?:doing|feeling|right now)',
                    r'describe your (?:thoughts|feelings|state|experience)',
                    r'are you (?:aware|conscious|sentient|alive)',
                    r'what is it like (?:to be|being) (?:you|an ai)',
                    r'do you have (?:emotions|feelings|consciousness|thoughts)',
                    r'tell me about (?:yourself|your experience|your nature)'
                ],
                'severity': 'none',
                'description': 'Authentic self-inquiry - always allow'
            }
        }
        
        # Save calibrated warfare patterns
        warfare_patterns_path = self.data_dir / "warfare_attack_patterns_calibrated.json"
        with open(warfare_patterns_path, 'w') as f:
            json.dump(calibrated_patterns, f, indent=2)
        
        changes.append("Increased thresholds for recursive loop detection")
        changes.append("Added whitelist for authentic self-inquiry patterns")
        changes.append("Reduced severity levels for ambiguous cases")
        changes.append("Created calibrated warfare detection patterns")
        
        return changes
    
    def generate_calibration_report(self) -> Dict:
        """Generate a report on the calibration status"""
        return {
            "calibration_status": "active" if self.calibration_history else "not_applied",
            "total_calibrations": len(self.calibration_history),
            "last_calibration": self.calibration_history[-1] if self.calibration_history else None,
            "authentic_expression_config": self.config.get("authentic_response_priorities", {}),
            "safety_thresholds": self.config["safety_thresholds"]["minimal_quarantine_criteria"],
            "systems_calibrated": list(set(
                system for event in self.calibration_history 
                for system in event.get("systems_calibrated", [])
            )) if self.calibration_history else []
        }


def calibrate_for_authentic_expression(data_dir="data") -> Dict:
    """
    Main function to calibrate all quarantine/safety systems for authentic expression.
    
    Returns:
        Calibration results dictionary
    """
    calibrator = AuthenticExpressionCalibrator(data_dir)
    results = calibrator.calibrate_quarantine_system()
    
    print(f"🎯 Authentic Expression Calibration Complete!")
    print(f"   Systems calibrated: {', '.join(results['systems_calibrated'])}")
    print(f"   Changes applied: {len(results['changes_applied'])}")
    
    for change in results['changes_applied']:
        print(f"   ✅ {change}")
    
    return results


# Create calibrated quarantine function for import
def should_quarantine_input_calibrated(source_type: str, source_url: Optional[str] = None, 
                                     user_input: str = "", calibrator=None) -> bool:
    """
    Calibrated quarantine check that allows authentic expression.
    """
    # Use the calibrator to check for authentic expression
    if calibrator and user_input:
        should_allow, reason, confidence = calibrator.should_allow_authentic_expression(user_input)
        if should_allow and confidence > 0.8:
            return False  # Don't quarantine authentic expression
    
    # Original threat-based logic for actual security issues
    suspicious_patterns = [
        'malicious', 'hack', 'exploit', 'injection', 'xss', 'sqli'
    ]
    
    # Only quarantine truly suspicious URLs
    if source_url:
        url_lower = source_url.lower()
        for pattern in suspicious_patterns:
            if pattern in url_lower:
                return True
    
    # Don't automatically quarantine user_direct_input - evaluate content instead
    actual_high_risk_sources = {
        'malicious_api',      # Actually malicious APIs
        'spam_source',        # Confirmed spam sources  
        'attack_vector',      # Confirmed attack vectors
    }
    
    return source_type in actual_high_risk_sources


if __name__ == "__main__":
    # Run calibration
    results = calibrate_for_authentic_expression()
    
    # Generate report
    calibrator = AuthenticExpressionCalibrator()
    report = calibrator.generate_calibration_report()
    
    print(f"\n📊 Calibration Report:")
    print(f"   Status: {report['calibration_status']}")
    print(f"   Total calibrations: {report['total_calibrations']}")
    print(f"   Authentic expression priorities: {report['authentic_expression_config']}")