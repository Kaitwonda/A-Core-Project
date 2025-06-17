# Calibrated quarantine for authentic expression
from typing import Optional


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
