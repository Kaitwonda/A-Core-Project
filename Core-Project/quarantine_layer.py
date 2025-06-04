# quarantine_layer.py - Base quarantine functionality for adaptive system

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List


def should_quarantine_input(source_type: str, source_url: Optional[str] = None) -> bool:
    """
    Basic source-based quarantine check.
    Used by vector_memory.py and other components.
    
    Args:
        source_type: Type of source (e.g., 'user_direct_input', 'web_scrape', etc.)
        source_url: Optional URL or identifier of the source
        
    Returns:
        bool: True if the input should be quarantined based on source
    """
    # High-risk source types that need quarantine consideration
    high_risk_sources = {
        'user_direct_input',  # Direct user messages
        'social_media',       # Social media content
        'untrusted_api',      # Untrusted external APIs
        'anonymous_upload',   # Anonymous file uploads
    }
    
    # Suspicious URL patterns
    suspicious_patterns = [
        'malicious',
        'hack',
        'exploit',
        'injection',
        'xss',
        'sqli'
    ]
    
    # Check if source type is high risk
    if source_type in high_risk_sources:
        return True
        
    # Check URL for suspicious patterns
    if source_url:
        url_lower = source_url.lower()
        for pattern in suspicious_patterns:
            if pattern in url_lower:
                return True
                
    # Default: don't quarantine
    return False


class UserMemoryQuarantine:
    """
    Base quarantine class with minimal functionality.
    Extended by AdaptiveQuarantine for smart behavior.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Quarantine storage directory
        self.quarantine_dir = self.data_dir / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic file paths
        self.quarantine_log = self.quarantine_dir / "quarantine_log.json"
        self.pattern_database = self.quarantine_dir / "pattern_database.json"
        
        # Initialize files if needed
        self._init_files()
        
    def _init_files(self):
        """Initialize storage files if they don't exist"""
        if not self.quarantine_log.exists():
            with open(self.quarantine_log, 'w') as f:
                json.dump([], f)
                
        if not self.pattern_database.exists():
            with open(self.pattern_database, 'w') as f:
                json.dump({}, f)
    
    def get_quarantine_statistics(self) -> Dict:
        """
        Get basic quarantine statistics.
        Extended by adaptive version for more detail.
        """
        if not self.quarantine_log.exists():
            return {
                'total_quarantines': 0,
                'active_quarantines': 0
            }
            
        with open(self.quarantine_log, 'r') as f:
            log = json.load(f)
            
        return {
            'total_quarantines': len(log),
            'active_quarantines': len(log)  # Simplified - adaptive version handles expiry
        }
    
    def quarantine(self, zone_id: str, reason: str = "manual", severity: str = "medium") -> Dict:
        """
        Basic quarantine method - just logs the entry.
        Adaptive version does the smart processing.
        """
        # Simple logging for compatibility
        record = {
            'zone_id': zone_id,
            'reason': reason,
            'severity': severity,
            'timestamp': str(Path.ctime(Path()))  # Simple timestamp
        }
        
        # Add to log
        if self.quarantine_log.exists():
            with open(self.quarantine_log, 'r') as f:
                log = json.load(f)
        else:
            log = []
            
        log.append(record)
        
        with open(self.quarantine_log, 'w') as f:
            json.dump(log, f)
            
        return {
            'success': True,
            'quarantine_id': zone_id,
            'severity': severity
        }
    
    def check_user_history(self, user_id: str) -> Dict:
        """
        Basic user history check.
        Returns minimal info for compatibility.
        """
        return {
            'risk_level': 'low',
            'history_length': 0,
            'recent_patterns': []
        }
    
    def load_all_quarantined_memory(self) -> list:
        """
        Load quarantined records for visualization.
        Returns empty list in base implementation.
        """
        if not self.quarantine_log.exists():
            return []
            
        try:
            with open(self.quarantine_log, 'r') as f:
                log = json.load(f)
                
            # Return sanitized records for visualization
            sanitized = []
            for record in log:
                sanitized.append({
                    'pattern_signature': record.get('pattern_signature', 'unknown'),
                    'zone_tags': record.get('zone_tags', {}),
                    'contamination_type': record.get('contamination_type', 'unknown'),
                    'severity': record.get('severity', 'medium')
                })
                
            return sanitized
        except:
            return []
    
    def check_contamination_risk(self, zone_output: Dict) -> Dict:
        """
        Check if a zone output might be contaminated by quarantined patterns.
        Base implementation always returns low risk.
        """
        return {
            'contamination_detected': False,
            'risk_level': 'low',
            'recommendation': 'proceed_normal'
        }
    
    def quarantine_user_input(self, text: str, user_id: str, 
                             source_url: Optional[str] = None, 
                             detected_emotions: Optional[Dict] = None,
                             matched_symbols: Optional[List] = None,
                             current_phase: int = 0) -> Dict:
        """
        Quarantine user input directly.
        Base implementation just logs it.
        """
        quarantine_id = f"q_{len(text)}_{datetime.now().timestamp()}"
        
        # Simple quarantine for compatibility
        result = self.quarantine(
            zone_id=quarantine_id,
            reason="user_input_quarantine",
            severity="medium"
        )
        
        return {
            'quarantine_id': quarantine_id,
            'success': result['success'],
            'warfare_detected': False,
            'threats': []
        }