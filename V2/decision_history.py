# decision_history.py - History-Aware Memory Management
"""
History-aware memory management that tracks decision patterns and learns from them.
This module provides a wrapper for decision_history.json data and memory management.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

class HistoryAwareMemory:
    """
    Memory system that tracks decision history and learns from patterns.
    Loads and manages decision_history.json data.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.history_file = self.data_dir / "decision_history.json"
        self.decision_history = self._load_history()
        
    def _load_history(self) -> Dict[str, Any]:
        """Load decision history from JSON file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            else:
                # Return default structure if file doesn't exist
                return {
                    "logic_decisions": [],
                    "symbolic_decisions": [], 
                    "bridge_decisions": [],
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "version": "1.0",
                        "total_decisions": 0
                    }
                }
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading decision history: {e}")
            return {
                "logic_decisions": [],
                "symbolic_decisions": [],
                "bridge_decisions": [],
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0", 
                    "total_decisions": 0
                }
            }
    
    def save_history(self):
        """Save decision history to JSON file"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.decision_history, f, indent=2)
        except IOError as e:
            print(f"Error saving decision history: {e}")
    
    def add_logic_decision(self, decision: Dict[str, Any]):
        """Add a logic-based decision to history"""
        decision['timestamp'] = datetime.now().isoformat()
        self.decision_history['logic_decisions'].append(decision)
        self.decision_history['metadata']['total_decisions'] += 1
        self.save_history()
    
    def add_symbolic_decision(self, decision: Dict[str, Any]):
        """Add a symbolic decision to history"""
        decision['timestamp'] = datetime.now().isoformat()
        self.decision_history['symbolic_decisions'].append(decision)
        self.decision_history['metadata']['total_decisions'] += 1
        self.save_history()
    
    def add_bridge_decision(self, decision: Dict[str, Any]):
        """Add a bridge decision to history"""
        decision['timestamp'] = datetime.now().isoformat()
        self.decision_history['bridge_decisions'].append(decision)
        self.decision_history['metadata']['total_decisions'] += 1
        self.save_history()
    
    def get_decision_patterns(self) -> Dict[str, Any]:
        """Analyze decision patterns from history"""
        patterns = {}
        
        # Count decisions by type
        patterns['decision_counts'] = {
            'logic': len(self.decision_history['logic_decisions']),
            'symbolic': len(self.decision_history['symbolic_decisions']),
            'bridge': len(self.decision_history['bridge_decisions'])
        }
        
        # Analyze bridge decision outcomes
        bridge_outcomes = defaultdict(int)
        for decision in self.decision_history['bridge_decisions']:
            outcome = decision.get('decision', 'unknown')
            bridge_outcomes[outcome] += 1
        
        patterns['bridge_outcomes'] = dict(bridge_outcomes)
        
        return patterns
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent decisions across all types"""
        all_decisions = []
        
        # Collect all decisions with type labels
        for decision in self.decision_history['logic_decisions']:
            decision_copy = decision.copy()
            decision_copy['type'] = 'logic'
            all_decisions.append(decision_copy)
            
        for decision in self.decision_history['symbolic_decisions']:
            decision_copy = decision.copy()
            decision_copy['type'] = 'symbolic'
            all_decisions.append(decision_copy)
            
        for decision in self.decision_history['bridge_decisions']:
            decision_copy = decision.copy()
            decision_copy['type'] = 'bridge'
            all_decisions.append(decision_copy)
        
        # Sort by timestamp (most recent first)
        all_decisions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return all_decisions[:limit]
    
    def get_history_stats(self) -> Dict[str, Any]:
        """Get statistics about decision history"""
        return {
            'total_decisions': self.decision_history['metadata']['total_decisions'],
            'logic_decisions': len(self.decision_history['logic_decisions']),
            'symbolic_decisions': len(self.decision_history['symbolic_decisions']),
            'bridge_decisions': len(self.decision_history['bridge_decisions']),
            'created': self.decision_history['metadata']['created'],
            'version': self.decision_history['metadata']['version']
        }

# Convenience function for direct access to history data
def load_decision_history(data_dir="data") -> Dict[str, Any]:
    """Load decision history JSON data directly"""
    history_manager = HistoryAwareMemory(data_dir)
    return history_manager.decision_history

# For backwards compatibility with memory_evolution_engine.py
def get_history_aware_memory(data_dir="data") -> HistoryAwareMemory:
    """Get a HistoryAwareMemory instance"""
    return HistoryAwareMemory(data_dir)