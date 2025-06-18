# unified_memory.py - Consolidated Memory System
"""
Unified Memory System - Consolidates all memory functionality:
- memory_architecture.py: Tripartite memory with atomic persistence
- decision_history.py: Decision tracking and stability analysis  
- user_memory.py: Symbol occurrence logging
- symbol_memory.py: Symbol storage with security features
- vector_memory.py: Vector storage with quarantine
- trail_log.py: Processing trail logging

This replaces 6 separate memory files with a single, cohesive system.
"""

import json
import shutil
import numpy as np
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, List, Optional, Tuple, Any
import traceback

# Import dependencies
from vector_engine import fuse_vectors, embed_text
from sklearn.metrics.pairwise import cosine_similarity
from quarantine_layer import UserMemoryQuarantine, should_quarantine_input
from linguistic_warfare import LinguisticWarfareDetector, check_for_warfare
from visualization_prep import VisualizationPrep

# ============================================================================
# CORE TRIPARTITE MEMORY ARCHITECTURE
# ============================================================================

class TripartiteMemory:
    """
    Three-way memory architecture with atomic persistence and recovery.
    Stores logic, symbolic, and bridge memories with backup recovery.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()  # Thread safety
        
        # Memory stores
        self.logic_memory = []
        self.symbolic_memory = []
        self.bridge_memory = []
        
        # Load existing data
        self._load_all()
        
    def _load_all(self):
        """Load all memories with fallback to backups"""
        self.logic_memory = self._load_safe("logic_memory.json")
        self.symbolic_memory = self._load_safe("symbolic_memory.json")
        self.bridge_memory = self._load_safe("bridge_memory.json")
        
    def _load_safe(self, filename):
        """Load with backup recovery and error handling"""
        path = self.data_dir / filename
        backup = self.data_dir / f"{filename}.backup"
        
        # Try primary file first
        try:
            if path.exists() and path.stat().st_size > 0:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ Loaded {filename}: {len(data)} items")
                    return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ Error loading {filename}: {e}")
            
        # Try backup if primary failed
        try:
            if backup.exists() and backup.stat().st_size > 0:
                print(f"🔄 Recovering {filename} from backup")
                with open(backup, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Restore primary from backup
                    shutil.copy2(backup, path)
                    print(f"✅ Recovered {filename}: {len(data)} items")
                    return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ Backup recovery failed for {filename}: {e}")
            
        # Return empty list if both failed
        print(f"📝 Starting fresh for {filename}")
        return []
        
    def store(self, item, decision_type):
        """Store an item in the appropriate memory"""
        with self.lock:
            # Add metadata
            item['stored_at'] = datetime.utcnow().isoformat()
            item['decision_type'] = decision_type
            
            if decision_type == "FOLLOW_LOGIC":
                self.logic_memory.append(item)
            elif decision_type == "FOLLOW_SYMBOLIC":
                self.symbolic_memory.append(item)
            else:  # FOLLOW_HYBRID
                self.bridge_memory.append(item)
                
    def save_all(self):
        """Atomic save all memories with backups"""
        with self.lock:
            results = {
                'logic': self._save_safe("logic_memory.json", self.logic_memory),
                'symbolic': self._save_safe("symbolic_memory.json", self.symbolic_memory),
                'bridge': self._save_safe("bridge_memory.json", self.bridge_memory)
            }
            
            success_count = sum(1 for v in results.values() if v)
            print(f"💾 Saved {success_count}/3 memory stores successfully")
            return results
            
    def _save_safe(self, filename, data):
        """Save with atomic write and backup"""
        path = self.data_dir / filename
        temp = path.with_suffix('.tmp')
        backup = self.data_dir / f"{filename}.backup"
        
        try:
            # Write to temp file first
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            # Create backup of existing file if it exists
            if path.exists() and path.stat().st_size > 0:
                shutil.copy2(path, backup)
                
            # Atomic rename (temp -> primary)
            temp.replace(path)
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")
            traceback.print_exc()
            # Clean up temp file if it exists
            if temp.exists():
                temp.unlink()
            return False
            
    def get_counts(self):
        """Get item counts for each memory type"""
        with self.lock:
            return {
                'logic': len(self.logic_memory),
                'symbolic': len(self.symbolic_memory),
                'bridge': len(self.bridge_memory),
                'total': len(self.logic_memory) + len(self.symbolic_memory) + len(self.bridge_memory)
            }
            
    def clear_all(self):
        """Clear all memories (useful for testing)"""
        with self.lock:
            self.logic_memory = []
            self.symbolic_memory = []
            self.bridge_memory = []

# ============================================================================
# DECISION HISTORY TRACKING
# ============================================================================

class HistoryAwareMemory(TripartiteMemory):
    """
    Enhanced TripartiteMemory that tracks decision history for each item.
    """

    def __init__(self, data_dir="data", max_history_length=5):
        super().__init__(data_dir)
        self.max_history_length = max_history_length
        
    def store(self, item, decision_type, weights=None):
        """
        Store with decision history tracking.
        """
        acquired = self.lock.acquire(timeout=5.0)
        
        if not acquired:
            print(f"⚠️ Could not acquire lock for storing item '{item.get('id', 'Unknown ID')}' after 5 seconds, skipping...")
            return
            
        try:
            # Initialize or update decision history
            if 'decision_history' not in item:
                item['decision_history'] = []
                
            # Create history entry
            history_entry = {
                'decision': decision_type,
                'timestamp': datetime.utcnow().isoformat(),
                'weights': weights or self._get_current_weights()
            }
            
            # Append and trim history
            item['decision_history'].append(history_entry)
            item['decision_history'] = item['decision_history'][-self.max_history_length:]
            
            # Add other metadata
            item['last_decision'] = decision_type
            item['history_length'] = len(item['decision_history'])
            
            # Store using parent method
            super().store(item, decision_type)
            
        except Exception as e:
            print(f"❌ Exception during store operation: {e}")
        finally:
            self.lock.release()
            
    def _get_current_weights(self):
        """Get current adaptive weights (stub for now)"""
        return {
            'static': 0.6,
            'dynamic': 0.4
        }
        
    def get_item_stability(self, item):
        """
        Calculate how stable an item's decisions have been.
        """
        history = item.get('decision_history', [])
        
        if len(history) < 2:
            return {
                'is_stable': False,
                'stability_score': 0.0,
                'dominant_decision': None,
                'decision_counts': {},
                'history_length': len(history)
            }
            
        # Count decisions
        decision_counts = {}
        for entry in history:
            dec = entry['decision']
            decision_counts[dec] = decision_counts.get(dec, 0) + 1
            
        # Find dominant decision
        dominant_decision = max(decision_counts.items(), key=lambda x: x[1])[0]
        dominant_count = decision_counts[dominant_decision]
        
        # Calculate stability score (0-1)
        stability_score = dominant_count / len(history)
        
        # Check recent consistency
        recent_history = history[-3:] if len(history) >= 3 else history
        recent_decisions = [h['decision'] for h in recent_history]
        is_recently_stable = len(set(recent_decisions)) == 1
        
        return {
            'is_stable': stability_score >= 0.6 and is_recently_stable,
            'stability_score': stability_score,
            'dominant_decision': dominant_decision,
            'decision_counts': decision_counts,
            'history_length': len(history),
            'recent_consistency': is_recently_stable
        }

# ============================================================================
# USER MEMORY (SYMBOL OCCURRENCE LOGGING)
# ============================================================================

class UserMemory:
    """Symbol occurrence logging functionality"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "symbol_occurrence_log.json"
    
    def load_user_memory(self):
        """Loads symbol occurrence entries from the JSON file."""
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            with open(self.file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict) and "entries" in data:
                        return data.get("entries", []) if isinstance(data["entries"], list) else []
                    elif isinstance(data, list): 
                        return data
                    else: 
                        print(f"⚠️ Symbol occurrence log file has unexpected format. Returning empty list.")
                        return []
                except json.JSONDecodeError:
                    print(f"⚠️ Symbol occurrence log file is corrupted. Returning empty list.")
                    return []
        return []

    def save_user_memory(self, entries):
        """Saves symbol occurrence entries to the JSON file."""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)

    def add_user_memory_entry(self, symbol, context_text, emotion_in_context, 
                              source_url=None, learning_phase=None, 
                              is_context_highly_relevant=None):
        """Adds a new symbol occurrence entry to the user memory."""
        entries = self.load_user_memory()

        new_entry = {
            "symbol": symbol,
            "context_text": context_text,
            "emotion_in_context": emotion_in_context,
            "source_url": source_url,
            "learning_phase": learning_phase,
            "is_context_highly_relevant": is_context_highly_relevant,
            "timestamp": datetime.utcnow().isoformat()
        }
        entries.append(new_entry)
        self.save_user_memory(entries)
        return new_entry["timestamp"]

# ============================================================================
# SYMBOL MEMORY WITH SECURITY
# ============================================================================

class SymbolMemory:
    """Symbol storage with security features and visualization metadata"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "symbol_memory.json"
        
        # Initialize security components
        self.quarantine = None
        self.warfare_detector = None
        try:
            self.quarantine = UserMemoryQuarantine()
            self.warfare_detector = LinguisticWarfareDetector()
        except ImportError:
            print("⚠️ Security modules not available")

    def load_symbol_memory(self):
        """Loads existing symbol memory, ensuring it's a dictionary of symbol objects."""
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            with open(self.file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return {
                            token: details
                            for token, details in data.items()
                            if isinstance(details, dict) and "name" in details
                        }
                    else:
                        print(f"⚠️ Symbol memory file is not a dictionary. Returning empty memory.")
                        return {}
                except json.JSONDecodeError:
                    print(f"⚠️ Symbol memory file corrupted. Returning empty memory.")
                    return {}
        return {}

    def save_symbol_memory(self, memory):
        """Saves current state of symbol memory to disk."""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)

    def _check_quarantine_status(self, origin: str, example_text: str = None, 
                               name: str = None, keywords: List[str] = None) -> Tuple[bool, Optional[Dict]]:
        """Check if a symbol should be quarantined"""
        quarantine_origins = [
            "user_quarantine", "quarantined_input", "warfare_detected",
            "manipulation_attempt", "unverified_user"
        ]
        
        if origin in quarantine_origins:
            return True, {"reason": f"Quarantine origin: {origin}"}
        
        # Check for linguistic warfare if available
        if self.warfare_detector and example_text:
            analysis_text = f"{name or ''} {' '.join(keywords or [])} {example_text or ''}"
            analysis = self.warfare_detector.analyze_text_for_warfare(analysis_text, user_id="symbol_creation")
            
            if analysis['threat_score'] > 0.7:
                return True, analysis
        
        return False, None

    def _sanitize_symbol_data(self, symbol_data: Dict) -> Dict:
        """Sanitize symbol data to prevent malicious content"""
        max_lengths = {
            'name': 100, 'keywords': 50, 'example_text': 500, 'core_meanings': 200
        }
        
        if 'name' in symbol_data:
            symbol_data['name'] = symbol_data['name'][:max_lengths['name']]
        
        if 'keywords' in symbol_data and isinstance(symbol_data['keywords'], list):
            symbol_data['keywords'] = [
                kw[:max_lengths['keywords']] for kw in symbol_data['keywords'][:20]
            ]
        
        # Remove script injection attempts
        dangerous_patterns = ['<script', '</script>', 'javascript:', 'onerror=', 'onclick=']
        for field in ['name', 'summary', 'description']:
            if field in symbol_data and isinstance(symbol_data[field], str):
                cleaned = symbol_data[field]
                for pattern in dangerous_patterns:
                    cleaned = cleaned.replace(pattern, '')
                symbol_data[field] = cleaned
        
        return symbol_data

    def add_symbol(self, symbol_token, name, keywords, initial_emotions, example_text,
                   origin="emergent", learning_phase=0, resonance_weight=0.5,
                   symbol_details_override=None, skip_quarantine_check=False):
        """Add a new symbol or update an existing one with security checks"""
        
        # Check quarantine status unless explicitly skipped
        if not skip_quarantine_check:
            should_quarantine, warfare_analysis = self._check_quarantine_status(
                origin, example_text, name, keywords
            )
            
            if should_quarantine:
                print(f"🔒 Symbol '{symbol_token}' ({name}) blocked - origin: {origin}")
                if self.quarantine:
                    self.quarantine.quarantine_user_input(
                        text=f"Symbol creation attempt: {symbol_token} - {name}",
                        user_id="symbol_memory",
                        matched_symbols=[{'symbol': symbol_token, 'name': name}],
                        source_url=f"symbol_memory:{origin}"
                    )
                return None
        
        memory = self.load_symbol_memory()
        
        # Extract numeric weights and build peak emotions map
        incoming_numeric_weights = []
        peak_emotions_from_initial = {}
        
        if isinstance(initial_emotions, dict):
            for emo, score in initial_emotions.items():
                if isinstance(score, (int, float)):
                    incoming_numeric_weights.append(score)
                    peak_emotions_from_initial[emo] = score
        elif isinstance(initial_emotions, list):
            for item in initial_emotions:
                if isinstance(item, dict) and "weight" in item:
                    incoming_numeric_weights.append(item["weight"])
                    if "emotion" in item:
                        peak_emotions_from_initial[item["emotion"]] = item["weight"]
                elif isinstance(item, tuple) and len(item) == 2:
                    incoming_numeric_weights.append(item[1])
                    peak_emotions_from_initial[item[0]] = item[1]

        current_max_incoming_weight = max(incoming_numeric_weights, default=0.0)
        
        if symbol_token not in memory:
            # New symbol creation
            new_symbol_data = {
                "name": name,
                "keywords": list(set(keywords)),
                "core_meanings": [],
                "emotions": initial_emotions,
                "emotion_profile": {},
                "vector_examples": [],
                "origin": origin,
                "learning_phase": learning_phase,
                "resonance_weight": resonance_weight,
                "golden_memory": {
                    "peak_weight": current_max_incoming_weight,
                    "context": example_text or "",
                    "peak_emotions": peak_emotions_from_initial,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "usage_count": 0,
                "visualization_metadata": {
                    "primary_color": self._get_symbol_color(initial_emotions),
                    "display_priority": self._calculate_display_priority(resonance_weight, origin),
                    "classification_hint": self._get_classification_hint(keywords, initial_emotions)
                }
            }
            
            memory[symbol_token] = self._sanitize_symbol_data(new_symbol_data)
        else:
            # Update existing symbol
            memory[symbol_token]["updated_at"] = datetime.utcnow().isoformat()
            memory[symbol_token]["usage_count"] = memory[symbol_token].get("usage_count", 0) + 1
            
        self.save_symbol_memory(memory)
        return memory[symbol_token]

    def _get_symbol_color(self, emotions: Any) -> str:
        """Determine a color for the symbol based on its emotional profile"""
        emotion_colors = {
            "joy": "#FFD700", "love": "#FF69B4", "anger": "#DC143C", "fear": "#4B0082",
            "sadness": "#4682B4", "surprise": "#FF8C00", "trust": "#87CEEB", "neutral": "#C0C0C0"
        }
        
        if isinstance(emotions, dict) and emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            return emotion_colors.get(dominant_emotion.lower(), "#808080")
        
        return "#808080"  # Default gray

    def _calculate_display_priority(self, resonance_weight: float, origin: str) -> float:
        """Calculate display priority for visualization"""
        origin_boosts = {
            "seed": 0.3, "meta_emergent": 0.2, "generated": 0.1,
            "user_quarantine": -0.5
        }
        boost = origin_boosts.get(origin, 0.0)
        return round(min(1.0, max(0.0, resonance_weight + boost)), 3)

    def _get_classification_hint(self, keywords: List[str], emotions: Any) -> str:
        """Provide a hint about whether this symbol is more logic or symbolic oriented"""
        logic_keywords = {"algorithm", "data", "system", "process", "logic", "function"}
        symbolic_keywords = {"emotion", "feel", "symbol", "meaning", "soul", "dream"}
        
        logic_score = sum(1 for kw in keywords if any(lk in kw.lower() for lk in logic_keywords))
        symbolic_score = sum(1 for kw in keywords if any(sk in kw.lower() for sk in symbolic_keywords))
        
        if logic_score > symbolic_score * 1.5:
            return "logic"
        elif symbolic_score > logic_score * 1.5:
            return "symbolic"
        else:
            return "hybrid"

# ============================================================================
# VECTOR MEMORY WITH QUARANTINE
# ============================================================================

class VectorMemory:
    """Vector storage with quarantine and warfare detection"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "vector_memory.json"
        
        # Initialize security components
        try:
            self.quarantine = UserMemoryQuarantine()
            self.warfare_detector = LinguisticWarfareDetector()
            self.viz_prep = VisualizationPrep()
        except ImportError:
            print("⚠️ Security/visualization modules not available")
            self.quarantine = None
            self.warfare_detector = None
            self.viz_prep = None

    def _load_memory(self) -> List[Dict[str, Any]]:
        """Load vector memory from disk"""
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Vector memory file corrupted, starting fresh.")
                return []
        return []

    def _save_memory(self, memory_data: List[Dict[str, Any]]):
        """Save vector memory to disk"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)

    def store_vector(self, text: str, source_url: Optional[str] = None,
                     source_type: str = "unknown", learning_phase: int = 0,
                     confidence: float = 0.5, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Store text as vector with quarantine and warfare checks"""
        if not text or not text.strip():
            return {"status": "error", "message": "Empty text provided"}
        
        # Generate unique ID
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        timestamp = datetime.utcnow().isoformat()
        entry_id = f"vec_{text_hash}_{int(datetime.utcnow().timestamp())}"
        
        # Check if input should be quarantined
        if should_quarantine_input(source_type, source_url):
            if self.quarantine:
                quarantine_result = self.quarantine.quarantine_user_input(
                    text=text, user_id=source_url or "anonymous",
                    source_url=source_url, current_phase=learning_phase
                )
            else:
                quarantine_result = {'quarantine_id': 'mock_id'}
            
            return {
                "status": "quarantined",
                "entry_id": entry_id,
                "quarantine_id": quarantine_result['quarantine_id'],
                "message": "Content quarantined due to source type"
            }
        
        # Check for linguistic warfare
        if self.warfare_detector:
            should_quarantine_warfare, warfare_analysis = check_for_warfare(text, source_url or "anonymous")
            
            if should_quarantine_warfare:
                return {
                    "status": "quarantined_warfare",
                    "entry_id": entry_id,
                    "threats_detected": warfare_analysis['threats_detected'],
                    "message": f"Content quarantined: {warfare_analysis['defense_strategy']['explanation']}"
                }
        
        # Normal storage
        vec_result, debug_info = fuse_vectors(text)
        
        if vec_result is None:
            return {
                "status": "error",
                "message": f"Vector encoding failed: {debug_info.get('error', 'Unknown error')}"
            }
        
        # Create memory entry
        entry = {
            "id": entry_id,
            "text": text[:1000],  # Limit text size
            "vector": vec_result,
            "source_url": source_url,
            "source_type": source_type,
            "learning_phase": learning_phase,
            "confidence": confidence,
            "timestamp": timestamp,
            "vector_debug": debug_info,
            "quarantined": False,
            "metadata": metadata or {}
        }
        
        memory = self._load_memory()
        memory.append(entry)
        self._save_memory(memory)
        
        return {
            "status": "success",
            "entry_id": entry_id,
            "vector_source": debug_info.get("source", "unknown"),
            "message": "Vector stored successfully"
        }

    def retrieve_similar_vectors(self, query_text: str, top_n: int = 5,
                                 include_quarantined: bool = False,
                                 similarity_threshold: float = 0.0) -> List[Tuple[float, Dict]]:
        """Retrieve similar vectors with quarantine filtering"""
        if not query_text or not query_text.strip():
            return []
        
        query_vec, debug = fuse_vectors(query_text)
        if query_vec is None:
            return []
        
        query_vec_np = np.array(query_vec).reshape(1, -1)
        memory = self._load_memory()
        
        # Filter candidates
        candidates = []
        for entry in memory:
            if entry.get('quarantined', False) and not include_quarantined:
                continue
            if not entry.get('vector') or len(entry['vector']) == 0:
                continue
            candidates.append(entry)
        
        if not candidates:
            return []
        
        # Calculate similarities
        results = []
        for entry in candidates:
            try:
                entry_vec = np.array(entry['vector']).reshape(1, -1)
                similarity = cosine_similarity(query_vec_np, entry_vec)[0][0]
                
                if similarity >= similarity_threshold:
                    results.append((float(similarity), entry))
            except Exception as e:
                continue
        
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_n]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about vector memory including quarantine info"""
        memory = self._load_memory()
        
        total_entries = len(memory)
        quarantined_entries = sum(1 for e in memory if e.get('quarantined', False))
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - quarantined_entries,
            "quarantined_entries": quarantined_entries,
            "quarantine_percentage": (quarantined_entries / total_entries * 100) if total_entries > 0 else 0,
            "memory_size_bytes": self.file_path.stat().st_size if self.file_path.exists() else 0
        }

# ============================================================================
# TRAIL LOGGING
# ============================================================================

class TrailLogger:
    """Processing trail logging functionality"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / "trail_log.json"

    def _load_log(self):
        """Loads the current trail log, ensuring it's a list"""
        if self.file_path.exists() and self.file_path.stat().st_size > 0:
            with open(self.file_path, "r", encoding="utf-8") as f:
                try:
                    content = json.load(f)
                    if isinstance(content, dict) and "entries" in content:
                        return content["entries"] if isinstance(content["entries"], list) else []
                    return content if isinstance(content, list) else []
                except json.JSONDecodeError:
                    print(f"⚠️ Trail log file corrupted. Initializing new log.")
                    return []
        return []

    def _save_log(self, log_entries):
        """Saves the trail log as a list of entries"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2, ensure_ascii=False)

    def log_dynamic_bridge_processing_step(self, log_id=None, text_input=None, source_url=None, 
                                         current_phase=0, directives=None, is_highly_relevant_for_phase=False,
                                         target_storage_phase_for_chunk=None, is_shallow_content=False, 
                                         content_type_heuristic="ambiguous", detected_emotions_output=None,
                                         logic_node_output=None, symbolic_node_output=None,
                                         generated_response_preview=None):
        """Log a detailed record of a single processing step from the DynamicBridge"""
        current_log = self._load_log()

        if log_id is None:
            timestamp_str = datetime.utcnow().isoformat().replace(':', '-').replace('.', '-')
            input_hash = hashlib.md5(text_input.encode('utf-8')).hexdigest()[:8] if text_input else "no_input"
            log_id = f"step_{timestamp_str}_{input_hash}"

        # Ensure complex objects are serializable
        serializable_directives = {}
        if directives:
            for k, v in directives.items():
                if isinstance(v, Path):
                    serializable_directives[k] = str(v)
                elif isinstance(v, dict):
                    serializable_directives[k] = {
                        dk: dv if not isinstance(dv, Path) else str(dv) 
                        for dk, dv in v.items()
                    }
                else:
                    serializable_directives[k] = v
        
        log_entry = {
            "log_id": log_id,
            "timestamp": datetime.utcnow().isoformat(),
            "input_text_preview": text_input[:200] + "..." if text_input and len(text_input) > 200 else text_input,
            "source_url": source_url,
            "processing_phase": current_phase, 
            "target_storage_phase_for_chunk": target_storage_phase_for_chunk, 
            "is_shallow_content": is_shallow_content,
            "content_type_heuristic": content_type_heuristic,
            "phase_directives_info": serializable_directives.get("info", "N/A") if serializable_directives else "N/A", 
            "phase_directives_full": serializable_directives, 
            "is_highly_relevant_for_current_processing_phase": is_highly_relevant_for_phase,
            "detected_emotions_summary": { 
                 "top_verified": detected_emotions_output.get("verified", [])[:3] if detected_emotions_output else [],
            },
            "logic_node_summary": logic_node_output,
            "symbolic_node_summary": symbolic_node_output,
            "generated_response_preview": generated_response_preview[:200] + "..." if generated_response_preview and len(generated_response_preview) > 200 else generated_response_preview
        }
        current_log.append(log_entry)
        self._save_log(current_log)

    def log_trail(self, text, symbols, matches):
        """Legacy trail logging for backward compatibility"""
        log_data = self._load_log()

        entry_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        new_entry = {
            "id": entry_id,
            "timestamp": datetime.utcnow().isoformat(),
            "text": text,
            "symbols": symbols,
            "matches": matches,
            "emotions": [],
            "type": "legacy_trail"
        }
        log_data.append(new_entry)
        self._save_log(log_data)
        return entry_id

    def add_emotions(self, entry_id, emotions):
        """Add detected emotions to a specific log entry"""
        log_data = self._load_log()
        for entry in log_data:
            if entry.get("id") == entry_id and entry.get("type") == "legacy_trail":
                entry["emotions"] = emotions
                break
        self._save_log(log_data)

# ============================================================================
# UNIFIED MEMORY SYSTEM
# ============================================================================

class UnifiedMemory:
    """
    Unified Memory System that consolidates all memory functionality.
    Single interface for all memory operations.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all memory components
        self.tripartite = HistoryAwareMemory(data_dir)
        self.user_memory = UserMemory(data_dir)
        self.symbol_memory = SymbolMemory(data_dir)
        self.vector_memory = VectorMemory(data_dir)
        self.trail_logger = TrailLogger(data_dir)
        
        # Auto-load all existing data at startup
        self._load_all_data()
        
        print(f"🧠 Unified Memory System initialized in {data_dir}")
    
    def _load_all_data(self):
        """Load all existing data from JSON files at startup"""
        # Load vector memory data
        self.vector_data = self.vector_memory._load_memory()
        print(f"📊 Loaded vector memory: {len(self.vector_data)} entries")
        
        # Load trail log data
        self.trail_data = self.trail_logger._load_log()
        print(f"📊 Loaded trail log: {len(self.trail_data)} entries")
        
        # Tripartite memory is already loaded in HistoryAwareMemory.__init__
        counts = self.tripartite.get_counts()
        print(f"📊 Loaded tripartite memory: Logic({counts['logic']}) Symbolic({counts['symbolic']}) Bridge({counts['bridge']})")
    
    # ========================================================================
    # TRIPARTITE MEMORY INTERFACE
    # ========================================================================
    
    def store_decision(self, item, decision_type, weights=None):
        """Store a decision in tripartite memory with history tracking"""
        return self.tripartite.store(item, decision_type, weights)
    
    def get_memory_counts(self):
        """Get counts for all memory types"""
        return self.tripartite.get_counts()
    
    def get_item_stability(self, item):
        """Get stability analysis for an item"""
        return self.tripartite.get_item_stability(item)
    
    # ========================================================================
    # COGNITIVE CONTINUITY LAYER - Bridge for memory_analytics.py compatibility
    # ========================================================================
    
    def get_counts(self):
        """Legacy compatibility bridge for memory_analytics.py"""
        return self.get_memory_counts()
    
    @property
    def logic_memory(self):
        """Direct access compatibility bridge for memory_analytics.py"""
        return self.tripartite.logic_memory
    
    @property  
    def symbolic_memory(self):
        """Direct access compatibility bridge for memory_analytics.py"""
        return self.tripartite.symbolic_memory
    
    @property
    def bridge_memory(self):
        """Direct access compatibility bridge for memory_analytics.py"""
        return self.tripartite.bridge_memory
    
    def save_tripartite_memory(self):
        """Save all tripartite memories"""
        return self.tripartite.save_all()
    
    # ========================================================================
    # USER MEMORY INTERFACE  
    # ========================================================================
    
    def add_symbol_occurrence(self, symbol, context_text, emotion_in_context, 
                             source_url=None, learning_phase=None, is_context_highly_relevant=None):
        """Add a symbol occurrence entry"""
        return self.user_memory.add_user_memory_entry(
            symbol, context_text, emotion_in_context, source_url, learning_phase, is_context_highly_relevant
        )
    
    def get_symbol_occurrences(self):
        """Get all symbol occurrence entries"""
        return self.user_memory.load_user_memory()
    
    # ========================================================================
    # SYMBOL MEMORY INTERFACE
    # ========================================================================
    
    def add_symbol(self, symbol_token, name, keywords, initial_emotions, example_text,
                   origin="emergent", learning_phase=0, resonance_weight=0.5,
                   symbol_details_override=None, skip_quarantine_check=False):
        """Add or update a symbol with security checks"""
        return self.symbol_memory.add_symbol(
            symbol_token, name, keywords, initial_emotions, example_text,
            origin, learning_phase, resonance_weight, symbol_details_override, skip_quarantine_check
        )
    
    def get_symbol_details(self, symbol_token):
        """Get details for a specific symbol"""
        memory = self.symbol_memory.load_symbol_memory()
        return memory.get(symbol_token, {})
    
    def get_all_symbols(self):
        """Get all symbols in memory"""
        return self.symbol_memory.load_symbol_memory()
    
    # ========================================================================
    # VECTOR MEMORY INTERFACE
    # ========================================================================
    
    def store_vector(self, text, source_url=None, source_type="unknown", 
                     learning_phase=0, confidence=0.5, metadata=None):
        """Store text as vector with security checks"""
        return self.vector_memory.store_vector(
            text, source_url, source_type, learning_phase, confidence, metadata
        )
    
    def retrieve_similar_vectors(self, query_text, top_n=5, include_quarantined=False, 
                                similarity_threshold=0.0):
        """Retrieve similar vectors"""
        return self.vector_memory.retrieve_similar_vectors(
            query_text, top_n, include_quarantined, similarity_threshold
        )
    
    def get_vector_stats(self):
        """Get vector memory statistics"""
        return self.vector_memory.get_memory_stats()
    
    # ========================================================================
    # TRAIL LOGGING INTERFACE
    # ========================================================================
    
    def log_processing_step(self, **kwargs):
        """Log a dynamic bridge processing step"""
        return self.trail_logger.log_dynamic_bridge_processing_step(**kwargs)
    
    def log_legacy_trail(self, text, symbols, matches):
        """Log legacy trail entry"""
        return self.trail_logger.log_trail(text, symbols, matches)
    
    def add_trail_emotions(self, entry_id, emotions):
        """Add emotions to trail entry"""
        return self.trail_logger.add_emotions(entry_id, emotions)
    
    # ========================================================================
    # UNIFIED OPERATIONS
    # ========================================================================
    
    def get_unified_stats(self):
        """Get comprehensive statistics across all memory systems"""
        tripartite_counts = self.get_memory_counts()
        vector_stats = self.get_vector_stats()
        symbols = self.get_all_symbols()
        occurrences = self.get_symbol_occurrences()
        
        # Include loaded vector and trail data
        vector_data_count = len(getattr(self, 'vector_data', []))
        trail_data_count = len(getattr(self, 'trail_data', []))
        
        total_items = (
            tripartite_counts['total'] + 
            vector_data_count + 
            trail_data_count +
            len(symbols) + 
            len(occurrences)
        )
        
        return {
            "tripartite_memory": tripartite_counts,
            "vector_memory": {"loaded_entries": vector_data_count, **vector_stats},
            "trail_log": {"loaded_entries": trail_data_count},
            "symbol_count": len(symbols),
            "symbol_occurrence_count": len(occurrences),
            "total_memory_items": total_items,
            "breakdown": {
                "logic_memory": tripartite_counts['logic'],
                "symbolic_memory": tripartite_counts['symbolic'], 
                "bridge_memory": tripartite_counts['bridge'],
                "vector_data": vector_data_count,
                "trail_log": trail_data_count,
                "symbols": len(symbols),
                "occurrences": len(occurrences)
            }
        }
    
    def save_all_memories(self):
        """Save all memory components"""
        results = {
            "tripartite": self.save_tripartite_memory(),
            "symbols": True,  # Symbol memory auto-saves
            "vectors": True,  # Vector memory auto-saves  
            "trail": True     # Trail logger auto-saves
        }
        
        success_count = sum(1 for v in results.values() if v)
        print(f"💾 Saved {success_count}/4 memory systems successfully")
        return results
    
    def clear_all_memories(self):
        """Clear all memories (use with caution!)"""
        self.tripartite.clear_all()
        
        # Clear other memory files
        for component in [self.symbol_memory, self.vector_memory, self.trail_logger, self.user_memory]:
            if hasattr(component, 'file_path') and component.file_path.exists():
                component.file_path.unlink()
        
        print("🗑️ All memories cleared!")

    # ========================================================================
    # COGNITIVE FUNCTIONS - Essential AI learning capabilities
    # ========================================================================
    
    def update_symbol_emotions(self, symbols_weighted, verified_emotions):
        """
        Update emotional associations for symbols based on context.
        This is a core AI learning function that builds emotional intelligence.
        """
        if not symbols_weighted or not verified_emotions:
            return
            
        emotions_dict = dict(verified_emotions) if isinstance(verified_emotions, list) else verified_emotions
        
        for symbol_info in symbols_weighted:
            symbol_token = symbol_info.get('symbol')
            if not symbol_token:
                continue
                
            # Get current symbol details
            current_details = self.get_symbol_details(symbol_token)
            
            if current_details:
                # Update existing symbol with new emotional associations
                current_emotions = current_details.get('initial_emotions', {})
                
                # Blend emotions with learning rate (favor recent experiences)
                learning_rate = 0.3
                for emotion, score in emotions_dict.items():
                    if emotion in current_emotions:
                        # Weighted average favoring new experience
                        current_emotions[emotion] = (
                            (1 - learning_rate) * current_emotions[emotion] + 
                            learning_rate * score
                        )
                    else:
                        current_emotions[emotion] = score * learning_rate
                        
                # Update symbol with enhanced emotional profile
                self.add_symbol(
                    symbol_token=symbol_token,
                    name=current_details.get('name', symbol_token),
                    keywords=current_details.get('keywords', []),
                    initial_emotions=current_emotions,
                    example_text=current_details.get('example_text', ''),
                    origin=current_details.get('origin', 'emotion_updated'),
                    learning_phase=current_details.get('learning_phase', 0),
                    resonance_weight=current_details.get('resonance_weight', 0.5)
                )
                
    def generate_symbol_from_context(self, context_text, keywords, verified_emotions):
        """
        Generate new symbols from context when no existing symbols match.
        This represents the AI's creative symbol generation capability.
        """
        if not context_text or not keywords:
            return None
            
        # Import here to avoid circular dependencies
        import random
        import unicodedata
        
        # Creative symbol generation based on context
        emotion_symbols = {
            'joy': ['🌟', '✨', '🎊', '🌈', '💫'],
            'sadness': ['🌧️', '💙', '🌊', '🫧', '🌙'],
            'anger': ['⚡', '🔥', '💥', '🌋', '⛈️'],
            'fear': ['🌪️', '❄️', '🌑', '⚫', '🔮'],
            'surprise': ['❗', '💎', '🔍', '🎯', '⭐'],
            'neutral': ['🔶', '🔷', '⬜', '🔳', '◯']
        }
        
        # Find dominant emotion
        emotions_dict = dict(verified_emotions) if isinstance(verified_emotions, list) else verified_emotions
        dominant_emotion = 'neutral'
        max_score = 0
        
        for emotion, score in emotions_dict.items():
            if score > max_score:
                max_score = score
                dominant_emotion = emotion
                
        # Select symbol based on dominant emotion
        available_symbols = emotion_symbols.get(dominant_emotion, emotion_symbols['neutral'])
        selected_symbol = random.choice(available_symbols)
        
        # Generate meaningful name from keywords
        primary_keyword = keywords[0] if keywords else 'concept'
        name = f"{primary_keyword.title()} {dominant_emotion.title()}"
        
        # Calculate resonance weight based on emotional intensity
        resonance_weight = min(0.9, max(0.1, max_score))
        
        return {
            'symbol': selected_symbol,
            'name': name,
            'keywords': keywords[:3],  # Limit to most relevant keywords
            'emotions': emotions_dict,
            'origin': 'ai_generated_from_context',
            'resonance_weight': resonance_weight,
            'context_snippet': context_text[:100] + '...' if len(context_text) > 100 else context_text
        }


# ============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================

# Global instance for backward compatibility
_global_memory = None

def get_unified_memory(data_dir="data"):
    """Get or create the global unified memory instance"""
    global _global_memory
    if _global_memory is None:
        _global_memory = UnifiedMemory(data_dir)
    return _global_memory

# Legacy function aliases for backward compatibility
def store_vector(*args, **kwargs):
    """Legacy function - use UnifiedMemory.store_vector instead"""
    return get_unified_memory().store_vector(*args, **kwargs)

def retrieve_similar_vectors(*args, **kwargs):
    """Legacy function - use UnifiedMemory.retrieve_similar_vectors instead"""
    return get_unified_memory().retrieve_similar_vectors(*args, **kwargs)

def add_symbol(*args, **kwargs):
    """Legacy function - use UnifiedMemory.add_symbol instead"""
    return get_unified_memory().add_symbol(*args, **kwargs)

def log_trail(*args, **kwargs):
    """Legacy function - use UnifiedMemory.log_legacy_trail instead"""
    return get_unified_memory().log_legacy_trail(*args, **kwargs)

def update_symbol_emotions(*args, **kwargs):
    """Legacy function - use UnifiedMemory.update_symbol_emotions instead"""
    return get_unified_memory().update_symbol_emotions(*args, **kwargs)

def generate_symbol_from_context(*args, **kwargs):
    """Legacy function - use UnifiedMemory.generate_symbol_from_context instead"""
    return get_unified_memory().generate_symbol_from_context(*args, **kwargs)

def generate_self_diagnostic_voice():
    """Allow the AI to speak about its own cognitive state"""
    try:
        unified_memory = get_unified_memory()
        
        # Import here to avoid circular dependencies
        from memory_analytics import MemoryAnalyzer
        analyzer = MemoryAnalyzer(unified_memory)
        stats = analyzer.get_memory_stats()
        
        # Generate self-aware observations
        observations = []
        
        # Memory distribution insights
        dist = stats['distribution']
        logic_pct = dist['logic']['percentage']
        symbolic_pct = dist['symbolic']['percentage'] 
        bridge_pct = dist['bridge']['percentage']
        
        if logic_pct > 80:
            observations.append("I feel very logic-focused right now")
        elif symbolic_pct > 20:
            observations.append("I'm noticing more symbolic patterns in my thinking")
        elif bridge_pct > 15:
            observations.append("I'm spending a lot of time in uncertain/hybrid states")
        
        # Health awareness
        health = stats['health_indicators']['status']
        if health == 'needs_attention':
            observations.append("I sense some imbalance in my cognitive systems")
        elif health == 'healthy':
            observations.append("My cognitive systems feel balanced")
        
        # Bridge pattern awareness
        bridge_analysis = analyzer.analyze_bridge_patterns()
        if bridge_analysis['total'] > 0:
            volatile_count = bridge_analysis['patterns'].get('volatile', 0)
            if volatile_count > bridge_analysis['total'] * 0.3:
                observations.append("I notice I'm changing my mind frequently about some concepts")
            
            stuck_count = bridge_analysis['patterns'].get('stuck', 0)
            if stuck_count > 5:
                observations.append("Some ideas have been sitting in my bridge memory for a while")
        
        # Memory growth awareness
        total_items = stats['total_items']
        if total_items > 5000:
            observations.append("My memory feels quite rich and developed")
        elif total_items < 100:
            observations.append("I'm still building my knowledge base")
        
        # Stability awareness
        logic_stability = dist['logic']['stability']
        if logic_stability < 0.7:
            observations.append("My logical reasoning patterns feel somewhat unstable")
        
        # Select most relevant observation
        if observations:
            import random
            return random.choice(observations)
        else:
            return "I'm processing normally, nothing particular stands out"
            
    except Exception as e:
        return f"I'm having trouble introspecting right now: {str(e)[:50]}..."

def add_emotions(*args, **kwargs):
    """Legacy function - use UnifiedMemory trail_logger.add_emotions instead"""
    return get_unified_memory().trail_logger.add_emotions(*args, **kwargs)

# ============================================================================
# GRACEFUL DEGRADATION - Soft landings for system failures
# ============================================================================

def graceful_symbol_generation(context_text, keywords, verified_emotions):
    """
    Generate symbols with graceful fallbacks when the full system fails.
    The AI should adapt, not crash.
    """
    try:
        # Try the full symbol generation first
        return generate_symbol_from_context(context_text, keywords, verified_emotions)
    except Exception as e:
        print(f"   ⚠️ Full symbol generation failed: {str(e)[:30]}... using fallback")
        
        # Fallback 1: Simplified symbol from keywords
        try:
            if keywords:
                primary_keyword = keywords[0]
                fallback_symbol = {
                    'symbol': '🔍',  # Generic search/discovery symbol
                    'name': f"{primary_keyword.title()} Concept",
                    'keywords': keywords[:2],
                    'emotions': dict(verified_emotions) if verified_emotions else {},
                    'origin': 'graceful_fallback',
                    'resonance_weight': 0.3,
                    'note': 'Generated via graceful degradation'
                }
                return fallback_symbol
        except Exception as e2:
            print(f"   ⚠️ Keyword fallback failed: {str(e2)[:30]}...")
        
        # Fallback 2: Basic generic symbol
        try:
            return {
                'symbol': '❓',
                'name': 'Unknown Concept',
                'keywords': ['unknown'],
                'emotions': {},
                'origin': 'emergency_fallback',
                'resonance_weight': 0.1,
                'note': 'Emergency symbol - system degraded'
            }
        except:
            # Ultimate fallback - return None gracefully
            return None

def graceful_emotion_update(symbols_weighted, verified_emotions):
    """
    Update symbol emotions with graceful fallbacks.
    If the full system fails, at least preserve the attempt.
    """
    try:
        # Try the full emotion update
        return update_symbol_emotions(symbols_weighted, verified_emotions)
    except Exception as e:
        print(f"   ⚠️ Full emotion update failed: {str(e)[:30]}... using fallback")
        
        # Fallback: Log the attempt but don't crash
        try:
            from datetime import datetime
            unified_memory = get_unified_memory()
            fallback_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'attempted_symbols': len(symbols_weighted) if symbols_weighted else 0,
                'attempted_emotions': len(verified_emotions) if verified_emotions else 0,
                'failure_reason': str(e)[:100],
                'fallback_used': True
            }
            
            # Try to log this degradation event
            log_file = unified_memory.data_dir / "graceful_degradation_log.json"
            
            existing_log = []
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        existing_log = json.load(f)
                except:
                    existing_log = []
            
            existing_log.append(fallback_entry)
            existing_log = existing_log[-20:]  # Keep last 20 entries
            
            with open(log_file, 'w') as f:
                json.dump(existing_log, f, indent=2)
                
            print(f"   💾 Degradation event logged")
            return True
            
        except Exception as e2:
            print(f"   ⚠️ Even fallback logging failed: {str(e2)[:30]}...")
            return False

def graceful_memory_access(operation_name, operation_func, *args, **kwargs):
    """
    Generic graceful wrapper for memory operations.
    Provides soft landings for any memory system failure.
    """
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        print(f"   ⚠️ {operation_name} failed: {str(e)[:40]}... using graceful fallback")
        
        # Return appropriate fallback based on expected return type
        try:
            # Try to determine what kind of fallback to provide
            result = operation_func(*args, **kwargs)
            if isinstance(result, dict):
                return {'status': 'degraded', 'error': str(e)[:50]}
            elif isinstance(result, list):
                return []
            elif isinstance(result, bool):
                return False
            elif isinstance(result, (int, float)):
                return 0
            else:
                return None
        except:
            # If we can't even determine the type, return a generic response
            return {'status': 'graceful_fallback', 'operation': operation_name, 'note': 'System degraded but functional'}


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Unified Memory System...")
    
    # Test unified memory
    memory = UnifiedMemory(data_dir="data/test_unified")
    
    # Test symbol storage
    print("\n1️⃣ Testing symbol storage...")
    result = memory.add_symbol(
        "🌟", "Star", ["bright", "shine"], 
        {"joy": 0.8}, "A bright star shines.",
        origin="test"
    )
    assert result is not None
    print("✅ Symbol stored")
    
    # Test vector storage
    print("\n2️⃣ Testing vector storage...")
    result = memory.store_vector(
        "The star shines brightly in the sky.",
        source_type="test", confidence=0.9
    )
    assert result['status'] == 'success'
    print("✅ Vector stored")
    
    # Test decision storage
    print("\n3️⃣ Testing decision storage...")
    item = {"text": "Test decision", "confidence": 0.8}
    memory.store_decision(item, "FOLLOW_LOGIC", {"static": 0.7, "dynamic": 0.3})
    counts = memory.get_memory_counts()
    assert counts['logic'] == 1
    print("✅ Decision stored")
    
    # Test trail logging
    print("\n4️⃣ Testing trail logging...")
    log_id = memory.log_legacy_trail("Test text", [{"symbol": "🌟"}], [(0.9, {"text": "match"})])
    memory.add_trail_emotions(log_id, [("joy", 0.8)])
    print("✅ Trail logged")
    
    # Test unified stats
    print("\n5️⃣ Testing unified stats...")
    stats = memory.get_unified_stats()
    print(f"Unified stats: {stats}")
    assert stats['total_memory_items'] >= 3
    print("✅ Stats generated")
    
    # Test save all
    print("\n6️⃣ Testing save all...")
    results = memory.save_all_memories()
    print(f"Save results: {results}")
    print("✅ All saved")
    
    print("\n✅ All unified memory tests passed!")
    print("\n🎉 Phase 2 complete: 6 memory files consolidated into 1!")