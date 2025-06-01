# symbol_memory.py - Updated with Quarantine, Warfare Detection, and Visualization Integration
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# File path for symbol memory
SYMBOL_MEMORY_PATH = Path("data/symbol_memory.json")

# Import the new modules for integration
try:
    from quarantine_layer import UserMemoryQuarantine
    from linguistic_warfare import LinguisticWarfareDetector
    from visualization_prep import VisualizationPrep
    SECURITY_MODULES_LOADED = True
except ImportError:
    SECURITY_MODULES_LOADED = False
    print("[SYMBOL_MEMORY-WARNING] Security modules not loaded. Operating without quarantine/warfare protection.")

def load_symbol_memory(file_path=SYMBOL_MEMORY_PATH):
    """Loads existing symbol memory, ensuring it's a dictionary of symbol objects."""
    current_path = Path(file_path) if isinstance(file_path, str) else file_path
    
    current_path.parent.mkdir(parents=True, exist_ok=True)
    if current_path.exists() and current_path.stat().st_size > 0:
        with open(current_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    return {
                        token: details
                        for token, details in data.items()
                        if isinstance(details, dict) and "name" in details # Basic check
                    }
                else:
                    print(f"[SYMBOL_MEMORY-WARNING] Symbol memory file {current_path} is not a dictionary. Returning empty memory.")
                    return {}
            except json.JSONDecodeError:
                print(f"[SYMBOL_MEMORY-WARNING] Symbol memory file {current_path} corrupted. Returning empty memory.")
                return {}
    return {}

def save_symbol_memory(memory, file_path=SYMBOL_MEMORY_PATH):
    """Saves current state of symbol memory to disk."""
    current_path = Path(file_path) if isinstance(file_path, str) else file_path
    current_path.parent.mkdir(parents=True, exist_ok=True)
    with open(current_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def _check_quarantine_status(origin: str, example_text: str = None, 
                           name: str = None, keywords: List[str] = None) -> Tuple[bool, Optional[Dict]]:
    """
    Check if a symbol should be quarantined based on origin and content.
    Returns (should_quarantine, warfare_analysis)
    """
    # Direct quarantine origins
    quarantine_origins = [
        "user_quarantine",
        "quarantined_input",
        "warfare_detected",
        "manipulation_attempt",
        "unverified_user"
    ]
    
    if origin in quarantine_origins:
        return True, {"reason": f"Quarantine origin: {origin}"}
    
    # Check for linguistic warfare if modules are loaded
    if SECURITY_MODULES_LOADED and example_text:
        detector = LinguisticWarfareDetector()
        # Combine all text for analysis
        analysis_text = f"{name or ''} {' '.join(keywords or [])} {example_text or ''}"
        analysis = detector.analyze_text_for_warfare(analysis_text, user_id="symbol_creation")
        
        if analysis['threat_score'] > 0.7:
            return True, analysis
    
    return False, None

def _sanitize_symbol_data(symbol_data: Dict) -> Dict:
    """
    Sanitize symbol data to prevent malicious content.
    """
    # Limit string lengths
    max_lengths = {
        'name': 100,
        'keywords': 50,  # per keyword
        'example_text': 500,
        'core_meanings': 200  # per meaning
    }
    
    if 'name' in symbol_data:
        symbol_data['name'] = symbol_data['name'][:max_lengths['name']]
    
    if 'keywords' in symbol_data and isinstance(symbol_data['keywords'], list):
        symbol_data['keywords'] = [
            kw[:max_lengths['keywords']] for kw in symbol_data['keywords'][:20]  # Max 20 keywords
        ]
    
    if 'core_meanings' in symbol_data and isinstance(symbol_data['core_meanings'], list):
        symbol_data['core_meanings'] = [
            meaning[:max_lengths['core_meanings']] for meaning in symbol_data['core_meanings'][:10]
        ]
    
    # Remove any potential script injection attempts
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return str(text)
        # Remove script tags and other potentially harmful content
        dangerous_patterns = ['<script', '</script>', 'javascript:', 'onerror=', 'onclick=']
        cleaned = text
        for pattern in dangerous_patterns:
            cleaned = cleaned.replace(pattern, '')
        return cleaned
    
    # Clean all text fields
    for field in ['name', 'summary', 'description']:
        if field in symbol_data and isinstance(symbol_data[field], str):
            symbol_data[field] = clean_text(symbol_data[field])
    
    return symbol_data

def add_symbol(symbol_token, name, keywords, initial_emotions, example_text,
               origin="emergent", learning_phase=0, resonance_weight=0.5,
               symbol_details_override=None, skip_quarantine_check=False,
               file_path=SYMBOL_MEMORY_PATH):
    """
    Adds a new symbol or updates an existing one in the symbol_memory.json.
    
    NEW PARAMETERS:
    - skip_quarantine_check: If True, bypasses quarantine checks (use carefully!)
    
    Initial_emotions can be a list of emotion strings, a dict {'emotion': score},
    a list of tuples [('emotion', score)], or list of dicts [{"emotion":"name", "weight":0.8}].
    """
    
    # Check quarantine status unless explicitly skipped
    if not skip_quarantine_check:
        should_quarantine, warfare_analysis = _check_quarantine_status(
            origin, example_text, name, keywords
        )
        
        if should_quarantine:
            print(f"[SYMBOL_MEMORY-QUARANTINE] Symbol '{symbol_token}' ({name}) blocked - origin: {origin}")
            if warfare_analysis and 'threats' in warfare_analysis:
                print(f"  Threats detected: {[t['type'] for t in warfare_analysis['threats']]}")
            
            # Log to quarantine if available
            if SECURITY_MODULES_LOADED:
                quarantine = UserMemoryQuarantine()
                quarantine.quarantine_user_input(
                    text=f"Symbol creation attempt: {symbol_token} - {name}",
                    user_id="symbol_memory",
                    matched_symbols=[{'symbol': symbol_token, 'name': name}],
                    source_url=f"symbol_memory:{origin}"
                )
            
            return None  # Don't add to core memory
    
    memory = load_symbol_memory(file_path)
    
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
            if isinstance(item, dict) and "weight" in item and isinstance(item["weight"], (int, float)):
                incoming_numeric_weights.append(item["weight"])
                if "emotion" in item:
                    peak_emotions_from_initial[item["emotion"]] = item["weight"]
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
                incoming_numeric_weights.append(item[1])
                peak_emotions_from_initial[item[0]] = item[1]

    current_max_incoming_weight = max(incoming_numeric_weights, default=0.0)
    
    # Filter peak_emotions to only include those at the peak weight
    peak_emotions_at_max = {
        emo: weight 
        for emo, weight in peak_emotions_from_initial.items() 
        if weight == current_max_incoming_weight
    } if current_max_incoming_weight > 0 else {}

    if symbol_token not in memory:
        # New symbol creation
        if symbol_details_override and isinstance(symbol_details_override, dict):
            # Sanitize override data
            symbol_details_override = _sanitize_symbol_data(symbol_details_override.copy())
            
            memory[symbol_token] = symbol_details_override.copy()
            # Initialize critical fields that might be missing
            memory[symbol_token].setdefault("vector_examples", [])
            memory[symbol_token].setdefault("usage_count", 0)
            memory[symbol_token].setdefault("golden_memory", {
                "peak_weight": current_max_incoming_weight,
                "context": example_text or memory[symbol_token].get("summary", ""),
                "peak_emotions": peak_emotions_at_max,
                "timestamp": datetime.utcnow().isoformat()
            })
            # Set other fields from parameters if not in override
            memory[symbol_token].setdefault("name", name)
            memory[symbol_token].setdefault("keywords", list(set(keywords)))
            memory[symbol_token].setdefault("emotions", initial_emotions) 
            memory[symbol_token].setdefault("emotion_profile", {})
            memory[symbol_token].setdefault("origin", origin) 
            memory[symbol_token].setdefault("learning_phase", learning_phase) 
            memory[symbol_token].setdefault("resonance_weight", resonance_weight) 
            memory[symbol_token].setdefault("created_at", datetime.utcnow().isoformat())
            memory[symbol_token].setdefault("updated_at", datetime.utcnow().isoformat())
            memory[symbol_token].setdefault("core_meanings", [])
            
            # Add visualization metadata
            memory[symbol_token].setdefault("visualization_metadata", {
                "primary_color": _get_symbol_color(initial_emotions),
                "display_priority": _calculate_display_priority(resonance_weight, origin),
                "classification_hint": _get_classification_hint(keywords, initial_emotions)
            })
            
        else:
            # Creating a brand new symbol without override
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
                    "peak_emotions": peak_emotions_at_max,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "usage_count": 0,
                "visualization_metadata": {
                    "primary_color": _get_symbol_color(initial_emotions),
                    "display_priority": _calculate_display_priority(resonance_weight, origin),
                    "classification_hint": _get_classification_hint(keywords, initial_emotions)
                }
            }
            
            # Sanitize new symbol data
            memory[symbol_token] = _sanitize_symbol_data(new_symbol_data)
        
        # Add the example_text if provided
        if example_text:
            max_example_len = 300
            example_to_store = example_text[:max_example_len] + "..." if len(example_text) > max_example_len else example_text
            
            memory[symbol_token]["vector_examples"].append({
                "text": example_to_store,
                "timestamp": datetime.utcnow().isoformat(),
                "source_phase": learning_phase
            })
            
            if memory[symbol_token].get("usage_count", 0) == 0:
                memory[symbol_token]["usage_count"] = 1

    else:
        # Symbol exists, update it
        memory[symbol_token]["updated_at"] = datetime.utcnow().isoformat()
        memory[symbol_token]["usage_count"] = memory[symbol_token].get("usage_count", 0) + 1
        
        # Update keywords
        if keywords:
            existing_kws = set(memory[symbol_token].get("keywords", []))
            for kw in keywords:
                existing_kws.add(kw)
            memory[symbol_token]["keywords"] = sorted(list(existing_kws))[:50]  # Limit keywords

        # Add new example
        if example_text:
            max_example_len = 300
            example_to_store = example_text[:max_example_len] + "..." if len(example_text) > max_example_len else example_text
            current_examples = memory[symbol_token].get("vector_examples", [])
            
            # Limit to 10 examples
            if len(current_examples) > 10:
                current_examples.pop(0)
            
            # Check for duplicates in recent examples
            is_duplicate_example = False
            for ex_entry in current_examples[-3:]:
                if isinstance(ex_entry, dict) and ex_entry.get("text") == example_to_store:
                    is_duplicate_example = True
                    break
            
            if not is_duplicate_example:
                current_examples.append({
                    "text": example_to_store,
                    "timestamp": datetime.utcnow().isoformat(),
                    "source_phase": learning_phase
                })
            memory[symbol_token]["vector_examples"] = current_examples
        
        # Update golden memory if this is a new peak
        if "golden_memory" not in memory[symbol_token]:
            memory[symbol_token]["golden_memory"] = {
                "peak_weight": 0.0, 
                "context": "", 
                "peak_emotions": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
        if incoming_numeric_weights:
            current_peak = memory[symbol_token]["golden_memory"].get("peak_weight", 0.0)
            if current_max_incoming_weight > current_peak:
                memory[symbol_token]["golden_memory"] = {
                    "peak_weight": current_max_incoming_weight,
                    "context": example_text or memory[symbol_token]["golden_memory"].get("context", ""),
                    "peak_emotions": peak_emotions_at_max,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        # Update visualization metadata
        memory[symbol_token]["visualization_metadata"] = {
            "primary_color": _get_symbol_color(memory[symbol_token].get("emotion_profile", {})),
            "display_priority": _calculate_display_priority(
                memory[symbol_token].get("resonance_weight", 0.5),
                memory[symbol_token].get("origin", "unknown")
            ),
            "classification_hint": _get_classification_hint(
                memory[symbol_token].get("keywords", []),
                memory[symbol_token].get("emotion_profile", {})
            )
        }
        
    save_symbol_memory(memory, file_path)
    return memory[symbol_token]

def get_symbol_details(symbol_token, file_path=SYMBOL_MEMORY_PATH):
    """Gets details for a specific symbol."""
    memory = load_symbol_memory(file_path)
    return memory.get(symbol_token, {})

def update_symbol_emotional_profile(symbol_token, emotion_changes, file_path=SYMBOL_MEMORY_PATH):
    """Updates the cumulative emotional profile of a symbol."""
    memory = load_symbol_memory(file_path)
    if symbol_token in memory:
        profile = memory[symbol_token].get("emotion_profile", {})
        for emotion, change in emotion_changes.items():
            profile[emotion] = profile.get(emotion, 0) + change
            profile[emotion] = round(max(0, min(1, profile[emotion])), 3)
        memory[symbol_token]["emotion_profile"] = profile
        memory[symbol_token]["updated_at"] = datetime.utcnow().isoformat()
        
        # Update visualization metadata
        memory[symbol_token]["visualization_metadata"]["primary_color"] = _get_symbol_color(profile)
        
        save_symbol_memory(memory, file_path)
        return True
    return False

def prune_duplicates(file_path=SYMBOL_MEMORY_PATH):
    """Removes duplicate examples from symbol vector_examples."""
    memory = load_symbol_memory(file_path)
    changed = False
    for symbol_token in memory:
        if "vector_examples" in memory[symbol_token] and isinstance(memory[symbol_token]["vector_examples"], list):
            unique_examples = []
            seen_texts = set()
            original_len = len(memory[symbol_token]["vector_examples"])
            for example_entry in memory[symbol_token]["vector_examples"]:
                if isinstance(example_entry, dict) and "text" in example_entry:
                    example_text_val = example_entry["text"]
                    if example_text_val not in seen_texts:
                        unique_examples.append(example_entry)
                        seen_texts.add(example_text_val)
                elif isinstance(example_entry, str): 
                    if example_entry not in seen_texts:
                        unique_examples.append({"text": example_entry, "timestamp": "unknown", "source_phase": 0})
                        seen_texts.add(example_entry)
            if len(unique_examples) < original_len:
                changed = True
            memory[symbol_token]["vector_examples"] = unique_examples
    if changed:
        print(f"[SYMBOL_MEMORY-INFO] Pruned duplicate examples.")
        save_symbol_memory(memory, file_path)

def get_emotion_profile(symbol_token, file_path=SYMBOL_MEMORY_PATH):
    """Gets the cumulative emotion profile for a symbol."""
    memory = load_symbol_memory(file_path)
    symbol_data = memory.get(symbol_token)
    if symbol_data and isinstance(symbol_data, dict) and "emotion_profile" in symbol_data:
        return symbol_data["emotion_profile"]
    return {}

def get_golden_memory(symbol_token, file_path=SYMBOL_MEMORY_PATH):
    """Gets the golden memory (peak state) for a symbol."""
    memory = load_symbol_memory(file_path)
    symbol_data = memory.get(symbol_token)
    if symbol_data and isinstance(symbol_data, dict) and "golden_memory" in symbol_data:
        return symbol_data["golden_memory"]
    return None

# New functions for visualization support
def _get_symbol_color(emotions: Any) -> str:
    """
    Determine a color for the symbol based on its emotional profile.
    Returns hex color code.
    """
    # Default color
    default_color = "#808080"  # Gray
    
    # Color mapping for emotions
    emotion_colors = {
        "joy": "#FFD700",      # Gold
        "love": "#FF69B4",     # Hot Pink
        "anger": "#DC143C",    # Crimson
        "fear": "#4B0082",     # Indigo
        "sadness": "#4682B4",  # Steel Blue
        "surprise": "#FF8C00", # Dark Orange
        "disgust": "#228B22",  # Forest Green
        "trust": "#87CEEB",    # Sky Blue
        "anticipation": "#DA70D6", # Orchid
        "curiosity": "#9370DB",    # Medium Purple
        "hope": "#98FB98",     # Pale Green
        "pride": "#FFB6C1",    # Light Pink
        "shame": "#8B4513",    # Saddle Brown
        "guilt": "#A0522D",    # Sienna
        "envy": "#2E8B57",     # Sea Green
        "gratitude": "#F0E68C", # Khaki
        "confusion": "#D3D3D3", # Light Gray
        "neutral": "#C0C0C0"   # Silver
    }
    
    if not emotions:
        return default_color
        
    # Handle different emotion formats
    if isinstance(emotions, dict):
        # Find dominant emotion
        if emotions:
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            return emotion_colors.get(dominant_emotion.lower(), default_color)
    elif isinstance(emotions, list) and emotions:
        # Handle list of tuples or dicts
        if isinstance(emotions[0], tuple):
            dominant_emotion = emotions[0][0]
        elif isinstance(emotions[0], dict) and 'emotion' in emotions[0]:
            dominant_emotion = emotions[0]['emotion']
        else:
            return default_color
        return emotion_colors.get(dominant_emotion.lower(), default_color)
    
    return default_color

def _calculate_display_priority(resonance_weight: float, origin: str) -> float:
    """
    Calculate display priority for visualization.
    Higher values = more prominent display.
    """
    base_priority = resonance_weight
    
    # Boost priority based on origin
    origin_boosts = {
        "seed": 0.3,
        "meta_emergent": 0.2,
        "meta_analysis": 0.2,
        "generated": 0.1,
        "emergent": 0.1,
        "user_direct_input": -0.1,  # Lower priority for user input
        "user_quarantine": -0.5     # Very low priority
    }
    
    boost = origin_boosts.get(origin, 0.0)
    return round(min(1.0, max(0.0, base_priority + boost)), 3)

def _get_classification_hint(keywords: List[str], emotions: Any) -> str:
    """
    Provide a hint about whether this symbol is more logic or symbolic oriented.
    """
    # Logic-oriented keywords
    logic_keywords = {"algorithm", "data", "system", "process", "compute", "logic", 
                     "structure", "function", "binary", "code", "network", "protocol"}
    
    # Symbolic-oriented keywords  
    symbolic_keywords = {"emotion", "feel", "symbol", "meaning", "soul", "spirit",
                        "dream", "myth", "metaphor", "archetype", "story", "journey"}
    
    # Count matches
    logic_score = sum(1 for kw in keywords if any(lk in kw.lower() for lk in logic_keywords))
    symbolic_score = sum(1 for kw in keywords if any(sk in kw.lower() for sk in symbolic_keywords))
    
    # Consider emotions
    if emotions:
        if isinstance(emotions, dict) and len(emotions) > 2:
            symbolic_score += 2
        elif isinstance(emotions, list) and len(emotions) > 2:
            symbolic_score += 2
    
    if logic_score > symbolic_score * 1.5:
        return "logic"
    elif symbolic_score > logic_score * 1.5:
        return "symbolic"
    else:
        return "hybrid"

# New security-aware functions
def validate_symbol_token(symbol_token: str) -> bool:
    """
    Validate that a symbol token is safe and reasonable.
    """
    if not symbol_token or not isinstance(symbol_token, str):
        return False
    
    # Length check
    if len(symbol_token) > 50:
        return False
    
    # Check for dangerous patterns
    dangerous_patterns = ['<script', 'javascript:', 'onerror', '../', '\\', '\x00']
    for pattern in dangerous_patterns:
        if pattern in symbol_token.lower():
            return False
    
    return True

def get_symbols_for_visualization(limit: int = 100, 
                                 min_usage: int = 0,
                                 exclude_quarantined: bool = True,
                                 file_path=SYMBOL_MEMORY_PATH) -> List[Dict]:
    """
    Get symbols formatted for visualization.
    """
    memory = load_symbol_memory(file_path)
    symbols_for_viz = []
    
    for token, details in memory.items():
        # Skip if usage is too low
        if details.get("usage_count", 0) < min_usage:
            continue
            
        # Skip quarantined origins if requested
        if exclude_quarantined and details.get("origin") in ["user_quarantine", "quarantined_input"]:
            continue
        
        viz_data = {
            "token": token,
            "name": details.get("name", token),
            "usage_count": details.get("usage_count", 0),
            "resonance_weight": details.get("resonance_weight", 0.5),
            "emotion_profile": details.get("emotion_profile", {}),
            "golden_memory": details.get("golden_memory", {}),
            "visualization_metadata": details.get("visualization_metadata", {}),
            "origin": details.get("origin", "unknown"),
            "learning_phase": details.get("learning_phase", 0),
            "keywords": details.get("keywords", [])[:5]  # Limit keywords for display
        }
        
        symbols_for_viz.append(viz_data)
    
    # Sort by display priority and usage
    symbols_for_viz.sort(
        key=lambda x: (
            x["visualization_metadata"].get("display_priority", 0.5),
            x["usage_count"]
        ),
        reverse=True
    )
    
    return symbols_for_viz[:limit]

def quarantine_existing_symbol(symbol_token: str, reason: str = "manual_quarantine", 
                              file_path=SYMBOL_MEMORY_PATH) -> bool:
    """
    Move an existing symbol to quarantine status.
    """
    memory = load_symbol_memory(file_path)
    
    if symbol_token not in memory:
        return False
    
    # Update the symbol's origin to indicate quarantine
    memory[symbol_token]["origin"] = "quarantined_" + memory[symbol_token].get("origin", "unknown")
    memory[symbol_token]["quarantine_reason"] = reason
    memory[symbol_token]["quarantined_at"] = datetime.utcnow().isoformat()
    memory[symbol_token]["resonance_weight"] = 0.0  # Zero out resonance
    
    # Update visualization metadata
    memory[symbol_token]["visualization_metadata"]["display_priority"] = 0.0
    memory[symbol_token]["visualization_metadata"]["quarantined"] = True
    
    save_symbol_memory(memory, file_path)
    
    print(f"[SYMBOL_MEMORY-QUARANTINE] Symbol '{symbol_token}' quarantined: {reason}")
    return True


if __name__ == '__main__':
    print("Testing symbol_memory.py with security integrations...")
    
    test_path = Path("data/test_symbol_memory_secure.json")
    if test_path.exists(): 
        test_path.unlink()
    
    # Test 1: Normal symbol addition
    print("\n--- Test 1: Normal symbol addition ---")
    result = add_symbol(
        "🌟", "Star", ["shine", "bright"], 
        [{"emotion": "joy", "weight": 0.8}], 
        "A bright star shines with joy.",
        origin="test", learning_phase=1, file_path=test_path
    )
    assert result is not None
    print("✅ Normal symbol added successfully")
    
    # Test 2: Quarantined origin
    print("\n--- Test 2: Quarantine origin blocking ---")
    result = add_symbol(
        "💀", "Danger", ["harm", "threat"], 
        [{"emotion": "fear", "weight": 0.9}], 
        "This is dangerous content.",
        origin="user_quarantine", learning_phase=1, file_path=test_path
    )
    assert result is None
    print("✅ Quarantined origin blocked successfully")
    
    # Test 3: Skip quarantine check
    print("\n--- Test 3: Skip quarantine check ---")
    result = add_symbol(
        "🔒", "Lock", ["secure", "protect"], 
        [{"emotion": "trust", "weight": 0.7}], 
        "Security bypass for system symbols.",
        origin="user_quarantine", learning_phase=1, 
        skip_quarantine_check=True, file_path=test_path
    )
    assert result is not None
    print("✅ Quarantine check bypassed successfully")
    
    # Test 4: Visualization metadata
    print("\n--- Test 4: Visualization metadata ---")
    star_details = get_symbol_details("🌟", file_path=test_path)
    assert "visualization_metadata" in star_details
    assert "primary_color" in star_details["visualization_metadata"]
    print(f"✅ Visualization metadata: {star_details['visualization_metadata']}")
    
    # Test 5: Get symbols for visualization
    print("\n--- Test 5: Get symbols for visualization ---")
    viz_symbols = get_symbols_for_visualization(file_path=test_path)
    assert len(viz_symbols) == 2  # Should exclude the quarantined one
    print(f"✅ Got {len(viz_symbols)} symbols for visualization")
    
    # Test 6: Quarantine existing symbol
    print("\n--- Test 6: Quarantine existing symbol ---")
    success = quarantine_existing_symbol("🌟", "test_quarantine", file_path=test_path)
    assert success == True
    star_details = get_symbol_details("🌟", file_path=test_path)
    assert star_details["resonance_weight"] == 0.0
    print("✅ Existing symbol quarantined successfully")
    
    # Test 7: Validate symbol token
    print("\n--- Test 7: Symbol token validation ---")
    assert validate_symbol_token("🌟") == True
    assert validate_symbol_token("<script>alert('hi')</script>") == False
    assert validate_symbol_token("a" * 100) == False
    print("✅ Symbol token validation working")
    
    print("\n✅ All security integration tests passed!")