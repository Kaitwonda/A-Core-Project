# symbol_memory.py
from pathlib import Path
import json
from datetime import datetime

# File path for symbol memory
SYMBOL_MEMORY_PATH = Path("data/symbol_memory.json")

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

def add_symbol(symbol_token, name, keywords, initial_emotions, example_text,
               origin="emergent", learning_phase=0, resonance_weight=0.5,
               symbol_details_override=None,
               file_path=SYMBOL_MEMORY_PATH):
    """
    Adds a new symbol or updates an existing one in the symbol_memory.json.
    Initial_emotions can be a list of emotion strings, a dict {'emotion': score},
    a list of tuples [('emotion', score)], or list of dicts [{"emotion":"name", "weight":0.8}].
    """
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
            
        else:
            # Creating a brand new symbol without override
            memory[symbol_token] = {
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
                "usage_count": 0
            }
        
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
            memory[symbol_token]["keywords"] = sorted(list(existing_kws))

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


if __name__ == '__main__':
    print("Testing symbol_memory.py with complete Golden Memory feature...")
    
    test_path = Path("data/test_symbol_memory_golden_complete.json")
    if test_path.exists(): 
        test_path.unlink()
    
    # Test 1: New symbol with initial emotions
    print("\n--- Test 1: New symbol with emotions ---")
    initial_emotions_t1 = [
        {"emotion": "joy", "weight": 0.8},
        {"emotion": "curiosity", "weight": 0.8},  # Same peak weight
        {"emotion": "fear", "weight": 0.3}
    ]
    
    result = add_symbol(
        "🌟", "Star", ["shine", "bright"], 
        initial_emotions_t1, 
        "A bright star shines with joy.",
        origin="test", learning_phase=1, file_path=test_path
    )
    
    golden = result["golden_memory"]
    print(f"Golden memory: {json.dumps(golden, indent=2)}")
    assert golden["peak_weight"] == 0.8
    assert "joy" in golden["peak_emotions"] and golden["peak_emotions"]["joy"] == 0.8
    assert "curiosity" in golden["peak_emotions"] and golden["peak_emotions"]["curiosity"] == 0.8
    assert "fear" not in golden["peak_emotions"]  # Not at peak
    assert "timestamp" in golden
    assert golden["context"] == "A bright star shines with joy."
    
    # Test 2: Update with higher peak
    print("\n--- Test 2: Update with higher peak ---")
    update_emotions = [("excitement", 0.95), ("joy", 0.95)]
    
    add_symbol(
        "🌟", "Star", ["glow"], 
        update_emotions,
        "The star glows with incredible excitement!",
        file_path=test_path
    )
    
    updated_details = get_symbol_details("🌟", file_path=test_path)
    golden = updated_details["golden_memory"]
    print(f"Updated golden memory: {json.dumps(golden, indent=2)}")
    assert golden["peak_weight"] == 0.95
    assert golden["peak_emotions"] == {"excitement": 0.95, "joy": 0.95}
    assert golden["context"] == "The star glows with incredible excitement!"
    
    # Test 3: Update with lower peak (should not change golden)
    print("\n--- Test 3: Update with lower peak (no change expected) ---")
    old_timestamp = golden["timestamp"]
    
    add_symbol(
        "🌟", "Star", ["dim"], 
        [("sadness", 0.6)],
        "The star dims sadly.",
        file_path=test_path
    )
    
    unchanged_details = get_symbol_details("🌟", file_path=test_path)
    golden = unchanged_details["golden_memory"]
    print(f"Golden memory after lower update: {json.dumps(golden, indent=2)}")
    assert golden["peak_weight"] == 0.95  # Unchanged
    assert golden["context"] == "The star glows with incredible excitement!"  # Unchanged
    
    # Test 4: Symbol with override
    print("\n--- Test 4: Symbol with override preserving golden memory ---")
    override = {
        "name": "Moon",
        "summary": "Celestial body",
        "keywords": ["lunar", "night"]
    }
    
    add_symbol(
        "🌙", "Moon Override Name", ["moon"], 
        [{"emotion": "tranquility", "weight": 0.7}],
        "The moon brings tranquility.",
        symbol_details_override=override,
        file_path=test_path
    )
    
    moon_details = get_symbol_details("🌙", file_path=test_path)
    assert "golden_memory" in moon_details
    assert moon_details["golden_memory"]["peak_weight"] == 0.7
    assert moon_details["golden_memory"]["peak_emotions"] == {"tranquility": 0.7}
    
    # Test 5: Retrieve golden memory
    print("\n--- Test 5: Get golden memory helper ---")
    star_golden = get_golden_memory("🌟", file_path=test_path)
    print(f"Retrieved golden memory: {json.dumps(star_golden, indent=2)}")
    assert star_golden is not None
    assert star_golden["peak_weight"] == 0.95
    
    print("\n✅ All golden memory tests passed!")