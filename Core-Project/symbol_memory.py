# symbol_memory.py
from pathlib import Path
import json
from datetime import datetime # Added for timestamp in add_symbol if needed later

# File path for symbol memory
SYMBOL_MEMORY_PATH = Path("data/symbol_memory.json")

# APPENDED/MODIFIED: load_symbol_memory function
def load_symbol_memory(file_path=SYMBOL_MEMORY_PATH): # Added file_path parameter
    """Loads existing symbol memory, ensuring it's a dictionary of symbol objects."""
    # Ensure file_path is a Path object if a string is passed
    current_path = Path(file_path) if isinstance(file_path, str) else file_path
    
    current_path.parent.mkdir(parents=True, exist_ok=True)
    if current_path.exists() and current_path.stat().st_size > 0: # Check size
        with open(current_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    # Filter out any top-level keys that are not symbol details dicts
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
    # print(f"[SYMBOL_MEMORY-INFO] Symbol memory file {current_path} not found or empty. Returning empty memory.")
    return {}

# MODIFIED: save_symbol_memory to ensure it uses its parameter
def save_symbol_memory(memory, file_path=SYMBOL_MEMORY_PATH): # file_path already a parameter
    """Saves current state of symbol memory to disk."""
    current_path = Path(file_path) if isinstance(file_path, str) else file_path
    current_path.parent.mkdir(parents=True, exist_ok=True)
    with open(current_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

# MODIFIED: add_symbol to ensure it uses its file_path parameter for load/save
def add_symbol(symbol_token, name, keywords, initial_emotions, example_text, 
               origin="emergent", learning_phase=0, resonance_weight=0.5,
               symbol_details_override=None, # For meta-symbols primarily
               file_path=SYMBOL_MEMORY_PATH):
    """
    Adds a new symbol or updates an existing one in the symbol_memory.json.
    Initial_emotions can be a list of emotion strings or a dict like {'emotion': score}.
    """
    memory = load_symbol_memory(file_path) # Use parameter
    
    if symbol_token not in memory:
        if symbol_details_override and isinstance(symbol_details_override, dict):
            memory[symbol_token] = symbol_details_override
            if "created_at" not in memory[symbol_token]: # Ensure timestamp if overridden
                 memory[symbol_token]["created_at"] = datetime.utcnow().isoformat()
            if "updated_at" not in memory[symbol_token]:
                 memory[symbol_token]["updated_at"] = datetime.utcnow().isoformat()
        else:
            memory[symbol_token] = {
                "name": name,
                "keywords": list(set(keywords)), # Ensure unique keywords
                "core_meanings": [], # To be populated or refined
                "emotions": initial_emotions if isinstance(initial_emotions, list) else [], # Store as list
                "emotion_profile": {}, # Aggregated emotional weights over time
                "vector_examples": [], # List of text snippets where this symbol appeared
                "origin": origin, # "seed", "emergent", "meta_emergent"
                "learning_phase": learning_phase, # Phase when created/significantly updated
                "resonance_weight": resonance_weight, # How strongly this symbol "pulls" attention
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "usage_count": 0
            }
        if example_text: # Add initial example if provided
             # Ensure example text is not overly long for storage
            max_example_len = 300 
            example_to_store = example_text[:max_example_len] + "..." if len(example_text) > max_example_len else example_text
            memory[symbol_token]["vector_examples"].append({
                "text": example_to_store, 
                "timestamp": datetime.utcnow().isoformat(),
                "source_phase": learning_phase # Phase in which this example was encountered
            })
            memory[symbol_token]["usage_count"] = 1

    else: # Symbol exists, update it
        memory[symbol_token]["updated_at"] = datetime.utcnow().isoformat()
        memory[symbol_token]["usage_count"] = memory[symbol_token].get("usage_count", 0) + 1
        
        if keywords: # Add new unique keywords
            existing_kws = set(memory[symbol_token].get("keywords", []))
            for kw in keywords: existing_kws.add(kw)
            memory[symbol_token]["keywords"] = sorted(list(existing_kws))

        if example_text:
            max_example_len = 300
            example_to_store = example_text[:max_example_len] + "..." if len(example_text) > max_example_len else example_text
            # Avoid too many examples, keep maybe last 5-10 or unique ones
            if len(memory[symbol_token]["vector_examples"]) > 10:
                memory[symbol_token]["vector_examples"].pop(0) # Remove oldest
            
            # Add if not a recent duplicate
            is_duplicate_example = False
            for ex_entry in memory[symbol_token]["vector_examples"][-3:]: # Check last 3
                if ex_entry["text"] == example_to_store:
                    is_duplicate_example = True
                    break
            if not is_duplicate_example:
                 memory[symbol_token]["vector_examples"].append({
                    "text": example_to_store, 
                    "timestamp": datetime.utcnow().isoformat(),
                    "source_phase": learning_phase
                })


        # Emotion profile updates are handled by symbol_emotion_updater based on symbol_emotion_map.
        # This function could also update the 'emotions' list if new inherent emotions are discovered.
        # For now, 'emotions' stores defined/seed emotions.

    save_symbol_memory(memory, file_path) # Use parameter
    return memory[symbol_token]


# MODIFIED: get_symbol_details to use file_path parameter
def get_symbol_details(symbol_token, file_path=SYMBOL_MEMORY_PATH): # Added file_path
    memory = load_symbol_memory(file_path) # Use parameter
    return memory.get(symbol_token, {})

# MODIFIED: update_symbol_emotional_profile to use file_path (though not directly used by external modules yet)
def update_symbol_emotional_profile(symbol_token, emotion_changes, file_path=SYMBOL_MEMORY_PATH): # Added file_path
    """
    Updates the persistent emotion_profile for a symbol.
    emotion_changes: dict like {'anger': 0.1, 'joy': -0.05}
    """
    memory = load_symbol_memory(file_path) # Use parameter
    if symbol_token in memory:
        profile = memory[symbol_token].get("emotion_profile", {})
        for emotion, change in emotion_changes.items():
            profile[emotion] = profile.get(emotion, 0) + change
            profile[emotion] = round(max(0, min(1, profile[emotion])), 3) # Clamp 0-1
        memory[symbol_token]["emotion_profile"] = profile
        memory[symbol_token]["updated_at"] = datetime.utcnow().isoformat()
        save_symbol_memory(memory, file_path) # Use parameter
        return True
    return False

def prune_duplicates(file_path=SYMBOL_MEMORY_PATH): # Added file_path
    """Removes duplicate entries from vector_examples for each symbol."""
    # This is a more conceptual function; actual vector similarity would be needed for true semantic duplicates.
    # For now, it removes exact text duplicates in vector_examples.
    memory = load_symbol_memory(file_path) # Use parameter
    for symbol_token in memory:
        if "vector_examples" in memory[symbol_token] and isinstance(memory[symbol_token]["vector_examples"], list):
            unique_examples = []
            seen_texts = set()
            for example_entry in memory[symbol_token]["vector_examples"]:
                # example_entry is now expected to be a dict {'text': "...", 'timestamp': "...", 'source_phase': ...}
                if isinstance(example_entry, dict) and "text" in example_entry:
                    example_text = example_entry["text"]
                    if example_text not in seen_texts:
                        unique_examples.append(example_entry)
                        seen_texts.add(example_text)
                # Keep old string-only examples if they exist, but don't add new ones like that
                elif isinstance(example_entry, str): 
                    if example_entry not in seen_texts:
                         unique_examples.append({"text": example_entry, "timestamp": "unknown", "source_phase": 0}) # Convert old format
                         seen_texts.add(example_entry)


            if len(unique_examples) < len(memory[symbol_token]["vector_examples"]):
                print(f"[SYMBOL_MEMORY-INFO] Pruned duplicate examples for symbol '{symbol_token}'.")
            memory[symbol_token]["vector_examples"] = unique_examples
    save_symbol_memory(memory, file_path) # Use parameter

# get_emotion_profile, now less critical if main profile comes from symbol_emotion_map,
# but useful if symbol_memory.json itself stores an evolving profile.
# MODIFIED to use file_path
def get_emotion_profile(symbol_token, file_path=SYMBOL_MEMORY_PATH): # Added file_path
    memory = load_symbol_memory(file_path) # Use parameter
    symbol_data = memory.get(symbol_token)
    if symbol_data and isinstance(symbol_data, dict) and "emotion_profile" in symbol_data:
        return symbol_data["emotion_profile"]
    return {}


if __name__ == '__main__':
    print("Testing symbol_memory.py with phase tagging and robust loading/saving with file_path...")
    
    # Use a temporary test file
    test_sm_path = Path("data/test_symbol_memory_v2.json")
    if test_sm_path.exists(): 
        try: test_sm_path.unlink()
        except OSError as e: print(f"Could not clear {test_sm_path}: {e}")
    
    # Test add_symbol with file_path
    add_symbol("🔥", "Fire", ["heat", "burn"], [{"emotion":"anger", "weight":0.8}], "The fire raged.", 
               origin="seed", learning_phase=1, file_path=test_sm_path)
    add_symbol("💧", "Water", ["flow", "wet"], [{"emotion":"calm", "weight":0.9}], "The water was calm.", 
               origin="seed", learning_phase=1, file_path=test_sm_path)
    
    # Test load_symbol_memory with file_path
    loaded_mem = load_symbol_memory(test_sm_path)
    print(f"\nLoaded test memory ({len(loaded_mem)} entries): {loaded_mem}")
    assert "🔥" in loaded_mem
    assert loaded_mem["🔥"]["learning_phase"] == 1
    assert loaded_mem["💧"]["name"] == "Water"
    assert isinstance(loaded_mem["🔥"]["vector_examples"], list)
    assert len(loaded_mem["🔥"]["vector_examples"]) == 1
    assert isinstance(loaded_mem["🔥"]["vector_examples"][0], dict)
    assert "text" in loaded_mem["🔥"]["vector_examples"][0]

    # Test updating an existing symbol
    add_symbol("🔥", "Big Fire", ["inferno"], [{"emotion":"fear", "weight":0.6}], "The inferno grew.", 
               origin="emergent_update", learning_phase=2, file_path=test_sm_path)
    updated_mem = load_symbol_memory(test_sm_path)
    print(f"\nUpdated test memory for 🔥: {updated_mem['🔥']}")
    assert "inferno" in updated_mem["🔥"]["keywords"] # Check if keywords were added
    assert updated_mem["🔥"]["usage_count"] == 2 # Should be 1 (initial) + 1 (update) = 2
    assert len(updated_mem["🔥"]["vector_examples"]) == 2 # Second example added

    # Test get_symbol_details with file_path
    fire_details = get_symbol_details("🔥", file_path=test_sm_path)
    print(f"\nDetails for 🔥: {fire_details}")
    assert fire_details["name"] == "Fire" # Name doesn't change on simple update unless specifically passed

    # Test prune_duplicates
    # Add a duplicate example manually for testing prune
    current_mem = load_symbol_memory(test_sm_path)
    if "🔥" in current_mem:
        current_mem["🔥"]["vector_examples"].append({"text": "The fire raged.", "timestamp": "now", "source_phase": 1}) # Add duplicate
        save_symbol_memory(current_mem, test_sm_path)
    
    prune_duplicates(file_path=test_sm_path)
    pruned_mem = load_symbol_memory(test_sm_path)
    print(f"\nMemory after prune for 🔥 vector_examples: {pruned_mem['🔥']['vector_examples']}")
    assert len(pruned_mem["🔥"]["vector_examples"]) == 2 # Should have removed the exact text duplicate

    # Test symbol_details_override for meta-symbols
    meta_details = {"name": "FireCycle", "based_on": "🔥", "summary":"A cycle of fire", "keywords": ["fire", "cycle"], "learning_phase": 2, "origin":"meta"}
    add_symbol("🔥⟳", "Fire Cycle Old Name (will be overridden)", ["oldkw"], [], "old example", 
               symbol_details_override=meta_details, file_path=test_sm_path)
    meta_mem = load_symbol_memory(test_sm_path)
    print(f"\nMeta symbol entry 🔥⟳: {meta_mem['🔥⟳']}")
    assert meta_mem["🔥⟳"]["name"] == "FireCycle"
    assert meta_mem["🔥⟳"]["summary"] == "A cycle of fire"
    assert meta_mem["🔥⟳"]["origin"] == "meta"
    assert "created_at" in meta_mem["🔥⟳"]
    assert "updated_at" in meta_mem["🔥⟳"]


    print("\n✅ symbol_memory.py tests completed with file_path parameterization.")
    # if test_sm_path.exists(): test_sm_path.unlink() # Optional: clean up test file