# symbol_memory.py
from pathlib import Path # NO LEADING SPACES/TABS ON THIS LINE
import json           # NO LEADING SPACES/TABS ON THIS LINE
# from collections import defaultdict # Not strictly needed

# File path for symbol memory
SYMBOL_MEMORY_PATH = Path("data/symbol_memory.json")

def load_symbol_memory():
    """Loads existing symbol memory, ensuring it's a dictionary of symbol objects."""
    SYMBOL_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SYMBOL_MEMORY_PATH.exists():
        with open(SYMBOL_MEMORY_PATH, "r", encoding="utf-8") as f:
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
                    print(f"[WARNING] Symbol memory file {SYMBOL_MEMORY_PATH} is not a dictionary. Returning empty memory.")
                    return {}
            except json.JSONDecodeError:
                print(f"[WARNING] Symbol memory file {SYMBOL_MEMORY_PATH} corrupted. Returning empty memory.")
                return {}
    return {}

def save_symbol_memory(memory):
    """Saves current state of symbol memory to disk."""
    SYMBOL_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SYMBOL_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

def add_symbol(symbol, name, keywords, emotions, example_text, origin="emergent", learning_phase=0):
    """
    Adds or updates a symbol entry in the symbol memory, including its learning phase.
    """
    memory = load_symbol_memory()

    if symbol not in memory:
        memory[symbol] = {
            "name": name,
            "keywords": [], 
            "emotion_profile": {}, 
            "vector_examples": [], 
            "origin": origin,
            "learning_phase": learning_phase, 
            "last_updated_phase": learning_phase,
            "resonance_weight": 0.5 
        }
    elif "learning_phase" not in memory[symbol]: 
        memory[symbol]["learning_phase"] = learning_phase 
    
    memory[symbol]["last_updated_phase"] = max(memory[symbol].get("last_updated_phase", 0), learning_phase)
    
    if "name" not in memory[symbol] or \
       (memory[symbol].get("name") != name and (origin != "seed" or memory[symbol].get("origin") == "seed")):
        memory[symbol]["name"] = name 
    
    if "origin" not in memory[symbol]: 
         memory[symbol]["origin"] = origin
    if "resonance_weight" not in memory[symbol]: 
        memory[symbol]["resonance_weight"] = 0.5

    current_profile = memory[symbol].get("emotion_profile", {})
    for emotion_str, score_float in emotions.items(): 
        current_profile[emotion_str] = current_profile.get(emotion_str, 0.0) + score_float 
    memory[symbol]["emotion_profile"] = current_profile

    if "vector_examples" not in memory[symbol] or not isinstance(memory[symbol]["vector_examples"], list): 
        memory[symbol]["vector_examples"] = []
    memory[symbol]["vector_examples"].append({
        "text": example_text,
        "context_emotions": list(emotions.keys()), 
        "example_phase": learning_phase 
    })
    memory[symbol]["vector_examples"] = memory[symbol]["vector_examples"][-20:] 

    existing_keywords = set(memory[symbol].get("keywords", []))
    for kw in keywords:
        if kw: existing_keywords.add(str(kw).lower()) 
    memory[symbol]["keywords"] = sorted(list(existing_keywords))
    
    save_symbol_memory(memory)

def prune_duplicates():
    memory = load_symbol_memory()
    for symbol_token in list(memory.keys()): 
        if "vector_examples" in memory[symbol_token] and isinstance(memory[symbol_token]["vector_examples"], list):
            seen_texts = set()
            unique_examples = []
            for example in memory[symbol_token]["vector_examples"]:
                if isinstance(example, dict) and "text" in example:
                    example_text = example["text"]
                    if example_text not in seen_texts:
                        unique_examples.append(example)
                        seen_texts.add(example_text)
                else: 
                    unique_examples.append(example) 
            
            if len(unique_examples) < len(memory[symbol_token]["vector_examples"]):
                print(f"[INFO] Pruned duplicate examples for symbol '{symbol_token}'.")
            memory[symbol_token]["vector_examples"] = unique_examples
    save_symbol_memory(memory)

def get_emotion_profile(symbol_token):
    memory = load_symbol_memory()
    symbol_data = memory.get(symbol_token)
    if symbol_data and isinstance(symbol_data, dict) and "emotion_profile" in symbol_data:
        return symbol_data["emotion_profile"]
    return {}

if __name__ == '__main__':
    print("Testing symbol_memory.py with phase tagging and robust loading...")
    if SYMBOL_MEMORY_PATH.exists(): 
        try:
            SYMBOL_MEMORY_PATH.unlink() 
            print(f"Cleared {SYMBOL_MEMORY_PATH} for fresh test.")
        except OSError as e:
            print(f"Could not clear {SYMBOL_MEMORY_PATH}: {e}")

    add_symbol("🔥", "Fire", ["heat", "burn"], {"anger": 0.8, "passion": 0.7}, "The fire raged.", origin="seed", learning_phase=1)
    add_symbol("🔥", "Inferno", ["destruction", "burn"], {"fear": 0.9, "anger": 0.5}, "An inferno consumed the forest.", origin="emergent", learning_phase=2)
    # ... (rest of test code)
