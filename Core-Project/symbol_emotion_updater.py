# symbol_emotion_updater.py

import json
from pathlib import Path
# from collections import defaultdict # Not strictly needed here if map ensures keys

# Default path, used if no path is provided to functions
DEFAULT_MAP_PATH = Path("data/symbol_emotion_map.json")
DEFAULT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure data dir exists at import

# APPENDED/MODIFIED: load_emotion_map function
def load_emotion_map(file_path=DEFAULT_MAP_PATH): # Added file_path parameter
    """Loads the symbol emotion map from the specified file path."""
    current_path = Path(file_path) if isinstance(file_path, str) else file_path
    current_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    if current_path.exists() and current_path.stat().st_size > 0:
        with open(current_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, dict) else {} # Expecting a dict
            except json.JSONDecodeError:
                print(f"[SEU-WARNING] Emotion map file {current_path} corrupted. Returning empty map.")
                return {}
    return {}

# APPENDED/MODIFIED: save_emotion_map function
def save_emotion_map(emotion_map, file_path=DEFAULT_MAP_PATH): # Added file_path parameter
    """Saves the symbol emotion map to the specified file path."""
    current_path = Path(file_path) if isinstance(file_path, str) else file_path
    current_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    with open(current_path, "w", encoding="utf-8") as f:
        json.dump(emotion_map, f, indent=2, ensure_ascii=False)

# APPENDED/MODIFIED: update_symbol_emotions function
def update_symbol_emotions(matched_symbols_weighted, verified_emotions, file_path=DEFAULT_MAP_PATH):
    """
    Updates the symbol_emotion_map.json based on currently matched symbols and detected emotions.
    Args:
        matched_symbols_weighted (list of dicts): Each dict has 'symbol', 'final_weight', etc.
                                                 (Output from P_Parser.parse_with_emotion)
        verified_emotions (list of tuples): e.g., [('joy', 0.8), ('fear', 0.5)]
                                            (This should be emotion_handler.predict_emotions()['verified'])
        file_path (Path or str, optional): The path to the symbol emotion map JSON file.
                                           Defaults to DEFAULT_MAP_PATH.
    """
    if not matched_symbols_weighted or not verified_emotions:
        # print("[SEU-INFO] No symbols matched or no emotions detected. Skipping emotion map update.")
        return

    emotion_map = load_emotion_map(file_path) # Use file_path

    for symbol_entry in matched_symbols_weighted:
        sym_token = symbol_entry.get("symbol")
        # Use the final_weight of the symbol match as a modulator for how much this instance influences the map
        symbol_instance_weight = symbol_entry.get("final_weight", 0.3) # Default to 0.3 if no weight

        if not sym_token:
            continue

        if sym_token not in emotion_map:
            emotion_map[sym_token] = {}

        for emotion_label, emotion_strength_in_text in verified_emotions:
            if not isinstance(emotion_label, str) or not isinstance(emotion_strength_in_text, (float, int)):
                continue # Skip malformed emotion entries

            current_cumulative_strength = emotion_map[sym_token].get(emotion_label.lower(), 0.0)
            
            # Weighted additive update:
            # The strength of the emotion in the current text, modulated by the symbol's relevance in this context.
            # A small learning rate (e.g., 0.1 to 0.25) prevents rapid, unstable shifts.
            learning_rate = 0.15 
            change = emotion_strength_in_text * symbol_instance_weight * learning_rate
            
            updated_strength = current_cumulative_strength + change
            # We are not clamping here; the map reflects cumulative association strength.
            # Clamping might be useful if scores are percentages, but here they are accumulating weights.
            # Let's round for cleaner storage.
            emotion_map[sym_token][emotion_label.lower()] = round(updated_strength, 4)

    save_emotion_map(emotion_map, file_path) # Use file_path
    # print(f"  [SEU] Updated symbol emotion map at {file_path}")


if __name__ == '__main__':
    print("Testing symbol_emotion_updater.py with file_path parameterization...")

    # Use a temporary test file
    test_seu_map_path = Path("data/test_symbol_emotion_map_v2.json")
    if test_seu_map_path.exists():
        try: test_seu_map_path.unlink()
        except OSError as e: print(f"Could not delete {test_seu_map_path}: {e}")
    
    # Initial state (empty map)
    save_emotion_map({}, test_seu_map_path) # Ensure clean start

    # Sample data mimicking output from parser and emotion_handler
    sample_symbols_1 = [
        {"symbol": "💡", "name": "Idea", "final_weight": 0.8},
        {"symbol": "🔥", "name": "Fire", "final_weight": 0.6}
    ]
    sample_emotions_1 = [("curiosity", 0.9), ("excitement", 0.7)]

    print("\n--- First update ---")
    update_symbol_emotions(sample_symbols_1, sample_emotions_1, file_path=test_seu_map_path)
    map_after_1 = load_emotion_map(test_seu_map_path)
    print("Map after first update:", json.dumps(map_after_1, indent=2))
    assert "💡" in map_after_1
    assert "curiosity" in map_after_1["💡"]
    assert map_after_1["💡"]["curiosity"] > 0 # e.g. 0.9 * 0.8 * 0.15 = 0.108
    assert "🔥" in map_after_1
    assert "excitement" in map_after_1["🔥"]


    sample_symbols_2 = [
        {"symbol": "💡", "name": "Idea", "final_weight": 0.7}, # Idea appears again
        {"symbol": "⚙️", "name": "Gear", "final_weight": 0.9}  # New symbol
    ]
    sample_emotions_2 = [("curiosity", 0.8), ("frustration", 0.5)] # Curiosity again, new emotion frustration

    print("\n--- Second update ---")
    update_symbol_emotions(sample_symbols_2, sample_emotions_2, file_path=test_seu_map_path)
    map_after_2 = load_emotion_map(test_seu_map_path)
    print("Map after second update:", json.dumps(map_after_2, indent=2))
    
    assert "💡" in map_after_2
    assert "curiosity" in map_after_2["💡"]
    # Expecting curiosity score for 💡 to be higher than after first update
    # Initial 💡 curiosity: 0.9 (text) * 0.8 (symbol weight) * 0.15 (rate) = 0.108
    # Second 💡 curiosity: 0.8 (text) * 0.7 (symbol weight) * 0.15 (rate) = 0.084. Total = 0.108 + 0.084 = 0.192
    assert map_after_2["💡"]["curiosity"] == round(0.108 + 0.084, 4)
    assert "frustration" in map_after_2["💡"] # Frustration also associated with Idea now

    assert "⚙️" in map_after_2
    assert "curiosity" in map_after_2["⚙️"]
    assert "frustration" in map_after_2["⚙️"]
    assert map_after_2["⚙️"]["frustration"] > 0 # e.g. 0.5 * 0.9 * 0.15 = 0.0675

    print("\n✅ symbol_emotion_updater.py tests completed.")
    # if test_seu_map_path.exists(): test_seu_map_path.unlink() # Optional: clean up