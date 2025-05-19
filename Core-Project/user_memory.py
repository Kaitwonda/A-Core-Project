# user_memory.py (conceptually symbol_occurrence_log.py)
import json
from pathlib import Path
from datetime import datetime

DEFAULT_USER_MEMORY_PATH = Path("data/symbol_occurrence_log.json") 

def load_user_memory(file_path=DEFAULT_USER_MEMORY_PATH):
    """Loads symbol occurrence entries from the specified JSON file."""
    current_path = Path(file_path) if isinstance(file_path, str) else file_path
    current_path.parent.mkdir(parents=True, exist_ok=True) 
    if current_path.exists() and current_path.stat().st_size > 0:
        with open(current_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and "entries" in data:
                    return data.get("entries", []) if isinstance(data["entries"], list) else []
                elif isinstance(data, list): 
                    return data
                else: 
                    print(f"[UM-WARNING] Symbol occurrence log file {current_path} has unexpected format. Returning empty list.")
                    return []
            except json.JSONDecodeError:
                print(f"[UM-WARNING] Symbol occurrence log file {current_path} is corrupted. Returning empty list.")
                return []
    return []

def save_user_memory(entries, file_path=DEFAULT_USER_MEMORY_PATH):
    """Saves symbol occurrence entries to the specified JSON file."""
    current_path = Path(file_path) if isinstance(file_path, str) else file_path
    current_path.parent.mkdir(parents=True, exist_ok=True)
    with open(current_path, "w", encoding="utf-8") as f:
        # Always save in the {"entries": [...]} format for consistency
        json.dump({"entries": entries}, f, indent=2, ensure_ascii=False)

def add_user_memory_entry(symbol, context_text, emotion_in_context, 
                          source_url=None, learning_phase=None, 
                          is_context_highly_relevant=None,
                          file_path=DEFAULT_USER_MEMORY_PATH):
    """Adds a new symbol occurrence entry to the user memory (symbol occurrence log)."""
    current_path = Path(file_path) if isinstance(file_path, str) else file_path
    entries = load_user_memory(current_path) # Load using the specific path

    new_entry = {
        "symbol": symbol, # Parameter name matches definition
        "context_text": context_text,
        "emotion_in_context": emotion_in_context,
        "source_url": source_url,
        "learning_phase": learning_phase,
        "is_context_highly_relevant": is_context_highly_relevant,
        "timestamp": datetime.utcnow().isoformat()
    }
    entries.append(new_entry)
    save_user_memory(entries, current_path) # Save using the specific path
    return new_entry["timestamp"] # Return something to confirm, like the timestamp

if __name__ == '__main__':
    print("Testing user_memory.py (as symbol_occurrence_log) with correct parameters...")
    test_log_path = Path("data/test_symbol_occurrence_v2.json")
    if test_log_path.exists():
        try: test_log_path.unlink()
        except OSError as e: print(f"Could not delete {test_log_path}: {e}")

    # Test using keyword arguments that match the definition
    add_user_memory_entry(symbol="🔥", context_text="A fire started.", emotion_in_context="intense", 
                          source_url="http://example.com/volcanoes", learning_phase=2, 
                          is_context_highly_relevant=True, file_path=test_log_path)
    add_user_memory_entry(symbol="💧", context_text="Sad tears.", emotion_in_context="sadness", 
                          learning_phase=3, is_context_highly_relevant=True, file_path=test_log_path) 

    log_content = load_user_memory(test_log_path)
    print(f"\nTest Log Content ({len(log_content)} entries):")
    for entry in log_content:
        print(f"  Symbol: {entry['symbol']}, Emotion: {entry['emotion_in_context']}, Phase: {entry['learning_phase']}")

    assert len(log_content) == 2
    assert log_content[0]['symbol'] == "🔥"
    assert log_content[1]['emotion_in_context'] == "sadness"

    # Test loading from a file that was just a list (old format support)
    list_only_path = Path("data/test_user_memory_list_only.json")
    with open(list_only_path, "w", encoding="utf-8") as f:
        json.dump([{"symbol":"X", "context_text":"old list format", "emotion_in_context":"test"}], f)
    loaded_from_list = load_user_memory(list_only_path)
    assert len(loaded_from_list) == 1
    assert loaded_from_list[0]['symbol'] == "X"
    if list_only_path.exists(): list_only_path.unlink()

    print("\n✅ user_memory.py tests completed.")