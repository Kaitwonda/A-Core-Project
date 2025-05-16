# user_memory.py (conceptually symbol_occurrence_log.py)
import json
from pathlib import Path
from datetime import datetime

# To be renamed to DEFAULT_SYMBOL_OCCURRENCE_LOG_PATH later
DEFAULT_USER_MEMORY_PATH = Path("data/symbol_occurrence_log.json") 

def load_user_memory(file_path=DEFAULT_USER_MEMORY_PATH):
    """Loads symbol occurrence entries from the specified JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True) 
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Ensure it's a list of entries, even if file was just {} or {"entries": null}
                if isinstance(data, dict) and "entries" in data:
                    return data.get("entries", []) if isinstance(data["entries"], list) else []
                elif isinstance(data, list): # Support for old format that was just a list
                    return data
                else: 
                    print(f"[WARNING] Symbol occurrence log file {file_path} has unexpected format. Returning empty list.")
                    return []
            except json.JSONDecodeError:
                print(f"[WARNING] Symbol occurrence log file {file_path} is corrupted. Returning empty list.")
                return []
    return []

def save_user_memory(entries, file_path=DEFAULT_USER_MEMORY_PATH):
    """Saves symbol occurrence entries to the specified JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Always save as a dictionary with an "entries" key for consistency
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2)
    # print(f"💾 Symbol occurrence log saved to {file_path} with {len(entries)} entries.")

def add_user_memory_entry(symbol_token, context_text, emotion_tag, 
                          source_url=None, learning_phase=0, 
                          is_context_highly_relevant=True, # Added new argument with default
                          file_path=DEFAULT_USER_MEMORY_PATH):
    """Adds a single symbol occurrence entry to the log."""
    entries = load_user_memory(file_path) 
    
    timestamp = datetime.utcnow().isoformat()
    
    new_entry = {
        "symbol": symbol_token,
        "context": context_text[:300], # Limit context length for log
        "emotion_in_context": emotion_tag, 
        "source_url": source_url, 
        "learning_phase": learning_phase, 
        "is_context_highly_relevant": is_context_highly_relevant, # Store the new flag
        "timestamp": timestamp
    }
    entries.append(new_entry)
    save_user_memory(entries, file_path)
    # print(f"📝 Logged occurrence: Symbol {symbol_token} (Phase {learning_phase}, Relevant: {is_context_highly_relevant}) from {source_url if source_url else 'direct'}")

if __name__ == '__main__':
    print("Testing user_memory.py (as symbol_occurrence_log) with phase, source_url, and relevance...")
    test_log_path = Path("data/test_symbol_occurrence_log.json")
    if test_log_path.exists():
        test_log_path.unlink()

    add_user_memory_entry("🔥", "A fire started on the web page about volcanoes.", "intense", 
                          source_url="http://example.com/volcanoes", learning_phase=2, 
                          is_context_highly_relevant=True, file_path=test_log_path)
    add_user_memory_entry("💧", "The user typed: sad tears.", "sadness", 
                          learning_phase=3, is_context_highly_relevant=True, file_path=test_log_path) 
    add_user_memory_entry("🔥", "Another fire mentioned, this time in a story (low relevance context).", "dramatic", 
                          source_url="http://example.com/stories/fire", learning_phase=2, 
                          is_context_highly_relevant=False, file_path=test_log_path)

    log_content = load_user_memory(test_log_path)
    print(f"\nTest Log Content ({len(log_content)} entries):")
    for entry in log_content:
        print(f"  Symbol: {entry['symbol']}, Emotion: {entry['emotion_in_context']}, "
              f"Phase: {entry['learning_phase']}, Relevant: {entry.get('is_context_highly_relevant')}, " 
              f"Source: {entry.get('source_url', 'N/A')}, Context: {entry['context'][:30]}...")
    
    if test_log_path.exists():
        test_log_path.unlink()