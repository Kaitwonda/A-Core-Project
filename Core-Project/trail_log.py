# trail_log.py
import json
import hashlib # For generating a more unique log_id if needed, though DynamicBridge provides one
from pathlib import Path
from datetime import datetime

TRAIL_LOG_FILE_PATH = Path("data/trail_log.json")
# Ensure the data directory exists when the module is loaded
TRAIL_LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_log():
    """Loads the current trail log, ensuring it's a list."""
    if TRAIL_LOG_FILE_PATH.exists() and TRAIL_LOG_FILE_PATH.stat().st_size > 0:
        with open(TRAIL_LOG_FILE_PATH, "r", encoding="utf-8") as f:
            try:
                content = json.load(f)
                # Expecting a list of entries. Handle if it was saved as {"entries": []} previously.
                if isinstance(content, dict) and "entries" in content:
                    return content["entries"] if isinstance(content["entries"], list) else []
                return content if isinstance(content, list) else []
            except json.JSONDecodeError:
                print(f"[TRAIL_LOG-WARNING] Trail log file {TRAIL_LOG_FILE_PATH} corrupted. Initializing new log.")
                return []
    return []

def _save_log(log_entries):
    """Saves the trail log, always as a list of entries."""
    with open(TRAIL_LOG_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2, ensure_ascii=False)

# APPENDED: log_dynamic_bridge_processing_step function
def log_dynamic_bridge_processing_step(log_id=None, # Added log_id as first param
                                     text_input=None, source_url=None, current_phase=0,
                                     directives=None, is_highly_relevant_for_phase=False,
                                     target_storage_phase_for_chunk=None, # New param
                                     is_shallow_content=False, # New param
                                     detected_emotions_output=None,
                                     logic_node_output=None,
                                     symbolic_node_output=None,
                                     generated_response_preview=None): # Added for completeness
    """
    Logs a detailed record of a single processing step from the DynamicBridge.
    """
    current_log = _load_log()

    if log_id is None: # Generate a default log_id if not provided
        timestamp_str = datetime.utcnow().isoformat().replace(':', '-').replace('.', '-')
        input_hash = hashlib.md5(text_input.encode('utf-8')).hexdigest()[:8] if text_input else "no_input"
        log_id = f"step_{timestamp_str}_{input_hash}"

    # Ensure complex objects are serializable (e.g., Path objects in directives)
    serializable_directives = {}
    if directives:
        for k, v in directives.items():
            if isinstance(v, Path):
                serializable_directives[k] = str(v)
            elif isinstance(v, dict): # Handle nested dicts like phase_keywords
                serializable_directives[k] = {
                    dk: dv if not isinstance(dv, Path) else str(dv) 
                    for dk, dv in v.items()
                }
            else:
                serializable_directives[k] = v
    
    # Summarize outputs for brevity if they are too verbose, especially lists of texts
    # Logic node output already seems summarized by DynamicBridge.
    # Symbolic node output seems summarized by DynamicBridge.
    # Detected emotions output: The 'verified' list of tuples should be fine.

    log_entry = {
        "log_id": log_id,
        "timestamp": datetime.utcnow().isoformat(),
        "input_text_preview": text_input[:200] + "..." if text_input and len(text_input) > 200 else text_input,
        "source_url": source_url,
        "processing_phase": current_phase, # Phase in which this processing occurred
        "target_storage_phase_for_chunk": target_storage_phase_for_chunk, # Phase this chunk is deemed most relevant for
        "is_shallow_content": is_shallow_content,
        "phase_directives_info": serializable_directives.get("info", "N/A") if serializable_directives else "N/A", # A summary of directives
        "phase_directives_full": serializable_directives, # Store all directives for detailed review
        "is_highly_relevant_for_current_processing_phase": is_highly_relevant_for_phase,
        "detected_emotions_summary": { # Summarize emotions
             "top_verified": detected_emotions_output.get("verified", [])[:3] if detected_emotions_output else [],
             # "top_hartmann": detected_emotions_output.get("hartmann_emotions", [])[:2] if detected_emotions_output else [] # Optional more detail
        },
        "logic_node_summary": logic_node_output,
        "symbolic_node_summary": symbolic_node_output,
        "generated_response_preview": generated_response_preview[:200] + "..." if generated_response_preview and len(generated_response_preview) > 200 else generated_response_preview
    }
    current_log.append(log_entry)
    _save_log(current_log)
    # print(f"  [TRAIL_LOG] Logged step {log_id} for phase {current_phase}.")


# --- Old logging functions, can be kept for compatibility or refactored/removed later ---
# These are likely used by main.py or memory_optimizer.py

def log_trail(text, symbols, matches, file_path=TRAIL_LOG_FILE_PATH): # Ensure file_path consistency
    """Appends an entry to the trail log for older interaction style."""
    # This function logs a different structure than log_dynamic_bridge_processing_step
    # It might be from an older version or for a different purpose (e.g., direct user interaction logging)
    # For now, we keep it separate. If it's meant to be the same, it needs refactoring.
    log_data = _load_log() # Uses the main TRAIL_LOG_FILE_PATH

    entry_id = hashlib.md5(text.encode('utf-8')).hexdigest() # Simple ID for this style
    new_entry = {
        "id": entry_id,
        "timestamp": datetime.utcnow().isoformat(),
        "text": text,
        "symbols": symbols, # Expects a list of symbol dicts from parser
        "matches": matches, # Expects list of (score, item_dict) from vector_memory
        "emotions": [] # Placeholder, to be filled by add_emotions
    }
    log_data.append(new_entry)
    _save_log(log_data) # Uses the main TRAIL_LOG_FILE_PATH
    return entry_id

def add_emotions(entry_id, emotions, file_path=TRAIL_LOG_FILE_PATH): # Ensure file_path consistency
    """Adds detected emotions to a specific log entry for older style."""
    log_data = _load_log()
    for entry in log_data:
        if entry.get("id") == entry_id: # Check if "id" key exists
            entry["emotions"] = emotions # Assumes emotions is a serializable list/dict
            break
    _save_log(log_data)


if __name__ == '__main__':
    print("Testing trail_log.py with new log_dynamic_bridge_processing_step...")

    # Dummy data for testing log_dynamic_bridge_processing_step
    dummy_text = "This is a test chunk processed by DynamicBridge."
    dummy_url = "http://example.com/test_page"
    dummy_phase = 1
    dummy_target_phase = 1
    dummy_directives = {
        "phase": 1, "info": "Test Phase 1 directives", 
        "logic_node_access_max_phase": 1, "symbolic_node_access_max_phase": 1,
        "phase_keywords_primary": ["test", "chunk"]
    }
    dummy_relevance = True
    dummy_is_shallow = False
    dummy_emotions_out = {"verified": [("joy", 0.8), ("curiosity", 0.6)], "other_models_summary": "mostly positive"}
    dummy_logic_out = {"retrieved_memories_count": 1, "top_retrieved_texts": [{"text": "Similar old chunk...", "similarity": 0.9, "phase_learned": 1}]}
    dummy_symbolic_out = {"matched_symbols_count": 1, "top_matched_symbols": [{"symbol": "💡", "name": "Idea", "emotional_weight": 0.75, "influencing_emotions": [("curiosity", 0.45)]}], "generated_symbol": None}
    dummy_response = "[BRIDGE] Processed test chunk."

    # Test with a clean log or a temporary one for the new function
    temp_bridge_log_path = Path("data/test_dynamic_bridge_trail_log.json") 
    if temp_bridge_log_path.exists(): temp_bridge_log_path.unlink()
    
    original_global_path = TRAIL_LOG_FILE_PATH # Save global
    globals()['TRAIL_LOG_FILE_PATH'] = temp_bridge_log_path # Override global for this test call

    log_dynamic_bridge_processing_step(
        text_input=dummy_text, source_url=dummy_url, current_phase=dummy_phase,
        directives=dummy_directives, is_highly_relevant_for_phase=dummy_relevance,
        target_storage_phase_for_chunk=dummy_target_phase, is_shallow_content=dummy_is_shallow,
        detected_emotions_output=dummy_emotions_out, logic_node_output=dummy_logic_out,
        symbolic_node_output=dummy_symbolic_out, generated_response_preview=dummy_response
    )
    
    log_dynamic_bridge_processing_step( # Log another to ensure list append works
        text_input="Another chunk, less relevant.", source_url=dummy_url, current_phase=dummy_phase,
        directives=dummy_directives, is_highly_relevant_for_phase=False,
        target_storage_phase_for_chunk=2, is_shallow_content=True, # e.g. shallow content for next phase
        detected_emotions_output={"verified": [("neutral", 0.9)]}, logic_node_output={"retrieved_memories_count": 0},
        symbolic_node_output={"matched_symbols_count": 0, "generated_symbol": None}
    )

    # Verify the new log
    assert temp_bridge_log_path.exists()
    with open(temp_bridge_log_path, "r") as f:
        bridge_log_data = json.load(f)
    print(f"\nContent of {temp_bridge_log_path} ({len(bridge_log_data)} entries):")
    for i, entry in enumerate(bridge_log_data):
        print(f"  Entry {i+1}: Log ID: {entry['log_id']}, Phase: {entry['processing_phase']}, TargetStore: {entry.get('target_storage_phase_for_chunk')}, Relevant: {entry['is_highly_relevant_for_current_processing_phase']}")
    assert len(bridge_log_data) == 2
    assert bridge_log_data[0]["processing_phase"] == 1
    assert bridge_log_data[0]["target_storage_phase_for_chunk"] == 1
    assert bridge_log_data[1]["target_storage_phase_for_chunk"] == 2
    assert bridge_log_data[1]["is_shallow_content"] is True


    # Test old functions (they use the global TRAIL_LOG_FILE_PATH, which is currently our temp path)
    print("\n--- Testing old log_trail and add_emotions (will write to same temp log) ---")
    old_log_id = log_trail("Old style log test", [{"symbol":"X"}], [(0.5, {"text":"match"})])
    add_emotions(old_log_id, [("test_emo", 0.99)])
    
    bridge_log_data_after_old = json.load(open(temp_bridge_log_path, "r"))
    print(f"Log now has {len(bridge_log_data_after_old)} entries.")
    assert len(bridge_log_data_after_old) == 3 # 2 new style, 1 old style
    assert any(e.get("id") == old_log_id and e.get("emotions") for e in bridge_log_data_after_old)


    globals()['TRAIL_LOG_FILE_PATH'] = original_global_path # Restore global
    print(f"\n✅ trail_log.py tests completed (using {temp_bridge_log_path}).")