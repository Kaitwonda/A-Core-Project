# trail_log.py
import json # NO LEADING SPACES/TABS ON THIS LINE
import hashlib
from pathlib import Path
from datetime import datetime

TRAIL_LOG_FILE_PATH = Path("data/trail_log.json")
# Ensure the data directory exists when the module is loaded
TRAIL_LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_log():
    """Loads the current trail log."""
    if TRAIL_LOG_FILE_PATH.exists():
        with open(TRAIL_LOG_FILE_PATH, "r", encoding="utf-8") as f:
            try:
                content = json.load(f)
                # Expecting a list of entries. Handle if it was saved as {"entries": []} by mistake.
                if isinstance(content, dict) and "entries" in content:
                    return content["entries"] if isinstance(content["entries"], list) else []
                return content if isinstance(content, list) else []
            except json.JSONDecodeError:
                print(f"[WARNING] Trail log file {TRAIL_LOG_FILE_PATH} corrupted. Initializing new log.")
                return []
    return []

def _save_log(log_entries):
    """Saves the trail log, always as a list of entries."""
    with open(TRAIL_LOG_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2, ensure_ascii=False)

def log_dynamic_bridge_processing_step(
    text_input,
    source_url,
    current_phase,
    directives,
    is_highly_relevant_for_phase,
    detected_emotions_output,
    logic_node_output,
    symbolic_node_output,
    final_response_for_user=None
):
    """
    Logs a comprehensive snapshot of a processing step managed by the DynamicBridge.
    """
    timestamp = datetime.utcnow().isoformat()
    input_hash_part = hashlib.sha256(text_input[:50].encode("utf-8")).hexdigest()[:8]
    # Ensure log_id is filename-safe and more unique
    log_id = f"step_{timestamp.replace(':', '-').replace('.', '-')}_{input_hash_part}"

    logic_summary = {
        "retrieved_memories_count": len(logic_node_output.get("retrieved_memories", [])),
        "top_retrieved_texts": [
            {
                "text": mem.get("text", "")[:100] + "...",
                "similarity": round(mem.get("similarity_score", 0.0), 3),
                "phase_learned": mem.get("phase_learned", 0),
                "source_url": mem.get("source_url", "N/A")
            } for mem in logic_node_output.get("retrieved_memories", [])[:2]
        ]
    }

    symbols_for_log = symbolic_node_output.get("matched_symbols", [])

    symbolic_summary = {
        "matched_symbols_count": len(symbols_for_log),
        "top_matched_symbols": [
            {
                "symbol": s.get("symbol", "N/A") if isinstance(s, dict) else str(s),
                "name": s.get("name", "N/A") if isinstance(s, dict) else "N/A",
                "emotional_weight": round(s.get("emotional_weight", 0.0), 3) if isinstance(s, dict) else 0.0,
                "influencing_emotions": s.get("influencing_emotions", []) if isinstance(s, dict) else []
            } for s in symbols_for_log[:3]
        ],
        "generated_symbol": symbolic_node_output.get("generated_symbol_details"),
        "top_detected_emotions_input": detected_emotions_output.get('verified', [])[:3]
    }

    log_entry = {
        "log_id": log_id,
        "timestamp": timestamp,
        "input_text_preview": text_input[:200] + ("..." if len(text_input) > 200 else ""),
        "source_url": source_url,
        "processing_phase": current_phase,
        "phase_directives_info": directives.get("info", "N/A"),
        "phase_directives_full": directives, # Log all directives
        "is_highly_relevant_for_phase": is_highly_relevant_for_phase,
        "logic_node_summary": logic_summary,
        "symbolic_node_summary": symbolic_summary,
        "final_user_response_preview": (final_response_for_user[:200] + "..." if final_response_for_user and len(final_response_for_user) > 200 else final_response_for_user) if final_response_for_user else "N/A (Autonomous Learning Step)"
    }

    current_log_entries = _load_log()
    current_log_entries.append(log_entry)
    _save_log(current_log_entries)
    # print(f"📜 Trail logged for DynamicBridge step (Phase {current_phase}, Relevant: {is_highly_relevant_for_phase}).") # Can be verbose

# --- Old logging functions (can be deprecated or refactored if no longer used by main.py directly) ---
def log_trail(text, symbols, matches, source_url=None, trust=None): # Old function
    print("[DEPRECATED WARNING] log_trail called. Use log_dynamic_bridge_processing_step from DynamicBridge.")
    pass


def add_emotions(text, emotion_output): # Old function
    print("[DEPRECATED WARNING] add_emotions called. Emotions should be part of the main log entry.")
    pass

if __name__ == '__main__':
    print("Testing enhanced trail_log.py...")
    # Create dummy data for testing
    dummy_text = "This is a test input for the dynamic bridge logging."
    dummy_url = "http://example.com/test_page"
    dummy_phase = 1
    dummy_directives = {"info": "Test Phase 1 directives", "logic_node_access_max_phase": 1, "phase": 1}
    dummy_relevance = True
    dummy_emotions_out = {"verified": [("joy", 0.8), ("curiosity", 0.6)], "raw_hartmann": []}
    dummy_logic_out = {"retrieved_memories": [{"text": "fact 1 from phase 1", "similarity_score": 0.9, "phase_learned": 1, "source_url":"http://a.com"}]}
    dummy_symbolic_out = {
        "matched_symbols": [
            {"symbol": "💡", "name": "Idea", "emotional_weight": 0.75, "influencing_emotions": [("curiosity", 0.45)]}
        ],
        "generated_symbol_details": None
    }
    dummy_response = "[Phase 1: Test Phase 1 directives]\nInput Text Emotional Tone (Top 3): [('joy', 0.8), ('curiosity', 0.6)]\n[Symbolic Interpretation (Contextual Resonance)]:\n  [Symbol (Defined Phase 0)] 💡 (Idea), Current Emotional Weight: 0.75"

    # Test with a clean log or a temporary one
    temp_log_path = Path("data/test_trail_log.json") 
    if temp_log_path.exists(): temp_log_path.unlink()
    
    original_global_path = TRAIL_LOG_FILE_PATH 
    globals()['TRAIL_LOG_FILE_PATH'] = temp_log_path 

    log_dynamic_bridge_processing_step(
        text_input=dummy_text, source_url=dummy_url, current_phase=dummy_phase,
        directives=dummy_directives, is_highly_relevant_for_phase=dummy_relevance,
        detected_emotions_output=dummy_emotions_out, logic_node_output=dummy_logic_out,
        symbolic_node_output=dummy_symbolic_out, final_response_for_user=dummy_response
    )
    print(f"Test log entry added to {TRAIL_LOG_FILE_PATH}") 

    log_content = _load_log()
    if log_content:
        print("Content of the first test log entry:")
        print(json.dumps(log_content[0], indent=2))

    if temp_log_path.exists(): temp_log_path.unlink() 
    globals()['TRAIL_LOG_FILE_PATH'] = original_global_path