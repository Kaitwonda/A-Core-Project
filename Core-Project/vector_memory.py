import os
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime # APPENDED: Import datetime
from vector_engine import fuse_vectors # fuse_vectors returns (fused_vector_list, debug_info_dict)
from sklearn.metrics.pairwise import cosine_similarity

# Define where vector memory will be stored
memory_file = Path("data/vector_memory.json")
memory_file.parent.mkdir(parents=True, exist_ok=True)

# Initialize memory_file if it doesn't exist or is empty
if not memory_file.exists() or os.path.getsize(memory_file) == 0:
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump([], f) # Store as an empty list

def load_vectors(): # Renamed from load_memory for clarity if used elsewhere
    """Loads all vector entries from the JSON file."""
    if memory_file.exists() and os.path.getsize(memory_file) > 0 : # Check size
        with open(memory_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else [] # Ensure it's a list
            except json.JSONDecodeError:
                print(f"[ERROR] Corrupted vector memory file: {memory_file}. Returning empty list.")
                return []
    return []

def save_vectors(memory_entries): # Renamed from save_memory for clarity
    """Saves all vector entries to the JSON file."""
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_entries, f, indent=2, ensure_ascii=False)

def store_vector(text, source_type="user", source_url=None, learning_phase=0,
                 exploration_depth="deep", confidence=0.7, source_trust="unknown"): # Added new fields
    """
    Stores the text, its fused vector, and metadata including learning phase,
    exploration depth, confidence, and source trust.
    """
    hash_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    fused_vector_list, debug_info = fuse_vectors(text)

    record = {
        "id": hash_key,
        "text": text,
        "vector": fused_vector_list,
        "source_model_info": debug_info["source"], # e.g., "fused", "minilm-dominant", "e5-dominant"
        "model_similarity_score": float(debug_info["similarity"]), # Similarity between MiniLM and E5
        "source_type": source_type, # e.g., "user_direct_input", "web_scrape_deep", "web_scrape_shallow"
        "source_url": source_url,
        "source_trust": source_trust, # New: e.g., "wikipedia", "user_provided", "blog_low_trust"
        "learning_phase": learning_phase, # The phase this info is *most relevant for*
        "exploration_depth": exploration_depth, # New: "deep" or "shallow"
        "confidence": confidence, # New: Confidence in the factualness/utility of this chunk (0.0-1.0)
        "timestamp": datetime.utcnow().isoformat() # Uses the newly imported datetime
    }

    current_memory = load_vectors()
    
    entry_exists = False
    for i, entry in enumerate(current_memory):
        if entry["id"] == hash_key:
            current_memory[i] = record # Update existing entry
            entry_exists = True
            break
    
    if not entry_exists:
        current_memory.append(record)

    save_vectors(current_memory)
    return hash_key

def retrieve_similar_vectors(query_text, top_n=5, max_phase_allowed=None, min_confidence=0.0): # Added min_confidence
    """
    Retrieves the top_n most similar vectors to the query_text,
    optionally filtered by max_phase_allowed and min_confidence.
    """
    if not query_text:
        return []

    all_entries = load_vectors()
    if not all_entries:
        return []

    query_vector_list, _ = fuse_vectors(query_text)
    if query_vector_list is None: # fuse_vectors might return None if text is empty
        print(f"[WARNING] Could not generate vector for query: '{query_text[:50]}...'")
        return []
    query_vector_np = np.array(query_vector_list).reshape(1, -1)


    candidate_entries = []
    for entry in all_entries:
        if max_phase_allowed is not None and entry.get("learning_phase", 0) > max_phase_allowed:
            continue
        if entry.get("confidence", 0.0) < min_confidence:
            continue
        candidate_entries.append(entry)
    
    if not candidate_entries:
        return []

    # Ensure all entry vectors are valid
    valid_entry_vectors = []
    valid_candidate_entries = []
    for entry in candidate_entries:
        if entry.get("vector") and isinstance(entry["vector"], list) and len(entry["vector"]) > 0:
             # Assuming all vectors should have the same dimension as query_vector_np
            if len(entry["vector"]) == query_vector_np.shape[1]:
                valid_entry_vectors.append(entry["vector"])
                valid_candidate_entries.append(entry)
            else:
                print(f"[WARNING] Skipping entry with mismatched vector dimension: ID {entry.get('id', 'N/A')}")
        else:
            print(f"[WARNING] Skipping entry with invalid or missing vector: ID {entry.get('id', 'N/A')}")

    if not valid_entry_vectors:
        return []
        
    entry_vectors_np = np.array(valid_entry_vectors)
    
    try:
        similarities = cosine_similarity(query_vector_np, entry_vectors_np)[0]
    except ValueError as e:
        print(f"[ERROR] Cosine similarity calculation error: {e}")
        print(f"Query vector shape: {query_vector_np.shape}")
        print(f"Candidate entry vectors shape: {entry_vectors_np.shape if entry_vectors_np.size > 0 else 'empty'}")
        return []

    scored_entries = []
    for i, entry in enumerate(valid_candidate_entries): # Use valid_candidate_entries here
        scored_entries.append((float(similarities[i]), entry))

    scored_entries.sort(key=lambda x: x[0], reverse=True)
    
    return scored_entries[:top_n]


if __name__ == "__main__":
    print("Testing vector_memory.py with new fields (exploration_depth, confidence, source_trust, timestamp)...")
    test_memory_file_path = Path("data/test_vector_memory_v2_fixed.json") # Changed name for fresh test
    if test_memory_file_path.exists():
        test_memory_file_path.unlink()
    
    original_memory_file = globals().get('memory_file') # Safer way to get global
    globals()['memory_file'] = test_memory_file_path

    store_vector("Deep learning helps AI.", source_type="test_deep", learning_phase=1,
                 exploration_depth="deep", confidence=0.9, source_trust="academic_paper")
    store_vector("AI winter was a period of reduced funding.", source_type="test_shallow", learning_phase=3,
                 exploration_depth="shallow", confidence=0.65, source_trust="wikipedia_summary")
    store_vector("Symbolic AI uses logic.", source_type="test_deep", learning_phase=1,
                 exploration_depth="deep", confidence=0.8, source_trust="textbook")
    store_vector("AGI is hypothetical for now.", source_type="test_deep", learning_phase=4,
                 exploration_depth="deep", confidence=0.7, source_trust="research_blog")
    store_vector("", source_type="empty_string_test", learning_phase=0) # Test empty string resilience

    print(f"\n--- Contents of {test_memory_file_path} ---")
    final_test_memory = load_vectors()
    assert len(final_test_memory) == 4 # Empty string should not store due to fuse_vectors returning None
    if final_test_memory: # Check if not empty
        for entry in final_test_memory:
            print(f"  Text: {entry['text'][:30]}..., Phase: {entry['learning_phase']}, Depth: {entry['exploration_depth']}, Conf: {entry['confidence']}, Trust: {entry['source_trust']}")
        assert final_test_memory[0]["exploration_depth"] == "deep"
        assert final_test_memory[1]["confidence"] == 0.65


    print("\n--- Testing retrieval ---")
    query = "Artificial Intelligence concepts"
    
    print(f"\nRetrieving for '{query}' (max_phase_allowed=1, min_confidence=0.75):")
    results_p1_c75 = retrieve_similar_vectors(query, max_phase_allowed=1, min_confidence=0.75)
    for score, item in results_p1_c75:
        print(f"  Score: {score:.3f}, Phase: {item.get('learning_phase')}, Conf: {item.get('confidence')}, Text: {item['text'][:30]}...")
    assert len(results_p1_c75) >= 2
    assert any("Deep learning" in item["text"] for _, item in results_p1_c75)
    assert any("Symbolic AI" in item["text"] for _, item in results_p1_c75)

    print(f"\nRetrieving for '{query}' (max_phase_allowed=4, min_confidence=0.6):")
    results_p4_c60 = retrieve_similar_vectors(query, max_phase_allowed=4, min_confidence=0.60)
    for score, item in results_p4_c60:
        print(f"  Score: {score:.3f}, Phase: {item.get('learning_phase')}, Conf: {item.get('confidence')}, Text: {item['text'][:30]}...")
    assert len(results_p4_c60) == 4

    print(f"\nRetrieving for '{query}' (max_phase_allowed=3, min_confidence=0.95):")
    results_p3_c95 = retrieve_similar_vectors(query, max_phase_allowed=3, min_confidence=0.95)
    print(f"Found {len(results_p3_c95)} items.")
    assert len(results_p3_c95) == 0
    
    print(f"\nRetrieving for empty query string:")
    results_empty_query = retrieve_similar_vectors("")
    assert len(results_empty_query) == 0
    print("Correctly returned empty list for empty query.")


    if original_memory_file is not None: # Restore only if it was set
        globals()['memory_file'] = original_memory_file
    print("\n✅ vector_memory.py tests completed with datetime import.")