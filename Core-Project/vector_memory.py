import os
import json
import hashlib
import numpy as np
from pathlib import Path
from vector_engine import fuse_vectors # fuse_vectors returns (fused_vector_list, debug_info_dict)
from sklearn.metrics.pairwise import cosine_similarity

# Define where vector memory will be stored
memory_file = Path("data/vector_memory.json")
memory_file.parent.mkdir(parents=True, exist_ok=True)

def store_vector(text, source_type="user", source_url=None, learning_phase=0): # Added learning_phase
    """
    Stores the text, its fused vector, and metadata including the learning phase.
    """
    hash_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    # Check if this text (by hash) already exists to avoid duplicates
    # This requires loading the memory first, which can be inefficient for many calls.
    # A more robust system might use a database or an in-memory set of hashes for quick checks.
    # For now, we'll allow duplicates but this is an area for optimization.
    
    fused_vector_list, debug_info = fuse_vectors(text) # fuse_vectors returns list, not np.array directly

    record = {
        "id": hash_key,
        "text": text,
        "vector": fused_vector_list, # Storing as list as it's JSON serializable
        "source_model_info": debug_info["source"], # e.g., "fused", "minilm-dominant"
        "fusion_similarity": debug_info["similarity"], # Similarity score between minilm and e5
        "source_type": source_type, # "user", "web_chunk"
        "learning_phase": learning_phase # Store the phase
    }

    # Trust weighting by domain
    trust_map = {
        "wikipedia.org": "high",
        "nature.com": "high",
        "sciencemag.org": "high", # Common for Science journal
        "sciencedirect.com": "high",
        "plato.stanford.edu": "high", # Stanford Encyclopedia of Philosophy
        "iep.utm.edu": "high", # Internet Encyclopedia of Philosophy
        "gutenberg.org": "medium", # Project Gutenberg (classic texts)
        "arxiv.org": "medium", # Research papers, pre-print
        "medium.com": "low", # Blog platform
        "reddit.com": "low",
        "unknown": "unknown"
    }

    if source_url:
        try:
            # Basic domain extraction
            domain_parts = source_url.split("://")[-1].split("/")[0].split('.')
            if len(domain_parts) >= 2:
                domain = f"{domain_parts[-2]}.{domain_parts[-1]}" # e.g. wikipedia.org
                if domain_parts[-2] == "co" and len(domain_parts) >=3: # for .co.uk etc.
                    domain = f"{domain_parts[-3]}.{domain_parts[-2]}.{domain_parts[-1]}"
            else:
                domain = "unknown"
        except Exception:
            domain = "unknown"
            
        record["source_url"] = source_url
        record["source_trust"] = trust_map.get(domain, "unknown")
    else:
        record["source_trust"] = "user_input" # Or some other default for non-URL sources


    current_memory = []
    if memory_file.exists():
        with open(memory_file, "r", encoding="utf-8") as f:
            try:
                current_memory = json.load(f)
                if not isinstance(current_memory, list): # Ensure it's a list
                    print(f"⚠️ Vector memory file {memory_file} was not a list. Resetting.")
                    current_memory = []
            except json.JSONDecodeError:
                print(f"⚠️ Vector memory file {memory_file} corrupted. Resetting.")
                current_memory = []
    
    current_memory.append(record)
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(current_memory, f, indent=2)

    print(f"🧠 Vector stored for: '{text[:60]}...' (Phase {learning_phase}) [Model: {debug_info['source']}, FusionSim: {debug_info['similarity']:.2f}]")


def retrieve_similar_vectors(query_text, top_k=5, threshold=0.4, max_phase_allowed=None): # Added max_phase_allowed
    """
    Retrieves vectors similar to the query text, optionally filtered by learning phase.
    """
    if not memory_file.exists():
        print("⚠️ Vector memory file not found for retrieval.")
        return []

    with open(memory_file, "r", encoding="utf-8") as f:
        try:
            memory = json.load(f)
            if not isinstance(memory, list):
                 print(f"⚠️ Vector memory file {memory_file} was not a list. Cannot retrieve.")
                 return []
        except json.JSONDecodeError:
            print(f"⚠️ Vector memory file {memory_file} corrupted. Cannot retrieve.")
            return []

    if not memory:
        return []

    query_vector_list, _ = fuse_vectors(query_text)
    query_vector_np = np.array(query_vector_list)

    similarities = []
    for record in memory:
        # Phase filtering
        if max_phase_allowed is not None:
            record_phase = record.get("learning_phase", 0) # Default to 0 if not tagged
            if record_phase > max_phase_allowed:
                continue # Skip this memory if it's from a future phase

        record_vector_np = np.array(record["vector"])
        # Ensure vectors are 2D for cosine_similarity: [[vec1]], [[vec2]]
        sim = cosine_similarity(query_vector_np.reshape(1, -1), record_vector_np.reshape(1, -1))[0][0]
        if sim >= threshold:
            similarities.append((float(sim), record)) # Store as float for consistency

    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]

if __name__ == '__main__':
    print("Testing vector_memory.py with phase tagging...")
    # Ensure vector_engine models are downloaded by running download_models.py if needed
    
    # Test storing
    store_vector("This is a test for phase 1 knowledge.", source_type="test", learning_phase=1)
    store_vector("Another test, this one for phase 2.", source_type="test", learning_phase=2)
    store_vector("A generic piece of information without explicit phase.", source_type="test") # Defaults to phase 0

    # Test retrieval
    print("\nRetrieving for 'phase 1 test':")
    results_all = retrieve_similar_vectors("phase 1 test")
    for score, item in results_all:
        print(f"  Score: {score:.3f}, Phase: {item.get('learning_phase', 'N/A')}, Text: {item['text'][:30]}...")

    print("\nRetrieving for 'phase 1 test' (max_phase_allowed=1):")
    results_phase1 = retrieve_similar_vectors("phase 1 test", max_phase_allowed=1)
    for score, item in results_phase1:
        print(f"  Score: {score:.3f}, Phase: {item.get('learning_phase', 'N/A')}, Text: {item['text'][:30]}...")
    
    print("\nRetrieving for 'phase 2 test' (max_phase_allowed=1):") # Should ideally not find phase 2 item
    results_phase2_restricted = retrieve_similar_vectors("phase 2 test", max_phase_allowed=1)
    if not results_phase2_restricted:
        print("  Correctly found no phase 2 items when restricted to phase 1.")
    for score, item in results_phase2_restricted:
        print(f"  Score: {score:.3f}, Phase: {item.get('learning_phase', 'N/A')}, Text: {item['text'][:30]}...")

