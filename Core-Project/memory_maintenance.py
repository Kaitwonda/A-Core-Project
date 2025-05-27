# memory_maintenance.py
import json
from pathlib import Path
from processing_nodes import detect_content_type, CurriculumManager # For phase keywords and content detection
# Assuming P_Parser is accessible for spacy instance
import parser as P_Parser # For P_Parser.nlp
from vector_memory import load_vectors as vm_load_vectors, save_vectors as vm_save_vectors, memory_file as default_vm_memory_path

def score_text_against_phase1_keywords(text_content, phase1_directives):
    """
    Scores text specifically against Phase 1 keywords to determine its relevance.
    Uses a simplified scoring model for this specific pruning task.
    """
    if not text_content or not isinstance(text_content, str): return 0.0
    text_lower = text_content.lower()
    score = 0.0
    
    # Using get with default empty list for safety if keys are missing in directives
    primary_keywords = phase1_directives.get("phase_keywords_primary", [])
    secondary_keywords = phase1_directives.get("phase_keywords_secondary", [])
    anti_keywords = phase1_directives.get("phase_keywords_anti", [])
    
    for kw in primary_keywords:
        if kw.lower() in text_lower: score += 2.0
    for kw in secondary_keywords:
        if kw.lower() in text_lower: score += 1.0
    for kw in anti_keywords: # Penalize if it matches anti-keywords for phase 1
        if kw.lower() in text_lower: score -= 3.0 # Strong penalty for anti-keywords
    return score

def prune_phase1_symbolic_vectors(vector_memory_path_str=None, archive_path_str=None):
    """
    Identifies vector entries learned in Phase 1 that are highly symbolic 
    or no longer align well with Phase 1's logic-focused keywords.
    These entries are moved to an archive file.

    Args:
        vector_memory_path_str (str, optional): Path to the vector memory JSON file. 
                                               Defaults to vector_memory.memory_file.
        archive_path_str (str, optional): Path to the archive JSON file for pruned entries.
                                          If None, pruned entries are not archived separately.
    """
    print("🧹 Starting Phase 1 symbolic vector pruning...")
    
    # Determine the vector memory path to use
    current_vector_memory_path = Path(vector_memory_path_str) if vector_memory_path_str else default_vm_memory_path

    all_vectors = vm_load_vectors() # This function now uses its internal default path if no specific one is passed
                                    # Or, if vm_load_vectors is changed to accept a path, pass current_vector_memory_path
    if not all_vectors:
        print("  No vectors found in memory to prune.")
        return

    kept_vectors = []
    archived_vectors = []
    
    # Get Phase 1 directives to understand its keyword profile
    temp_curriculum_manager = CurriculumManager() # Instantiate to get directives
    phase1_directives = temp_curriculum_manager.get_processing_directives(1)
    
    # Thresholds for pruning decision (can be tuned or moved to directives)
    # If a Phase 1 entry's keyword score is below this, and it's symbolic/ambiguous, consider pruning.
    MIN_PHASE1_KEYWORD_SCORE_TO_KEEP_IF_SYMBOLIC = 1.0 
    # If a Phase 1 entry's keyword score is below this, regardless of type, consider pruning.
    ABSOLUTE_MIN_PHASE1_KEYWORD_SCORE_TO_KEEP = 0.1 


    pruned_count = 0
    for entry in all_vectors:
        if entry.get("learning_phase") == 1:
            text_to_evaluate = entry.get("text", "")
            # Ensure P_Parser.nlp is loaded if detect_content_type relies on it
            spacy_instance = None
            if P_Parser.NLP_MODEL_LOADED and P_Parser.nlp:
                spacy_instance = P_Parser.nlp
            
            content_type = detect_content_type(text_to_evaluate, spacy_instance)
            phase1_keyword_score = score_text_against_phase1_keywords(text_to_evaluate, phase1_directives)

            should_prune = False
            reason = ""

            if content_type == "symbolic" and phase1_keyword_score < MIN_PHASE1_KEYWORD_SCORE_TO_KEEP_IF_SYMBOLIC:
                should_prune = True
                reason = f"Symbolic content with low Phase 1 keyword score ({phase1_keyword_score:.2f})"
            elif content_type == "ambiguous" and phase1_keyword_score < MIN_PHASE1_KEYWORD_SCORE_TO_KEEP_IF_SYMBOLIC: # Stricter for ambiguous
                should_prune = True
                reason = f"Ambiguous content with low Phase 1 keyword score ({phase1_keyword_score:.2f})"
            elif phase1_keyword_score < ABSOLUTE_MIN_PHASE1_KEYWORD_SCORE_TO_KEEP: # Prune if it just doesn't match P1 at all
                should_prune = True
                reason = f"Very low Phase 1 keyword score ({phase1_keyword_score:.2f})"
            
            if should_prune:
                archived_vectors.append(entry)
                pruned_count +=1
                # print(f"  Pruning Phase 1 entry (ID: ...{entry.get('id', 'N/A')[-6:]}): {reason}")
            else:
                kept_vectors.append(entry)
        else:
            kept_vectors.append(entry) # Keep entries from other phases

    if pruned_count > 0:
        print(f"  Pruned {pruned_count} Phase 1 entries due to symbolic nature or low Phase 1 keyword alignment.")
        # Save the modified list back using the same path it was loaded from
        vm_save_vectors(kept_vectors) 

        if archive_path_str:
            archive_file = Path(archive_path_str)
            archive_file.parent.mkdir(parents=True, exist_ok=True)
            existing_archived_data = []
            if archive_file.exists() and archive_file.stat().st_size > 0:
                with open(archive_file, "r", encoding="utf-8") as f_arch_read:
                    try: 
                        loaded_archive = json.load(f_arch_read)
                        if isinstance(loaded_archive, list):
                             existing_archived_data = loaded_archive
                    except json.JSONDecodeError: 
                        print(f"  [WARN] Archive file {archive_file} corrupted. Starting new archive.")
                        pass 
            
            # Combine existing archive with newly archived vectors
            combined_archived_vectors = existing_archived_data + archived_vectors
            
            # Optional: De-duplicate archive by 'id' to prevent identical entries if run multiple times on same data
            unique_archived_map = {item['id']: item for item in combined_archived_vectors if 'id' in item}
            final_archive_list = list(unique_archived_map.values())

            with open(archive_file, "w", encoding="utf-8") as f_arch_write:
                json.dump(final_archive_list, f_arch_write, indent=2, ensure_ascii=False)
            print(f"  Archived {len(archived_vectors)} pruned entries to {archive_file} (Total in archive: {len(final_archive_list)})")
    else:
        print("  No Phase 1 entries met pruning criteria for this run.")

if __name__ == "__main__":
    print("Testing memory_maintenance.py (Phase 1 Pruning)...")
    
    # Setup a dummy vector_memory.json for testing
    dummy_vm_path_mm = Path("data/test_vm_for_pruning_mm.json") # Unique name for this test
    dummy_archive_path_mm = Path("data/test_vm_archive_pruned_mm.json")

    # Clean up previous test files if they exist
    if dummy_vm_path_mm.exists(): dummy_vm_path_mm.unlink()
    if dummy_archive_path_mm.exists(): dummy_archive_path_mm.unlink()

    # Sample data for testing
    test_vectors_mm = [
        {"id": "v1mm", "text": "This is a core algorithm data structure for Phase 1 about computational complexity.", "learning_phase": 1, "confidence": 0.9, "source_url":"http://example.com/algo"},
        {"id": "v2mm", "text": "A deep philosophical dream about code and binary stars, evoking strong emotions.", "learning_phase": 1, "confidence": 0.4, "source_url":"http://example.com/dream"}, # Should be pruned
        {"id": "v3mm", "text": "Another computer science fact about memory addressing in software architecture.", "learning_phase": 1, "confidence": 0.8, "source_url":"http://example.com/cs"},
        {"id": "v4mm", "text": "The spirit of the machine yearns for understanding mythology and ancient philosophy.", "learning_phase": 1, "confidence": 0.3, "source_url":"http://example.com/myth"}, # Should be pruned
        {"id": "v5mm", "text": "Important notes on quantum physics for phase 4 about string theory.", "learning_phase": 4, "confidence": 0.7, "source_url":"http://example.com/qphys"}, # Should be kept
        {"id": "v6mm", "text": "An old algorithm for sorting, perhaps a bit subjective.", "learning_phase": 1, "confidence": 0.5, "source_url":"http://example.com/oldalgo"}, # Borderline, depends on scoring
        {"id": "v7mm", "text": "A text with no phase 1 keywords, just random words and things.", "learning_phase": 1, "confidence": 0.6} # Should be pruned (low P1 keyword score)
    ]
    with open(dummy_vm_path_mm, "w", encoding="utf-8") as f: json.dump(test_vectors_mm, f)

    # Temporarily override vector_memory's default path for this test module
    # This requires vector_memory.py to use its global 'memory_file' variable.
    original_vm_memory_file_in_module = default_vm_memory_path # Save the actual default
    import vector_memory # Ensure module is loaded to modify its global
    vector_memory.memory_file = dummy_vm_path_mm # Set to our test path

    # Ensure P_Parser.nlp is loaded if not already (for detect_content_type)
    if not P_Parser.NLP_MODEL_LOADED:
        try:
            import spacy
            P_Parser.nlp = spacy.load("en_core_web_sm")
            P_Parser.NLP_MODEL_LOADED = True
            print("   Loaded spaCy for memory_maintenance test.")
        except Exception as e:
            print(f"   Could not load spaCy for memory_maintenance test: {e}")


    prune_phase1_symbolic_vectors(archive_path_str=str(dummy_archive_path_mm))

    # Check results
    if dummy_vm_path_mm.exists():
        pruned_vm_content = json.load(open(dummy_vm_path_mm))
        print(f"Entries remaining in vector memory ({dummy_vm_path_mm}): {len(pruned_vm_content)}")
        # Expected: v1mm, v3mm, v5mm definitely. v6mm might remain or be pruned.
        assert len(pruned_vm_content) <= 4 
        assert not any(entry["id"] == "v2mm" for entry in pruned_vm_content)
        assert not any(entry["id"] == "v4mm" for entry in pruned_vm_content)
        assert not any(entry["id"] == "v7mm" for entry in pruned_vm_content) # Should be pruned due to low keyword score
        assert any(entry["id"] == "v5mm" for entry in pruned_vm_content) 
    else:
        print(f"ERROR: Test vector memory file {dummy_vm_path_mm} not found after pruning.")
        assert False

    if dummy_archive_path_mm.exists():
        archived_content = json.load(open(dummy_archive_path_mm))
        print(f"Entries in archive ({dummy_archive_path_mm}): {len(archived_content)}")
        assert len(archived_content) >= 3 # v2mm, v4mm, v7mm should be archived
        assert any(entry["id"] == "v2mm" for entry in archived_content)
        assert any(entry["id"] == "v4mm" for entry in archived_content)
        assert any(entry["id"] == "v7mm" for entry in archived_content)
    else:
        # This can happen if no files were pruned, which would be a test failure if we expect pruning.
        print(f"WARN: Archive file {dummy_archive_path_mm} not created. Check if pruning occurred as expected.")
        # If we expect items to be pruned, this is an issue.
        assert not (len(test_vectors_mm) == len(pruned_vm_content)) # Fails if nothing was pruned when it should have been


    # Restore original path in vector_memory module and clean up test files
    vector_memory.memory_file = original_vm_memory_file_in_module
    if dummy_vm_path_mm.exists(): dummy_vm_path_mm.unlink()
    if dummy_archive_path_mm.exists(): dummy_archive_path_mm.unlink()
    
    print("✅ memory_maintenance.py tests completed.")