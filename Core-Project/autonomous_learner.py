# autonomous_learner.py

import time
import random
from pathlib import Path
import json

from processing_nodes import LogicNode, SymbolicNode, DynamicBridge, CurriculumManager
import web_parser # For process_web_url and chunk_text
import parser as P_Parser # For extract_keywords (used for chunk relevance check)

# --- Configuration ---
PHASE_URL_SOURCES = {
    1: [ 
        "https://en.wikipedia.org/wiki/Turing_machine",
        "https://en.wikipedia.org/wiki/Von_Neumann_architecture",
        "https://plato.stanford.edu/entries/logic-classical/",
        "https://en.wikipedia.org/wiki/Algorithm",
        "https://en.wikipedia.org/wiki/Data_structure"
    ],
    2: [
        "https://en.wikipedia.org/wiki/History_of_science",
        "https://en.wikipedia.org/wiki/Physics",
        "https://en.wikipedia.org/wiki/Biology",
        "https://en.wikipedia.org/wiki/World_history",
        "https://www.nature.com/news", 
    ],
    3: [ 
        "https://www.gutenberg.org/ebooks/11", # Alice's Adventures in Wonderland
        "https://www.psychologytoday.com/us/basics/emotion",
        "https://www.poetryfoundation.org/poems/44272/the-raven-56d22275e0cde", 
        "https://en.wikipedia.org/wiki/Narrative_structure",
    ],
    4: [ 
        "https://plato.stanford.edu/entries/category-theory/", 
        "https://iep.utm.edu/category/s-h-epistemology/metaphysics/",
        "https://aeon.co/philosophy",
        "https://www.sacred-texts.com/egy/bookt.htm", 
        "https://en.wikipedia.org/wiki/List_of_paradoxes"
    ]
}

ANALYZE_META_SYMBOLS_EVERY_N_RELEVANT_CHUNKS = 30 
CHECK_PHASE_ADVANCEMENT_EVERY_N_RELEVANT_URLS = 2 
MAX_URLS_TO_ATTEMPT_PER_PHASE_SESSION = 5 
MAX_RELEVANT_URLS_TO_PROCESS_PER_PHASE_SESSION = 2 
FETCH_DELAY_SECONDS = 3 

def select_urls_for_current_phase(phase, all_phase_urls, directives, num_urls_to_select):
    candidate_urls = all_phase_urls.get(phase, [])
    if not candidate_urls: return []
    # Ensure filter_keywords is a set of lowercase strings
    filter_keywords = set(str(d).lower() for d in directives.get("data_source_filter_keywords", []))
    selected_urls = []

    if filter_keywords:
        keyword_matched_urls = [url for url in candidate_urls if any(kw in url.lower() for kw in filter_keywords)]
        random.shuffle(keyword_matched_urls) 
        for url in keyword_matched_urls:
            if url not in selected_urls: selected_urls.append(url)
            if len(selected_urls) >= num_urls_to_select: break
    
    if len(selected_urls) < num_urls_to_select:
        remaining_pool = [url for url in candidate_urls if url not in selected_urls]
        random.shuffle(remaining_pool)
        selected_urls.extend(remaining_pool[:num_urls_to_select - len(selected_urls)])
            
    return selected_urls[:num_urls_to_select]

# Corrected function signature and usage
def check_chunk_relevance_for_phase1(chunk_text, current_phase1_focus_keywords, relevance_threshold, min_matches):
    """Checks if a text chunk is relevant for Phase 1 based on keyword overlap."""
    if not current_phase1_focus_keywords: return True 
    
    chunk_keywords = set(P_Parser.extract_keywords(chunk_text))
    overlap_count = len(chunk_keywords & current_phase1_focus_keywords) # Use passed argument
    
    relevance_score_as_percentage_of_focus = overlap_count / len(current_phase1_focus_keywords) if current_phase1_focus_keywords else 0
    
    if overlap_count >= min_matches or relevance_score_as_percentage_of_focus >= relevance_threshold:
        print(f"    Chunk relevant for Phase 1 (Overlap: {overlap_count}, Score: {relevance_score_as_percentage_of_focus:.2f})")
        return True
    else:
        print(f"    Chunk low relevance for Phase 1 (Overlap: {overlap_count}, Score: {relevance_score_as_percentage_of_focus:.2f}). Shallow processing.")
        return False

def run_autonomous_learning_cycle():
    print("🚀 Initializing Autonomous Learning Cycle...")
    Path("data").mkdir(parents=True, exist_ok=True) 
    
    # Load seed symbols here to pass count to CurriculumManager
    seed_symbols_data = {}
    seed_path = Path("data/seed_symbols.json")
    if seed_path.exists():
        with open(seed_path, "r", encoding="utf-8") as f:
            try:
                seed_symbols_data = json.load(f)
            except json.JSONDecodeError:
                print(f"[ERROR] Could not decode seed_symbols.json. Ensure it's valid JSON.")
    seed_symbols_count = len(seed_symbols_data)


    logic_node = LogicNode()
    symbolic_node = SymbolicNode() 
    curriculum_manager = CurriculumManager(initial_seed_symbols_count=seed_symbols_count) # Pass count
    dynamic_bridge = DynamicBridge(logic_node, symbolic_node, curriculum_manager)

    total_relevant_chunks_processed_ever = 0
    total_relevant_urls_processed_ever = 0
    
    try:
        while curriculum_manager.get_current_phase() <= 4:
            current_phase = curriculum_manager.get_current_phase()
            phase_description = curriculum_manager.get_phase_description()
            print(f"\n📘====== Starting Curriculum Phase {current_phase}: {phase_description} ======")
            
            directives = curriculum_manager.get_processing_directives()
            print(f"Phase Directives: Info: '{directives.get('info')}'")
            print(f"Web Scraping Allowed: {directives.get('allow_web_scraping')}")
            # Ensure phase_focus_keywords is correctly fetched and lowercased
            phase_focus_keywords_from_directives = set(str(d).lower() for d in directives.get("data_source_filter_keywords", []))
            print(f"Focus Keywords for data selection: {phase_focus_keywords_from_directives if phase_focus_keywords_from_directives else 'Any'}")

            if not directives.get("allow_web_scraping", False):
                print(f"Phase {current_phase}: Web scraping not allowed by directives. Simulating internal checks.")
                curriculum_manager.update_metrics(logic_node, symbolic_node, processed_chunk_in_phase_1=(current_phase == 1))
                if curriculum_manager.advance_phase_if_ready(): continue
                else:
                    print(f"Cannot advance from Phase {current_phase}. Ending cycle.")
                    break
            
            urls_to_attempt = select_urls_for_current_phase(
                current_phase, PHASE_URL_SOURCES, directives, MAX_URLS_TO_ATTEMPT_PER_PHASE_SESSION
            )

            if not urls_to_attempt:
                print(f"No suitable URLs selected for Phase {current_phase}. Checking advancement.")
                curriculum_manager.update_metrics(logic_node, symbolic_node)
                if curriculum_manager.advance_phase_if_ready(): continue
                else:
                    print(f"Could not advance from Phase {current_phase}. Ending cycle.")
                    break

            processed_relevant_urls_this_session = 0
            for url_to_scrape in urls_to_attempt:
                if processed_relevant_urls_this_session >= MAX_RELEVANT_URLS_TO_PROCESS_PER_PHASE_SESSION:
                    print(f"Max relevant URLs ({MAX_RELEVANT_URLS_TO_PROCESS_PER_PHASE_SESSION}) for this session in Phase {current_phase} reached.")
                    break 
                
                print(f"\n🔗 Attempting URL: {url_to_scrape}")
                source_url, text_chunks = web_parser.process_web_url(url_to_scrape)

                if not text_chunks:
                    print(f"No text chunks from {source_url}. Skipping.")
                    continue

                any_relevant_chunk_found_in_url = False
                for i, chunk in enumerate(text_chunks):
                    if not chunk.strip(): 
                        print(f"    Chunk {i+1} is empty. Skipping.")
                        continue
                    
                    is_chunk_highly_relevant = True 
                    if current_phase == 1:
                        # Pass the correctly scoped phase_focus_keywords_from_directives
                        is_chunk_highly_relevant = check_chunk_relevance_for_phase1(
                            chunk, 
                            phase_focus_keywords_from_directives, # Correct variable passed
                            directives.get("phase1_strict_chunk_relevance_threshold", 0.1),
                            directives.get("phase1_min_keyword_matches", 1)
                        )
                    
                    dynamic_bridge.route_and_respond(
                        chunk, 
                        source_url=source_url, 
                        is_user_interaction=False,
                        is_highly_relevant_for_phase=is_chunk_highly_relevant 
                    )
                    
                    if is_chunk_highly_relevant:
                        any_relevant_chunk_found_in_url = True
                        total_relevant_chunks_processed_ever += 1
                        curriculum_manager.update_metrics(logic_node, symbolic_node, processed_chunk_in_phase_1=(current_phase == 1 and is_chunk_highly_relevant))
                        # print(f"    Relevant chunk processed. Total relevant chunks ever: {total_relevant_chunks_processed_ever}") # Can be verbose

                        if total_relevant_chunks_processed_ever > 0 and total_relevant_chunks_processed_ever % ANALYZE_META_SYMBOLS_EVERY_N_RELEVANT_CHUNKS == 0:
                            print(f"\n🔄 Running periodic meta-symbol analysis (after {total_relevant_chunks_processed_ever} total relevant chunks)...")
                            meta_results = symbolic_node.run_meta_symbol_analysis(current_phase=current_phase, directives=directives)
                            print(f"Meta-symbol analysis results: {meta_results.get('status')}")
                            if meta_results.get("new_meta_symbols"):
                                for ms in meta_results["new_meta_symbols"]:
                                    print(f"  ✨ New meta-symbol created: {ms.get('symbol')} - {ms.get('name')}")
                
                if any_relevant_chunk_found_in_url:
                    total_relevant_urls_processed_ever += 1
                    processed_relevant_urls_this_session += 1
                    curriculum_manager.update_metrics(logic_node, symbolic_node, processed_url_is_relevant=True)


                if total_relevant_urls_processed_ever > 0 and total_relevant_urls_processed_ever % CHECK_PHASE_ADVANCEMENT_EVERY_N_RELEVANT_URLS == 0:
                    print(f"\n📈 Checking for curriculum phase advancement (after {total_relevant_urls_processed_ever} total relevant URLs)...")
                    if curriculum_manager.advance_phase_if_ready():
                        if curriculum_manager.get_current_phase() != current_phase:
                            print("Phase advanced! Restarting phase loop.")
                            break 
                
                if curriculum_manager.get_current_phase() != current_phase: break 
                
                print(f"Politely waiting for {FETCH_DELAY_SECONDS} seconds before next URL...")
                time.sleep(FETCH_DELAY_SECONDS)
            
            if curriculum_manager.get_current_phase() != current_phase: continue 

            print(f"\n📈 End of URL batch for Phase {current_phase}. Checking for advancement...")
            curriculum_manager.update_metrics(logic_node, symbolic_node) 
            if not curriculum_manager.advance_phase_if_ready() and not urls_to_attempt : 
                 print(f"Stuck in Phase {current_phase} without processable URLs or advancement. Ending.")
                 break
        
        if curriculum_manager.get_current_phase() > 4:
            print("\n🎉🎉🎉 All curriculum phases completed! Autonomous learning cycle finished. 🎉🎉🎉")
        else:
            print(f"\n🏁 Learning cycle ended at Phase {curriculum_manager.get_current_phase()}. Review thresholds, URL sources, or relevance filters.")

    except KeyboardInterrupt: print("\n🛑 Autonomous learning cycle interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Final System State Summary ---")
        print(logic_node.get_knowledge_summary())
        active_lex = symbolic_node._get_active_symbol_lexicon() 
        print(f"Symbolic Node: Seeds={len(symbolic_node.SEED_SYMBOLS)}, Learned={len(symbolic_node.fn_load_symbol_memory())}, Meta={len(symbolic_node.meta_symbols)}, Total Active Lexicon: {len(active_lex)}")
        print(f"Final Curriculum Phase: {curriculum_manager.get_current_phase()} - {curriculum_manager.get_phase_description()}")
        print(f"Total Relevant URLs processed: {total_relevant_urls_processed_ever}")
        print(f"Total Relevant Chunks processed: {total_relevant_chunks_processed_ever}")
        print("💾 All data should be saved by respective modules.")

if __name__ == "__main__":
    data_files_to_check = {
        "data/vector_memory.json": [], "data/symbol_memory.json": {},
        "data/symbol_emotion_map.json": {}, 
        "data/symbol_occurrence_log.json": {"entries": []}, 
        "data/meta_symbols.json": {},
        "data/seed_symbols.json": { 
            "🔥": {"name": "Fire", "keywords": ["fire", "flame", "computation", "logic"], "core_meanings": ["heat"], "emotions": ["anger"], "archetypes": ["destroyer"], "learning_phase": 0, "resonance_weight": 0.7},
            "💧": {"name": "Water", "keywords": ["water", "liquid", "data", "flow"], "core_meanings": ["flow"], "emotions": ["calm"], "archetypes": ["healer"], "learning_phase": 0, "resonance_weight": 0.7},
            "💻": {"name": "Computer", "keywords": ["computer", "computation", "cpu", "binary", "code", "algorithm", "system", "architecture"], "core_meanings": ["processing", "logic unit"], "emotions": ["neutral", "focus"], "archetypes": ["tool", "oracle"], "learning_phase": 0, "resonance_weight": 0.8}
        }
    }
    Path("data").mkdir(parents=True, exist_ok=True)
    for file_path_str, default_content in data_files_to_check.items():
        file_path = Path(file_path_str)
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(default_content, f, indent=2)
            print(f"Created initial data file: {file_path}")
    
    print("Please ensure you have run 'python download_models.py' and 'python -m spacy download en_core_web_sm' first.")
    # input("Press Enter to start the autonomous learning cycle...") 
    run_autonomous_learning_cycle()
