# processing_nodes.py
# (Assuming all previous imports and LogicNode class are here and correct
#  as per the version that fixed the emoji SyntaxError)
# ... (LogicNode class definition as before) ...

import json
from pathlib import Path
import hashlib 
from datetime import datetime 
from collections import Counter, defaultdict

# --- Node-specific imports ---
from vector_engine import fuse_vectors 
from vector_memory import store_vector as vm_store_vector, \
                          retrieve_similar_vectors as vm_retrieve_similar_vectors, \
                          memory_file as vm_memory_path

import parser as P_Parser 
import emotion_handler as EH_EmotionHandler
import symbol_emotion_updater as SEU_SymbolEmotionUpdater
import symbol_memory as SM_SymbolMemory
import symbol_generator as SG_Refactored_SymbolGenerator 
import user_memory as UM_UserMemory 
import trail_log as TL_TrailLog 

class LogicNode: # Copied from latest for completeness
    def __init__(self, vector_memory_path_str=None):
        self.memory_path = Path(vector_memory_path_str) if vector_memory_path_str else vm_memory_path
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists(): 
            with open(self.memory_path, "w", encoding="utf-8") as f: json.dump([], f) 
        print(f"🧠 LogicNode initialized. Memory path: {self.memory_path}")

    def process_input_for_facts(self, text, source_type="user", source_url=None, current_phase=1, is_highly_relevant_for_phase=True):
        effective_source_type = source_type
        if current_phase == 1 and not is_highly_relevant_for_phase and source_type == "web_chunk":
            effective_source_type = "web_low_relevance_p1" 
        vm_store_vector(text, source_type=effective_source_type, source_url=source_url, learning_phase=current_phase)
        retrieved_memories = vm_retrieve_similar_vectors(text, top_k=5, threshold=0.35) 
        formatted_memories = []
        if retrieved_memories:
            for score, entry in retrieved_memories:
                formatted_memories.append({
                    "text": entry.get("text", ""), "similarity_score": score,
                    "trust_level": entry.get("source_trust", "unknown"),
                    "source_url": entry.get("source_url", None),
                    "original_source_type": entry.get("source_type", "unknown"), 
                    "phase_learned": entry.get("learning_phase", 0) 
                })
        return {"retrieved_memories": formatted_memories}

    def get_knowledge_summary(self):
        if self.memory_path.exists():
            with open(self.memory_path, "r", encoding="utf-8") as f:
                try:
                    memory_data = json.load(f)
                    return f"LogicNode has {len(memory_data)} entries in vector memory."
                except json.JSONDecodeError: return "LogicNode vector memory file is corrupted."
        return "LogicNode vector memory is empty or not found."

class SymbolicNode: 
    def __init__(self,
                 seed_symbols_path_str="data/seed_symbols.json",
                 symbol_memory_path_str="data/symbol_memory.json",
                 symbol_emotion_map_path_str="data/symbol_emotion_map.json",
                 symbol_occurrence_log_path_str="data/symbol_occurrence_log.json",
                 meta_symbols_path_str="data/meta_symbols.json"
                 ):
        self.seed_symbols_path = Path(seed_symbols_path_str)
        self.symbol_memory_path = Path(symbol_memory_path_str)
        self.symbol_emotion_map_path = Path(symbol_emotion_map_path_str)
        self.symbol_occurrence_log_path = Path(symbol_occurrence_log_path_str)
        self.meta_symbols_path = Path(meta_symbols_path_str)

        for p_str_obj in [self.seed_symbols_path, self.symbol_memory_path, self.symbol_emotion_map_path,
                      self.symbol_occurrence_log_path, self.meta_symbols_path]:
            p_str_obj.parent.mkdir(parents=True, exist_ok=True)
            if ".json" in p_str_obj.name and not p_str_obj.exists():
                 with open(p_str_obj, "w", encoding="utf-8") as f:
                    if "log" in p_str_obj.name or "user_memory" in p_str_obj.name : json.dump({"entries":[]}, f)
                    else: json.dump({}, f) # Initialize maps/dicts as empty objects

        self.fn_predict_emotions = EH_EmotionHandler.predict_emotions
        self.fn_parse_with_emotion = P_Parser.parse_with_emotion
        self.fn_extract_keywords = P_Parser.extract_keywords
        self.fn_update_symbol_emotions = SEU_SymbolEmotionUpdater.update_symbol_emotions
        self.fn_add_symbol_to_memory = SM_SymbolMemory.add_symbol
        self.fn_load_symbol_memory = SM_SymbolMemory.load_symbol_memory
        self.fn_generate_emergent_symbol_dict = SG_Refactored_SymbolGenerator.generate_symbol_from_context
        self.fn_load_occurrence_log = UM_UserMemory.load_user_memory
        self.fn_add_occurrence_entry = UM_UserMemory.add_user_memory_entry
        self.fn_Counter = Counter

        # Ensure SEED_SYMBOLS is a dictionary
        self.SEED_SYMBOLS = {} 
        if self.seed_symbols_path.exists():
            with open(self.seed_symbols_path, "r", encoding="utf-8") as f:
                try: 
                    loaded_seeds = json.load(f)
                    if isinstance(loaded_seeds, dict):
                        self.SEED_SYMBOLS = loaded_seeds
                    else:
                        print(f"[ERROR] Seed symbols file {self.seed_symbols_path} is not a dictionary. Using empty seeds.")
                except json.JSONDecodeError: 
                    print(f"[WARNING] Seed symbols file {self.seed_symbols_path} is corrupted. Using empty seeds.")
        else: 
            print(f"[WARNING] Seed symbols file not found at {self.seed_symbols_path}. Using empty seeds.")
        
        self.meta_symbols = self._load_meta_symbols() # Ensures it returns a dict
        
        # Ensure fn_load_symbol_memory also returns a dict
        initial_evolving_symbols = self.fn_load_symbol_memory()
        if not isinstance(initial_evolving_symbols, dict):
            print(f"[WARNING] Symbol memory from {self.symbol_memory_path} was not a dict. Resetting to empty.")
            initial_evolving_symbols = {}
            SM_SymbolMemory.save_symbol_memory({}) # Save an empty dict to fix the file

        print(f"🔮 SymbolicNode initialized. Seed symbols: {len(self.SEED_SYMBOLS)}. "
              f"Evolving symbols: {len(initial_evolving_symbols)}. Meta-symbols: {len(self.meta_symbols)}.")
              
    def _load_meta_symbols(self):
        if self.meta_symbols_path.exists():
            with open(self.meta_symbols_path, "r", encoding="utf-8") as f:
                try: 
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
                except json.JSONDecodeError: 
                    print(f"[WARNING] Meta-symbols file {self.meta_symbols_path} corrupted. Starting empty.")
                    return {}
        return {}
        
    def _save_meta_symbols(self): 
        with open(self.meta_symbols_path, "w", encoding="utf-8") as f: json.dump(self.meta_symbols, f, indent=2)
        
    def _get_active_symbol_lexicon(self, current_phase=4, directives=None): 
        memory_symbols = self.fn_load_symbol_memory()
        if not isinstance(memory_symbols, dict): # Safety check
            print(f"[ERROR] Corrupted symbol_memory.json, expected dict, got {type(memory_symbols)}. Using empty.")
            memory_symbols = {}

        # Ensure all components are dictionaries before merging
        seed_dict = self.SEED_SYMBOLS if isinstance(self.SEED_SYMBOLS, dict) else {}
        meta_dict = self.meta_symbols if isinstance(self.meta_symbols, dict) else {}

        combined = {**seed_dict, **memory_symbols, **meta_dict}
        
        if directives and directives.get("symbolic_node_access_max_phase") is not None:
            max_phase = directives["symbolic_node_access_max_phase"]
            filtered_lexicon = {}
            for token, data_item in combined.items():
                if not isinstance(data_item, dict): # Skip if an item in combined is not a dict
                    print(f"[WARNING] Malformed data for token '{token}' in combined lexicon. Skipping.")
                    continue
                symbol_phase = data_item.get("learning_phase", 0 if token in seed_dict else current_phase) 
                if symbol_phase <= max_phase:
                    filtered_lexicon[token] = data_item
            return filtered_lexicon
        return combined
        
    def process_input_for_symbols(self, text, source_url=None, current_phase=1, directives=None, pre_detected_emotions_output=None, is_highly_relevant_for_phase=True): 
        # print(f"🔮 SymbolicNode (Phase {current_phase}, Relevant: {is_highly_relevant_for_phase}) processing: '{text[:30]}...'")
        emotions_output = pre_detected_emotions_output if pre_detected_emotions_output else self.fn_predict_emotions(text)
        verified_emotions = emotions_output.get('verified', [])
        active_lexicon = self._get_active_symbol_lexicon(current_phase, directives)
        
        # This now uses the updated parser.py that accepts current_lexicon
        matched_symbols_from_parser = self.fn_parse_with_emotion(text, verified_emotions, current_lexicon=active_lexicon)
        
        phase_filtered_matches = [] # Initialize as empty list
        if directives and directives.get("symbolic_node_access_max_phase") is not None:
            max_phase_sym = directives["symbolic_node_access_max_phase"]
            for sym_info in matched_symbols_from_parser:
                # Ensure sym_info itself is a dict and has a 'symbol' key
                if not isinstance(sym_info, dict) or "symbol" not in sym_info:
                    continue 
                symbol_data = active_lexicon.get(sym_info["symbol"]) # Check against the already phase-filtered active_lexicon
                if symbol_data and isinstance(symbol_data, dict) and \
                   symbol_data.get("learning_phase", 0 if sym_info["symbol"] in self.SEED_SYMBOLS else current_phase) <= max_phase_sym:
                    phase_filtered_matches.append(sym_info)
        else: # No specific directive, so all initially matched symbols are considered phase-appropriate for learning
            phase_filtered_matches = matched_symbols_from_parser
        
        # print(f"🔮 Matched symbols (emotion-weighted, phase-filtered): {len(phase_filtered_matches)}")

        if is_highly_relevant_for_phase:
            self.fn_update_symbol_emotions(phase_filtered_matches, verified_emotions)
            if phase_filtered_matches:
                for sym_entry_from_parser in phase_filtered_matches:
                    if not isinstance(sym_entry_from_parser, dict) or "symbol" not in sym_entry_from_parser: continue # Safety
                    symbol_token = sym_entry_from_parser["symbol"]
                    symbol_data_from_lexicon = active_lexicon.get(symbol_token) 
                    if symbol_data_from_lexicon and isinstance(symbol_data_from_lexicon, dict):
                        current_emotions_dict = {emo_str: score for emo_str, score in verified_emotions}
                        keywords_for_memory = symbol_data_from_lexicon.get("keywords", 
                                            symbol_data_from_lexicon.get("core_meanings", 
                                            [sym_entry_from_parser.get("matched_keyword", "")] if sym_entry_from_parser.get("matched_keyword") else []))
                        self.fn_add_symbol_to_memory(symbol=symbol_token, name=symbol_data_from_lexicon.get("name", sym_entry_from_parser.get("name", "Unknown")),
                            keywords=list(set(keywords_for_memory)), emotions=current_emotions_dict, example_text=text, 
                            origin=symbol_data_from_lexicon.get("origin", "seed"), learning_phase=current_phase )
        # else: print(f"🔮 Shallow symbolic processing for low-relevance chunk in Phase {current_phase}.")

        for sym_entry_from_parser in phase_filtered_matches: # Log occurrences of phase-appropriate symbols
            if not isinstance(sym_entry_from_parser, dict) or "symbol" not in sym_entry_from_parser: continue # Safety
            top_emotion_tag = verified_emotions[0][0] if verified_emotions else "(unspecified)"
            self.fn_add_occurrence_entry(symbol_token=sym_entry_from_parser["symbol"], context_text=text, emotion_tag=top_emotion_tag, 
                                         file_path=self.symbol_occurrence_log_path, source_url=source_url, learning_phase=current_phase, 
                                         is_context_highly_relevant=is_highly_relevant_for_phase)
        generated_symbol_details = None
        allow_gen = directives.get("allow_new_symbol_generation", True) if directives else True
        is_match_weak = not phase_filtered_matches or (phase_filtered_matches and phase_filtered_matches[0].get('emotional_weight',0.0) < 0.1)
        if allow_gen and is_highly_relevant_for_phase and is_match_weak :
            # print("🔮 No strong symbol match for relevant chunk, attempting to generate emergent symbol...")
            keywords_for_gen = self.fn_extract_keywords(text)
            new_symbol_dict = self.fn_generate_emergent_symbol_dict(text, keywords_for_gen, verified_emotions)
            if new_symbol_dict and isinstance(new_symbol_dict, dict):
                generated_symbol_details = new_symbol_dict 
                self.fn_add_symbol_to_memory(symbol=generated_symbol_details["symbol"], name=generated_symbol_details["name"],
                    keywords=generated_symbol_details.get("keywords",[]), emotions=generated_symbol_details.get("emotions",{}),
                    example_text=text, origin="emergent", learning_phase=current_phase)
                print(f"✨ SymbolicNode formally added new emergent symbol: {generated_symbol_details.get('symbol')} - {generated_symbol_details.get('name')} in Phase {current_phase}")
                phase_filtered_matches.append({"symbol": generated_symbol_details["symbol"], "name": generated_symbol_details["name"],
                    "matched_keyword": ", ".join(generated_symbol_details.get("keywords", [])), "emotional_weight": 0.5, 
                    "influencing_emotions": verified_emotions })
                top_emotion_tag = verified_emotions[0][0] if verified_emotions else "(unspecified)"
                self.fn_add_occurrence_entry(symbol_token=generated_symbol_details["symbol"], context_text=text,
                    emotion_tag=top_emotion_tag, file_path=self.symbol_occurrence_log_path, source_url=source_url, 
                    learning_phase=current_phase, is_context_highly_relevant=True)
        return {"detected_emotions_output": emotions_output, "matched_symbols": phase_filtered_matches, 
                "generated_symbol_details": generated_symbol_details}
                
    def run_meta_symbol_analysis(self, min_occurrences=5, min_distinct_emotions=3, current_phase=1, directives=None): 
        # print(f"🔮 Analyzing for meta-symbol candidates (context: Phase {current_phase})...")
        occurrence_log_entries = self.fn_load_occurrence_log(self.symbol_occurrence_log_path)
        max_phase_for_analysis = current_phase 
        if directives and directives.get("meta_symbol_analysis_max_phase") is not None:
            max_phase_for_analysis = directives["meta_symbol_analysis_max_phase"]
        filtered_log_entries = [e for e in occurrence_log_entries if e.get('learning_phase', 0) <= max_phase_for_analysis and e.get('is_context_highly_relevant', True)]
        # print(f"🔮 Using {len(filtered_log_entries)} relevant log entries for meta-analysis up to phase {max_phase_for_analysis}.")
        if not filtered_log_entries: return {"status": "Filtered occurrence log empty."}
        symbol_counts = self.fn_Counter(entry['symbol'] for entry in filtered_log_entries)
        potential_loops = defaultdict(lambda: {"occurrences": 0, "emotions": set(), "contexts": [], "phases_seen": set()})
        for entry in filtered_log_entries:
            symbol = entry['symbol']
            if symbol_counts[symbol] >= min_occurrences: 
                potential_loops[symbol]["occurrences"] += 1; potential_loops[symbol]["emotions"].add(entry['emotion_in_context'])
                potential_loops[symbol]["contexts"].append(entry['context']); potential_loops[symbol]["phases_seen"].add(entry.get('learning_phase', 0))
        newly_bound_meta_symbols_info = []
        active_lexicon_for_check = self._get_active_symbol_lexicon(current_phase, directives)
        for symbol, data in potential_loops.items():
            if data["occurrences"] >= min_occurrences and len(data['emotions']) >= min_distinct_emotions:
                if "⟳" in symbol or symbol in self.meta_symbols: continue
                original_symbol_data = active_lexicon_for_check.get(symbol)
                original_name = original_symbol_data.get("name", symbol) if original_symbol_data else symbol
                new_meta_token, new_meta_name = f"{symbol}⟳", f"{original_name} Cycle"
                all_loop_keywords = [kw for ctx in data['contexts'][:5] for kw in self.fn_extract_keywords(ctx)]
                common_keywords_in_loop = self.fn_Counter(all_loop_keywords).most_common(3)
                derived_summary = (f"Recurring pattern: '{original_name}' with emotions {list(data['emotions'])[:3]}. Context keywords: {', '.join(k[0] for k in common_keywords_in_loop)}.")
                bound_meta_dict = self._bind_meta_symbol(original_token=symbol, new_token=new_meta_token, name=new_meta_name, summary=derived_summary,
                    base_keywords=original_symbol_data.get("keywords", []) if original_symbol_data else [], learning_phase=current_phase )
                if bound_meta_dict: newly_bound_meta_symbols_info.append(bound_meta_dict)
        status_msg = "New meta-symbols created." if newly_bound_meta_symbols_info else "No new meta-symbols from analysis."
        return {"status": status_msg, "new_meta_symbols": newly_bound_meta_symbols_info}
        
    def _bind_meta_symbol(self, original_token, new_token, name, summary, base_keywords=None, learning_phase=1): 
        active_lexicon = self._get_active_symbol_lexicon(learning_phase) 
        if new_token in active_lexicon:
            print(f"[WARNING] Meta-symbol {new_token} already exists. Skipping.") 
            return None
        timestamp = datetime.utcnow().isoformat()
        meta_keywords = list(set(self.fn_extract_keywords(name + " " + summary) + (base_keywords or [])))
        new_meta_entry = {"name": name, "symbol": new_token, "based_on": original_token, "summary": summary,
            "keywords": meta_keywords, "core_meanings": [kw.lower() for kw in meta_keywords[:3]],
            "emotion_profile": {}, "origin": "meta_binding", "created": timestamp,
            "resonance_weight": 0.7, "learning_phase": learning_phase}
        self.meta_symbols[new_token] = new_meta_entry; self._save_meta_symbols()
        print(f"[INFO] Created meta-symbol '{new_token}' ({name}) based on '{original_token}' in Phase {learning_phase}.") 
        self.fn_add_symbol_to_memory(symbol=new_token, name=name, keywords=meta_keywords, emotions={},
            example_text=summary, origin="meta_binding", learning_phase=learning_phase)
        return new_meta_entry
        
    def get_symbol_details(self, symbol_token, current_phase=4, directives=None): 
        active_lexicon = self._get_active_symbol_lexicon(current_phase, directives)
        return active_lexicon.get(symbol_token)

class DynamicBridge: 
    def __init__(self, logic_node: LogicNode, symbolic_node: SymbolicNode, curriculum_manager):
        self.logic_node = logic_node
        self.symbolic_node = symbolic_node
        self.curriculum_manager = curriculum_manager 
        print("🌉 DynamicBridge initialized.")

    def route_and_respond(self, text_input, source_url=None, is_user_interaction=False, is_highly_relevant_for_phase=True):
        current_phase = self.curriculum_manager.get_current_phase()
        directives = self.curriculum_manager.get_processing_directives()
        # print(f"🌉 DynamicBridge (Phase {current_phase}, Relevant: {is_highly_relevant_for_phase}) processing: '{text_input[:30]}...'")
        
        emotions_output = EH_EmotionHandler.predict_emotions(text_input)

        logic_output = self.logic_node.process_input_for_facts(
            text_input, 
            source_type="web_chunk" if source_url else ("user_direct" if is_user_interaction else "internal_curated"), 
            source_url=source_url, current_phase=current_phase,
            is_highly_relevant_for_phase=is_highly_relevant_for_phase
        )
        
        symbolic_output = self.symbolic_node.process_input_for_symbols(
            text_input, source_url=source_url, current_phase=current_phase,
            directives=directives, pre_detected_emotions_output=emotions_output,
            is_highly_relevant_for_phase=is_highly_relevant_for_phase
        )

        TL_TrailLog.log_dynamic_bridge_processing_step(
            text_input=text_input, source_url=source_url, current_phase=current_phase,
            directives=directives, is_highly_relevant_for_phase=is_highly_relevant_for_phase,
            detected_emotions_output=emotions_output,
            logic_node_output=logic_output, symbolic_node_output=symbolic_output,
            final_response_for_user=None if not is_user_interaction else "Response_Generation_Pending" 
        )

        if not is_user_interaction: return None 

        response_parts = []
        response_parts.append(f"[Responding in context of Phase {current_phase}: {self.curriculum_manager.get_phase_description()}]")
        response_parts.append(f"Input Text Emotional Tone (Top 3): {[(e, round(s,2)) for e,s in emotions_output.get('verified',[])[:3]]}")
        filtered_logic_memories = []
        max_phase_mem_access = directives.get("logic_node_access_max_phase", current_phase)
        for mem in logic_output["retrieved_memories"]:
            if mem.get("phase_learned", 0) <= max_phase_mem_access:
                if current_phase > 1 or mem.get("original_source_type") != "web_low_relevance_p1":
                    filtered_logic_memories.append(mem)
        phase_appropriate_symbols = symbolic_output["matched_symbols"]
        if filtered_logic_memories:
            response_parts.append("\n[Information Recall (Contextual Relevance)]:")
            for mem in filtered_logic_memories[:2]:
                 response_parts.append(f"  [Fact (Learned Phase {mem['phase_learned']})] \"{mem['text'][:100]}...\" (Similarity: {mem['similarity_score']:.2f}, Trust: {mem['trust_level']})")
        if phase_appropriate_symbols:
            response_parts.append("\n[Symbolic Interpretation (Contextual Resonance)]:")
            for sym_info in phase_appropriate_symbols[:3]:
                 symbol_details_for_response = self.symbolic_node.get_symbol_details(sym_info['symbol'], current_phase, directives)
                 if symbol_details_for_response: 
                    response_parts.append(f"  [Symbol (Defined Phase {symbol_details_for_response.get('learning_phase',0)})] {sym_info['symbol']} ({sym_info.get('name', 'N/A')}), "
                                        f"Current Emotional Weight: {sym_info.get('emotional_weight',0.0):.2f}")
        elif symbolic_output["generated_symbol_details"]:
            gsd = symbolic_output["generated_symbol_details"]
            response_parts.append(f"\n[Newly Emerged Symbol (Phase {gsd.get('learning_phase',current_phase)})]: {gsd.get('symbol')} ({gsd.get('name')})")
        final_response_str = "\n".join(response_parts)
        if len(response_parts) <= 2 : 
            final_response_str = f"{response_parts[0]}\n{response_parts[1]}\nI've processed that. No specific output to share based on current phase access rules."
        return final_response_str

class CurriculumManager: 
    def __init__(self, initial_seed_symbols_count=0): 
        self.current_phase = 1
        self.phase_data_sources_keywords = {
            1: ["computation theory", "logic gate", "binary code", "algorithm", "data structure", "system architecture", "artificial intelligence definition", "machine learning basics", "programming language principle", "cpu architecture", "memory management os", "turing machine", "von neumann architecture", "information theory"],
            2: ["history of science", "timeline of earth", "physics laws", "biology evolution", "world geography", "astronomy basics", "scientific method application", "ancient civilizations", "cultural anthropology overview"],
            3: ["emotion theory", "psychology basics", "narrative structure", "poetry analysis", "human relationships", "empathy development", "ethical dilemma", "character archetype"],
            4: ["philosophy of mind", "metaphysics concepts", "paradox examples", "cosmology advanced theory", "quantum physics interpretation", "comparative mythology", "consciousness studies", "symbolism deep dive"]
        }
        self.phase_details = {
            1: "Foundational Self-Model: Computational identity, systems theory, AI architecture.",
            2: "Contextual Integration: Historical, scientific, broad cultural corpora.",
            3: "Emotional-Symbolic Differentiation: Psychology, literature, empathy layers.",
            4: "Abstract Expansion: Philosophy, metaphysics, paradox, complex mythologies."
        }
        self.phase_metrics = {
            "processed_relevant_chunks_phase_1": 0, "vector_memory_size": 0, "symbol_lexicon_size": 0, 
            "meta_symbols_created": 0, "symbol_emotion_map_richness": 0, 
            "processed_relevant_urls_phase_2": 0, "processed_relevant_urls_phase_3": 0, "processed_relevant_urls_phase_4": 0,
        }
        
        seed_len = initial_seed_symbols_count 
        self.phase_advancement_thresholds = {
            1: {"processed_relevant_chunks_phase_1": 50, "symbol_lexicon_size": seed_len + 10},
            2: {"processed_relevant_urls_phase_2": 5, "vector_memory_size": 100, "symbol_emotion_map_richness": 10},
            3: {"processed_relevant_urls_phase_3": 5, "meta_symbols_created": 1, "symbol_lexicon_size": seed_len + 25},
            4: {"processed_relevant_urls_phase_4": 3} 
        }
        print(f"📚 CurriculumManager initialized. Current phase: {self.current_phase} - {self.get_phase_description()}")

    def get_current_phase(self): return self.current_phase
    def get_phase_description(self): return self.phase_details.get(self.current_phase, "Unknown Phase")
    def update_metrics(self, logic_node: LogicNode, symbolic_node: SymbolicNode, processed_url_is_relevant=False, processed_chunk_is_relevant_in_phase_1=False):
        if logic_node.memory_path.exists():
             with open(logic_node.memory_path, "r", encoding="utf-8") as f:
                try: self.phase_metrics["vector_memory_size"] = len(json.load(f))
                except json.JSONDecodeError: pass
        active_lexicon = symbolic_node._get_active_symbol_lexicon(self.current_phase, self.get_processing_directives()) 
        self.phase_metrics["symbol_lexicon_size"] = len(active_lexicon)
        self.phase_metrics["meta_symbols_created"] = len(symbolic_node.meta_symbols)
        s_emo_map = SEU_SymbolEmotionUpdater.load_emotion_map()
        richness = sum(1 for profile in s_emo_map.values() if len([s for s in profile.values() if s > 0.5]) >= 3)
        self.phase_metrics["symbol_emotion_map_richness"] = richness
        if processed_url_is_relevant: 
            if self.current_phase == 2: self.phase_metrics["processed_relevant_urls_phase_2"] += 1
            elif self.current_phase == 3: self.phase_metrics["processed_relevant_urls_phase_3"] += 1
            elif self.current_phase == 4: self.phase_metrics["processed_relevant_urls_phase_4"] += 1
        if processed_chunk_is_relevant_in_phase_1 and self.current_phase == 1:
            self.phase_metrics["processed_relevant_chunks_phase_1"] +=1
            
    def advance_phase_if_ready(self):
        if self.current_phase >= 4: return False 
        current_thresholds = self.phase_advancement_thresholds.get(self.current_phase)
        if not current_thresholds: return False
        ready_to_advance = all(self.phase_metrics.get(metric, 0) >= threshold_value 
                               for metric, threshold_value in current_thresholds.items())
        if ready_to_advance:
            self.current_phase += 1
            print(f"🎉📚 Curriculum Advanced to Phase {self.current_phase}: {self.get_phase_description()}")
            # Reset metrics for the new phase if they are phase-specific counters
            if self.current_phase == 2: self.phase_metrics["processed_relevant_urls_phase_2"] = 0
            if self.current_phase == 3: self.phase_metrics["processed_relevant_urls_phase_3"] = 0
            if self.current_phase == 4: self.phase_metrics["processed_relevant_urls_phase_4"] = 0
            if self.current_phase == 1: self.phase_metrics["processed_relevant_chunks_phase_1"] = 0 # Should not happen if advancing from 1
            return True
        return False
        
    def get_processing_directives(self):
        phase = self.current_phase
        directives = {"phase": phase, "info": self.get_phase_description()}
        directives["logic_node_access_max_phase"] = phase 
        directives["symbolic_node_access_max_phase"] = phase
        directives["meta_symbol_analysis_max_phase"] = phase
        directives["allow_new_symbol_generation"] = True 
        if phase == 1:
            directives["focus"] = "self_model_computational_identity"
            directives["allow_web_scraping"] = True 
            directives["data_source_filter_keywords"] = self.phase_data_sources_keywords[1]
            directives["phase1_strict_chunk_relevance_threshold"] = 0.25 
            directives["phase1_min_keyword_matches"] = 2 
            directives["knowledge_integration_level"] = "isolated_foundational"
        elif phase == 2:
            directives["focus"] = "contextual_knowledge_factual_historical_scientific"
            directives["allow_web_scraping"] = True
            directives["data_source_filter_keywords"] = self.phase_data_sources_keywords[2]
            directives["knowledge_integration_level"] = "contextual_grounding_cross_linking_phase1_2"
        elif phase == 3:
            directives["focus"] = "emotional_symbolic_depth_psychological_literary"
            directives["allow_web_scraping"] = True
            directives["data_source_filter_keywords"] = self.phase_data_sources_keywords[3]
            directives["knowledge_integration_level"] = "affective_symbolic_linking_to_factual_context"
        elif phase == 4:
            directives["focus"] = "abstract_conceptualization_philosophy_paradox_myth"
            directives["allow_web_scraping"] = True
            directives["data_source_filter_keywords"] = self.phase_data_sources_keywords[4]
            directives["knowledge_integration_level"] = "holistic_synthesis_cross_phase_correlation"
        return directives

