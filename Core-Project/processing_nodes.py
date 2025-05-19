# processing_nodes.py
import json
import hashlib
from pathlib import Path
from datetime import datetime 
import re 
from collections import Counter, defaultdict

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


def detect_content_type(text_input: str, spacy_nlp_instance=None) -> str:
    if not text_input or not isinstance(text_input, str):
        return "ambiguous"
    text_lower = text_input.lower()
    factual_markers = [
        "according to", "study shows", "research indicates", "published in", "cited in", "evidence suggests",
        "data shows", "statistics indicate", "found that", "confirmed that", "demonstrated that",
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "dr.", "prof.", "university of", "institute of", "journal of", ".gov", ".edu", ".org",
        "theorem", "equation", "formula", "law of", "principle of",
        "born on", "died on", "founded in", "established in",
        "kg", "km", "meter", "liter", "celsius", "fahrenheit", "%", "$", "€", "¥"
    ]
    symbolic_markers = [
        "love", "hate", "fear", "joy", "sadness", "anger", "hope", "dream", "nightmare",
        "like a", "as if", "metaphor", "symbolizes", "represents", "signifies", "embodies", "evokes",
        "spirit", "soul", "ghost", "magic", "myth", "legend", "folklore", "ritual", "omen",
        "🔥", "💧", "🌀", "💡", "🧩", "♾️", 
        "heart", "light", "darkness", "shadow", "journey", "quest", "fate", "destiny",
        "feels like", "seems as though", "one might say", "could be seen as"
    ]
    f_count = sum(marker in text_lower for marker in factual_markers)
    s_count = sum(marker in text_lower for marker in symbolic_markers)
    numbers = re.findall(r'(?<!\w)[-+]?\d*\.?\d+(?!\w)', text_lower)
    if len(numbers) > 2: f_count += 1
    if len(numbers) > 5: f_count +=1
    if spacy_nlp_instance:
        doc = spacy_nlp_instance(text_input[:spacy_nlp_instance.max_length])
        entity_factual_boost = 0
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
                entity_factual_boost += 0.5
            elif ent.label_ in ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"]:
                entity_factual_boost += 0.25
        f_count += entity_factual_boost
    if f_count > s_count * 1.5: return "factual"
    elif s_count > f_count * 1.5: return "symbolic"
    else:
        if f_count == 0 and s_count == 0:
            if len(text_input.split()) < 5 : return "ambiguous"
            if len(numbers) > 0 : return "factual"
            return "ambiguous"
        elif f_count > s_count : return "factual"
        elif s_count > f_count : return "symbolic"
        return "ambiguous"

class LogicNode:
    def __init__(self, vector_memory_path_str=None):
        self.memory_path = Path(vector_memory_path_str) if vector_memory_path_str else vm_memory_path
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists() or self.memory_path.stat().st_size == 0:
            with open(self.memory_path, "w", encoding="utf-8") as f: json.dump([], f)
        print(f"🧠 LogicNode initialized. Memory path: {self.memory_path}")
    def store_memory(self, text_input, source_url=None, source_type="web_scrape",
                     current_processing_phase=0, target_storage_phase=0,     
                     is_highly_relevant_for_current_phase=False, is_shallow_content=False,
                     confidence_score=0.7):
        exploration_depth = "shallow" if is_shallow_content else "deep"
        source_trust = "unknown" 
        if source_url:
            if "wikipedia.org" in source_url or "stanford.edu" in source_url:
                source_trust = "high_academic_encyclopedic"
            elif any(domain_part in source_url for domain_part in [".gov", ".edu", "ac.uk", "uni-", "-university", "ieee.org", "nature.com", "sciencemag.org"]):
                source_trust = "high_authoritative"
            elif any(bad_domain in source_url for bad_domain in ["randomblog.blogspot.com", "forum.example", "personal-site.tripod"]):
                source_trust = "low_unverified"
        vm_store_vector(text=text_input, source_type=source_type, source_url=source_url,
            learning_phase=target_storage_phase, exploration_depth=exploration_depth,
            confidence=confidence_score, source_trust=source_trust)
        return {"status": "success", "action": "stored_logic_memory", "target_phase": target_storage_phase}
    def retrieve_memories(self, query_text, current_phase_directives):
        max_phase = current_phase_directives.get("logic_node_access_max_phase", 0)
        min_conf = current_phase_directives.get("logic_node_min_confidence_retrieve", 0.3)
        results = vm_retrieve_similar_vectors(query_text, max_phase_allowed=max_phase, top_n=5, min_confidence=min_conf)
        formatted_results = []
        for score, item in results:
            formatted_results.append({"text": item.get("text", "")[:150] + "...", "similarity": round(score, 4),
                "phase_learned": item.get("learning_phase", "N/A"), "source_url": item.get("source_url", "N/A"),
                "confidence": item.get("confidence", "N/A")})
        return {"retrieved_memories_count": len(formatted_results), "top_retrieved_texts": formatted_results[:2]}

class SymbolicNode:
    def __init__(self, seed_symbols_path_str="data/seed_symbols.json",
                 symbol_memory_path_str="data/symbol_memory.json",
                 symbol_occurrence_log_path_str="data/symbol_occurrence_log.json",
                 symbol_emotion_map_path_str="data/symbol_emotion_map.json",
                 meta_symbols_path_str="data/meta_symbols.json"):
        self.seed_symbols_path = Path(seed_symbols_path_str)
        self.symbol_memory_path = Path(symbol_memory_path_str)
        self.symbol_occurrence_log_path = Path(symbol_occurrence_log_path_str)
        self.symbol_emotion_map_path = Path(symbol_emotion_map_path_str)
        self.meta_symbols_path = Path(meta_symbols_path_str)
        self._ensure_data_files()
        self.seed_symbols = P_Parser.load_seed_symbols(file_path=self.seed_symbols_path)
        self.symbol_memory = SM_SymbolMemory.load_symbol_memory(file_path=self.symbol_memory_path)
        self.meta_symbols = self._load_meta_symbols()
        print(f"⚛️ SymbolicNode initialized. Loaded {len(self.seed_symbols)} seed symbols, {len(self.symbol_memory)} learned symbols, {len(self.meta_symbols)} meta-symbols.")
    def _ensure_data_files(self):
        paths = [self.symbol_memory_path, self.symbol_occurrence_log_path, self.symbol_emotion_map_path, self.meta_symbols_path]
        self.seed_symbols_path.parent.mkdir(parents=True, exist_ok=True)
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists() or p.stat().st_size == 0:
                if p.suffix == '.json':
                    with open(p, "w", encoding="utf-8") as f:
                        if "map" in p.name or "meta" in p.name or ("memory" in p.name and "occurrence" not in p.name) : json.dump({}, f)
                        elif "occurrence" in p.name: json.dump({"entries": []}, f)
                        else: json.dump({},f)
    def _load_meta_symbols(self):
        if self.meta_symbols_path.exists() and self.meta_symbols_path.stat().st_size > 0:
            with open(self.meta_symbols_path, "r", encoding="utf-8") as f:
                try: return json.load(f)
                except json.JSONDecodeError: print(f"[SN-WARNING] Meta-symbols file {self.meta_symbols_path} corrupted."); return {}
        return {}
    def _save_meta_symbols(self):
        with open(self.meta_symbols_path, "w", encoding="utf-8") as f:
            json.dump(self.meta_symbols, f, indent=2, ensure_ascii=False)
    def _get_active_symbol_lexicon(self, current_phase_directives):
        max_phase = current_phase_directives.get("symbolic_node_access_max_phase", 0)
        active_lexicon = {}
        if self.seed_symbols:
            for token, details in self.seed_symbols.items():
                if details.get("learning_phase", 0) <= max_phase: active_lexicon[token] = details
        self.symbol_memory = SM_SymbolMemory.load_symbol_memory(file_path=self.symbol_memory_path)
        if self.symbol_memory:
            for token, details in self.symbol_memory.items():
                if details.get("learning_phase", 0) <= max_phase: active_lexicon[token] = details
        self.meta_symbols = self._load_meta_symbols()
        if self.meta_symbols:
            for token, details in self.meta_symbols.items():
                base_symbol_token = details.get("based_on")
                if base_symbol_token and base_symbol_token in active_lexicon:
                    active_lexicon[token] = {"name": details.get("name", token), "keywords": details.get("keywords", []) + P_Parser.extract_keywords(details.get("summary","")), "core_meanings": [details.get("summary", "meta-symbol")], "emotions": [], "archetypes": [], "learning_phase": active_lexicon[base_symbol_token].get("learning_phase",0), "resonance_weight": details.get("resonance_weight", 0.8), "is_meta": True}
        return active_lexicon
    def process_input_for_symbols(self, text_input, detected_emotions_output, 
                                  current_processing_phase, target_storage_phase, 
                                  current_phase_directives, source_url=None, 
                                  is_highly_relevant_for_current_phase=False, is_shallow_content=False,
                                  confidence_score=0.7):
        active_lexicon = self._get_active_symbol_lexicon(current_phase_directives)
        verified_emotions = detected_emotions_output.get("verified", []) if isinstance(detected_emotions_output, dict) else []
        matched_symbols_weighted = P_Parser.parse_with_emotion(text_input, verified_emotions, current_lexicon=active_lexicon)
        if matched_symbols_weighted and verified_emotions:
            SEU_SymbolEmotionUpdater.update_symbol_emotions(matched_symbols_weighted, verified_emotions, file_path=self.symbol_emotion_map_path)
        generated_symbol_details = None
        if not matched_symbols_weighted and current_phase_directives.get("allow_new_symbol_generation", False) and is_highly_relevant_for_current_phase and confidence_score > 0.5: 
            keywords_for_gen = P_Parser.extract_keywords(text_input)
            if keywords_for_gen:
                new_symbol_proposal = SG_Refactored_SymbolGenerator.generate_symbol_from_context(text_input, keywords_for_gen, verified_emotions)
                SM_SymbolMemory.add_symbol(symbol_token=new_symbol_proposal['symbol'], name=new_symbol_proposal['name'], keywords=new_symbol_proposal['keywords'], initial_emotions=new_symbol_proposal['emotions'], example_text=text_input, origin=new_symbol_proposal['origin'], learning_phase=target_storage_phase, resonance_weight=new_symbol_proposal.get('resonance_weight', 0.5), file_path=self.symbol_memory_path)
                self.symbol_memory = SM_SymbolMemory.load_symbol_memory(file_path=self.symbol_memory_path)
                generated_symbol_details = new_symbol_proposal
                print(f"    🌱 New symbol generated: {new_symbol_proposal['symbol']} ({new_symbol_proposal['name']}) for phase {target_storage_phase}")
                matched_symbols_weighted.append({'symbol': new_symbol_proposal['symbol'], 'name': new_symbol_proposal['name'], 'matched_keyword': new_symbol_proposal['keywords'][0] if new_symbol_proposal['keywords'] else 'emergent', 'final_weight': 0.7, 'influencing_emotions': new_symbol_proposal['emotions']})
        for sym_match in matched_symbols_weighted:
            primary_emotion_in_context_str = verified_emotions[0][0] if verified_emotions else "(unspecified)"
            UM_UserMemory.add_user_memory_entry(
                symbol=sym_match['symbol'], context_text=text_input, emotion_in_context=primary_emotion_in_context_str, 
                source_url=source_url, learning_phase=target_storage_phase, 
                is_context_highly_relevant=is_highly_relevant_for_current_phase,
                file_path=self.symbol_occurrence_log_path
            )
        summary_matched_symbols = []
        for s_match in matched_symbols_weighted[:3]:
            symbol_details_from_mem = SM_SymbolMemory.get_symbol_details(s_match.get("symbol"), file_path=self.symbol_memory_path)
            summary_matched_symbols.append({"symbol": s_match.get("symbol"), "name": s_match.get("name", symbol_details_from_mem.get("name", "Unknown")), "emotional_weight": s_match.get("final_weight"), "influencing_emotions": s_match.get("influencing_emotions", [])})
        return {"matched_symbols_count": len(summary_matched_symbols), "top_matched_symbols": summary_matched_symbols, "generated_symbol": generated_symbol_details, "top_detected_emotions_input": verified_emotions[:3]}
    
    # MODIFIED: run_meta_symbol_analysis to ensure new_meta_entry has all needed fields for add_symbol
    def run_meta_symbol_analysis(self, max_phase_to_consider):
        print(f"[SymbolicNode] Running meta-symbol analysis (considering up to phase {max_phase_to_consider})...")
        occurrence_log = UM_UserMemory.load_user_memory(file_path=self.symbol_occurrence_log_path)
        phase_filtered_log = [entry for entry in occurrence_log if entry.get("learning_phase", 0) <= max_phase_to_consider]
        if not phase_filtered_log: print("    No symbol occurrences found for meta-symbol generation."); return

        symbol_emotion_counts = defaultdict(lambda: {"count": 0, "emotions": Counter()})
        for entry in phase_filtered_log:
            symbol_token, emotion = entry["symbol"], entry["emotion_in_context"]
            symbol_emotion_counts[symbol_token]["count"] += 1
            if emotion != "(unspecified)": symbol_emotion_counts[symbol_token]["emotions"][emotion] += 1
        
        MIN_OCCURRENCES_FOR_META, MIN_DISTINCT_EMOTIONS_FOR_META = 5, 2
        for symbol_token, data in symbol_emotion_counts.items():
            if data["count"] >= MIN_OCCURRENCES_FOR_META and len(data["emotions"]) >= MIN_DISTINCT_EMOTIONS_FOR_META:
                meta_token_candidate_base = symbol_token + "⟳"
                self.meta_symbols = self._load_meta_symbols() 
                self.symbol_memory = SM_SymbolMemory.load_symbol_memory(file_path=self.symbol_memory_path) 
                if meta_token_candidate_base in self.meta_symbols or meta_token_candidate_base in self.symbol_memory: continue

                base_symbol_details = SM_SymbolMemory.get_symbol_details(symbol_token, file_path=self.symbol_memory_path) or self.seed_symbols.get(symbol_token, {})
                base_name = base_symbol_details.get("name", symbol_token)
                top_emotions = [emo for emo, count in data["emotions"].most_common(3)]
                
                # Ensure new_meta_entry has all fields add_symbol might expect if symbol_details_override is used,
                # including those it might try to append to (like vector_examples) or increment (usage_count).
                new_meta_entry = {
                    "name": f"{base_name} Cycle", 
                    "based_on": symbol_token,
                    "summary": f"Recurring pattern or complex emotional field for '{base_name}'. Often involves: {', '.join(top_emotions)}.",
                    "keywords": base_symbol_details.get("keywords", []) + ["cycle", "recursion", "complex emotion"],
                    "core_meanings": [f"recurring {base_name}", "emotional complexity"], 
                    "emotions": [], # Meta-symbols define their own emotional basis through usage
                    "emotion_profile": {}, # Initialize empty
                    "archetypes": ["transformation", "pattern"],
                    "created_at": datetime.utcnow().isoformat(), 
                    "updated_at": datetime.utcnow().isoformat(),
                    "origin": "meta_analysis", # Origin from meta-analysis perspective
                    "learning_phase": max_phase_to_consider, 
                    "resonance_weight": round(base_symbol_details.get("resonance_weight", 0.5) * 1.2, 2),
                    "vector_examples": [], # Initialize as empty list
                    "usage_count": 0       # Initialize usage count
                }
                self.meta_symbols[meta_token_candidate_base] = new_meta_entry # Store in meta_symbols.json
                print(f"    🔗 New meta-symbol bound: {meta_token_candidate_base} based on {symbol_token}")
                
                # Add to main symbol_memory.json
                SM_SymbolMemory.add_symbol(
                    symbol_token=meta_token_candidate_base, 
                    # name, keywords etc. will be taken from symbol_details_override
                    name=new_meta_entry["name"], # Redundant if override works perfectly but good for clarity
                    keywords=new_meta_entry["keywords"], 
                    initial_emotions=[], 
                    example_text=new_meta_entry["summary"], # This will be added as the first example
                    origin="meta_emergent", # This will be the origin in symbol_memory
                    learning_phase=max_phase_to_consider, 
                    resonance_weight=new_meta_entry["resonance_weight"],
                    symbol_details_override=new_meta_entry, # Pass the fully formed dict
                    file_path=self.symbol_memory_path
                )
        self._save_meta_symbols()


class CurriculumManager: # No changes from previous full script
    def __init__(self):
        self.current_phase = 1
        self.max_phases = 4
        self.phase_metrics = {phase: {"chunks_processed": 0, "relevant_chunks_processed": 0, "urls_visited": 0, "new_symbols_generated":0, "meta_symbols_bound":0} for phase in range(1, self.max_phases + 1)}
        self.phase_data_sources_keywords = {
             1: {"primary": ["code", "algorithm", "software", "hardware", "computer", "programming", "language construct", "binary", "data structure", "turing machine", "von neumann", "cpu", "memory unit", "logic gate", "boolean algebra", "processor", "silicon", "semiconductor", "compiler", "operating system", "network protocol"], "secondary": ["technology", "system", "architecture", "computation", "information theory", "digital logic", "circuit", "bus", "interface"], "anti_keywords": ["history", "war", "philosophy", "art", "novel", "poem", "ancient", "medieval", "renaissance", "century", "mythology", "emotion", "belief", "spirituality", "quantum field", "metaphysics", "geology", "biology", "astronomy"]},
             2: {"primary": ["emotion", "feeling", "affect", "mood", "sentiment", "psychology", "cognition", "perception", "bias", "stress", "trauma", "joy", "sadness", "anger", "fear", "surprise", "disgust", "empathy"], "secondary": ["myth", "symbolism", "archetype", "metaphor", "narrative structure", "dream analysis", "subconscious", "consciousness studies (psychological)", "attachment theory", "behavioral psychology", "cognitive dissonance"], "anti_keywords": ["quantum physics", "spacetime", "relativity", "particle physics", "geopolitics", "economic policy", "software engineering", "circuit design"]},
             3: {"primary": ["history", "event", "timeline", "discovery", "science", "physics", "biology", "chemistry", "geology", "astronomy", "year", "date", "century", "civilization", "empire", "war", "revolution", "mineral", "element", "energy", "matter", "force", "motion", "genetics", "evolution"], "secondary": ["archaeology", "anthropology", "society", "invention", "exploration", "culture", "human migration", "industrial revolution", "world war", "cold war", "space race", "internet history", "1990", "1991", "2000s"], "anti_keywords": ["metaphysical philosophy", "esoteric spirituality", "literary critique (unless historical)", "fine art analysis (unless historical context)"]},
             4: {"primary": ["philosophy", "metaphysics", "ontology", "epistemology", "ethics", "religion", "spirituality", "quantum mechanics", "quantum field theory", "general relativity", "string theory", "consciousness (philosophical/speculative)", "theorem", "paradox", "reality", "existence", "multiverse", "simulation theory", "artificial general intelligence", "emergence", "complexity theory", "chaos theory"], "secondary": ["logic (philosophical)", "reason", "truth", "meaning", "purpose", "free will", "determinism", "theology", "cosmology (speculative)", "future studies", "transhumanism", "veda", "upanishad", "dharma", "karma", "moksha", "atman", "brahman"], "anti_keywords": ["pop culture critique", "celebrity gossip", "daily news (unless highly theoretical implications)", "product reviews"]}
        }
        self.phase_info_descriptions = {1: "Computational Identity: Foundational understanding of computation, computer science, logic, and the AI's own architectural concepts.", 2: "Emotional and Symbolic Awareness: Learning about human (and potentially machine) emotions, psychological concepts, foundational myths, and basic symbolism.", 3: "Historical and Scientific Context: Broadening knowledge to include world history, major scientific disciplines (physics, biology, etc.), and how events and discoveries are situated in time.", 4: "Abstract and Philosophical Exploration: Engaging with complex, abstract, and speculative ideas like philosophy, metaphysics, advanced/theoretical science (quantum, cosmology), ethics, and the nature of reality/consciousness."}
    def get_current_phase(self): return self.current_phase
    def get_max_phases(self): return self.max_phases
    def get_phase_context_description(self, phase): return self.phase_info_descriptions.get(phase, "General Learning Phase")
    def get_processing_directives(self, phase):
        if not (1 <= phase <= self.max_phases): phase = 1 
        directives = {"phase": phase, "info": self.get_phase_context_description(phase), "logic_node_access_max_phase": phase, "symbolic_node_access_max_phase": phase, "meta_symbol_analysis_max_phase": phase, "allow_new_symbol_generation": True, "focus": f"phase_{phase}_focus", "allow_web_scraping": True, "phase_keywords_primary": self.phase_data_sources_keywords.get(phase, {}).get("primary", []), "phase_keywords_secondary": self.phase_data_sources_keywords.get(phase, {}).get("secondary", []), "phase_keywords_anti": self.phase_data_sources_keywords.get(phase, {}).get("anti", []), "phase_min_primary_keyword_matches_for_link_follow": 1, "phase_min_total_keyword_score_for_link_follow": 2.5,    "phase_min_primary_keyword_matches_for_chunk_relevance": 1, "phase_min_total_keyword_score_for_chunk_relevance": 1.0, "allow_shallow_dive_for_future_phase_links": True, "shallow_dive_max_chars": 500, "max_exploration_depth_from_seed_url": 5, "max_urls_to_process_per_phase_session": 2, "logic_node_min_confidence_retrieve": 0.3, "symbolic_node_min_confidence_retrieve": 0.25, "factual_heuristic_confidence_threshold": 0.6, "symbolic_heuristic_confidence_threshold": 0.5, "link_score_weight_static": 0.6, "link_score_weight_dynamic": 0.4, "max_dynamic_link_score_bonus": 5.0, "max_session_hot_keywords": 20, "min_session_hot_keyword_freq": 2}
        if phase == 1: directives["allow_new_symbol_generation"] = False
        return directives
    def update_metrics(self, phase, chunks_processed_increment=0, relevant_chunks_increment=0, urls_visited_increment=0, new_symbols_increment=0, meta_symbols_increment=0):
        if phase in self.phase_metrics:
            self.phase_metrics[phase]["chunks_processed"] += chunks_processed_increment; self.phase_metrics[phase]["relevant_chunks_processed"] += relevant_chunks_increment; self.phase_metrics[phase]["urls_visited"] += urls_visited_increment; self.phase_metrics[phase]["new_symbols_generated"] += new_symbols_increment; self.phase_metrics[phase]["meta_symbols_bound"] += meta_symbols_increment
    def advance_phase_if_ready(self, current_completed_phase_num):
        metrics = self.phase_metrics.get(current_completed_phase_num)
        if not metrics: return False
        if current_completed_phase_num == 1:
            if metrics["relevant_chunks_processed"] >= 2 and metrics["urls_visited"] >= 1: self.current_phase = 2; return True
        elif current_completed_phase_num == 2:
            if metrics["relevant_chunks_processed"] >= 3 and metrics["new_symbols_generated"] >= 0: self.current_phase = 3; return True
        elif current_completed_phase_num == 3:
            if metrics["relevant_chunks_processed"] >= 3 and metrics["urls_visited"] >= 1: self.current_phase = 4; return True
        elif current_completed_phase_num == 4: print("[CurriculumManager] Phase 4 (max phase) completed."); return False 
        return False

class DynamicBridge:
    def __init__(self, logic_node: LogicNode, symbolic_node: SymbolicNode, curriculum_manager: CurriculumManager):
        self.logic_node = logic_node
        self.symbolic_node = symbolic_node
        self.curriculum_manager = curriculum_manager
        self.trail_logger = TL_TrailLog 
        self.spacy_nlp = P_Parser.nlp if P_Parser.NLP_MODEL_LOADED else None
        print("🌉 DynamicBridge initialized.")
    def _detect_emotions(self, text_input):
        return EH_EmotionHandler.predict_emotions(text_input)
    @staticmethod
    def _score_text_for_phase(text_content, phase_directives):
        text_lower = text_content.lower()
        score, primary_matches, secondary_matches = 0.0, 0, 0 
        for kw in phase_directives.get("phase_keywords_primary", []):
            if kw.lower() in text_lower: score += 2.0; primary_matches += 1
        for kw in phase_directives.get("phase_keywords_secondary", []):
            if kw.lower() in text_lower: score += 1.0; secondary_matches += 1 
        for kw in phase_directives.get("phase_keywords_anti", []):
            if kw.lower() in text_lower: score -= 3.0
        return score, primary_matches, secondary_matches
    def is_chunk_relevant_for_current_phase(self, text_chunk, current_processing_phase_num, directives):
        score, prim_matches, _ = self._score_text_for_phase(text_chunk, directives)
        min_prim_matches = directives.get("phase_min_primary_keyword_matches_for_chunk_relevance", 1)
        min_total_score = directives.get("phase_min_total_keyword_score_for_chunk_relevance", 1.0)
        return prim_matches >= min_prim_matches and score >= min_total_score
    def determine_target_storage_phase(self, text_chunk, current_processing_phase_num):
        best_phase, highest_score = current_processing_phase_num, -float('inf')
        current_phase_directives = self.curriculum_manager.get_processing_directives(current_processing_phase_num)
        current_score, _, _ = self._score_text_for_phase(text_chunk, current_phase_directives)
        highest_score = current_score
        min_score_current_phase_relevance = current_phase_directives.get("phase_min_total_keyword_score_for_chunk_relevance", 1.0)
        for phase_idx in range(1, self.curriculum_manager.get_max_phases() + 1):
            phase_directives_for_eval = self.curriculum_manager.get_processing_directives(phase_idx)
            score, _, _ = self._score_text_for_phase(text_chunk, phase_directives_for_eval)
            min_target_phase_score_relevance = phase_directives_for_eval.get("phase_min_total_keyword_score_for_chunk_relevance", 1.0)
            if score > highest_score and score >= min_target_phase_score_relevance:
                highest_score, best_phase = score, phase_idx
            elif current_score < min_score_current_phase_relevance and score >= min_target_phase_score_relevance:
                if score > highest_score : highest_score, best_phase = score, phase_idx
                elif best_phase == current_processing_phase_num and highest_score < min_score_current_phase_relevance :
                    highest_score, best_phase = score, phase_idx
        if highest_score < 0.5 and best_phase != current_processing_phase_num:
            if current_score >= 0.1 : return current_processing_phase_num
            return best_phase 
        elif highest_score < 0.1 : return current_processing_phase_num
        return best_phase
    def route_chunk_for_processing(self, text_input, source_url, 
                                   current_processing_phase, target_storage_phase,
                                   is_highly_relevant_for_current_phase, is_shallow_content=False,
                                   base_confidence=0.7):
        log_entry_id = f"step_{datetime.utcnow().isoformat().replace(':', '-').replace('.', '-')}_{hashlib.md5(text_input.encode('utf-8')).hexdigest()[:8]}"
        detected_emotions_output = self._detect_emotions(text_input)
        current_phase_processing_directives = self.curriculum_manager.get_processing_directives(current_processing_phase)
        content_type = detect_content_type(text_input, spacy_nlp_instance=self.spacy_nlp)
        effective_confidence = base_confidence
        if not is_highly_relevant_for_current_phase: effective_confidence *= 0.8
        if is_shallow_content: effective_confidence *= 0.7
        effective_confidence = round(max(0.1, effective_confidence), 2)
        logic_node_output, symbolic_node_output = None, None
        if content_type == "factual":
            self.logic_node.store_memory(text_input=text_input, source_url=source_url, source_type="web_scrape_deep_factual" if not is_shallow_content else "web_scrape_shallow_factual", current_processing_phase=current_processing_phase, target_storage_phase=target_storage_phase, is_highly_relevant_for_current_phase=is_highly_relevant_for_current_phase, is_shallow_content=is_shallow_content, confidence_score=effective_confidence)
            logic_node_output = self.logic_node.retrieve_memories(text_input, current_phase_processing_directives)
        elif content_type == "symbolic":
            symbolic_node_output = self.symbolic_node.process_input_for_symbols(text_input=text_input, detected_emotions_output=detected_emotions_output, current_processing_phase=current_processing_phase, target_storage_phase=target_storage_phase, current_phase_directives=current_phase_processing_directives, source_url=source_url, is_highly_relevant_for_current_phase=is_highly_relevant_for_current_phase, is_shallow_content=is_shallow_content, confidence_score=effective_confidence)
            logic_node_output = self.logic_node.retrieve_memories(text_input, current_phase_processing_directives)
        else: # ambiguous
            self.logic_node.store_memory(text_input=text_input, source_url=source_url, source_type="web_scrape_ambiguous_deep" if not is_shallow_content else "web_scrape_ambiguous_shallow", current_processing_phase=current_processing_phase, target_storage_phase=target_storage_phase, is_highly_relevant_for_current_phase=is_highly_relevant_for_current_phase, is_shallow_content=is_shallow_content, confidence_score=round(max(0.1, effective_confidence * 0.65),2))
            logic_node_output = self.logic_node.retrieve_memories(text_input, current_phase_processing_directives)
            symbolic_node_output = self.symbolic_node.process_input_for_symbols(text_input=text_input, detected_emotions_output=detected_emotions_output, current_processing_phase=current_processing_phase, target_storage_phase=target_storage_phase, current_phase_directives=current_phase_processing_directives, source_url=source_url, is_highly_relevant_for_current_phase=is_highly_relevant_for_current_phase, is_shallow_content=is_shallow_content, confidence_score=round(max(0.1, effective_confidence * 0.65),2))
        if logic_node_output is None: logic_node_output = {"retrieved_memories_count": 0, "top_retrieved_texts": []}
        if symbolic_node_output is None: symbolic_node_output = {"matched_symbols_count": 0, "top_matched_symbols": [], "generated_symbol": None, "top_detected_emotions_input": []}
        self.trail_logger.log_dynamic_bridge_processing_step(
            log_id=log_entry_id, text_input=text_input, source_url=source_url,
            current_phase=current_processing_phase, directives=current_phase_processing_directives,
            is_highly_relevant_for_phase=is_highly_relevant_for_current_phase,
            target_storage_phase_for_chunk=target_storage_phase, 
            is_shallow_content=is_shallow_content, 
            detected_emotions_output=detected_emotions_output,
            logic_node_output=logic_node_output, symbolic_node_output=symbolic_node_output,
        )
        new_sym_count = 1 if symbolic_node_output.get("generated_symbol") else 0
        self.curriculum_manager.update_metrics(current_processing_phase, chunks_processed_increment=1, 
            relevant_chunks_increment=1 if is_highly_relevant_for_current_phase else 0,
            new_symbols_increment=new_sym_count)
    def generate_response_for_user(self, user_input_text, source_url=None):
        current_phase = self.curriculum_manager.get_current_phase()
        directives = self.curriculum_manager.get_processing_directives(current_phase)
        target_storage_phase, is_relevant = current_phase, True 
        content_type = detect_content_type(user_input_text, spacy_nlp_instance=self.spacy_nlp)
        self.route_chunk_for_processing(text_input=user_input_text, source_url=source_url, current_processing_phase=current_phase, target_storage_phase=target_storage_phase, is_highly_relevant_for_current_phase=is_relevant, is_shallow_content=False, base_confidence=0.85)
        logic_sum = self.logic_node.retrieve_memories(user_input_text, directives)
        full_emotions_output = self._detect_emotions(user_input_text)
        sym_sum = self.symbolic_node.process_input_for_symbols(user_input_text, full_emotions_output, current_phase, current_phase, directives, source_url, True, False )
        response_parts = [f"[BRIDGE - Phase {current_phase} ({directives.get('info')}) | InputType: {content_type}] Processed."]
        if logic_sum["retrieved_memories_count"] > 0 and logic_sum["top_retrieved_texts"]:
            response_parts.append(f"  Logic Recall: {logic_sum['retrieved_memories_count']} facts. Top: '{logic_sum['top_retrieved_texts'][0]['text'][:40]}...' (Conf: {logic_sum['top_retrieved_texts'][0]['confidence']})")
        if sym_sum["matched_symbols_count"] > 0 and sym_sum["top_matched_symbols"]:
            response_parts.append(f"  Symbolic Matches: {sym_sum['matched_symbols_count']} symbols. Top: {sym_sum['top_matched_symbols'][0]['symbol']} ({sym_sum['top_matched_symbols'][0]['name']})")
        if sym_sum.get("generated_symbol"): response_parts.append(f"  Emerged Symbol: {sym_sum['generated_symbol']['symbol']} ({sym_sum['generated_symbol']['name']})")
        return "\n".join(response_parts)

if __name__ == '__main__':
    print("Testing processing_nodes.py components with all accumulated fixes...")
    if P_Parser.NLP_MODEL_LOADED and P_Parser.nlp is None:
        try: P_Parser.nlp = spacy.load("en_core_web_sm"); print("   spaCy model re-loaded for processing_nodes.py tests.")
        except OSError: print("   spaCy model still not found. Entity heuristic in detect_content_type will be skipped for tests.")
    fact_text_main = "The study published in Nature on May 10th, 2023, confirmed values."
    sym_text_main = "Her laughter was like a dream, a symbol of joy."
    assert detect_content_type(fact_text_main, P_Parser.nlp) == "factual"
    assert detect_content_type(sym_text_main, P_Parser.nlp) == "symbolic"
    TEST_FILE_SUFFIX = "_full_final_test_v3.json" # Incremented suffix for clean test files
    test_logic_mem_path, test_sym_mem_path, test_sym_occur_path, test_sym_emo_map_path, test_meta_sym_path, test_seed_sym_path, test_trail_log_path = [Path(f"data/test_{name}{TEST_FILE_SUFFIX}") for name in ["logic_memory", "symbol_memory", "symbol_occurrence", "symbol_emotion_map", "meta_symbols", "seed_symbols", "trail_log"]]
    for p in [test_logic_mem_path, test_sym_mem_path, test_sym_occur_path, test_sym_emo_map_path, test_meta_sym_path, test_seed_sym_path, test_trail_log_path]:
        if p.exists(): p.unlink()
    with open(test_seed_sym_path, "w", encoding="utf-8") as f: json.dump({"💡": {"name": "Idea", "keywords": ["idea", "thought"], "emotions": ["curiosity"], "learning_phase":0, "resonance_weight": 0.5}}, f)
    logic_node_main_test = LogicNode(vector_memory_path_str=str(test_logic_mem_path))
    symbolic_node_main_test = SymbolicNode(seed_symbols_path_str=str(test_seed_sym_path), symbol_memory_path_str=str(test_sym_mem_path), symbol_occurrence_log_path_str=str(test_sym_occur_path), symbol_emotion_map_path_str=str(test_sym_emo_map_path), meta_symbols_path_str=str(test_meta_sym_path))
    curriculum_manager_main_test = CurriculumManager()
    original_tl_path = TL_TrailLog.TRAIL_LOG_FILE_PATH 
    TL_TrailLog.TRAIL_LOG_FILE_PATH = test_trail_log_path
    dynamic_bridge_main_test = DynamicBridge(logic_node_main_test, symbolic_node_main_test, curriculum_manager_main_test)
    print("\n--- Testing DynamicBridge route_chunk_for_processing (full test with all fixes) ---")
    print("Processing factual text...")
    dynamic_bridge_main_test.route_chunk_for_processing(text_input=fact_text_main, source_url="http://example.com/fact_article", current_processing_phase=1, target_storage_phase=1, is_highly_relevant_for_current_phase=True)
    print("Processing symbolic text...")
    dynamic_bridge_main_test.route_chunk_for_processing(text_input=sym_text_main, source_url="http://example.com/symbolic_story", current_processing_phase=2, target_storage_phase=2, is_highly_relevant_for_current_phase=True)
    mixed_text_main = "The 2023 report felt like a dream, its numbers symbolizing hope."
    print("Processing mixed text (should be ambiguous)...")
    dynamic_bridge_main_test.route_chunk_for_processing(text_input=mixed_text_main, source_url="http://example.com/mixed_report", current_processing_phase=3, target_storage_phase=3, is_highly_relevant_for_current_phase=True)
    if test_trail_log_path.exists():
        with open(test_trail_log_path, "r", encoding="utf-8") as f: log_data = json.load(f)
        print(f"Full test trail log has {len(log_data)} entries.")
        assert len(log_data) == 3; assert "log_id" in log_data[0]
    else: print(f"ERROR: Test trail log {test_trail_log_path} not created!")
    if test_logic_mem_path.exists():
        with open(test_logic_mem_path, "r", encoding="utf-8") as f: vec_mem_data = json.load(f)
        print(f"Logic memory has {len(vec_mem_data)} entries.")
        assert len(vec_mem_data) == 2 ; assert vec_mem_data[0]['text'] == fact_text_main; assert vec_mem_data[1]['text'] == mixed_text_main
    else: print(f"ERROR: Test logic memory {test_logic_mem_path} not created!")
    if test_sym_occur_path.exists() and 'log_data' in locals() and log_data: # Check if log_data exists and is not empty
        processed_by_symbolic_node = False
        for entry in log_data: 
            if isinstance(entry, dict) and "symbolic_node_summary" in entry and \
               entry["symbolic_node_summary"].get("matched_symbols_count", 0) > 0:
                processed_by_symbolic_node = True; break
        if processed_by_symbolic_node:
            with open(test_sym_occur_path, "r", encoding="utf-8") as f: sym_occur_data = json.load(f)
            print(f"Symbol occurrence log has {len(sym_occur_data.get('entries',[]))} entries.")
            assert len(sym_occur_data.get('entries',[])) > 0 
        else: print(f"WARN: No symbols were matched by SymbolicNode according to trail_log, so symbol_occurrence_log might be empty.")
    elif not test_sym_occur_path.exists(): print(f"WARN: Test symbol occurrence log {test_sym_occur_path} not created.")
    TL_TrailLog.TRAIL_LOG_FILE_PATH = original_tl_path
    print("\n✅ processing_nodes.py ALL ACCUMULATED FIXES integration tests completed.")