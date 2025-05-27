# autonomous_learner.py

import time
import random
from pathlib import Path
import json
from collections import defaultdict, Counter
from urllib.parse import urljoin, urlparse
from datetime import datetime  # Import at module level

from processing_nodes import LogicNode, SymbolicNode, DynamicBridge, CurriculumManager
from brain_metrics import BrainMetrics
import web_parser
import parser as P_Parser

# Import tripartite memory for proper storage
from memory_architecture import TripartiteMemory
from decision_history import HistoryAwareMemory

# Import adaptive weight loading from memory_optimizer
from memory_optimizer import load_adaptive_config, ADAPTIVE_CONFIG_PATH

# --- Configuration ---
PHASE_URL_SOURCES = {
    "logical": [
        "https://en.wikipedia.org/wiki/Logic",
        "https://en.wikipedia.org/wiki/Mathematical_logic",
        "https://en.wikipedia.org/wiki/Propositional_calculus",
        "https://plato.stanford.edu/entries/logic-classical/",
        "https://oli.cmu.edu/courses/logic-proofs/",
        "https://scholarworks.smith.edu/textbooks/1/",
        "https://ies.ed.gov/rel-northeast-islands/2025/01/tool-4",
        "https://www.cdc.gov/library/research-guides/logic-models.html"
    ],
    "symbolic": [
        "https://en.wikipedia.org/wiki/Symbolic_logic",
        "https://philosophy.lander.edu/logic/symbolic.html",
        "https://logic.stanford.edu/intrologic/miscellaneous/symbolic.html",
        "https://www.ccsf.edu/node/164316"
    ],
    "hybrid": [
        "https://en.wikipedia.org/wiki/Hybrid_logic",
        "https://plato.stanford.edu/entries/logic-hybrid/"
    ]
}

deferred_urls_by_phase = defaultdict(list)
visited_urls_globally = set()
DATA_DIR = Path("data")

# Initialize BrainMetrics
print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Initializing BrainMetrics...", flush=True)
brain_metrics = BrainMetrics()
print(f"[{time.strftime('%H:%M:%S')}] DEBUG: BrainMetrics initialized", flush=True)

# Local metrics tracking (fallback if CurriculumManager doesn't have get_all_metrics)
LOCAL_METRICS = {
    "urls_visited_per_phase": defaultdict(int),
    "chunks_processed_per_phase": defaultdict(int),
    "symbols_found_per_phase": defaultdict(int),
    "phase_start_times": {},
    "phase_end_times": {},
    "links_evaluated_per_phase": defaultdict(int),
    "links_followed_per_phase": defaultdict(int),
    "links_deferred_per_phase": defaultdict(int)
}

# Global directives for adaptive weights - will be updated by load_adaptive_config()
ADAPTIVE_DIRECTIVES = {
    "link_score_weight_static": 0.6,
    "link_score_weight_dynamic": 0.4,
    "last_weight_update": None,
    "update_count": 0
}

# --- Helper function to load adaptive config ---
def load_adaptive_weights():
    """Load adaptive weights from disk config"""
    global ADAPTIVE_DIRECTIVES
    if ADAPTIVE_CONFIG_PATH.exists():
        try:
            with open(ADAPTIVE_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                ADAPTIVE_DIRECTIVES.update(config)
                print(f"🔧 Loaded adaptive weights: Static={ADAPTIVE_DIRECTIVES['link_score_weight_static']:.1%}, "
                      f"Dynamic={ADAPTIVE_DIRECTIVES['link_score_weight_dynamic']:.1%}")
        except Exception as e:
            print(f"⚠️ Could not load adaptive config: {e}")

# --- Tripartite Memory Storage Function ---
def store_to_tripartite_memory(memory, item, decision_type):
    """
    Store an item in the appropriate tripartite memory location.
    
    Args:
        memory: The HistoryAwareMemory instance to store in
        item: Dictionary containing text, scores, and metadata
        decision_type: FOLLOW_LOGIC, FOLLOW_SYMBOLIC, or FOLLOW_HYBRID
    """
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: store_to_tripartite_memory called with decision_type={decision_type}", flush=True)
    
    # Ensure item has required fields
    if 'logic_score' not in item:
        item['logic_score'] = 0.0
    if 'symbolic_score' not in item:
        item['symbolic_score'] = 0.0
    if 'text' not in item:
        item['text'] = item.get('content', '')
    if 'id' not in item:
        item['id'] = f"{decision_type}_{item.get('source_url', 'unknown_source').split('/')[-1]}_{int(time.time() * 1000)}"
    
    # Get current weights for history
    weights = {
        'static': ADAPTIVE_DIRECTIVES.get('link_score_weight_static', 0.6),
        'dynamic': ADAPTIVE_DIRECTIVES.get('link_score_weight_dynamic', 0.4)
    }
    
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: About to call memory.store() for item ID: {item['id']}", flush=True)
    # Store with history tracking
    memory.store(item, decision_type, weights)
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: memory.store() completed for item ID: {item['id']}", flush=True)
    
    return item

# --- Confidence Gate Helper ---
def evaluate_link_with_confidence_gates(logic_score, symbolic_score, logic_scale=2.0, sym_scale=1.0):
    """
    Returns (decision_type, final_score).
    decision_type: "FOLLOW_LOGIC", "FOLLOW_SYMBOLIC", or "FOLLOW_HYBRID"
    final_score: the score to use for threshold comparison.
    """
    # Normalize into [0,1]
    logic_conf = min(1.0, logic_score / logic_scale if logic_scale > 0 else (1.0 if logic_score > 0 else 0.0))
    sym_conf = min(1.0, symbolic_score / sym_scale if sym_scale > 0 else (1.0 if symbolic_score > 0 else 0.0))
    
    # High-confidence overrides
    if logic_conf > 0.8 and sym_conf < 0.3:
        return "FOLLOW_LOGIC", logic_score
    elif sym_conf > 0.8 and logic_conf < 0.3:
        return "FOLLOW_SYMBOLIC", symbolic_score
    else:
        # Hybrid blend using adaptive weights
        static_weight = ADAPTIVE_DIRECTIVES.get('link_score_weight_static', 0.6)
        dynamic_weight = ADAPTIVE_DIRECTIVES.get('link_score_weight_dynamic', 0.4)
        combined = (logic_score * static_weight) + (symbolic_score * dynamic_weight)
        return "FOLLOW_HYBRID", combined

# --- Helper Functions ---
def initialize_data_files_if_needed():
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Starting initialize_data_files_if_needed()", flush=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_files_to_check = {
        "data/vector_memory.json": [],
        "data/symbol_memory.json": {},
        "data/symbol_occurrence_log.json": {"entries": []},
        "data/symbol_emotion_map.json": {},
        "data/meta_symbols.json": {},
        "data/trail_log.json": [],
        "data/deferred_urls_log.json": {},
        "data/seed_symbols.json": {
            "🔥": {"name": "Fire", "keywords": ["fire", "flame", "computation", "logic"], "core_meanings": ["heat"], "emotions": ["anger"], "archetypes": ["destroyer"], "learning_phase": 0, "resonance_weight": 0.7},
            "💧": {"name": "Water", "keywords": ["water", "liquid", "data", "flow"], "core_meanings": ["flow"], "emotions": ["calm"], "archetypes": ["healer"], "learning_phase": 0, "resonance_weight": 0.7},
            "💻": {"name": "Computer", "keywords": ["computer", "computation", "cpu", "binary", "code", "algorithm", "system", "architecture"], "core_meanings": ["processing", "logic unit"], "emotions": ["neutral", "focus"], "archetypes": ["tool", "oracle"], "learning_phase": 0, "resonance_weight": 0.8}
        },
        "data/symbol_cooccurrence.json": {},
        "data/link_decisions.csv": "",
        "data/logic_memory.json": [],
        "data/symbolic_memory.json": [],
        "data/bridge_memory.json": []
    }
    
    for file_path_str, default_content in data_files_to_check.items():
        print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Checking file {file_path_str}", flush=True)
        file_path = Path(file_path_str)
        if not file_path.exists() or file_path.stat().st_size == 0:
            try:
                if file_path_str.endswith('.csv'):
                    with open(file_path, "w", newline='', encoding="utf-8") as f:
                        f.write("timestamp,url,logic_score,symbol_score,decision,confidence_type,phase,link_text\n")
                else:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(default_content, f, indent=2, ensure_ascii=False)
                print(f"Initialized {file_path_str}")
            except Exception as e:
                print(f"Error initializing {file_path_str}: {e}")
        else:
            if file_path_str not in ["data/seed_symbols.json", "data/link_decisions.csv"]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        if file_path_str.endswith(".json"):
                            if file_path_str == "data/deferred_urls_log.json":
                                loaded_data = json.load(f)
                                if not isinstance(loaded_data, dict):
                                    raise json.JSONDecodeError("Not a dict", file_path_str, 0)
                            elif file_path.stat().st_size > 0:
                                json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {file_path_str} was corrupted or not valid JSON. Re-initializing.")
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(default_content, f, indent=2, ensure_ascii=False)
    
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Completed initialize_data_files_if_needed()", flush=True)

def load_deferred_urls():
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Starting load_deferred_urls()", flush=True)
    global deferred_urls_by_phase
    deferred_log_path = DATA_DIR / "deferred_urls_log.json"
    if deferred_log_path.exists() and deferred_log_path.stat().st_size > 0:
        try:
            with open(deferred_log_path, "r", encoding="utf-8") as f:
                loaded_deferred = json.load(f)
                deferred_urls_by_phase = defaultdict(list)
                for phase_key, url_list in loaded_deferred.items():
                    try:
                        phase_num = int(phase_key)
                        processed_url_list = []
                        for item in url_list:
                            if isinstance(item, str):
                                processed_url_list.append({
                                    "url": item, "anchor": Path(item).name,
                                    "priority_score_at_deferral": 1.0,
                                    "original_discovery_phase": 0,
                                    "original_discovery_hop_count": 0
                                })
                            elif isinstance(item, dict) and "url" in item:
                                entry = {
                                    "url": item["url"],
                                    "anchor": item.get("anchor", Path(item["url"]).name),
                                    "priority_score_at_deferral": item.get("priority_score_at_deferral", 1.0),
                                    "original_discovery_phase": item.get("original_discovery_phase", 0),
                                    "original_discovery_hop_count": item.get("original_discovery_hop_count", 0)
                                }
                                processed_url_list.append(entry)
                        deferred_urls_by_phase[phase_num].extend(processed_url_list)
                    except ValueError:
                        print(f"[WARN] Invalid phase key '{phase_key}' in {deferred_log_path}. Skipping.")
            print(f"Loaded {sum(len(v) for v in deferred_urls_by_phase.values())} deferred URLs from {deferred_log_path}")
        except json.JSONDecodeError:
            print(f"[ERROR] Could not decode {deferred_log_path}. Starting with empty deferred URLs.")
            deferred_urls_by_phase = defaultdict(list)
    else:
        print(f"Deferred URLs log not found or empty at {deferred_log_path}. Starting with empty deferred URLs.")
        deferred_urls_by_phase = defaultdict(list)
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Completed load_deferred_urls()", flush=True)

def save_deferred_urls():
    processed_deferred_urls = {}
    for phase, entries in deferred_urls_by_phase.items():
        unique_phase_entries = []
        seen_urls_in_phase = set()
        for entry_dict in entries:
            if isinstance(entry_dict, dict) and "url" in entry_dict:
                url_val = entry_dict["url"]
                if url_val not in seen_urls_in_phase:
                    unique_phase_entries.append(entry_dict)
                    seen_urls_in_phase.add(url_val)
        processed_deferred_urls[str(phase)] = unique_phase_entries
    try:
        with open(DATA_DIR / "deferred_urls_log.json", "w", encoding="utf-8") as f:
            json.dump(processed_deferred_urls, f, indent=2, ensure_ascii=False)
        print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Saved deferred URLs.", flush=True)
    except Exception as e:
        print(f"Error saving deferred URLs: {e}")

def save_curriculum_metrics(curriculum_manager):
    """Save curriculum metrics to disk for analysis by optimizer"""
    try:
        cm_metrics = None
        if hasattr(curriculum_manager, 'get_all_metrics'):
            cm_metrics = curriculum_manager.get_all_metrics()
        
        if cm_metrics is None:
            print("   Using fallback metrics (CurriculumManager.get_all_metrics not found or returned None)")
            max_ph = 5
            if hasattr(curriculum_manager, 'get_max_phases'):
                max_ph = curriculum_manager.get_max_phases()
            elif hasattr(curriculum_manager, 'max_phases'):
                max_ph = curriculum_manager.max_phases

            cm_metrics = {
                'current_phase': curriculum_manager.current_phase if hasattr(curriculum_manager, 'current_phase') else 0,
                'max_phases': max_ph,
                'metrics_by_phase': {},
                'phase_timings': {
                    'start_times': dict(LOCAL_METRICS['phase_start_times']),
                    'end_times': dict(LOCAL_METRICS['phase_end_times'])
                },
                'last_updated': datetime.utcnow().isoformat()
            }
            
            for phase_num_int in range(1, cm_metrics['max_phases'] + 1):
                phase_str = str(phase_num_int)
                cm_metrics['metrics_by_phase'][phase_str] = {
                    'urls_visited': LOCAL_METRICS['urls_visited_per_phase'].get(phase_num_int, 0),
                    'chunks_processed': LOCAL_METRICS['chunks_processed_per_phase'].get(phase_num_int, 0),
                    'symbols_discovered': LOCAL_METRICS['symbols_found_per_phase'].get(phase_num_int, 0),
                    'links_evaluated': LOCAL_METRICS['links_evaluated_per_phase'].get(phase_num_int, 0),
                    'links_followed': LOCAL_METRICS['links_followed_per_phase'].get(phase_num_int, 0),
                    'links_deferred': LOCAL_METRICS['links_deferred_per_phase'].get(phase_num_int, 0),
                    'phase_completed': phase_num_int in LOCAL_METRICS['phase_end_times']
                }
        
        metrics_path = DATA_DIR / "curriculum_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(cm_metrics, f, indent=2, default=str)
        
        print(f"✅ Saved curriculum metrics to {metrics_path}")
    except Exception as e:
        print(f"Error saving curriculum metrics: {e}")
        import traceback
        traceback.print_exc()

def score_text_against_keywords(text_content, primary_keywords, secondary_keywords=None, anti_keywords=None):
    if not text_content or not isinstance(text_content, str):
        return 0.0, 0
    text_lower = text_content.lower()
    score = 0.0
    primary_matches = 0
    for kw in primary_keywords:
        if kw.lower() in text_lower:
            score += 2.0
            primary_matches += 1
    if secondary_keywords:
        for kw in secondary_keywords:
            if kw.lower() in text_lower:
                score += 1.0
    if anti_keywords:
        for kw in anti_keywords:
            if kw.lower() in text_lower:
                score -= 3.0
    return max(0.0, score), primary_matches

def score_text_for_symbolic_relevance(text_content, phase_directives=None):
    """
    Score text based on symbolic/emotional markers.
    Enhanced to be more sensitive to symbolic content.
    """
    if not text_content or not isinstance(text_content, str):
        return 0.0
    
    text_lower = text_content.lower()
    symbolic_score = 0.0
    
    emotion_markers = phase_directives.get("emotion_keywords", 
        ["emotion", "feeling", "joy", "fear", "anger", "love", "hope", "dream", 
         "passion", "sadness", "happy", "anxious", "excited"]) if phase_directives else []
    for marker in emotion_markers:
        if marker in text_lower:
            symbolic_score += 1.5
    
    symbolic_language_markers = phase_directives.get("symbolic_keywords",
        ["symbolize", "represents", "metaphor", "archetype", "myth", "legend",
         "meaning", "interpretation", "significance", "symbolic"]) if phase_directives else []
    for marker in symbolic_language_markers:
        if marker in text_lower:
            symbolic_score += 2.0
    
    import re
    emoji_pattern_str = phase_directives.get("emoji_pattern", r'[🔥💧💻⚙️🌀💡🧩🔗🌐⚖️🕊️⟳]') if phase_directives else r'[🔥💧💻⚙️🌀💡🧩🔗🌐⚖️🕊️⟳]'
    try:
        emoji_pattern = re.compile(emoji_pattern_str)
        emoji_count = len(emoji_pattern.findall(text_content))
        symbolic_score += emoji_count * 1.0
    except re.error:
        print(f"Warning: Invalid emoji regex pattern: {emoji_pattern_str}")

    narrative_markers = phase_directives.get("narrative_keywords",
        ["story", "tale", "narrative", "journey", "quest", "character"]) if phase_directives else []
    for marker in narrative_markers:
        if marker in text_lower:
            symbolic_score += 1.0
    
    return symbolic_score

def log_link_decision(url, logic_score, symbol_score, decision_type, confidence_type, phase, link_text=""):
    """Log link decisions to CSV for analysis"""
    import csv
    
    log_path = DATA_DIR / "link_decisions.csv"
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp","url","logic_score","symbol_score","decision","confidence_type","phase","link_text"])
        writer.writerow([
            datetime.utcnow().isoformat(),
            url[:255],
            round(logic_score, 3),
            round(symbol_score, 3),
            decision_type,
            confidence_type,
            phase,
            link_text[:100].replace('\n', ' ').replace('\r', '')
        ])

def evaluate_link_action(link_anchor_text, link_url,
                         current_processing_phase_num, curriculum_manager,
                         session_hot_keywords=None, force_phase1_focus=False):
    """
    Evaluates a link with confidence-gated decision making.
    Enhanced with better scoring, logging, and fallback rule.
    """
    current_phase_directives = curriculum_manager.get_processing_directives(current_processing_phase_num)
    phase1_directives = curriculum_manager.get_processing_directives(1)

    # Inject adaptive weights into directives
    current_phase_directives["link_score_weight_static"] = ADAPTIVE_DIRECTIVES["link_score_weight_static"]
    current_phase_directives["link_score_weight_dynamic"] = ADAPTIVE_DIRECTIVES["link_score_weight_dynamic"]

    # Score against current phase (logic-based)
    logic_score, logic_primary_matches = score_text_against_keywords(
        link_anchor_text,
        current_phase_directives.get("phase_keywords_primary", []),
        current_phase_directives.get("phase_keywords_secondary", []),
        current_phase_directives.get("phase_keywords_anti", [])
    )

    # Score for symbolic relevance
    symbolic_score = score_text_for_symbolic_relevance(link_anchor_text, current_phase_directives)

    # Add dynamic scoring based on hot keywords
    dynamic_score = 0.0
    if session_hot_keywords:
        text_lower_for_hot = link_anchor_text.lower()
        for hot_kw, freq in session_hot_keywords.items():
            if hot_kw.lower() in text_lower_for_hot:
                dynamic_score += (1.0 * freq)
        dynamic_score = min(dynamic_score, current_phase_directives.get("max_dynamic_link_score_bonus", 5.0))

    combined_logic_score = logic_score + dynamic_score

    decision_type, final_priority_score = evaluate_link_with_confidence_gates(
        combined_logic_score,
        symbolic_score
    )
    original_decision_type = decision_type

    if hasattr(brain_metrics, 'log_decision'):
        brain_metrics.log_decision(
            decision_type,
            link_url,
            combined_logic_score,
            symbolic_score,
            phase=current_processing_phase_num,
            link_text=link_anchor_text
        )
    
    # Fallback Rule Block
    min_primary_follow_fallback = current_phase_directives.get("phase_min_primary_keyword_matches_for_link_follow", 1)
    min_total_score_follow_fallback = current_phase_directives.get("phase_min_total_keyword_score_for_link_follow", 2.5)

    if final_priority_score >= 0.6 and logic_primary_matches > 0 and decision_type != "FOLLOW_LOGIC":
        if logic_primary_matches >= min_primary_follow_fallback and combined_logic_score >= min_total_score_follow_fallback:
            sanitized_anchor_fallback_print = link_anchor_text[:30].replace('\n', ' ').replace('\r', '')
            print(
                f"    🛡️ Fallback Rule: Promoting to FOLLOW_LOGIC (LPM: {logic_primary_matches}, "
                f"OrigDec: {original_decision_type}, Score: {combined_logic_score:.1f}) "
                f"for link '{sanitized_anchor_fallback_print}...'"
            )
            decision_type = "FOLLOW_LOGIC"
            final_priority_score = combined_logic_score

    # Enhanced logging
    if combined_logic_score > 0 or symbolic_score > 0:
        sanitized_anchor_log_print = link_anchor_text[:50].replace('\n',' ').replace('\r', '')
        print(
            f"    🧠 Link: '{sanitized_anchor_log_print}...' → "
            f"Logic={logic_score:.1f}+Dyn={dynamic_score:.1f} (TotalL={combined_logic_score:.1f}), "
            f"Sym={symbolic_score:.1f} → OrigDec:{original_decision_type}, FinalDec:{decision_type} "
            f"(score={final_priority_score:.1f})"
        )
    
    log_link_decision(link_url, combined_logic_score, symbolic_score,
                     decision_type, original_decision_type, current_processing_phase_num, link_anchor_text)

    LOCAL_METRICS['links_evaluated_per_phase'][current_processing_phase_num] += 1

    min_primary_follow = current_phase_directives.get("phase_min_primary_keyword_matches_for_link_follow", 1)
    min_total_score_follow = current_phase_directives.get("phase_min_total_keyword_score_for_link_follow", 2.5)
    symbolic_follow_score_multiplier = current_phase_directives.get("symbolic_follow_score_multiplier", 0.7)

    if force_phase1_focus:
        phase1_score, phase1_primary_matches = score_text_against_keywords(
            link_anchor_text,
            phase1_directives.get("phase_keywords_primary", []),
            phase1_directives.get("phase_keywords_secondary", []),
            phase1_directives.get("phase_keywords_anti", [])
        )
        
        min_primary_follow_p1 = phase1_directives.get("phase_min_primary_keyword_matches_for_link_follow", 2)
        min_total_score_follow_p1 = phase1_directives.get("phase_min_total_keyword_score_for_link_follow", 5.0)

        if phase1_primary_matches >= min_primary_follow_p1 and phase1_score >= min_total_score_follow_p1:
            LOCAL_METRICS['links_followed_per_phase'][current_processing_phase_num] += 1
            return "FOLLOW_NOW", current_processing_phase_num, phase1_score
        elif phase1_score >= 2.0:
            LOCAL_METRICS['links_deferred_per_phase'][1] += 1
            return "DEFER_SHALLOW", 1, phase1_score
        else:
            return "IGNORE", None, final_priority_score

    # Normal evaluation
    if decision_type == "FOLLOW_LOGIC":
        if logic_primary_matches >= min_primary_follow and final_priority_score >= min_total_score_follow:
            LOCAL_METRICS['links_followed_per_phase'][current_processing_phase_num] += 1
            return "FOLLOW_NOW", current_processing_phase_num, final_priority_score
    elif decision_type == "FOLLOW_HYBRID":
        if logic_primary_matches >= min_primary_follow and final_priority_score >= min_total_score_follow:
            LOCAL_METRICS['links_followed_per_phase'][current_processing_phase_num] += 1
            return "FOLLOW_NOW", current_processing_phase_num, final_priority_score
    elif decision_type == "FOLLOW_SYMBOLIC":
        if final_priority_score >= (min_total_score_follow * symbolic_follow_score_multiplier):
            LOCAL_METRICS['links_followed_per_phase'][current_processing_phase_num] += 1
            return "FOLLOW_NOW", current_processing_phase_num, final_priority_score

    # Check for deferral to other phases
    best_future_phase_score = -float('inf')
    best_future_phase_num = None
    for future_phase_idx in range(1, curriculum_manager.get_max_phases() + 1):
        if future_phase_idx == current_processing_phase_num:
            continue
        
        future_phase_directives = curriculum_manager.get_processing_directives(future_phase_idx)
        defer_score, defer_primary_matches = score_text_against_keywords(
            link_anchor_text,
            future_phase_directives.get("phase_keywords_primary", []),
            future_phase_directives.get("phase_keywords_secondary", []),
            future_phase_directives.get("phase_keywords_anti", [])
        )
        
        min_primary_defer = future_phase_directives.get("phase_min_primary_keyword_matches_for_link_follow", 1)
        min_total_score_defer = future_phase_directives.get("phase_min_total_keyword_score_for_link_follow", 2.0)

        if defer_primary_matches >= min_primary_defer and defer_score >= min_total_score_defer:
            if defer_score > best_future_phase_score:
                best_future_phase_score = defer_score
                best_future_phase_num = future_phase_idx
    
    if best_future_phase_num is not None:
        LOCAL_METRICS['links_deferred_per_phase'][best_future_phase_num] += 1
        action = "DEFER_SHALLOW" if current_phase_directives.get("allow_shallow_dive_for_future_phase_links", True) else "DEFER_URL_ONLY"
        
        sanitized_anchor_text = link_anchor_text[:30].replace('\n', ' ').replace('\r', '')
        print(f"    🔗 Deferring link '{sanitized_anchor_text}...' to Phase {best_future_phase_num} with score {best_future_phase_score:.1f}")
        
        return action, best_future_phase_num, best_future_phase_score
    
    return "IGNORE", None, final_priority_score

def process_chunk_to_tripartite(tripartite_memory, chunk_text, source_url, current_phase, symbols_found=0):
    """
    Process a chunk and store it in tripartite memory.
    
    Args:
        tripartite_memory: The HistoryAwareMemory instance to store in
        chunk_text: Text to process
        source_url: Source URL
        current_phase: Current phase number
        symbols_found: Number of symbols found
    
    Returns: dict with processing results
    """
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: process_chunk_to_tripartite called for URL {source_url}, phase {current_phase}", flush=True)
    
    temp_cm = CurriculumManager()
    phase_directives = temp_cm.get_processing_directives(current_phase)
    
    # Inject adaptive weights
    phase_directives["link_score_weight_static"] = ADAPTIVE_DIRECTIVES["link_score_weight_static"]
    phase_directives["link_score_weight_dynamic"] = ADAPTIVE_DIRECTIVES["link_score_weight_dynamic"]
    
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: About to calculate chunk scores", flush=True)
    # Calculate scores for the chunk
    logic_score, _ = score_text_against_keywords(
        chunk_text,
        phase_directives.get("phase_keywords_primary", []),
        phase_directives.get("phase_keywords_secondary", []),
        phase_directives.get("phase_keywords_anti", [])
    )
    
    symbolic_score = score_text_for_symbolic_relevance(chunk_text, phase_directives)
    
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Chunk scores calculated: logic={logic_score:.2f}, symbolic={symbolic_score:.2f}", flush=True)
    
    # Determine decision type using the same confidence gate logic as links
    chunk_logic_scale = phase_directives.get("chunk_logic_scale", 2.0)
    chunk_sym_scale = phase_directives.get("chunk_sym_scale", 1.0)

    decision_type, confidence_score = evaluate_link_with_confidence_gates(
        logic_score,
        symbolic_score,
        logic_scale=chunk_logic_scale,
        sym_scale=chunk_sym_scale
    )
    
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Chunk decision type: {decision_type}, Confidence score: {confidence_score:.2f}", flush=True)
    
    # Create item for storage
    item_id_suffix = f"{source_url.split('/')[-1]}_{current_phase}_{int(time.time() * 1000)}_{random.randint(1000,9999)}"
    item = {
        'id': f"{decision_type}_{item_id_suffix}",
        'text': chunk_text[:2000],
        'source_url': source_url,
        'logic_score': logic_score,
        'symbolic_score': symbolic_score,
        'confidence_score': confidence_score,
        'processing_phase': current_phase,
        'timestamp': datetime.utcnow().isoformat(),
        'symbols_found_in_chunk': symbols_found
    }
    
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: About to store chunk to tripartite memory (ID: {item['id']})", flush=True)
    # Store to tripartite memory
    stored_item = store_to_tripartite_memory(tripartite_memory, item, decision_type)
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Stored chunk to tripartite memory (ID: {item['id']})", flush=True)
    
    return {
        'decision_type': decision_type,
        'confidence_score': confidence_score,
        'stored_item_id': stored_item.get('id')
    }

def autonomous_learning_cycle(focus_only_on_phase_1=True):
    global deferred_urls_by_phase
    global visited_urls_globally

    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Starting autonomous_learning_cycle", flush=True)
    
    # Load adaptive weights at the start of the cycle
    load_adaptive_weights()
    
    initialize_data_files_if_needed()
    load_deferred_urls()

    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Creating LogicNode...", flush=True)
    logic_node = LogicNode()
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: LogicNode created", flush=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Creating SymbolicNode...", flush=True)
    symbolic_node = SymbolicNode()
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: SymbolicNode created", flush=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Creating CurriculumManager...", flush=True)
    curriculum_manager = CurriculumManager()
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: CurriculumManager created", flush=True)
    
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Creating DynamicBridge...", flush=True)
    dynamic_bridge = DynamicBridge(logic_node, symbolic_node, curriculum_manager)
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: DynamicBridge created", flush=True)
    
    # Initialize tripartite memory
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Creating HistoryAwareMemory...", flush=True)
    tripartite_memory = HistoryAwareMemory(data_dir=str(DATA_DIR))
    print(f"[{time.strftime('%H:%M:%S')}] DEBUG: HistoryAwareMemory created successfully!", flush=True)
    initial_counts = tripartite_memory.get_counts()
    print(f"    📊 Initial memory: Logic={initial_counts['logic']}, Symbolic={initial_counts['symbolic']}, Bridge={initial_counts['bridge']}")

    chunks_processed_since_last_save = 0
    SAVE_EVERY_N_CHUNKS = 50

    print(f"🚀 Starting Autonomous Learning Cycle... (Phase 1 Focus: {focus_only_on_phase_1})")

    # Determine phases to process based on focus_only_on_phase_1
    max_system_phases = curriculum_manager.get_max_phases()
    phases_to_process_indices = range(1, 2) if focus_only_on_phase_1 else range(1, max_system_phases + 1)

    for current_phase_processing_num in phases_to_process_indices:
        LOCAL_METRICS['phase_start_times'][current_phase_processing_num] = datetime.utcnow().isoformat()
        
        curriculum_manager.current_phase = current_phase_processing_num
        current_phase_directives = curriculum_manager.get_processing_directives(current_phase_processing_num)
        
        # Inject adaptive weights into phase directives
        current_phase_directives["link_score_weight_static"] = ADAPTIVE_DIRECTIVES["link_score_weight_static"]
        current_phase_directives["link_score_weight_dynamic"] = ADAPTIVE_DIRECTIVES["link_score_weight_dynamic"]
        
        print(f"\n🌀 =====  PHASE {current_phase_processing_num} ({current_phase_directives.get('info','N/A')})  =====")
        
        priority_queue_this_phase = []
        
        # Add seed URLs for the current phase
        for seed_url in PHASE_URL_SOURCES.get(current_phase_processing_num, []):
            seed_anchor = Path(seed_url).name
            static_score, _ = score_text_against_keywords(
                seed_anchor,
                current_phase_directives.get("phase_keywords_primary", []),
                current_phase_directives.get("phase_keywords_secondary", []),
                current_phase_directives.get("phase_keywords_anti", [])
            )
            priority_queue_this_phase.append((static_score if static_score > 0 else 5.0, seed_url, seed_anchor, 0))

        # Add deferred URLs relevant to this phase
        phase_deferred_entries = []
        if current_phase_processing_num in deferred_urls_by_phase:
            phase_deferred_entries = deferred_urls_by_phase.pop(current_phase_processing_num, [])
        
        for deferred_entry in phase_deferred_entries:
            deferred_url = deferred_entry["url"]
            deferred_anchor = deferred_entry.get("anchor", Path(deferred_url).name)
            static_score, _ = score_text_against_keywords(
                deferred_anchor,
                current_phase_directives.get("phase_keywords_primary", []),
                current_phase_directives.get("phase_keywords_secondary", []),
                current_phase_directives.get("phase_keywords_anti", [])
            )
            priority = max(static_score if static_score > 0 else 1.0, deferred_entry.get("priority_score_at_deferral", 1.0))
            original_hop = deferred_entry.get("original_discovery_hop_count", 0)
            priority_queue_this_phase.append((priority, deferred_url, deferred_anchor, original_hop))

        # Deduplicate and sort the priority queue
        temp_url_seen_in_queue = set()
        unique_priority_queue_items = []
        for item_tuple in priority_queue_this_phase:
            url_to_check = item_tuple[1]
            if url_to_check not in temp_url_seen_in_queue:
                unique_priority_queue_items.append(item_tuple)
                temp_url_seen_in_queue.add(url_to_check)
        priority_queue_this_phase = unique_priority_queue_items
        priority_queue_this_phase.sort(key=lambda x: x[0], reverse=True)

        urls_processed_in_session_count = 0
        if focus_only_on_phase_1 and current_phase_processing_num == 1:
            phase1_specific_directives = curriculum_manager.get_processing_directives(1)
            max_urls_for_session = phase1_specific_directives.get("max_urls_to_process_per_phase_session", 5)
        else:
            max_urls_for_session = current_phase_directives.get("max_urls_to_process_per_phase_session", 2)

        session_hot_keywords = Counter()
        MAX_HOT_KEYWORDS = current_phase_directives.get("max_session_hot_keywords", 20)
        MIN_HOT_KEYWORD_FREQ = current_phase_directives.get("min_session_hot_keyword_freq", 2)

        session_items_stored = defaultdict(int)

        while priority_queue_this_phase and urls_processed_in_session_count < max_urls_for_session:
            current_priority, current_url, current_anchor, current_hop_count = priority_queue_this_phase.pop(0)

            if current_url in visited_urls_globally:
                print(f"    ⏭️ Already visited globally: {current_url}. Skipping.")
                continue
            
            max_hop_depth = current_phase_directives.get("max_exploration_depth_from_seed_url", 3)
            if current_hop_count > max_hop_depth:
                print(f"    Max hop depth ({current_hop_count}/{max_hop_depth}) reached for {current_url}. Skipping.")
                continue
            
            print(f"\n🔗 Processing URL ({urls_processed_in_session_count + 1}/{max_urls_for_session}, hop {current_hop_count}, prio {current_priority:.2f}): {current_url}")
            visited_urls_globally.add(current_url)
            urls_processed_in_session_count += 1
            
            LOCAL_METRICS['urls_visited_per_phase'][current_phase_processing_num] += 1
            if hasattr(curriculum_manager, 'update_metrics'):
                curriculum_manager.update_metrics(current_phase_processing_num, urls_visited_increment=1)

            print(f"[{time.strftime('%H:%M:%S')}] DEBUG: About to fetch raw HTML for {current_url}", flush=True)
            raw_html_content = web_parser.fetch_raw_html(current_url)
            if not raw_html_content:
                print(f"    ⚠️ Failed to fetch HTML for {current_url}. Skipping.")
                time.sleep(random.uniform(1,2))
                continue
            
            print(f"[{time.strftime('%H:%M:%S')}] DEBUG: HTML fetched, extracting links from {current_url}", flush=True)
            extracted_links_with_anchor = web_parser.extract_links_with_text_from_html(current_url, raw_html_content)
            
            print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Cleaning HTML to text for {current_url}", flush=True)
            main_page_text = web_parser.clean_html_to_text(raw_html_content)
            if not main_page_text:
                print(f"    ⚠️ No text content after cleaning HTML for {current_url}. Skipping.")
                time.sleep(random.uniform(1,2))
                continue

            print(f"[{time.strftime('%H:%M:%S')}] DEBUG: About to chunk content for {current_url}", flush=True)
            page_chunks = P_Parser.chunk_content(main_page_text, max_chunk_size=current_phase_directives.get("max_chunk_size", 1000))
            
            MAX_CHUNKS_PER_PAGE = current_phase_directives.get("max_chunks_per_page", 50)
            if len(page_chunks) > MAX_CHUNKS_PER_PAGE:
                print(f"    📄 Page has {len(page_chunks)} chunks. Limiting to {MAX_CHUNKS_PER_PAGE} for performance...")
                step = max(1, len(page_chunks) // MAX_CHUNKS_PER_PAGE)
                page_chunks = page_chunks[::step][:MAX_CHUNKS_PER_PAGE]
            else:
                print(f"    📄 Page yielded {len(page_chunks)} chunks. Processing...")
            
            print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Starting chunk processing loop for {current_url}", flush=True)
            
            symbols_found_this_page = 0
            url_start_time = time.time()
            
            for chunk_num, chunk_text_content in enumerate(page_chunks):
                print(f"[{time.strftime('%H:%M:%S')}] About to process chunk {chunk_num+1}/{len(page_chunks)} from {current_url}", flush=True)
                
                min_chunk_len = current_phase_directives.get("min_chunk_processing_length", 100)
                if len(chunk_text_content.strip()) < min_chunk_len:
                    print(f"[{time.strftime('%H:%M:%S')}] Skipping short chunk {chunk_num+1} (len: {len(chunk_text_content.strip())})", flush=True)
                    continue
                
                print(f"[{time.strftime('%H:%M:%S')}] Scoring chunk {chunk_num+1}", flush=True)
                chunk_score_current_phase, prim_matches_current = score_text_against_keywords(
                    chunk_text_content,
                    current_phase_directives.get("phase_keywords_primary", []),
                    current_phase_directives.get("phase_keywords_secondary", []),
                    current_phase_directives.get("phase_keywords_anti", [])
                )
                
                # Skip irrelevant chunks strategy
                skip_irrelevant_threshold = current_phase_directives.get("skip_irrelevant_chunk_score_threshold", 0.5)
                skip_after_n_chunks = current_phase_directives.get("skip_irrelevant_chunk_after_n", 10)
                if chunk_score_current_phase < skip_irrelevant_threshold and chunk_num > skip_after_n_chunks:
                    print(f"[{time.strftime('%H:%M:%S')}] Skipping low-score chunk {chunk_num+1} (score: {chunk_score_current_phase}) after {skip_after_n_chunks} chunks.", flush=True)
                    continue
                
                is_relevant_for_current_processing = (
                    prim_matches_current >= current_phase_directives.get("phase_min_primary_keyword_matches_for_chunk_relevance", 1) and
                    chunk_score_current_phase >= current_phase_directives.get("phase_min_total_keyword_score_for_chunk_relevance", 1.0)
                )

                print(f"[{time.strftime('%H:%M:%S')}] Parsing symbols from chunk {chunk_num+1}", flush=True)
                chunk_symbols_data = P_Parser.parse_input(chunk_text_content)
                num_symbols_in_chunk = 0
                if chunk_symbols_data and 'symbols' in chunk_symbols_data:
                    num_symbols_in_chunk = len(chunk_symbols_data['symbols'])
                    symbols_found_this_page += num_symbols_in_chunk

                print(f"[{time.strftime('%H:%M:%S')}] About to store chunk {chunk_num+1} to tripartite memory", flush=True)
                result = process_chunk_to_tripartite(
                    tripartite_memory,
                    chunk_text=chunk_text_content,
                    source_url=current_url,
                    current_phase=current_phase_processing_num,
                    symbols_found=num_symbols_in_chunk
                )
                print(f"[{time.strftime('%H:%M:%S')}] Stored chunk {chunk_num+1}, decision: {result['decision_type']}, ID: {result.get('stored_item_id','N/A')}", flush=True)
                
                session_items_stored[result['decision_type']] += 1
                
                if (chunk_num + 1) % 10 == 0 and len(page_chunks) > 10:
                    elapsed = time.time() - url_start_time
                    rate = (chunk_num + 1) / elapsed if elapsed > 0 else 0
                    print(f"    ⏳ Progress: {chunk_num + 1}/{len(page_chunks)} chunks "
                          f"({elapsed:.1f}s elapsed, {rate:.1f} chunks/sec)", flush=True)
                
                LOCAL_METRICS['chunks_processed_per_phase'][current_phase_processing_num] += 1
                LOCAL_METRICS['symbols_found_per_phase'][current_phase_processing_num] += num_symbols_in_chunk
                
                if is_relevant_for_current_processing:
                    chunk_kws = P_Parser.extract_keywords(chunk_text_content, max_keywords=5)
                    session_hot_keywords.update(chunk_kws)
                
                chunks_processed_since_last_save += 1
                
                if chunks_processed_since_last_save >= SAVE_EVERY_N_CHUNKS:
                    print(f"[{time.strftime('%H:%M:%S')}] Periodic save after {chunks_processed_since_last_save} chunks...", flush=True)
                    if hasattr(tripartite_memory, 'save_all'):
                        tripartite_memory.save_all()
                    counts = tripartite_memory.get_counts()
                    print(f"    📊 Current totals: Logic={counts['logic']}, Symbolic={counts['symbolic']}, Bridge={counts['bridge']}")
                    chunks_processed_since_last_save = 0
            
            # Save after each URL processing is complete
            print(f"[{time.strftime('%H:%M:%S')}] Saving after URL {current_url} processing...", flush=True)
            if hasattr(tripartite_memory, 'save_all'):
                tripartite_memory.save_all()
            counts = tripartite_memory.get_counts()
            print(f"    📊 Memory after URL: Logic={counts['logic']}, Symbolic={counts['symbolic']}, Bridge={counts['bridge']}")
            
            total_url_time = time.time() - url_start_time
            chunks_per_sec_url = (len(page_chunks) / total_url_time) if total_url_time > 0 else 0
            print(f"    ✅ URL processed in {total_url_time:.1f}s ({chunks_per_sec_url:.1f} chunks/sec)")
            
            if session_hot_keywords:
                pruned_hot_keywords = Counter({
                    kw: count for kw, count in session_hot_keywords.most_common(MAX_HOT_KEYWORDS * 2)
                    if count >= MIN_HOT_KEYWORD_FREQ
                })
                session_hot_keywords = Counter(dict(pruned_hot_keywords.most_common(MAX_HOT_KEYWORDS)))
                if session_hot_keywords:
                    print(f"    🔥 Session Hot Keywords (Top {len(session_hot_keywords)}): {list(session_hot_keywords.keys())}")

            # Process extracted links
            links_found_count = len(extracted_links_with_anchor)
            links_scored_meaningfully = 0
            
            print(f"[{time.strftime('%H:%M:%S')}] Processing {links_found_count} links from {current_url}", flush=True)
            
            new_links_to_consider_tuples = []
            for link_url_abs, link_anchor_text in extracted_links_with_anchor:
                if not link_anchor_text or len(link_anchor_text.strip()) < 3:
                    continue

                try:
                    parsed_link_domain = urlparse(link_url_abs).netloc
                    parsed_current_domain = urlparse(current_url).netloc
                    if not (parsed_link_domain == parsed_current_domain or 
                            parsed_link_domain.endswith("." + parsed_current_domain) or
                            parsed_current_domain.endswith("." + parsed_link_domain)):
                        continue
                except Exception as e_parse:
                    print(f"      ⚠️ Error parsing link URL {link_url_abs} or current URL {current_url}: {e_parse}")
                    continue
                
                if link_url_abs in visited_urls_globally:
                    continue

                action, target_phase_for_link, link_priority_score = evaluate_link_action(
                    link_anchor_text, link_url_abs, current_phase_processing_num, curriculum_manager, session_hot_keywords,
                    force_phase1_focus=(focus_only_on_phase_1 and current_phase_processing_num == 1)
                )
                
                if link_priority_score > 0:
                    links_scored_meaningfully += 1

                if action == "FOLLOW_NOW":
                    if focus_only_on_phase_1 and current_phase_processing_num == 1 and target_phase_for_link != 1:
                        pass
                    else:
                        if not any(item[1] == link_url_abs for item in priority_queue_this_phase):
                            new_links_to_consider_tuples.append((link_priority_score, link_url_abs, link_anchor_text, current_hop_count + 1))

                elif action == "DEFER_SHALLOW" or action == "DEFER_URL_ONLY":
                    if focus_only_on_phase_1 and current_phase_processing_num == 1 and target_phase_for_link != 1:
                        pass
                    else:
                        is_already_deferred_to_target = any(
                            entry.get("url") == link_url_abs for entry in deferred_urls_by_phase.get(target_phase_for_link, [])
                            if isinstance(entry, dict)
                        )
                        if not is_already_deferred_to_target:
                            deferred_urls_by_phase[target_phase_for_link].append({
                                "url": link_url_abs,
                                "anchor": link_anchor_text,
                                "priority_score_at_deferral": link_priority_score,
                                "original_discovery_phase": current_phase_processing_num,
                                "original_discovery_hop_count": current_hop_count + 1
                            })

                            sanitized_anchor_text = link_anchor_text[:30].replace('\n', ' ').replace('\r', '')
                            print(
                                f"    📥 Deferred link '{sanitized_anchor_text}...' "
                                f"to Phase {target_phase_for_link} (Score: {link_priority_score:.1f})"
                            )
                            
                            if action == "DEFER_SHALLOW" and link_url_abs not in visited_urls_globally:
                                shallow_content = web_parser.fetch_shallow(link_url_abs, max_chars=current_phase_directives.get("shallow_dive_max_chars", 500))
                                if shallow_content:
                                    shallow_result = process_chunk_to_tripartite(
                                        tripartite_memory,
                                        chunk_text=shallow_content,
                                        source_url=link_url_abs,
                                        current_phase=current_phase_processing_num,
                                        symbols_found=0
                                    )
                                    if 'decision_type' in shallow_result:
                                        session_items_stored[shallow_result['decision_type']] += 1
            
            print(f"    📊 Links summary: {links_found_count} found, {links_scored_meaningfully} scored > 0.")
            
            if new_links_to_consider_tuples:
                print(f"    🔗 Adding {len(new_links_to_consider_tuples)} new links to phase queue.")
                priority_queue_this_phase.extend(new_links_to_consider_tuples)
                priority_queue_this_phase.sort(key=lambda x: x[0], reverse=True)

            # Be nice to servers
            time.sleep(random.uniform(current_phase_directives.get("min_crawl_delay", 0.5),
                                     current_phase_directives.get("max_crawl_delay", 1.5)))

        # End of while loop for URLs in a phase
        print(f"\n💾 Final save for Phase {current_phase_processing_num}...")
        if hasattr(tripartite_memory, 'save_all'):
            tripartite_memory.save_all()
        final_counts = tripartite_memory.get_counts()
        
        print(f"\n📊 Phase {current_phase_processing_num} summary:")
        print(f"  Items stored this session: {sum(session_items_stored.values())} "
              f"(L: {session_items_stored.get('FOLLOW_LOGIC', 0)}, "
              f"S: {session_items_stored.get('FOLLOW_SYMBOLIC', 0)}, "
              f"H: {session_items_stored.get('FOLLOW_HYBRID', 0)})")
        print(f"  Total in memory: {final_counts['total']} "
              f"(L: {final_counts['logic']}, S: {final_counts['symbolic']}, B: {final_counts['bridge']})")

        LOCAL_METRICS['phase_end_times'][current_phase_processing_num] = datetime.utcnow().isoformat()
        
        print(f"\n🏁 Phase {current_phase_processing_num} scraping session complete. Processed {urls_processed_in_session_count} URLs.")
        
        save_curriculum_metrics(curriculum_manager)
        
        # Run meta-symbol analysis (conditionally)
        if not focus_only_on_phase_1 or current_phase_processing_num == 1:
            print(f"🔬 Running meta-symbol analysis for data up to phase {current_phase_processing_num}...")
            if hasattr(symbolic_node, 'run_meta_symbol_analysis'):
                symbolic_node.run_meta_symbol_analysis(max_phase_to_consider=current_phase_processing_num)
        
        try:
            save_deferred_urls()
        except Exception as e_save_deferred:
            print(f"[ERROR] Could not save deferred URLs after phase {current_phase_processing_num}: {e_save_deferred}")

        # Curriculum advancement logic
        if not focus_only_on_phase_1:
            if hasattr(curriculum_manager, 'advance_phase_if_ready'):
                if curriculum_manager.advance_phase_if_ready(current_phase_processing_num):
                    print(f"🎉 Advanced to Phase {curriculum_manager.get_current_phase()} based on metrics!")
                else:
                    if current_phase_processing_num < curriculum_manager.get_max_phases():
                        print(f"📊 Metrics not yet met to advance from Phase {current_phase_processing_num}.")
            elif current_phase_processing_num == curriculum_manager.get_max_phases():
                print("🏁 All curriculum phases processed in this learning cycle.")
        elif current_phase_processing_num == 1 and focus_only_on_phase_1:
            print(f"🏁 Phase 1 (Focus Mode) processing complete.")

    # End of for loop for phases
    print("\n✅ Autonomous Learning Cycle Finished.")

    # Save brain metrics if available
    if hasattr(brain_metrics, 'save_session_metrics'):
        brain_metrics.save_session_metrics()

    # Save curriculum metrics and deferred URLs
    save_curriculum_metrics(curriculum_manager)
    save_deferred_urls()

    # Final memory save
    if hasattr(tripartite_memory, 'save_all'):
        tripartite_memory.save_all()

    # Global summary
    print(f"Total URLs visited globally during this cycle: {len(visited_urls_globally)}")

    print("\n📊 Session Summary (Local Metrics):")
    for phase_num_int_summary in range(1, curriculum_manager.get_max_phases() + 1):
        if LOCAL_METRICS['urls_visited_per_phase'].get(phase_num_int_summary, 0) > 0:
            print(f"  Phase {phase_num_int_summary}:")
            print(f"    - URLs visited: {LOCAL_METRICS['urls_visited_per_phase'][phase_num_int_summary]}")
            print(f"    - Chunks processed: {LOCAL_METRICS['chunks_processed_per_phase'][phase_num_int_summary]}")
            print(f"    - Symbols found: {LOCAL_METRICS['symbols_found_per_phase'][phase_num_int_summary]}")
            print(f"    - Links evaluated: {LOCAL_METRICS['links_evaluated_per_phase'][phase_num_int_summary]}")
            print(f"    - Links followed: {LOCAL_METRICS['links_followed_per_phase'][phase_num_int_summary]}")
            print(f"    - Links deferred: {LOCAL_METRICS['links_deferred_per_phase'][phase_num_int_summary]}")

if __name__ == "__main__":
   # Determine focus mode (e.g., from command-line arg, config, or hardcode for testing)
   phase_1_only_focus = True
   print(f"Starting autonomous learning process... Phase 1 Focus: {phase_1_only_focus}")
   
   # Temporary curriculum manager instance for saving metrics in case of error before full init
   temp_cm_for_error_handling = None
   try:
       temp_cm_for_error_handling = CurriculumManager()
       autonomous_learning_cycle(focus_only_on_phase_1=phase_1_only_focus)
   except KeyboardInterrupt:
       print("\n🛑 Autonomous learning interrupted by user.")
       if hasattr(brain_metrics, 'save_session_metrics'):
           brain_metrics.save_session_metrics()
       save_deferred_urls()
       if temp_cm_for_error_handling:
           save_curriculum_metrics(temp_cm_for_error_handling)
   except Exception as e:
       print(f"💥 An error occurred during autonomous learning: {e}")
       import traceback
       traceback.print_exc()
       if hasattr(brain_metrics, 'save_session_metrics'):
           brain_metrics.save_session_metrics()
       save_deferred_urls()
       if temp_cm_for_error_handling:
           save_curriculum_metrics(temp_cm_for_error_handling)
   
   # Quick Confidence-Gate Tests (with new scales)
   print("\n--- Testing evaluate_link_with_confidence_gates() ---")
   # Test cases with new scales (logic_scale=2.0, sym_scale=1.0)
   # logic_conf = logic_score / 2.0
   # sym_conf = symbolic_score / 1.0
   test_cases = [
       # logic_score, symbolic_score, expected_decision (logic_conf > 0.8 and sym_conf < 0.3)
       (1.8, 0.1, "FOLLOW_LOGIC"),     # logic_conf=0.9 (>0.8), sym_conf=0.1 (<0.3) -> LOGIC
       # (sym_conf > 0.8 and logic_conf < 0.3)
       (0.2, 0.9, "FOLLOW_SYMBOLIC"),  # logic_conf=0.1 (<0.3), sym_conf=0.9 (>0.8) -> SYMBOLIC
       # Hybrid
       (1.0, 0.5, "FOLLOW_HYBRID"),    # logic_conf=0.5, sym_conf=0.5 -> HYBRID
       (0.5, 0.2, "FOLLOW_HYBRID"),    # logic_conf=0.25, sym_conf=0.2 -> HYBRID (neither override met)
       (3.0, 0.0, "FOLLOW_LOGIC"),     # logic_conf=1.0 (>0.8), sym_conf=0.0 (<0.3) -> LOGIC
       (0.0, 2.0, "FOLLOW_SYMBOLIC"),  # logic_conf=0.0 (<0.3), sym_conf=1.0 (>0.8) -> SYMBOLIC (sym_conf capped at 1.0)
   ]
   
   all_tests_passed = True
   for i, (logic, sym, expected_decision) in enumerate(test_cases):
       dec, score = evaluate_link_with_confidence_gates(logic, sym)  # Uses new default scales
       # Calculate confidences for assertion explanation
       l_conf = min(1.0, logic / 2.0 if 2.0 > 0 else (1.0 if logic > 0 else 0.0))
       s_conf = min(1.0, sym / 1.0 if 1.0 > 0 else (1.0 if sym > 0 else 0.0))
       
       print(f"  Test {i+1}: Logic={logic:.1f} (conf={l_conf:.2f}), Sym={sym:.1f} (conf={s_conf:.2f})  →  Decision={dec}, Score={score:.2f}")
       if dec != expected_decision:
           all_tests_passed = False
           print(f"    ASSERTION FAILED: Expected {expected_decision}, got {dec}")
           # Explain why it might have failed based on confidence values
           if expected_decision == "FOLLOW_LOGIC":
               print(f"    Expected LOGIC override: logic_conf ({l_conf:.2f}) > 0.8 AND sym_conf ({s_conf:.2f}) < 0.3")
           elif expected_decision == "FOLLOW_SYMBOLIC":
               print(f"    Expected SYMBOLIC override: sym_conf ({s_conf:.2f}) > 0.8 AND logic_conf ({l_conf:.2f}) < 0.3")
           else:  # HYBRID
               print(f"    Expected HYBRID: NOT (logic_conf > 0.8 AND sym_conf < 0.3) AND NOT (sym_conf > 0.8 AND logic_conf < 0.3)")

   if all_tests_passed:
       print("✅ Confidence-gate tests passed!")
   else:
       print("❌ Some confidence-gate tests FAILED!")
   print("--- End of tests ---\n")