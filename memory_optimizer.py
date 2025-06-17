# memory_optimizer.py - Updated with Quarantine, Linguistic Warfare, and Visualization

import sys
import re
import argparse
import unicodedata
import time
from pathlib import Path
import json
import hashlib
import csv
from datetime import datetime

# Assuming these are the correct import paths based on your project structure
import parser as P_Parser
from parser import load_seed_symbols

from web_parser import fetch_raw_html, extract_links_with_text_from_html, clean_html_to_text
from unified_memory import VectorMemory
vector_memory = VectorMemory()
from symbol_cluster import cluster_vectors_and_plot
from trail_log import log_trail, add_emotions
from trail_graph import show_trail_graph
from emotion_handler import predict_emotions
from symbol_emotion_updater import update_symbol_emotions
from symbol_generator import generate_symbol_from_context
from symbol_drift_plot import show_symbol_drift
from symbol_emotion_cluster import show_emotion_clusters

# Updated import for symbol_memory module
import symbol_memory as SM_SymbolMemory

# New import for Phase 1 pruning
from memory_maintenance import prune_phase1_symbolic_vectors
# For getting phase directives, even in this interactive script
from processing_nodes import CurriculumManager

# New import for brain metrics and adaptive weights
from brain_metrics import BrainMetrics

# New import for memory evolution
from memory_evolution_engine import run_memory_evolution

# NEW IMPORTS FOR QUARANTINE, WARFARE, AND VISUALIZATION
from quarantine_layer import UserMemoryQuarantine
from linguistic_warfare import LinguisticWarfareDetector, check_for_warfare
from visualization_prep import VisualizationPrep, visualize_processing_result

# New import for system analytics plots
try:
    from system_analytics import plot_node_activation_timeline, plot_symbol_popularity_timeline, plot_curriculum_metrics
    SYSTEM_ANALYTICS_LOADED = True
except ImportError:
    SYSTEM_ANALYTICS_LOADED = False
    # Skip or dummy out these functions if analytics module not present
    def plot_node_activation_timeline(*args, **kwargs): pass
    def plot_symbol_popularity_timeline(*args, **kwargs): pass
    def plot_curriculum_metrics(*args, **kwargs): pass

# Track input count
input_counter = 0
# Renamed from INPUT_THRESHOLD to be more specific
INPUT_THRESHOLD_SYMBOL_PRUNE = 10  # Run symbol example duplicate pruning every N inputs
INPUT_THRESHOLD_PHASE1_VECTOR_PRUNE = 20  # Run Phase 1 specific vector pruning every M inputs
INPUT_THRESHOLD_WEIGHT_RECOMPUTE = 5  # Recompute adaptive weights every N inputs
INPUT_THRESHOLD_MEMORY_EVOLUTION = 30  # Run memory evolution every N inputs

# Global directives that can be dynamically adjusted
ADAPTIVE_DIRECTIVES = {
    "link_score_weight_static": 0.6,  # Default
    "link_score_weight_dynamic": 0.4,  # Default
    "last_weight_update": None,
    "update_count": 0
}

# Path for persisting adaptive configuration
ADAPTIVE_CONFIG_PATH = Path("data/adaptive_config.json")

# Initialize quarantine and warfare detector globally
quarantine = UserMemoryQuarantine(data_dir="data")
warfare_detector = LinguisticWarfareDetector(data_dir="data")
viz_prep = VisualizationPrep(data_dir="data")

def load_adaptive_config():
    """Load adaptive configuration from disk"""
    global ADAPTIVE_DIRECTIVES
    if ADAPTIVE_CONFIG_PATH.exists():
        try:
            with open(ADAPTIVE_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                ADAPTIVE_DIRECTIVES.update(config)
                print(f"🔧 Loaded adaptive config: Static={ADAPTIVE_DIRECTIVES['link_score_weight_static']:.1%}, "
                      f"Dynamic={ADAPTIVE_DIRECTIVES['link_score_weight_dynamic']:.1%}")
        except Exception as e:
            print(f"⚠️ Could not load adaptive config: {e}")

def save_adaptive_config():
    """Save adaptive configuration to disk"""
    try:
        ADAPTIVE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ADAPTIVE_CONFIG_PATH, 'w') as f:
            json.dump(ADAPTIVE_DIRECTIVES, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save adaptive config: {e}")

def recompute_adaptive_link_weights(force=False):
    """
    Recompute adaptive weights based on brain metrics history.
    Only updates if there's sufficient data and confidence.
    """
    bm = BrainMetrics()
    rec = bm.get_adaptive_weights()
    
    if rec and (force or rec['confidence'] in ['medium', 'high']):
        old_static = ADAPTIVE_DIRECTIVES["link_score_weight_static"]
        old_dynamic = ADAPTIVE_DIRECTIVES["link_score_weight_dynamic"]
        
        # Apply with learning rate to avoid drastic changes
        learning_rate = 0.3 if rec['confidence'] == 'high' else 0.2
        
        new_static = (1 - learning_rate) * old_static + learning_rate * rec["link_score_weight_static"]
        new_dynamic = (1 - learning_rate) * old_dynamic + learning_rate * rec["link_score_weight_dynamic"]
        
        # Normalize
        total = new_static + new_dynamic
        new_static /= total
        new_dynamic /= total
        
        ADAPTIVE_DIRECTIVES["link_score_weight_static"] = round(new_static, 3)
        ADAPTIVE_DIRECTIVES["link_score_weight_dynamic"] = round(new_dynamic, 3)
        ADAPTIVE_DIRECTIVES["last_weight_update"] = datetime.utcnow().isoformat()
        ADAPTIVE_DIRECTIVES["update_count"] = ADAPTIVE_DIRECTIVES.get("update_count", 0) + 1
        
        print(f"🔧 Adaptive weights updated: Static={new_static:.1%} (was {old_static:.1%}), "
              f"Dynamic={new_dynamic:.1%} (was {old_dynamic:.1%}) "
              f"based on {rec['based_on_sessions']} sessions ({rec['confidence']} confidence)")
        
        save_adaptive_config()
        return True
    elif rec:
        print(f"🔧 Not updating weights yet (confidence: {rec['confidence']}). "
              f"Need more data for reliable adjustment.")
    return False

# Regex to detect URLs
def is_url(text):
    return re.match(r"https?://", text.strip()) is not None

# Regex to find emojis (common blocks)
EMOJI_PATTERN = re.compile(
    r"([\U0001F300-\U0001F5FF]|[\U0001F600-\U0001F64F]|"
    r"[\U0001F680-\U0001F6FF]|[\U0001F700-\U0001F77F]|"
    r"[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]|"
    r"[\U0001F900-\U0001F9FF]|[\U0001FA00-\U0001FA6F])"
)

def extract_new_emojis(text, existing_lexicon):
    """
    Return any emoji characters in text that are not yet defined in the lexicon.
    """
    found = set(EMOJI_PATTERN.findall(text))
    return [e for e in found if e not in existing_lexicon]

# --- Acceptance resolution constants ---
ACCEPTANCE_SYMBOL = "🕊️"
ACCEPTANCE_NAME   = "Acceptance"
ACCEPTANCE_KEYWORDS = ["release", "surrender", "let go"]
RESOLUTION_MIN_DEPTH    = 3     # require at least 3 arrows
RESOLUTION_WEIGHT_CUTOFF = 0.25  # resonance_weight below this

# --- Acceptance resolution on meta_symbols.json ---
def perform_acceptance_resolution():
    """
    Scan data/meta_symbols.json for deep, low-weight recursive chains and mark them resolved.
    Also triggers adaptive weight recomputation.
    """
    meta_path = Path("data") / "meta_symbols.json"
    if not meta_path.exists():
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    updated = False
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Ensure acceptance symbol seed
    if ACCEPTANCE_SYMBOL not in meta:
        meta[ACCEPTANCE_SYMBOL] = {
            "name": ACCEPTANCE_NAME,
            "keywords": ACCEPTANCE_KEYWORDS[:],
            "resonance_weight": 0.3,
            "origin": "system_seed",
            "created_at": timestamp,
            "learning_phase": 1,
            "vector_examples": [],
            "usage_count": 0
        }
        updated = True

    for token, entry in meta.items():
        if token == ACCEPTANCE_SYMBOL or entry.get("resolved", False):
            continue
        depth = token.count("⟳")
        weight = entry.get("resonance_weight", 0.0)
        if depth >= RESOLUTION_MIN_DEPTH and weight < RESOLUTION_WEIGHT_CUTOFF:
            entry["resolved"]        = True
            entry["resolved_at"]     = timestamp
            entry["resolved_with"]   = ACCEPTANCE_SYMBOL
            entry["peak_context"]    = entry.get("summary", entry.get("vector_examples", [{}])[0].get("text",""))
            entry["peak_weight"]     = weight
            entry["resolution_note"] = f"depth {depth}, weight {weight:.2f}"
            updated = True

    if updated:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[ACCEPTANCE] Resolved tagging in meta_symbols.json completed.")
    
    # Trigger adaptive weight recomputation after resolution
    recompute_adaptive_link_weights()

def process_web_url_placeholder(url, current_phase_for_storage=1, general_lexicon_for_url=None):
    """
    Simplified processing for a web URL in interactive mode.
    Stores a summary and performs basic analysis.
    Uses adaptive weights for scoring.
    """
    print(f"[INFO] Interactive mode: Processing URL {url} for summary.")
    raw_html = fetch_raw_html(url)
    if raw_html:
        cleaned_text = clean_html_to_text(raw_html)
        if cleaned_text:
            summary_to_store = cleaned_text[:2000]
            
            # Check for warfare patterns in URL content
            should_quarantine_url, warfare_analysis = check_for_warfare(summary_to_store, user_id="url_fetch")
            
            if should_quarantine_url:
                print(f"⚠️ URL content contains warfare patterns. Quarantining...")
                q_result = quarantine.quarantine_user_input(
                    text=summary_to_store,
                    user_id="url_fetch",
                    source_url=url,
                    current_phase=current_phase_for_storage
                )
                return None
            
            vector_memory.store_vector(
                text=summary_to_store,
                source_url=url,
                source_type="user_url_summary_interactive",
                learning_phase=current_phase_for_storage
            )
            print(f"  Stored summary for {url} (tagged for phase {current_phase_for_storage})")

            emotions_output = predict_emotions(summary_to_store)
            verified_emotions = emotions_output.get("verified", [])

            if general_lexicon_for_url is None:
                general_lexicon_for_url = P_Parser.load_seed_symbols()
                general_lexicon_for_url.update(SM_SymbolMemory.load_symbol_memory())

            symbols_weighted = P_Parser.parse_with_emotion(
                summary_to_store,
                verified_emotions,
                current_lexicon=general_lexicon_for_url
            )
            if symbols_weighted:
                print("  Symbols detected in URL summary:")
                for s_info in symbols_weighted[:2]:
                    print(
                        f"    → {s_info['symbol']} ({s_info['name']}) "
                        f"W: {s_info.get('final_weight',0):.2f}"
                    )
                update_symbol_emotions(symbols_weighted, verified_emotions)
                for sym_match_info in symbols_weighted:
                    SM_SymbolMemory.add_symbol(
                        symbol_token=sym_match_info["symbol"],
                        name=sym_match_info["name"],
                        keywords=[sym_match_info.get("matched_keyword","url_summary_match")],
                        initial_emotions=dict(verified_emotions),
                        example_text=summary_to_store,
                        origin="user_url_interactive_match",
                        learning_phase=current_phase_for_storage
                    )
            return summary_to_store
        else:
            print(f"  Could not extract clean text from {url}")
    else:
        print(f"  Could not fetch content from {url}")
    return None

def generate_response(user_input_text, extracted_symbols_weighted, current_phase_directives, visualization_data=None):
    """
    Generates a response based on similar vectors and extracted symbols.
    Uses adaptive weights for similarity scoring.
    Optionally includes visualization data.
    """
    # Apply adaptive weights when retrieving similar vectors
    similar = vector_memory.retrieve_similar_vectors(
        query_text=user_input_text,
        top_n=3,
        similarity_threshold=current_phase_directives.get("logic_node_min_confidence_retrieve", 0.3)
    )

    if not similar and not extracted_symbols_weighted:
        return "I'm still learning and processing that. Nothing specific comes to mind yet based on that input."

    response_parts = []

    if similar:
        response_parts.append("🧠 Here's what I remember that seems related:")
        trust_order = {
            "high_academic_encyclopedic":0, "high_authoritative": 0,
            "high": 1, "user_direct_input": 1,
            "medium": 2, "user_url_summary_interactive": 2,
            "unknown": 3,
            "low_unverified": 4, "low": 4
        }
        similar.sort(key=lambda x: (trust_order.get(x[1].get("source_trust", "unknown"), 3), -x[0]))
        
        for sim_score, memory_item in similar:
            txt = memory_item.get("text", "Unknown text")
            trust = memory_item.get("source_trust", "unknown")
            source_url_mem = memory_item.get("source_url")
            source_note = f" (Source: {source_url_mem})" if source_url_mem else ""
            trust_note = f" (Trust: {trust})" if trust != "unknown" else ""
            phase_learned = memory_item.get("learning_phase", "N/A")

            response_parts.append(
                f"  - \"{txt[:120]}...\" "
                f"(Sim: {sim_score:.2f}, Phase: {phase_learned}{trust_note}{source_note})"
            )
    else:
        response_parts.append("🧠 I don't have a strong factual memory related to that precise input right now.")

    if extracted_symbols_weighted:
        response_parts.append("\n🔗 Symbolic cues I detected in your input:")
        for sym_info in extracted_symbols_weighted[:3]:
            response_parts.append(
                f"  → {sym_info['symbol']} ({sym_info['name']}) "
                f"- Relevance: {sym_info.get('final_weight', 0):.2f}"
            )
    
    # Show current adaptive weights
    response_parts.append(f"\n📊 Current balance: Logic {ADAPTIVE_DIRECTIVES['link_score_weight_static']:.0%} / "
                         f"Symbolic {ADAPTIVE_DIRECTIVES['link_score_weight_dynamic']:.0%}")
    
    # Add visualization summary if available
    if visualization_data:
        response_parts.append("\n🎨 Content Analysis:")
        segment_types = {}
        for seg in visualization_data.get('segments', []):
            seg_type = seg['classification']
            segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
        
        for seg_type, count in segment_types.items():
            emoji = {'logic': '🧮', 'symbolic': '❤️', 'bridge': '🤔'}.get(seg_type, '❓')
            response_parts.append(f"  {emoji} {seg_type.capitalize()}: {count} segments")
    
    return "\n".join(response_parts)

def main():
    global input_counter

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-diagnostics", action="store_true",
        help="Skip heavy exit diagnostics and plotting"
    )
    parser.add_argument(
        "--reset-weights", action="store_true",
        help="Reset adaptive weights to defaults"
    )
    parser.add_argument(
        "--disable-quarantine", action="store_true",
        help="Disable quarantine for testing (not recommended)"
    )
    args = parser.parse_args()

    # Load adaptive configuration at startup
    if args.reset_weights:
        print("🔧 Resetting adaptive weights to defaults...")
        ADAPTIVE_DIRECTIVES["link_score_weight_static"] = 0.6
        ADAPTIVE_DIRECTIVES["link_score_weight_dynamic"] = 0.4
        save_adaptive_config()
    else:
        load_adaptive_config()

    print("🧠 Hybrid AI: Symbolic + Vector Memory (Optimizer Mode with Adaptive Weights)")
    print("🛡️ Quarantine & Warfare Protection: " + ("DISABLED ⚠️" if args.disable_quarantine else "ENABLED ✅"))
    print("\nCommands:")
    print("  - Type text or paste a URL to process")
    print("  - Type 'evolve' to run memory evolution")
    print("  - Type 'stats' to see quarantine/warfare statistics")
    print("  - Type 'exit' or 'quit' to end session")
    print()

    # Initial weight computation based on historical data
    recompute_adaptive_link_weights()

    temp_curriculum_manager = CurriculumManager()
    current_phase_for_interaction = temp_curriculum_manager.get_current_phase()
    
    general_lexicon = P_Parser.load_seed_symbols()
    general_lexicon.update(SM_SymbolMemory.load_symbol_memory())
    print(f"Interactive mode using context of Phase {current_phase_for_interaction}. Loaded {len(general_lexicon)} symbols for matching.")

    while True:
        current_interaction_directives = temp_curriculum_manager.get_processing_directives(current_phase_for_interaction)
        # Override to allow new symbol generation
        current_interaction_directives["allow_new_symbol_generation"] = True
        # Apply adaptive weights to directives
        current_interaction_directives["link_score_weight_static"] = ADAPTIVE_DIRECTIVES["link_score_weight_static"]
        current_interaction_directives["link_score_weight_dynamic"] = ADAPTIVE_DIRECTIVES["link_score_weight_dynamic"]

        user_input = input("💬 You: ").strip()
        
        # Check for special commands
        if user_input.lower() == "evolve":
            print("\n🧬 Running Memory Evolution Cycle...")
            evolution_config = {
                'reverse_audit_confidence_threshold': 0.3,
                'enable_reverse_migration': True,
                'enable_weight_evolution': True,
                'save_detailed_logs': True
            }
            try:
                evolution_results = run_memory_evolution(data_dir='data', config=evolution_config)
                print(f"\n✅ Evolution complete!")
                
                # Update the general lexicon with any changes
                general_lexicon = P_Parser.load_seed_symbols()
                general_lexicon.update(SM_SymbolMemory.load_symbol_memory())
            except Exception as e:
                print(f"❌ Evolution failed: {e}")
                import traceback
                traceback.print_exc()
            continue
        
        if user_input.lower() == "stats":
            print("\n📊 System Statistics:")
            q_stats = quarantine.get_quarantine_stats()
            w_stats = warfare_detector.get_defense_statistics()
            
            print(f"\n🔒 Quarantine:")
            print(f"  Total quarantined: {q_stats['total_quarantined']}")
            print(f"  Warfare attempts: {q_stats['warfare_attempts']} ({q_stats['warfare_percentage']:.1f}%)")
            print(f"  Unique users: {q_stats['unique_users']}")
            
            print(f"\n🛡️ Warfare Defense:")
            print(f"  Total checks: {w_stats['total_checks']}")
            print(f"  Threats detected: {w_stats['threats_detected']} ({w_stats['threat_percentage']:.1f}%)")
            print(f"  Last 24h: {w_stats['checks_last_24h']} checks, {w_stats['threats_last_24h']} threats")
            continue
        
        if user_input.lower() in ("exit", "quit"):
            break

        # Initialize visualization data
        visualization_data = None

        if is_url(user_input):
            print("🌐 Detected URL. Processing summary...")
            user_input_for_response = process_web_url_placeholder(
                user_input,
                current_phase_for_interaction,
                general_lexicon
            )
            if not user_input_for_response:
                print("  Could not process URL content.")
                continue
        else:
            # User text input - check for warfare patterns first
            print("📝 Detected text input. Analyzing for security threats...")
            
            if not args.disable_quarantine:
                # Check for linguistic warfare
                should_quarantine, warfare_analysis = check_for_warfare(user_input, user_id="interactive_user")
                
                if should_quarantine:
                    print(f"\n⚠️ SECURITY ALERT: {warfare_analysis['defense_strategy']['explanation']}")
                    print(f"🔒 Input quarantined. Strategy: {warfare_analysis['defense_strategy']['strategy']}")
                    
                    # Quarantine the input
                    q_result = quarantine.quarantine_user_input(
                        text=user_input,
                        user_id="interactive_user",
                        current_phase=current_phase_for_interaction
                    )
                    
                    # Show threat details
                    for threat in warfare_analysis['threats_detected'][:3]:
                        print(f"  - {threat['type']}: {threat['description']}")
                    
                    # Still provide a minimal response
                    response_parts = [
                        "I've detected potentially harmful patterns in your input.",
                        f"Threat level: {warfare_analysis['threat_score']:.1%}",
                        "Your input has been quarantined for safety.",
                        "Please try rephrasing your request in a more constructive way."
                    ]
                    print("\n🗣️ AI Response:")
                    print("\n".join(response_parts))
                    continue
                else:
                    print("✅ No threats detected. Processing normally...")
            
            # Normal processing for safe input
            user_input_for_response = user_input
            vector_memory.store_vector(
                text=user_input, 
                source_type="user_direct_input",
                learning_phase=current_phase_for_interaction
            )

        # Process emotions and symbols
        emotions_output = predict_emotions(user_input_for_response)
        verified_emotions = emotions_output.get("verified", [])
        print("\n💓 Emotions detected in input:")
        for tag, score in verified_emotions:
            print(f"   → {tag} ({score:.2f})")
        
        symbols_weighted = P_Parser.parse_with_emotion(
            user_input_for_response,
            verified_emotions,
            current_lexicon=general_lexicon
        )
        
        # Auto-capture and store novel emojis
        new_emojis = extract_new_emojis(user_input_for_response, general_lexicon)
        for e in new_emojis:
            name = unicodedata.name(e, e)
            SM_SymbolMemory.add_symbol(
                symbol_token=e, name=name,
                keywords=[name.lower()],
                initial_emotions={},
                example_text=user_input_for_response,
                origin="auto_emoji_capture",
                learning_phase=current_phase_for_interaction
            )
        if new_emojis:
            general_lexicon.update(SM_SymbolMemory.load_symbol_memory())

        if symbols_weighted:
            update_symbol_emotions(symbols_weighted, verified_emotions)
            for s_info in symbols_weighted:
                SM_SymbolMemory.add_symbol(
                    symbol_token=s_info["symbol"],
                    name=s_info["name"],
                    keywords=[s_info.get("matched_keyword","unknown_match")],
                    initial_emotions=dict(verified_emotions),
                    example_text=user_input_for_response,
                    origin="user_interaction_match",
                    learning_phase=current_phase_for_interaction
                )
            print("\n✨ Symbols extracted from input:")
            for s in symbols_weighted:
                print(f"   → {s['symbol']} ({s['name']})" +
                      f" [Matched: {s.get('matched_keyword','N/A')}, W: {s.get('final_weight',0):.2f}]")
        else:
            print("🌀 No specific symbolic units extracted from input based on current lexicon.")

        if not symbols_weighted and current_interaction_directives.get("allow_new_symbol_generation", False):
            keywords_for_gen = P_Parser.extract_keywords(user_input_for_response)
            if keywords_for_gen:
                new_sym = generate_symbol_from_context(
                    user_input_for_response, keywords_for_gen, verified_emotions
                )
                if new_sym:
                    print(f"💡 Suggested new symbol: {new_sym['symbol']} - {new_sym['name']}")
                    SM_SymbolMemory.add_symbol(
                        symbol_token=new_sym['symbol'],
                        name=new_sym['name'],
                        keywords=new_sym['keywords'],
                        initial_emotions=new_sym['emotions'],
                        example_text=user_input_for_response,
                        origin=new_sym['origin'],
                        learning_phase=current_phase_for_interaction,
                        resonance_weight=new_sym.get('resonance_weight',0.5)
                    )
                    general_lexicon.update(SM_SymbolMemory.load_symbol_memory())

        # Prepare visualization data
        processing_result = {
            'decision_type': 'FOLLOW_HYBRID',  # Simplified for interactive mode
            'logic_score': len(P_Parser.extract_keywords(user_input_for_response)) * 2,
            'symbolic_score': len(symbols_weighted) * 3,
            'confidence': 0.7,
            'source_type': 'user_direct_input',
            'processing_phase': current_phase_for_interaction,
            'symbols_found': len(symbols_weighted)
        }
        
        visualization_data = viz_prep.prepare_text_for_display(
            user_input_for_response,
            processing_result,
            include_emotions=True,
            include_symbols=True
        )

        response = generate_response(
            user_input_for_response,
            symbols_weighted,
            current_interaction_directives,
            visualization_data
        )
        print("\n🗣️ AI Response:")
        print(response)

        input_counter += 1
        
        # Periodic maintenance tasks
        if input_counter % INPUT_THRESHOLD_SYMBOL_PRUNE == 0:
            print("\n🧹 Periodic symbol duplicate pruning...")
            SM_SymbolMemory.prune_duplicates()
            
        if input_counter % INPUT_THRESHOLD_PHASE1_VECTOR_PRUNE == 0:
            if current_phase_for_interaction == 1:
                print("\n🧹 Periodic Phase 1 vector pruning...")
                prune_phase1_symbolic_vectors(archive_path_str="data/optimizer_archived_phase1_vectors.json")
            else:
                print(f"\n(Skipping Phase 1 prune; Phase {current_phase_for_interaction})")
                
        if input_counter % INPUT_THRESHOLD_WEIGHT_RECOMPUTE == 0:
            print("\n🔧 Periodic adaptive weight recomputation...")
            recompute_adaptive_link_weights()
            
        # Periodic memory evolution
        if input_counter % INPUT_THRESHOLD_MEMORY_EVOLUTION == 0:
            print("\n🧬 Periodic memory evolution...")
            try:
                evolution_results = run_memory_evolution(data_dir='data')
                if evolution_results['migrated'] > 0 or evolution_results['reversed'] > 0:
                    print(f"  → Evolved: {evolution_results['migrated']} migrated, "
                          f"{evolution_results['reversed']} reversed")
            except:
                pass  # Silent fail for periodic runs

    # --- On exit: run full diagnostics ---
    print("\n--- Session End Diagnostics ---")

    if not args.skip_diagnostics:
        try: 
            cluster_vectors_and_plot(show_graph=True)
        except: 
            pass
        try: 
            show_trail_graph()
        except: 
            pass
        try: 
            show_symbol_drift()
        except: 
            pass
        try: 
            show_emotion_clusters()
        except: 
            pass
        if SYSTEM_ANALYTICS_LOADED:
            try:
                plot_node_activation_timeline()
                plot_symbol_popularity_timeline()
                plot_curriculum_metrics()
            except: 
                pass
        SM_SymbolMemory.prune_duplicates()
        prune_phase1_symbolic_vectors(archive_path_str="data/optimizer_archived_phase1_vectors_final.json")
        
        # Run memory evolution as part of cleanup
        print("\n🧬 Running final memory evolution...")
        try:
            evolution_config = {
                'reverse_audit_confidence_threshold': 0.3,
                'enable_reverse_migration': True,
                'enable_weight_evolution': True,
                'save_detailed_logs': True
            }
            evolution_results = run_memory_evolution(data_dir='data', config=evolution_config)
            print(f"✅ Final evolution: {evolution_results['migrated']} migrated, "
                  f"{evolution_results['reversed']} reversed")
        except Exception as e:
            print(f"⚠️ Memory evolution skipped: {e}")

    # Acceptance resolution for unresolved recursion patterns
    perform_acceptance_resolution()
    
    # Final adaptive weight computation
    print("\n🔧 Final adaptive weight check...")
    if recompute_adaptive_link_weights():
        print("   Weights updated based on session data.")
    
    # Show brain metrics summary
    try:
        from brain_metrics import display_metrics_summary
        display_metrics_summary()
    except:
        pass
    
    # Final quarantine and warfare statistics
    print("\n📊 Final Security Statistics:")
    q_stats = quarantine.get_quarantine_stats()
    w_stats = warfare_detector.get_defense_statistics()
    print(f"  Quarantined: {q_stats['total_quarantined']} items")
    print(f"  Warfare attempts: {q_stats['warfare_attempts']}")
    print(f"  Defense success rate: {100 - w_stats['threat_percentage']:.1f}%")

    print("\nOptimizer session ended. Goodbye!")

if __name__ == "__main__":
    # Ensure data directories are set up
    DATA_DIR_MO = Path("data")
    DATA_DIR_MO.mkdir(parents=True, exist_ok=True)
    # Initialize memory files if missing
    for fn in ["symbol_memory.json","seed_symbols.json","meta_symbols.json"]:
        path = DATA_DIR_MO/fn
        if not path.exists() or path.stat().st_size==0:
            with open(path,"w",encoding="utf-8") as f:
                json.dump({},f)
    main()