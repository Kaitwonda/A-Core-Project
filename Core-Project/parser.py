# parser.py
import spacy
import json
from pathlib import Path

# Load spaCy model
NLP_MODEL_LOADED = False
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
    NLP_MODEL_LOADED = True
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("[ERROR] spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    print("         Keyword extraction will be limited.")

# Load seed symbols (archetypes, meanings, etc.)
SEED_PATH = Path("data/seed_symbols.json")
SEED_SYMBOLS = {} # Initialize as empty dict
if SEED_PATH.exists():
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        try:
            SEED_SYMBOLS = json.load(f)
            print(f"🌱 Seed symbols loaded: {len(SEED_SYMBOLS)} entries from {SEED_PATH}.")
        except json.JSONDecodeError:
            print(f"[WARNING] Seed symbols file {SEED_PATH} is corrupted. Using empty seeds.")
else:
    print(f"[WARNING] Seed symbols file not found at {SEED_PATH}. Using empty seeds.")


# Load evolving emotional weight map (used for emotion weighting in parse_with_emotion)
EMOTION_MAP_PATH = Path("data/symbol_emotion_map.json")
def load_emotion_map(): 
    if EMOTION_MAP_PATH.exists():
        with open(EMOTION_MAP_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"[WARNING] Symbol emotion map {EMOTION_MAP_PATH} corrupted. Returning empty map.")
                return {}
    # print(f"ℹ️ Symbol emotion map not found at {EMOTION_MAP_PATH}. Returning empty map.") # Can be chatty
    return {}

# --- CORE PARSING MODULES ---

def extract_keywords(text):
    """Extract lemmatized key tokens (nouns, verbs, adjectives) from input text."""
    if not NLP_MODEL_LOADED or nlp is None: # Check if model loaded
        # print("[WARNING] spaCy model not loaded. Keyword extraction might be basic (splitting text).")
        # Basic fallback if spaCy isn't available
        return [word.lower() for word in text.split() if len(word) > 3] # Very simple fallback
    
    doc = nlp(text)
    keywords = set()
    for token in doc:
        if token.pos_ in {"NOUN", "VERB", "ADJ"} and not token.is_stop and len(token.lemma_) > 2: # Min length for lemma
            keywords.add(token.lemma_.lower())
    return list(keywords)

def match_keywords_to_symbols(keywords, lexicon_to_search): # No default, always expect a lexicon
    """
    Match input keywords to symbols from the provided lexicon.
    Args:
        keywords (list): A list of keywords extracted from text.
        lexicon_to_search (dict): The symbol lexicon to search (e.g., combined seeds, memory, meta).
    Returns:
        list: A list of matched symbol entries (dictionaries containing symbol token, name, matched_keyword).
    """
    if not lexicon_to_search: 
        # print("[INFO] Parser: Lexicon to search is empty in match_keywords_to_symbols.")
        return []
        
    extracted_matches = []
    processed_keywords = set(str(kw).lower() for kw in keywords if kw) # Normalize input keywords once

    for symbol_token, symbol_data in lexicon_to_search.items():
        if not isinstance(symbol_data, dict): 
            continue

        # Build a comprehensive list of searchable terms for the current symbol
        searchable_terms_for_symbol = set() # Use a set for faster lookups
        
        # 1. 'keywords' field (from symbol_memory, meta_symbols)
        for kw_val in symbol_data.get("keywords", []):
            if kw_val: searchable_terms_for_symbol.add(str(kw_val).lower())
        
        # 2. 'core_meanings' (from seed_symbols, meta_symbols)
        for cm_val in symbol_data.get("core_meanings", []):
            if cm_val: searchable_terms_for_symbol.add(str(cm_val).lower())
        
        # 3. 'name' of the symbol
        symbol_name = symbol_data.get("name")
        if symbol_name: 
            searchable_terms_for_symbol.add(str(symbol_name).lower())
            # Also add individual words from the name if it's multi-word
            searchable_terms_for_symbol.update(str(symbol_name).lower().split())


        # 4. Emotion names associated with the symbol's definition (not the learned emotion_profile)
        # Seed symbols have 'emotions': ["anger", "passion"]
        # Symbol memory (from generator) has 'emotions': {"anger": 0.8} (context of creation)
        raw_emotions_field = symbol_data.get("emotions", []) # Default to list
        if isinstance(raw_emotions_field, list): 
            for e_val in raw_emotions_field:
                if e_val: searchable_terms_for_symbol.add(str(e_val).lower())
        elif isinstance(raw_emotions_field, dict): 
            for e_key in raw_emotions_field.keys():
                if e_key: searchable_terms_for_symbol.add(str(e_key).lower())
        
        # 5. 'archetypes' (common in seed_symbols)
        for arch_val in symbol_data.get("archetypes", []):
            if arch_val: searchable_terms_for_symbol.add(str(arch_val).lower())
        
        # Perform matching
        for input_kw in processed_keywords: 
            if input_kw in searchable_terms_for_symbol:
                extracted_matches.append({
                    "symbol": symbol_token, 
                    "name": symbol_data.get("name", "Unknown Symbol Name"),
                    "matched_keyword": input_kw
                })
                break # Match found for this symbol based on one keyword, move to the next symbol
    return extracted_matches

def parse_with_emotion(text, detected_emotions, current_lexicon): # current_lexicon is now mandatory
    """
    Enhanced parser that adjusts symbol matches using current emotional context
    and searches within the provided (active) lexicon.
    Args:
        text (str): The input text.
        detected_emotions (list): List of tuples like [('fear', 0.76), ...] from emotion_handler.
        current_lexicon (dict): The full symbol lexicon (seeds, memory, meta) to use.
    Returns:
        list: Sorted list of matched symbol entries, each a dict with relevant fields.
    """
    if not current_lexicon:
        # print("[INFO] Parser: parse_with_emotion called with an empty current_lexicon.")
        return []

    emotion_map_for_weighting = load_emotion_map() # This is data/symbol_emotion_map.json
    input_keywords = extract_keywords(text)
            
    # Get initial keyword-based matches from the active lexicon
    initial_matches = match_keywords_to_symbols(input_keywords, current_lexicon)
    
    symbols_with_emotional_weight = []

    for match_info in initial_matches:
        symbol_token = match_info["symbol"]
        
        emotional_weight = 0.0
        symbol_specific_learned_emotion_profile = emotion_map_for_weighting.get(symbol_token, {})
        contributing_influencers = [] 

        for detected_emo_str, detected_emo_score in detected_emotions: 
            learned_association_strength = symbol_specific_learned_emotion_profile.get(detected_emo_str, 0.0)
            contribution = learned_association_strength * detected_emo_score 
            if contribution > 0.01: 
                contributing_influencers.append((detected_emo_str, round(contribution,3)))
            emotional_weight += contribution
        
        symbols_with_emotional_weight.append({
            "symbol": symbol_token,
            "name": match_info["name"], 
            "matched_keyword": match_info["matched_keyword"], 
            "emotional_weight": round(emotional_weight, 3),
            "influencing_emotions": sorted(contributing_influencers, key=lambda x: -x[1])
        })

    symbols_with_emotional_weight.sort(key=lambda x: x["emotional_weight"], reverse=True)
    return symbols_with_emotional_weight


if __name__ == '__main__':
    print("Testing parser.py with active lexicon handling...")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Ensure SEED_SYMBOLS is populated for the test
    if not SEED_SYMBOLS:
        dummy_seeds = {
            "🔥": {"name": "Fire", "keywords": ["fire", "flame", "heat"], "core_meanings": ["destruction", "passion"], "emotions": ["anger", "passion"], "archetypes": ["destroyer"], "learning_phase": 0},
            "💧": {"name": "Water", "keywords": ["water", "rain", "tear"], "core_meanings": ["emotion", "flow"], "emotions": ["sadness", "calm"], "archetypes": ["healer"], "learning_phase": 0}
        }
        # This global modification is just for the __main__ test block
        globals()['SEED_SYMBOLS'] = dummy_seeds 
        print(f"Using dummy SEED_SYMBOLS for test: {len(SEED_SYMBOLS)} entries.")


    dummy_emotion_map = {
        "🔥": {"anger": 0.8, "passion": 0.7, "fear": 0.3},
        "💧": {"sadness": 0.9, "calm": 0.6, "joy": 0.1},
    }
    if not EMOTION_MAP_PATH.exists():
        with open(EMOTION_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(dummy_emotion_map, f, indent=2)
        print(f"Created dummy {EMOTION_MAP_PATH} for testing.")
    
    test_text_1 = "The raging fire brought immense passion but also anger and destruction."
    print(f"\nInput text 1: {test_text_1}")
    keywords_1 = extract_keywords(test_text_1)
    print(f"Extracted Keywords 1: {keywords_1}")

    # Test match_keywords_to_symbols with SEED_SYMBOLS
    matched_basic_1 = match_keywords_to_symbols(keywords_1, SEED_SYMBOLS) # Explicitly pass SEED_SYMBOLS
    print(f"\nBasic Symbol Matches for Text 1 (from SEED_SYMBOLS):")
    for match in matched_basic_1:
        print(f"  Symbol: {match['symbol']} ({match['name']}), Matched Keyword: {match['matched_keyword']}")

    sample_detected_emotions_1 = [("anger", 0.9), ("passion", 0.8), ("fear", 0.5)]
    print(f"\nTesting parse_with_emotion for Text 1 with detected emotions: {sample_detected_emotions_1}")
    emotion_weighted_matches_1 = parse_with_emotion(test_text_1, sample_detected_emotions_1, current_lexicon=SEED_SYMBOLS)
    print("Emotion-Weighted Symbol Matches for Text 1 (from SEED_SYMBOLS):")
    for match in emotion_weighted_matches_1:
        print(f"  Symbol: {match['symbol']} ({match['name']}), Matched: '{match['matched_keyword']}', "
              f"Weight: {match['emotional_weight']}, Influencers: {match['influencing_emotions']}")

    print("-" * 30)
    
    # Simulate an active lexicon that includes a meta-symbol and a learned symbol
    learned_symbol_example = {
        "💻": {"name": "Computation", "keywords": ["computation", "algorithm", "processing"], "origin": "emergent", "learning_phase": 1}
    }
    meta_symbol_example = {
        "🔥⟳": {"name": "Fire Cycle", "keywords": ["fire", "cycle", "rebirth", "transformation"], "core_meanings": ["fire cycle"], "origin":"meta_binding", "learning_phase": 2}
    }
    active_test_lexicon = {**SEED_SYMBOLS, **learned_symbol_example, **meta_symbol_example}
    
    # Ensure symbol_emotion_map has entries for these test symbols if we want them weighted
    # For this test, we'll assume they might not have strong learned emotional profiles yet.
    # emotion_map_for_weighting = load_emotion_map()
    # emotion_map_for_weighting["💻"] = {"focus": 0.6}
    # emotion_map_for_weighting["🔥⟳"] = {"transformation": 0.7, "hope": 0.5}
    # with open(EMOTION_MAP_PATH, "w", encoding="utf-8") as f: json.dump(emotion_map_for_weighting, f, indent=2)


    test_text_2 = "The fire cycle represents a computational algorithm of rebirth."
    print(f"\nInput text 2: {test_text_2}")
    keywords_2 = extract_keywords(test_text_2)
    print(f"Extracted Keywords 2: {keywords_2}")
    
    sample_detected_emotions_2 = [("transformation", 0.8), ("hope", 0.6), ("focus", 0.5)]
    print(f"\nTesting parse_with_emotion for Text 2 with detected emotions: {sample_detected_emotions_2} using an active_test_lexicon.")
    emotion_weighted_matches_2 = parse_with_emotion(test_text_2, sample_detected_emotions_2, current_lexicon=active_test_lexicon)
    print("Emotion-Weighted Symbol Matches for Text 2 (from active_test_lexicon):")
    if not emotion_weighted_matches_2: print("  No matches found.")
    for match in emotion_weighted_matches_2:
        print(f"  Symbol: {match['symbol']} ({match['name']}), Matched: '{match['matched_keyword']}', "
              f"Weight: {match['emotional_weight']}, Influencers: {match['influencing_emotions']}")
