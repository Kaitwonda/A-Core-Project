# parser.py
import spacy
import json
from pathlib import Path
import re # For chunk_content

# Load spaCy model
NLP_MODEL_LOADED = False
nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
    NLP_MODEL_LOADED = True
    print("✅ spaCy model 'en_core_web_sm' loaded successfully for parser.py.")
except OSError:
    print("[ERROR] spaCy model 'en_core_web_sm' not found for parser.py. Please run: python -m spacy download en_core_web_sm")
    print("         Keyword extraction and advanced parsing will be limited.")

SEED_PATH = Path("data/seed_symbols.json")
EMOTION_MAP_PATH = Path("data/symbol_emotion_map.json") # Used by parse_with_emotion
EMOTION_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_seed_symbols(file_path=SEED_PATH):
    file_path_obj = Path(file_path)
    file_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if file_path_obj.exists() and file_path_obj.stat().st_size > 0: # Check size
        with open(file_path_obj, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # print(f"🌱 Seed symbols loaded: {len(data)} entries from {file_path_obj}.")
                return data
            except json.JSONDecodeError:
                print(f"[PARSER-WARNING] Seed symbols file {file_path_obj} is corrupted. Using empty seeds.")
                return {}
    else:
        # print(f"[PARSER-WARNING] Seed symbols file not found at {file_path_obj} or empty. Creating/using default and returning empty for now.")
        default_seeds = {
            "🔥": {"name": "Fire", "keywords": ["fire", "flame", "computation", "logic"], "core_meanings": ["heat"], "emotions": ["anger"], "archetypes": ["destroyer"], "learning_phase": 0, "resonance_weight": 0.7},
            "💧": {"name": "Water", "keywords": ["water", "liquid", "data", "flow"], "core_meanings": ["flow"], "emotions": ["calm"], "archetypes": ["healer"], "learning_phase": 0, "resonance_weight": 0.7},
            "💻": {"name": "Computer", "keywords": ["computer", "computation", "cpu", "binary", "code", "algorithm", "system", "architecture"], "core_meanings": ["processing", "logic unit"], "emotions": ["neutral", "focus"], "archetypes": ["tool", "oracle"], "learning_phase": 0, "resonance_weight": 0.8}
        }
        if not file_path_obj.exists() or file_path_obj.stat().st_size == 0:
            try:
                with open(file_path_obj, "w", encoding="utf-8") as f:
                    json.dump(default_seeds, f, indent=2, ensure_ascii=False)
                # print(f"   Created default seed file at {file_path_obj}")
            except Exception as e:
                print(f"   [PARSER-ERROR] Could not create default seed file: {e}")
        return {}

def load_emotion_map(file_path=EMOTION_MAP_PATH):
    if file_path.exists() and file_path.stat().st_size > 0: # Check size
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"[PARSER-WARNING] Symbol emotion map {file_path} is corrupted. Using empty map.")
                return {}
    return {}

def extract_keywords(text_input, max_keywords=10):
    if not text_input or not isinstance(text_input, str): return []
    keywords = []
    if NLP_MODEL_LOADED and nlp:
        doc = nlp(text_input.lower()[:nlp.max_length]) # Added max_length guard
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]: # Added ADJ
                    keywords.append(token.lemma_)
        keywords = list(dict.fromkeys(keywords))
    else:
        words = re.findall(r'\b\w{3,}\b', text_input.lower())
        stop_words = {"the", "a", "is", "in", "it", "to", "and", "of", "for", "on", "with", "as", "by", "an", "this", "that", "was", "were", "be"}
        keywords = [word for word in words if word not in stop_words]
        keywords = list(dict.fromkeys(keywords))
    return keywords[:max_keywords]

def chunk_content(text, max_chunk_size=1000, overlap=100):
    if not text or not isinstance(text, str): return []
    chunks = []
    if NLP_MODEL_LOADED and nlp:
        doc = nlp(text[:nlp.max_length]) # Guard very long texts for sentence tokenization
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    current_chunk_parts = []
    current_length = 0
    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)
        if not sentence: continue

        # If current sentence itself is too long, split it first
        if sentence_len > max_chunk_size:
            if current_chunk_parts: # Save accumulated before hard split
                chunks.append(" ".join(current_chunk_parts))
                current_chunk_parts, current_length = [], 0
            
            # Hard split the long sentence
            sub_sentence_start = 0
            while sub_sentence_start < sentence_len:
                end_point = min(sub_sentence_start + max_chunk_size, sentence_len)
                # Try to find a space if possible, otherwise hard break
                actual_end_point = sentence.rfind(" ", sub_sentence_start, end_point) if " " in sentence[sub_sentence_start:end_point] else end_point
                if actual_end_point <= sub_sentence_start : actual_end_point = end_point
                
                chunks.append(sentence[sub_sentence_start:actual_end_point].strip())
                sub_sentence_start = actual_end_point + (1 if sentence[actual_end_point:actual_end_point+1] == " " else 0)
            continue # Move to next sentence from input

        # If adding the new sentence exceeds max length for the current_chunk
        if current_length + sentence_len + (1 if current_chunk_parts else 0) > max_chunk_size and current_chunk_parts:
            chunks.append(" ".join(current_chunk_parts))
            
            # Create overlap
            overlap_parts = []
            overlap_len = 0
            # Iterate backwards through sentences of the chunk just added
            temp_overlap_source = list(current_chunk_parts) # Work on a copy
            while temp_overlap_source:
                part_to_add = temp_overlap_source.pop() # Get last sentence from previous chunk
                part_len = len(part_to_add)
                if overlap_len + part_len + (1 if overlap_parts else 0) <= overlap:
                    overlap_parts.insert(0, part_to_add)
                    overlap_len += part_len + (1 if len(overlap_parts) > 1 else 0)
                else:
                    break # Overlap is full enough or this sentence is too big
            
            current_chunk_parts = overlap_parts
            current_length = overlap_len
        
        # Add current sentence to the (potentially new/overlapped) chunk
        if current_chunk_parts: current_chunk_parts.append(sentence)
        else: current_chunk_parts = [sentence]
        current_length += sentence_len + (1 if len(current_chunk_parts) > 1 else 0)

    if current_chunk_parts:
        chunks.append(" ".join(current_chunk_parts))
        
    return [c for c in chunks if c.strip()]


def extract_symbolic_units(text_input, current_lexicon):
    if not current_lexicon: return []
    extracted = []
    text_keywords = extract_keywords(text_input, max_keywords=25) # Get more keywords for matching
    text_lower_for_direct_match = text_input.lower()

    for token_symbol, details in current_lexicon.items():
        symbol_keywords = details.get("keywords", [])
        # More robust matching:
        # 1. Check if symbol token itself is in text (e.g., "🔥" in "Fire 🔥 is hot")
        # 2. Check if any symbol keyword is a WHOLE WORD in the text (to avoid "art" in "start")
        # 3. Check if any symbol keyword is in extracted text_keywords (lemma match)
        
        matched_kw = None
        symbol_token_lower = token_symbol.lower()

        if f" {symbol_token_lower} " in f" {text_lower_for_direct_match} " or \
           text_lower_for_direct_match.startswith(f"{symbol_token_lower} ") or \
           text_lower_for_direct_match.endswith(f" {symbol_token_lower}") or \
           text_lower_for_direct_match == symbol_token_lower:
            matched_kw = token_symbol # Matched the symbol token itself
        else:
            for sk in symbol_keywords:
                sk_lower = sk.lower()
                # Whole word match in text
                if re.search(r'\b' + re.escape(sk_lower) + r'\b', text_lower_for_direct_match):
                    matched_kw = sk
                    break
                # Lemma match in extracted keywords
                if sk_lower in text_keywords: # text_keywords are already lemmas and lowercased
                    matched_kw = sk
                    break
        
        if matched_kw:
            extracted.append({
                "symbol": token_symbol,
                "name": details.get("name", "Unknown Symbol"),
                "matched_keyword": matched_kw
            })
    return extracted

def parse_input(text_input, current_lexicon=None):
    if current_lexicon is None: current_lexicon = load_seed_symbols()
    keywords = extract_keywords(text_input)
    symbols = extract_symbolic_units(text_input, current_lexicon)
    return {"keywords": keywords, "symbols": symbols}


# APPENDED/MODIFIED: parse_with_emotion function
def parse_with_emotion(text_input, detected_emotions_verified, current_lexicon):
    """
    Identifies symbols in text and weights their relevance based on:
    1. Base emotional profile of the symbol (from global symbol_emotion_map.json).
    2. Current contextual emotions detected in the text_input.
    Args:
        text_input (str): The text to parse.
        detected_emotions_verified (list of tuples): e.g., [('joy', 0.8), ('curiosity', 0.6)]
                                                     This should be the 'verified' list from emotion_handler.
        current_lexicon (dict): The active symbols to search for.
    Returns:
        list: List of matched symbols with their contextual emotional weights.
              e.g., [{'symbol': '💡', 'name': 'Idea', ..., 'final_weight': Z, 'influencing_emotions': [('curiosity', 0.6)]}]
    """
    if not current_lexicon:
        print("[PARSER-ERROR] parse_with_emotion requires a current_lexicon.")
        return []
    if not isinstance(detected_emotions_verified, list): # Expecting a list of tuples
        print(f"[PARSER-WARNING] detected_emotions_verified is not a list: {detected_emotions_verified}. Using empty emotions.")
        detected_emotions_verified = []

    symbol_emotion_profiles_map = load_emotion_map() # data/symbol_emotion_map.json
    matched_symbols_in_text = extract_symbolic_units(text_input, current_lexicon)
    
    emotionally_weighted_symbols = []
    current_text_emotions_map = {emo_label.lower(): score for emo_label, score in detected_emotions_verified if emo_label and isinstance(score, (float, int))}

    for matched_sym_info in matched_symbols_in_text:
        symbol_token = matched_sym_info["symbol"]
        symbol_details = current_lexicon.get(symbol_token, {})
        symbol_name = symbol_details.get("name", symbol_token)

        # 1. Base Emotional Weight: How much does the symbol's *learned general emotional profile*
        #    (from symbol_emotion_map.json) align with the *current text's emotions*?
        base_weight = 0.3 # Default base weight, slightly above ignore threshold
        symbol_general_profile = symbol_emotion_profiles_map.get(symbol_token, {})
        
        if symbol_general_profile:
            alignment_score = 0
            common_emotions_count = 0
            for text_emo, text_emo_strength in current_text_emotions_map.items():
                if text_emo in symbol_general_profile: # text_emo is already lowercased
                    alignment_score += symbol_general_profile[text_emo] * text_emo_strength # Weighted by text emotion strength
                    common_emotions_count +=1
            if common_emotions_count > 0:
                base_weight = alignment_score / common_emotions_count
            elif not symbol_general_profile and not current_text_emotions_map: # No emotion info anywhere
                base_weight = 0.5 # Neutral if no info
            # If symbol_general_profile exists but no common_emotions, base_weight remains low (default 0.3)
        
        base_weight = min(1.0, max(0.05, base_weight)) # Normalize

        # 2. Contextual Emotional Weight: How much do the *symbol's inherent/defined emotions*
        #    (from seed_symbols.json or symbol_memory.json) resonate with the *current text's emotions*?
        contextual_weight = 0.0
        influencing_emotions_for_this_match = []
        
        # Get symbol's defined emotions (e.g., from seed_symbols "emotions" list or symbol_memory)
        defined_symbol_emotions_raw = symbol_details.get("emotions", [])
        defined_symbol_emotions_lc = []
        
        # FIXED: Handle emotions whether they're stored as dict or list
        if isinstance(defined_symbol_emotions_raw, dict):
            # If someone stored a dict (emotion->weight), treat keys as emotions
            defined_symbol_emotions_lc = [emo.lower() for emo in defined_symbol_emotions_raw.keys()]
        elif isinstance(defined_symbol_emotions_raw, list) and defined_symbol_emotions_raw:
            # If it's a list, only index into it when it really is a list
            first = defined_symbol_emotions_raw[0]
            if isinstance(first, dict):
                defined_symbol_emotions_lc = [
                    e.get("emotion","").lower() 
                    for e in defined_symbol_emotions_raw 
                    if isinstance(e, dict) and e.get("emotion")
                ]
            elif isinstance(first, tuple):
                defined_symbol_emotions_lc = [
                    e[0].lower() 
                    for e in defined_symbol_emotions_raw 
                    if isinstance(e, tuple) and len(e)>=1 and isinstance(e[0], str)
                ]
            elif isinstance(first, str):
                defined_symbol_emotions_lc = [e.lower() for e in defined_symbol_emotions_raw]
        # Else leave defined_symbol_emotions_lc empty

        for text_emo, text_emo_strength in current_text_emotions_map.items():
            if text_emo in defined_symbol_emotions_lc: # text_emo is already lowercased
                contextual_weight += text_emo_strength # Add strength of current text emotion
                influencing_emotions_for_this_match.append((text_emo, text_emo_strength)) # Store tuple

        # If no specific resonance, use a fraction of the overall emotional intensity of the text
        if not influencing_emotions_for_this_match and current_text_emotions_map:
             contextual_weight = (sum(current_text_emotions_map.values()) / len(current_text_emotions_map)) * 0.25 # 25% of avg intensity
        
        contextual_weight = min(1.0, max(0.0, contextual_weight))

        # 3. Final Weight: Combine base and contextual. Give more importance to direct contextual resonance.
        #    This heuristic can be tuned.
        final_symbol_weight = (base_weight * 0.3) + (contextual_weight * 0.7) # Contextual more important
        # Add a small boost if the symbol has a high resonance_weight defined
        final_symbol_weight += symbol_details.get("resonance_weight", 0.5) * 0.1 # 10% of its resonance weight

        final_symbol_weight = min(1.0, max(0.05, final_symbol_weight)) # Ensure it's within a range 0.05 - 1.0

        emotionally_weighted_symbols.append({
            "symbol": symbol_token,
            "name": symbol_name,
            "matched_keyword": matched_sym_info["matched_keyword"],
            "base_emotional_weight": round(base_weight, 3),
            "contextual_emotional_weight": round(contextual_weight, 3),
            "final_weight": round(final_symbol_weight, 3),
            "influencing_emotions": sorted(influencing_emotions_for_this_match, key=lambda x:x[1], reverse=True)
        })

    emotionally_weighted_symbols.sort(key=lambda x: x["final_weight"], reverse=True)
    return emotionally_weighted_symbols


if __name__ == '__main__':
    print("Testing parser.py with new helper functions and refined parse_with_emotion...")

    # Ensure spaCy model is loaded for tests that rely on it
    if not NLP_MODEL_LOADED and nlp is None : # Attempt to load if failed at module level
        try:
            nlp = spacy.load("en_core_web_sm")
            NLP_MODEL_LOADED = True
            print("   spaCy model re-loaded for tests.")
        except OSError:
             print("   spaCy model still not found for tests. Keyword extraction will be basic.")


    # Test load_seed_symbols
    print("\n--- Testing load_seed_symbols ---")
    dummy_seed_path = Path("data/test_dummy_seeds_parser.json")
    dummy_seed_data = {"🌟": {"name": "Star", "keywords": ["star", "shine"], "learning_phase": 0, "emotions": ["wonder", "hope"]}}
    with open(dummy_seed_path, "w", encoding="utf-8") as f: json.dump(dummy_seed_data, f)
    loaded_seeds = load_seed_symbols(dummy_seed_path)
    print(f"Loaded seeds: {loaded_seeds}")
    assert "🌟" in loaded_seeds
    if dummy_seed_path.exists(): dummy_seed_path.unlink()

    # Test extract_keywords
    print("\n--- Testing extract_keywords ---")
    test_text_kw = "This is a test sentence with several important Nouns, VERBS, and ProperNouns like London and interesting adjectives."
    keywords = extract_keywords(test_text_kw)
    print(f"Text: '{test_text_kw}'\nKeywords: {keywords}")
    expected_kws = ["test", "sentence", "noun", "verb", "propernoun", "london", "interesting", "adjective"] # Lemmas
    missing_kws = [ek for ek in expected_kws if ek not in keywords]
    if NLP_MODEL_LOADED:
         assert not missing_kws or len(missing_kws) < 3 # Allow some minor variation if spaCy used
    else:
         assert "nouns" in keywords or "important" in keywords # Fallback is different

    # Test chunk_content
    print("\n--- Testing chunk_content (refined) ---")
    long_text_chunk = "First sentence. Second sentence is a bit longer. Third sentence provides more detail. Fourth one. Fifth. Sixth makes it very long indeed. Seventh is the absolute charm. Eighth, ninth, and the tenth sentence will conclude this. A single sentence that is way too long for any normal chunk size, it just keeps going on and on and on without any punctuation to break it up naturally so it must be hard split several times over."
    chunks = chunk_content(long_text_chunk, max_chunk_size=70, overlap=15)
    print(f"Original length: {len(long_text_chunk)}, Chunks generated: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} (len {len(chunk)}): '{chunk[:80]}...'")
        assert len(chunk) <= 70 or (len(chunk)>70 and " " not in chunk[:70]) # Max size or hard split word
    assert len(chunks) > 5 # Expecting multiple chunks

    # Test parse_with_emotion (focus on data flow)
    print("\n--- Testing parse_with_emotion (refined) ---")
    dummy_active_lexicon = {
        "💡": {"name": "Idea", "keywords": ["idea", "thought", "innovation"], "emotions": ["curiosity", "excitement"], "resonance_weight": 0.6},
        "🔥": {"name": "Fire", "keywords": ["fire", "passion"], "emotions": ["passion", "anger"], "resonance_weight": 0.8}
    }
    # Ensure data/symbol_emotion_map.json is usable or create a dummy one for test
    dummy_emo_map_path_parser = Path("data/test_dummy_symbol_emotion_map_parser.json")
    dummy_emo_map_data = {"💡": {"curiosity": 0.7, "excitement": 0.5}, "🔥": {"passion": 0.8, "anger": 0.6}}
    with open(dummy_emo_map_path_parser, "w", encoding="utf-8") as f: json.dump(dummy_emo_map_data, f)
    
    original_parser_emo_map_path = EMOTION_MAP_PATH
    globals()['EMOTION_MAP_PATH'] = dummy_emo_map_path_parser # Override path for test

    sample_verified_emotions = [("curiosity", 0.9), ("passion", 0.7)] # This is what emotion_handler.predict_emotions()['verified'] would provide
    test_text_pwe = "A fiery new idea sparked with passion and curiosity."
    
    emotion_weighted_matches = parse_with_emotion(
        test_text_pwe,
        sample_verified_emotions,
        current_lexicon=dummy_active_lexicon
    )
    print(f"Input: '{test_text_pwe}' with emotions {sample_verified_emotions}")
    print("Emotion-weighted symbol matches:")
    found_idea = False
    found_fire = False
    for match in emotion_weighted_matches:
        print(f"  Symbol: {match['symbol']} ({match['name']}), Final W: {match['final_weight']:.3f}, BaseEW: {match['base_emotional_weight']:.3f}, CtxEW: {match['contextual_emotional_weight']:.3f}, Influencing: {match['influencing_emotions']}")
        if match['symbol'] == '💡': found_idea = True; assert match['final_weight'] > 0.3
        if match['symbol'] == '🔥': found_fire = True; assert match['final_weight'] > 0.3
        # Check if influencing emotions are correctly passed and are a subset of sample_verified_emotions
        if match['influencing_emotions']:
            for inf_emo, inf_score in match['influencing_emotions']:
                assert any(s_emo == inf_emo and s_score == inf_score for s_emo, s_score in sample_verified_emotions)

    assert found_idea and found_fire
    
    globals()['EMOTION_MAP_PATH'] = original_parser_emo_map_path # Restore
    if dummy_emo_map_path_parser.exists(): dummy_emo_map_path_parser.unlink()

    print("\n✅ parser.py refined tests completed.")