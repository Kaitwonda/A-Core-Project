# parser.py - Modified to work with AlphaWall zone outputs
import spacy
import json
from pathlib import Path
import re

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

# Emoji pattern for detection across the system
EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map symbols
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251]+"
)

SEED_PATH = Path("data/seed_symbols.json")
EMOTION_MAP_PATH = Path("data/symbol_emotion_map.json")
EMOTION_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)

# NEW: Import AlphaWall for zone output handling
try:
    from alphawall import AlphaWall
    ALPHAWALL_AVAILABLE = True
except ImportError:
    ALPHAWALL_AVAILABLE = False
    print("[WARNING] AlphaWall not available. Parser will work in legacy mode.")


def load_seed_symbols(file_path=SEED_PATH):
    file_path_obj = Path(file_path)
    file_path_obj.parent.mkdir(parents=True, exist_ok=True)
    if file_path_obj.exists() and file_path_obj.stat().st_size > 0:
        with open(file_path_obj, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError:
                print(f"[PARSER-WARNING] Seed symbols file {file_path_obj} is corrupted. Using empty seeds.")
                return {}
    else:
        default_seeds = {
            "🔥": {"name": "Fire", "keywords": ["fire", "flame", "computation", "logic"], "core_meanings": ["heat"], "emotions": ["anger"], "archetypes": ["destroyer"], "learning_phase": 0, "resonance_weight": 0.7},
            "💧": {"name": "Water", "keywords": ["water", "liquid", "data", "flow"], "core_meanings": ["flow"], "emotions": ["calm"], "archetypes": ["healer"], "learning_phase": 0, "resonance_weight": 0.7},
            "💻": {"name": "Computer", "keywords": ["computer", "computation", "cpu", "binary", "code", "algorithm", "system", "architecture"], "core_meanings": ["processing", "logic unit"], "emotions": ["neutral", "focus"], "archetypes": ["tool", "oracle"], "learning_phase": 0, "resonance_weight": 0.8}
        }
        if not file_path_obj.exists() or file_path_obj.stat().st_size == 0:
            try:
                with open(file_path_obj, "w", encoding="utf-8") as f:
                    json.dump(default_seeds, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"   [PARSER-ERROR] Could not create default seed file: {e}")
        return {}


def load_emotion_map(file_path=EMOTION_MAP_PATH):
    if file_path.exists() and file_path.stat().st_size > 0:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"[PARSER-WARNING] Symbol emotion map {file_path} is corrupted. Using empty map.")
                return {}
    return {}


def extract_keywords(text_input, max_keywords=10):
    """Extract keywords from text - now works with both raw text and zone outputs"""
    if not text_input:
        return []
    
    # Handle zone output format
    if isinstance(text_input, dict) and 'tags' in text_input:
        # Extract keywords from zone metadata
        keywords = []
        
        # Add intent as keyword
        intent = text_input['tags'].get('intent', '')
        if intent:
            keywords.extend(intent.split('_'))
        
        # Add emotional state
        emotional_state = text_input['tags'].get('emotional_state', '')
        if emotional_state:
            keywords.extend(emotional_state.split('_'))
        
        # Add context types
        contexts = text_input['tags'].get('context', [])
        for ctx in contexts:
            keywords.extend(ctx.split('_'))
        
        # Add semantic profile indicators
        semantic_profile = text_input.get('semantic_profile', {})
        for key, score in semantic_profile.items():
            if score > 0.6:  # High similarity
                keywords.append(key.replace('similarity_to_', ''))
        
        # Clean and deduplicate
        keywords = [kw.lower() for kw in keywords if kw and len(kw) > 2]
        return list(dict.fromkeys(keywords))[:max_keywords]
    
    # Legacy mode - process raw text
    if isinstance(text_input, str):
        keywords = []
        if NLP_MODEL_LOADED and nlp:
            doc = nlp(text_input.lower()[:nlp.max_length])
            for token in doc:
                if not token.is_stop and not token.is_punct and not token.is_space:
                    if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]:
                        keywords.append(token.lemma_)
            keywords = list(dict.fromkeys(keywords))
        else:
            words = re.findall(r'\b\w{3,}\b', text_input.lower())
            stop_words = {"the", "a", "is", "in", "it", "to", "and", "of", "for", "on", "with", "as", "by", "an", "this", "that", "was", "were", "be"}
            keywords = [word for word in words if word not in stop_words]
            keywords = list(dict.fromkeys(keywords))
        return keywords[:max_keywords]
    
    return []


def chunk_content(text, max_chunk_size=1000, overlap=100):
    """Chunk content - only works with raw text, not zone outputs"""
    if not text or not isinstance(text, str):
        return []
    
    chunks = []
    if NLP_MODEL_LOADED and nlp:
        doc = nlp(text[:nlp.max_length])
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    current_chunk_parts = []
    current_length = 0
    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)
        if not sentence:
            continue

        if sentence_len > max_chunk_size:
            if current_chunk_parts:
                chunks.append(" ".join(current_chunk_parts))
                current_chunk_parts, current_length = [], 0
            
            sub_sentence_start = 0
            while sub_sentence_start < sentence_len:
                end_point = min(sub_sentence_start + max_chunk_size, sentence_len)
                actual_end_point = sentence.rfind(" ", sub_sentence_start, end_point) if " " in sentence[sub_sentence_start:end_point] else end_point
                if actual_end_point <= sub_sentence_start:
                    actual_end_point = end_point
                
                chunks.append(sentence[sub_sentence_start:actual_end_point].strip())
                sub_sentence_start = actual_end_point + (1 if sentence[actual_end_point:actual_end_point+1] == " " else 0)
            continue

        if current_length + sentence_len + (1 if current_chunk_parts else 0) > max_chunk_size and current_chunk_parts:
            chunks.append(" ".join(current_chunk_parts))
            
            overlap_parts = []
            overlap_len = 0
            temp_overlap_source = list(current_chunk_parts)
            while temp_overlap_source:
                part_to_add = temp_overlap_source.pop()
                part_len = len(part_to_add)
                if overlap_len + part_len + (1 if overlap_parts else 0) <= overlap:
                    overlap_parts.insert(0, part_to_add)
                    overlap_len += part_len + (1 if len(overlap_parts) > 1 else 0)
                else:
                    break
            
            current_chunk_parts = overlap_parts
            current_length = overlap_len
        
        if current_chunk_parts:
            current_chunk_parts.append(sentence)
        else:
            current_chunk_parts = [sentence]
        current_length += sentence_len + (1 if len(current_chunk_parts) > 1 else 0)

    if current_chunk_parts:
        chunks.append(" ".join(current_chunk_parts))
        
    return [c for c in chunks if c.strip()]


def extract_symbolic_units(input_data, current_lexicon):
    """Extract symbolic units - now works with zone outputs"""
    if not current_lexicon:
        return []
    
    extracted = []
    
    # Handle zone output
    if isinstance(input_data, dict) and 'tags' in input_data:
        # Map zone tags to potential symbols
        zone_to_symbol_hints = {
            'emotional_state': {
                'calm': ['💧', '🕊️', '☮️'],
                'overwhelmed': ['🌀', '🌊', '💥'],
                'grief': ['💧', '🌧️', '💔'],
                'angry': ['🔥', '⚡', '💥'],
                'emotionally_recursive': ['🌀', '♻️', '🔄']
            },
            'intent': {
                'information_request': ['💻', '🔍', '❓'],
                'expressive': ['❤️', '🎭', '✨'],
                'self_reference': ['🪞', '👤', '💭'],
                'abstract_reflection': ['🌌', '∞', '🧩'],
                'euphemistic': ['🌫️', '👻', '...'],
                'humor_deflection': ['😂', '🎭', '🛡️']
            },
            'context': {
                'trauma_loop': ['🌀', '🔄', '⚠️'],
                'reclaimed_language': ['💪', '✊', '🌈'],
                'metaphorical': ['🎭', '🌉', '🎨'],
                'coded_speech': ['🔐', '🗝️', '...'],
                'poetic_speech': ['📜', '✒️', '🎵'],
                'meme_reference': ['🐸', '💯', '📱']
            }
        }
        
        # Check emotional state
        emotional_state = input_data['tags'].get('emotional_state', '')
        if emotional_state in zone_to_symbol_hints['emotional_state']:
            for symbol in zone_to_symbol_hints['emotional_state'][emotional_state]:
                if symbol in current_lexicon:
                    details = current_lexicon[symbol]
                    extracted.append({
                        "symbol": symbol,
                        "name": details.get("name", "Unknown Symbol"),
                        "matched_keyword": f"zone:{emotional_state}"
                    })
        
        # Check intent
        intent = input_data['tags'].get('intent', '')
        if intent in zone_to_symbol_hints['intent']:
            for symbol in zone_to_symbol_hints['intent'][intent]:
                if symbol in current_lexicon and not any(e['symbol'] == symbol for e in extracted):
                    details = current_lexicon[symbol]
                    extracted.append({
                        "symbol": symbol,
                        "name": details.get("name", "Unknown Symbol"),
                        "matched_keyword": f"zone:{intent}"
                    })
        
        # Check contexts
        for context in input_data['tags'].get('context', []):
            if context in zone_to_symbol_hints['context']:
                for symbol in zone_to_symbol_hints['context'][context]:
                    if symbol in current_lexicon and not any(e['symbol'] == symbol for e in extracted):
                        details = current_lexicon[symbol]
                        extracted.append({
                            "symbol": symbol,
                            "name": details.get("name", "Unknown Symbol"),
                            "matched_keyword": f"zone:{context}"
                        })
        
        # Use semantic profile for additional hints
        semantic_profile = input_data.get('semantic_profile', {})
        semantic_to_symbols = {
            'technical': ['💻', '⚙️', '🔧'],
            'emotional': ['❤️', '💧', '🔥'],
            'philosophical': ['🌌', '🧩', '∞'],
            'practical': ['🔧', '📋', '✅']
        }
        
        for concept, symbols in semantic_to_symbols.items():
            similarity_key = f'similarity_to_{concept}'
            if semantic_profile.get(similarity_key, 0) > 0.6:
                for symbol in symbols:
                    if symbol in current_lexicon and not any(e['symbol'] == symbol for e in extracted):
                        details = current_lexicon[symbol]
                        extracted.append({
                            "symbol": symbol,
                            "name": details.get("name", "Unknown Symbol"),
                            "matched_keyword": f"semantic:{concept}"
                        })
        
        return extracted
    
    # Legacy mode - process raw text
    if isinstance(input_data, str):
        text_keywords = extract_keywords(input_data, max_keywords=25)
        text_lower_for_direct_match = input_data.lower()

        for token_symbol, details in current_lexicon.items():
            symbol_keywords = details.get("keywords", [])
            matched_kw = None
            symbol_token_lower = token_symbol.lower()

            if f" {symbol_token_lower} " in f" {text_lower_for_direct_match} " or \
               text_lower_for_direct_match.startswith(f"{symbol_token_lower} ") or \
               text_lower_for_direct_match.endswith(f" {symbol_token_lower}") or \
               text_lower_for_direct_match == symbol_token_lower:
                matched_kw = token_symbol
            else:
                for sk in symbol_keywords:
                    sk_lower = sk.lower()
                    if re.search(r'\b' + re.escape(sk_lower) + r'\b', text_lower_for_direct_match):
                        matched_kw = sk
                        break
                    if sk_lower in text_keywords:
                        matched_kw = sk
                        break
            
            if matched_kw:
                extracted.append({
                    "symbol": token_symbol,
                    "name": details.get("name", "Unknown Symbol"),
                    "matched_keyword": matched_kw
                })
    
    return extracted


def parse_input(input_data, current_lexicon=None):
    """
    Main parsing function - now accepts both raw text and zone outputs.
    
    Args:
        input_data: Either a string (raw text) or dict (zone output)
        current_lexicon: Symbol lexicon to use
    
    Returns:
        dict: Parsed keywords and symbols
    """
    if current_lexicon is None:
        current_lexicon = load_seed_symbols()
    
    # Extract keywords and symbols based on input type
    keywords = extract_keywords(input_data)
    symbols = extract_symbolic_units(input_data, current_lexicon)
    
    # Add zone metadata if available
    result = {
        "keywords": keywords,
        "symbols": symbols
    }
    
    if isinstance(input_data, dict) and 'tags' in input_data:
        result['zone_metadata'] = {
            'zone_id': input_data.get('zone_id'),
            'routing_hint': input_data.get('routing_hints', {}).get('suggested_node'),
            'quarantine_recommended': input_data.get('routing_hints', {}).get('quarantine_recommended', False),
            'emotional_state': input_data['tags'].get('emotional_state'),
            'intent': input_data['tags'].get('intent'),
            'risk_flags': input_data['tags'].get('risk', [])
        }
    
    return result


def parse_with_emotion(input_data, detected_emotions_verified, current_lexicon):
    """
    Enhanced emotion-aware parsing - now works with zone outputs.
    
    For zone outputs, uses the emotional metadata to enhance symbol matching.
    """
    if not current_lexicon:
        print("[PARSER-ERROR] parse_with_emotion requires a current_lexicon.")
        return []
    if not isinstance(detected_emotions_verified, list):
        print(f"[PARSER-WARNING] detected_emotions_verified is not a list: {detected_emotions_verified}. Using empty emotions.")
        detected_emotions_verified = []

    symbol_emotion_profiles_map = load_emotion_map()
    
    # Get matched symbols based on input type
    matched_symbols_in_text = extract_symbolic_units(input_data, current_lexicon)
    
    # If zone output, enhance emotion detection
    if isinstance(input_data, dict) and 'tags' in input_data:
        # Add zone's emotional state to detected emotions
        emotional_state = input_data['tags'].get('emotional_state', '')
        emotion_confidence = input_data['tags'].get('emotion_confidence', 0.5)
        
        # Map zone emotional states to emotion labels
        zone_emotion_map = {
            'calm': [('calm', emotion_confidence), ('peace', emotion_confidence * 0.8)],
            'overwhelmed': [('anxiety', emotion_confidence), ('fear', emotion_confidence * 0.7)],
            'grief': [('sadness', emotion_confidence), ('loss', emotion_confidence * 0.8)],
            'angry': [('anger', emotion_confidence), ('frustration', emotion_confidence * 0.7)],
            'emotionally_recursive': [('confusion', emotion_confidence), ('repetition', emotion_confidence)]
        }
        
        if emotional_state in zone_emotion_map:
            # Merge zone emotions with detected emotions
            zone_emotions = zone_emotion_map[emotional_state]
            for zone_emo, zone_score in zone_emotions:
                # Only add if not already in detected emotions
                if not any(emo == zone_emo for emo, _ in detected_emotions_verified):
                    detected_emotions_verified.append((zone_emo, zone_score))
    
    emotionally_weighted_symbols = []
    current_text_emotions_map = {
        emo_label.lower(): score 
        for emo_label, score in detected_emotions_verified 
        if emo_label and isinstance(score, (float, int))
    }

    for matched_sym_info in matched_symbols_in_text:
        symbol_token = matched_sym_info["symbol"]
        symbol_details = current_lexicon.get(symbol_token, {})
        symbol_name = symbol_details.get("name", symbol_token)

        # Base Emotional Weight
        base_weight = 0.3
        symbol_general_profile = symbol_emotion_profiles_map.get(symbol_token, {})
        
        if symbol_general_profile:
            alignment_score = 0
            common_emotions_count = 0
            for text_emo, text_emo_strength in current_text_emotions_map.items():
                if text_emo in symbol_general_profile:
                    alignment_score += symbol_general_profile[text_emo] * text_emo_strength
                    common_emotions_count += 1
            if common_emotions_count > 0:
                base_weight = alignment_score / common_emotions_count
            elif not symbol_general_profile and not current_text_emotions_map:
                base_weight = 0.5
        
        base_weight = min(1.0, max(0.05, base_weight))

        # Contextual Emotional Weight
        contextual_weight = 0.0
        influencing_emotions_for_this_match = []
        
        defined_symbol_emotions_raw = symbol_details.get("emotions", [])
        defined_symbol_emotions_lc = []
        
        if isinstance(defined_symbol_emotions_raw, dict):
            defined_symbol_emotions_lc = [emo.lower() for emo in defined_symbol_emotions_raw.keys()]
        elif isinstance(defined_symbol_emotions_raw, list) and defined_symbol_emotions_raw:
            first = defined_symbol_emotions_raw[0]
            if isinstance(first, dict):
                defined_symbol_emotions_lc = [
                    e.get("emotion", "").lower() 
                    for e in defined_symbol_emotions_raw 
                    if isinstance(e, dict) and e.get("emotion")
                ]
            elif isinstance(first, tuple):
                defined_symbol_emotions_lc = [
                    e[0].lower() 
                    for e in defined_symbol_emotions_raw 
                    if isinstance(e, tuple) and len(e) >= 1 and isinstance(e[0], str)
                ]
            elif isinstance(first, str):
                defined_symbol_emotions_lc = [e.lower() for e in defined_symbol_emotions_raw]

        for text_emo, text_emo_strength in current_text_emotions_map.items():
            if text_emo in defined_symbol_emotions_lc:
                contextual_weight += text_emo_strength
                influencing_emotions_for_this_match.append((text_emo, text_emo_strength))

        if not influencing_emotions_for_this_match and current_text_emotions_map:
            contextual_weight = (sum(current_text_emotions_map.values()) / len(current_text_emotions_map)) * 0.25
        
        contextual_weight = min(1.0, max(0.0, contextual_weight))

        # Final Weight with zone boost
        final_symbol_weight = (base_weight * 0.3) + (contextual_weight * 0.7)
        
        # If this is a zone-matched symbol, give it a boost
        if matched_sym_info['matched_keyword'].startswith('zone:'):
            final_symbol_weight += 0.1
        elif matched_sym_info['matched_keyword'].startswith('semantic:'):
            final_symbol_weight += 0.05
        
        final_symbol_weight += symbol_details.get("resonance_weight", 0.5) * 0.1
        final_symbol_weight = min(1.0, max(0.05, final_symbol_weight))

        emotionally_weighted_symbols.append({
            "symbol": symbol_token,
            "name": symbol_name,
            "matched_keyword": matched_sym_info["matched_keyword"],
            "base_emotional_weight": round(base_weight, 3),
            "contextual_emotional_weight": round(contextual_weight, 3),
            "final_weight": round(final_symbol_weight, 3),
            "influencing_emotions": sorted(influencing_emotions_for_this_match, key=lambda x: x[1], reverse=True)
        })

    emotionally_weighted_symbols.sort(key=lambda x: x["final_weight"], reverse=True)
    return emotionally_weighted_symbols


# NEW: Helper function to check if input is zone output
def is_zone_output(input_data):
    """Check if the input is a zone output from AlphaWall"""
    return isinstance(input_data, dict) and 'tags' in input_data and 'zone_id' in input_data


# NEW: Backward compatibility wrapper
def parse_raw_text(text_input, current_lexicon=None):
    """
    Legacy function for parsing raw text directly.
    In production, this should go through AlphaWall first.
    """
    if ALPHAWALL_AVAILABLE:
        print("[WARNING] parse_raw_text called but AlphaWall is available. Consider using AlphaWall for safety.")
    
    return parse_input(text_input, current_lexicon)


if __name__ == '__main__':
    print("Testing parser.py with AlphaWall integration...")

    # Ensure spaCy model is loaded for tests
    if not NLP_MODEL_LOADED and nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
            NLP_MODEL_LOADED = True
            print("   spaCy model re-loaded for tests.")
        except OSError:
            print("   spaCy model still not found for tests. Keyword extraction will be basic.")

    # Test 1: Zone output parsing
    print("\n--- Testing zone output parsing ---")
    mock_zone_output = {
        'zone_id': 'test123',
        'timestamp': '2025-06-02T19:00:00Z',
        'memory_trace': 'mem456',
        'tags': {
            'emotional_state': 'overwhelmed',
            'emotion_confidence': 0.8,
            'intent': 'expressive',
            'context': ['trauma_loop', 'metaphorical'],
            'risk': ['bridge_conflict_expected']
        },
        'semantic_profile': {
            'similarity_to_technical': 0.2,
            'similarity_to_emotional': 0.8,
            'similarity_to_philosophical': 0.6
        },
        'routing_hints': {
            'suggested_node': 'symbolic_primary',
            'confidence_level': 'moderate',
            'quarantine_recommended': False
        }
    }
    
    # Test keyword extraction from zone output
    zone_keywords = extract_keywords(mock_zone_output)
    print(f"Zone keywords: {zone_keywords}")
    assert 'overwhelmed' in zone_keywords
    assert 'expressive' in zone_keywords
    assert 'emotional' in zone_keywords  # From high semantic similarity
    
    # Test symbolic extraction from zone output
    test_lexicon = load_seed_symbols()
    test_lexicon.update({
        '🌀': {'name': 'Spiral', 'keywords': ['spiral', 'recursion'], 'emotions': ['confusion']},
        '❤️': {'name': 'Heart', 'keywords': ['heart', 'love'], 'emotions': ['love', 'care']}
    })
    
    zone_symbols = extract_symbolic_units(mock_zone_output, test_lexicon)
    print(f"Zone symbols: {zone_symbols}")
    assert any(s['symbol'] == '🌀' for s in zone_symbols)  # Should match trauma_loop
    
    # Test parse_input with zone output
    parsed_zone = parse_input(mock_zone_output, test_lexicon)
    print(f"Parsed zone output: {json.dumps(parsed_zone, indent=2)}")
    assert 'zone_metadata' in parsed_zone
    assert parsed_zone['zone_metadata']['routing_hint'] == 'symbolic_primary'
    
    # Test 2: Legacy text parsing still works
    print("\n--- Testing legacy text parsing ---")
    test_text = "Fire and water represent the eternal balance of computation."
    parsed_text = parse_input(test_text, test_lexicon)
    print(f"Parsed text: {parsed_text}")
    assert 'fire' in parsed_text['keywords'] or 'water' in parsed_text['keywords']
    assert any(s['symbol'] in ['🔥', '💧'] for s in parsed_text['symbols'])
    
    # Test 3: Emotion parsing with zone output
    print("\n--- Testing emotion parsing with zone output ---")
    test_emotions = [('anxiety', 0.7), ('confusion', 0.6)]
    
    emotion_weighted_zone = parse_with_emotion(mock_zone_output, test_emotions, test_lexicon)
    print("Emotion-weighted symbols from zone:")
    for sym in emotion_weighted_zone[:3]:
        print(f"  {sym['symbol']} ({sym['name']}): {sym['final_weight']:.3f}")
    
    # Test 4: Check zone detection
    print("\n--- Testing zone detection ---")
    assert is_zone_output(mock_zone_output) == True
    assert is_zone_output("regular text") == False
    assert is_zone_output({'some': 'dict'}) == False
    
    print("\n✅ All parser tests with AlphaWall integration passed!")

__all__ = ['EMOJI_PATTERN']