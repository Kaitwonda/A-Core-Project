# symbol_generator.py
import json
import hashlib 
from pathlib import Path 
import random
import string

# Expanded pool to include more than just emojis
# You can curate this list extensively
SYMBOL_TOKEN_POOL = [ 
    # Emojis (as before, ensure a good variety)
    "🌀", "💡", "🧩", "🔗", "🌐", "⚖️", "🗝️", "🌱", "⚙️", "🧭",
    "📜", "🧱", "💬", "👁️‍🗨️", "🧠", "🎨", "🎼", "🎭", "🌌", "⏳",
    "🌠", "✨", "❓", "❗", "♾️", "🕳️", "💠", "💎", "🧬", "🔭",
    "🔬", "🕊️", "🪞", "🛡️", "🕰️", "🌍", "💭", "👁️", "👂",
    "👣", "🌳", "🌲", "🌿", "🍄", "🌊", "💧", "🌬️", "💨", "🌪️",
    "🌋", "⛰️", "☀️", "🌙", "🌕", "🌑", "🪐", "📚", "🎲", "🔮", 
    "⚗️", "🕮", "🕯️", "⌛", "⚖", "⚓", "⚛️", "⚜️", "⚙", "♾",
    # Greek Letters (Commonly used ones)
    "Δ", "Φ", "Ψ", "Ω", "Σ", "Π", "Λ", "Θ", "Ξ", "α", "β", "γ", "δ", "ε", "φ", "ψ", "ω",
    # Geometric Shapes & Basic Symbols
    "○", "●", "□", "■", "△", "▲", "▽", "▼", "◇", "♦", "→", "←", "↔", "⇒", "⇔", "∴", "∵",
    "+", "-", "*", "/", "=", "<", ">", "≠", "≈", "≡", "∑", "∫", "√",
    # Other interesting Unicode symbols (Alchemical, Astrological, etc. - use with care for meaning)
    "☉", "☽", "☿", "♀", "♂", "♃", "♄", "♅", "♆", "♇", # Planets
    "♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓", # Zodiac
    "🜁", "🜂", "🜃", "🜄", # Alchemical elements (Air, Fire, Water, Earth)
    # Short Textual Sigils (examples - can be generated more dynamically later)
    # "[Core]", "[Flow]", "[Nexus]", "[Void]", "[Path]" 
    # For now, the generator picks one character. Generating multi-char sigils needs more logic.
]


def generate_symbol_from_context(text, keywords, emotions_list_of_tuples):
    """
    Constructs a new symbol dictionary based on context. Does NOT save it.
    Args:
        text (str): The context text.
        keywords (list): List of keywords from the text.
        emotions_list_of_tuples (list): List of (emotion_str, score_float) from emotion_handler.
    Returns:
        dict: A dictionary representing the new symbol, or None if no basis for generation.
    """
    if not keywords and not emotions_list_of_tuples:
        return None

    name_parts = []
    if keywords:
        name_parts.extend([kw.title() for kw in keywords[:2]]) # Use up to 2 top keywords
    
    if emotions_list_of_tuples:
        top_emotion_str = emotions_list_of_tuples[0][0].title()
        if not name_parts: # If no keywords, use top emotion for name
            name_parts.append(top_emotion_str)
        elif len(name_parts) == 1: # If one keyword, append emotion
             name_parts.append(f"({top_emotion_str})")
        # If two keywords, name might be long enough, or consider adding emotion too.
        # else: name_parts.append(f"[{top_emotion_str}]")


    if not name_parts: return None
    name = " ".join(name_parts)
    if not name: name = "Unnamed Concept"


    # Select a symbol token from the expanded pool
    # TODO: Could add logic to check if token is already in use in the active lexicon
    #       to ensure uniqueness if that's desired for newly generated symbols.
    #       This would require passing the active_lexicon to this function.
    symbol_token = random.choice(list(set(SYMBOL_TOKEN_POOL))) 
    
    creation_context_emotions = {emo: score for emo, score in emotions_list_of_tuples[:3]}

    new_symbol_entry = {
        "symbol": symbol_token, 
        "name": name,
        "keywords": list(set(keywords[:5])), 
        "core_meanings": [kw.lower() for kw in keywords[:2]], 
        "emotions": creation_context_emotions, 
        "emotion_profile": {}, 
        "origin": "emergent",
        "resonance_weight": 0.5, 
    }
    
    # print(f"✨ Symbol Generator proposed new emergent symbol: {new_symbol_entry['symbol']} - {new_symbol_entry['name']}")
    return new_symbol_entry

if __name__ == '__main__':
    print("Testing symbol_generator.py (refactored with diverse pool)...")
    
    # Mock P_Parser.extract_keywords for standalone testing if parser.py is not in PYTHONPATH
    # or if you want to avoid its spaCy dependency for this specific test.
    def mock_extract_keywords(text_input):
        return [w.lower() for w in text_input.split() if len(w) > 3]

    sample_text_1 = "A new computational paradigm emerged, causing both excitement and confusion and wonder."
    sample_keywords_1 = mock_extract_keywords(sample_text_1) 
    sample_emotions_1 = [("excitement", 0.8), ("confusion", 0.7), ("wonder", 0.65), ("curiosity", 0.5)]

    for _ in range(3): # Generate a few to see different tokens
        generated_symbol1 = generate_symbol_from_context(sample_text_1, sample_keywords_1, sample_emotions_1)
        if generated_symbol1:
            print("\nGenerated Symbol:")
            print(json.dumps(generated_symbol1, indent=2))
        else:
            print("Symbol not generated.")
