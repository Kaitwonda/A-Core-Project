# symbol_emotion_updater.py

import json
from pathlib import Path
from collections import defaultdict

MAP_PATH = Path("data/symbol_emotion_map.json")
MAP_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_emotion_map():
    if MAP_PATH.exists():
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_emotion_map(emotion_map):
    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(emotion_map, f, indent=2)

def update_symbol_emotions(symbols, emotions):
    """
    symbols: List of dicts with 'symbol' keys (from parser)
    emotions: List of tuples like [('fear', 0.76), ('anger', 0.54)] (from emotion_handler['verified'])
    """
    emotion_map = load_emotion_map()

    for symbol_entry in symbols:
        sym = symbol_entry["symbol"]
        if sym not in emotion_map:
            emotion_map[sym] = {}

        for emotion, strength in emotions:
            current = emotion_map[sym].get(emotion, 0)
            # Weighted additive update — small increases to prevent runaway growth
            updated = current + (strength * 0.25)
            emotion_map[sym][emotion] = round(updated, 3)

    save_emotion_map(emotion_map)
    print("🧬 Symbol-emotion map updated.")
