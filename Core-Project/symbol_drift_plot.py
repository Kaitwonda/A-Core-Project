# symbol_drift_plot.py

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

TRAIL_PATH = Path("data/trail_log.json")  # Make sure this path is correct or update accordingly

def show_symbol_drift(symbol_filter=None):
    if not TRAIL_PATH.exists():
        print("❌ No trail log found.")
        return

    with open(TRAIL_PATH, "r", encoding="utf-8") as f:
        trail = json.load(f)

    # history[symbol][emotion] = [(timestamp, weight), ...]
    history = defaultdict(lambda: defaultdict(list))

    for entry in trail:
        timestamp = entry.get("timestamp")
        if not timestamp:
            continue
        try:
            ts = datetime.fromisoformat(timestamp)
        except ValueError:
            continue  # Skip invalid timestamps

        for symbol in entry.get("symbols", []):
            sym_id = symbol.get("symbol")
            if symbol_filter and sym_id != symbol_filter:
                continue

            emotional_weight = symbol.get("emotional_weight", 0)

            for emotion_tuple in symbol.get("influencing_emotions", []):
                if isinstance(emotion_tuple, list) and len(emotion_tuple) >= 1:
                    emotion = emotion_tuple[0]
                    history[sym_id][emotion].append((ts, emotional_weight))

    if not history:
        print("⚠️ No matching symbols or emotions found.")
        return

    # Plotting each symbol's emotional weight over time
    for symbol, emotion_data in history.items():
        plt.figure(figsize=(9, 5))
        for emotion, points in emotion_data.items():
            points.sort()
            times, weights = zip(*points)
            plt.plot(times, weights, label=emotion)
        plt.title(f"Symbol Drift for {symbol}")
        plt.xlabel("Time")
        plt.ylabel("Emotional Weight")
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example usage
# show_symbol_drift("🦞")  # or call without a filter to see all
