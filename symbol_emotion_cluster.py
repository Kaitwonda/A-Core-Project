import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
import numpy as np

MAP_PATH = Path("data/symbol_emotion_map.json")

def show_emotion_clusters():
    if not MAP_PATH.exists():
        print("❌ No symbol emotion map found.")
        return

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        emo_map = json.load(f)

    all_emotions = sorted({e for symbol in emo_map.values() for e in symbol})
    matrix = []
    labels = []

    for symbol, emo_weights in emo_map.items():
        vector = [emo_weights.get(e, 0) for e in all_emotions]
        matrix.append(vector)
        labels.append(symbol)

    if len(matrix) < 2:
        print("📉 Not enough symbols for clustering.")
        return

    reduced = TSNE(n_components=2, perplexity=5).fit_transform(np.array(matrix))

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, label, fontsize=9)
    plt.title("Symbol Clusters by Emotional Signature")
    plt.tight_layout()
    plt.show()
