import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from cluster_namer import pick_cluster_name

VECTOR_PATH = Path("data/vector_memory.json")

def cluster_vectors_and_plot(show_graph=True):
    if not VECTOR_PATH.exists():
        print("❌ No vector memory found.")
        return

    with open(VECTOR_PATH, "r", encoding="utf-8") as f:
        memory = json.load(f)

    # ✅ Handle both dict and list memory formats
    if isinstance(memory, list):
        entries = memory
    elif isinstance(memory, dict):
        entries = list(memory.values())
    else:
        print("❌ Unrecognized vector memory format.")
        return

    texts = [entry["text"] for entry in entries if "text" in entry]
    if len(texts) < 3:
        print("🪞 Not enough entries for clustering.")
        return

    vectorizer = TfidfVectorizer(max_features=300)
    X = vectorizer.fit_transform(texts).toarray()

    n_clusters = min(5, len(texts) // 2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced = tsne.fit_transform(X)

    # Analyze cluster topics
    label_map = defaultdict(list)
    for idx, lbl in enumerate(labels):
        label_map[lbl].append(texts[idx])

    def dummy_generate_cluster_id(texts):
        return hash(" ".join(texts)) % 1000000

    # Plot
    if show_graph:
        plt.figure(figsize=(10, 7))
        for lbl, points in label_map.items():
            pts = reduced[[i for i, l in enumerate(labels) if l == lbl]]

            # Collect sample keywords
            cluster_keywords = []
            for txt in points:
                cluster_keywords.extend(txt.lower().split())

            # Pick name and emoji
            label = pick_cluster_name(cluster_keywords, [], [], cluster_id=dummy_generate_cluster_id(points))
            xs, ys = zip(*pts)
            plt.scatter(xs, ys, label=label)

        plt.legend()
        plt.title("Clustered Memories")
        plt.tight_layout()
        plt.show()
