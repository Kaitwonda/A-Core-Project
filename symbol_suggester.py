import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
from pathlib import Path

# 📂 Paths
VECTOR_STORE = Path("memory/vector_memory.json")
SYMBOL_GRAPH = Path("graph/symbol_graph.json")

# 📦 Load existing vectors
def load_vectors():
    if VECTOR_STORE.exists():
        with open(VECTOR_STORE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    return []

# 💾 Save updated symbol graph
def save_symbol(symbol_obj):
    if SYMBOL_GRAPH.exists():
        with open(SYMBOL_GRAPH, 'r', encoding='utf-8') as f:
            graph = json.load(f)
    else:
        graph = {}

    symbol_id = str(uuid4())
    graph[symbol_id] = symbol_obj

    with open(SYMBOL_GRAPH, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2)

# 🔍 Suggest symbols from clusters
def suggest_symbols_from_vectors(min_cluster_size=3):
    vectors_data = load_vectors()
    if not vectors_data:
        print("❌ No vectors found.")
        return

    embeddings = np.array([entry["vector"] for entry in vectors_data])
    db = DBSCAN(eps=0.3, min_samples=min_cluster_size, metric='cosine').fit(embeddings)

    labels = db.labels_
    unique_labels = set(labels)
    print(f"🔍 Found {len(unique_labels) - (1 if -1 in labels else 0)} potential clusters.")

    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        
        indices = np.where(labels == label)[0]
        if len(indices) < min_cluster_size:
            continue

        cluster_texts = [vectors_data[i]["text"] for i in indices]
        cluster_vectors = [vectors_data[i]["vector"] for i in indices]
        centroid = np.mean(cluster_vectors, axis=0).tolist()

        proposed_symbol = {
            "emoji": "🌀",  # Temporary placeholder
            "core_meanings": [],  # To be refined
            "based_on": cluster_texts[:3],
            "vector_centroid": centroid,
            "resonance_weight": round(len(indices) / len(vectors_data), 3),
            "origin": "auto_cluster",
            "timestamp": str(Path().stat().st_mtime)
        }

        print(f"✨ Proposed new symbol from cluster of {len(indices)}: 🌀")
        save_symbol(proposed_symbol)

# 🔁 Main trigger
if __name__ == "__main__":
    suggest_symbols_from_vectors()
