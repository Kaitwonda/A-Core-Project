from sklearn.cluster import KMeans
import numpy as np

def cluster_memory(memory_data, n_clusters=2):
    vectors = [item["vector"] for item in memory_data]
    texts = [item["text"] for item in memory_data]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(texts[i])

    return clusters
