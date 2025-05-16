# cluster_namer.py

import json

from pathlib import Path

from collections import Counter

import random



SYMBOL_PATH = Path("data/symbol_memory.json")

NAMED_CLUSTERS_PATH = Path("data/cluster_names.json")



EMOJI_POOL = ["🌀", "💔", "🛡️", "🌱", "🔥", "🌙", "🌊", "🧭", "⚡", "🪞"]



# Add this to fix the missing function issue

def generate_cluster_id(texts):

    return hash(" ".join(texts)) % 1000000



def pick_cluster_name(texts, keywords, emotions, cluster_id=None):

    if cluster_id is None:

        cluster_id = generate_cluster_id(texts)

    return generate_cluster_name(cluster_id, keywords, emotions)



def generate_cluster_name(cluster_id, keywords, emotions):

    if keywords:

        top = Counter(keywords).most_common(1)[0][0].title()

    elif emotions:

        top = max(emotions, key=lambda x: x[1])[0].title()

    else:

        top = "Theme"



    emoji = random.choice(EMOJI_POOL)

    return f"{emoji} {top}"



def load_symbols():

    if SYMBOL_PATH.exists():

        with open(SYMBOL_PATH, "r", encoding="utf-8") as f:

            return json.load(f)

    return {}



def extract_texts(symbols):

    return [

        f"{v['name']} {' '.join(v['keywords'])} {' '.join(v['emotions'].keys())}"

        for v in symbols.values()

    ]



def cluster_symbols(symbols, threshold=0.5):

    from sklearn.metrics.pairwise import cosine_similarity

    from sklearn.feature_extraction.text import TfidfVectorizer



    texts = extract_texts(symbols)

    vectorizer = TfidfVectorizer().fit_transform(texts)

    sim_matrix = cosine_similarity(vectorizer)



    clusters = []

    visited = set()



    for i in range(len(symbols)):

        if i in visited:

            continue

        cluster = [i]

        visited.add(i)

        for j in range(len(symbols)):

            if j not in visited and sim_matrix[i][j] > threshold:

                cluster.append(j)

                visited.add(j)

        clusters.append(cluster)



    return clusters



def summarize_cluster(symbols, indices):

    texts = [symbols[i] for i in indices]

    names = [s['name'] for s in texts]

    all_words = sum([s['keywords'] for s in texts], []) + sum([list(s['emotions'].keys()) for s in texts], [])

    most_common = Counter(all_words).most_common(3)

    summary = names[0] if names else "Theme"

    if most_common:

        summary = most_common[0][0].title()

    emoji = random.choice(EMOJI_POOL)

    return emoji, summary



def assign_cluster_names():

    raw = load_symbols()

    if not raw:

        print("❌ No symbols to cluster.")

        return



    symbol_list = list(raw.values())

    clusters = cluster_symbols(symbol_list)



    named_clusters = {}

    for i, group in enumerate(clusters):

        emoji, name = summarize_cluster(symbol_list, group)

        named_clusters[f"cluster_{i}"] = {

            "name": name,

            "emoji": emoji,

            "members": [symbol_list[idx]['symbol'] for idx in group]

        }



    with open(NAMED_CLUSTERS_PATH, "w", encoding="utf-8") as f:

        json.dump(named_clusters, f, indent=2)



    print(f"✅ Named {len(named_clusters)} clusters.")