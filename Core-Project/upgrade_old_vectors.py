import json
from pathlib import Path
from vector_engine import encode_with_minilm, encode_with_e5
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

memory_file = Path("data/vector_memory.json")

def upgrade_vectors():
    if not memory_file.exists():
        print("❌ No memory file found.")
        return

    with open(memory_file, "r", encoding="utf-8") as f:
        memory = json.load(f)

    upgraded = []
    for entry in memory:
        text = entry["text"]
        vec_minilm = encode_with_minilm(text)
        vec_e5 = encode_with_e5(text)

        similarity = cosine_similarity([vec_minilm], [vec_e5])[0][0]
        fused = (vec_minilm + vec_e5) / 2 if similarity >= 0.7 else vec_minilm
        source = "fused" if similarity >= 0.7 else "minilm-dominant"

        entry["vector"] = fused.tolist()
        entry["similarity"] = float(similarity)
        entry["source"] = source
        upgraded.append(entry)

    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(upgraded, f, indent=2)

    print(f"✅ Upgraded {len(upgraded)} vector entries with fused data.")

if __name__ == "__main__":
    upgrade_vectors()
