import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Load memory
memory_file = Path("data/vector_memory.json")


def load_memory():
    if not memory_file.exists():
        return []
    with open(memory_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_symbol_chains(min_similarity=0.4):
    memory = load_memory()
    symbol_map = {}

    for entry in memory:
        symbol = entry.get("symbol")
        if not symbol:
            continue

        if symbol not in symbol_map:
            symbol_map[symbol] = []

        symbol_map[symbol].append(entry)

    chains = {}

    for symbol, entries in symbol_map.items():
        chains[symbol] = []
        vectors = [np.array(e["vector"]).reshape(1, -1) for e in entries]

        for i, vec in enumerate(vectors):
            for j, other_vec in enumerate(vectors):
                if i != j:
                    sim = cosine_similarity(vec, other_vec)[0][0]
                    if sim >= min_similarity:
                        text = entries[j]["text"]
                        chains[symbol].append((sim, text))

        chains[symbol].sort(reverse=True, key=lambda x: x[0])

    return chains


def print_symbol_chains():
    chains = build_symbol_chains()
    for symbol, connections in chains.items():
        print(f"\n🔗 Symbol Chain for {symbol}:")
        for sim, text in connections[:3]:
            print(f"  → ({sim:.2f}) {text[:80]}...")


if __name__ == "__main__":
    print_symbol_chains()
