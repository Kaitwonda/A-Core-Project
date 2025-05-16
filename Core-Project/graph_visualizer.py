import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Load symbol definitions
with open(Path("seed_symbols.json"), "r", encoding="utf-8") as f:
    symbols = json.load(f)

# Load user memory (if it exists)
memory_path = Path("user_memory.json")
memory = []

if memory_path.exists():
    with open(memory_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        memory = data.get("entries", [])

# Count emotional intensity per symbol
emotion_counts = {}
for entry in memory:
    sym = entry["symbol"]
    emotion_counts[sym] = emotion_counts.get(sym, 0) + 1

# Build graph
G = nx.Graph()

for symbol, data in symbols.items():
    label = f"{symbol}\n{data['name']}"
    weight = emotion_counts.get(symbol, 1)
    G.add_node(symbol, label=label, weight=weight)

    for linked in data.get("linked_symbols", []):
        if linked in symbols:
            G.add_edge(symbol, linked)

# Layout + visual params
pos = nx.spring_layout(G, seed=42)
sizes = [G.nodes[n]["weight"] * 600 for n in G.nodes]
labels = nx.get_node_attributes(G, "label")

plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="skyblue", edgecolors="black")
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, labels, font_size=10)

plt.title("🧠 Symbolic Mind Graph (Emotion-Weighted)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
