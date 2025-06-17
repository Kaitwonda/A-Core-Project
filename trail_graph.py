import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def show_trail_graph():
    trail_path = Path("data/trail_log.json")
    if not trail_path.exists():
        print("❌ No trail log found.")
        return

    with open(trail_path, "r", encoding="utf-8") as f:
        trail = json.load(f)

    G = nx.DiGraph()

    for i, entry in enumerate(trail):
        base_node = f"T{i}"
        G.add_node(base_node, label=entry["text"][:30] + "...")

        # Link to matched memories
        for j, match in enumerate(entry.get("matches", [])):
            match_node = f"{base_node}_m{j}"
            match_label = match["text"][:30] + "..."
            G.add_node(match_node, label=match_label)
            G.add_edge(base_node, match_node)

        # Link to symbols
        for symbol in entry.get("symbols", []):
            sym_node = f"{base_node}_{symbol}"
            G.add_node(sym_node, label=symbol)
            G.add_edge(base_node, sym_node)

    pos = nx.spring_layout(G, seed=42)
    labels = nx.get_node_attributes(G, 'label')

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=900)
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos, labels, font_size=9)
    plt.title("🧭 Thought Trail + Symbol Map")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Optional standalone execution
if __name__ == "__main__":
    show_trail_graph()
