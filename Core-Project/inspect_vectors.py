import json
from pathlib import Path

memory_file = Path("data/vector_memory.json")

def inspect_memory(limit=5):
    if not memory_file.exists():
        print("❌ No vector memory file found.")
        return

    with open(memory_file, "r", encoding="utf-8") as f:
        memory = json.load(f)

    print(f"🧠 Loaded {len(memory)} vector entries.\n")

    # Show the most recent entries (last n)
    for entry in memory[-limit:]:
        print(f"📝 Text: {entry['text']}")
        print(f"🔹 Source: {entry.get('source', 'unknown')}")

        similarity = entry.get("similarity", None)
        if isinstance(similarity, (float, int)):
            print(f"📈 Similarity (MiniLM vs E5): {similarity:.2f}")
        else:
            print("📈 Similarity (MiniLM vs E5): n/a")

        print(f"🔑 ID: {entry['id'][:10]}...\n")

if __name__ == "__main__":
    inspect_memory()
