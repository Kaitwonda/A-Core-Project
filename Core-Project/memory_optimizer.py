import sys
import re
from parser import parse_input, extract_symbolic_units, parse_with_emotion
from web_parser import process_web_url
from vector_memory import store_vector, retrieve_similar_vectors
from symbol_cluster import cluster_vectors_and_plot
from trail_log import log_trail, add_emotions
from trail_graph import show_trail_graph
from emotion_handler import predict_emotions
from symbol_emotion_updater import update_symbol_emotions
from symbol_memory import add_symbol, prune_duplicates
from symbol_generator import generate_symbol_from_context
from symbol_drift_plot import show_symbol_drift
from symbol_emotion_cluster import show_emotion_clusters

# Track input count
input_counter = 0
INPUT_THRESHOLD = 10  # Run maintenance every 10 inputs

def is_url(text):
    return re.match(r"https?://", text.strip()) is not None

def generate_response(user_input, extracted_symbols):
    similar = retrieve_similar_vectors(user_input)
    if not similar:
        return "I'm still learning. Nothing comes to mind yet."

    trust_order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    similar.sort(key=lambda x: (trust_order.get(x[1].get("source_trust", "unknown"), 3), -x[0]))

    if any(entry[1].get("source_trust") in ("high", "medium") for entry in similar):
        similar = [entry for entry in similar if entry[1].get("source_trust", "unknown") in ("high", "medium")]

    response = "🧠 Here's what I remember:\n"
    for sim, memory in similar:
        txt = memory["text"]
        trust = memory.get("source_trust", "unknown")
        source_note = f" (source: {memory['source_url']})" if memory.get("source_url") else ""

        if trust == "high":
            trust_note = " — from a trusted source"
        elif trust == "medium":
            trust_note = " — moderately trusted"
        elif trust == "low":
            trust_note = " — caution: low-trust source"
        else:
            trust_note = " — source unknown"

        response += f" - {txt[:100]}...{source_note}{trust_note} (sim={sim:.2f})\n"

    if extracted_symbols:
        response += "\n🔗 Symbolic cues detected:"
        for sym in extracted_symbols:
            response += f"\n → {sym['symbol']} ({sym['name']})"

    return response

def main():
    global input_counter

    print("🧠 Hybrid AI: Symbolic + Vector Memory")
    print("Type a thought or paste a URL (type 'exit' to quit).\n")

    while True:
        user_input = input("💬 You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        if is_url(user_input):
            print("🌐 Detected URL. Parsing web content...")
            process_web_url(user_input)
        else:
            print("📝 Detected input. Parsing and storing...")
            parse_input(user_input)
            store_vector(user_input)

            emotions = predict_emotions(user_input)
            print("\n💓 Emotions detected:")
            for tag, score in emotions["verified"]:
                print(f"   → {tag} ({score:.2f})")

            add_emotions(user_input, emotions)
            symbols = parse_with_emotion(user_input, emotions["verified"])
            update_symbol_emotions(symbols, emotions["verified"])

            for s in symbols:
                s["influencing_emotions"] = emotions["verified"]

            for sym in symbols:
                add_symbol(
                    symbol=sym["symbol"],
                    name=sym["name"],
                    keywords=[sym["matched_keyword"]],
                    emotions=dict(emotions["verified"]),
                    example_text=user_input,
                    origin="emergent"
                )

            if symbols:
                print("\n✨ Extracted symbols:")
                for s in symbols:
                    print(f"   → {s['symbol']} ({s['name']}) [matched: {s['matched_keyword']}]")
            else:
                print("🌀 No symbolic units extracted.")

            if not symbols:
                keywords = [k for k in extract_symbolic_units(user_input)]
                new_sym = generate_symbol_from_context(user_input, keywords, emotions["verified"])
                if new_sym:
                    print(f"✨ Created new emergent symbol: {new_sym}")

            matches = retrieve_similar_vectors(user_input)
            log_trail(user_input, symbols, matches)

            response = generate_response(user_input, symbols)
            print("\n🗣️ Response:")
            print(response)

            # ✅ Check threshold for memory pruning
            input_counter += 1
            if input_counter % INPUT_THRESHOLD == 0:
                print("🧹 Auto-optimizing memory and symbols...")
                prune_duplicates()

    # On exit: run full diagnostics
    print("\n🔍 Checking for emergent patterns...")
    cluster_vectors_and_plot(show_graph=True)

    print("🧭 Visualizing trail of connections...")
    show_trail_graph()

    print("📈 Showing symbol drift over time...")
    show_symbol_drift()

    print("🎨 Visualizing emotional similarity between symbols...")
    show_emotion_clusters()

    print("🧹 Final memory optimization...")
    prune_duplicates()

if __name__ == "__main__":
    main()
