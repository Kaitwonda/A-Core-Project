from transformers import AutoModelForSequenceClassification, AutoTokenizer

models = [
    "j-hartmann/emotion-english-distilroberta-base",  # replaces joeddav/go-emotions
    "bhadresh-savani/distilbert-base-uncased-emotion",
    "nateraw/bert-base-uncased-emotion"
]

for m in models:
    print(f"📦 Downloading {m}...")
    AutoModelForSequenceClassification.from_pretrained(m)
    AutoTokenizer.from_pretrained(m)

print("✅ All models downloaded and cached.")
