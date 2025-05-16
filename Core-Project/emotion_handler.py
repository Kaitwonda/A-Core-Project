# emotion_handler.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

print("Loading emotion models...")

# Updated model: nuanced emotion model
hartmann_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
hartmann_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Fast general tone model
distil_tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
distil_model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")

# Slightly deeper verification
bert_tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
bert_model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")

# Emotion labels (static)
hartmann_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]

distil_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
bert_labels = distil_labels

def predict_emotions(text):
    emotions = {}

    # --- Hartmann (fine-grained multi-label) ---
    inputs = hartmann_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = hartmann_model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze()
    top_hartmann = [(hartmann_labels[i], float(probs[i])) for i in range(len(probs)) if probs[i] > 0.3]
    emotions['hartmann_emotions'] = sorted(top_hartmann, key=lambda x: x[1], reverse=True)

    # --- DistilBERT (fast tone) ---
    inputs = distil_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = distil_model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    top_distil = (distil_labels[torch.argmax(probs)], float(torch.max(probs)))
    emotions['distil_emotion'] = top_distil

    # --- BERT (deep check) ---
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze()
    top_bert = [(bert_labels[i], float(probs[i])) for i in range(len(probs)) if probs[i] > 0.3]
    emotions['bert_emotions'] = sorted(top_bert, key=lambda x: x[1], reverse=True)

    # --- Verified Emotion Merge ---
    verified_emotions = []
    for emo, score in emotions['hartmann_emotions']:
        if emo in distil_labels and emo == emotions['distil_emotion'][0]:
            verified_emotions.append((emo, score + 0.1))  # Boost if overlap
        else:
            verified_emotions.append((emo, score))

    verified_emotions = sorted(verified_emotions, key=lambda x: x[1], reverse=True)[:5]
    emotions['verified'] = verified_emotions

    return emotions
