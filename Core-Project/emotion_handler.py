# emotion_handler.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn.functional as F
from collections import defaultdict # For verified emotion merging

# --- Model Loading ---
# (Assuming models are loaded as per your existing script)
# For brevity, I'll skip reloading them here, but in your actual file they are present.
# Make sure download_models.py has been run and models are accessible.

MODELS_LOADED_EMOTION = False
try:
    hartmann_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    hartmann_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    hartmann_labels = hartmann_model.config.id2label # Dynamically get labels

    distil_tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    distil_model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    distil_labels = list(distil_model.config.id2label.values()) # This model has labels like 'sadness', 'joy', etc.

    bert_tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
    bert_model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
    # BERT model specific labels (from its model card or config)
    bert_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 
                   'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 
                   'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 
                   'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 
                   'remorse', 'sadness', 'surprise', 'neutral'] # Ensure these match model output

    MODELS_LOADED_EMOTION = True
    print("✅ Emotion models loaded successfully for emotion_handler.py.")

except Exception as e:
    print(f"⚠️ Error loading emotion models in emotion_handler.py: {e}")
    print("   Emotion detection will be severely limited or non-functional.")
    # Set models to None so predict_emotions can handle it
    hartmann_tokenizer, hartmann_model, hartmann_labels = None, None, []
    distil_tokenizer, distil_model, distil_labels = None, None, []
    bert_tokenizer, bert_model, bert_labels = None, None, []


def predict_emotions(text: str, top_n_verified=5):
    """
    Predicts emotions using multiple models and provides a merged 'verified' list.
    Args:
        text (str): The input text.
        top_n_verified (int): Number of top verified emotions to return.
    Returns:
        dict: Dictionary containing raw outputs from each model and a 'verified' list
              of (emotion_label, score) tuples, sorted by score.
              Returns empty lists/defaults if models are not loaded or text is empty.
    """
    emotions_summary = {
        'hartmann_emotions': [],
        'distil_emotion': (None, 0.0),
        'bert_emotions': [],
        'verified': []
    }

    if not MODELS_LOADED_EMOTION or not text or not text.strip():
        print("[EMOTION_HANDLER] Models not loaded or text is empty. Returning empty emotion summary.")
        return emotions_summary

    # --- Hartmann (j-hartmann/emotion-english-distilroberta-base) ---
    if hartmann_model and hartmann_tokenizer:
        inputs = hartmann_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = hartmann_model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze() # Sigmoid for multi-label
        
        top_hartmann = []
        for i in range(len(probs)):
            label_id = i # In new HF, logits/probs are in order of config.id2label
            label_name = hartmann_labels.get(label_id, f"unknown_label_{label_id}")
            score = float(probs[i])
            if score > 0.1: # Threshold for including an emotion
                top_hartmann.append((label_name, score))
        emotions_summary['hartmann_emotions'] = sorted(top_hartmann, key=lambda x: x[1], reverse=True)

    # --- DistilBERT (bhadresh-savani/distilbert-base-uncased-emotion) ---
    # This model is typically single-label classification (softmax output)
    if distil_model and distil_tokenizer:
        inputs = distil_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = distil_model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()
        top_prob_val, top_idx = torch.max(probs, dim=0)
        top_distil_label = distil_labels[top_idx.item()] if top_idx.item() < len(distil_labels) else "unknown"
        emotions_summary['distil_emotion'] = (top_distil_label, float(top_prob_val))

    # --- BERT (nateraw/bert-base-uncased-emotion) ---
    # This one is also multi-label (sigmoid output) like Hartmann's
    if bert_model and bert_tokenizer:
        inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = bert_model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze() # Sigmoid for multi-label
        
        top_bert = []
        for i in range(len(probs)):
            label_name = bert_labels[i] if i < len(bert_labels) else f"unknown_bert_label_{i}"
            score = float(probs[i])
            if score > 0.1: # Threshold
                top_bert.append((label_name, score))
        emotions_summary['bert_emotions'] = sorted(top_bert, key=lambda x: x[1], reverse=True)

    # --- Verified Emotion Merge ---
    # A more robust merge: average scores for common emotions, boost if multiple models agree.
    merged_emotions = defaultdict(lambda: {"total_score": 0.0, "count": 0, "sources": []})

    all_emotion_lists = [
        emotions_summary['hartmann_emotions'],
        emotions_summary['bert_emotions']
    ]
    # Add distil_emotion if its score is decent and it's not None
    if emotions_summary['distil_emotion'][0] is not None and emotions_summary['distil_emotion'][1] > 0.2:
        all_emotion_lists.append([emotions_summary['distil_emotion']])


    for i, emotion_list in enumerate(all_emotion_lists):
        source_name = ["hartmann", "distilbert", "bert"][i] # crude mapping for example
        for emotion, score in emotion_list:
            if score < 0.25: continue # Minimum confidence for an emotion to be considered in merge
            
            # Normalize emotion labels if necessary (e.g., some models might have "sad" vs "sadness")
            # For now, assume labels are reasonably consistent or use a mapping.
            normalized_emotion = emotion.lower() # Simple normalization

            merged_emotions[normalized_emotion]["total_score"] += score
            merged_emotions[normalized_emotion]["count"] += 1
            merged_emotions[normalized_emotion]["sources"].append(source_name)

    verified_emotions_final = []
    for emotion, data in merged_emotions.items():
        avg_score = data["total_score"] / data["count"]
        # Boost score if multiple models agree (e.g., agreement_boost)
        agreement_boost = 1.0 + (0.1 * (data["count"] - 1)) # 10% boost for each additional model agreeing
        final_score = min(1.0, avg_score * agreement_boost) # Cap at 1.0
        
        # Only include if final score is reasonably high
        if final_score > 0.3: # Threshold for "verified"
            verified_emotions_final.append((emotion, round(final_score, 4)))

    emotions_summary['verified'] = sorted(verified_emotions_final, key=lambda x: x[1], reverse=True)[:top_n_verified]
    
    return emotions_summary

if __name__ == '__main__':
    print("Testing emotion_handler.py...")
    if not MODELS_LOADED_EMOTION:
        print("Emotion models not loaded. Cannot run tests.")
    else:
        sample_text_1 = "I am so happy and excited about this new project! It's amazing."
        emotions1 = predict_emotions(sample_text_1)
        print(f"\nEmotions for: '{sample_text_1}'")
        print(f"  Hartmann: {emotions1['hartmann_emotions'][:3]}")
        print(f"  DistilBERT: {emotions1['distil_emotion']}")
        print(f"  BERT: {emotions1['bert_emotions'][:3]}")
        print(f"  VERIFIED: {emotions1['verified']}")
        assert len(emotions1['verified']) > 0
        assert emotions1['verified'][0][1] > 0.5 # Expecting a strong primary emotion

        sample_text_2 = "This is terrifying, I'm so scared and full of dread and anger."
        emotions2 = predict_emotions(sample_text_2)
        print(f"\nEmotions for: '{sample_text_2}'")
        print(f"  VERIFIED: {emotions2['verified']}")
        assert len(emotions2['verified']) > 0
        # Check if fear or anger is prominent
        assert any(emo[0] in ["fear", "anger", "dread"] for emo in emotions2['verified'][:2])


        sample_text_3 = "The weather is calm and the sky is clear." # More neutral
        emotions3 = predict_emotions(sample_text_3)
        print(f"\nEmotions for: '{sample_text_3}'")
        print(f"  VERIFIED: {emotions3['verified']}")
        # For neutral, 'verified' might be empty or have low scores / neutral
        if emotions3['verified']:
            assert emotions3['verified'][0][1] < 0.7 # Expect lower scores for neutral text
            assert any(emo[0] in ["neutral", "calm", "optimism", "relief"] for emo in emotions3['verified'])


        empty_text_emotions = predict_emotions("")
        print(f"\nEmotions for empty text: {empty_text_emotions}")
        assert empty_text_emotions['verified'] == []
        
        print("\n✅ emotion_handler.py tests completed.")