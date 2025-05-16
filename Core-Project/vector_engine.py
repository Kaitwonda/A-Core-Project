from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load models
minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
e5_model = SentenceTransformer('intfloat/e5-small-v2')  # or 'e5-base-v2'

def encode_with_minilm(text):
    return minilm_model.encode(text, convert_to_numpy=True)

def encode_with_e5(text):
    formatted = f"query: {text}"  # E5 expects 'query:' prefix for queries
    return e5_model.encode(formatted, convert_to_numpy=True)

def embed_text(text, model="minilm"):
    if model == "minilm":
        return encode_with_minilm(text).tolist()
    elif model == "e5":
        return encode_with_e5(text).tolist()
    else:
        raise ValueError("Unknown model.")

def fuse_vectors(text, threshold=0.7):
    vec_minilm = encode_with_minilm(text)
    vec_e5 = encode_with_e5(text)

    similarity = cosine_similarity([vec_minilm], [vec_e5])[0][0]

    if similarity >= threshold:
        fused = (vec_minilm + vec_e5) / 2
        source = "fused"
    else:
        fused = vec_minilm
        source = "minilm-dominant"

    return fused.tolist(), {
        "similarity": float(similarity),
        "source": source,
        "minilm": vec_minilm.tolist(),
        "e5": vec_e5.tolist()
    }

