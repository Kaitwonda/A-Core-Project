# File: vector_engine.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load models once at import time to avoid repeated downloads
try:
    minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    e5_model    = SentenceTransformer('intfloat/e5-small-v2')
    MODELS_LOADED = True
    print("✅ Vector embedding models loaded: MiniLM & E5")
except Exception as e:
    MODELS_LOADED = False
    print(f"⚠️ Failed to load embedding models: {e}")
    minilm_model = None
    e5_model = None


def encode_with_minilm(text: str) -> np.ndarray:
    """Return a MiniLM embedding or a zero‐vector if unavailable/empty."""
    if not MODELS_LOADED or minilm_model is None:
        dim = 384
    else:
        dim = minilm_model.get_sentence_embedding_dimension()
    if not text or not isinstance(text, str) or not text.strip():
        return np.zeros(dim)
    return minilm_model.encode(text, convert_to_numpy=True)


def encode_with_e5(text: str) -> np.ndarray:
    """Return an E5 embedding or a zero‐vector if unavailable/empty."""
    if not MODELS_LOADED or e5_model is None:
        dim = 384
    else:
        dim = e5_model.get_sentence_embedding_dimension()
    if not text or not isinstance(text, str) or not text.strip():
        return np.zeros(dim)
    formatted = f"query: {text}"
    return e5_model.encode(formatted, convert_to_numpy=True)


def fuse_vectors(text: str, threshold: float = 0.7):
    """
    Encode `text` with both MiniLM and E5, compute their cosine similarity,
    and either:
      - Average them if sim ≥ threshold (source="fused")
      - Otherwise pick MiniLM by default (source="minilm_dominant"), 
        or the non-zero one if the other is zero.
    Returns: (vector_list or None, debug_info dict)
    """
    if not MODELS_LOADED:
        return None, {"similarity": 0.0, "source": "models_not_loaded", "error": "Models failed to load"}

    if not text or not text.strip():
        return None, {"similarity": 0.0, "source": "empty_input", "error": "Empty or blank text"}

    v_minilm = encode_with_minilm(text)
    v_e5     = encode_with_e5(text)

    # if both are zero‐vectors, bail out
    if np.all(v_minilm == 0) and np.all(v_e5 == 0):
        return None, {"similarity": 0.0, "source": "zero_vectors", "error": "Both encoders returned zero"}

    # compute similarity
    try:
        sim = float(cosine_similarity(v_minilm.reshape(1, -1), v_e5.reshape(1, -1))[0][0])
    except Exception as e:
        # fallback to MiniLM if similarity fails
        return v_minilm.tolist(), {"similarity": 0.0, "source": "minilm_fallback", "error": str(e)}

    # decide fusion or fallback
    if sim >= threshold:
        fused = ((v_minilm + v_e5) / 2.0)
        vec_out = fused
        src = "fused"
    else:
        # if one is zero, pick the other
        if np.all(v_minilm == 0) and not np.all(v_e5 == 0):
            vec_out, src = v_e5, "e5_dominant_minilm_zero"
        elif not np.all(v_minilm == 0) and np.all(v_e5 == 0):
            vec_out, src = v_minilm, "minilm_dominant_e5_zero"
        else:
            vec_out, src = v_minilm, "minilm_dominant_low_similarity"

    debug = {"similarity": sim, "source": src}
    return vec_out.tolist(), debug


def embed_text(text: str, model_choice: str = "minilm"):
    """
    Encode text with a single specified model.
    Returns vector_list or None on error/empty input.
    """
    if not MODELS_LOADED:
        return None
    if not text or not text.strip():
        return None

    if model_choice.lower() == "minilm":
        vec = encode_with_minilm(text)
    elif model_choice.lower() == "e5":
        vec = encode_with_e5(text)
    else:
        return None

    if np.all(vec == 0):
        return None
    return vec.tolist()


if __name__ == "__main__":
    print("🔬 Testing vector_engine.py")

    if not MODELS_LOADED:
        print("Models are not loaded; tests will be skipped.")
    else:
        # dynamic dimension fetch
        dim = minilm_model.get_sentence_embedding_dimension()
        samples = [
            ("Test similarity between two close sentences.", True),
            ("A completely unrelated sentence about quantum physics.", False),
            ("", None),
            ("   ", None),
        ]

        for text, expect_similarity in samples:
            vec, info = fuse_vectors(text)
            print(f">>> '{text[:30]}...'  ➔  source={info['source']}, sim={info['similarity']:.4f}")
            if expect_similarity is None:
                assert vec is None
            else:
                assert vec is not None
                assert len(vec) == dim

        # test embed_text
        single = embed_text("Hello world", model_choice="minilm")
        assert single is None or len(single) == dim

        print("✅ All tests in vector_engine.py passed.")
