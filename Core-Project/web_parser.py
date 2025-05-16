import requests
import trafilatura # For robust main content extraction
from bs4 import BeautifulSoup # For fallback and further cleaning if needed
from pathlib import Path
import time
import re # For splitting in chunk_text fallback

# Removed imports for parser and vector_memory as web_parser will no longer call them directly.

def fetch_and_clean_with_trafilatura(url): # Removed timeout from signature here as well
    """Fetches content using trafilatura, which is good for main article text."""
    try:
        # print(f"Attempting to fetch with trafilatura: {url}")
        # CORRECTED: Removed timeout from trafilatura.fetch_url()
        downloaded = trafilatura.fetch_url(url) 
        if downloaded:
            # Allow trafilatura's own fallback mechanism by default
            clean_text = trafilatura.extract(downloaded, include_comments=False, include_tables=True, no_fallback=False) 
            if clean_text:
                return clean_text
            # else:
                # print(f"[INFO] Trafilatura extracted no main text from {url} even with its fallback.")
        # else:
            # print(f"[INFO] Trafilatura fetch_url failed to download content from {url}.")
        return None
    except Exception as e:
        print(f"[ERROR] Trafilatura failed for {url}: {e}")
        return None

def fallback_extract_text_with_bs4(url, timeout=10):
    """Fallback: basic HTML-to-text using BeautifulSoup if trafilatura fails or yields little."""
    try:
        # print(f"Attempting fallback fetch with BeautifulSoup: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.text, "html.parser") 
        
        for unwanted_tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "link", "meta"]): 
            unwanted_tag.decompose()
        
        text_blocks = []
        for element in soup.find_all(['p', 'div', 'article', 'section', 'main', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'pre', 'blockquote']):
            block_text = element.get_text(separator='\n', strip=True) 
            if block_text:
                text_blocks.append(block_text)
        
        full_text = '\n\n'.join(text_blocks) 
        return ' '.join(full_text.split()) if full_text else ""

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Fallback requests failed for {url}: {e}")
        return ""
    except Exception as e:
        print(f"[ERROR] Fallback BeautifulSoup extraction failed for {url}: {e}")
        return ""

def chunk_text(text, max_chunk_length=1000, overlap_sentences=1):
    if not text: return []
    
    sentences = []
    nlp_spacy = None # Initialize to None
    try:
        import spacy
        try:
            nlp_spacy = spacy.load("en_core_web_sm") 
            doc = nlp_spacy(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except OSError:
            print("[WARNING] spaCy 'en_core_web_sm' model not found for chunking. Run: python -m spacy download en_core_web_sm. Falling back.")
    except ImportError:
        print("[WARNING] spaCy not installed. Falling back to basic newline/punctuation splitting for chunking.")
    
    if not sentences: 
        potential_sentences = []
        for paragraph_like in re.split(r'\n\s*\n', text):
            for line_like in paragraph_like.split('\n'):
                line_like = line_like.strip()
                if len(line_like) > 3: 
                    potential_sentences.extend(re.split(r'(?<=[.!?])\s+', line_like)) 
        sentences = [s.strip() for s in potential_sentences if s.strip()]
        if not sentences and text.strip(): 
             sentences = [text.strip()]
        elif not sentences and not text.strip(): 
            return []

    chunks = []
    current_chunk_sentences = []
    current_length = 0
    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)
        if not sentence: continue

        if current_length > 0 and (current_length + sentence_len + 1 > max_chunk_length):
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = []
            current_length = 0
        
        if sentence_len > max_chunk_length:
            if current_chunk_sentences: 
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_length = 0
            for j in range(0, sentence_len, max_chunk_length):
                chunks.append(sentence[j : j + max_chunk_length])
        else:
            current_chunk_sentences.append(sentence)
            current_length += sentence_len + (1 if len(current_chunk_sentences) > 1 else 0)
                
    if current_chunk_sentences: 
        chunks.append(" ".join(current_chunk_sentences))
        
    return [c for c in chunks if c] 

def process_web_url(url):
    print(f"🌐 Fetching and cleaning URL: {url}")
    text_content = fetch_and_clean_with_trafilatura(url) # Corrected call
    
    if not text_content or len(text_content) < 100: 
        print(f"[INFO] Trafilatura yielded minimal/no content from {url}. Attempting BS4 fallback.")
        time.sleep(0.25) 
        text_content = fallback_extract_text_with_bs4(url)

    if not text_content:
        print(f"[ERROR] No content could be extracted from URL: {url}")
        return url, [] 

    chunks = chunk_text(text_content, max_chunk_length=1000) 
    
    if chunks:
        print(f"📄 Extracted {len(chunks)} text chunks from {url}.")
    else:
        print(f"[WARNING] No usable text chunks extracted from {url} after cleaning and chunking.")
    
    return url, chunks

if __name__ == '__main__':
    print("Testing web_parser.py...")
    test_urls = ["https://example.com"] 
    for test_url in test_urls:
        source_url, text_chunks = process_web_url(test_url)
        print(f"\n--- Chunks from: {source_url} ---")
        if text_chunks:
            for i, chunk in enumerate(text_chunks[:3]): 
                print(f"Chunk {i+1}/{len(text_chunks)} (length: {len(chunk)}):\n'{chunk[:200]}...'")
        else:
            print("No chunks returned.")
        print("-" * 30)

    long_text_example = ("This is the first sentence. This is the second sentence which is a bit longer. "
                         "The third sentence provides even more detail and context. And a fourth one. "
                         "Fifth sentence here. Sixth sentence makes it longer. Seventh is the charm. "
                         "Eighth, ninth, and the tenth sentence will conclude this example paragraph for chunking. "
                         "A very very very very very very very very very very very very very very very very very very very "
                         "very very very very very very very very very very very very very very very very very very very "
                         "long sentence without any punctuation just to test the hard splitting logic if it ever gets this far.")
    print("\n--- Testing chunk_text directly (max_chunk_length=100) ---")
    chunks_from_string = chunk_text(long_text_example, max_chunk_length=100)
    for i, chunk in enumerate(chunks_from_string):
        print(f"String Chunk {i+1} (len: {len(chunk)}): '{chunk}'")
    
    print("\n--- Testing chunk_text directly (max_chunk_length=50) ---")
    chunks_from_string_short = chunk_text(long_text_example, max_chunk_length=50)
    for i, chunk in enumerate(chunks_from_string_short):
        print(f"String Chunk Short {i+1} (len: {len(chunk)}): '{chunk}'")
