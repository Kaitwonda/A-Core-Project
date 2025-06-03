# content_utils.py - Content type detection utilities

import re

def detect_content_type(text_input: str, spacy_nlp_instance=None) -> str:
    """
    Detect whether content is factual, symbolic, or ambiguous.
    Moved here to avoid circular imports.
    """
    if not text_input or not isinstance(text_input, str):
        return "ambiguous"
    
    text_lower = text_input.lower()
    
    factual_markers = [
        "according to", "study shows", "research indicates", "published in", "cited in", "evidence suggests",
        "data shows", "statistics indicate", "found that", "confirmed that", "demonstrated that",
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "dr.", "prof.", "university of", "institute of", "journal of", ".gov", ".edu", ".org",
        "theorem", "equation", "formula", "law of", "principle of",
        "born on", "died on", "founded in", "established in",
        "kg", "km", "meter", "liter", "celsius", "fahrenheit", "%", "$", "€", "¥"
    ]
    
    symbolic_markers = [
        "love", "hate", "fear", "joy", "sadness", "anger", "hope", "dream", "nightmare",
        "like a", "as if", "metaphor", "symbolizes", "represents", "signifies", "embodies", "evokes",
        "spirit", "soul", "ghost", "magic", "myth", "legend", "folklore", "ritual", "omen",
        "🔥", "💧", "🌀", "💡", "🧩", "♾️",
        "heart", "light", "darkness", "shadow", "journey", "quest", "fate", "destiny",
        "feels like", "seems as though", "one might say", "could be seen as"
    ]
    
    f_count = sum(marker in text_lower for marker in factual_markers)
    s_count = sum(marker in text_lower for marker in symbolic_markers)
    
    numbers = re.findall(r'(?<!\w)[-+]?\d*\.?\d+(?!\w)', text_lower)
    if len(numbers) > 2: 
        f_count += 1
    if len(numbers) > 5: 
        f_count += 1
        
    if spacy_nlp_instance:
        doc = spacy_nlp_instance(text_input[:spacy_nlp_instance.max_length])
        entity_factual_boost = 0
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
                entity_factual_boost += 0.5
            elif ent.label_ in ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"]:
                entity_factual_boost += 0.25
        f_count += entity_factual_boost
        
    if f_count > s_count * 1.5: 
        return "factual"
    elif s_count > f_count * 1.5: 
        return "symbolic"
    else:
        if f_count == 0 and s_count == 0:
            if len(text_input.split()) < 5: 
                return "ambiguous"
            if len(numbers) > 0: 
                return "factual"
            return "ambiguous"
        elif f_count > s_count: 
            return "factual"
        elif s_count > f_count: 
            return "symbolic"
        return "ambiguous"