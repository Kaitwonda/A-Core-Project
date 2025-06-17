#!/usr/bin/env python3
"""
Tripartite Memory Migration Script

Migrates vector data from the old single-file format to the new tripartite memory system.
Analyzes each vector and routes it to logic, symbolic, or bridge memory based on content.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import re

# Import the evaluator to analyze content
try:
    from link_evaluator import evaluate_link_with_context
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    print("⚠️  Link evaluator not available - using basic heuristics")

def analyze_content(text: str, existing_data: Dict = None) -> Tuple[str, float, float]:
    """
    Analyze content to determine if it should go to logic, symbolic, or bridge memory.
    
    Returns:
        (decision_type, logic_score, symbolic_score)
    """
    
    if EVALUATOR_AVAILABLE and len(text.strip()) > 10:
        try:
            # Use the actual evaluator if available
            result = evaluate_link_with_context(text)
            decision = result.get('routing_decision', 'FOLLOW_HYBRID')
            logic_score = result.get('logic_score', 5.0)
            symbolic_score = result.get('symbolic_score', 5.0)
            return decision, logic_score, symbolic_score
        except:
            pass
    
    # Fallback to heuristic analysis
    text_lower = text.lower()
    
    # Logic indicators
    logic_keywords = [
        'algorithm', 'function', 'method', 'process', 'logic', 'compute', 'calculate',
        'data', 'structure', 'analysis', 'system', 'technical', 'implementation',
        'code', 'programming', 'database', 'network', 'protocol', 'specification',
        'definition', 'explanation', 'how', 'what', 'when', 'where', 'steps',
        'procedure', 'instruction', 'guide', 'documentation', 'manual'
    ]
    
    # Symbolic indicators  
    symbolic_keywords = [
        'feel', 'emotion', 'beautiful', 'meaning', 'purpose', 'soul', 'heart',
        'dream', 'hope', 'fear', 'love', 'hate', 'passion', 'inspire', 'create',
        'art', 'poetry', 'music', 'story', 'narrative', 'metaphor', 'symbol',
        'journey', 'growth', 'transformation', 'spiritual', 'consciousness',
        'experience', 'wisdom', 'insight', 'intuition', 'creativity'
    ]
    
    # Count matches
    logic_score = sum(1 for kw in logic_keywords if kw in text_lower)
    symbolic_score = sum(1 for kw in symbolic_keywords if kw in text_lower)
    
    # Check for questions (often logical)
    if any(text_lower.strip().startswith(q) for q in ['what', 'how', 'why', 'when', 'where']):
        logic_score += 2
    
    # Check for exclamations and emotional punctuation (symbolic)
    if '!' in text or text_lower.strip().endswith('?') and ('feel' in text_lower or 'think' in text_lower):
        symbolic_score += 1
    
    # Check length - very short text is often factual/logical
    if len(text.strip()) < 20:
        logic_score += 1
    
    # Determine decision
    if logic_score > symbolic_score + 1:
        decision = "FOLLOW_LOGIC"
        final_logic = min(10.0, 5.0 + logic_score)
        final_symbolic = max(1.0, 5.0 - logic_score/2)
    elif symbolic_score > logic_score + 1:
        decision = "FOLLOW_SYMBOLIC"
        final_logic = max(1.0, 5.0 - symbolic_score/2)
        final_symbolic = min(10.0, 5.0 + symbolic_score)
    else:
        decision = "FOLLOW_HYBRID"
        final_logic = 5.0 + (logic_score - symbolic_score) * 0.5
        final_symbolic = 5.0 + (symbolic_score - logic_score) * 0.5
    
    return decision, final_logic, final_symbolic

def convert_vector_to_tripartite_format(vector_item: Dict, index: int) -> Dict:
    """Convert old vector format to new tripartite memory format"""
    
    text = vector_item.get('text', '')
    
    # Analyze content for routing
    decision_type, logic_score, symbolic_score = analyze_content(text, vector_item)
    
    # Create unique ID
    if 'id' in vector_item:
        item_id = vector_item['id']
    else:
        # Create ID from content hash and index
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        item_id = f"{decision_type.lower()}_{text_hash}_{index}"
    
    # Build the tripartite memory item
    tripartite_item = {
        "id": item_id,
        "text": text,
        "source_url": vector_item.get('source_url', ''),
        "logic_score": logic_score,
        "symbolic_score": symbolic_score,
        "confidence": vector_item.get('confidence', 0.5),
        "processing_phase": vector_item.get('learning_phase', 1),
        "storage_phase": vector_item.get('learning_phase', 1),
        "is_shallow": vector_item.get('exploration_depth', 'shallow') == 'shallow',
        "is_highly_relevant": logic_score > 7 or symbolic_score > 7,
        "timestamp": vector_item.get('timestamp', datetime.utcnow().isoformat()),
        "content_type": vector_item.get('source_type', 'unknown'),
        "emotions": {},  # Would need emotion analysis
        "symbols_found": 0,  # Would need symbol analysis  
        "symbols_list": [],
        "keywords": [],  # Could extract from text
        "decision_type": decision_type,
        "stored_at": datetime.utcnow().isoformat(),
        # Preserve original vector data
        "original_vector": vector_item.get('vector', []),
        "migrated_from": "vector_memory"
    }
    
    # Add any additional metadata
    if 'metadata' in vector_item:
        tripartite_item['original_metadata'] = vector_item['metadata']
    
    return tripartite_item

def migrate_vectors_to_tripartite():
    """Main migration function"""
    
    data_dir = Path("data")
    
    # Source file
    vector_memory_file = data_dir / "vector_memory.json"
    
    # Target files
    logic_memory_file = data_dir / "logic_memory.json"
    symbolic_memory_file = data_dir / "symbolic_memory.json"
    bridge_memory_file = data_dir / "bridge_memory.json"
    
    print("🔄 Tripartite Memory Migration")
    print("=" * 50)
    
    # Load vector data
    if not vector_memory_file.exists():
        print(f"❌ No vector memory file found at {vector_memory_file}")
        return False
    
    print(f"📁 Loading vectors from {vector_memory_file}")
    try:
        with open(vector_memory_file, 'r', encoding='utf-8') as f:
            vectors = json.load(f)
        print(f"✅ Loaded {len(vectors)} vectors")
    except Exception as e:
        print(f"❌ Error loading vectors: {e}")
        return False
    
    if not vectors:
        print("❌ No vectors to migrate")
        return False
    
    # Prepare memory stores
    logic_items = []
    symbolic_items = []
    bridge_items = []
    
    # Process each vector
    print("🔄 Analyzing and routing vectors...")
    
    for i, vector_item in enumerate(vectors):
        try:
            tripartite_item = convert_vector_to_tripartite_format(vector_item, i)
            decision = tripartite_item['decision_type']
            
            if decision == "FOLLOW_LOGIC":
                logic_items.append(tripartite_item)
            elif decision == "FOLLOW_SYMBOLIC":
                symbolic_items.append(tripartite_item)
            else:  # FOLLOW_HYBRID
                bridge_items.append(tripartite_item)
                
        except Exception as e:
            print(f"⚠️  Error processing vector {i}: {e}")
            continue
    
    # Show distribution
    print(f"📊 Distribution results:")
    print(f"   Logic memory: {len(logic_items)} items")
    print(f"   Symbolic memory: {len(symbolic_items)} items")  
    print(f"   Bridge memory: {len(bridge_items)} items")
    print(f"   Total: {len(logic_items) + len(symbolic_items) + len(bridge_items)} items")
    
    # Backup existing files if they have data
    backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for file_path, name in [(logic_memory_file, 'logic'), (symbolic_memory_file, 'symbolic'), (bridge_memory_file, 'bridge')]:
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
                if existing_data:  # Not empty
                    backup_path = data_dir / f"{name}_memory_backup_{backup_timestamp}.json"
                    with open(backup_path, 'w') as f:
                        json.dump(existing_data, f, indent=2)
                    print(f"💾 Backed up existing {name} memory to {backup_path}")
            except:
                pass
    
    # Save to tripartite files
    success = True
    
    try:
        print(f"💾 Saving logic memory ({len(logic_items)} items)...")
        with open(logic_memory_file, 'w', encoding='utf-8') as f:
            json.dump(logic_items, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Saving symbolic memory ({len(symbolic_items)} items)...")
        with open(symbolic_memory_file, 'w', encoding='utf-8') as f:
            json.dump(symbolic_items, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Saving bridge memory ({len(bridge_items)} items)...")
        with open(bridge_memory_file, 'w', encoding='utf-8') as f:
            json.dump(bridge_items, f, indent=2, ensure_ascii=False)
            
        print("✅ Successfully migrated vectors to tripartite memory!")
        
    except Exception as e:
        print(f"❌ Error saving tripartite memory: {e}")
        success = False
    
    return success

def fix_symbol_memory():
    """Fix symbol memory format to be a dictionary"""
    
    data_dir = Path("data")
    symbol_file = data_dir / "symbol_memory.json"
    
    if not symbol_file.exists():
        print("ℹ️  No symbol memory file to fix")
        return True
    
    print("🔧 Fixing symbol memory format...")
    
    try:
        with open(symbol_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If it's already a dictionary, nothing to do
        if isinstance(data, dict):
            print("✅ Symbol memory is already in correct format")
            return True
        
        # If it's a list, convert to dictionary
        if isinstance(data, list):
            symbol_dict = {}
            
            for item in data:
                if isinstance(item, dict) and 'symbol' in item:
                    symbol = item['symbol']
                    symbol_dict[symbol] = {
                        "name": item.get('name', symbol),
                        "keywords": [],
                        "core_meanings": item.get('contexts', [])[:3],  # First 3 contexts as meanings
                        "emotions": [],
                        "emotion_profile": {},
                        "vector_examples": [],
                        "origin": "migrated",
                        "learning_phase": item.get('learning_phase', 1),
                        "resonance_weight": 0.5,
                        "created_at": item.get('timestamp', datetime.utcnow().isoformat()),
                        "updated_at": datetime.utcnow().isoformat(),
                        "usage_count": item.get('occurrences', 1)
                    }
            
            # Save as dictionary
            with open(symbol_file, 'w', encoding='utf-8') as f:
                json.dump(symbol_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Converted {len(symbol_dict)} symbols to dictionary format")
            return True
        
        else:
            print("⚠️  Unknown symbol memory format, creating empty dictionary")
            with open(symbol_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)
            return True
            
    except Exception as e:
        print(f"❌ Error fixing symbol memory: {e}")
        return False

def main():
    """Main migration function"""
    
    print("🚀 Starting tripartite memory migration...")
    print()
    
    success = True
    
    # Fix symbol memory format first
    if not fix_symbol_memory():
        success = False
    
    print()
    
    # Migrate vectors to tripartite system
    if not migrate_vectors_to_tripartite():
        success = False
    
    print()
    
    if success:
        print("🎉 Migration completed successfully!")
        print()
        print("Next steps:")
        print("1. Restart your AI system")
        print("2. The system should now show the correct memory counts:")
        print("   - Logic memory: factual, technical content")
        print("   - Symbolic memory: emotional, creative content")  
        print("   - Bridge memory: hybrid content spanning both domains")
        print("3. Your AI will have access to its previous learning in the new architecture")
    else:
        print("❌ Migration failed or incomplete")
    
    return success

if __name__ == "__main__":
    main()