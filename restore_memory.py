#!/usr/bin/env python3
"""
Memory Restoration Script

This script restores archived memory data back to the active memory files.
Specifically handles the optimizer_archived_phase1_vectors_final.json file.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

def restore_archived_vectors():
    """Restore archived vectors to active memory"""
    
    data_dir = Path("data")
    
    # Source files (archived data)
    archived_vectors = data_dir / "optimizer_archived_phase1_vectors_final.json"
    
    # Target files (active memory)
    vector_memory = data_dir / "vector_memory.json"
    
    print("🔄 Memory Restoration Tool")
    print("=" * 50)
    
    # Check if archived data exists
    if not archived_vectors.exists():
        print(f"❌ No archived vectors found at {archived_vectors}")
        return False
    
    # Load archived vectors
    print(f"📁 Loading archived vectors from {archived_vectors}")
    try:
        with open(archived_vectors, 'r', encoding='utf-8') as f:
            archived_data = json.load(f)
        
        print(f"✅ Found {len(archived_data)} archived vector entries")
    except Exception as e:
        print(f"❌ Error loading archived vectors: {e}")
        return False
    
    # Backup current vector memory if it exists and has data
    if vector_memory.exists():
        try:
            with open(vector_memory, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
            
            if current_data:  # If there's existing data
                backup_path = data_dir / f"vector_memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                shutil.copy2(vector_memory, backup_path)
                print(f"💾 Backed up current vector memory to {backup_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not backup current vectors: {e}")
    
    # Restore archived vectors to active memory
    print(f"🔄 Restoring vectors to {vector_memory}")
    try:
        with open(vector_memory, 'w', encoding='utf-8') as f:
            json.dump(archived_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully restored {len(archived_data)} vectors to active memory")
        
        # Show some stats
        if archived_data:
            # Check if vectors have the expected structure
            sample = archived_data[0]
            if 'text' in sample and 'vector' in sample:
                texts_with_content = [item for item in archived_data if item.get('text', '').strip()]
                print(f"📊 Memory stats:")
                print(f"   - Total entries: {len(archived_data)}")
                print(f"   - Entries with text: {len(texts_with_content)}")
                print(f"   - Vector dimensions: {len(sample['vector']) if 'vector' in sample else 'Unknown'}")
                
                # Show sample texts
                print(f"📝 Sample entries:")
                for i, item in enumerate(archived_data[:3]):
                    text = item.get('text', '')[:100]
                    print(f"   {i+1}. {text}{'...' if len(item.get('text', '')) > 100 else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error restoring vectors: {e}")
        return False

def restore_symbol_data():
    """Restore symbol occurrence log data to active symbol memory"""
    
    data_dir = Path("data")
    
    # Source files
    symbol_log = data_dir / "symbol_occurrence_log.json"
    seed_symbols_file = data_dir / "seed_symbols.json"
    
    # Target file
    symbol_memory = data_dir / "symbol_memory.json"
    
    restored_count = 0
    
    if symbol_log.exists():
        print(f"📁 Processing symbol occurrence log...")
        try:
            with open(symbol_log, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            if 'entries' in log_data:
                entries = log_data['entries']
                print(f"✅ Found {len(entries)} symbol entries in log")
                
                # Extract unique symbols
                symbols = {}
                for entry in entries:
                    symbol = entry.get('symbol')
                    if symbol and symbol not in symbols:
                        symbols[symbol] = {
                            'symbol': symbol,
                            'contexts': [entry.get('context', '')[:200]],
                            'emotions': [entry.get('emotion_in_context', 'neutral')],
                            'source_urls': [entry.get('source_url', '')],
                            'learning_phase': entry.get('learning_phase', 1),
                            'timestamp': entry.get('timestamp', ''),
                            'occurrences': 1
                        }
                    elif symbol:
                        # Accumulate data for existing symbol
                        symbols[symbol]['contexts'].append(entry.get('context', '')[:200])
                        symbols[symbol]['emotions'].append(entry.get('emotion_in_context', 'neutral'))
                        symbols[symbol]['source_urls'].append(entry.get('source_url', ''))
                        symbols[symbol]['occurrences'] += 1
                
                restored_count += len(symbols)
                
                # Save to symbol memory
                symbol_list = list(symbols.values())
                with open(symbol_memory, 'w', encoding='utf-8') as f:
                    json.dump(symbol_list, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Restored {len(symbols)} unique symbols to active memory")
                
        except Exception as e:
            print(f"❌ Error processing symbol log: {e}")
    
    return restored_count > 0

def main():
    """Main restoration function"""
    
    print("🚀 Starting memory restoration...")
    print()
    
    success = False
    
    # Restore vectors
    if restore_archived_vectors():
        success = True
    
    print()
    
    # Restore symbols  
    if restore_symbol_data():
        success = True
    
    print()
    
    if success:
        print("🎉 Memory restoration completed successfully!")
        print()
        print("Next steps:")
        print("1. Restart your AI system to load the restored memory")
        print("2. The system should now show non-zero memory counts on startup")
        print("3. Your AI should have access to its previous learning")
    else:
        print("❌ Memory restoration failed or no data to restore")
    
    return success

if __name__ == "__main__":
    main()