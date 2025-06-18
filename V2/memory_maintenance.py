# memory_maintenance.py - Memory maintenance and cleanup functions
"""
Memory maintenance functions for cleaning up and archiving memory data.
Provides compatibility for memory_optimizer.py maintenance operations.
"""

import json
from pathlib import Path
from datetime import datetime
from unified_memory import get_unified_memory

def prune_phase1_symbolic_vectors(archive_path_str="data/archived_phase1_vectors.json"):
    """
    Prune Phase 1 symbolic vectors from the vector memory system.
    Archives removed vectors for potential recovery.
    """
    try:
        print(f"🧹 Starting Phase 1 symbolic vector pruning...")
        
        unified_memory = get_unified_memory()
        vector_data = getattr(unified_memory, 'vector_data', [])
        
        if not vector_data:
            print("   No vector data to prune")
            return 0
        
        # Identify Phase 1 symbolic vectors
        phase1_symbolic = []
        remaining_vectors = []
        
        for vector_entry in vector_data:
            learning_phase = vector_entry.get('learning_phase', 0)
            source_type = vector_entry.get('source_type', '')
            
            # Identify Phase 1 symbolic content
            is_phase1 = (learning_phase == 1)
            is_symbolic = ('symbolic' in source_type.lower() or 
                          'emotion' in source_type.lower() or
                          vector_entry.get('contains_symbols', False))
            
            if is_phase1 and is_symbolic:
                phase1_symbolic.append(vector_entry)
            else:
                remaining_vectors.append(vector_entry)
        
        if not phase1_symbolic:
            print("   No Phase 1 symbolic vectors found to prune")
            return 0
        
        # Archive the pruned vectors
        archive_path = Path(archive_path_str)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing archive if it exists
        archived_data = []
        if archive_path.exists():
            try:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archived_data = json.load(f)
            except:
                archived_data = []
        
        # Add new archived items with timestamp
        for item in phase1_symbolic:
            item['archived_at'] = datetime.utcnow().isoformat()
            item['archive_reason'] = 'phase1_symbolic_prune'
            archived_data.append(item)
        
        # Save updated archive
        with open(archive_path, 'w', encoding='utf-8') as f:
            json.dump(archived_data, f, indent=2, ensure_ascii=False)
        
        # Update the vector memory with remaining vectors
        unified_memory.vector_data = remaining_vectors
        
        # Update the vector memory file if it exists
        vector_file = unified_memory.data_dir / "vector_memory.json"
        if vector_file.exists():
            with open(vector_file, 'w', encoding='utf-8') as f:
                json.dump(remaining_vectors, f, indent=2, ensure_ascii=False)
        
        pruned_count = len(phase1_symbolic)
        print(f"   ✅ Pruned {pruned_count} Phase 1 symbolic vectors")
        print(f"   📁 Archived to: {archive_path}")
        print(f"   📊 Remaining vectors: {len(remaining_vectors)}")
        
        return pruned_count
        
    except Exception as e:
        print(f"   ❌ Error during Phase 1 pruning: {e}")
        return 0

def cleanup_old_archives(max_archive_age_days=30):
    """Clean up old archive files to prevent disk bloat"""
    try:
        data_dir = Path("data")
        archive_files = list(data_dir.glob("*archived*.json"))
        
        cleaned = 0
        for archive_file in archive_files:
            # Check file age
            age_days = (datetime.now() - datetime.fromtimestamp(archive_file.stat().st_mtime)).days
            
            if age_days > max_archive_age_days:
                archive_file.unlink()
                cleaned += 1
                
        if cleaned > 0:
            print(f"🗑️ Cleaned up {cleaned} old archive files")
            
        return cleaned
        
    except Exception as e:
        print(f"❌ Error during archive cleanup: {e}")
        return 0

def get_maintenance_stats():
    """Get statistics about memory maintenance status"""
    try:
        data_dir = Path("data")
        archive_files = list(data_dir.glob("*archived*.json"))
        
        total_archived = 0
        for archive_file in archive_files:
            try:
                with open(archive_file, 'r') as f:
                    archived_data = json.load(f)
                    total_archived += len(archived_data)
            except:
                pass
        
        unified_memory = get_unified_memory()
        current_vectors = len(getattr(unified_memory, 'vector_data', []))
        
        return {
            'current_vectors': current_vectors,
            'total_archived': total_archived,
            'archive_files': len(archive_files),
            'last_maintenance': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error getting maintenance stats: {e}")
        return {}