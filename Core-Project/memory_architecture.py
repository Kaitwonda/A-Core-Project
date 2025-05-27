# memory_architecture.py - Core Storage & Persistence Layer

import json
import shutil
from pathlib import Path
from datetime import datetime
from threading import Lock
import traceback

class TripartiteMemory:
    """
    Three-way memory architecture with atomic persistence and recovery.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()  # Thread safety
        
        # Memory stores
        self.logic_memory = []
        self.symbolic_memory = []
        self.bridge_memory = []
        
        # Load existing data
        self._load_all()
        
    def _load_all(self):
        """Load all memories with fallback to backups"""
        self.logic_memory = self._load_safe("logic_memory.json")
        self.symbolic_memory = self._load_safe("symbolic_memory.json")
        self.bridge_memory = self._load_safe("bridge_memory.json")
        
    def _load_safe(self, filename):
        """Load with backup recovery and error handling"""
        path = self.data_dir / filename
        backup = self.data_dir / f"{filename}.backup"
        
        # Try primary file first
        try:
            if path.exists() and path.stat().st_size > 0:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"✅ Loaded {filename}: {len(data)} items")
                    return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ Error loading {filename}: {e}")
            
        # Try backup if primary failed
        try:
            if backup.exists() and backup.stat().st_size > 0:
                print(f"🔄 Recovering {filename} from backup")
                with open(backup, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Restore primary from backup
                    shutil.copy2(backup, path)
                    print(f"✅ Recovered {filename}: {len(data)} items")
                    return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ Backup recovery failed for {filename}: {e}")
            
        # Return empty list if both failed
        print(f"📝 Starting fresh for {filename}")
        return []
        
    def store(self, item, decision_type):
        """Store an item in the appropriate memory"""
        with self.lock:
            # Add metadata
            item['stored_at'] = datetime.utcnow().isoformat()
            item['decision_type'] = decision_type
            
            if decision_type == "FOLLOW_LOGIC":
                self.logic_memory.append(item)
            elif decision_type == "FOLLOW_SYMBOLIC":
                self.symbolic_memory.append(item)
            else:  # FOLLOW_HYBRID
                self.bridge_memory.append(item)
                
    def save_all(self):
        """Atomic save all memories with backups"""
        with self.lock:
            results = {
                'logic': self._save_safe("logic_memory.json", self.logic_memory),
                'symbolic': self._save_safe("symbolic_memory.json", self.symbolic_memory),
                'bridge': self._save_safe("bridge_memory.json", self.bridge_memory)
            }
            
            success_count = sum(1 for v in results.values() if v)
            print(f"💾 Saved {success_count}/3 memory stores successfully")
            return results
            
    def _save_safe(self, filename, data):
        """Save with atomic write and backup"""
        path = self.data_dir / filename
        temp = path.with_suffix('.tmp')
        backup = self.data_dir / f"{filename}.backup"
        
        try:
            # Write to temp file first
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            # Create backup of existing file if it exists
            if path.exists() and path.stat().st_size > 0:
                shutil.copy2(path, backup)
                
            # Atomic rename (temp -> primary)
            temp.replace(path)
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")
            traceback.print_exc()
            # Clean up temp file if it exists
            if temp.exists():
                temp.unlink()
            return False
            
    def get_counts(self):
        """Get item counts for each memory type"""
        with self.lock:
            return {
                'logic': len(self.logic_memory),
                'symbolic': len(self.symbolic_memory),
                'bridge': len(self.bridge_memory),
                'total': len(self.logic_memory) + len(self.symbolic_memory) + len(self.bridge_memory)
            }
            
    def clear_all(self):
        """Clear all memories (useful for testing)"""
        with self.lock:
            self.logic_memory = []
            self.symbolic_memory = []
            self.bridge_memory = []
            

# Unit tests
# Make imports available for memory_optimizer.py
__all__ = ['TripartiteMemory']

if __name__ == "__main__":
    import tempfile
    import os
    from concurrent.futures import ThreadPoolExecutor
    
    print("🧪 Testing TripartiteMemory...")
    
    # Test 1: Basic storage and retrieval
    print("\n1️⃣ Test: Basic storage and retrieval")
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = TripartiteMemory(data_dir=tmpdir)
        
        # Store items
        mem.store({'text': 'Logic item', 'logic_score': 0.9}, 'FOLLOW_LOGIC')
        mem.store({'text': 'Symbolic item', 'symbolic_score': 0.9}, 'FOLLOW_SYMBOLIC')
        mem.store({'text': 'Bridge item', 'logic_score': 0.5, 'symbolic_score': 0.5}, 'FOLLOW_HYBRID')
        
        counts = mem.get_counts()
        assert counts['logic'] == 1, f"Expected 1 logic item, got {counts['logic']}"
        assert counts['symbolic'] == 1, f"Expected 1 symbolic item, got {counts['symbolic']}"
        assert counts['bridge'] == 1, f"Expected 1 bridge item, got {counts['bridge']}"
        print("✅ Basic storage works")
        
    # Test 2: Persistence across instances
    print("\n2️⃣ Test: Persistence across instances")
    with tempfile.TemporaryDirectory() as tmpdir:
        # First instance
        mem1 = TripartiteMemory(data_dir=tmpdir)
        mem1.store({'text': 'Persistent item'}, 'FOLLOW_LOGIC')
        mem1.save_all()
        
        # Second instance should load the data
        mem2 = TripartiteMemory(data_dir=tmpdir)
        counts = mem2.get_counts()
        assert counts['logic'] == 1, "Data not persisted"
        assert mem2.logic_memory[0]['text'] == 'Persistent item'
        print("✅ Persistence works")
        
    # Test 3: Backup recovery
    print("\n3️⃣ Test: Backup recovery from corruption")
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = TripartiteMemory(data_dir=tmpdir)
        mem.store({'text': 'Important data'}, 'FOLLOW_LOGIC')
        mem.save_all()
        
        # Corrupt the primary file
        logic_path = Path(tmpdir) / "logic_memory.json"
        logic_path.write_text("corrupted{json")
        
        # New instance should recover from backup
        mem2 = TripartiteMemory(data_dir=tmpdir)
        assert len(mem2.logic_memory) == 1
        assert mem2.logic_memory[0]['text'] == 'Important data'
        print("✅ Backup recovery works")
        
    # Test 4: Concurrent writes
    print("\n4️⃣ Test: Concurrent writes")
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = TripartiteMemory(data_dir=tmpdir)
        
        def store_many(thread_id):
            for i in range(10):
                mem.store({'text': f'Thread {thread_id} item {i}'}, 'FOLLOW_LOGIC')
            mem.save_all()
            
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(store_many, i) for i in range(3)]
            for f in futures:
                f.result()
                
        counts = mem.get_counts()
        assert counts['logic'] == 30, f"Expected 30 items, got {counts['logic']}"
        print("✅ Thread safety works")
        
    # Test 5: Empty file handling
    print("\n5️⃣ Test: Empty file handling")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty files
        for fname in ['logic_memory.json', 'symbolic_memory.json', 'bridge_memory.json']:
            (Path(tmpdir) / fname).touch()
            
        mem = TripartiteMemory(data_dir=tmpdir)
        counts = mem.get_counts()
        assert counts['total'] == 0, "Should handle empty files gracefully"
        print("✅ Empty file handling works")
        
    print("\n✅ All tests passed!")