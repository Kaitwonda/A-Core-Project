# weight_evolution.py - Progressive Weight Evolution System

import json
from pathlib import Path
from datetime import datetime

class WeightEvolver:
    """
    Manages progressive evolution of weights toward specialization.
    Includes momentum to prevent oscillation and feedback from memory distribution.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.weights_file = self.data_dir / "adaptive_weights.json"
        self.momentum_file = self.data_dir / "weight_momentum.json"
        self.history_file = self.data_dir / "weight_evolution_history.json"
        
        # Load current state
        self.weights = self._load_weights()
        self.momentum = self._load_momentum()
        self.history = self._load_history()
        
    def _load_weights(self):
        """Load current weights"""
        try:
            if self.weights_file.exists():
                with open(self.weights_file, 'r') as f:
                    data = json.load(f)
                    return {
                        'static': data.get('link_score_weight_static', 0.6),
                        'dynamic': data.get('link_score_weight_dynamic', 0.4),
                        'last_updated': data.get('last_updated')
                    }
        except Exception:
            pass
        return {'static': 0.6, 'dynamic': 0.4, 'last_updated': None}
        
    def _save_weights(self):
        """Save current weights"""
        data = {
            'link_score_weight_static': self.weights['static'],
            'link_score_weight_dynamic': self.weights['dynamic'],
            'last_updated': datetime.utcnow().isoformat()
        }
        with open(self.weights_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def _load_momentum(self):
        """Load momentum state"""
        try:
            if self.momentum_file.exists():
                with open(self.momentum_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            'static_wins': 0,
            'dynamic_wins': 0,
            'last_direction': None,
            'consecutive_moves': 0
        }
        
    def _save_momentum(self):
        """Save momentum state"""
        with open(self.momentum_file, 'w') as f:
            json.dump(self.momentum, f, indent=2)
            
    def _load_history(self):
        """Load evolution history"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []
        
    def _save_history(self):
        """Save evolution history (keep last 50 entries)"""
        self.history = self.history[-50:]
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
            
    def get_current_specialization(self):
        """Calculate current specialization level (0=balanced, 1=fully specialized)"""
        return abs(self.weights['static'] - self.weights['dynamic'])
        
    def calculate_target_specialization(self, run_count, memory_stats=None):
        """
        Calculate target specialization based on:
        1. Number of optimization runs
        2. Bridge memory size (if provided)
        """
        # Base target increases with runs
        base_target = min(0.8, 0.3 + (run_count * 0.05))
        
        if memory_stats and 'distribution' in memory_stats:
            bridge_pct = memory_stats['distribution'].get('bridge_pct', 50)
            
            # Accelerate specialization if bridge is shrinking well
            if bridge_pct < 20:
                return min(0.9, base_target * 1.2)
            elif bridge_pct < 30:
                return min(0.85, base_target * 1.1)
            elif bridge_pct > 50:
                # Slow down if bridge is too large
                return base_target * 0.9
                
        return base_target
        
    def evolve_weights(self, run_count, memory_stats=None, performance_stats=None):
        """
        Evolve weights toward specialization with momentum.
        
        Args:
            run_count: Number of optimization runs
            memory_stats: Optional memory distribution stats
            performance_stats: Optional performance metrics (win rates, etc.)
        """
        old_static = self.weights['static']
        old_dynamic = self.weights['dynamic']
        
        # Current state
        current_spec = self.get_current_specialization()
        target_spec = self.calculate_target_specialization(run_count, memory_stats)
        
        print(f"\n⚡ Weight Evolution:")
        print(f"  Current: static={old_static:.3f}, dynamic={old_dynamic:.3f}")
        print(f"  Specialization: {current_spec:.3f} → target {target_spec:.3f}")
        
        # Check if we need to evolve
        if current_spec >= target_spec:
            print("  → Already at target specialization")
            return False
            
        # Determine direction based on current bias or performance
        if performance_stats:
            # Use performance to guide direction
            logic_wins = performance_stats.get('logic_win_rate', 0)
            symbol_wins = performance_stats.get('symbol_win_rate', 0)
            
            if logic_wins > symbol_wins + 0.1:  # Significant logic advantage
                direction = 'static'
            elif symbol_wins > logic_wins + 0.1:  # Significant symbolic advantage
                direction = 'dynamic'
            else:
                # No clear winner, continue current direction
                direction = 'static' if old_static > old_dynamic else 'dynamic'
        else:
            # Continue in current direction
            direction = 'static' if old_static > old_dynamic else 'dynamic'
            
        # Calculate step size with momentum
        base_step = 0.02
        
        # Apply momentum if moving in same direction
        if self.momentum['last_direction'] == direction:
            self.momentum['consecutive_moves'] += 1
            momentum_factor = 1 + (self.momentum['consecutive_moves'] * 0.1)
            step = min(0.05, base_step * momentum_factor)  # Cap at 0.05
        else:
            # Direction change, reset momentum
            self.momentum['consecutive_moves'] = 1
            step = base_step
            
        # Update momentum tracking
        self.momentum['last_direction'] = direction
        if direction == 'static':
            self.momentum['static_wins'] += 1
            self.momentum['dynamic_wins'] = 0
        else:
            self.momentum['dynamic_wins'] += 1
            self.momentum['static_wins'] = 0
            
        # Apply the evolution
        if direction == 'static':
            new_static = min(0.9, old_static + step)
            new_dynamic = 1.0 - new_static
        else:
            new_dynamic = min(0.9, old_dynamic + step)
            new_static = 1.0 - new_dynamic
            
        # Update weights
        self.weights['static'] = round(new_static, 3)
        self.weights['dynamic'] = round(new_dynamic, 3)
        
        # Record in history
        self.history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'run_count': run_count,
            'old_weights': {'static': old_static, 'dynamic': old_dynamic},
            'new_weights': {'static': self.weights['static'], 'dynamic': self.weights['dynamic']},
            'target_specialization': target_spec,
            'actual_specialization': self.get_current_specialization(),
            'momentum': self.momentum.copy(),
            'memory_stats': memory_stats,
            'performance_stats': performance_stats
        })
        
        # Save everything
        self._save_weights()
        self._save_momentum()
        self._save_history()
        
        print(f"  → Evolved to: static={self.weights['static']:.3f}, dynamic={self.weights['dynamic']:.3f}")
        print(f"  Momentum: {self.momentum['consecutive_moves']} consecutive {direction} moves")
        
        return True
        
    def get_evolution_summary(self):
        """Get summary of weight evolution over time"""
        if not self.history:
            return {
                'total_evolutions': 0,
                'current_weights': self.weights,
                'current_specialization': self.get_current_specialization()
            }
            
        # Analyze history
        first_entry = self.history[0]
        last_entry = self.history[-1]
        
        # Calculate total drift
        initial_spec = abs(first_entry['old_weights']['static'] - first_entry['old_weights']['dynamic'])
        current_spec = self.get_current_specialization()
        
        # Find dominant direction
        static_moves = sum(1 for h in self.history if h.get('momentum', {}).get('last_direction') == 'static')
        dynamic_moves = len(self.history) - static_moves
        
        return {
            'total_evolutions': len(self.history),
            'current_weights': self.weights,
            'current_specialization': current_spec,
            'specialization_increase': current_spec - initial_spec,
            'dominant_direction': 'static' if static_moves > dynamic_moves else 'dynamic',
            'direction_ratio': {
                'static_moves': static_moves,
                'dynamic_moves': dynamic_moves
            },
            'momentum_state': self.momentum
        }
        

# Unit tests
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing Progressive Weight Evolution...")
    
    # Test 1: Basic evolution
    print("\n1️⃣ Test: Basic weight evolution")
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = WeightEvolver(data_dir=tmpdir)
        
        # Initial state
        assert evolver.weights['static'] == 0.6
        assert evolver.weights['dynamic'] == 0.4
        
        # Evolve once
        evolved = evolver.evolve_weights(run_count=1)
        assert evolved == True, "Should evolve"
        assert evolver.get_current_specialization() > 0.2, "Specialization should increase"
        
        # Check weights sum to 1
        assert abs(evolver.weights['static'] + evolver.weights['dynamic'] - 1.0) < 0.001
        
        print("✅ Basic evolution works")
        
    # Test 2: Target specialization with memory feedback
    print("\n2️⃣ Test: Memory-aware evolution")
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = WeightEvolver(data_dir=tmpdir)
        
        # With small bridge, should accelerate
        memory_stats = {
            'distribution': {
                'logic_pct': 45,
                'symbolic_pct': 40,
                'bridge_pct': 15  # Small bridge
            }
        }
        
        target = evolver.calculate_target_specialization(5, memory_stats)
        base_target = evolver.calculate_target_specialization(5, None)
        
        assert target > base_target, "Should accelerate with small bridge"
        
        print("✅ Memory-aware targeting works")
        
    # Test 3: Momentum acceleration
    print("\n3️⃣ Test: Momentum acceleration")
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = WeightEvolver(data_dir=tmpdir)
        
        # Force consistent direction
        old_spec = evolver.get_current_specialization()
        
        # Multiple evolutions in same direction
        for i in range(3):
            evolver.evolve_weights(run_count=i+1)
            
        # Check momentum built up
        assert evolver.momentum['consecutive_moves'] >= 3
        new_spec = evolver.get_current_specialization()
        assert new_spec > old_spec + 0.1, "Momentum should accelerate evolution"
        
        print("✅ Momentum acceleration works")
        
    # Test 4: Performance-guided evolution
    print("\n4️⃣ Test: Performance-guided evolution")
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = WeightEvolver(data_dir=tmpdir)
        
        # Simulate logic winning more
        performance_stats = {
            'logic_win_rate': 0.7,
            'symbol_win_rate': 0.3,
            'hybrid_rate': 0.0
        }
        
        evolver.evolve_weights(run_count=1, performance_stats=performance_stats)
        
        # Should favor static (logic)
        assert evolver.weights['static'] > 0.6, "Should increase static weight"
        assert evolver.momentum['last_direction'] == 'static'
        
        print("✅ Performance-guided evolution works")
        
    # Test 5: Evolution history and persistence
    print("\n5️⃣ Test: Evolution history and persistence")
    with tempfile.TemporaryDirectory() as tmpdir:
        # First session
        evolver1 = WeightEvolver(data_dir=tmpdir)
        
        # Evolve multiple times
        for i in range(3):
            evolver1.evolve_weights(run_count=i+1)
            
        summary1 = evolver1.get_evolution_summary()
        assert summary1['total_evolutions'] == 3
        
        # Second session - should load history
        evolver2 = WeightEvolver(data_dir=tmpdir)
        summary2 = evolver2.get_evolution_summary()
        
        assert summary2['total_evolutions'] == 3, "History should persist"
        assert evolver2.weights == evolver1.weights, "Weights should persist"
        assert evolver2.momentum == evolver1.momentum, "Momentum should persist"
        
        print("✅ History and persistence work")
        
    # Test 6: Target capping
    print("\n6️⃣ Test: Specialization limits")
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = WeightEvolver(data_dir=tmpdir)
        
        # Evolve many times
        for i in range(20):
            evolver.evolve_weights(run_count=i+1)
            
        # Check we don't exceed limits
        assert evolver.weights['static'] <= 0.9
        assert evolver.weights['dynamic'] >= 0.1
        assert evolver.get_current_specialization() <= 0.8
        
        print("✅ Specialization limits respected")
        
    print("\n✅ All weight evolution tests passed!")