# weight_evolution.py - Progressive Weight Evolution System (Autonomous Version)

import json
from pathlib import Path
from datetime import datetime

class WeightEvolver:
    """
    Manages progressive evolution of weights based on actual data patterns.
    Truly autonomous - learns optimal balance from content distribution.
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
        Calculate target specialization based on actual data distribution.
        This makes it truly autonomous - it learns from what it's seeing.
        """
        
        # If no memory stats, start balanced
        if not memory_stats or 'distribution' not in memory_stats:
            return 0.0  # Start balanced until we have data
        
        dist = memory_stats['distribution']
        logic_pct = dist.get('logic_pct', 0)
        symbolic_pct = dist.get('symbolic_pct', 0)
        bridge_pct = dist.get('bridge_pct', 0)
        
        # Key insight: If bridge is large, we're not classifying well
        # The system should adjust weights to reduce bridge size
        
        if bridge_pct > 40:
            # Large bridge means we need to help items migrate
            # Look at what's accumulating more
            if logic_pct > symbolic_pct * 2:
                # We have way more logic than symbolic
                # Maybe we're over-weighting logic? Try balancing
                return 0.0  # Push toward balance
            elif symbolic_pct > logic_pct * 2:
                # We have way more symbolic than logic
                # Maybe we're over-weighting symbolic? Try balancing
                return 0.0  # Push toward balance
            else:
                # Bridge is large but distribution is balanced
                # Try slight specialization to help classification
                return 0.2
        
        # If bridge is small, the system is working well
        # Let it continue with current specialization
        if bridge_pct < 10:
            current_spec = self.get_current_specialization()
            return min(0.6, current_spec * 1.1)  # Slightly increase what's working
        
        # Medium bridge (10-40%) - adjust based on content ratio
        # This is the autonomous part - learn from the data!
        if logic_pct > 0 and symbolic_pct > 0:
            # Calculate natural ratio in the data
            total_classified = logic_pct + symbolic_pct
            logic_ratio = logic_pct / total_classified
            symbolic_ratio = symbolic_pct / total_classified
            
            # If data is naturally 70% logic, 30% symbolic
            # then weights should reflect that to minimize bridge
            if logic_ratio > 0.7:
                # Data is logic-heavy, allow weights to specialize
                return 0.4  # This allows up to 70/30 split
            elif symbolic_ratio > 0.7:
                # Data is symbolic-heavy, allow weights to specialize
                return 0.4  # This allows up to 30/70 split
            else:
                # Data is balanced, keep weights balanced
                return 0.1  # Allow only slight specialization
        
        # Default: slight specialization
        return 0.2
        
    def evolve_weights(self, run_count, memory_stats=None, performance_stats=None):
        """
        Evolve weights toward specialization based on actual data patterns.
        Now truly autonomous - learns from content distribution.
        """
        old_static = self.weights['static']
        old_dynamic = self.weights['dynamic']
        
        # Current state
        current_spec = self.get_current_specialization()
        target_spec = self.calculate_target_specialization(run_count, memory_stats)
        
        print(f"\n⚡ Weight Evolution:")
        print(f"  Current: static={old_static:.3f}, dynamic={old_dynamic:.3f}")
        print(f"  Specialization: {current_spec:.3f} → target {target_spec:.3f}")
        
        # If we have memory stats, show why we're making this decision
        if memory_stats and 'distribution' in memory_stats:
            dist = memory_stats['distribution']
            print(f"  Data distribution: Logic={dist.get('logic_pct', 0):.1f}%, "
                  f"Symbolic={dist.get('symbolic_pct', 0):.1f}%, "
                  f"Bridge={dist.get('bridge_pct', 0):.1f}%")
        
        # Check if we need to evolve
        if abs(current_spec - target_spec) < 0.02:
            print("  → Already close to target specialization")
            return False
        
        # Determine direction based on actual data patterns
        if memory_stats and 'distribution' in memory_stats:
            dist = memory_stats['distribution']
            logic_pct = dist.get('logic_pct', 0)
            symbolic_pct = dist.get('symbolic_pct', 0)
            
            # Autonomous decision: follow the data
            if logic_pct > symbolic_pct * 1.5 and current_spec < target_spec:
                direction = 'static'  # Strengthen logic
            elif symbolic_pct > logic_pct * 1.5 and current_spec < target_spec:
                direction = 'dynamic'  # Strengthen symbolic
            elif current_spec > target_spec:
                # Need to reduce specialization
                direction = 'reduce'
            else:
                # Use performance stats if available
                if performance_stats:
                    logic_wins = performance_stats.get('logic_win_rate', 0)
                    symbol_wins = performance_stats.get('symbol_win_rate', 0)
                    direction = 'static' if logic_wins > symbol_wins else 'dynamic'
                else:
                    # No clear signal, maintain current bias
                    direction = 'static' if old_static > old_dynamic else 'dynamic'
        else:
            # No data, evolve slowly toward balance
            direction = 'reduce' if current_spec > 0.1 else 'static'
        
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
        elif direction == 'dynamic':
            self.momentum['dynamic_wins'] += 1
            self.momentum['static_wins'] = 0
            
        # Apply the evolution
        if direction == 'reduce':
            # Move toward balance
            if old_static > old_dynamic:
                new_static = max(0.5, old_static - step)
                new_dynamic = 1.0 - new_static
            else:
                new_dynamic = max(0.5, old_dynamic - step)
                new_static = 1.0 - new_dynamic
        elif direction == 'static':
            new_static = min(0.9, old_static + step)
            new_dynamic = 1.0 - new_static
        else:  # dynamic
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
            'performance_stats': performance_stats,
            'decision_reason': f"Direction: {direction}, Bridge%: {memory_stats.get('distribution', {}).get('bridge_pct', 'N/A') if memory_stats else 'N/A'}"
        })
        
        # Save everything
        self._save_weights()
        self._save_momentum()
        self._save_history()
        
        print(f"  → Evolved to: static={self.weights['static']:.3f}, dynamic={self.weights['dynamic']:.3f}")
        print(f"  Decision: {direction} (based on data distribution)")
        
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
        dynamic_moves = sum(1 for h in self.history if h.get('momentum', {}).get('last_direction') == 'dynamic')
        reduce_moves = sum(1 for h in self.history if h.get('momentum', {}).get('last_direction') == 'reduce')
        
        return {
            'total_evolutions': len(self.history),
            'current_weights': self.weights,
            'current_specialization': current_spec,
            'specialization_increase': current_spec - initial_spec,
            'dominant_direction': 'static' if static_moves > dynamic_moves else 'dynamic',
            'direction_counts': {
                'static_moves': static_moves,
                'dynamic_moves': dynamic_moves,
                'reduce_moves': reduce_moves
            },
            'momentum_state': self.momentum
        }
        

# Unit tests
if __name__ == "__main__":
    import tempfile
    
    print("🧪 Testing Autonomous Weight Evolution...")
    
    # Test 1: Basic evolution
    print("\n1️⃣ Test: Basic weight evolution")
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = WeightEvolver(data_dir=tmpdir)
        
        # Initial state
        assert evolver.weights['static'] == 0.6
        assert evolver.weights['dynamic'] == 0.4
        
        # Evolve with balanced data - should move toward balance
        memory_stats = {
            'distribution': {
                'logic_pct': 45,
                'symbolic_pct': 45,
                'bridge_pct': 10
            }
        }
        
        evolved = evolver.evolve_weights(run_count=1, memory_stats=memory_stats)
        assert evolved == True, "Should evolve"
        
        # Should move toward balance with balanced data
        new_spec = evolver.get_current_specialization()
        assert new_spec < 0.2, f"Should reduce specialization with balanced data, got {new_spec}"
        
        print("✅ Basic evolution works")
        
    # Test 2: Autonomous learning from data
    print("\n2️⃣ Test: Autonomous learning from data patterns")
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = WeightEvolver(data_dir=tmpdir)
        
        # Simulate logic-heavy data
        memory_stats = {
            'distribution': {
                'logic_pct': 70,
                'symbolic_pct': 20,
                'bridge_pct': 10
            }
        }
        
        # Should learn to specialize toward logic
        target = evolver.calculate_target_specialization(5, memory_stats)
        assert target > 0.3, f"Should target specialization for logic-heavy data, got {target}"
        
        # Simulate symbolic-heavy data
        memory_stats['distribution'] = {
            'logic_pct': 20,
            'symbolic_pct': 70,
            'bridge_pct': 10
        }
        
        target = evolver.calculate_target_specialization(5, memory_stats)
        assert target > 0.3, f"Should target specialization for symbolic-heavy data, got {target}"
        
        print("✅ Autonomous learning works")
        
    # Test 3: Bridge-aware evolution
    print("\n3️⃣ Test: Bridge-aware evolution")
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = WeightEvolver(data_dir=tmpdir)
        
        # Large bridge should trigger rebalancing
        memory_stats = {
            'distribution': {
                'logic_pct': 50,
                'symbolic_pct': 5,
                'bridge_pct': 45  # Large bridge!
            }
        }
        
        target = evolver.calculate_target_specialization(5, memory_stats)
        assert target == 0.0, f"Should target balance with large bridge, got {target}"
        
        print("✅ Bridge-aware evolution works")
        
    # Test 4: Self-correction
    print("\n4️⃣ Test: Self-correction when imbalanced")
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = WeightEvolver(data_dir=tmpdir)
        
        # Force imbalanced weights
        evolver.weights['static'] = 0.85
        evolver.weights['dynamic'] = 0.15
        
        # With very imbalanced distribution opposite to weights
        memory_stats = {
            'distribution': {
                'logic_pct': 10,  # Very little logic
                'symbolic_pct': 80,  # Lots of symbolic
                'bridge_pct': 10
            }
        }
        
        # Should try to reduce specialization
        evolved = evolver.evolve_weights(run_count=1, memory_stats=memory_stats)
        
        # Weights should move toward balance
        assert evolver.weights['static'] < 0.85, "Should reduce static weight"
        assert evolver.weights['dynamic'] > 0.15, "Should increase dynamic weight"
        
        print("✅ Self-correction works")
        
    # Test 5: Working system maintenance
    print("\n5️⃣ Test: Maintaining working system")
    with tempfile.TemporaryDirectory() as tmpdir:
        evolver = WeightEvolver(data_dir=tmpdir)
        
        # Small bridge = system working well
        memory_stats = {
            'distribution': {
                'logic_pct': 60,
                'symbolic_pct': 35,
                'bridge_pct': 5  # Very small bridge
            }
        }
        
        current_spec = evolver.get_current_specialization()
        target = evolver.calculate_target_specialization(5, memory_stats)
        
        # Should maintain or slightly increase current specialization
        assert target >= current_spec, "Should maintain working system"
        
        print("✅ Working system maintenance works")
        
    print("\n✅ All autonomous evolution tests passed!")