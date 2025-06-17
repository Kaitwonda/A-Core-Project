# reset_weights.py - Reset adaptive weights to balanced values

import json
from pathlib import Path

def reset_adaptive_weights():
    """Reset the adaptive weights to more balanced values"""
    
    # Path to adaptive config
    config_path = Path("data/adaptive_config.json")
    
    # New balanced configuration
    balanced_config = {
        "link_score_weight_static": 0.5,  # Changed from 0.9
        "link_score_weight_dynamic": 0.5,  # Changed from 0.1
        "last_weight_update": "2025-06-11T18:30:00",
        "update_count": 0,
        "target_specialization": 0.0,  # 0.0 = balanced, not specialized
        "momentum": {
            "static": 0.0,
            "dynamic": 0.0
        }
    }
    
    # Save the new config
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(balanced_config, f, indent=2)
    
    print("✅ Reset adaptive weights to balanced (50/50)")
    print(f"   Static weight: {balanced_config['link_score_weight_static']}")
    print(f"   Dynamic weight: {balanced_config['link_score_weight_dynamic']}")
    print(f"   Target specialization: {balanced_config['target_specialization']}")
    
    # Also check if weight evolution history exists and reset it
    evolution_path = Path("data/weight_evolution_history.json")
    if evolution_path.exists():
        with open(evolution_path, 'w') as f:
            json.dump([], f)
        print("✅ Cleared weight evolution history")
    
    # Reset migration age to allow easier migration
    migration_age_path = Path("data/migration_age.json")
    if migration_age_path.exists():
        with open(migration_age_path, 'w') as f:
            json.dump({"age": 0, "last_updated": "2025-06-11T18:30:00"}, f)
        print("✅ Reset migration age (threshold back to 0.9)")

if __name__ == "__main__":
    reset_adaptive_weights()