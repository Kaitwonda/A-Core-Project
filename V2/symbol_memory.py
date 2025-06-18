# symbol_memory.py - Bridge module for memory_optimizer.py compatibility
"""
Compatibility bridge for the old symbol_memory module.
This provides the interface that memory_optimizer.py expects while
delegating to the unified memory system.
"""

from unified_memory import get_unified_memory

def load_symbol_memory():
    """Load all symbols from the unified memory system"""
    return get_unified_memory().get_all_symbols()

def add_symbol(symbol_token, name, keywords, initial_emotions, example_text,
               origin="emergent", learning_phase=0, resonance_weight=0.5,
               symbol_details_override=None, skip_quarantine_check=False):
    """Add a symbol through the unified memory system"""
    return get_unified_memory().add_symbol(
        symbol_token=symbol_token,
        name=name,
        keywords=keywords,
        initial_emotions=initial_emotions,
        example_text=example_text,
        origin=origin,
        learning_phase=learning_phase,
        resonance_weight=resonance_weight,
        symbol_details_override=symbol_details_override,
        skip_quarantine_check=skip_quarantine_check
    )

def prune_duplicates():
    """Prune duplicate symbols (delegated to unified memory system)"""
    # The unified memory system handles this internally
    # This is a placeholder for backward compatibility
    print("🧹 Symbol pruning handled by unified memory system")
    return True

# Additional compatibility functions if needed
def get_symbol_details(symbol_token):
    """Get details for a specific symbol"""
    return get_unified_memory().get_symbol_details(symbol_token)

def save_symbol_memory():
    """Save symbol memory (handled automatically by unified system)"""
    return True