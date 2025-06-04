# link_utils.py - Shared utilities for link evaluation

from typing import Tuple

def evaluate_link_with_confidence_gates(logic_score: float, 
                                      symbolic_score: float,
                                      logic_scale: float = 2.0,
                                      sym_scale: float = 1.0) -> Tuple[str, float]:
    """
    Evaluate link scores and determine routing decision with confidence.
    
    This is a pure function that can be used by any component without
    creating circular dependencies.
    
    Args:
        logic_score: Raw logic pathway score
        symbolic_score: Raw symbolic pathway score  
        logic_scale: Scaling factor for logic scores
        sym_scale: Scaling factor for symbolic scores
        
    Returns:
        Tuple of (decision_type, confidence)
        where decision_type is one of:
        - 'FOLLOW_LOGIC'
        - 'FOLLOW_SYMBOLIC' 
        - 'FOLLOW_HYBRID'
    """
    # Apply scales
    scaled_logic = logic_score * logic_scale
    scaled_symbolic = symbolic_score * sym_scale
    
    # Determine decision type
    if scaled_logic > scaled_symbolic * 1.5:
        decision_type = 'FOLLOW_LOGIC'
        confidence = min(1.0, scaled_logic / 10.0)
    elif scaled_symbolic > scaled_logic * 1.5:
        decision_type = 'FOLLOW_SYMBOLIC'
        confidence = min(1.0, scaled_symbolic / 10.0)
    else:
        decision_type = 'FOLLOW_HYBRID'
        confidence = min(1.0, (scaled_logic + scaled_symbolic) / 20.0)
        
    return decision_type, round(confidence, 3)