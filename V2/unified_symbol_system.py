# unified_symbol_system.py - Comprehensive Symbol Processing System
"""
Unified Symbol System - Consolidates all symbol functionality:
- vector_symbols.py: Vector-based symbolic system with semantic vectors
- symbol_discovery.py: Autonomous symbol discovery from text  
- symbol_generator.py: Dynamic symbol generation from context
- symbol_emotion_updater.py: Symbol-emotion mapping and updates
- parser.py (symbol parts): Symbol parsing and lexicon management
- emotion_handler.py (symbol parts): Symbol-emotion interactions
- seed_symbols.json: Initial symbol definitions
- meta_symbols.json: Meta-symbol storage

This replaces 8 symbol-related components with 2 unified files.
"""

import numpy as np
import json
import re
import hashlib
import random
import unicodedata
import logging
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class SymbolMatch:
    """Represents a match between text and a vector symbol"""
    symbol: str
    glyph: str
    similarity: float
    mathematical_resonance: float
    metaphorical_resonance: float
    combined_score: float
    matched_concept: str

@dataclass  
class DiscoveredSymbol:
    """A newly discovered symbol with extracted meaning"""
    symbol: str
    name: str
    context_snippet: str
    mathematical_concepts: List[str]
    metaphorical_concepts: List[str]
    confidence: float
    source_url: str
    discovery_timestamp: str

# ============================================================================
# VECTOR SYMBOL SYSTEM
# ============================================================================

class VectorSymbol:
    """
    A symbol that exists in both mathematical and metaphorical space
    Encoded as semantic vectors for deep pattern matching
    """
    
    def __init__(self, glyph: str, name: str, mathematical_concepts: List[str], 
                 metaphorical_concepts: List[str], learning_phase: int = 1):
        self.glyph = glyph
        self.name = name
        self.mathematical_concepts = mathematical_concepts
        self.metaphorical_concepts = metaphorical_concepts
        self.learning_phase = learning_phase
        
        # Will be populated by VectorSymbolSystem
        self.math_vector = None
        self.metaphor_vector = None
        self.combined_vector = None
        
        # Learning metrics
        self.usage_count = 0
        self.successful_matches = 0
        self.failed_matches = 0
        self.context_adaptations = []
        
    def calculate_resonance_weights(self) -> Tuple[float, float]:
        """Calculate dynamic weights based on learning history"""
        if self.usage_count == 0:
            return 0.5, 0.5  # Balanced starting point
            
        # Adapt based on success patterns
        success_rate = self.successful_matches / self.usage_count if self.usage_count > 0 else 0.5
        
        # If mathematical contexts work better, favor math vector
        if success_rate > 0.7:
            math_weight = 0.6 + (success_rate - 0.7) * 0.4  # Scale up to 1.0
            metaphor_weight = 1.0 - math_weight
        elif success_rate < 0.3:
            # If struggling, balance more evenly 
            math_weight = 0.4
            metaphor_weight = 0.6
        else:
            # Balanced approach for moderate success
            math_weight = 0.5
            metaphor_weight = 0.5
            
        return math_weight, metaphor_weight
    
    def update_from_match(self, similarity: float, was_successful: bool, context: str = ""):
        """Update symbol based on a matching attempt"""
        self.usage_count += 1
        
        if was_successful:
            self.successful_matches += 1
            if context and len(self.context_adaptations) < 10:  # Keep recent contexts
                self.context_adaptations.append({
                    'context': context[:100],
                    'similarity': float(similarity),  # Convert numpy types to Python types
                    'timestamp': datetime.utcnow().isoformat()
                })
        else:
            self.failed_matches += 1
    
    def get_success_rate(self) -> float:
        """Get current success rate"""
        return self.successful_matches / self.usage_count if self.usage_count > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'glyph': self.glyph,
            'name': self.name,
            'mathematical_concepts': self.mathematical_concepts,
            'metaphorical_concepts': self.metaphorical_concepts,
            'learning_phase': self.learning_phase,
            'usage_count': self.usage_count,
            'successful_matches': self.successful_matches,
            'failed_matches': self.failed_matches,
            'context_adaptations': self.context_adaptations
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VectorSymbol':
        """Create from dictionary"""
        symbol = cls(
            glyph=data['glyph'],
            name=data['name'],
            mathematical_concepts=data['mathematical_concepts'],
            metaphorical_concepts=data['metaphorical_concepts'],
            learning_phase=data.get('learning_phase', 1)
        )
        symbol.usage_count = data.get('usage_count', 0)
        symbol.successful_matches = data.get('successful_matches', 0)
        symbol.failed_matches = data.get('failed_matches', 0)
        symbol.context_adaptations = data.get('context_adaptations', [])
        return symbol

class VectorSymbolSystem:
    """
    Core vector-based symbol system that encodes symbols as semantic vectors
    for autonomous learning and deep pattern matching
    """
    
    def __init__(self, encoder_model="all-MiniLM-L6-v2", data_dir="data"):
        self.encoder = SentenceTransformer(encoder_model)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Symbol storage
        self.symbols: Dict[str, VectorSymbol] = {}
        self.learning_file = self.data_dir / "vector_symbol_learning.json"
        
        # Learning parameters
        self.similarity_threshold = 0.3
        self.learning_rate = 0.1
        
        # Initialize with ancient mathematical-metaphorical symbols
        self._initialize_ancient_symbols()
        
        # Load any existing learning data
        self._load_learning_data()
        
        print(f"🔮 Vector Symbol System initialized with {len(self.symbols)} symbols")
    
    def _initialize_ancient_symbols(self):
        """Initialize with ancient symbols that have both mathematical and metaphorical meaning"""
        ancient_symbols = [
            VectorSymbol("Φ", "Golden Ratio", 
                        ["golden ratio", "fibonacci", "proportion", "harmony", "mathematics"],
                        ["divine proportion", "beauty", "perfection", "natural harmony", "cosmic balance"]),
            
            VectorSymbol("Ψ", "Psi Function", 
                        ["wave function", "quantum state", "probability", "complex numbers"],
                        ["consciousness", "mind", "psyche", "hidden knowledge", "soul essence"]),
            
            VectorSymbol("Λ", "Lambda", 
                        ["eigenvalue", "wavelength", "function", "calculus", "parameter"],
                        ["transformation", "change", "evolution", "bridge", "transition"]),
            
            VectorSymbol("Δ", "Delta", 
                        ["change", "difference", "derivative", "increment", "triangle"],
                        ["transformation", "journey", "path", "ascension", "mountain"]),
            
            VectorSymbol("Ω", "Omega", 
                        ["resistance", "frequency", "end", "final", "completion"],
                        ["ultimate", "finality", "wisdom", "ending", "transcendence"]),
            
            VectorSymbol("𓂀", "Eye of Ra", 
                        ["observation", "measurement", "vision", "detection", "monitoring"],
                        ["divine sight", "awareness", "consciousness", "protection", "illumination"]),
            
            VectorSymbol("𓊖", "Spiral of Life", 
                        ["logarithmic spiral", "growth", "sequence", "iteration", "recursion"],
                        ["life force", "evolution", "eternal cycle", "cosmic energy", "renewal"]),
            
            VectorSymbol("⬟", "Hexagon", 
                        ["efficiency", "optimization", "structure", "geometry", "tessellation"],
                        ["community", "cooperation", "natural order", "perfection", "stability"]),
            
            VectorSymbol("∞", "Infinity", 
                        ["limitless", "unbounded", "infinite series", "convergence", "mathematics"],
                        ["eternal", "boundless", "endless potential", "cosmic vastness", "transcendence"]),
            
            VectorSymbol("∅", "Empty Set", 
                        ["null", "zero", "empty", "void", "nothing", "absence"],
                        ["potential", "beginning", "void", "space for creation", "primordial silence"])
        ]
        
        for symbol in ancient_symbols:
            self.symbols[symbol.glyph] = symbol
            
        # Encode all symbols as vectors
        self._encode_all_symbols()
    
    def _encode_all_symbols(self):
        """Encode all symbols as semantic vectors"""
        for symbol in self.symbols.values():
            # Create combined concept text for encoding
            math_text = " ".join(symbol.mathematical_concepts)
            metaphor_text = " ".join(symbol.metaphorical_concepts)
            
            # Generate vectors
            symbol.math_vector = self.encoder.encode(math_text)
            symbol.metaphor_vector = self.encoder.encode(metaphor_text)
            
            # Create combined vector (weighted average)
            math_weight, metaphor_weight = symbol.calculate_resonance_weights()
            symbol.combined_vector = (
                math_weight * symbol.math_vector + 
                metaphor_weight * symbol.metaphor_vector
            )
    
    def add_symbol(self, glyph: str, name: str, mathematical_concepts: List[str], 
                   metaphorical_concepts: List[str], learning_phase: int = 1) -> VectorSymbol:
        """Add a new symbol to the system"""
        symbol = VectorSymbol(glyph, name, mathematical_concepts, metaphorical_concepts, learning_phase)
        self.symbols[glyph] = symbol
        
        # Encode the new symbol
        math_text = " ".join(mathematical_concepts)
        metaphor_text = " ".join(metaphorical_concepts)
        
        symbol.math_vector = self.encoder.encode(math_text)
        symbol.metaphor_vector = self.encoder.encode(metaphor_text)
        
        math_weight, metaphor_weight = symbol.calculate_resonance_weights()
        symbol.combined_vector = (
            math_weight * symbol.math_vector + 
            metaphor_weight * symbol.metaphor_vector
        )
        
        print(f"🌟 Added new symbol: {glyph} ({name})")
        return symbol
    
    def find_matches(self, text: str, top_k: int = 5, threshold: float = None) -> List[SymbolMatch]:
        """Find symbols that resonate with the given text"""
        if threshold is None:
            threshold = self.similarity_threshold
            
        # Encode the input text
        text_vector = self.encoder.encode(text)
        
        matches = []
        
        for symbol in self.symbols.values():
            if symbol.combined_vector is None:
                continue
                
            # Calculate similarities
            math_similarity = float(np.dot(text_vector, symbol.math_vector) / (
                np.linalg.norm(text_vector) * np.linalg.norm(symbol.math_vector)
            ))
            
            metaphor_similarity = float(np.dot(text_vector, symbol.metaphor_vector) / (
                np.linalg.norm(text_vector) * np.linalg.norm(symbol.metaphor_vector)
            ))
            
            combined_similarity = float(np.dot(text_vector, symbol.combined_vector) / (
                np.linalg.norm(text_vector) * np.linalg.norm(symbol.combined_vector)
            ))
            
            if combined_similarity >= threshold:
                # Determine which concept matched best
                best_concept = "mathematical" if math_similarity > metaphor_similarity else "metaphorical"
                
                match = SymbolMatch(
                    symbol=symbol.glyph,
                    glyph=symbol.glyph,
                    similarity=combined_similarity,
                    mathematical_resonance=math_similarity,
                    metaphorical_resonance=metaphor_similarity,
                    combined_score=combined_similarity,
                    matched_concept=best_concept
                )
                
                matches.append(match)
        
        # Sort by combined score
        matches.sort(key=lambda x: x.combined_score, reverse=True)
        
        return matches[:top_k]
    
    def learn_from_routing(self, text: str, decision_type: str, was_successful: bool):
        """Learn from routing decisions to improve symbol matching"""
        matches = self.find_matches(text, top_k=3, threshold=0.1)  # Lower threshold for learning
        
        for match in matches:
            symbol = self.symbols.get(match.symbol)
            if symbol:
                symbol.update_from_match(match.combined_score, was_successful, text[:100])
        
        # Re-encode symbols that have been updated
        updated_symbols = [s for s in self.symbols.values() if s.usage_count > 0]
        if updated_symbols:
            self._encode_updated_symbols(updated_symbols)
        
        # Save learning progress
        self._save_learning_data()
    
    def _encode_updated_symbols(self, symbols: List[VectorSymbol]):
        """Re-encode symbols that have learning updates"""
        for symbol in symbols:
            math_weight, metaphor_weight = symbol.calculate_resonance_weights()
            symbol.combined_vector = (
                math_weight * symbol.math_vector + 
                metaphor_weight * symbol.metaphor_vector
            )
    
    def route_with_unified_weights(self, text: str) -> Tuple[str, float, Dict]:
        """Determine routing based on symbol resonance"""
        matches = self.find_matches(text, top_k=5)
        
        if not matches:
            return "FOLLOW_LOGIC", 0.1, {"reason": "no_symbol_matches"}
        
        # Calculate symbolic strength
        total_resonance = sum(match.combined_score for match in matches)
        avg_resonance = total_resonance / len(matches)
        
        # Determine routing
        if avg_resonance > 0.7:
            return "FOLLOW_SYMBOLIC", avg_resonance, {
                "matched_symbols": [{"glyph": m.symbol, "score": m.combined_score} for m in matches[:3]],
                "average_resonance": avg_resonance
            }
        elif avg_resonance > 0.4:
            return "FOLLOW_HYBRID", avg_resonance, {
                "matched_symbols": [{"glyph": m.symbol, "score": m.combined_score} for m in matches[:3]],
                "average_resonance": avg_resonance
            }
        else:
            return "FOLLOW_LOGIC", avg_resonance, {
                "matched_symbols": [{"glyph": m.symbol, "score": m.combined_score} for m in matches[:3]],
                "average_resonance": avg_resonance
            }
    
    def get_symbol_insights(self) -> Dict:
        """Get insights about the symbol system's learning"""
        total_symbols = len(self.symbols)
        active_symbols = sum(1 for s in self.symbols.values() if s.usage_count > 0)
        
        # Get performance data
        symbol_performance = []
        for symbol in self.symbols.values():
            if symbol.usage_count > 0:
                symbol_performance.append({
                    'glyph': symbol.glyph,
                    'name': symbol.name,
                    'usage_count': symbol.usage_count,
                    'success_rate': symbol.get_success_rate(),
                    'learning_phase': symbol.learning_phase
                })
        
        # Sort by usage
        symbol_performance.sort(key=lambda x: x['usage_count'], reverse=True)
        
        return {
            'total_symbols': total_symbols,
            'active_symbols': active_symbols,
            'current_threshold': self.similarity_threshold,
            'symbol_performance': symbol_performance
        }
    
    def _save_learning_data(self):
        """Save learning data to disk"""
        data = {'system_threshold': self.similarity_threshold}
        
        for glyph, symbol in self.symbols.items():
            if symbol.usage_count > 0:  # Only save symbols with usage
                data[glyph] = symbol.to_dict()
        
        with open(self.learning_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_learning_data(self):
        """Load learning data from disk"""
        if not self.learning_file.exists():
            return
            
        try:
            with open(self.learning_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'system_threshold' in data:
                self.similarity_threshold = data['system_threshold']
            
            # Load symbol learning data
            for glyph, symbol_data in data.items():
                if glyph == 'system_threshold':
                    continue
                    
                if glyph in self.symbols:
                    symbol = self.symbols[glyph]
                    symbol.usage_count = symbol_data.get('usage_count', 0)
                    symbol.successful_matches = symbol_data.get('successful_matches', 0)
                    symbol.failed_matches = symbol_data.get('failed_matches', 0)
                    symbol.context_adaptations = symbol_data.get('context_adaptations', [])
            
        except Exception as e:
            print(f"⚠️ Error loading learning data: {e}")

# ============================================================================
# SYMBOL DISCOVERY ENGINE
# ============================================================================

class SymbolDiscoveryEngine:
    """
    Autonomous symbol discovery system that learns new symbols from reading
    """
    
    def __init__(self, encoder_model="all-MiniLM-L6-v2", data_dir="data"):
        self.encoder = SentenceTransformer(encoder_model)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Symbol detection patterns
        self._init_detection_patterns()
        
        # Mathematical and metaphorical concept extractors
        self._init_concept_extractors()
        
        # Discovery history setup
        self.discoveries_file = self.data_dir / "symbol_discoveries.json"
        self.discovery_history: List[Dict] = []
        
        # Known symbols to avoid rediscovery
        self.known_symbols: Set[str] = set()
        
        # Load existing discoveries
        self._load_discoveries()
        
        print(f"🔍 Symbol Discovery Engine initialized with {len(self.discovery_history)} known discoveries")
    
    def _init_detection_patterns(self):
        """Initialize patterns for detecting symbol explanations"""
        self.symbol_patterns = [
            # Direct explanations
            r'(?:the\s+)?([^a-zA-Z\s]{1,3})\s+(?:symbol|sign|character|glyph)\s+(?:represents?|means?|signifies?|denotes?)\s+([^.!?]+)',
            r'([^a-zA-Z\s]{1,3})\s+(?:is|represents?|means?|stands for|symbolizes?)\s+([^.!?]+)',
            
            # Mathematical context
            r'(?:using|with|the)\s+([^a-zA-Z\s]{1,3})\s+(?:function|operator|constant|number|ratio|formula)\s+([^.!?]+)',
            r'([^a-zA-Z\s]{1,3})\s+(?:in mathematics|in physics|in geometry)\s+(?:represents?|means?|is)\s+([^.!?]+)',
            
            # Named symbols
            r'(?:the\s+)?([a-zA-Z]+\s+(?:ratio|number|constant|symbol))\s+\(([^a-zA-Z\s]{1,3})\)\s+([^.!?]+)',
            r'([^a-zA-Z\s]{1,3})\s+\((?:the\s+)?([a-zA-Z]+\s+(?:ratio|number|constant|symbol))\)\s+([^.!?]+)'
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.symbol_patterns]
    
    def _init_concept_extractors(self):
        """Initialize concept extraction patterns"""
        self.math_indicators = {
            'equation', 'formula', 'function', 'derivative', 'integral', 'limit', 'theorem', 'proof',
            'algorithm', 'computation', 'calculation', 'mathematics', 'geometry', 'algebra', 'calculus',
            'number', 'ratio', 'proportion', 'sequence', 'series', 'matrix', 'vector', 'scalar',
            'probability', 'statistics', 'physics', 'engineering', 'science', 'measurement'
        }
        
        self.metaphor_indicators = {
            'represents', 'symbolizes', 'embodies', 'signifies', 'meaning', 'spiritual', 'divine',
            'sacred', 'mystical', 'philosophical', 'wisdom', 'consciousness', 'soul', 'essence',
            'journey', 'transformation', 'evolution', 'growth', 'harmony', 'balance', 'unity',
            'cosmic', 'universal', 'eternal', 'infinite', 'transcendent', 'enlightenment'
        }
    
    def discover_symbols_from_text(self, text: str, source_url: str = "unknown") -> List[DiscoveredSymbol]:
        """Main discovery function - finds new symbols and their meanings in text"""
        discoveries = []
        
        # Split text into sentences for better pattern matching
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Try each pattern
            for pattern in self.compiled_patterns:
                matches = pattern.finditer(sentence)
                
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        symbol_candidate, explanation = groups[0].strip(), groups[-1].strip()
                        
                        # Validate symbol candidate
                        if self._is_valid_symbol(symbol_candidate):
                            discovery = self._create_discovery(
                                symbol_candidate, explanation, sentence, source_url
                            )
                            
                            if discovery and discovery.symbol not in self.known_symbols:
                                discoveries.append(discovery)
                                self.known_symbols.add(discovery.symbol)
        
        # Save new discoveries
        if discoveries:
            self._save_discoveries(discoveries)
            
        return discoveries
    
    def _is_valid_symbol(self, candidate: str) -> bool:
        """Check if a candidate string is a valid symbol"""
        if not candidate or len(candidate) > 5:
            return False
            
        # Check for Unicode symbols, mathematical symbols, or special characters
        for char in candidate:
            category = unicodedata.category(char)
            if category.startswith(('Sm', 'So', 'Ps', 'Pe')) or ord(char) > 127:
                return True
                
        return False
    
    def _create_discovery(self, symbol: str, explanation: str, context: str, source_url: str) -> Optional[DiscoveredSymbol]:
        """Create a discovery object from extracted information"""
        # Extract concepts
        math_concepts = self._extract_mathematical_concepts(explanation)
        metaphor_concepts = self._extract_metaphorical_concepts(explanation)
        
        # Calculate confidence based on concept richness and context
        confidence = self._calculate_confidence(math_concepts, metaphor_concepts, context)
        
        if confidence < 0.3:  # Minimum confidence threshold
            return None
        
        # Generate a name for the symbol
        name = self._generate_symbol_name(symbol, explanation, math_concepts, metaphor_concepts)
        
        return DiscoveredSymbol(
            symbol=symbol,
            name=name,
            context_snippet=context[:200],
            mathematical_concepts=math_concepts,
            metaphorical_concepts=metaphor_concepts,
            confidence=confidence,
            source_url=source_url,
            discovery_timestamp=datetime.utcnow().isoformat()
        )
    
    def _extract_mathematical_concepts(self, text: str) -> List[str]:
        """Extract mathematical concepts from text"""
        text_lower = text.lower()
        concepts = []
        
        for indicator in self.math_indicators:
            if indicator in text_lower:
                concepts.append(indicator)
        
        # Add specific mathematical terms found
        math_terms = re.findall(r'\b(?:formula|equation|function|derivative|integral|theorem|algorithm)\w*\b', text_lower)
        concepts.extend(math_terms)
        
        return list(set(concepts))  # Remove duplicates
    
    def _extract_metaphorical_concepts(self, text: str) -> List[str]:
        """Extract metaphorical concepts from text"""
        text_lower = text.lower()
        concepts = []
        
        for indicator in self.metaphor_indicators:
            if indicator in text_lower:
                concepts.append(indicator)
        
        # Add emotional/spiritual terms
        metaphor_terms = re.findall(r'\b(?:spiritual|divine|sacred|wisdom|consciousness|harmony|balance|transformation)\w*\b', text_lower)
        concepts.extend(metaphor_terms)
        
        return list(set(concepts))  # Remove duplicates
    
    def _calculate_confidence(self, math_concepts: List[str], metaphor_concepts: List[str], context: str) -> float:
        """Calculate confidence score for a discovery"""
        # Base confidence from concept richness
        concept_score = min((len(math_concepts) + len(metaphor_concepts)) / 10.0, 1.0)
        
        # Context quality (length and structure)
        context_score = min(len(context) / 200.0, 1.0)
        
        # Balanced concepts bonus
        if math_concepts and metaphor_concepts:
            balance_bonus = 0.2
        else:
            balance_bonus = 0.0
        
        # Specific quality indicators
        quality_indicators = ['represents', 'means', 'symbolizes', 'denotes', 'function', 'ratio']
        quality_score = sum(0.1 for indicator in quality_indicators if indicator in context.lower())
        quality_score = min(quality_score, 0.3)
        
        total_confidence = concept_score * 0.4 + context_score * 0.3 + balance_bonus + quality_score
        
        return min(total_confidence, 1.0)
    
    def _generate_symbol_name(self, symbol: str, explanation: str, math_concepts: List[str], metaphor_concepts: List[str]) -> str:
        """Generate a descriptive name for the symbol"""
        # Try to extract name from explanation
        name_patterns = [
            r'(?:the\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+(?:symbol|sign|character|glyph|function|constant|ratio)',
            r'called\s+(?:the\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
            r'known\s+as\s+(?:the\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)*)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, explanation, re.IGNORECASE)
            if match:
                name = match.group(1).strip().title()
                if len(name) > 2 and name.lower() not in {'the', 'this', 'that', 'symbol'}:
                    return name
        
        # Generate name from concepts
        if math_concepts and metaphor_concepts:
            return f"{math_concepts[0].title()} {metaphor_concepts[0].title()}"
        elif math_concepts:
            return f"{math_concepts[0].title()} Symbol"
        elif metaphor_concepts:
            return f"{metaphor_concepts[0].title()} Symbol"
        else:
            return f"Symbol {symbol}"
    
    def get_discovery_insights(self) -> Dict:
        """Get insights about symbol discoveries"""
        if not self.discovery_history:
            return {
                'total_discoveries': 0,
                'average_confidence': 0.0,
                'symbols_by_confidence': []
            }
        
        total_discoveries = len(self.discovery_history)
        avg_confidence = sum(d['confidence'] for d in self.discovery_history) / total_discoveries
        
        # Sort by confidence
        sorted_discoveries = sorted(self.discovery_history, key=lambda x: x['confidence'], reverse=True)
        
        # High confidence discoveries
        high_confidence = [d for d in self.discovery_history if d['confidence'] > 0.7]
        
        return {
            'total_discoveries': total_discoveries,
            'average_confidence': avg_confidence,
            'high_confidence_discoveries': len(high_confidence),
            'symbols_by_confidence': sorted_discoveries
        }
    
    def _save_discoveries(self, new_discoveries: List[DiscoveredSymbol]):
        """Save new discoveries to file"""
        # Convert to dictionaries
        new_data = [asdict(discovery) for discovery in new_discoveries]
        
        # Add to history
        self.discovery_history.extend(new_data)
        
        # Save to file
        with open(self.discoveries_file, 'w', encoding='utf-8') as f:
            json.dump(self.discovery_history, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(new_discoveries)} new symbol discoveries")
    
    def _load_discoveries(self):
        """Load existing discoveries from file"""
        if not self.discoveries_file.exists():
            return
            
        try:
            with open(self.discoveries_file, 'r', encoding='utf-8') as f:
                self.discovery_history = json.load(f)
            
            # Populate known symbols set
            for discovery in self.discovery_history:
                self.known_symbols.add(discovery['symbol'])
                
        except Exception as e:
            print(f"⚠️ Error loading discoveries: {e}")
            self.discovery_history = []

# ============================================================================
# SYMBOL GENERATOR
# ============================================================================

class SymbolGenerator:
    """Dynamic symbol generation from context and emotions"""
    
    # Expanded symbol pool including various Unicode categories
    SYMBOL_TOKEN_POOL = [
        # Emojis
        "🌀", "💡", "🧩", "🔗", "🌐", "⚖️", "🗝️", "🌱", "⚙️", "🧭",
        "📜", "🧱", "💬", "👁️‍🗨️", "🧠", "🎨", "🎼", "🎭", "🌌", "⏳",
        "🌠", "✨", "❓", "❗", "♾️", "🕳️", "💠", "💎", "🧬", "🔭",
        "🔬", "🕊️", "🪞", "🛡️", "🕰️", "🌍", "💭", "👁️", "👂",
        "👣", "🌳", "🌲", "🌿", "🍄", "🌊", "💧", "🌬️", "💨", "🌪️",
        "🌋", "⛰️", "☀️", "🌙", "🌕", "🌑", "🪐", "📚", "🎲", "🔮", 
        "⚗️", "🕮", "🕯️", "⌛", "⚖", "⚓", "⚛️", "⚜️", "⚙", "♾",
        
        # Greek Letters
        "Δ", "Φ", "Ψ", "Ω", "Σ", "Π", "Λ", "Θ", "Ξ", "α", "β", "γ", "δ", "ε", "φ", "ψ", "ω",
        
        # Geometric Shapes & Mathematical Symbols  
        "○", "●", "□", "■", "△", "▲", "▽", "▼", "◇", "♦", "→", "←", "↔", "⇒", "⇔", "∴", "∵",
        "+", "-", "*", "/", "=", "<", ">", "≠", "≈", "≡", "∑", "∫", "√",
        
        # Celestial and Alchemical
        "☉", "☽", "☿", "♀", "♂", "♃", "♄", "♅", "♆", "♇",  # Planets
        "♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓",  # Zodiac
        "🜁", "🜂", "🜃", "🜄",  # Alchemical elements
    ]
    
    def __init__(self):
        self.generation_history = []
        
    def generate_symbol_from_context(self, text: str, keywords: List[str], 
                                   emotions_list_of_tuples: List[Tuple[str, float]]) -> Optional[Dict]:
        """
        Generate a new symbol based on context, keywords, and emotions
        
        Args:
            text: The context text
            keywords: List of keywords from the text
            emotions_list_of_tuples: List of (emotion_str, score_float) from emotion detection
            
        Returns:
            Dict representing the new symbol, or None if no basis for generation
        """
        if not keywords and not emotions_list_of_tuples:
            return None
        
        # Build name components
        name_parts = []
        if keywords:
            name_parts.extend([kw.title() for kw in keywords[:2]])  # Use up to 2 top keywords
        
        if emotions_list_of_tuples:
            primary_emotion = emotions_list_of_tuples[0][0] if emotions_list_of_tuples else "neutral"
            if primary_emotion not in [part.lower() for part in name_parts]:
                name_parts.append(primary_emotion.title())
        
        # Generate symbol name
        if len(name_parts) == 1:
            symbol_name = name_parts[0]
        else:
            symbol_name = " ".join(name_parts[:2])  # Limit to 2 parts for readability
        
        # Select symbol token based on context and emotions
        symbol_token = self._select_contextual_symbol(text, keywords, emotions_list_of_tuples)
        
        # Calculate resonance weight based on emotion intensity and keyword relevance
        resonance_weight = self._calculate_resonance_weight(emotions_list_of_tuples, keywords)
        
        # Determine origin
        origin = "generated_from_context"
        if any(emotion[1] > 0.8 for emotion in emotions_list_of_tuples):
            origin = "high_emotion_generated"
        elif len(keywords) > 3:
            origin = "keyword_rich_generated"
        
        # Create symbol definition
        symbol_definition = {
            "symbol": symbol_token,
            "name": symbol_name,
            "keywords": keywords[:5],  # Limit keywords
            "emotions": emotions_list_of_tuples[:3],  # Top 3 emotions
            "origin": origin,
            "resonance_weight": resonance_weight,
            "generated_from": text[:100] + "..." if len(text) > 100 else text,
            "generation_timestamp": datetime.utcnow().isoformat()
        }
        
        # Track generation
        self.generation_history.append(symbol_definition)
        
        print(f"🎭 Generated symbol: {symbol_token} ({symbol_name})")
        return symbol_definition
    
    def _select_contextual_symbol(self, text: str, keywords: List[str], 
                                emotions: List[Tuple[str, float]]) -> str:
        """Select an appropriate symbol based on context"""
        text_lower = text.lower()
        
        # Emotion-based selection
        if emotions:
            primary_emotion = emotions[0][0].lower()
            emotion_symbols = {
                'joy': ['🌟', '☀️', '✨', '🌠'],
                'anger': ['🔥', '⚡', '🌋', '△'],  
                'sadness': ['💧', '🌊', '☽', '∅'],
                'fear': ['🕳️', '🌑', '👁️', 'Ψ'],
                'surprise': ['❓', '💡', '🔮', 'Ω'],
                'love': ['♾️', '💎', '🌹', 'Φ'],
                'trust': ['🛡️', '⚖️', '🗝️', 'Λ']
            }
            
            if primary_emotion in emotion_symbols:
                return random.choice(emotion_symbols[primary_emotion])
        
        # Keyword-based selection  
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if any(math_term in keyword_lower for math_term in ['math', 'number', 'calculate', 'equation']):
                return random.choice(['Δ', 'Σ', 'Π', 'Φ', '∑', '∫', '√'])
            elif any(sci_term in keyword_lower for sci_term in ['science', 'physics', 'chemistry']):
                return random.choice(['⚛️', '🔬', '🧬', 'Ψ', '🜁', '🜂'])
            elif any(nature_term in keyword_lower for nature_term in ['nature', 'life', 'growth', 'organic']):
                return random.choice(['🌱', '🌿', '🍄', '🌳', '🌊', '☀️'])
            elif any(tech_term in keyword_lower for tech_term in ['technology', 'digital', 'compute', 'system']):
                return random.choice(['⚙️', '🔗', '🌐', '💾', '🧩', '●'])
        
        # Default random selection
        return random.choice(self.SYMBOL_TOKEN_POOL)
    
    def _calculate_resonance_weight(self, emotions: List[Tuple[str, float]], keywords: List[str]) -> float:
        """Calculate resonance weight based on emotional intensity and keyword richness"""
        base_weight = 0.5
        
        # Emotion intensity contribution
        if emotions:
            max_emotion_score = max(score for _, score in emotions)
            emotion_contribution = min(max_emotion_score * 0.3, 0.3)
            base_weight += emotion_contribution
        
        # Keyword richness contribution
        keyword_contribution = min(len(keywords) * 0.05, 0.2)
        base_weight += keyword_contribution
        
        # Ensure within bounds
        return min(max(base_weight, 0.1), 1.0)
    
    def get_generation_stats(self) -> Dict:
        """Get statistics about symbol generation"""
        if not self.generation_history:
            return {'total_generated': 0, 'average_weight': 0.0, 'origins': {}}
        
        total = len(self.generation_history)
        avg_weight = sum(s['resonance_weight'] for s in self.generation_history) / total
        
        # Count by origin
        origins = defaultdict(int)
        for symbol in self.generation_history:
            origins[symbol['origin']] += 1
        
        return {
            'total_generated': total,
            'average_weight': avg_weight,
            'origins': dict(origins)
        }

# ============================================================================
# SYMBOL-EMOTION INTEGRATION
# ============================================================================

class SymbolEmotionMapper:
    """Manages relationships between symbols and emotions"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.emotion_map_file = self.data_dir / "symbol_emotion_map.json"
        
        # Load existing emotion map
        self.emotion_map = self._load_emotion_map()
        
    def _load_emotion_map(self) -> Dict:
        """Load symbol emotion map from file"""
        if not self.emotion_map_file.exists():
            return {}
            
        try:
            with open(self.emotion_map_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            print(f"⚠️ Emotion map file corrupted. Starting fresh.")
            return {}
    
    def _save_emotion_map(self):
        """Save emotion map to file"""
        with open(self.emotion_map_file, 'w', encoding='utf-8') as f:
            json.dump(self.emotion_map, f, indent=2, ensure_ascii=False)
    
    def update_symbol_emotions(self, matched_symbols_weighted: List[Dict], 
                             verified_emotions: List[Tuple[str, float]]):
        """
        Update symbol-emotion associations based on matches and detected emotions
        
        Args:
            matched_symbols_weighted: List of dicts with 'symbol', 'final_weight', etc.
            verified_emotions: List of (emotion_str, score_float) tuples
        """
        if not matched_symbols_weighted or not verified_emotions:
            return
        
        for symbol_match in matched_symbols_weighted:
            symbol = symbol_match.get('symbol')
            if not symbol:
                continue
                
            # Initialize symbol in map if not present
            if symbol not in self.emotion_map:
                self.emotion_map[symbol] = {
                    'total_occurrences': 0,
                    'emotion_associations': {},
                    'average_weights': {}
                }
            
            symbol_data = self.emotion_map[symbol]
            symbol_data['total_occurrences'] += 1
            
            # Update emotion associations
            for emotion, score in verified_emotions:
                if emotion not in symbol_data['emotion_associations']:
                    symbol_data['emotion_associations'][emotion] = []
                
                symbol_data['emotion_associations'][emotion].append(score)
                
                # Keep only recent associations (last 20)
                if len(symbol_data['emotion_associations'][emotion]) > 20:
                    symbol_data['emotion_associations'][emotion] = symbol_data['emotion_associations'][emotion][-20:]
                
                # Update average
                avg_score = sum(symbol_data['emotion_associations'][emotion]) / len(symbol_data['emotion_associations'][emotion])
                symbol_data['average_weights'][emotion] = avg_score
        
        # Save updated map
        self._save_emotion_map()
    
    def get_symbol_emotion_profile(self, symbol: str) -> Dict:
        """Get emotion profile for a specific symbol"""
        if symbol not in self.emotion_map:
            return {}
            
        return self.emotion_map[symbol]
    
    def get_symbols_by_emotion(self, emotion: str, min_score: float = 0.5) -> List[Tuple[str, float]]:
        """Get symbols associated with a specific emotion"""
        results = []
        
        for symbol, data in self.emotion_map.items():
            if emotion in data.get('average_weights', {}):
                score = data['average_weights'][emotion]
                if score >= min_score:
                    results.append((symbol, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_emotion_map_stats(self) -> Dict:
        """Get statistics about the emotion map"""
        if not self.emotion_map:
            return {'total_symbols': 0, 'total_associations': 0, 'top_emotions': []}
        
        total_symbols = len(self.emotion_map)
        total_associations = sum(data['total_occurrences'] for data in self.emotion_map.values())
        
        # Count emotions across all symbols
        emotion_counts = defaultdict(int)
        for symbol_data in self.emotion_map.values():
            for emotion in symbol_data.get('average_weights', {}):
                emotion_counts[emotion] += 1
        
        # Top emotions
        top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_symbols': total_symbols,
            'total_associations': total_associations,
            'top_emotions': top_emotions,
            'emotions_tracked': len(emotion_counts)
        }

# ============================================================================
# SYMBOL PARSING AND LEXICON
# ============================================================================

class SymbolParser:
    """Symbol parsing and lexicon management"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Load lexicons
        self.seed_symbols = self._load_seed_symbols()
        self.meta_symbols = self._load_meta_symbols()
        
    def _load_seed_symbols(self) -> Dict:
        """Load seed symbols from JSON file"""
        seed_file = self.data_dir / "seed_symbols.json"
        if not seed_file.exists():
            return {}
            
        try:
            with open(seed_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Seed symbols file corrupted.")
            return {}
    
    def _load_meta_symbols(self) -> Dict:
        """Load meta symbols from JSON file"""
        meta_file = self.data_dir / "meta_symbols.json"
        if not meta_file.exists():
            return {}
            
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Meta symbols file corrupted.")
            return {}
    
    def build_active_lexicon(self, max_phase: int = 999) -> Dict:
        """Build active symbol lexicon up to specified phase"""
        active_lexicon = {}
        
        # Add seed symbols within phase limit
        if self.seed_symbols:
            for token, details in self.seed_symbols.items():
                if details.get("learning_phase", 0) <= max_phase:
                    active_lexicon[token] = details
        
        # Add meta symbols within phase limit
        if self.meta_symbols:
            for token, details in self.meta_symbols.items():
                if details.get("learning_phase", 0) <= max_phase:
                    active_lexicon[token] = details
        
        return active_lexicon
    
    def extract_symbols_from_text(self, text: str, current_lexicon: Dict) -> List[Dict]:
        """Extract symbols found in text using current lexicon"""
        extracted = []
        
        # Simple token matching
        for token_symbol, details in current_lexicon.items():
            if token_symbol in text:
                matched_kw = f"direct_match:{token_symbol}"
                extracted.append({
                    "symbol": token_symbol,
                    "name": details.get("name", "Unknown Symbol"),
                    "matched_keyword": matched_kw
                })
        
        return extracted
    
    def parse_with_emotion(self, text: str, current_lexicon: Dict) -> Dict:
        """Parse text for symbols and return structured result"""
        extracted_symbols = self.extract_symbols_from_text(text, current_lexicon)
        
        # Calculate weights based on symbol relevance
        symbols_weighted = []
        for symbol_info in extracted_symbols:
            # Simple weight calculation - can be enhanced
            base_weight = 0.5
            
            # Boost for direct matches
            if "direct_match" in symbol_info.get("matched_keyword", ""):
                base_weight += 0.3
            
            symbol_info["final_weight"] = min(base_weight, 1.0)
            symbols_weighted.append(symbol_info)
        
        return {
            "matched_symbols_weighted": symbols_weighted,
            "total_symbolic_weight": sum(s["final_weight"] for s in symbols_weighted),
            "symbols_found_count": len(symbols_weighted)
        }

# ============================================================================
# UNIFIED SYMBOL SYSTEM
# ============================================================================

class UnifiedSymbolSystem:
    """
    Unified Symbol System that integrates all symbol functionality
    """
    
    def __init__(self, encoder_model="all-MiniLM-L6-v2", data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize all subsystems
        self.vector_symbols = VectorSymbolSystem(encoder_model, data_dir)
        self.discovery_engine = SymbolDiscoveryEngine(encoder_model, data_dir)
        self.generator = SymbolGenerator()
        self.emotion_mapper = SymbolEmotionMapper(data_dir)
        self.parser = SymbolParser(data_dir)
        
        print(f"🌟 Unified Symbol System initialized")
    
    def process_text_for_symbols(self, text: str, source_url: str = "unknown", 
                                max_phase: int = 999) -> Dict:
        """
        Complete symbol processing pipeline for input text
        
        Returns comprehensive analysis including matches, discoveries, and new generations
        """
        results = {
            'vector_matches': [],
            'discovered_symbols': [], 
            'generated_symbols': [],
            'parser_matches': [],
            'routing_decision': None,
            'total_symbolic_score': 0.0
        }
        
        # 1. Vector symbol matching
        vector_matches = self.vector_symbols.find_matches(text, top_k=5)
        results['vector_matches'] = [
            {
                'symbol': m.symbol,
                'glyph': m.glyph,
                'score': m.combined_score,
                'math_resonance': m.mathematical_resonance,
                'metaphor_resonance': m.metaphorical_resonance,
                'matched_concept': m.matched_concept
            }
            for m in vector_matches
        ]
        
        # 2. Symbol discovery from text
        discovered = self.discovery_engine.discover_symbols_from_text(text, source_url)
        results['discovered_symbols'] = [
            {
                'symbol': d.symbol,
                'name': d.name,
                'confidence': d.confidence,
                'math_concepts': d.mathematical_concepts,
                'metaphor_concepts': d.metaphorical_concepts
            }
            for d in discovered
        ]
        
        # 3. Parser-based symbol extraction
        active_lexicon = self.parser.build_active_lexicon(max_phase)
        parser_result = self.parser.parse_with_emotion(text, active_lexicon)
        results['parser_matches'] = parser_result['matched_symbols_weighted']
        
        # 4. Calculate total symbolic score
        vector_score = sum(m.combined_score for m in vector_matches)
        parser_score = parser_result['total_symbolic_weight']
        discovery_score = sum(d.confidence for d in discovered)
        
        results['total_symbolic_score'] = vector_score + parser_score + discovery_score * 0.5
        
        # 5. Routing decision
        routing_decision, confidence, metadata = self.vector_symbols.route_with_unified_weights(text)
        results['routing_decision'] = {
            'decision': routing_decision,
            'confidence': confidence,
            'metadata': metadata
        }
        
        return results
    
    def learn_from_interaction(self, text: str, decision_type: str, was_successful: bool, 
                             emotions: List[Tuple[str, float]] = None):
        """Learn from user interaction to improve symbol system"""
        
        # 1. Update vector symbol learning
        self.vector_symbols.learn_from_routing(text, decision_type, was_successful)
        
        # 2. Update emotion mappings if emotions provided
        if emotions:
            # Get current matches to update emotion associations
            active_lexicon = self.parser.build_active_lexicon()
            parser_result = self.parser.parse_with_emotion(text, active_lexicon)
            
            if parser_result['matched_symbols_weighted']:
                self.emotion_mapper.update_symbol_emotions(
                    parser_result['matched_symbols_weighted'], 
                    emotions
                )
    
    def generate_contextual_symbol(self, text: str, keywords: List[str] = None, 
                                 emotions: List[Tuple[str, float]] = None) -> Optional[Dict]:
        """Generate a new symbol based on context"""
        if keywords is None:
            keywords = []
            
        if emotions is None:
            emotions = []
            
        return self.generator.generate_symbol_from_context(text, keywords, emotions)
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics about the symbol system"""
        return {
            'vector_symbols': self.vector_symbols.get_symbol_insights(),
            'discoveries': self.discovery_engine.get_discovery_insights(),
            'generations': self.generator.get_generation_stats(),
            'emotion_mapping': self.emotion_mapper.get_emotion_map_stats(),
            'system_status': {
                'total_seed_symbols': len(self.parser.seed_symbols),
                'total_meta_symbols': len(self.parser.meta_symbols),
                'data_directory': str(self.data_dir)
            }
        }
    
    def integrate_discovered_symbol_to_vector_system(self, discovered: DiscoveredSymbol) -> VectorSymbol:
        """Integrate a discovered symbol into the vector symbol system"""
        vector_symbol = self.vector_symbols.add_symbol(
            glyph=discovered.symbol,
            name=discovered.name,
            mathematical_concepts=discovered.mathematical_concepts,
            metaphorical_concepts=discovered.metaphorical_concepts,
            learning_phase=1  # Start at phase 1 for discovered symbols
        )
        
        print(f"🔗 Integrated discovery '{discovered.symbol}' into vector system")
        return vector_symbol
    
    def save_all_data(self):
        """Save all symbol system data"""
        self.vector_symbols._save_learning_data()
        self.emotion_mapper._save_emotion_map()
        print("💾 All symbol system data saved")


# ============================================================================
# BACKWARD COMPATIBILITY AND CONVENIENCE FUNCTIONS
# ============================================================================

# Global instance for backward compatibility
_global_symbol_system = None

def get_unified_symbol_system(data_dir="data"):
    """Get or create the global unified symbol system instance"""
    global _global_symbol_system
    if _global_symbol_system is None:
        _global_symbol_system = UnifiedSymbolSystem(data_dir=data_dir)
    return _global_symbol_system

# Legacy function aliases
def generate_symbol_from_context(*args, **kwargs):
    """Legacy function - use UnifiedSymbolSystem.generate_contextual_symbol instead"""
    return get_unified_symbol_system().generate_contextual_symbol(*args, **kwargs)

def update_symbol_emotions(*args, **kwargs):
    """Legacy function - use UnifiedSymbolSystem.emotion_mapper.update_symbol_emotions instead"""
    return get_unified_symbol_system().emotion_mapper.update_symbol_emotions(*args, **kwargs)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testing Unified Symbol System...")
    
    # Test unified symbol system
    symbol_system = UnifiedSymbolSystem(data_dir="data/test_symbols")
    
    # Test 1: Text processing
    print("\n1️⃣ Testing symbol processing...")
    test_text = "The golden ratio Φ represents divine proportion in mathematics and beauty."
    results = symbol_system.process_text_for_symbols(test_text)
    
    print(f"Vector matches: {len(results['vector_matches'])}")
    print(f"Discoveries: {len(results['discovered_symbols'])}")
    print(f"Total symbolic score: {results['total_symbolic_score']:.3f}")
    print(f"Routing: {results['routing_decision']['decision']}")
    
    # Test 2: Symbol discovery
    print("\n2️⃣ Testing symbol discovery...")
    discovery_text = "The infinity symbol ∞ represents endless quantities in mathematics and eternal concepts in philosophy."
    discoveries = symbol_system.discovery_engine.discover_symbols_from_text(discovery_text)
    print(f"Discovered {len(discoveries)} symbols")
    
    # Test 3: Symbol generation  
    print("\n3️⃣ Testing symbol generation...")
    generated = symbol_system.generate_contextual_symbol(
        "A bright star illuminates the cosmic void",
        keywords=["star", "cosmic", "illumination"],
        emotions=[("wonder", 0.8), ("joy", 0.6)]
    )
    print(f"Generated: {generated['symbol'] if generated else 'None'}")
    
    # Test 4: Learning
    print("\n4️⃣ Testing learning...")
    symbol_system.learn_from_interaction(
        test_text, "FOLLOW_SYMBOLIC", True, 
        emotions=[("beauty", 0.9), ("harmony", 0.7)]
    )
    print("✅ Learning completed")
    
    # Test 5: Stats
    print("\n5️⃣ Testing comprehensive stats...")
    stats = symbol_system.get_comprehensive_stats()
    print(f"Stats: {len(stats)} categories")
    
    # Test 6: Save
    print("\n6️⃣ Testing save...")
    symbol_system.save_all_data()
    print("✅ All data saved")
    
    print("\n✅ All unified symbol system tests passed!")
    print("\n🎉 Symbol system consolidation complete!")