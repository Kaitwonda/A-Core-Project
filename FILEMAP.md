# Core-Project File Map

A comprehensive AI training system with dual-node architecture, tripartite memory, and adaptive learning capabilities.

## 📁 Core Architecture

### 🧠 Processing Nodes
- **`processing_nodes.py`** - Main processing nodes (LogicNode, SymbolicNode, DynamicBridge) with confidence-gated routing
- **`autonomous_learner.py`** - Autonomous web crawling and learning system with phase-based curriculum
- **`curriculum_manager.py`** - Multi-phase learning progression (computational → symbolic → historical → philosophical)

### 💾 Memory System
- **`memory_architecture.py`** - Core tripartite memory storage (Logic/Symbolic/Bridge) with atomic persistence
- **`decision_history.py`** - Decision history tracking and stability analysis for memory items
- **`adaptive_migration.py`** - Confidence-based migration system with overlap detection
- **`reverse_migration.py`** - Audit system to catch and correct misclassifications
- **`memory_evolution_engine.py`** - Complete memory evolution orchestrator
- **`memory_analytics.py`** - Deep analytics and observability for memory distribution
- **`memory_maintenance.py`** - Pruning and cleanup utilities for Phase 1 symbolic content

## 🔤 Symbol & Language Processing

### 📊 Symbol Management
- **`symbol_memory.py`** - Symbol storage with golden memory (peak emotional states)
- **`symbol_generator.py`** - Emergent symbol creation from context and emotions
- **`symbol_emotion_updater.py`** - Updates symbol-emotion associations based on usage
- **`meta_symbols.json`** - Meta-symbols for recursive/complex patterns (e.g., "🔥⟳" for burn cycles)

### 🔍 Symbol Analysis
- **`symbol_cluster.py`** - Clustering and visualization of symbol relationships
- **`symbol_chainer.py`** - Building chains of related symbolic concepts
- **`symbol_suggester.py`** - Auto-generation of symbols from vector clusters
- **`cluster_namer.py`** - Automatic naming and emoji assignment for clusters

## 🎯 Text Processing & Parsing

### 📝 Core Parsing
- **`parser.py`** - Main text parsing with emotion-weighted symbol extraction
- **`web_parser.py`** - Web content extraction with Trafilatura + BeautifulSoup fallback
- **`emotion_handler.py`** - Multi-model emotion detection with confidence merging

### 🔢 Vector Processing
- **`vector_engine.py`** - Dual embedding system (MiniLM + E5) with fusion logic
- **`vector_memory.py`** - Vector storage with learning phases and confidence tracking
- **`upgrade_old_vectors.py`** - Migration utility for legacy vector formats

## 🧬 Adaptive Learning

### ⚖️ Weight Evolution
- **`weight_evolution.py`** - Progressive weight evolution toward specialization with momentum
- **`memory_optimizer.py`** - Interactive optimizer with adaptive weight adjustment
- **`brain_metrics.py`** - Decision tracking and adaptive weight recommendations

### 📈 Analytics & Visualization
- **`system_analytics.py`** - Node activation timelines and symbol popularity tracking
- **`trail_log.py`** - Processing step logging for the DynamicBridge system
- **`trail_graph.py`** - Network visualization of thought trails and symbol connections

## 🎨 Visualization & Analysis

### 📊 Plotting & Charts
- **`symbol_drift_plot.py`** - Symbol emotional weight changes over time
- **`symbol_emotion_cluster.py`** - Emotional similarity clustering visualization
- **`graph_visualizer.py`** - Network graphs of symbol relationships
- **`inspect_vectors.py`** - Vector memory inspection utility

## 🚀 Execution & Control

### 🔧 Main Interfaces
- **`main.py`** - Primary interactive interface for the hybrid AI system
- **`run_pipeline.py`** - Complete learning and evolution pipeline orchestrator

### 📦 Data Files
- **`seed_symbols.json`** - Initial symbol definitions (🔥 Fire, 💧 Water, 💻 Computer)  
- **`symbol_emotion_map.json`** - Cumulative emotion associations for symbols
- **`user_memory.json`** - User interaction history with emotional context

### 🧪 Testing & Demos
- **`symbol_test.ipynb`** - Interactive Jupyter notebook for symbol system exploration

