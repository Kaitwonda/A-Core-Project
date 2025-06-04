#!/usr/bin/env python3
"""
AI Memory Tracking Dashboard & Logging System - Enhanced Version
Integrates with your existing tripartite memory architecture
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from typing import Dict, List, Any, Optional
import threading
from queue import Queue

# Try to import autorefresh, fallback to manual refresh if not available
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ==================== CONFIGURATION ====================

EMOTION_COLORS = {
    'joy': '#FFD700',      # Gold
    'anger': '#DC143C',    # Crimson
    'fear': '#8B008B',     # Dark Magenta
    'sadness': '#4682B4',  # Steel Blue
    'neutral': '#808080',  # Gray
    'peace': '#98FB98',    # Pale Green
    'curiosity': '#FF69B4', # Hot Pink
    'amusement': '#FFA500', # Orange
    'annoyance': '#CD5C5C', # Indian Red
    'excitement': '#FF1493' # Deep Pink
}

# ==================== VALIDATION HELPERS ====================

def validate_schema(data: Any, expected_type: type, schema_name: str = "") -> bool:
    """Validate data matches expected type schema"""
    if not isinstance(data, expected_type):
        st.error(f"Schema validation failed for {schema_name}: expected {expected_type.__name__}, got {type(data).__name__}")
        return False
    return True

@st.cache_data(ttl=5)  # Cache for 5 seconds to prevent constant reloading
def safe_load_json(filename: str, expected_type: type = dict) -> Any:
    """Safely load and validate JSON file with caching"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if validate_schema(data, expected_type, filename):
            return data
        return {} if expected_type == dict else []
    except FileNotFoundError:
        st.warning(f"File not found: {filename}")
        return {} if expected_type == dict else []
    except json.JSONDecodeError as e:
        st.error(f"JSON decode error in {filename}: {e}")
        return {} if expected_type == dict else []

@st.cache_data(show_spinner=False)
def compute_memory_metrics(symbol_memory: dict, logic_memory: list, bridge_memory: list) -> dict:
    """Compute memory metrics with caching to prevent drift"""
    recursive_symbols = len([s for s in symbol_memory if '⟳' in s])
    max_recursion = max([s.count('⟳') for s in symbol_memory.keys()] + [0])
    
    return {
        'total_symbols': len(symbol_memory),
        'recursive_symbols': recursive_symbols,
        'total_logic': len(logic_memory),
        'total_bridge': len(bridge_memory),
        'max_recursion_depth': max_recursion
    }

# ==================== ENHANCED LOGGING SYSTEM ====================

class EventLogger:
    """Convert your JSON logs to efficient JSONL format with buffering"""
    
    def __init__(self, log_dir="logs", buffer_size=100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_log = self.log_dir / f"trail_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.buffer = []
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log events in JSONL format with buffering"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "session_id": st.session_state.get('session_id', 'default'),
            "data": data
        }
        
        with self.lock:
            self.buffer.append(event)
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Write buffer to disk"""
        if self.buffer:
            with open(self.current_log, 'a') as f:
                for event in self.buffer:
                    f.write(json.dumps(event) + '\n')
            self.buffer.clear()
    
    def log_bridge_decision(self, url: str, logic_score: float, symbol_score: float, decision: str) -> None:
        self.log_event("bridge_decision", {
            "url": url,
            "logic_score": logic_score,
            "symbol_score": symbol_score,
            "decision": decision
        })
    
    def log_symbol_activation(self, symbol: str, emotion: str, resonance: float, context: str, origin_trace: Optional[Dict] = None) -> None:
        self.log_event("symbol_activation", {
            "symbol": symbol,
            "emotion": emotion,
            "resonance": resonance,
            "context": context[:100],  # Truncate for efficiency
            "origin_trace": origin_trace  # New: track where symbol came from
        })
    
    def log_memory_migration(self, item_id: str, from_node: str, to_node: str, reason: str, bridge_event_id: Optional[str] = None) -> None:
        self.log_event("memory_migration", {
            "item_id": item_id,
            "from_node": from_node,
            "to_node": to_node,
            "reason": reason,
            "bridge_event_id": bridge_event_id  # New: track triggering bridge event
        })

# ==================== ENHANCED DASHBOARD ====================

st.set_page_config(page_title="AI Memory System Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Tripartite Memory System Dashboard v2.0")

# Dashboard controls
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("🔄 Refresh Data", disabled=st.session_state.get('freeze_dashboard', False)):
        st.cache_data.clear()
        st.rerun()
with col2:
	freeze_state = st.checkbox(
    				"❄️ Freeze Dashboard",
    				value=st.session_state.get('freeze_dashboard', False),
	    help="Prevent automatic updates and recalculations"
	)                  
	st.session_state.freeze_dashboard = freeze_state
with col3:
    if freeze_state:
        st.info("Dashboard frozen - data won't update until unfrozen")

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
if 'realtime_data' not in st.session_state:
    st.session_state.realtime_data = Queue()

# NEW: Add AI running state and last metrics
if 'ai_running' not in st.session_state:
    st.session_state.ai_running = False
if 'last_metrics' not in st.session_state:
    st.session_state.last_metrics = {
        'events_sec': 0,
        'active_symbols': 0,
        'bridge_queue': 0,
        'memory_usage': 0,
        'recursion_status': '⚫ Offline'
    }

# Initialize logger
logger = EventLogger()

# Sidebar for navigation
page = st.sidebar.selectbox("Select View", [
    "System Overview",
    "Symbol Network",
    "Bridge Analytics", 
    "Memory Evolution",
    "Session Replay",
    "Real-time Monitor",
    "Origin Tracer"  # New feature
])

# ==================== SYSTEM OVERVIEW ====================

if page == "System Overview":
    st.header("📊 System Health & Metrics")
    
    # Load data with caching if not frozen
    if not st.session_state.get('freeze_dashboard', False):
        symbol_memory = safe_load_json('symbol_memory.json', dict)
        logic_memory = safe_load_json('logic_memory.json', list)
        bridge_memory = safe_load_json('bridge_memory.json', list)
    else:
        # Use session state for frozen data
        if 'frozen_symbol_memory' not in st.session_state:
            st.session_state.frozen_symbol_memory = safe_load_json('symbol_memory.json', dict)
            st.session_state.frozen_logic_memory = safe_load_json('logic_memory.json', list)
            st.session_state.frozen_bridge_memory = safe_load_json('bridge_memory.json', list)
        symbol_memory = st.session_state.frozen_symbol_memory
        logic_memory = st.session_state.frozen_logic_memory
        bridge_memory = st.session_state.frozen_bridge_memory
    
    # Compute metrics with caching
    metrics = compute_memory_metrics(symbol_memory, logic_memory, bridge_memory)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Active Symbols", metrics['total_symbols'], 
                delta=f"+{metrics['recursive_symbols']} recursive")
    col2.metric("Logic Nodes", metrics['total_logic'])
    col3.metric("Bridge Queue", metrics['total_bridge'])
    col4.metric("Max Recursion Depth", metrics['max_recursion_depth'])
    
    # Memory balance chart with enhanced styling
    fig_balance = go.Figure(data=[
        go.Bar(name='Memory Distribution', 
               x=['Symbolic', 'Logic', 'Bridge'],
               y=[len(symbol_memory), len(logic_memory), len(bridge_memory)],
               marker_color=['#FF69B4', '#4682B4', '#FFD700'])
    ])
    fig_balance.update_layout(
        title="Memory Node Distribution",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_balance, use_container_width=True)
    
    # Weight evolution trend
    st.subheader("⚖️ Weight Evolution Trend")
    weight_history = safe_load_json('weight_evolution_history.json', list)
    
    if weight_history:
        # Extract weight data from nested structure
        weight_data = []
        for entry in weight_history:
            if isinstance(entry, dict) and 'timestamp' in entry:
                row = {'timestamp': entry['timestamp']}
                
                # Extract weights from old_weights and new_weights
                if 'new_weights' in entry and isinstance(entry['new_weights'], dict):
                    row['static_weight'] = entry['new_weights'].get('static', 0)
                    row['dynamic_weight'] = entry['new_weights'].get('dynamic', 0)
                elif 'old_weights' in entry and isinstance(entry['old_weights'], dict):
                    row['static_weight'] = entry['old_weights'].get('static', 0)
                    row['dynamic_weight'] = entry['old_weights'].get('dynamic', 0)
                
                # Add other metrics
                if 'momentum' in entry and isinstance(entry['momentum'], dict):
                    # Calculate momentum difference for plotting
                    momentum_static = entry['momentum'].get('static', 0)
                    momentum_dynamic = entry['momentum'].get('dynamic', 0)
                    row['momentum_diff'] = momentum_static - momentum_dynamic
                    row['momentum_static'] = momentum_static
                    row['momentum_dynamic'] = momentum_dynamic
                else:
                    row['momentum_diff'] = 0
                    row['momentum_static'] = 0
                    row['momentum_dynamic'] = 0
                
                row['actual_specialization'] = entry.get('actual_specialization', 0)
                
                weight_data.append(row)
        
        if weight_data:
            df_weights = pd.DataFrame(weight_data)
            df_weights['timestamp'] = pd.to_datetime(df_weights['timestamp'])
            
            # Create subplot figure for multiple metrics
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Weight Evolution', 'Specialization & Momentum'),
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            # Weight evolution
            fig.add_trace(
                go.Scatter(x=df_weights['timestamp'], y=df_weights['static_weight'],
                          name='Static Weight', 
                          line=dict(color='#4682B4', width=2),
                          hovertemplate='Static: %{y:.3f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_weights['timestamp'], y=df_weights['dynamic_weight'],
                          name='Dynamic Weight', 
                          line=dict(color='#FF69B4', width=2),
                          hovertemplate='Dynamic: %{y:.3f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>'),
                row=1, col=1
            )
            
            # Specialization and momentum
            fig.add_trace(
                go.Scatter(x=df_weights['timestamp'], y=df_weights['actual_specialization'],
                          name='Specialization', 
                          line=dict(color='#32CD32', width=2),
                          hovertemplate='Specialization: %{y:.3f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_weights['timestamp'], y=df_weights['momentum_diff'],
                          name='Momentum Δ (Static - Dynamic)', 
                          line=dict(color='#FFA500', width=2, dash='dash'),
                          hovertemplate='Momentum Δ: %{y:.3f}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>'),
                row=2, col=1
            )
            
            # Add zero line for momentum
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Weight", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=2, col=1)
            
            fig.update_layout(
                height=600, 
                title_text="Weight Evolution & System Dynamics",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                ),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show current state
            if len(df_weights) > 0:
                latest = df_weights.iloc[-1]
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Static Weight", f"{latest['static_weight']:.3f}")
                col2.metric("Current Dynamic Weight", f"{latest['dynamic_weight']:.3f}")
                
                # Handle momentum as dict
                momentum_diff = latest.get('momentum_diff', 0)
                col3.metric("Momentum Direction", 
                           "↑ Static" if momentum_diff > 0 else "↓ Dynamic",
                           delta=f"{abs(momentum_diff):.3f}")
        else:
            st.warning("Could not extract weight data from history file.")

# ==================== ENHANCED SYMBOL NETWORK ====================

elif page == "Symbol Network":
    st.header("🔮 Symbol Relationship Network")
    
    symbol_cooccurrence = safe_load_json('symbol_cooccurrence.json', dict)
    symbol_emotion_map = safe_load_json('symbol_emotion_map.json', dict)
    
    # Create network graph with enhanced features
    G = nx.Graph()
    
    # Add nodes with emotion colors
    for symbol in symbol_cooccurrence.keys():
        emotion_data = symbol_emotion_map.get(symbol, {})
        if emotion_data:
            top_emotion = max(emotion_data.items(), key=lambda x: x[1])[0]
            emotion_score = emotion_data[top_emotion]
        else:
            top_emotion = 'neutral'
            emotion_score = 0
        
        G.add_node(symbol, 
                  emotion=top_emotion,
                  emotion_score=emotion_score,
                  size=15 + symbol.count('⟳') * 10,
                  color=EMOTION_COLORS.get(top_emotion, '#808080'))
    
    # Add weighted edges
    for symbol1, connections in symbol_cooccurrence.items():
        if isinstance(connections, dict):
            for symbol2, weight in connections.items():
                if symbol1 != symbol2 and symbol2 in G.nodes:
                    G.add_edge(symbol1, symbol2, weight=float(weight))
    
    # Create Plotly network visualization with enhanced styling
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Create edge traces with varying thickness
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2].get('weight', 1)
        
        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=0.5 + weight * 0.5, color=f'rgba(128,128,128,{min(0.2 + weight * 0.1, 0.8)})'),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Create node trace with emotion colors - FIXED VERSION
    # Initialize lists (not tuples!)
    node_x = []
    node_y = []
    node_text = []
    node_customdata = []
    node_sizes = []
    node_colors = []
    
    # Build the data
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_sizes.append(G.nodes[node].get('size', 15))
        node_colors.append(G.nodes[node].get('emotion_score', 0))
        # Make sure customdata entries are lists, not tuples
        node_customdata.append([
            G.nodes[node].get('emotion', 'neutral'), 
            G.nodes[node].get('emotion_score', 0)
        ])
    
    # Create the trace with all data at once
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y, 
        text=node_text, 
        mode='markers+text',
        textposition="top center",
        hovertemplate='%{text}<br>Emotion: %{customdata[0]}<br>Score: %{customdata[1]:.2f}<extra></extra>',
        customdata=node_customdata,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_sizes,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Emotion Intensity',
                    side='right'
                ),
                xanchor='left'
            )
        )
    )
    
    fig_network = go.Figure(data=edge_traces + [node_trace],
                           layout=go.Layout(
                               title='Symbol Co-occurrence Network (size = recursion depth, opacity = connection strength)',
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20, l=5, r=5, t=40),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               height=600
                           ))
    
    st.plotly_chart(fig_network, use_container_width=True)
    
    # Symbol emotion breakdown with color coding
    st.subheader("😊 Symbol Emotional Profiles")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_symbol = st.selectbox("Select a symbol to analyze", list(symbol_emotion_map.keys()))
    
    if selected_symbol and selected_symbol in symbol_emotion_map:
        emotions = symbol_emotion_map[selected_symbol]
        df_emotions = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Score'])
        df_emotions = df_emotions.sort_values('Score', ascending=False)
        
        # Add colors to emotions
        df_emotions['Color'] = df_emotions['Emotion'].map(lambda e: EMOTION_COLORS.get(e, '#808080'))
        
        fig_emotions = go.Figure(data=[
            go.Bar(x=df_emotions['Emotion'], y=df_emotions['Score'],
                   marker_color=df_emotions['Color'],
                   text=df_emotions['Score'].round(2),
                   textposition='outside')
        ])
        fig_emotions.update_layout(
            title=f"Emotional Profile for {selected_symbol}",
            xaxis_title="Emotion",
            yaxis_title="Score",
            showlegend=False,
            height=400
        )
        
        with col2:
            st.plotly_chart(fig_emotions, use_container_width=True)

# ==================== BRIDGE ANALYTICS ====================

elif page == "Bridge Analytics":
    st.header("🌉 Bridge Decision Analytics")
    
    # Load bridge conflicts
    bridge_conflicts = safe_load_json('bridge_conflicts.json', list)
    
    # Decision distribution
    if Path('link_decisions.csv').exists():
        decisions_df = pd.read_csv('link_decisions.csv')
        
        if not decisions_df.empty and 'decision' in decisions_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                decision_counts = decisions_df['decision'].value_counts()
                fig_decisions = px.pie(values=decision_counts.values, names=decision_counts.index,
                                      title="Bridge Decision Distribution",
                                      color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_decisions, use_container_width=True)
            
            with col2:
                # Logic vs Symbol score scatter with decision regions
                fig_scatter = px.scatter(decisions_df, x='logic_score', y='symbol_score', 
                                        color='decision', title="Logic vs Symbol Score Distribution",
                                        hover_data=['url'],
                                        color_discrete_sequence=px.colors.qualitative.Set3)
                
                # Add decision boundary regions
                fig_scatter.add_shape(type="line", x0=0, y0=0, x1=10, y1=10,
                                     line=dict(color="gray", width=1, dash="dash"))
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Conflict heatmap
            if bridge_conflicts:
                st.subheader("🔥 Bridge Conflict Analysis")
                
                # Create conflict matrix
                conflict_matrix = defaultdict(lambda: defaultdict(int))
                for conflict in bridge_conflicts:
                    if isinstance(conflict, dict) and 'from_type' in conflict and 'to_type' in conflict:
                        conflict_matrix[conflict['from_type']][conflict['to_type']] += 1
                
                if conflict_matrix:
                    # Convert to DataFrame
                    node_types = list(set(list(conflict_matrix.keys()) + 
                                         [k for v in conflict_matrix.values() for k in v.keys()]))
                    matrix_data = []
                    for from_type in node_types:
                        row = []
                        for to_type in node_types:
                            row.append(conflict_matrix[from_type][to_type])
                        matrix_data.append(row)
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=matrix_data,
                        x=node_types,
                        y=node_types,
                        colorscale='Reds',
                        text=matrix_data,
                        texttemplate="%{text}",
                        textfont={"size": 10}
                    ))
                    fig_heatmap.update_layout(
                        title="Bridge Conflict Heatmap (From → To)",
                        xaxis_title="To Node Type",
                        yaxis_title="From Node Type",
                        height=500
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

# ==================== MEMORY EVOLUTION ====================

elif page == "Memory Evolution":
    st.header("📈 Memory Evolution & Optimization")
    
    evolution_sessions = safe_load_json('evolution_sessions.json', list)
    curriculum_metrics = safe_load_json('curriculum_metrics.json', dict)
    
    if curriculum_metrics:
        # Process curriculum data
        metrics_data = []
        for phase, data in curriculum_metrics.items():
            if isinstance(data, dict):
                metrics_data.append({
                    'Phase': phase,
                    'Chunks Processed': data.get('chunks_processed', 0),
                    'Symbols Generated': data.get('symbols_generated', 0),
                    'Logic Nodes': data.get('logic_nodes_created', 0),
                    'Bridge Events': data.get('bridge_events', 0)
                })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            
            # Create subplots for curriculum progress
            fig_curriculum = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Learning Progress', 'Cumulative Growth', 
                               'Processing Rate', 'Symbol/Logic Ratio'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Learning progress bar chart
            for i, col in enumerate(['Chunks Processed', 'Symbols Generated', 'Logic Nodes']):
                fig_curriculum.add_trace(
                    go.Bar(name=col, x=df_metrics['Phase'], y=df_metrics[col]),
                    row=1, col=1
                )
            
            # Cumulative growth line chart
            df_metrics['Cumulative Symbols'] = df_metrics['Symbols Generated'].cumsum()
            df_metrics['Cumulative Logic'] = df_metrics['Logic Nodes'].cumsum()
            
            fig_curriculum.add_trace(
                go.Scatter(name='Cumulative Symbols', x=df_metrics['Phase'], 
                          y=df_metrics['Cumulative Symbols'], mode='lines+markers'),
                row=1, col=2
            )
            fig_curriculum.add_trace(
                go.Scatter(name='Cumulative Logic', x=df_metrics['Phase'], 
                          y=df_metrics['Cumulative Logic'], mode='lines+markers'),
                row=1, col=2
            )
            
            # Processing rate
            if len(df_metrics) > 1:
                df_metrics['Process Rate'] = df_metrics['Chunks Processed'].diff().fillna(0)
                fig_curriculum.add_trace(
                    go.Scatter(name='Processing Rate', x=df_metrics['Phase'], 
                              y=df_metrics['Process Rate'], mode='lines+markers',
                              line=dict(shape='spline')),
                    row=2, col=1
                )
            
            # Symbol/Logic ratio
            df_metrics['Symbol/Logic Ratio'] = (df_metrics['Symbols Generated'] / 
                                               df_metrics['Logic Nodes'].replace(0, 1))
            fig_curriculum.add_trace(
                go.Scatter(name='Symbol/Logic Ratio', x=df_metrics['Phase'], 
                          y=df_metrics['Symbol/Logic Ratio'], mode='lines+markers',
                          line=dict(color='purple')),
                row=2, col=2
            )
            
            fig_curriculum.update_layout(height=800, showlegend=True,
                                       title_text="Curriculum Learning Analytics")
            st.plotly_chart(fig_curriculum, use_container_width=True)
    
    # Memory drift analysis
    st.subheader("🌊 Symbol Drift Analysis")
    meta_symbols = safe_load_json('meta_symbols.json', dict)
    
    if meta_symbols:
        drift_data = []
        for symbol, data in meta_symbols.items():
            if isinstance(data, dict):
                recursion_depth = symbol.count('⟳')
                drift_data.append({
                    'Symbol': symbol,
                    'Recursion Depth': recursion_depth,
                    'Summary Length': len(data.get('summary', '')),
                    'Origin': data.get('origin_symbol', 'unknown'),
                    'Transformation': data.get('transformation_tag', 'none')
                })
        
        if drift_data:
            df_drift = pd.DataFrame(drift_data)
            
            # Create 3D scatter plot for drift visualization
            fig_drift = px.scatter_3d(df_drift, x='Recursion Depth', y='Summary Length',
                                     z=df_drift.index, color='Transformation',
                                     hover_data=['Symbol', 'Origin'],
                                     title="Symbol Complexity Evolution in 3D Space")
            fig_drift.update_layout(height=600)
            st.plotly_chart(fig_drift, use_container_width=True)

# ==================== SESSION REPLAY ====================

elif page == "Session Replay":
    st.header("🎬 Session Replay Tool")
    
    # Session selector
    log_files = list(Path('logs').glob('*.jsonl')) if Path('logs').exists() else []
    
    if log_files:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_log = st.selectbox("Select log file", log_files, 
                                       format_func=lambda x: f"{x.name} ({x.stat().st_size / 1024:.1f} KB)")
        
        # Load events
        events = []
        try:
            with open(selected_log, 'r') as f:
                for line in f:
                    events.append(json.loads(line))
        except Exception as e:
            st.error(f"Error loading log: {e}")
        
        if events:
            # Replay controls
            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            
            with col1:
                if st.button("▶️ Play", key="play"):
                    st.session_state.playing = True
            with col2:
                if st.button("⏸️ Pause", key="pause"):
                    st.session_state.playing = False
            with col3:
                playback_speed = st.select_slider("Speed", options=[0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
            
            event_index = st.slider("Event Timeline", 0, len(events)-1, 0, key="event_slider")
            
            # Display current event with syntax highlighting
            current_event = events[event_index]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📋 Event Details")
                st.json(current_event)
            
            with col2:
                st.subheader("📊 Event Visualization")
                
                # Visualize based on event type
                if current_event['event_type'] == 'bridge_decision':
                    data = current_event['data']
                    
                    # Radar chart for decision
                    fig = go.Figure(data=go.Scatterpolar(
                        r=[data.get('logic_score', 0), data.get('symbol_score', 0), 5],  # Added third dimension
                        theta=['Logic', 'Symbol', 'Hybrid'],
                        fill='toself',
                        name=data.get('decision', 'Unknown'),
                        line_color=EMOTION_COLORS.get('neutral', '#808080')
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 10])
                        ),
                        title=f"Decision Profile: {data.get('decision', 'Unknown')}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif current_event['event_type'] == 'symbol_activation':
                    data = current_event['data']
                    
                    # Emotion gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=data.get('resonance', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"{data.get('symbol', '?')} - {data.get('emotion', 'neutral')}"},
                        delta={'reference': 0.5},
                        gauge={'axis': {'range': [None, 1]},
                               'bar': {'color': EMOTION_COLORS.get(data.get('emotion', 'neutral'), '#808080')},
                               'steps': [
                                   {'range': [0, 0.25], 'color': "lightgray"},
                                   {'range': [0.25, 0.5], 'color': "gray"},
                                   {'range': [0.5, 0.75], 'color': "lightblue"},
                                   {'range': [0.75, 1], 'color': "lightgreen"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 0.9}}
                    ))
                    st.plotly_chart(fig, use_container_width=True)
            
            # Auto-play functionality
            if st.session_state.get('playing', False) and event_index < len(events) - 1:
                time.sleep(1.0 / playback_speed)
                st.session_state.event_slider = event_index + 1
                st.rerun()

# ==================== REAL-TIME MONITOR - FIXED VERSION ====================

elif page == "Real-time Monitor":
    st.header("📡 Real-time System Monitor")
    
    # AI Control Panel
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("▶️ Start AI" if not st.session_state.ai_running else "⏸️ Stop AI"):
            st.session_state.ai_running = not st.session_state.ai_running
    
    with col2:
        status = "🟢 RUNNING" if st.session_state.ai_running else "🔴 STOPPED"
        st.metric("AI Status", status)
    
    with col3:
        if not st.session_state.ai_running:
            st.info("AI is stopped. Displaying last known values.")
    
    # Use autorefresh if available and AI is running
    if HAS_AUTOREFRESH and st.session_state.ai_running:
        count = st_autorefresh(interval=1000, limit=None, key="realtime_counter")
    else:
        if st.button("🔄 Refresh"):
            st.rerun()
        if not HAS_AUTOREFRESH:
            st.info("Install streamlit-autorefresh for automatic updates: pip install streamlit-autorefresh")
    
    # Create layout
    metric_container = st.container()
    chart_container = st.container()
    log_container = st.container()
    
    # Real-time metrics
    with metric_container:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        if st.session_state.ai_running:
            # Try to read actual metrics from files
            try:
                # Read from your actual memory files
                symbol_memory = safe_load_json('symbol_memory.json', dict)
                logic_memory = safe_load_json('logic_memory.json', list)
                bridge_memory = safe_load_json('bridge_memory.json', list)
                
                # Calculate real metrics
                active_symbols = len(symbol_memory)
                bridge_queue = len(bridge_memory)
                
                # Check for high recursion
                max_recursion = max([s.count('⟳') for s in symbol_memory.keys()] + [0])
                recursion_status = "🟢 Normal" if max_recursion < 3 else ("🟡 High" if max_recursion < 5 else "🔴 Critical")
                
                # For events/sec, you'd need to analyze your log files
                # This is a placeholder - replace with actual log analysis
                events_sec = len(bridge_memory) if bridge_memory else 0
                
                # Memory usage (you could calculate actual size of JSON files)
                import os
                total_size = 0
                for f in ['symbol_memory.json', 'logic_memory.json', 'bridge_memory.json']:
                    if os.path.exists(f):
                        total_size += os.path.getsize(f)
                memory_usage = min(int(total_size / 1024 / 10), 100)  # Rough estimate
                
                # Update last known metrics
                st.session_state.last_metrics = {
                    'events_sec': events_sec,
                    'active_symbols': active_symbols,
                    'bridge_queue': bridge_queue,
                    'memory_usage': memory_usage,
                    'recursion_status': recursion_status
                }
                
            except Exception as e:
                st.error(f"Error reading actual metrics: {e}")
                # Fall back to last known values
                events_sec = st.session_state.last_metrics['events_sec']
                active_symbols = st.session_state.last_metrics['active_symbols']
                bridge_queue = st.session_state.last_metrics['bridge_queue']
                memory_usage = st.session_state.last_metrics['memory_usage']
                recursion_status = st.session_state.last_metrics['recursion_status']
        else:
            # AI is stopped - use last known values
            events_sec = st.session_state.last_metrics['events_sec']
            active_symbols = st.session_state.last_metrics['active_symbols']
            bridge_queue = st.session_state.last_metrics['bridge_queue']
            memory_usage = st.session_state.last_metrics['memory_usage']
            recursion_status = st.session_state.last_metrics['recursion_status']
        
        # Display metrics (no random changes when AI is stopped)
        col1.metric("Events/sec", events_sec, delta=None if not st.session_state.ai_running else "Live")
        col2.metric("Active Symbols", active_symbols)
        col3.metric("Bridge Queue", bridge_queue)
        col4.metric("Memory Usage", f"{memory_usage}%")
        col5.metric("Recursion Alert", recursion_status)
    
    # Real-time activity chart
    with chart_container:
        if st.session_state.ai_running:
            # Generate time series data based on actual logs if available
            timestamps = pd.date_range(end=datetime.now(), periods=60, freq='s')
            
            # Try to read actual activity from logs
            # This is a placeholder - implement actual log parsing
            data = pd.DataFrame({
                'timestamp': timestamps,
                'logic_activity': [len(logic_memory)] * 60 if 'logic_memory' in locals() else [0] * 60,
                'symbol_activity': [len(symbol_memory)] * 60 if 'symbol_memory' in locals() else [0] * 60,
                'bridge_activity': [len(bridge_memory)] * 60 if 'bridge_memory' in locals() else [0] * 60
            })
        else:
            # Show flat line when AI is stopped
            timestamps = pd.date_range(end=datetime.now(), periods=60, freq='s')
            data = pd.DataFrame({
                'timestamp': timestamps,
                'logic_activity': [0] * 60,
                'symbol_activity': [0] * 60,
                'bridge_activity': [0] * 60
            })
        
        fig = px.line(data, x='timestamp', 
                     y=['logic_activity', 'symbol_activity', 'bridge_activity'],
                     title="Real-time Node Activity (60s window)" + (" - AI STOPPED" if not st.session_state.ai_running else ""),
                     color_discrete_map={
                         'logic_activity': '#4682B4',
                         'symbol_activity': '#FF69B4', 
                         'bridge_activity': '#FFD700'
                     })
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent events log
    with log_container:
        st.subheader("📜 Recent Events")
        
        if st.session_state.ai_running:
            # Try to read actual recent events from your log files
            log_files = list(Path('logs').glob('*.jsonl')) if Path('logs').exists() else []
            
            if log_files:
                # Get the most recent log file
                latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                
                try:
                    # Read last 5 events
                    events = []
                    with open(latest_log, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-5:]:  # Get last 5 events
                            events.append(json.loads(line))
                    
                    # Display actual events
                    for event in reversed(events):  # Show newest first
                        timestamp = event.get('timestamp', 'Unknown')
                        event_type = event.get('event_type', 'Unknown')
                        data = event.get('data', {})
                        
                        # Format based on event type
                        if event_type == 'symbol_activation':
                            icon = "🔮"
                            message = f"{data.get('symbol', '?')} activated with {data.get('emotion', 'unknown')} (resonance: {data.get('resonance', 0):.2f})"
                        elif event_type == 'bridge_decision':
                            icon = "🌉"
                            message = f"{data.get('decision', 'Unknown')} for {data.get('url', 'unknown URL')}"
                        elif event_type == 'memory_migration':
                            icon = "📦"
                            message = f"Item {data.get('item_id', '?')} moved from {data.get('from_node', '?')} to {data.get('to_node', '?')}"
                        else:
                            icon = "📌"
                            message = f"{event_type}: {str(data)[:100]}..."
                        
                        col1, col2, col3 = st.columns([2, 1, 7])
                        col1.write(timestamp.split('T')[1].split('.')[0] if 'T' in timestamp else timestamp)
                        col2.write(icon)
                        col3.write(message)
                
                except Exception as e:
                    st.error(f"Error reading log events: {e}")
                    st.info("No recent events to display")
            else:
                st.info("No log files found. Events will appear here when the AI is running.")
        else:
            st.info("AI is stopped. No new events.")

# ==================== NEW: ORIGIN TRACER ====================

elif page == "Origin Tracer":
    st.header("🔍 Symbol & Memory Origin Tracer")
    
    st.markdown("""
    Trace the origin and evolution of symbols and memory nodes. 
    See what bridge events spawned them and track their transformation journey.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        trace_type = st.radio("Select trace type", ["Symbol", "Memory Node", "Bridge Event"])
        
        if trace_type == "Symbol":
            symbol_memory = safe_load_json('symbol_memory.json', dict)
            selected_item = st.selectbox("Select symbol", list(symbol_memory.keys()))
        elif trace_type == "Memory Node":
            st.info("Memory node tracing requires log analysis")
            selected_item = st.text_input("Enter node ID")
        else:
            st.info("Bridge event tracing requires event logs")
            selected_item = st.text_input("Enter event ID")
    
    with col2:
        if selected_item:
            st.subheader(f"📍 Origin Trace: {selected_item}")
            
            # Create origin visualization
            if trace_type == "Symbol" and '⟳' in selected_item:
                # Show recursion tree
                base_symbol = selected_item.replace('⟳', '')
                recursion_depth = selected_item.count('⟳')
                
                # Create tree visualization
                tree_data = []
                for i in range(recursion_depth + 1):
                    symbol = base_symbol + '⟳' * i
                    tree_data.append({
                        'Symbol': symbol,
                        'Depth': i,
                        'Type': 'Recursive' if i > 0 else 'Base'
                    })
                
                df_tree = pd.DataFrame(tree_data)
                
                fig_tree = px.sunburst(df_tree, path=['Type', 'Symbol'], 
                                      values=[1]*len(df_tree),
                                      title=f"Symbol Recursion Tree for {selected_item}")
                st.plotly_chart(fig_tree, use_container_width=True)
                
                # Show metadata if available
                meta_symbols = safe_load_json('meta_symbols.json', dict)
                if selected_item in meta_symbols:
                    st.json(meta_symbols[selected_item])

# ==================== UTILITY FUNCTIONS ====================

def convert_to_jsonl(json_file: str, output_dir: str = "logs") -> Optional[Path]:
    """Convert your existing JSON logs to JSONL format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            output_file = output_dir / f"{Path(json_file).stem}.jsonl"
            with open(output_file, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            return output_file
        else:
            st.error(f"{json_file} is not a list format")
            return None
    except Exception as e:
        st.error(f"Conversion error: {e}")
        return None

# Conversion utility in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Utilities")

# Check if trail_log.json exists before showing button
if Path("trail_log.json").exists():
    if st.sidebar.button("Convert trail_log.json to JSONL"):
        try:
            result = convert_to_jsonl("trail_log.json")
            if result:
                st.sidebar.success(f"✅ Converted to {result}")
        except Exception as e:
            st.sidebar.error(f"❌ Conversion failed: {e}")
else:
    st.sidebar.info("📄 trail_log.json not found")

# Always allow buffer flush
if st.sidebar.button("Flush Event Buffer"):
    logger._flush_buffer()
    st.sidebar.success("✅ Event buffer flushed")

# Show file status
st.sidebar.markdown("---")
st.sidebar.subheader("📊 File Status")
files_to_check = [
    ("symbol_memory.json", "Symbols"),
    ("logic_memory.json", "Logic"),
    ("bridge_memory.json", "Bridge"),
    ("link_decisions.csv", "Decisions")
]

for filename, label in files_to_check:
    if Path(filename).exists():
        size = Path(filename).stat().st_size / 1024
        st.sidebar.success(f"✅ {label}: {size:.1f} KB")
    else:
        st.sidebar.error(f"❌ {label}: Not found")

# Show current session info
st.sidebar.markdown("---")
st.sidebar.info(f"Session ID: {st.session_state.session_id}")
st.sidebar.info(f"Log location: {logger.current_log}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 **Tripartite Memory System v2.0**")
st.sidebar.markdown("Enhanced with origin tracing, weighted networks, and real-time monitoring")
st.sidebar.markdown("Built with ❤️ for transparent AI")