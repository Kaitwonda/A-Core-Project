#!/usr/bin/env python3
"""
Create a visual dependency graph for the Dual Brain AI System
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import numpy as np

def load_analysis():
    """Load the import analysis data."""
    try:
        with open("import_analysis.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Run analyze_imports.py first to generate import_analysis.json")
        return None

def create_dependency_graph(analysis):
    """Create a NetworkX graph from the dependency analysis."""
    G = nx.DiGraph()
    
    # Add all modules as nodes
    all_modules = set(analysis['internal_imports'])
    for module in all_modules:
        G.add_node(module)
    
    # Add edges from dependency graph
    dep_graph = analysis['dependency_graph']
    for source, targets in dep_graph.items():
        for target in targets:
            if target in all_modules:  # Only internal dependencies
                G.add_edge(source, target)
    
    return G

def identify_node_categories(G, analysis):
    """Categorize nodes for better visualization."""
    key_files = ['master_orchestrator', 'cli', 'processing_nodes', 'talk_to_ai', 
                'autonomous_learner', 'bridge_adapter']
    
    circular_nodes = set()
    for cycle in analysis['circular_dependencies']:
        circular_nodes.update(cycle)
    
    # Calculate node importance (in-degree + out-degree)
    importance = {}
    for node in G.nodes():
        importance[node] = G.in_degree(node) + G.out_degree(node)
    
    categories = {}
    for node in G.nodes():
        if node in key_files:
            categories[node] = 'key'
        elif node in circular_nodes:
            categories[node] = 'circular'
        elif importance[node] > 10:
            categories[node] = 'highly_connected'
        elif importance[node] > 5:
            categories[node] = 'medium_connected'
        else:
            categories[node] = 'low_connected'
    
    return categories, importance

def create_visualizations(G, analysis):
    """Create multiple visualizations of the dependency graph."""
    
    categories, importance = identify_node_categories(G, analysis)
    
    # Color mapping
    color_map = {
        'key': '#FF6B6B',           # Red for key files
        'circular': '#4ECDC4',      # Teal for circular dependencies
        'highly_connected': '#45B7D1', # Blue for highly connected
        'medium_connected': '#96CEB4', # Green for medium connected  
        'low_connected': '#FFEAA7'   # Yellow for low connected
    }
    
    # Size mapping based on importance
    sizes = [max(100, importance[node] * 50) for node in G.nodes()]
    colors = [color_map[categories[node]] for node in G.nodes()]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Full dependency graph
    ax1 = plt.subplot(221)
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    nx.draw(G, pos, 
           node_color=colors,
           node_size=sizes,
           with_labels=True,
           font_size=8,
           font_weight='bold',
           arrows=True,
           edge_color='gray',
           alpha=0.7,
           ax=ax1)
    
    ax1.set_title("Full Dependency Graph", fontsize=14, fontweight='bold')
    
    # Create legend
    legend_elements = []
    for category, color in color_map.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=category.replace('_', ' ').title()))
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    # 2. Key files and their immediate dependencies
    ax2 = plt.subplot(222)
    key_files = ['master_orchestrator', 'cli', 'processing_nodes', 'talk_to_ai', 
                'autonomous_learner', 'bridge_adapter']
    
    # Create subgraph with key files and their immediate neighbors
    key_subgraph_nodes = set(key_files)
    for key_file in key_files:
        if key_file in G:
            key_subgraph_nodes.update(G.successors(key_file))
            key_subgraph_nodes.update(G.predecessors(key_file))
    
    G_key = G.subgraph(key_subgraph_nodes)
    pos_key = nx.spring_layout(G_key, k=2, iterations=50, seed=42)
    
    key_colors = []
    key_sizes = []
    for node in G_key.nodes():
        if node in key_files:
            key_colors.append('#FF6B6B')  # Red for key files
            key_sizes.append(300)
        else:
            key_colors.append('#DDDDDD')  # Gray for dependencies
            key_sizes.append(150)
    
    nx.draw(G_key, pos_key,
           node_color=key_colors,
           node_size=key_sizes,
           with_labels=True,
           font_size=9,
           font_weight='bold',
           arrows=True,
           edge_color='gray',
           alpha=0.8,
           ax=ax2)
    
    ax2.set_title("Key Files and Dependencies", fontsize=14, fontweight='bold')
    
    # 3. Circular dependencies visualization
    ax3 = plt.subplot(223)
    circular_nodes = set()
    for cycle in analysis['circular_dependencies']:
        circular_nodes.update(cycle)
    
    if circular_nodes:
        G_circular = G.subgraph(circular_nodes)
        pos_circular = nx.circular_layout(G_circular)
        
        nx.draw(G_circular, pos_circular,
               node_color='#4ECDC4',
               node_size=400,
               with_labels=True,
               font_size=10,
               font_weight='bold',
               arrows=True,
               edge_color='red',
               edge_style='--',
               width=2,
               alpha=0.9,
               ax=ax3)
    
    ax3.set_title("Circular Dependencies", fontsize=14, fontweight='bold')
    
    # 4. Most connected modules
    ax4 = plt.subplot(224)
    
    # Get top 10 most connected nodes
    top_connected = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    top_nodes = [node for node, _ in top_connected]
    top_importance = [imp for _, imp in top_connected]
    
    # Create bar chart
    y_pos = np.arange(len(top_nodes))
    bars = ax4.barh(y_pos, top_importance, color='#45B7D1', alpha=0.7)
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([node.replace('_', ' ') for node in top_nodes])
    ax4.set_xlabel('Connection Count (In + Out Degree)')
    ax4.set_title('Most Connected Modules', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center')
    
    plt.tight_layout()
    return fig

def print_summary(analysis):
    """Print a summary of the analysis."""
    print("\n" + "="*60)
    print("📊 DEPENDENCY GRAPH ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n🔍 Graph Metrics:")
    print(f"   Total modules: {len(analysis['internal_imports'])}")
    print(f"   Total dependencies: {sum(len(deps) for deps in analysis['dependency_graph'].values())}")
    
    if analysis['circular_dependencies']:
        print(f"\n🔄 Circular Dependencies Found:")
        for i, cycle in enumerate(analysis['circular_dependencies'], 1):
            print(f"   {i}. {' → '.join(cycle)}")
    else:
        print(f"\n✅ No circular dependencies found")
    
    print(f"\n🎯 Key Files Status:")
    key_files = ['master_orchestrator.py', 'cli.py', 'processing_nodes.py', 
                'talk_to_ai.py', 'autonomous_learner.py', 'bridge_adapter.py']
    
    for key_file in key_files:
        if key_file in analysis['file_imports']:
            imports = analysis['file_imports'][key_file]
            error_count = len(imports['errors'])
            status = "❌" if error_count > 0 else "✅"
            print(f"   {status} {key_file}: {error_count} errors")

def main():
    """Main function to create dependency visualizations."""
    print("📈 Creating dependency graph visualizations...")
    
    # Load analysis data
    analysis = load_analysis()
    if not analysis:
        return
    
    # Create NetworkX graph
    G = create_dependency_graph(analysis)
    
    print(f"📊 Graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Create visualizations
    try:
        fig = create_visualizations(G, analysis)
        
        # Save the plot
        output_file = "dependency_graph.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ Dependency graph saved as {output_file}")
        
        # Also save as SVG for scalability
        plt.savefig("dependency_graph.svg", format='svg', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✅ Scalable version saved as dependency_graph.svg")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        print("Note: matplotlib may not be available in this environment")
    
    # Print text summary
    print_summary(analysis)
    
    # Save graph data as JSON for other tools
    graph_data = {
        'nodes': list(G.nodes()),
        'edges': [(source, target) for source, target in G.edges()],
        'circular_dependencies': analysis['circular_dependencies'],
        'node_metrics': {node: {'in_degree': G.in_degree(node), 
                               'out_degree': G.out_degree(node),
                               'total_degree': G.in_degree(node) + G.out_degree(node)}
                        for node in G.nodes()}
    }
    
    with open("dependency_graph_data.json", "w") as f:
        json.dump(graph_data, f, indent=2)
    print(f"✅ Graph data saved as dependency_graph_data.json")

if __name__ == "__main__":
    main()