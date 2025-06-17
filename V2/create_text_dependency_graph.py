#!/usr/bin/env python3
"""
Create a text-based dependency graph for the Dual Brain AI System
"""

import json
from collections import defaultdict, deque

def load_analysis():
    """Load the import analysis data."""
    try:
        with open("import_analysis.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Run analyze_imports.py first to generate import_analysis.json")
        return None

def create_text_dependency_tree(analysis):
    """Create a text-based dependency tree."""
    dep_graph = analysis['dependency_graph']
    key_files = ['master_orchestrator', 'cli', 'processing_nodes', 'talk_to_ai', 
                'autonomous_learner', 'bridge_adapter']
    
    print("\n" + "="*80)
    print("🌳 DEPENDENCY TREE FOR KEY FILES")
    print("="*80)
    
    def print_dependencies(module, visited=None, indent=0, max_depth=3):
        if visited is None:
            visited = set()
        
        if module in visited or indent > max_depth:
            if indent <= max_depth:
                print("  " * indent + f"└── {module} (circular/already shown)")
            return
        
        visited.add(module)
        dependencies = dep_graph.get(module, [])
        
        prefix = "  " * indent
        if indent == 0:
            print(f"\n📁 {module}")
        else:
            print(f"{prefix}├── {module}")
        
        for i, dep in enumerate(dependencies):
            if i == len(dependencies) - 1:
                print(f"{prefix}│   └── {dep}")
                if dep in dep_graph:
                    print_dependencies(dep, visited.copy(), indent + 2, max_depth)
            else:
                print(f"{prefix}│   ├── {dep}")
                if dep in dep_graph:
                    print_dependencies(dep, visited.copy(), indent + 2, max_depth)
    
    for key_file in key_files:
        if key_file in dep_graph or key_file in analysis['internal_imports']:
            print_dependencies(key_file)

def analyze_import_paths(analysis):
    """Analyze critical import paths."""
    print("\n" + "="*80)
    print("🔍 CRITICAL IMPORT PATH ANALYSIS")
    print("="*80)
    
    # Find paths to problematic dependencies
    critical_deps = ['spacy', 'pandas'] + analysis.get('missing_modules', [])
    
    print(f"\n🚨 Critical Missing Dependencies Impact:")
    for dep in critical_deps:
        print(f"\n📦 {dep}:")
        
        # Find which modules would be affected
        affected = []
        for file_path, imports in analysis['file_imports'].items():
            all_imports = imports['import'] + imports['from']
            for imp in all_imports:
                if dep in imp.lower():
                    affected.append(file_path)
        
        if affected:
            print(f"   Directly affects {len(affected)} files:")
            for file in sorted(affected)[:5]:  # Show first 5
                print(f"   ├── {file}")
            if len(affected) > 5:
                print(f"   └── ... and {len(affected) - 5} more files")
        else:
            print(f"   No direct imports found (may be transitive)")

def analyze_circular_dependencies(analysis):
    """Detailed analysis of circular dependencies."""
    print("\n" + "="*80)
    print("🔄 CIRCULAR DEPENDENCY DETAILED ANALYSIS")
    print("="*80)
    
    cycles = analysis['circular_dependencies']
    
    if not cycles:
        print("\n✅ No circular dependencies found!")
        return
    
    for i, cycle in enumerate(cycles, 1):
        print(f"\n🔄 Cycle {i}: {' → '.join(cycle)}")
        
        # Show specific import statements causing the cycle
        print(f"   Import chain:")
        for j in range(len(cycle) - 1):
            current = cycle[j]
            next_module = cycle[j + 1]
            
            # Find the specific import statement
            current_file = f"{current}.py"
            if current_file in analysis['file_imports']:
                imports = analysis['file_imports'][current_file]
                for imp in imports['import'] + imports['from']:
                    if next_module in imp:
                        print(f"   ├── {current}.py imports {next_module}: '{imp}'")
                        break
        
        # Suggest fixes
        print(f"   💡 Suggested fixes:")
        print(f"   ├── Move shared functionality to a new module")
        print(f"   ├── Use lazy imports (import inside functions)")
        print(f"   └── Apply dependency injection pattern")

def create_module_complexity_report(analysis):
    """Create a report on module complexity."""
    print("\n" + "="*80)
    print("📊 MODULE COMPLEXITY ANALYSIS")
    print("="*80)
    
    dep_graph = analysis['dependency_graph']
    
    # Calculate metrics for each module
    modules_metrics = {}
    for module in analysis['internal_imports']:
        dependencies = len(dep_graph.get(module, []))
        
        # Count how many modules depend on this one
        dependents = 0
        for other_module, other_deps in dep_graph.items():
            if module in other_deps:
                dependents += 1
        
        modules_metrics[module] = {
            'dependencies': dependencies,
            'dependents': dependents,
            'complexity': dependencies + dependents
        }
    
    # Sort by complexity
    sorted_modules = sorted(modules_metrics.items(), 
                          key=lambda x: x[1]['complexity'], 
                          reverse=True)
    
    print(f"\n📈 Top 15 Most Complex Modules:")
    print(f"{'Module':<25} {'Dependencies':<12} {'Dependents':<12} {'Total':<8}")
    print("-" * 65)
    
    for module, metrics in sorted_modules[:15]:
        print(f"{module:<25} {metrics['dependencies']:<12} "
              f"{metrics['dependents']:<12} {metrics['complexity']:<8}")
    
    # Identify potential refactoring candidates
    print(f"\n🔧 Refactoring Recommendations:")
    
    high_dependency = [m for m, metrics in modules_metrics.items() 
                      if metrics['dependencies'] > 10]
    high_dependents = [m for m, metrics in modules_metrics.items() 
                      if metrics['dependents'] > 5]
    
    if high_dependency:
        print(f"\n   📤 Modules with too many dependencies (>10):")
        for module in high_dependency[:5]:
            deps = modules_metrics[module]['dependencies']
            print(f"   ├── {module}: {deps} dependencies")
        print(f"   └── Consider: Break into smaller modules, use facades")
    
    if high_dependents:
        print(f"\n   📥 Modules that too many others depend on (>5):")
        for module in high_dependents[:5]:
            deps = modules_metrics[module]['dependents']
            print(f"   ├── {module}: {deps} dependents")
        print(f"   └── Consider: Create interfaces, split responsibilities")

def create_import_health_summary(analysis):
    """Create an overall health summary."""
    print("\n" + "="*80)
    print("🏥 IMPORT SYSTEM HEALTH SUMMARY")
    print("="*80)
    
    # Calculate health scores
    total_files = analysis['file_count']
    files_with_errors = len(analysis['import_errors'])
    circular_count = len(analysis['circular_dependencies'])
    missing_critical = len(analysis['critical_missing'])
    missing_internal = len(analysis['missing_modules'])
    
    # Health scoring
    error_score = max(0, 100 - (files_with_errors / total_files * 100))
    circular_score = max(0, 100 - (circular_count * 20))
    dependency_score = max(0, 100 - (missing_critical * 30) - (missing_internal * 10))
    
    overall_score = (error_score + circular_score + dependency_score) / 3
    
    print(f"\n📊 Health Metrics:")
    print(f"   Import Error Score:    {error_score:.1f}/100")
    print(f"   Circular Dep Score:    {circular_score:.1f}/100") 
    print(f"   Dependency Score:      {dependency_score:.1f}/100")
    print(f"   Overall Health:        {overall_score:.1f}/100")
    
    # Health status
    if overall_score >= 85:
        status = "🟢 EXCELLENT"
    elif overall_score >= 70:
        status = "🟡 GOOD"
    elif overall_score >= 50:
        status = "🟠 FAIR"
    else:
        status = "🔴 POOR"
    
    print(f"\n🎯 System Status: {status}")
    
    # Immediate actions needed
    print(f"\n🚨 Immediate Actions Required:")
    if missing_critical:
        print(f"   1. Install missing dependencies: {', '.join(analysis['critical_missing'])}")
    if circular_count > 0:
        print(f"   2. Resolve {circular_count} circular dependencies")
    if missing_internal:
        print(f"   3. Fix {missing_internal} missing internal modules")
    if files_with_errors:
        print(f"   4. Fix syntax errors in {files_with_errors} files")
    
    if overall_score >= 85:
        print(f"   ✅ System is healthy! Only minor optimizations needed.")

def create_action_plan(analysis):
    """Create a prioritized action plan."""
    print("\n" + "="*80)
    print("📋 PRIORITIZED ACTION PLAN")
    print("="*80)
    
    actions = []
    
    # High priority actions
    if analysis['critical_missing']:
        actions.append({
            'priority': 'HIGH',
            'action': 'Install missing critical dependencies',
            'details': f"pip install {' '.join(analysis['critical_missing'])}",
            'impact': 'Unblocks most system functionality'
        })
    
    if analysis['import_errors']:
        actions.append({
            'priority': 'HIGH', 
            'action': 'Fix syntax/parse errors',
            'details': f"{len(analysis['import_errors'])} files have import errors",
            'impact': 'Prevents module loading'
        })
    
    # Medium priority actions
    if analysis['circular_dependencies']:
        actions.append({
            'priority': 'MEDIUM',
            'action': 'Resolve circular dependencies', 
            'details': f"{len(analysis['circular_dependencies'])} cycles found",
            'impact': 'Improves reliability and maintainability'
        })
    
    if analysis['missing_modules']:
        actions.append({
            'priority': 'MEDIUM',
            'action': 'Fix missing internal modules',
            'details': f"{len(analysis['missing_modules'])} modules imported but not found",
            'impact': 'Fixes import errors and warnings'
        })
    
    # Low priority actions
    dep_graph = analysis['dependency_graph']
    high_complexity = [m for m, deps in dep_graph.items() if len(deps) > 8]
    if high_complexity:
        actions.append({
            'priority': 'LOW',
            'action': 'Refactor high-complexity modules',
            'details': f"{len(high_complexity)} modules have >8 dependencies",
            'impact': 'Improves maintainability'
        })
    
    # Print actions by priority
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        priority_actions = [a for a in actions if a['priority'] == priority]
        if priority_actions:
            print(f"\n🔴 {priority} PRIORITY:" if priority == 'HIGH' else 
                  f"\n🟡 {priority} PRIORITY:" if priority == 'MEDIUM' else
                  f"\n🟢 {priority} PRIORITY:")
            
            for i, action in enumerate(priority_actions, 1):
                print(f"   {i}. {action['action']}")
                print(f"      Details: {action['details']}")
                print(f"      Impact: {action['impact']}")
                print()

def main():
    """Main function to create text-based dependency analysis."""
    print("📝 Creating text-based dependency analysis...")
    
    # Load analysis data
    analysis = load_analysis()
    if not analysis:
        return
    
    # Create various analysis views
    create_text_dependency_tree(analysis)
    analyze_import_paths(analysis)
    analyze_circular_dependencies(analysis)
    create_module_complexity_report(analysis)
    create_import_health_summary(analysis)
    create_action_plan(analysis)
    
    print(f"\n" + "="*80)
    print("✅ DEPENDENCY ANALYSIS COMPLETE")
    print("="*80)
    print(f"📄 See IMPORT_ANALYSIS_REPORT.md for detailed findings")
    print(f"🔧 Follow the action plan above to resolve issues")

if __name__ == "__main__":
    main()