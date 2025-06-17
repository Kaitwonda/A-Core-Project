#!/usr/bin/env python3
"""
Import Dependency Analyzer for Dual Brain AI System
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
import json

def extract_imports_from_file(file_path):
    """Extract all import statements from a Python file."""
    imports = {'from': [], 'import': [], 'errors': []}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports['import'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                for alias in node.names:
                    imports['from'].append(f"{module}.{alias.name}" if module else alias.name)
                    
    except SyntaxError as e:
        imports['errors'].append(f"Syntax error: {e}")
    except Exception as e:
        imports['errors'].append(f"Parse error: {e}")
    
    return imports

def find_circular_dependencies(dependency_graph):
    """Find circular dependencies using DFS."""
    visited = set()
    rec_stack = set()
    cycles = []
    
    def dfs(node, path):
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return
        
        if node in visited:
            return
            
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in dependency_graph.get(node, []):
            dfs(neighbor, path[:])
        
        rec_stack.remove(node)
        path.pop()
    
    for node in dependency_graph:
        if node not in visited:
            dfs(node, [])
    
    return cycles

def analyze_project_imports(project_dir):
    """Analyze all Python imports in the project."""
    project_path = Path(project_dir)
    
    # Find all Python files (excluding venv)
    py_files = []
    for py_file in project_path.glob('**/*.py'):
        if 'venv' not in str(py_file) and '__pycache__' not in str(py_file):
            py_files.append(py_file)
    
    print(f"🔍 Analyzing {len(py_files)} Python files...")
    
    # Extract imports from each file
    file_imports = {}
    dependency_graph = defaultdict(list)
    all_imports = set()
    project_modules = set()
    
    # First pass: collect all project module names
    for py_file in py_files:
        module_name = py_file.stem
        if module_name != '__init__':
            project_modules.add(module_name)
    
    # Second pass: analyze imports
    for py_file in py_files:
        relative_path = py_file.relative_to(project_path)
        module_name = py_file.stem
        
        imports = extract_imports_from_file(py_file)
        file_imports[str(relative_path)] = imports
        
        # Build dependency graph for project modules
        for imp in imports['import'] + imports['from']:
            imp_base = imp.split('.')[0]
            all_imports.add(imp_base)
            
            if imp_base in project_modules and imp_base != module_name:
                dependency_graph[module_name].append(imp_base)
    
    # Categorize imports
    external_imports = all_imports - project_modules
    internal_imports = all_imports & project_modules
    
    # Find missing modules (those imported but not found as files)
    existing_modules = set()
    for py_file in py_files:
        existing_modules.add(py_file.stem)
    
    missing_modules = set()
    for imp in internal_imports:
        if imp not in existing_modules:
            missing_modules.add(imp)
    
    # Find circular dependencies
    cycles = find_circular_dependencies(dependency_graph)
    
    # Identify import errors
    import_errors = []
    critical_missing = set()
    
    for file_path, imports in file_imports.items():
        if imports['errors']:
            import_errors.append({
                'file': file_path,
                'errors': imports['errors']
            })
        
        # Check for critical missing imports
        for imp in imports['import'] + imports['from']:
            imp_base = imp.split('.')[0]
            if imp_base in ['spacy', 'numpy', 'pandas', 'sklearn', 'torch', 'tensorflow']:
                # These are common external dependencies that might be missing
                try:
                    __import__(imp_base)
                except ImportError:
                    critical_missing.add(imp_base)
    
    return {
        'file_count': len(py_files),
        'total_imports': len(all_imports),
        'external_imports': sorted(external_imports),
        'internal_imports': sorted(internal_imports),
        'missing_modules': sorted(missing_modules),
        'circular_dependencies': cycles,
        'import_errors': import_errors,
        'critical_missing': sorted(critical_missing),
        'dependency_graph': dict(dependency_graph),
        'file_imports': file_imports
    }

def print_analysis_report(analysis):
    """Print a comprehensive analysis report."""
    print("\n" + "="*80)
    print("🧠 DUAL BRAIN AI SYSTEM - IMPORT DEPENDENCY ANALYSIS")
    print("="*80)
    
    print(f"\n📊 PROJECT OVERVIEW:")
    print(f"   Total Python files analyzed: {analysis['file_count']}")
    print(f"   Total unique imports: {analysis['total_imports']}")
    print(f"   External dependencies: {len(analysis['external_imports'])}")
    print(f"   Internal modules: {len(analysis['internal_imports'])}")
    
    # Critical Missing Dependencies
    if analysis['critical_missing']:
        print(f"\n❌ CRITICAL MISSING DEPENDENCIES:")
        for dep in analysis['critical_missing']:
            print(f"   • {dep}")
        print(f"\n   💡 Install with: pip install {' '.join(analysis['critical_missing'])}")
    else:
        print(f"\n✅ All critical dependencies appear to be available")
    
    # Missing Internal Modules
    if analysis['missing_modules']:
        print(f"\n⚠️  MISSING INTERNAL MODULES:")
        for module in analysis['missing_modules']:
            print(f"   • {module}")
        print(f"\n   These modules are imported but no corresponding .py file was found")
    
    # Circular Dependencies
    if analysis['circular_dependencies']:
        print(f"\n🔄 CIRCULAR DEPENDENCIES DETECTED:")
        for i, cycle in enumerate(analysis['circular_dependencies'], 1):
            print(f"   {i}. {' → '.join(cycle)}")
        print(f"\n   ⚠️  Circular dependencies can cause import errors and should be resolved")
    else:
        print(f"\n✅ No circular dependencies detected")
    
    # Import Errors
    if analysis['import_errors']:
        print(f"\n💥 IMPORT ERRORS:")
        for error in analysis['import_errors']:
            print(f"   File: {error['file']}")
            for err in error['errors']:
                print(f"      • {err}")
    else:
        print(f"\n✅ No syntax/parse errors in import statements")
    
    # External Dependencies
    print(f"\n📦 EXTERNAL DEPENDENCIES:")
    print(f"   Standard library: {len([d for d in analysis['external_imports'] if d in ['os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'collections', 'typing', 'enum', 'dataclasses', 'random', 'hashlib', 're', 'csv', 'tempfile', 'urllib', 'shutil']])}")
    print(f"   Third-party packages: {len([d for d in analysis['external_imports'] if d not in ['os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'collections', 'typing', 'enum', 'dataclasses', 'random', 'hashlib', 're', 'csv', 'tempfile', 'urllib', 'shutil']])}")
    
    third_party = [d for d in analysis['external_imports'] if d not in ['os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'collections', 'typing', 'enum', 'dataclasses', 'random', 'hashlib', 're', 'csv', 'tempfile', 'urllib', 'shutil']]
    if third_party:
        print(f"   Third-party: {', '.join(sorted(third_party))}")
    
    # Key Files Analysis
    key_files = ['master_orchestrator.py', 'cli.py', 'processing_nodes.py', 'talk_to_ai.py', 'autonomous_learner.py', 'bridge_adapter.py']
    
    print(f"\n🔑 KEY FILES ANALYSIS:")
    for key_file in key_files:
        if key_file in analysis['file_imports']:
            imports = analysis['file_imports'][key_file]
            total_imports = len(imports['import']) + len(imports['from'])
            error_count = len(imports['errors'])
            status = "❌" if error_count > 0 else "✅"
            print(f"   {status} {key_file}: {total_imports} imports, {error_count} errors")
            
            if imports['errors']:
                for error in imports['errors']:
                    print(f"      • {error}")
    
    # Dependency Graph (most connected modules)
    print(f"\n🕸️  MOST CONNECTED MODULES:")
    dep_counts = [(module, len(deps)) for module, deps in analysis['dependency_graph'].items()]
    dep_counts.sort(key=lambda x: x[1], reverse=True)
    
    for module, count in dep_counts[:10]:
        if count > 0:
            print(f"   • {module}: depends on {count} other modules")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    priority_issues = []
    if analysis['critical_missing']:
        priority_issues.append("🔴 HIGH: Install missing critical dependencies")
    if analysis['circular_dependencies']:
        priority_issues.append("🟡 MEDIUM: Resolve circular dependencies")
    if analysis['missing_modules']:
        priority_issues.append("🟡 MEDIUM: Create missing internal modules or fix import paths")
    if analysis['import_errors']:
        priority_issues.append("🔴 HIGH: Fix syntax/parse errors in imports")
    
    if priority_issues:
        for issue in priority_issues:
            print(f"   {issue}")
    else:
        print(f"   ✅ Import structure looks healthy!")
    
    print(f"\n" + "="*80)

if __name__ == "__main__":
    project_dir = "."
    
    try:
        analysis = analyze_project_imports(project_dir)
        print_analysis_report(analysis)
        
        # Save detailed analysis to file
        with open("import_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n💾 Detailed analysis saved to import_analysis.json")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()