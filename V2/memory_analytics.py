# memory_analytics.py - Deep Analytics and Observability System

from datetime import datetime
from collections import Counter
import json
from pathlib import Path

class MemoryAnalyzer:
    """
    Provides deep analytics and insights into memory distribution,
    patterns, and evolution over time.
    """
    
    def __init__(self, memory, data_dir="data"):
        self.memory = memory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.analytics_file = self.data_dir / "memory_analytics_history.json"
        
    def get_memory_stats(self):
        """Get comprehensive statistics about memory distribution"""
        counts = self.memory.get_counts()
        
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'distribution': {
                'logic': {
                    'count': counts['logic'],
                    'percentage': (counts['logic'] / max(1, counts['total'])) * 100,
                    'avg_score': self._calculate_avg_score(self.memory.logic_memory, 'logic_score'),
                    'avg_age_days': self._calculate_avg_age(self.memory.logic_memory),
                    'stability': self._calculate_stability(self.memory.logic_memory)
                },
                'symbolic': {
                    'count': counts['symbolic'],
                    'percentage': (counts['symbolic'] / max(1, counts['total'])) * 100,
                    'avg_score': self._calculate_avg_score(self.memory.symbolic_memory, 'symbolic_score'),
                    'avg_age_days': self._calculate_avg_age(self.memory.symbolic_memory),
                    'stability': self._calculate_stability(self.memory.symbolic_memory)
                },
                'bridge': {
                    'count': counts['bridge'],
                    'percentage': (counts['bridge'] / max(1, counts['total'])) * 100,
                    'avg_logic_score': self._calculate_avg_score(self.memory.bridge_memory, 'logic_score'),
                    'avg_symbolic_score': self._calculate_avg_score(self.memory.bridge_memory, 'symbolic_score'),
                    'avg_age_days': self._calculate_avg_age(self.memory.bridge_memory),
                    'stability': self._calculate_stability(self.memory.bridge_memory),
                    'volatility': self._calculate_volatility(self.memory.bridge_memory)
                }
            },
            'total_items': counts['total'],
            'health_indicators': self._calculate_health_indicators(counts)
        }
        
        return stats
        
    def _calculate_avg_score(self, items, score_key):
        """Calculate average score for items"""
        if not items:
            return 0.0
        scores = [item.get(score_key, 0) for item in items]
        return sum(scores) / len(scores) if scores else 0.0
        
    def _calculate_avg_age(self, items):
        """Calculate average age in days"""
        if not items:
            return 0.0
            
        now = datetime.utcnow()
        ages = []
        
        for item in items:
            if 'stored_at' in item:
                try:
                    stored = datetime.fromisoformat(item['stored_at'].replace('Z', '+00:00'))
                    age_days = (now - stored).days
                    ages.append(age_days)
                except:
                    pass
                    
        return sum(ages) / len(ages) if ages else 0.0
        
    def _calculate_stability(self, items):
        """Calculate overall stability of items (0-1)"""
        if not items:
            return 1.0
            
        stable_count = 0
        for item in items:
            stability = self.memory.get_item_stability(item)
            if stability['is_stable']:
                stable_count += 1
                
        return stable_count / len(items)
        
    def _calculate_volatility(self, items):
        """Calculate volatility (frequency of decision changes)"""
        if not items:
            return 0.0
            
        volatilities = []
        for item in items:
            history = item.get('decision_history', [])
            if len(history) >= 2:
                # Count decision changes
                changes = sum(1 for i in range(1, len(history)) 
                            if history[i]['decision'] != history[i-1]['decision'])
                volatility = changes / (len(history) - 1)
                volatilities.append(volatility)
                
        return sum(volatilities) / len(volatilities) if volatilities else 0.0
        
    def _calculate_health_indicators(self, counts):
        """Calculate system health indicators"""
        total = counts['total']
        if total == 0:
            return {
                'status': 'empty', 
                'issues': ['No items in memory'],
                'recommendations': ['Process some content to build knowledge base']
            }
            
        bridge_pct = (counts['bridge'] / total) * 100
        
        indicators = {
            'status': 'healthy',
            'bridge_percentage': bridge_pct,
            'issues': [],
            'recommendations': []
        }
        
        # Check for issues
        if bridge_pct > 50:
            indicators['status'] = 'needs_attention'
            indicators['issues'].append(f'High bridge percentage: {bridge_pct:.1f}%')
            indicators['recommendations'].append('Consider lowering migration threshold')
            
        if counts['logic'] == 0 or counts['symbolic'] == 0:
            indicators['issues'].append('Severe imbalance: one memory type is empty')
            indicators['recommendations'].append('Review classification criteria')
            
        if total < 10:
            indicators['issues'].append('Very few items in memory')
            indicators['recommendations'].append('Process more content to build knowledge')
            
        return indicators
        
    def analyze_bridge_patterns(self):
        """Deep analysis of bridge memory patterns"""
        bridge_items = self.memory.bridge_memory
        
        if not bridge_items:
            return {
                'total': 0,
                'patterns': {},
                'insights': ['Bridge memory is empty']
            }
            
        analysis = {
            'total': len(bridge_items),
            'patterns': {},
            'insights': [],
            'keyword_analysis': {},
            'score_distribution': {}
        }
        
        # Categorize patterns
        patterns = {
            'volatile': [],  # Frequently changing classification
            'balanced': [],  # Similar logic/symbolic scores
            'conflicted': [],  # High scores in both
            'weak': [],  # Low scores in both
            'stuck': []  # Been in bridge for a long time
        }
        
        # Analyze each item
        for item in bridge_items:
            logic_score = item.get('logic_score', 0)
            symbolic_score = item.get('symbolic_score', 0)
            
            # Check patterns
            stability = self.memory.get_item_stability(item)
            
            if len(stability['decision_counts']) >= 3:
                patterns['volatile'].append(item)
                
            if abs(logic_score - symbolic_score) < 1.0:
                patterns['balanced'].append(item)
                
            if logic_score > 6 and symbolic_score > 6:
                patterns['conflicted'].append(item)
                
            if logic_score < 3 and symbolic_score < 3:
                patterns['weak'].append(item)
                
            # Check age
            if 'stored_at' in item:
                try:
                    stored = datetime.fromisoformat(item['stored_at'].replace('Z', '+00:00'))
                    age_days = (datetime.utcnow() - stored).days
                    if age_days > 7:
                        patterns['stuck'].append(item)
                except:
                    pass
                    
        # Store pattern counts
        analysis['patterns'] = {
            name: len(items) for name, items in patterns.items()
        }
        
        # Extract keywords from bridge items
        all_text = ' '.join(item.get('text', '') for item in bridge_items)
        # Simple keyword extraction (in real system, use proper NLP)
        words = all_text.lower().split()
        word_freq = Counter(word for word in words if len(word) > 4)
        analysis['keyword_analysis']['top_keywords'] = word_freq.most_common(10)
        
        # Score distribution
        analysis['score_distribution'] = {
            'avg_logic_score': self._calculate_avg_score(bridge_items, 'logic_score'),
            'avg_symbolic_score': self._calculate_avg_score(bridge_items, 'symbolic_score'),
            'high_both': len(patterns['conflicted']),
            'low_both': len(patterns['weak'])
        }
        
        # Generate insights
        if patterns['volatile']:
            pct = (len(patterns['volatile']) / len(bridge_items)) * 100
            analysis['insights'].append(
                f"{pct:.1f}% of bridge items are volatile (frequently changing classification)"
            )
            
        if patterns['stuck']:
            analysis['insights'].append(
                f"{len(patterns['stuck'])} items have been in bridge for over 7 days"
            )
            
        if patterns['conflicted']:
            analysis['insights'].append(
                f"{len(patterns['conflicted'])} items have high scores in both logic and symbolic"
            )
            
        if analysis['score_distribution']['avg_logic_score'] > analysis['score_distribution']['avg_symbolic_score'] + 2:
            analysis['insights'].append(
                "Bridge items lean heavily toward logic - consider adjusting weights"
            )
            
        return analysis
        
    def generate_evolution_report(self, migration_summary=None, audit_summary=None):
        """Generate comprehensive evolution report"""
        stats = self.get_memory_stats()
        bridge_analysis = self.analyze_bridge_patterns()
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_items': stats['total_items'],
                'distribution': {
                    'logic': f"{stats['distribution']['logic']['count']} ({stats['distribution']['logic']['percentage']:.1f}%)",
                    'symbolic': f"{stats['distribution']['symbolic']['count']} ({stats['distribution']['symbolic']['percentage']:.1f}%)",
                    'bridge': f"{stats['distribution']['bridge']['count']} ({stats['distribution']['bridge']['percentage']:.1f}%)"
                }
            },
            'health': stats['health_indicators'],
            'bridge_analysis': bridge_analysis,
            'stability_metrics': {
                'logic_stability': stats['distribution']['logic']['stability'],
                'symbolic_stability': stats['distribution']['symbolic']['stability'],
                'bridge_stability': stats['distribution']['bridge']['stability'],
                'bridge_volatility': stats['distribution']['bridge']['volatility']
            }
        }
        
        # Add migration info if provided
        if migration_summary:
            report['migrations'] = migration_summary
            
        if audit_summary:
            report['audit'] = audit_summary
            
        # Save to history
        self._save_report(report)
        
        return report
        
    def _save_report(self, report):
        """Save report to history file"""
        history = []
        
        # Load existing history
        if self.analytics_file.exists():
            try:
                with open(self.analytics_file, 'r') as f:
                    history = json.load(f)
            except:
                pass
                
        # Add new report and keep last 100
        history.append(report)
        history = history[-100:]
        
        # Save
        with open(self.analytics_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    def print_report(self, report):
        """Pretty print the evolution report"""
        print("\n" + "="*60)
        print("📊 MEMORY EVOLUTION REPORT")
        print("="*60)
        
        print(f"\n📅 Generated: {report['timestamp']}")
        
        print("\n📈 Distribution:")
        for mem_type, info in report['summary']['distribution'].items():
            print(f"  {mem_type.capitalize()}: {info}")
            
        print(f"\n💚 Health Status: {report['health']['status'].upper()}")
        if report['health']['issues']:
            print("  Issues:")
            for issue in report['health']['issues']:
                print(f"    ⚠️  {issue}")
        
        # FIXED: Safely check for recommendations
        if report['health'].get('recommendations'):
            print("  Recommendations:")
            for rec in report['health']['recommendations']:
                print(f"    💡 {rec}")
                
        print("\n🌉 Bridge Analysis:")
        print(f"  Total items: {report['bridge_analysis']['total']}")
        if report['bridge_analysis']['patterns']:
            print("  Patterns:")
            for pattern, count in report['bridge_analysis']['patterns'].items():
                if count > 0:
                    print(f"    {pattern}: {count}")
                    
        if report['bridge_analysis']['insights']:
            print("  Insights:")
            for insight in report['bridge_analysis']['insights']:
                print(f"    → {insight}")
                
        print("\n📊 Stability Metrics:")
        print(f"  Logic: {report['stability_metrics']['logic_stability']:.2%}")
        print(f"  Symbolic: {report['stability_metrics']['symbolic_stability']:.2%}")
        print(f"  Bridge: {report['stability_metrics']['bridge_stability']:.2%}")
        print(f"  Bridge Volatility: {report['stability_metrics']['bridge_volatility']:.2%}")
        
        if 'migrations' in report:
            print(f"\n🔄 Migrations: {report['migrations'].get('total_migrated', 0)} items")
            
        if 'audit' in report:
            print(f"\n🔍 Audit: {report['audit'].get('total_reversed', 0)} items reversed")
            
        print("\n" + "="*60)
        

# Unit tests
if __name__ == "__main__":
    import tempfile
    from decision_history import HistoryAwareMemory
    
    print("🧪 Testing Memory Analytics...")
    
    # Test 1: Basic statistics
    print("\n1️⃣ Test: Basic memory statistics")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        analyzer = MemoryAnalyzer(memory, data_dir=tmpdir)
        
        # Add some items
        for i in range(5):
            memory.store({'text': f'Logic item {i}', 'logic_score': 8, 'symbolic_score': 2}, 'FOLLOW_LOGIC')
        for i in range(3):
            memory.store({'text': f'Symbolic item {i}', 'logic_score': 2, 'symbolic_score': 8}, 'FOLLOW_SYMBOLIC')
        for i in range(7):
            memory.store({'text': f'Bridge item {i}', 'logic_score': 5, 'symbolic_score': 5}, 'FOLLOW_HYBRID')
            
        stats = analyzer.get_memory_stats()
        
        assert stats['total_items'] == 15
        assert stats['distribution']['logic']['count'] == 5
        assert stats['distribution']['symbolic']['count'] == 3
        assert stats['distribution']['bridge']['count'] == 7
        assert abs(stats['distribution']['bridge']['percentage'] - 46.7) < 1
        
        print("✅ Basic statistics work correctly")
        
    # Test 2: Bridge pattern analysis
    print("\n2️⃣ Test: Bridge pattern analysis")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        analyzer = MemoryAnalyzer(memory, data_dir=tmpdir)
        
        # Add various bridge patterns
        # Volatile item
        volatile = {'text': 'Volatile content', 'logic_score': 5, 'symbolic_score': 5}
        for dec in ['FOLLOW_LOGIC', 'FOLLOW_SYMBOLIC', 'FOLLOW_HYBRID', 'FOLLOW_LOGIC']:
            memory.store(volatile.copy(), dec)
            
        # Conflicted item
        memory.store({'text': 'High both', 'logic_score': 8, 'symbolic_score': 7}, 'FOLLOW_HYBRID')
        
        # Weak item
        memory.store({'text': 'Low both', 'logic_score': 2, 'symbolic_score': 2}, 'FOLLOW_HYBRID')
        
        analysis = analyzer.analyze_bridge_patterns()
        
        assert analysis['total'] > 0
        assert 'volatile' in analysis['patterns']
        assert len(analysis['insights']) > 0
        
        print("✅ Bridge pattern analysis works")
        
    # Test 3: Full evolution report
    print("\n3️⃣ Test: Full evolution report")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        analyzer = MemoryAnalyzer(memory, data_dir=tmpdir)
        
        # Add diverse items
        for i in range(10):
            score_type = i % 3
            if score_type == 0:
                memory.store({'text': f'Item {i}', 'logic_score': 8, 'symbolic_score': 2}, 'FOLLOW_LOGIC')
            elif score_type == 1:
                memory.store({'text': f'Item {i}', 'logic_score': 2, 'symbolic_score': 8}, 'FOLLOW_SYMBOLIC')
            else:
                memory.store({'text': f'Item {i}', 'logic_score': 5, 'symbolic_score': 5}, 'FOLLOW_HYBRID')
                
        # Generate report
        report = analyzer.generate_evolution_report(
            migration_summary={'total_migrated': 2, 'to_logic': 1, 'to_symbolic': 1},
            audit_summary={'total_reversed': 1, 'from_logic': 1}
        )
        
        assert 'summary' in report
        assert 'health' in report
        assert 'bridge_analysis' in report
        assert 'stability_metrics' in report
        
        # Test pretty printing
        analyzer.print_report(report)
        
        print("✅ Evolution report generation works")
        
    # Test 4: Health indicators
    print("\n4️⃣ Test: Health indicators")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        analyzer = MemoryAnalyzer(memory, data_dir=tmpdir)
        
        # Create unhealthy distribution (too much in bridge)
        for i in range(2):
            memory.store({'text': f'Logic {i}', 'logic_score': 8, 'symbolic_score': 2}, 'FOLLOW_LOGIC')
        for i in range(8):
            memory.store({'text': f'Bridge {i}', 'logic_score': 5, 'symbolic_score': 5}, 'FOLLOW_HYBRID')
            
        stats = analyzer.get_memory_stats()
        health = stats['health_indicators']
        
        assert health['status'] == 'needs_attention'
        assert len(health['issues']) > 0
        assert len(health['recommendations']) > 0
        assert health['bridge_percentage'] > 50
        
        print("✅ Health indicators detect issues correctly")
        
    # Test 5: Empty memory handling
    print("\n5️⃣ Test: Empty memory handling")
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = HistoryAwareMemory(data_dir=tmpdir)
        analyzer = MemoryAnalyzer(memory, data_dir=tmpdir)
        
        # Generate report with empty memory
        report = analyzer.generate_evolution_report()
        
        # This should not raise KeyError
        analyzer.print_report(report)
        
        assert report['health']['status'] == 'empty'
        assert 'recommendations' in report['health']
        assert len(report['health']['recommendations']) > 0
        
        print("✅ Empty memory handling works correctly")
        
    print("\n✅ All analytics tests passed!")