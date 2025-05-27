# brain_metrics.py
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict

class BrainMetrics:
    def __init__(self, metrics_path="data/brain_contribution_metrics.json",
                 conflicts_path="data/bridge_conflicts.json",
                 decisions_log_path="data/link_decisions.csv"):
        self.metrics_file = Path(metrics_path)
        self.conflicts_file = Path(conflicts_path)
        self.decisions_log = Path(decisions_log_path)
        
        # Ensure parent dirs exist
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing or initialize
        if not self.metrics_file.exists():
            self._write_json(self.metrics_file, [])
        if not self.conflicts_file.exists():
            self._write_json(self.conflicts_file, [])

        # Session tracking
        self.session = {
            "start_time": datetime.utcnow(),
            "logic_wins": 0,
            "symbol_wins": 0,
            "hybrid_decisions": 0,
            "conflicts": [],
            "phase_decisions": defaultdict(lambda: {"logic": 0, "symbol": 0, "hybrid": 0}),
            "url_decisions": []
        }

    def _write_json(self, path, data):
        """Write JSON data to file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _read_json(self, path):
        """Read JSON data from file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def log_decision(self, decision_type, url, logic_score, symbol_score, phase=1, link_text=""):
        """Log a single link decision"""
        # Count decisions by type
        if decision_type == "FOLLOW_LOGIC":
            self.session["logic_wins"] += 1
            self.session["phase_decisions"][phase]["logic"] += 1
        elif decision_type == "FOLLOW_SYMBOLIC":
            self.session["symbol_wins"] += 1
            self.session["phase_decisions"][phase]["symbol"] += 1
        else:  # FOLLOW_HYBRID
            self.session["hybrid_decisions"] += 1
            self.session["phase_decisions"][phase]["hybrid"] += 1

        # Record significant conflicts (where brains strongly disagree)
        logic_conf = min(1.0, logic_score / 10.0)
        symbol_conf = min(1.0, symbol_score / 5.0)
        
        # Conflict detection: one brain very confident, other not
        if (logic_conf > 0.7 and symbol_conf < 0.3) or (symbol_conf > 0.7 and logic_conf < 0.3):
            conflict_severity = abs(logic_score - symbol_score)
            if conflict_severity > 5.0:
                self.session["conflicts"].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "url": url[:100],
                    "link_text": link_text[:50],
                    "logic_score": round(logic_score, 2),
                    "symbol_score": round(symbol_score, 2),
                    "logic_confidence": round(logic_conf, 2),
                    "symbol_confidence": round(symbol_conf, 2),
                    "decision": decision_type,
                    "severity": round(conflict_severity, 2),
                    "phase": phase
                })

        # Store detailed decision
        self.session["url_decisions"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "url": url[:100],
            "decision_type": decision_type,
            "logic_score": round(logic_score, 2),
            "symbol_score": round(symbol_score, 2),
            "phase": phase
        })

    def save_session_metrics(self):
        """Save session metrics to persistent storage"""
        # Load history
        history = self._read_json(self.metrics_file)

        total = (self.session["logic_wins"] + 
                self.session["symbol_wins"] + 
                self.session["hybrid_decisions"])
        
        if total == 0:
            print("📊 No decisions made in this session to save.")
            return

        # Calculate rates
        metrics = {
            "timestamp": self.session["start_time"].isoformat(),
            "duration_seconds": (datetime.utcnow() - self.session["start_time"]).total_seconds(),
            "logic_win_rate": round(self.session["logic_wins"] / total, 3),
            "symbol_win_rate": round(self.session["symbol_wins"] / total, 3),
            "hybrid_rate": round(self.session["hybrid_decisions"] / total, 3),
            "total_decisions": total,
            "logic_wins": self.session["logic_wins"],
            "symbol_wins": self.session["symbol_wins"],
            "hybrid_decisions": self.session["hybrid_decisions"],
            "phase_breakdown": dict(self.session["phase_decisions"]),
            "conflicts_count": len(self.session["conflicts"])
        }
        
        history.append(metrics)
        self._write_json(self.metrics_file, history)
        
        print(f"📊 Session metrics saved: Logic={metrics['logic_win_rate']:.1%}, "
              f"Symbol={metrics['symbol_win_rate']:.1%}, Hybrid={metrics['hybrid_rate']:.1%} "
              f"({total} decisions)")

        # Append conflicts to conflicts file
        if self.session["conflicts"]:
            conflicts = self._read_json(self.conflicts_file)
            conflicts.extend(self.session["conflicts"])
            self._write_json(self.conflicts_file, conflicts)
            print(f"⚡ Logged {len(self.session['conflicts'])} brain conflicts")

    def generate_report(self):
        """Generate a visual report of brain contributions"""
        history = self._read_json(self.metrics_file)
        if not history:
            print("📊 No metrics history to report.")
            return

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Brain Metrics Analysis", fontsize=16)

        # 1. Win rates over time
        ax1 = axes[0, 0]
        timestamps = [pd.to_datetime(m['timestamp']) for m in history]
        logic_rates = [m['logic_win_rate'] for m in history]
        symbol_rates = [m['symbol_win_rate'] for m in history]
        hybrid_rates = [m['hybrid_rate'] for m in history]
        
        ax1.plot(timestamps, logic_rates, 'b-', label='Logic', marker='o')
        ax1.plot(timestamps, symbol_rates, 'r-', label='Symbolic', marker='s')
        ax1.plot(timestamps, hybrid_rates, 'g-', label='Hybrid', marker='^')
        ax1.set_xlabel('Session Time')
        ax1.set_ylabel('Decision Rate')
        ax1.set_title('Brain Decision Rates Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Total decisions pie chart (most recent session)
        ax2 = axes[0, 1]
        if history:
            latest = history[-1]
            sizes = [latest['logic_wins'], latest['symbol_wins'], latest['hybrid_decisions']]
            labels = ['Logic', 'Symbolic', 'Hybrid']
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Latest Session Distribution ({latest["total_decisions"]} decisions)')

        # 3. Conflicts analysis
        ax3 = axes[1, 0]
        conflicts = self._read_json(self.conflicts_file)
        if conflicts:
            severities = [c['severity'] for c in conflicts]
            ax3.hist(severities, bins=20, color='orange', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Conflict Severity (Score Difference)')
            ax3.set_ylabel('Count')
            ax3.set_title(f'Brain Conflict Distribution ({len(conflicts)} total)')
            ax3.grid(True, alpha=0.3)

        # 4. Phase breakdown (if available)
        ax4 = axes[1, 1]
        if history and 'phase_breakdown' in history[-1]:
            phase_data = history[-1]['phase_breakdown']
            if phase_data:
                phases = list(phase_data.keys())
                logic_counts = [phase_data[p]['logic'] for p in phases]
                symbol_counts = [phase_data[p]['symbol'] for p in phases]
                hybrid_counts = [phase_data[p]['hybrid'] for p in phases]
                
                x = range(len(phases))
                width = 0.25
                ax4.bar([i - width for i in x], logic_counts, width, label='Logic', color='#3498db')
                ax4.bar(x, symbol_counts, width, label='Symbolic', color='#e74c3c')
                ax4.bar([i + width for i in x], hybrid_counts, width, label='Hybrid', color='#2ecc71')
                
                ax4.set_xlabel('Phase')
                ax4.set_ylabel('Decisions')
                ax4.set_title('Decisions by Phase')
                ax4.set_xticks(x)
                ax4.set_xticklabels([f'Phase {p}' for p in phases])
                ax4.legend()
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        report_path = Path("data/brain_metrics_report.png")
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        print(f"📊 Report saved to {report_path}")
        plt.show()

    def get_adaptive_weights(self):
        """Calculate recommended weights based on performance"""
        history = self._read_json(self.metrics_file)
        if len(history) < 3:  # Need minimum history
            return None
            
        # Average over recent sessions
        recent = history[-5:]  # Last 5 sessions
        avg_logic_rate = sum(m['logic_win_rate'] for m in recent) / len(recent)
        avg_symbol_rate = sum(m['symbol_win_rate'] for m in recent) / len(recent)
        
        # Normalize to get weights
        total = avg_logic_rate + avg_symbol_rate
        if total > 0:
            recommended_static = avg_logic_rate / total
            recommended_dynamic = avg_symbol_rate / total
        else:
            recommended_static = 0.6
            recommended_dynamic = 0.4
            
        return {
            "link_score_weight_static": round(recommended_static, 3),
            "link_score_weight_dynamic": round(recommended_dynamic, 3),
            "based_on_sessions": len(recent),
            "confidence": "high" if len(history) > 10 else "medium" if len(history) > 5 else "low"
        }

    def analyze_conflicts(self):
        """Analyze conflict patterns"""
        conflicts = self._read_json(self.conflicts_file)
        if not conflicts:
            print("⚡ No conflicts recorded yet.")
            return
            
        print(f"\n⚡ Conflict Analysis ({len(conflicts)} total conflicts)")
        print("=" * 50)
        
        # Group by decision outcome
        by_decision = defaultdict(list)
        for c in conflicts:
            by_decision[c['decision']].append(c)
        
        for decision_type, conflict_list in by_decision.items():
            avg_severity = sum(c['severity'] for c in conflict_list) / len(conflict_list)
            print(f"\n{decision_type}:")
            print(f"  Count: {len(conflict_list)}")
            print(f"  Avg Severity: {avg_severity:.1f}")
            
            # Show top conflict
            if conflict_list:
                top_conflict = max(conflict_list, key=lambda x: x['severity'])
                print(f"  Worst conflict: Logic={top_conflict['logic_score']}, "
                      f"Symbol={top_conflict['symbol_score']}, "
                      f"URL: {top_conflict['url'][:50]}...")


# Standalone utility functions
def display_metrics_summary():
    """Quick summary of current metrics"""
    metrics = BrainMetrics()
    history = metrics._read_json(metrics.metrics_file)
    
    if not history:
        print("📊 No metrics history found.")
        return
        
    latest = history[-1]
    print("\n📊 Latest Session Summary:")
    print(f"  Time: {latest['timestamp']}")
    print(f"  Total Decisions: {latest['total_decisions']}")
    print(f"  Logic Win Rate: {latest['logic_win_rate']:.1%}")
    print(f"  Symbol Win Rate: {latest['symbol_win_rate']:.1%}")
    print(f"  Hybrid Rate: {latest['hybrid_rate']:.1%}")
    print(f"  Conflicts: {latest.get('conflicts_count', 0)}")
    
    # Overall trends
    if len(history) > 1:
        logic_trend = latest['logic_win_rate'] - history[-2]['logic_win_rate']
        symbol_trend = latest['symbol_win_rate'] - history[-2]['symbol_win_rate']
        print(f"\n📈 Trends vs Previous Session:")
        print(f"  Logic: {'+' if logic_trend >= 0 else ''}{logic_trend:.1%}")
        print(f"  Symbol: {'+' if symbol_trend >= 0 else ''}{symbol_trend:.1%}")


if __name__ == "__main__":
    # Test the module
    print("🧠 Brain Metrics Module")
    print("=" * 50)
    
    # Create instance
    metrics = BrainMetrics()
    
    # Display current summary
    display_metrics_summary()
    
    # Analyze conflicts
    metrics.analyze_conflicts()
    
    # Check for adaptive weights
    weights = metrics.get_adaptive_weights()
    if weights:
        print(f"\n🎯 Recommended Adaptive Weights ({weights['confidence']} confidence):")
        print(f"  Static (Logic): {weights['link_score_weight_static']:.1%}")
        print(f"  Dynamic (Symbol): {weights['link_score_weight_dynamic']:.1%}")
    
    # Generate visual report
    try:
        metrics.generate_report()
    except Exception as e:
        print(f"⚠️ Could not generate visual report: {e}")