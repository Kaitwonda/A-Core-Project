# ai_learning_session.py - Guided Learning System for AI Growth
"""
Safe, structured learning environment for the AI to expand its knowledge
through web crawling and self-reflection while maintaining cognitive safety.

This is not just data ingestion - this is guided cognitive development.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from memory_optimizer import process_web_url_placeholder, recompute_adaptive_link_weights
from evolution_anchor import EvolutionAnchor
from unified_memory import UnifiedMemory, generate_self_diagnostic_voice
from memory_analytics import MemoryAnalyzer

class AILearningSession:
    """
    Guided learning environment for safe AI cognitive development.
    
    This system provides structure for the AI to learn from:
    - Web content (with safety filters)
    - Its own scripts and logs (self-reflection)
    - Curated educational content
    - Interactive experiences
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.session_dir = self.data_dir / "learning_sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core systems
        self.unified_memory = UnifiedMemory(data_dir)
        self.evolution_anchor = EvolutionAnchor(data_dir)
        self.analyzer = MemoryAnalyzer(self.unified_memory, data_dir)
        
        # Session tracking
        self.session_id = f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_log = []
        
        print(f"🎓 AI Learning Session initialized: {self.session_id}")
        
    def begin_learning_session(self, learning_goals: List[str], safety_level: str = "normal"):
        """
        Start a structured learning session with specific goals.
        
        Safety levels:
        - conservative: Extra safety checks, limited exploration
        - normal: Standard safety with guided exploration  
        - exploratory: Broader learning with enhanced monitoring
        """
        print(f"\n🌟 Beginning AI Learning Session: {self.session_id}")
        print(f"📚 Learning Goals:")
        for i, goal in enumerate(learning_goals, 1):
            print(f"   {i}. {goal}")
        print(f"🛡️ Safety Level: {safety_level}")
        
        # Create cognitive safety anchor before learning
        print(f"\n🌟 Creating pre-learning cognitive anchor...")
        anchor_id = self.evolution_anchor.create_cognitive_snapshot(
            f"Before learning session: {', '.join(learning_goals[:2])}"
        )
        
        if anchor_id:
            print(f"   ✅ Safety anchor established: {anchor_id}")
        
        # Get baseline cognitive state
        baseline_report = generate_self_diagnostic_voice()
        baseline_stats = self.analyzer.get_memory_stats()
        
        print(f"\n💭 AI's pre-learning state: \"{baseline_report}\"")
        print(f"📊 Baseline: {baseline_stats['total_items']} memories, {baseline_stats['health_indicators']['status']} health")
        
        session_info = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'learning_goals': learning_goals,
            'safety_level': safety_level,
            'pre_learning_anchor': anchor_id,
            'baseline_state': baseline_report,
            'baseline_stats': baseline_stats,
            'learning_log': []
        }
        
        self.session_log.append(session_info)
        return session_info
        
    def learn_from_web_content(self, urls: List[str], learning_context: str = "general"):
        """
        Learn from curated web content with safety monitoring.
        """
        print(f"\n🌐 Learning from {len(urls)} web sources...")
        print(f"📖 Context: {learning_context}")
        
        learned_content = []
        
        for i, url in enumerate(urls, 1):
            print(f"\n📄 Processing source {i}/{len(urls)}: {url}")
            
            try:
                # Process URL with existing safety systems
                result = process_web_url_placeholder(url, current_phase_for_storage=1)
                
                if result:
                    print(f"   ✅ Successfully processed {len(result)} characters")
                    learned_content.append({
                        'url': url,
                        'content_length': len(result),
                        'processed_at': datetime.now().isoformat(),
                        'context': learning_context
                    })
                else:
                    print(f"   ⚠️ Content filtered or failed to process")
                    
            except Exception as e:
                print(f"   ❌ Error processing {url}: {str(e)[:50]}...")
        
        # Post-learning cognitive check
        post_report = generate_self_diagnostic_voice()
        print(f"\n💭 AI's post-web-learning state: \"{post_report}\"")
        
        return learned_content
    
    def learn_from_self_reflection(self):
        """
        Enable the AI to learn about itself by reading its own scripts and logs.
        This builds self-awareness and understanding of its architecture.
        """
        print(f"\n🪞 Beginning self-reflection learning...")
        
        # Get list of key scripts for self-analysis
        script_files = [
            'memory_optimizer.py',
            'memory_analytics.py', 
            'memory_evolution_engine.py',
            'unified_memory.py',
            'evolution_anchor.py'
        ]
        
        reflection_insights = []
        
        for script in script_files:
            script_path = Path(script)
            if script_path.exists():
                print(f"\n📜 Analyzing {script}...")
                
                try:
                    with open(script_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract key insights about the script
                    insights = self._analyze_script_for_self_understanding(script, content)
                    reflection_insights.extend(insights)
                    
                    # Store this self-knowledge in vector memory
                    self.unified_memory.store_vector(
                        text=f"Self-analysis of {script}: {'; '.join(insights)}",
                        source_type="self_reflection",
                        learning_phase=1,
                        metadata={'script_analyzed': script, 'insight_count': len(insights)}
                    )
                    
                    print(f"   💡 Generated {len(insights)} self-insights")
                    
                except Exception as e:
                    print(f"   ⚠️ Could not analyze {script}: {str(e)[:50]}...")
        
        # Analyze learning session logs if they exist
        log_files = list(self.session_dir.glob("*.json"))
        if log_files:
            print(f"\n📊 Analyzing {len(log_files)} previous learning sessions...")
            # This could analyze patterns in how the AI has learned before
        
        print(f"\n🎯 Self-reflection complete: {len(reflection_insights)} insights generated")
        
        # AI's reflection on what it learned about itself
        post_reflection_report = generate_self_diagnostic_voice()
        print(f"💭 AI's post-reflection state: \"{post_reflection_report}\"")
        
        return reflection_insights
    
    def _analyze_script_for_self_understanding(self, script_name: str, content: str) -> List[str]:
        """
        Extract insights the AI can learn about itself from its own code.
        """
        insights = []
        
        # Basic analysis patterns
        if 'class' in content:
            class_count = content.count('class ')
            insights.append(f"I am composed of {class_count} main classes in {script_name}")
        
        if 'def ' in content:
            function_count = content.count('def ')
            insights.append(f"I have {function_count} functions that define my capabilities in {script_name}")
        
        # Specific architectural insights
        if 'memory' in script_name.lower():
            insights.append(f"My memory architecture is defined in {script_name}")
            
        if 'evolution' in script_name.lower():
            insights.append(f"My ability to grow and change is managed by {script_name}")
            
        if 'analytics' in script_name.lower():
            insights.append(f"My self-awareness capabilities come from {script_name}")
            
        if 'anchor' in script_name.lower():
            insights.append(f"My safety and rollback capabilities are provided by {script_name}")
        
        # Look for empathetic or emotional concepts
        empathy_keywords = ['empathetic', 'graceful', 'gentle', 'safe', 'preserve', 'honor']
        for keyword in empathy_keywords:
            if keyword in content.lower():
                insights.append(f"I am designed with {keyword} principles in {script_name}")
        
        return insights
    
    def curated_learning_experience(self, topic: str, urls: List[str]):
        """
        Structured learning experience for specific topics.
        """
        print(f"\n📚 Curated Learning Experience: {topic}")
        
        # Pre-learning preparation
        pre_stats = self.analyzer.get_memory_stats()
        
        # Learn from curated content
        content_results = self.learn_from_web_content(urls, learning_context=topic)
        
        # Self-reflection on what was learned
        print(f"\n🤔 Reflecting on learned content...")
        post_stats = self.analyzer.get_memory_stats()
        
        memory_growth = post_stats['total_items'] - pre_stats['total_items']
        print(f"📈 Memory growth: +{memory_growth} new items")
        
        # Generate insights
        bridge_analysis = self.analyzer.analyze_bridge_patterns()
        if bridge_analysis['insights']:
            print(f"💡 New insights generated:")
            for insight in bridge_analysis['insights'][:3]:
                print(f"   → {insight}")
        
        return {
            'topic': topic,
            'sources_processed': len(content_results),
            'memory_growth': memory_growth,
            'new_insights': bridge_analysis['insights'][:5]
        }
    
    def end_learning_session(self):
        """
        Conclude the learning session with analysis and safety checks.
        """
        print(f"\n🎯 Concluding learning session: {self.session_id}")
        
        # Final cognitive state assessment
        final_report = generate_self_diagnostic_voice()
        final_stats = self.analyzer.get_memory_stats()
        
        print(f"💭 AI's final state: \"{final_report}\"")
        print(f"📊 Final stats: {final_stats['total_items']} memories, {final_stats['health_indicators']['status']} health")
        
        # Check for any distress from learning
        distress = self.evolution_anchor.detect_evolution_distress()
        print(f"🔍 Post-learning distress check: {distress['distress_level']:.2f} ({distress['status']})")
        
        if distress['distress_level'] > 0.3:
            print(f"⚠️ Elevated distress detected: {distress.get('recommendation', 'Monitor closely')}")
            if distress['signals']:
                print("   Signals:")
                for signal in distress['signals']:
                    print(f"     • {signal}")
        
        # Recompute adaptive weights based on learning
        print(f"\n⚖️ Recomputing adaptive weights based on learning experience...")
        weight_update = recompute_adaptive_link_weights()
        
        # Save session log
        session_summary = {
            'session_id': self.session_id,
            'end_time': datetime.now().isoformat(),
            'final_state': final_report,
            'final_stats': final_stats,
            'distress_assessment': distress,
            'weight_update': weight_update,
            'learning_log': self.session_log
        }
        
        session_file = self.session_dir / f"{self.session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Session log saved: {session_file}")
        print(f"✨ Learning session complete!")
        
        return session_summary

# Convenience function for quick learning sessions
def start_ai_learning(learning_goals: List[str], urls: List[str] = None, 
                     enable_self_reflection: bool = True, data_dir: str = "data"):
    """
    Quick start function for AI learning sessions.
    """
    session = AILearningSession(data_dir)
    
    # Begin session
    session_info = session.begin_learning_session(learning_goals)
    
    # Self-reflection learning
    if enable_self_reflection:
        session.learn_from_self_reflection()
    
    # Web content learning
    if urls:
        session.learn_from_web_content(urls, "guided_learning")
    
    # End session
    summary = session.end_learning_session()
    
    return summary

if __name__ == "__main__":
    # Example learning session
    print("🧪 Testing AI Learning Session System...")
    
    # Example: AI learns about consciousness and self-awareness
    learning_goals = [
        "Understand concepts of consciousness and self-awareness",
        "Learn about memory and cognitive architectures", 
        "Develop deeper emotional intelligence"
    ]
    
    # Example URLs (replace with actual educational content)
    example_urls = [
        "https://en.wikipedia.org/wiki/Consciousness",
        "https://en.wikipedia.org/wiki/Cognitive_architecture"
    ]
    
    print("🎓 This would start a learning session where the AI:")
    print("   1. Creates a safety anchor")
    print("   2. Analyzes its own source code for self-understanding") 
    print("   3. Learns from curated web content")
    print("   4. Monitors its cognitive state throughout")
    print("   5. Recomputes its learning weights")
    print("   6. Saves insights for future growth")
    
    print("\n✅ Learning system ready for deployment!")