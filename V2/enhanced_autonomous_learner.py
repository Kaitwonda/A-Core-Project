# enhanced_autonomous_learner.py - Massive Web Learning with Advanced Brain Integration
"""
Enhanced autonomous learning system that can:
1. Process 500+ URLs autonomously with smart link following
2. Use advanced brain architecture (tripartite memory, evolution, etc.)
3. Context-aware link evaluation and discovery
4. Full integration with security and cognitive safeguards
"""

import time
import random
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from urllib.parse import urlparse, urljoin
from collections import deque, defaultdict

# Core system imports
from unified_memory import UnifiedMemory
from memory_analytics import MemoryAnalyzer
from evolution_anchor import EvolutionAnchor
from web_parser import fetch_raw_html, extract_links_with_text_from_html, clean_html_to_text
from linguistic_warfare import check_for_warfare
from quarantine_layer import should_quarantine_input

class EnhancedAutonomousLearner:
    """
    Advanced autonomous learning system with massive web crawling capabilities
    and full brain integration.
    """
    
    def __init__(self, data_dir: str = "data"):
        print("🧠 Initializing Enhanced Autonomous Learner...")
        
        self.data_dir = Path(data_dir)
        self.session_dir = self.data_dir / "autonomous_sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Core brain components
        self.unified_memory = UnifiedMemory(data_dir)
        self.analyzer = MemoryAnalyzer(self.unified_memory, data_dir)
        self.evolution_anchor = EvolutionAnchor(data_dir)
        
        # Web crawling state
        self.url_queue = deque()
        self.processed_urls = set()
        self.deferred_urls = deque()
        self.session_hot_keywords = set()
        self.domain_stats = defaultdict(int)
        
        # Learning session tracking
        self.session_id = None
        self.session_stats = {
            'urls_processed': 0,
            'chunks_learned': 0,
            'symbols_discovered': 0,
            'links_followed': 0,
            'security_blocks': 0
        }
        
        # Safety and quality controls
        self.max_depth = 3
        self.max_urls_per_domain = 50
        self.content_similarity_threshold = 0.7
        self.safety_threshold = 0.8
        
        print("✅ Enhanced Autonomous Learner ready for massive learning!")
    
    def start_massive_learning_session(self, seed_urls: List[str], target_urls: int = 500, 
                                     learning_focus: str = "general"):
        """
        Start a massive autonomous learning session processing hundreds of URLs.
        """
        print(f"\n🚀 MASSIVE LEARNING SESSION STARTING")
        print(f"🎯 Target: {target_urls} URLs")
        print(f"📚 Focus: {learning_focus}")
        print("=" * 50)
        
        # Create session ID and safety anchor
        self.session_id = f"massive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        anchor_id = self.evolution_anchor.create_cognitive_snapshot(
            f"Before massive learning: {target_urls} URLs on {learning_focus}"
        )
        
        if anchor_id:
            print(f"🌟 Safety anchor created: {anchor_id}")
        
        # Initialize learning context
        self._initialize_learning_context(seed_urls, learning_focus)
        
        # Reset session stats
        self.session_stats = {k: 0 for k in self.session_stats}
        
        start_time = time.time()
        
        try:
            # Main learning loop
            while (len(self.processed_urls) < target_urls and 
                   (self.url_queue or self.deferred_urls)):
                
                # Process URLs from queue
                self._process_url_batch(batch_size=10)
                
                # Periodic cognitive health check
                if self.session_stats['urls_processed'] % 50 == 0:
                    self._cognitive_health_check()
                
                # Evolution cycle every 100 URLs
                if self.session_stats['urls_processed'] % 100 == 0:
                    self._run_evolution_cycle()
                
                # Brief pause to avoid overwhelming servers
                time.sleep(1)
            
            # Session complete
            elapsed_time = time.time() - start_time
            self._finalize_learning_session(elapsed_time)
            
        except KeyboardInterrupt:
            print("\n⚠️ Learning session interrupted by user")
            self._emergency_session_save()
        except Exception as e:
            print(f"\n❌ Learning session error: {e}")
            self._emergency_session_save()
    
    def _initialize_learning_context(self, seed_urls: List[str], learning_focus: str):
        """Initialize the learning context and seed the URL queue."""
        print(f"\n🌱 Initializing learning context...")
        
        # Set up focus keywords based on learning area
        focus_keywords = {
            'ai_consciousness': ['consciousness', 'artificial intelligence', 'cognition', 'awareness', 'sentience'],
            'science': ['research', 'study', 'analysis', 'experiment', 'discovery'],
            'philosophy': ['ethics', 'morality', 'existence', 'meaning', 'truth'],
            'technology': ['innovation', 'development', 'engineering', 'programming', 'algorithm'],
            'general': ['learning', 'knowledge', 'information', 'understanding', 'insight']
        }
        
        self.session_hot_keywords = set(focus_keywords.get(learning_focus, focus_keywords['general']))
        
        # Seed the queue with initial URLs
        for url in seed_urls:
            if self._is_safe_domain(url):
                self.url_queue.append({
                    'url': url,
                    'depth': 0,
                    'priority': 1.0,
                    'source': 'seed',
                    'context': learning_focus
                })
                print(f"   🌱 Seeded: {url}")
            else:
                print(f"   ⚠️ Skipped unsafe seed: {url}")
    
    def _process_url_batch(self, batch_size: int = 10):
        """Process a batch of URLs from the queue."""
        batch_urls = []
        
        # Get batch from queue
        for _ in range(min(batch_size, len(self.url_queue))):
            if self.url_queue:
                batch_urls.append(self.url_queue.popleft())
        
        # Process each URL in the batch
        for url_info in batch_urls:
            if url_info['url'] not in self.processed_urls:
                self._process_single_url(url_info)
    
    def _process_single_url(self, url_info: Dict):
        """Process a single URL with full brain integration."""
        url = url_info['url']
        
        print(f"\n📄 Processing: {url[:60]}...")
        
        try:
            # Fetch content
            html_content = fetch_raw_html(url)
            if not html_content:
                print("   ❌ Failed to fetch content")
                return
            
            # Clean and extract text
            text_content = clean_html_to_text(html_content)
            if not text_content or len(text_content) < 100:
                print("   ⚠️ Insufficient content")
                return
            
            # Security check
            is_warfare, warfare_analysis = check_for_warfare(text_content, user_id="autonomous_learner")
            if is_warfare:
                print(f"   🛡️ Blocked: {warfare_analysis.get('threat_score', 0):.1%} threat level")
                self.session_stats['security_blocks'] += 1
                return
            
            # Process content through unified brain
            result = self._process_content_with_brain(text_content, url, url_info)
            
            if result:
                # Extract and evaluate links for further exploration
                self._discover_and_queue_links(url, html_content, url_info)
                
                # Update session stats
                self.session_stats['urls_processed'] += 1
                self.session_stats['chunks_learned'] += 1
                
                print(f"   ✅ Processed successfully")
            
            # Mark as processed
            self.processed_urls.add(url)
            
        except Exception as e:
            print(f"   ❌ Error processing {url}: {str(e)[:50]}...")
    
    def _process_content_with_brain(self, text_content: str, source_url: str, url_info: Dict) -> bool:
        """Process content through the unified brain architecture."""
        try:
            # Store in vector memory with learning context
            self.unified_memory.store_vector(
                text=text_content[:2000],  # Reasonable chunk size
                source_url=source_url,
                source_type="autonomous_web_learning",
                learning_phase=1,
                metadata={
                    'learning_focus': url_info.get('context', 'general'),
                    'discovery_depth': url_info.get('depth', 0),
                    'session_id': self.session_id
                }
            )
            
            # Extract and update symbols
            from unified_memory import update_symbol_emotions, generate_symbol_from_context
            
            # Simple emotion prediction for context
            emotions = self._predict_content_emotions(text_content)
            
            # Extract keywords for symbol generation
            keywords = self._extract_keywords(text_content)
            
            if keywords:
                # Try to generate new symbols from content
                new_symbol = generate_symbol_from_context(text_content, keywords, emotions)
                if new_symbol:
                    self.session_stats['symbols_discovered'] += 1
                    print(f"   💡 Generated symbol: {new_symbol['symbol']} - {new_symbol['name']}")
            
            return True
            
        except Exception as e:
            print(f"   ⚠️ Brain processing error: {str(e)[:50]}...")
            return False
    
    def _discover_and_queue_links(self, base_url: str, html_content: str, parent_info: Dict):
        """Discover and intelligently queue related links for exploration."""
        if parent_info.get('depth', 0) >= self.max_depth:
            return
        
        # Extract all links
        links = extract_links_with_text_from_html(base_url, html_content)
        
        evaluated_links = []
        for link_url, anchor_text in links:
            # Evaluate link for relevance and safety
            action, priority, reason = self._evaluate_link_for_learning(
                link_url, anchor_text, parent_info
            )
            
            if action == "FOLLOW_NOW":
                evaluated_links.append((priority, link_url, anchor_text, parent_info['depth'] + 1))
        
        # Sort by priority and add to queue
        evaluated_links.sort(reverse=True, key=lambda x: x[0])
        
        added_count = 0
        for priority, link_url, anchor_text, depth in evaluated_links[:10]:  # Limit to top 10
            if link_url not in self.processed_urls and self._check_domain_limits(link_url):
                self.url_queue.append({
                    'url': link_url,
                    'depth': depth,
                    'priority': priority,
                    'source': base_url,
                    'context': parent_info.get('context', 'general'),
                    'anchor_text': anchor_text
                })
                added_count += 1
        
        if added_count > 0:
            print(f"   🔗 Queued {added_count} related links")
            self.session_stats['links_followed'] += added_count
    
    def _evaluate_link_for_learning(self, link_url: str, anchor_text: str, parent_info: Dict) -> Tuple[str, float, str]:
        """
        Evaluate a link for learning value using context-aware scoring.
        Returns: (action, priority, reason)
        """
        # Basic safety check
        if not self._is_safe_domain(link_url):
            return "SKIP", 0.0, "unsafe_domain"
        
        # Content relevance scoring
        relevance_score = 0.0
        
        # Check anchor text against session keywords
        anchor_lower = anchor_text.lower()
        keyword_matches = sum(1 for keyword in self.session_hot_keywords if keyword in anchor_lower)
        relevance_score += keyword_matches * 0.3
        
        # Educational content indicators
        educational_indicators = [
            'research', 'study', 'analysis', 'theory', 'concept', 'principle',
            'explanation', 'tutorial', 'guide', 'introduction', 'overview'
        ]
        education_score = sum(1 for indicator in educational_indicators if indicator in anchor_lower)
        relevance_score += education_score * 0.2
        
        # URL structure scoring
        url_lower = link_url.lower()
        if any(domain in url_lower for domain in ['edu', 'wikipedia', 'stanford', 'mit']):
            relevance_score += 0.4
        
        if any(path in url_lower for path in ['article', 'research', 'paper', 'study']):
            relevance_score += 0.2
        
        # Context matching with parent
        if parent_info.get('context') in anchor_lower:
            relevance_score += 0.3
        
        # Determine action based on score
        if relevance_score >= 0.7:
            return "FOLLOW_NOW", relevance_score, f"high_relevance_{relevance_score:.2f}"
        elif relevance_score >= 0.4:
            return "DEFER", relevance_score, f"moderate_relevance_{relevance_score:.2f}"
        else:
            return "SKIP", relevance_score, f"low_relevance_{relevance_score:.2f}"
    
    def _is_safe_domain(self, url: str) -> bool:
        """Check if a domain is safe for learning."""
        try:
            domain = urlparse(url).netloc.lower()
            
            # Blocked domains
            blocked_domains = [
                'malware', 'phishing', 'spam', 'adult', 'gambling',
                'torrent', 'illegal', 'hack', 'crack'
            ]
            
            if any(blocked in domain for blocked in blocked_domains):
                return False
            
            # Preferred educational domains
            educational_domains = [
                'wikipedia.org', 'edu', 'stanford.edu', 'mit.edu',
                'arxiv.org', 'scholar.google', 'researchgate.net'
            ]
            
            # Allow educational domains without further checks
            if any(edu_domain in domain for edu_domain in educational_domains):
                return True
            
            # General safety checks for other domains
            return len(domain) > 3 and '.' in domain
            
        except Exception:
            return False
    
    def _check_domain_limits(self, url: str) -> bool:
        """Check if we haven't exceeded domain processing limits."""
        try:
            domain = urlparse(url).netloc
            return self.domain_stats[domain] < self.max_urls_per_domain
        except Exception:
            return False
    
    def _predict_content_emotions(self, text: str) -> List[Tuple[str, float]]:
        """Simple emotion prediction for content context."""
        emotion_keywords = {
            'joy': ['happy', 'celebration', 'success', 'achievement', 'positive'],
            'curiosity': ['discover', 'explore', 'investigate', 'research', 'study'],
            'confidence': ['certain', 'proven', 'established', 'confirmed', 'verified'],
            'neutral': ['analysis', 'examination', 'review', 'evaluation', 'assessment']
        }
        
        text_lower = text.lower()
        emotions = []
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotions.append((emotion, min(0.8, score * 0.2)))
        
        return emotions if emotions else [('neutral', 0.5)]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text for symbol generation."""
        # Simple keyword extraction (could be enhanced with NLP)
        words = re.findall(r'\\b[a-zA-Z]{4,}\\b', text.lower())
        
        # Filter to relevant terms
        relevant_words = []
        for word in set(words):
            if (word in self.session_hot_keywords or 
                any(keyword in word for keyword in self.session_hot_keywords)):
                relevant_words.append(word)
        
        return relevant_words[:5]  # Limit to top 5
    
    def _cognitive_health_check(self):
        """Check AI's cognitive health during learning."""
        print(f"\\n🔍 Cognitive health check at {self.session_stats['urls_processed']} URLs...")
        
        # Check for learning distress
        distress = self.evolution_anchor.detect_evolution_distress()
        
        if distress['distress_level'] > 0.5:
            print(f"   ⚠️ Elevated distress: {distress['distress_level']:.2f}")
            print(f"   💭 Recommendation: {distress.get('recommendation', 'Monitor closely')}")
            
            # If distress is very high, pause learning
            if distress['distress_level'] > 0.8:
                print("   🛑 High distress detected - pausing learning for 30 seconds...")
                time.sleep(30)
        else:
            print("   ✅ Cognitive health stable")
    
    def _run_evolution_cycle(self):
        """Run a memory evolution cycle during learning."""
        print(f"\\n🧬 Running evolution cycle at {self.session_stats['urls_processed']} URLs...")
        
        try:
            from memory_evolution_engine import run_memory_evolution
            
            result = run_memory_evolution(data_dir=self.data_dir)
            
            if result and result.get('success'):
                print(f"   ✅ Evolution complete: {result.get('migrated', 0)} migrated, {result.get('reversed', 0)} reversed")
            else:
                print("   ⚠️ Evolution cycle had issues")
                
        except Exception as e:
            print(f"   ❌ Evolution error: {str(e)[:50]}...")
    
    def _finalize_learning_session(self, elapsed_time: float):
        """Finalize and save the learning session."""
        print(f"\\n🎯 MASSIVE LEARNING SESSION COMPLETE")
        print("=" * 50)
        
        # Final stats
        print(f"⏱️ Duration: {elapsed_time/60:.1f} minutes")
        print(f"📊 URLs processed: {self.session_stats['urls_processed']}")
        print(f"🧠 Chunks learned: {self.session_stats['chunks_learned']}")
        print(f"💡 Symbols discovered: {self.session_stats['symbols_discovered']}")
        print(f"🔗 Links followed: {self.session_stats['links_followed']}")
        print(f"🛡️ Security blocks: {self.session_stats['security_blocks']}")
        
        # Final cognitive assessment
        final_stats = self.analyzer.get_memory_stats()
        print(f"🧠 Final memory: {final_stats['total_items']} items, {final_stats['health_indicators']['status']} health")
        
        # Final evolution cycle
        print(f"\\n🧬 Running final evolution cycle...")
        self._run_evolution_cycle()
        
        # Save session log
        session_summary = {
            'session_id': self.session_id,
            'completed_at': datetime.now().isoformat(),
            'elapsed_time_minutes': elapsed_time / 60,
            'stats': self.session_stats,
            'final_memory_stats': final_stats,
            'processed_domains': dict(self.domain_stats)
        }
        
        session_file = self.session_dir / f"{self.session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_summary, f, indent=2)
        
        print(f"💾 Session saved: {session_file}")
    
    def _emergency_session_save(self):
        """Emergency save during interrupted sessions."""
        print(f"\\n💾 Emergency session save...")
        
        emergency_data = {
            'session_id': self.session_id,
            'interrupted_at': datetime.now().isoformat(),
            'partial_stats': self.session_stats,
            'urls_in_queue': len(self.url_queue),
            'processed_count': len(self.processed_urls)
        }
        
        emergency_file = self.session_dir / f"{self.session_id}_emergency.json"
        with open(emergency_file, 'w', encoding='utf-8') as f:
            json.dump(emergency_data, f, indent=2)
        
        print(f"✅ Emergency data saved: {emergency_file}")

# Convenience function for quick massive learning
def start_massive_web_learning(seed_urls: List[str], target_urls: int = 500, 
                             focus: str = "general", data_dir: str = "data"):
    """
    Quick start function for massive autonomous web learning.
    """
    learner = EnhancedAutonomousLearner(data_dir)
    learner.start_massive_learning_session(seed_urls, target_urls, focus)
    return learner

if __name__ == "__main__":
    # Example massive learning session
    seed_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Consciousness"
    ]
    
    print("🧪 Testing Enhanced Autonomous Learner...")
    print("This would start a massive learning session processing hundreds of URLs")
    print("with advanced brain integration and cognitive safety monitoring.")
    
    # Uncomment to run actual test:
    # start_massive_web_learning(seed_urls, target_urls=50, focus="ai_consciousness")