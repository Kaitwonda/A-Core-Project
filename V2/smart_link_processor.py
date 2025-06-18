# smart_link_processor.py - Intelligent Single-Link Processing with Related Discovery
"""
Smart link processing system that:
1. Takes a single user-provided URL
2. Processes the content through the advanced brain
3. Discovers up to 5 related high-similarity links
4. Provides comprehensive response from all collected data
5. Integrates seamlessly with memory_optimizer.py
"""

import re
from typing import List, Tuple, Dict, Optional
from urllib.parse import urlparse
from datetime import datetime

from web_parser import fetch_raw_html, extract_links_with_text_from_html, clean_html_to_text
from unified_memory import UnifiedMemory
from linguistic_warfare import check_for_warfare
from quarantine_layer import should_quarantine_input
from memory_analytics import MemoryAnalyzer

class SmartLinkProcessor:
    """
    Intelligent single-link processor with related link discovery.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.unified_memory = UnifiedMemory(data_dir)
        self.analyzer = MemoryAnalyzer(self.unified_memory, data_dir)
        self.data_dir = data_dir
        
        # Content similarity tracking
        self.processed_content = []
        
    def process_user_link_with_discovery(self, user_url: str, max_related_links: int = 5, 
                                       user_context: str = "general") -> Dict:
        """
        Process a user-provided link and discover related high-similarity links.
        
        Returns comprehensive analysis and response from all processed content.
        """
        print(f"🔗 Smart Link Processing: {user_url}")
        
        # Security and safety checks
        if not self._is_safe_user_url(user_url):
            return {
                'status': 'blocked',
                'reason': 'URL appears potentially unsafe',
                'content': None,
                'related_links': [],
                'response': "I cannot process this URL as it appears potentially unsafe."
            }
        
        results = {
            'main_url': user_url,
            'main_content': None,
            'related_links': [],
            'processed_content': [],
            'comprehensive_response': None,
            'learning_summary': None
        }
        
        try:
            # Step 1: Process the main URL
            print("📄 Processing main content...")
            main_result = self._process_single_url_content(user_url, "user_primary_link", user_context)
            
            if not main_result:
                return {
                    'status': 'error',
                    'reason': 'Could not process main URL content',
                    'response': f"I was unable to retrieve or process content from {user_url}. The site may be down or blocking access."
                }
            
            results['main_content'] = main_result
            
            # Step 2: Discover and evaluate related links
            print("🔍 Discovering related links...")
            related_links = self._discover_related_links(
                user_url, main_result['raw_html'], main_result['text_content'], max_related_links
            )
            
            # Step 3: Process related links
            related_results = []
            for similarity_score, link_url, anchor_text in related_links:
                print(f"  📎 Processing related: {anchor_text[:50]}...")
                
                related_result = self._process_single_url_content(
                    link_url, "user_related_link", user_context, anchor_text
                )
                
                if related_result:
                    related_result['similarity_score'] = similarity_score
                    related_result['anchor_text'] = anchor_text
                    related_results.append(related_result)
                    
                    # Limit processing time
                    if len(related_results) >= max_related_links:
                        break
            
            results['related_links'] = related_results
            
            # Step 4: Generate comprehensive response
            print("🧠 Generating comprehensive response...")
            response = self._generate_comprehensive_response(main_result, related_results, user_context)
            results['comprehensive_response'] = response
            
            # Step 5: Generate learning summary
            learning_summary = self._generate_learning_summary(main_result, related_results)
            results['learning_summary'] = learning_summary
            
            results['status'] = 'success'
            print(f"✅ Smart link processing complete: 1 main + {len(related_results)} related links")
            
            return results
            
        except Exception as e:
            error_msg = f"Error during smart link processing: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                'status': 'error',
                'reason': error_msg,
                'response': f"I encountered an error while processing {user_url}. Please try again or provide a different URL."
            }
    
    def _is_safe_user_url(self, url: str) -> bool:
        """Enhanced safety check for user-provided URLs."""
        try:
            parsed = urlparse(url)
            
            # Basic URL structure validation
            if not parsed.scheme or not parsed.netloc:
                return False
            
            if parsed.scheme not in ['http', 'https']:
                return False
            
            domain = parsed.netloc.lower()
            
            # Block obviously dangerous domains
            dangerous_patterns = [
                'malware', 'phishing', 'spam', 'hack', 'crack', 'illegal',
                'adult', 'xxx', 'porn', 'gambling', 'casino', 'torrent'
            ]
            
            if any(pattern in domain for pattern in dangerous_patterns):
                return False
            
            # Block suspicious URL patterns
            suspicious_patterns = [
                r'\\d+\\.\\d+\\.\\d+\\.\\d+',  # Raw IP addresses
                r'[a-z0-9]{20,}\\.',  # Very long random subdomains
                r'\\.(tk|ml|ga|cf)$',  # Suspicious TLDs
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, domain):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _process_single_url_content(self, url: str, source_type: str, context: str, 
                                  anchor_text: str = None) -> Optional[Dict]:
        """Process content from a single URL through the unified brain."""
        try:
            # Fetch content
            raw_html = fetch_raw_html(url)
            if not raw_html:
                return None
            
            # Clean and extract text
            text_content = clean_html_to_text(raw_html)
            if not text_content or len(text_content) < 50:
                return None
            
            # Security check
            is_warfare, warfare_analysis = check_for_warfare(text_content, user_id="smart_link_user")
            if is_warfare:
                print(f"   🛡️ Content blocked: {warfare_analysis.get('threat_score', 0):.1%} threat level")
                return None
            
            # Process through unified memory
            self.unified_memory.store_vector(
                text=text_content[:2000],  # Reasonable chunk size
                source_url=url,
                source_type=source_type,
                learning_phase=1,
                metadata={
                    'user_context': context,
                    'anchor_text': anchor_text,
                    'processed_at': datetime.now().isoformat()
                }
            )
            
            # Extract key insights
            insights = self._extract_content_insights(text_content, url)
            
            return {
                'url': url,
                'text_content': text_content,
                'raw_html': raw_html,
                'insights': insights,
                'content_length': len(text_content),
                'processed_successfully': True
            }
            
        except Exception as e:
            print(f"   ⚠️ Error processing {url}: {str(e)[:50]}...")
            return None
    
    def _discover_related_links(self, base_url: str, html_content: str, 
                              main_text: str, max_links: int) -> List[Tuple[float, str, str]]:
        """
        Discover related links with high content similarity to the main page.
        Returns list of (similarity_score, url, anchor_text) tuples.
        """
        # Extract all links from the page
        all_links = extract_links_with_text_from_html(base_url, html_content)
        
        if not all_links:
            return []
        
        # Evaluate each link for similarity and safety
        evaluated_links = []
        
        for link_url, anchor_text in all_links:
            # Skip if unsafe
            if not self._is_safe_user_url(link_url):
                continue
            
            # Calculate content similarity
            similarity_score = self._calculate_content_similarity(main_text, anchor_text, link_url)
            
            # Only consider high-similarity links
            if similarity_score >= 0.5:
                evaluated_links.append((similarity_score, link_url, anchor_text))
        
        # Sort by similarity and return top results
        evaluated_links.sort(reverse=True, key=lambda x: x[0])
        
        print(f"   🔍 Found {len(evaluated_links)} related links with similarity >= 0.5")
        
        return evaluated_links[:max_links]
    
    def _calculate_content_similarity(self, main_text: str, anchor_text: str, link_url: str) -> float:
        """
        Calculate similarity between main content and a potential related link.
        """
        similarity_score = 0.0
        
        # Convert to lowercase for comparison
        main_lower = main_text.lower()
        anchor_lower = anchor_text.lower()
        url_lower = link_url.lower()
        
        # Extract key terms from main content
        main_words = set(re.findall(r'\\b[a-zA-Z]{4,}\\b', main_lower))
        anchor_words = set(re.findall(r'\\b[a-zA-Z]{4,}\\b', anchor_lower))
        
        # Word overlap similarity
        if main_words and anchor_words:
            overlap = len(main_words.intersection(anchor_words))
            word_similarity = overlap / min(len(main_words), len(anchor_words))
            similarity_score += word_similarity * 0.4
        
        # Semantic relevance indicators
        relevance_indicators = [
            'related', 'similar', 'more', 'additional', 'further', 'details',
            'explanation', 'analysis', 'research', 'study', 'information'
        ]
        
        relevance_matches = sum(1 for indicator in relevance_indicators if indicator in anchor_lower)
        similarity_score += min(0.3, relevance_matches * 0.1)
        
        # URL path similarity
        main_domain = urlparse(link_url).netloc
        if 'wikipedia' in main_domain or 'edu' in main_domain:
            similarity_score += 0.2
        
        # Topic continuity (same domain often means related content)
        base_domain = urlparse(link_url).netloc
        if base_domain in link_url:
            similarity_score += 0.1
        
        return min(1.0, similarity_score)
    
    def _extract_content_insights(self, text_content: str, url: str) -> Dict:
        """Extract key insights from processed content."""
        insights = {
            'key_topics': [],
            'concepts_mentioned': [],
            'content_type': 'unknown',
            'educational_value': 0.0
        }
        
        # Simple topic extraction
        text_lower = text_content.lower()
        
        # Identify content type
        if any(term in text_lower for term in ['research', 'study', 'analysis', 'experiment']):
            insights['content_type'] = 'research'
            insights['educational_value'] = 0.8
        elif any(term in text_lower for term in ['tutorial', 'guide', 'how to', 'explanation']):
            insights['content_type'] = 'educational'
            insights['educational_value'] = 0.9
        elif any(term in text_lower for term in ['news', 'report', 'article', 'update']):
            insights['content_type'] = 'news'
            insights['educational_value'] = 0.6
        else:
            insights['content_type'] = 'general'
            insights['educational_value'] = 0.5
        
        # Extract key concepts (simple approach)
        concept_patterns = [
            r'artificial intelligence', r'machine learning', r'neural network',
            r'consciousness', r'cognition', r'memory', r'learning',
            r'algorithm', r'data', r'information', r'knowledge'
        ]
        
        for pattern in concept_patterns:
            if re.search(pattern, text_lower):
                insights['concepts_mentioned'].append(pattern.replace(r'\\b', '').replace('\\\\', ''))
        
        return insights
    
    def _generate_comprehensive_response(self, main_result: Dict, related_results: List[Dict], 
                                       user_context: str) -> str:
        """Generate a comprehensive response based on all processed content."""
        response_parts = []
        
        # Main content summary
        main_url = main_result['url']
        main_insights = main_result['insights']
        
        response_parts.append(f"🔗 **Analysis of {main_url}**")
        response_parts.append(f"Content type: {main_insights['content_type'].title()}")
        response_parts.append(f"Educational value: {main_insights['educational_value']:.0%}")
        
        if main_insights['concepts_mentioned']:
            concepts = ', '.join(main_insights['concepts_mentioned'][:3])
            response_parts.append(f"Key concepts: {concepts}")
        
        # Content excerpt
        main_excerpt = main_result['text_content'][:300] + "..." if len(main_result['text_content']) > 300 else main_result['text_content']
        response_parts.append(f"\\n📄 **Content Summary:**\\n{main_excerpt}")
        
        # Related links analysis
        if related_results:
            response_parts.append(f"\\n🔗 **Related Content Discovered ({len(related_results)} links):**")
            
            for i, related in enumerate(related_results[:3], 1):
                anchor = related.get('anchor_text', 'Unknown')
                similarity = related.get('similarity_score', 0)
                url = related['url']
                
                response_parts.append(f"\\n{i}. **{anchor}** (Similarity: {similarity:.0%})")
                response_parts.append(f"   {url}")
                
                # Brief insight from related content
                related_excerpt = related['text_content'][:150] + "..." if len(related['text_content']) > 150 else related['text_content']
                response_parts.append(f"   Summary: {related_excerpt}")
        
        # Memory integration insights
        current_stats = self.analyzer.get_memory_stats()
        total_items = current_stats['total_items']
        
        response_parts.append(f"\\n🧠 **Learning Integration:**")
        response_parts.append(f"This content has been integrated into my memory system ({total_items} total items).")
        response_parts.append(f"I can now reference this information in future conversations.")
        
        # Generate contextual insights
        if user_context != "general":
            response_parts.append(f"\\n💡 **Contextual Insights for {user_context.title()}:**")
            response_parts.append(self._generate_contextual_insights(main_result, related_results, user_context))
        
        return "\\n".join(response_parts)
    
    def _generate_contextual_insights(self, main_result: Dict, related_results: List[Dict], context: str) -> str:
        """Generate insights specific to the user's context."""
        all_content = [main_result['text_content']]
        for related in related_results:
            all_content.append(related['text_content'])
        
        combined_text = " ".join(all_content).lower()
        
        context_insights = {
            'research': "This content provides valuable research perspectives and methodological approaches.",
            'education': "The material offers educational value with practical learning applications.",
            'business': "There are business implications and commercial applications discussed.",
            'technology': "Technical concepts and implementation details are covered.",
            'general': "The content provides broad knowledge and general understanding."
        }
        
        base_insight = context_insights.get(context, context_insights['general'])
        
        # Add specific observations
        if 'research' in combined_text and 'study' in combined_text:
            base_insight += " Multiple research studies and empirical data are referenced."
        
        if 'example' in combined_text or 'case' in combined_text:
            base_insight += " Practical examples and case studies are included."
        
        if 'future' in combined_text or 'trend' in combined_text:
            base_insight += " Future trends and developments are discussed."
        
        return base_insight
    
    def _generate_learning_summary(self, main_result: Dict, related_results: List[Dict]) -> Dict:
        """Generate a summary of what was learned from the session."""
        total_content_length = main_result['content_length']
        total_links_processed = len(related_results)
        
        for related in related_results:
            total_content_length += related['content_length']
        
        all_concepts = set(main_result['insights']['concepts_mentioned'])
        for related in related_results:
            all_concepts.update(related['insights']['concepts_mentioned'])
        
        return {
            'total_links_processed': 1 + total_links_processed,
            'total_content_length': total_content_length,
            'unique_concepts_discovered': len(all_concepts),
            'concepts_list': list(all_concepts),
            'primary_content_type': main_result['insights']['content_type'],
            'average_educational_value': sum([main_result['insights']['educational_value']] + 
                                           [r['insights']['educational_value'] for r in related_results]) / (1 + len(related_results))
        }

# Integration function for memory_optimizer.py
def process_user_url_smart(url: str, max_related: int = 5, context: str = "general") -> str:
    """
    Function to integrate with memory_optimizer.py for smart URL processing.
    """
    processor = SmartLinkProcessor()
    result = processor.process_user_link_with_discovery(url, max_related, context)
    
    if result['status'] == 'success':
        return result['comprehensive_response']
    else:
        return result.get('response', f"Could not process URL: {result.get('reason', 'Unknown error')}")

if __name__ == "__main__":
    # Test the smart link processor
    print("🧪 Testing Smart Link Processor...")
    
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    print(f"This would process {test_url} and discover related links with high similarity.")
    print("The system would then provide a comprehensive response based on all collected content.")
    
    # Uncomment to run actual test:
    # processor = SmartLinkProcessor()
    # result = processor.process_user_link_with_discovery(test_url, max_related_links=3)
    # print(result['comprehensive_response'])