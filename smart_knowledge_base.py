#!/usr/bin/env python3
"""
Smart Knowledge Base with Semantic Topic Understanding
Enhanced RAG system with concept relationships and topic modeling
"""
import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import re
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import advanced NLP libraries
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_ADVANCED_AVAILABLE = True
except ImportError:
    SKLEARN_ADVANCED_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️  networkx not available. Install with: pip install networkx")

from config import config
from obsidian_knowledge_base import ObsidianKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConceptGraph:
    """Build and maintain a concept relationship graph"""
    
    def __init__(self):
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            self.graph = None
            logger.warning("NetworkX not available - concept graph disabled")
        
        self.concept_frequencies = Counter()
        self.concept_cooccurrences = defaultdict(Counter)
        self.topic_clusters = {}
    
    def add_concept(self, concept: str, document_id: str, context: str = ""):
        """Add a concept to the graph"""
        if not self.graph:
            return
            
        concept = concept.lower().strip()
        if len(concept) < 2:
            return
            
        # Add node if it doesn't exist
        if not self.graph.has_node(concept):
            self.graph.add_node(concept, 
                               frequency=0, 
                               documents=set(), 
                               contexts=[])
        
        # Update node data
        self.graph.nodes[concept]['frequency'] += 1
        self.graph.nodes[concept]['documents'].add(document_id)
        if context:
            self.graph.nodes[concept]['contexts'].append(context[:200])  # Limit context length
        
        self.concept_frequencies[concept] += 1
    
    def add_relationship(self, concept1: str, concept2: str, 
                        relationship_type: str = "related", strength: float = 1.0):
        """Add a relationship between concepts"""
        if not self.graph:
            return
            
        concept1, concept2 = concept1.lower().strip(), concept2.lower().strip()
        
        if concept1 == concept2 or len(concept1) < 2 or len(concept2) < 2:
            return
        
        # Ensure both nodes exist
        for concept in [concept1, concept2]:
            if not self.graph.has_node(concept):
                self.graph.add_node(concept, frequency=0, documents=set(), contexts=[])
        
        # Add or update edge
        if self.graph.has_edge(concept1, concept2):
            current_weight = self.graph[concept1][concept2].get('weight', 0)
            self.graph[concept1][concept2]['weight'] = current_weight + strength
        else:
            self.graph.add_edge(concept1, concept2, 
                              relationship=relationship_type, 
                              weight=strength)
        
        # Update co-occurrence tracking
        self.concept_cooccurrences[concept1][concept2] += 1
        self.concept_cooccurrences[concept2][concept1] += 1
    
    def find_related_concepts(self, concept: str, max_concepts: int = 10) -> List[Tuple[str, float]]:
        """Find concepts related to the given concept"""
        if not self.graph:
            return []
            
        concept = concept.lower().strip()
        if not self.graph.has_node(concept):
            return []
        
        related = []
        
        # Direct neighbors
        for neighbor in self.graph.neighbors(concept):
            weight = self.graph[concept][neighbor].get('weight', 1.0)
            frequency = self.graph.nodes[neighbor].get('frequency', 1)
            score = weight * np.log(frequency + 1)  # Weight by frequency
            related.append((neighbor, score))
        
        # Indirect neighbors (2 hops away)
        for neighbor in self.graph.neighbors(concept):
            for second_hop in self.graph.neighbors(neighbor):
                if second_hop != concept and second_hop not in [r[0] for r in related]:
                    weight = (self.graph[concept][neighbor].get('weight', 1.0) * 
                             self.graph[neighbor][second_hop].get('weight', 1.0)) * 0.5  # Decay
                    frequency = self.graph.nodes[second_hop].get('frequency', 1)
                    score = weight * np.log(frequency + 1)
                    related.append((second_hop, score))
        
        # Sort by score and return top concepts
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:max_concepts]
    
    def get_concept_cluster(self, concept: str) -> Set[str]:
        """Get the cluster of related concepts for a given concept"""
        if not self.graph:
            return {concept}
            
        concept = concept.lower().strip()
        if concept in self.topic_clusters:
            return self.topic_clusters[concept]
        
        # Find strongly connected components or use simple traversal
        try:
            if NETWORKX_AVAILABLE:
                # Find all concepts within 2 hops
                cluster = {concept}
                for neighbor in self.graph.neighbors(concept):
                    cluster.add(neighbor)
                    for second_hop in self.graph.neighbors(neighbor):
                        cluster.add(second_hop)
                
                self.topic_clusters[concept] = cluster
                return cluster
        except Exception:
            pass
        
        return {concept}
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        if not self.graph:
            return {}
        
        return {
            'total_concepts': self.graph.number_of_nodes(),
            'total_relationships': self.graph.number_of_edges(),
            'avg_connections': self.graph.number_of_edges() / max(self.graph.number_of_nodes(), 1),
            'most_connected': sorted(
                [(node, len(list(self.graph.neighbors(node)))) 
                 for node in self.graph.nodes()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }


class SmartKnowledgeBase(ObsidianKnowledgeBase):
    """Enhanced knowledge base with semantic understanding"""
    
    def __init__(self, vault_path: str = None):
        super().__init__(vault_path)
        
        # Additional components for smart understanding
        self.concept_graph = ConceptGraph()
        self.topic_model = None
        self.concept_embeddings = {}
        self.domain_classifiers = {}
        
        # Enhanced cache files
        self.concept_graph_file = self.index_dir / "concept_graph.pkl"
        self.topic_model_file = self.index_dir / "topic_model.pkl"
        
        # Topic extraction patterns
        self.topic_patterns = {
            'mathematical_concepts': r'\b(?:theorem|lemma|corollary|definition|axiom|property|equation|formula)\b',
            'methods_techniques': r'\b(?:method|technique|algorithm|approach|procedure|process|strategy)\b',
            'applications': r'\b(?:application|use|implementation|example|case study|real[- ]world)\b',
            'relationships': r'\b(?:related to|connected to|based on|derived from|implies|leads to)\b'
        }
    
    def _extract_concepts_from_document(self, doc: Dict[str, str]) -> List[str]:
        """Extract key concepts from a document"""
        content = doc.get('content', '')
        title = doc.get('title', '')
        
        concepts = set()
        
        # Extract from title
        title_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title)
        concepts.update(word.lower() for word in title_words if len(word) > 3)
        
        # Extract headers
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        for header in headers:
            clean_header = re.sub(r'[^\w\s]', '', header).strip().lower()
            if 4 <= len(clean_header) <= 50:
                concepts.add(clean_header)
        
        # Extract mathematical terms
        math_terms = re.findall(r'\$([^$]+)\$', content)
        for term in math_terms:
            clean_term = re.sub(r'[\\{}]', '', term).strip().lower()
            if 3 <= len(clean_term) <= 20 and not clean_term.isdigit():
                concepts.add(clean_term)
        
        # Extract capitalized technical terms
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', content)
        for term in technical_terms:
            if 4 <= len(term) <= 30:
                concepts.add(term.lower())
        
        # Extract bold/emphasized terms
        emphasized = re.findall(r'\*\*([^*]+)\*\*|\*([^*]+)\*', content)
        for match in emphasized:
            term = (match[0] or match[1]).strip().lower()
            if 3 <= len(term) <= 30:
                concepts.add(term)
        
        return list(concepts)
    
    def _extract_concept_relationships(self, doc: Dict[str, str], concepts: List[str]):
        """Extract relationships between concepts within a document"""
        content = doc['content'].lower()
        doc_id = doc['file_path']
        
        # Add concepts to graph
        for concept in concepts:
            self.concept_graph.add_concept(concept, doc_id, content[:500])
        
        # Find co-occurrence relationships
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Check if concepts appear near each other
                pattern1 = rf'\b{re.escape(concept1)}\b'
                pattern2 = rf'\b{re.escape(concept2)}\b'
                
                matches1 = [(m.start(), m.end()) for m in re.finditer(pattern1, content)]
                matches2 = [(m.start(), m.end()) for m in re.finditer(pattern2, content)]
                
                # Calculate minimum distance between concept mentions
                min_distance = float('inf')
                for start1, end1 in matches1:
                    for start2, end2 in matches2:
                        distance = min(abs(start1 - start2), abs(end1 - end2))
                        min_distance = min(min_distance, distance)
                
                # Add relationship if concepts are mentioned close together
                if min_distance < 200:  # Within 200 characters
                    strength = 1.0 / (1.0 + min_distance / 50.0)  # Closer = stronger
                    self.concept_graph.add_relationship(concept1, concept2, "co_occurs", strength)
        
        # Extract explicit relationships using patterns
        for pattern_name, pattern in self.topic_patterns.items():
            matches = re.findall(f'([^.!?]+{pattern}[^.!?]+)', content)
            for match in matches:
                # Find concepts in this sentence
                sentence_concepts = [c for c in concepts if c in match.lower()]
                if len(sentence_concepts) >= 2:
                    # Add stronger relationships for explicitly mentioned connections
                    for i, concept1 in enumerate(sentence_concepts):
                        for concept2 in sentence_concepts[i+1:]:
                            self.concept_graph.add_relationship(
                                concept1, concept2, pattern_name, 2.0)
    
    def index_vault(self, force_reindex: bool = False):
        """Enhanced vault indexing with concept extraction"""
        # First, run the base indexing
        success = super().index_vault(force_reindex)
        
        if not success or not self.documents:
            return False
        
        logger.info("🧠 Building concept graph and semantic relationships...")
        
        # Clear previous concept graph if reindexing
        if force_reindex:
            self.concept_graph = ConceptGraph()
        
        # Extract concepts and relationships from all documents
        for doc in self.documents:
            concepts = self._extract_concepts_from_document(doc)
            if concepts:
                self._extract_concept_relationships(doc, concepts)
        
        # Build topic model if enough documents
        if SKLEARN_ADVANCED_AVAILABLE and len(self.documents) >= 5:
            self._build_topic_model()
        
        # Save enhanced index
        self._save_enhanced_index()
        
        # Log statistics
        stats = self.concept_graph.get_statistics()
        logger.info(f"📊 Concept graph: {stats.get('total_concepts', 0)} concepts, {stats.get('total_relationships', 0)} relationships")
        
        return True
    
    def _build_topic_model(self):
        """Build LDA topic model from documents"""
        if not SKLEARN_ADVANCED_AVAILABLE:
            return
            
        try:
            # Prepare documents for topic modeling
            doc_texts = [doc['content'] for doc in self.documents if len(doc['content']) > 100]
            
            if len(doc_texts) < 5:
                return
            
            # Use TF-IDF for topic modeling
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2)
            )
            
            doc_term_matrix = vectorizer.fit_transform(doc_texts)
            
            # Fit LDA model
            n_topics = min(10, len(doc_texts) // 2)  # Reasonable number of topics
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
            
            lda.fit(doc_term_matrix)
            
            self.topic_model = {
                'model': lda,
                'vectorizer': vectorizer,
                'feature_names': vectorizer.get_feature_names_out()
            }
            
            logger.info(f"📈 Topic model built with {n_topics} topics")
            
        except Exception as e:
            logger.warning(f"Topic model creation failed: {e}")
    
    def smart_search(self, query: str, top_k: int = 5, 
                    include_related_concepts: bool = True) -> List[Dict]:
        """Enhanced search with concept understanding"""
        
        # Start with regular search
        base_results = self.search(query, top_k * 2)  # Get more results initially
        
        if not include_related_concepts or not self.concept_graph.graph:
            return base_results[:top_k]
        
        # Extract concepts from query
        query_concepts = self._extract_concepts_from_query(query)
        
        if not query_concepts:
            return base_results[:top_k]
        
        # Find related concepts
        expanded_concepts = set(query_concepts)
        for concept in query_concepts:
            related = self.concept_graph.find_related_concepts(concept, 5)
            expanded_concepts.update([rel[0] for rel in related])
        
        # Re-rank results based on concept relevance
        enhanced_results = []
        for result in base_results:
            doc_concepts = self._extract_concepts_from_document(result)
            
            # Calculate concept overlap score
            concept_overlap = len(set(doc_concepts) & expanded_concepts)
            concept_bonus = concept_overlap * 0.2  # Bonus for concept matches
            
            result['relevance_score'] += concept_bonus
            result['matched_concepts'] = list(set(doc_concepts) & expanded_concepts)
            enhanced_results.append(result)
        
        # Sort by enhanced relevance score
        enhanced_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return enhanced_results[:top_k]
    
    def _extract_concepts_from_query(self, query: str) -> List[str]:
        """Extract concepts from search query"""
        concepts = []
        
        # Simple extraction - look for technical terms
        technical_terms = re.findall(r'\b[A-Za-z][a-z]*(?:\s+[A-Za-z][a-z]*)*\b', query)
        for term in technical_terms:
            if len(term) >= 3 and term.lower() not in {'the', 'and', 'for', 'with', 'what', 'how', 'why'}:
                concepts.append(term.lower())
        
        return concepts
    
    def get_topic_suggestions(self, current_topics: Set[str], max_suggestions: int = 5) -> List[Dict]:
        """Get smart topic suggestions based on current knowledge"""
        if not self.concept_graph.graph:
            return []
        
        suggestions = []
        
        # Find concepts that are related to current topics but not already known
        for topic in current_topics:
            related = self.concept_graph.find_related_concepts(topic, 10)
            for concept, score in related:
                if concept not in current_topics:
                    # Get documents that mention this concept
                    if self.concept_graph.graph.has_node(concept):
                        docs = self.concept_graph.graph.nodes[concept].get('documents', set())
                        suggestion = {
                            'concept': concept,
                            'relevance_score': score,
                            'related_to': topic,
                            'document_count': len(docs),
                            'contexts': self.concept_graph.graph.nodes[concept].get('contexts', [])[:3]
                        }
                        suggestions.append(suggestion)
        
        # Sort by relevance and remove duplicates
        suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion['concept'] not in seen:
                seen.add(suggestion['concept'])
                unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= max_suggestions:
                    break
        
        return unique_suggestions
    
    def _save_enhanced_index(self):
        """Save the enhanced index with concept graph"""
        super()._save_index()  # Save base index
        
        # Save concept graph
        if self.concept_graph.graph and NETWORKX_AVAILABLE:
            try:
                with open(self.concept_graph_file, 'wb') as f:
                    pickle.dump({
                        'graph': self.concept_graph.graph,
                        'concept_frequencies': self.concept_graph.concept_frequencies,
                        'topic_clusters': self.concept_graph.topic_clusters
                    }, f)
            except Exception as e:
                logger.warning(f"Failed to save concept graph: {e}")
        
        # Save topic model
        if self.topic_model:
            try:
                with open(self.topic_model_file, 'wb') as f:
                    pickle.dump(self.topic_model, f)
            except Exception as e:
                logger.warning(f"Failed to save topic model: {e}")
    
    def _load_index(self):
        """Load the enhanced index"""
        success = super()._load_index()  # Load base index
        
        if not success:
            return False
        
        # Load concept graph
        if self.concept_graph_file.exists():
            try:
                with open(self.concept_graph_file, 'rb') as f:
                    graph_data = pickle.load(f)
                    self.concept_graph.graph = graph_data.get('graph')
                    self.concept_graph.concept_frequencies = graph_data.get('concept_frequencies', Counter())
                    self.concept_graph.topic_clusters = graph_data.get('topic_clusters', {})
                    
                logger.info("📊 Concept graph loaded")
            except Exception as e:
                logger.warning(f"Failed to load concept graph: {e}")
        
        # Load topic model
        if self.topic_model_file.exists():
            try:
                with open(self.topic_model_file, 'rb') as f:
                    self.topic_model = pickle.load(f)
                logger.info("📈 Topic model loaded")
            except Exception as e:
                logger.warning(f"Failed to load topic model: {e}")
        
        return True
    
    def get_enhanced_statistics(self) -> Dict:
        """Get comprehensive statistics about the knowledge base"""
        base_stats = super().get_statistics()
        
        enhanced_stats = {
            **base_stats,
            'concept_graph': self.concept_graph.get_statistics(),
            'has_topic_model': self.topic_model is not None,
            'enhanced_features': {
                'concept_relationships': NETWORKX_AVAILABLE,
                'topic_modeling': SKLEARN_ADVANCED_AVAILABLE,
                'smart_search': True
            }
        }
        
        return enhanced_stats


def main():
    """Test the smart knowledge base"""
    import sys
    
    vault_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    kb = SmartKnowledgeBase(vault_path)
    
    print("🧠 Testing Smart Knowledge Base...")
    
    # Index the vault
    success = kb.index_vault()
    if not success:
        print("❌ Failed to index vault")
        return
    
    # Show enhanced statistics
    stats = kb.get_enhanced_statistics()
    print(f"\n📊 Enhanced Knowledge Base Stats:")
    print(f"   📄 Documents: {stats.get('total_documents', 0)}")
    print(f"   🧠 Concepts: {stats.get('concept_graph', {}).get('total_concepts', 0)}")
    print(f"   🔗 Relationships: {stats.get('concept_graph', {}).get('total_relationships', 0)}")
    print(f"   📈 Topic Model: {'✅' if stats.get('has_topic_model') else '❌'}")
    
    # Test smart search
    test_queries = [
        "machine learning algorithms",
        "linear algebra eigenvalues",
        "probability distributions"
    ]
    
    print(f"\n🔍 Testing smart search...")
    for query in test_queries:
        results = kb.smart_search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['title']} (score: {result['relevance_score']:.3f})")
            if 'matched_concepts' in result:
                concepts = ', '.join(result['matched_concepts'][:5])
                print(f"      Concepts: {concepts}")
    
    # Test topic suggestions
    current_topics = {'machine learning', 'neural networks'}
    suggestions = kb.get_topic_suggestions(current_topics)
    
    if suggestions:
        print(f"\n💡 Topic suggestions based on current knowledge:")
        for suggestion in suggestions[:5]:
            print(f"   • {suggestion['concept']} (related to {suggestion['related_to']})")
    
    print("\n✅ Smart knowledge base testing completed!")

if __name__ == "__main__":
    main()