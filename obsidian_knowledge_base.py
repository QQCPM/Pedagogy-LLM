#!/usr/bin/env python3
"""
Obsidian Knowledge Base Integration
RAG system using your Obsidian vault for enhanced responses
"""
import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging

# Core libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Try to import sentence transformers for better embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Using TF-IDF for embeddings.")
    print("   Install with: pip install sentence-transformers")

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObsidianKnowledgeBase:
    """RAG system for Obsidian vault integration"""
    
    def __init__(self, vault_path: str = None):
        self.vault_path = Path(vault_path) if vault_path else Path("/Users/quangnguyen/Downloads/hello")
        self.data_dir = Path(config.data_dir)
        self.index_dir = self.data_dir / "knowledge_index"
        self.index_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.documents = []
        self.document_embeddings = None
        self.tfidf_matrix = None
        
        # Cache files
        self.cache_file = self.index_dir / "document_cache.pkl"
        self.embeddings_file = self.index_dir / "embeddings.npy"
        self.tfidf_file = self.index_dir / "tfidf.pkl"
        self.metadata_file = self.index_dir / "metadata.json"
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use a lightweight model that works well for educational content
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Loaded SentenceTransformer for semantic search")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}")
                self.embedding_model = None
        
        # Always initialize TF-IDF as fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        logger.info("✅ Initialized TF-IDF vectorizer")
    
    def _extract_text_from_markdown(self, file_path: Path) -> Dict[str, str]:
        """Extract and clean text from markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata if present
            metadata = {}
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    # YAML frontmatter (simple parsing)
                    frontmatter = parts[1].strip()
                    content = parts[2].strip()
                    
                    for line in frontmatter.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
            
            # Clean markdown formatting but preserve structure
            # Remove code blocks but keep their content
            content = re.sub(r'```[\w]*\n(.*?)\n```', r'\\1', content, flags=re.DOTALL)
            
            # Remove image links but keep alt text
            content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\\1', content)
            
            # Remove regular links but keep text
            content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\\1', content)
            
            # Remove bold/italic formatting
            content = re.sub(r'\*\*([^\*]+)\*\*', r'\\1', content)
            content = re.sub(r'\*([^\*]+)\*', r'\\1', content)
            
            # Remove headers but keep text
            content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
            
            # Clean up extra whitespace
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = content.strip()
            
            # Extract title (from filename or first header)
            title = file_path.stem
            if content:
                first_lines = content.split('\n')[:3]
                for line in first_lines:
                    if line.strip() and len(line.strip()) < 100:
                        title = line.strip()
                        break
            
            return {
                'title': title,
                'content': content,
                'metadata': metadata,
                'file_path': str(file_path),
                'last_modified': file_path.stat().st_mtime
            }
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def _needs_reindexing(self) -> bool:
        """Check if reindexing is needed"""
        if not self.cache_file.exists():
            return True
        
        # Check if any files are newer than cache
        try:
            cache_time = self.cache_file.stat().st_mtime
            
            for md_file in self.vault_path.rglob("*.md"):
                if md_file.stat().st_mtime > cache_time:
                    return True
            
            return False
        except Exception:
            return True
    
    def index_vault(self, force_reindex: bool = False):
        """Index all markdown files in the vault"""
        if not force_reindex and not self._needs_reindexing():
            logger.info("📚 Loading existing index...")
            return self._load_index()
        
        logger.info(f"🔍 Indexing Obsidian vault: {self.vault_path}")
        
        if not self.vault_path.exists():
            logger.error(f"❌ Vault path does not exist: {self.vault_path}")
            return False
        
        # Find all markdown files
        md_files = list(self.vault_path.rglob("*.md"))
        logger.info(f"📄 Found {len(md_files)} markdown files")
        
        # Extract text from all files
        documents = []
        for md_file in md_files:
            doc = self._extract_text_from_markdown(md_file)
            if doc and doc['content'].strip():  # Only include non-empty documents
                documents.append(doc)
        
        if not documents:
            logger.warning("❌ No valid documents found in vault")
            return False
        
        logger.info(f"📚 Processed {len(documents)} valid documents")
        self.documents = documents
        
        # Create TF-IDF embeddings
        logger.info("🔧 Creating TF-IDF embeddings...")
        texts = [doc['content'] for doc in documents]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Create semantic embeddings if available
        if self.embedding_model:
            logger.info("🧠 Creating semantic embeddings...")
            
            # Process in batches to avoid memory issues
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts)
                embeddings.extend(batch_embeddings)
            
            self.document_embeddings = np.array(embeddings)
            logger.info(f"✅ Created embeddings for {len(embeddings)} documents")
        
        # Save index
        self._save_index()
        
        logger.info("✅ Vault indexing completed!")
        return True
    
    def _save_index(self):
        """Save the index to disk"""
        # Save document cache
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save TF-IDF
        with open(self.tfidf_file, 'wb') as f:
            pickle.dump({
                'vectorizer': self.tfidf_vectorizer,
                'matrix': self.tfidf_matrix
            }, f)
        
        # Save semantic embeddings if available
        if self.document_embeddings is not None:
            np.save(self.embeddings_file, self.document_embeddings)
        
        # Save metadata
        metadata = {
            'indexed_at': datetime.now().isoformat(),
            'total_documents': len(self.documents),
            'has_semantic_embeddings': self.document_embeddings is not None,
            'vault_path': str(self.vault_path)
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Index saved to {self.index_dir}")
    
    def _load_index(self):
        """Load the index from disk"""
        try:
            # Load documents
            with open(self.cache_file, 'rb') as f:
                self.documents = pickle.load(f)
            
            # Load TF-IDF
            with open(self.tfidf_file, 'rb') as f:
                tfidf_data = pickle.load(f)
                self.tfidf_vectorizer = tfidf_data['vectorizer']
                self.tfidf_matrix = tfidf_data['matrix']
            
            # Load semantic embeddings if available
            if self.embeddings_file.exists():
                self.document_embeddings = np.load(self.embeddings_file)
            
            logger.info(f"✅ Loaded index with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, hybrid_weight: float = 0.7) -> List[Dict]:
        """
        Search the knowledge base using hybrid search
        
        Args:
            query: Search query
            top_k: Number of top results to return
            hybrid_weight: Weight for semantic search (0=only TF-IDF, 1=only semantic)
        """
        if not self.documents:
            logger.warning("❌ No documents indexed. Run index_vault() first.")
            return []
        
        results = []
        
        # TF-IDF search
        tfidf_scores = self._tfidf_search(query)
        
        # Semantic search if available
        if self.document_embeddings is not None and self.embedding_model is not None:
            semantic_scores = self._semantic_search(query)
            
            # Combine scores using hybrid weighting
            combined_scores = []
            for i in range(len(self.documents)):
                tfidf_score = tfidf_scores[i] if i < len(tfidf_scores) else 0
                semantic_score = semantic_scores[i] if i < len(semantic_scores) else 0
                
                combined_score = (1 - hybrid_weight) * tfidf_score + hybrid_weight * semantic_score
                combined_scores.append((combined_score, i))
        else:
            # Use only TF-IDF
            combined_scores = [(score, i) for i, score in enumerate(tfidf_scores)]
        
        # Sort by combined score and get top results
        combined_scores.sort(reverse=True)
        
        for score, doc_idx in combined_scores[:top_k]:
            if score > 0:  # Only include relevant results
                doc = self.documents[doc_idx].copy()
                doc['relevance_score'] = score
                doc['snippet'] = self._create_snippet(doc['content'], query)
                results.append(doc)
        
        return results
    
    def _tfidf_search(self, query: str) -> List[float]:
        """Perform TF-IDF search"""
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        return similarities.tolist()
    
    def _semantic_search(self, query: str) -> List[float]:
        """Perform semantic search using sentence embeddings"""
        if self.embedding_model is None or self.document_embeddings is None:
            return [0.0] * len(self.documents)
        
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings).flatten()
        return similarities.tolist()
    
    def _create_snippet(self, content: str, query: str, max_length: int = 300) -> str:
        """Create a relevant snippet from content"""
        # Find the most relevant sentence containing query terms
        query_terms = query.lower().split()
        sentences = content.split('. ')
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        if best_sentence:
            # Expand context around the best sentence
            sentence_idx = sentences.index(best_sentence)
            start_idx = max(0, sentence_idx - 1)
            end_idx = min(len(sentences), sentence_idx + 2)
            
            snippet = '. '.join(sentences[start_idx:end_idx])
        else:
            # Fallback to beginning of content
            snippet = content
        
        # Truncate if too long
        if len(snippet) > max_length:
            snippet = snippet[:max_length] + "..."
        
        return snippet.strip()
    
    def get_relevant_context(self, question: str, max_context_length: int = 2000) -> str:
        """Get relevant context from knowledge base for a question"""
        # Search for relevant documents
        relevant_docs = self.search(question, top_k=3)
        
        if not relevant_docs:
            return ""
        
        # Compile context from top documents
        context_parts = []
        current_length = 0
        
        for doc in relevant_docs:
            if current_length >= max_context_length:
                break
            
            # Add document with title and snippet
            doc_context = f"**{doc['title']}**:\n{doc['snippet']}\n"
            
            if current_length + len(doc_context) <= max_context_length:
                context_parts.append(doc_context)
                current_length += len(doc_context)
            else:
                # Add partial content to fit within limit
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Only if meaningful space remaining
                    truncated_context = doc_context[:remaining_space] + "..."
                    context_parts.append(truncated_context)
                break
        
        return "\n".join(context_parts)
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        if not self.metadata_file.exists():
            return {"error": "No index found"}
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Add current stats
        metadata.update({
            'documents_loaded': len(self.documents),
            'has_semantic_search': self.document_embeddings is not None,
            'vault_accessible': self.vault_path.exists()
        })
        
        return metadata

def main():
    """Test the knowledge base system"""
    import sys
    
    vault_path = None
    if len(sys.argv) > 1:
        vault_path = sys.argv[1]
    
    kb = ObsidianKnowledgeBase(vault_path)
    
    print("🚀 Testing Obsidian Knowledge Base...")
    
    # Index the vault
    success = kb.index_vault()
    if not success:
        print("❌ Failed to index vault")
        return
    
    # Show statistics
    stats = kb.get_statistics()
    print(f"\n📊 Knowledge Base Stats:")
    print(f"   📄 Documents: {stats.get('total_documents', 0)}")
    print(f"   🧠 Semantic Search: {'✅' if stats.get('has_semantic_embeddings') else '❌'}")
    print(f"   📁 Vault Path: {stats.get('vault_path')}")
    
    # Test search
    test_queries = [
        "machine learning",
        "linear algebra",
        "probability",
        "neural networks"
    ]
    
    print(f"\n🔍 Testing search functionality...")
    for query in test_queries:
        results = kb.search(query, top_k=2)
        print(f"\nQuery: '{query}'")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['title']} (score: {result['relevance_score']:.3f})")
                print(f"      {result['snippet'][:100]}...")
        else:
            print("   No results found")
    
    # Test context generation
    test_question = "Explain eigenvalues and eigenvectors"
    context = kb.get_relevant_context(test_question)
    
    if context:
        print(f"\n📚 Context for '{test_question}':")
        print(f"   {context[:200]}...")
    else:
        print(f"\n📚 No relevant context found for '{test_question}'")

if __name__ == "__main__":
    main()