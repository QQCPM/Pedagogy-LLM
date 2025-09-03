#!/usr/bin/env python3
"""
Concept Flowchart Generation System
Create visual diagrams to accompany educational explanations
"""
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("⚠️  graphviz not available. Install with: pip install graphviz")
    print("   Also ensure Graphviz is installed on system: brew install graphviz (macOS)")

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConceptFlowchartGenerator:
    """Generate concept flowcharts for educational topics"""
    
    def __init__(self):
        self.output_dir = Path(config.output_dir) / "flowcharts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Concept relationship patterns for different domains
        self.domain_patterns = {
            'mathematics': {
                'dependencies': ['requires', 'builds on', 'uses', 'assumes'],
                'applications': ['applies to', 'used in', 'enables'],
                'relationships': ['related to', 'connects to', 'similar to']
            },
            'machine_learning': {
                'dependencies': ['requires', 'built on', 'uses', 'trained on'],
                'applications': ['used for', 'applied to', 'enables'],
                'relationships': ['similar to', 'variant of', 'extends']
            },
            'physics': {
                'dependencies': ['governed by', 'based on', 'requires'],
                'applications': ['explains', 'predicts', 'describes'],
                'relationships': ['analogous to', 'related to', 'connected to']
            }
        }
        
        # Common concept types and their visual properties
        self.concept_styles = {
            'fundamental': {'shape': 'ellipse', 'color': 'lightblue', 'style': 'filled'},
            'theorem': {'shape': 'box', 'color': 'lightgreen', 'style': 'filled'},
            'definition': {'shape': 'diamond', 'color': 'lightcoral', 'style': 'filled'},
            'application': {'shape': 'hexagon', 'color': 'lightyellow', 'style': 'filled'},
            'example': {'shape': 'ellipse', 'color': 'lightgray', 'style': 'filled'},
            'method': {'shape': 'box', 'color': 'lightcyan', 'style': 'filled,rounded'},
            'property': {'shape': 'parallelogram', 'color': 'lavender', 'style': 'filled'}
        }
    
    def detect_topic_domain(self, question: str, response: str = "") -> str:
        """Detect the primary domain of a topic"""
        text = (question + " " + response).lower()
        
        domain_keywords = {
            'mathematics': ['matrix', 'vector', 'equation', 'theorem', 'proof', 'derivative', 'integral', 
                          'eigenvalue', 'linear', 'algebra', 'calculus', 'probability', 'statistics'],
            'machine_learning': ['neural', 'network', 'training', 'model', 'algorithm', 'learning',
                               'classification', 'regression', 'optimization', 'gradient', 'backprop'],
            'physics': ['force', 'energy', 'momentum', 'wave', 'quantum', 'relativity', 'field',
                       'particle', 'mechanics', 'thermodynamics', 'electromagnetic'],
            'computer_science': ['algorithm', 'data structure', 'complexity', 'sorting', 'graph',
                               'tree', 'hash', 'programming', 'software', 'database'],
            'engineering': ['design', 'system', 'control', 'signal', 'circuit', 'mechanical',
                          'electrical', 'optimization', 'simulation']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else 'general'
    
    def extract_concepts_from_response(self, response: str) -> List[Dict]:
        """Extract key concepts and their relationships from response text"""
        concepts = []
        
        # Extract section headers (markdown style)
        section_pattern = r'^#+\s+(.+)$'
        sections = re.findall(section_pattern, response, re.MULTILINE)
        
        for section in sections:
            # Clean section title
            clean_section = re.sub(r'[^\w\s-]', '', section.strip())
            if len(clean_section) > 3 and len(clean_section) < 50:  # Reasonable length
                concepts.append({
                    'name': clean_section,
                    'type': self.classify_concept_type(clean_section, section),
                    'context': section
                })
        
        # Extract mathematical terms (LaTeX expressions)
        latex_pattern = r'\$([^$]+)\$'
        math_terms = re.findall(latex_pattern, response)
        
        for term in math_terms:
            # Clean and extract meaningful mathematical concepts
            clean_term = re.sub(r'[\\{}]', '', term).strip()
            if len(clean_term) > 2 and len(clean_term) < 30:
                concepts.append({
                    'name': clean_term,
                    'type': 'mathematical',
                    'context': f'Mathematical expression: ${term}$'
                })
        
        # Extract key terms (capitalized words, technical terms)
        technical_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
        technical_terms = re.findall(technical_pattern, response)
        
        for term in technical_terms:
            if len(term.split()) <= 3 and len(term) > 3:  # Multi-word technical terms
                concepts.append({
                    'name': term,
                    'type': 'technical',
                    'context': f'Technical term: {term}'
                })
        
        # Remove duplicates and filter
        seen = set()
        filtered_concepts = []
        for concept in concepts:
            name_lower = concept['name'].lower()
            if name_lower not in seen and len(concept['name']) > 2:
                seen.add(name_lower)
                filtered_concepts.append(concept)
        
        return filtered_concepts[:15]  # Limit to prevent overcrowding
    
    def classify_concept_type(self, name: str, context: str) -> str:
        """Classify the type of concept for appropriate styling"""
        name_lower = name.lower()
        context_lower = context.lower()
        
        # Classification rules
        if any(word in name_lower for word in ['theorem', 'law', 'principle']):
            return 'theorem'
        elif any(word in name_lower for word in ['definition', 'what is', 'meaning']):
            return 'definition'
        elif any(word in name_lower for word in ['example', 'instance', 'case']):
            return 'example'
        elif any(word in name_lower for word in ['method', 'algorithm', 'approach', 'technique']):
            return 'method'
        elif any(word in name_lower for word in ['property', 'characteristic', 'feature']):
            return 'property'
        elif any(word in name_lower for word in ['application', 'use', 'implementation']):
            return 'application'
        elif any(word in context_lower for word in ['intuitive', 'understanding', 'foundation']):
            return 'fundamental'
        else:
            return 'concept'
    
    def extract_relationships(self, concepts: List[Dict], response: str, domain: str) -> List[Tuple]:
        """Extract relationships between concepts"""
        relationships = []
        
        if domain not in self.domain_patterns:
            domain = 'mathematics'  # Default
        
        patterns = self.domain_patterns[domain]
        concept_names = [c['name'].lower() for c in concepts]
        
        # Look for explicit relationship patterns
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i != j:
                    # Check for dependency relationships
                    for dep_word in patterns['dependencies']:
                        pattern = f"{re.escape(concept1['name'].lower())}.*?{dep_word}.*?{re.escape(concept2['name'].lower())}"
                        if re.search(pattern, response.lower()):
                            relationships.append((concept1['name'], concept2['name'], 'depends_on'))
                        
                        pattern = f"{re.escape(concept2['name'].lower())}.*?{dep_word}.*?{re.escape(concept1['name'].lower())}"
                        if re.search(pattern, response.lower()):
                            relationships.append((concept2['name'], concept1['name'], 'depends_on'))
                    
                    # Check for application relationships
                    for app_word in patterns['applications']:
                        pattern = f"{re.escape(concept1['name'].lower())}.*?{app_word}.*?{re.escape(concept2['name'].lower())}"
                        if re.search(pattern, response.lower()):
                            relationships.append((concept1['name'], concept2['name'], 'applies_to'))
        
        # Add hierarchical relationships based on section structure
        section_concepts = [c for c in concepts if c['type'] in ['fundamental', 'definition', 'theorem']]
        if len(section_concepts) > 1:
            # Create a simple hierarchy
            for i in range(len(section_concepts) - 1):
                relationships.append((section_concepts[i]['name'], section_concepts[i+1]['name'], 'leads_to'))
        
        return relationships
    
    def create_concept_flowchart(self, question: str, response: str, 
                               save_path: Optional[str] = None) -> Optional[str]:
        """Create a concept flowchart for the given question and response"""
        
        if not GRAPHVIZ_AVAILABLE:
            logger.warning("Graphviz not available - cannot generate flowchart")
            return None
        
        try:
            # Detect domain and extract concepts
            domain = self.detect_topic_domain(question, response)
            concepts = self.extract_concepts_from_response(response)
            
            if len(concepts) < 2:
                logger.info("Not enough concepts found to create flowchart")
                return None
            
            relationships = self.extract_relationships(concepts, response, domain)
            
            # Create graphviz diagram
            dot = graphviz.Digraph(comment=f'Concept Map: {question[:50]}')
            dot.attr(rankdir='TB', size='10,8', dpi='300')
            dot.attr('node', fontname='Arial', fontsize='10')
            dot.attr('edge', fontname='Arial', fontsize='8')
            
            # Add title
            title = question[:60] + ('...' if len(question) > 60 else '')
            dot.node('title', f'\n{title}\n', 
                    shape='plaintext', fontsize='14', fontname='Arial Bold')
            
            # Add concepts as nodes
            for concept in concepts:
                style = self.concept_styles.get(concept['type'], self.concept_styles['fundamental'])
                
                # Truncate long names
                display_name = concept['name'][:25] + ('...' if len(concept['name']) > 25 else '')
                
                dot.node(concept['name'], display_name, 
                        shape=style['shape'], 
                        color=style['color'], 
                        style=style['style'])
            
            # Add relationships as edges
            edge_styles = {
                'depends_on': {'color': 'red', 'style': 'solid', 'label': 'requires'},
                'applies_to': {'color': 'green', 'style': 'dashed', 'label': 'applies to'},
                'leads_to': {'color': 'blue', 'style': 'solid', 'label': 'leads to'},
                'relates_to': {'color': 'purple', 'style': 'dotted', 'label': 'relates to'}
            }
            
            for source, target, rel_type in relationships:
                style = edge_styles.get(rel_type, edge_styles['relates_to'])
                dot.edge(source, target, 
                        label=style['label'],
                        color=style['color'], 
                        style=style['style'])
            
            # Generate filename if not provided
            if not save_path:
                question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = self.output_dir / f"concept_map_{timestamp}_{question_hash}"
            
            # Render diagram
            output_path = dot.render(save_path, format='png', cleanup=True)
            
            logger.info(f"Concept flowchart generated: {output_path}")
            
            # Also save metadata
            metadata = {
                'question': question,
                'domain': domain,
                'concepts': concepts,
                'relationships': relationships,
                'generated_at': datetime.now().isoformat(),
                'image_path': output_path
            }
            
            metadata_path = str(save_path) + '_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating concept flowchart: {e}")
            return None
    
    def create_simple_diagram(self, concepts: List[str], title: str) -> Optional[str]:
        """Create a simple concept diagram from a list of concepts"""
        
        if not GRAPHVIZ_AVAILABLE or len(concepts) < 2:
            return None
        
        try:
            dot = graphviz.Digraph(comment=title)
            dot.attr(rankdir='TB', size='8,6', dpi='300')
            dot.attr('node', fontname='Arial', fontsize='10', shape='ellipse', 
                    color='lightblue', style='filled')
            
            # Add title
            dot.node('title', f'\n{title}\n', 
                    shape='plaintext', fontsize='14', fontname='Arial Bold')
            
            # Add concepts
            for concept in concepts:
                display_name = concept[:20] + ('...' if len(concept) > 20 else '')
                dot.node(concept, display_name)
            
            # Create simple linear connections
            for i in range(len(concepts) - 1):
                dot.edge(concepts[i], concepts[i + 1])
            
            # Generate filename
            title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.output_dir / f"simple_diagram_{timestamp}_{title_hash}"
            
            # Render
            output_path = dot.render(save_path, format='png', cleanup=True)
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating simple diagram: {e}")
            return None

def main():
    """Test the concept flowchart generator"""
    generator = ConceptFlowchartGenerator()
    
    # Test with a sample question and response
    test_question = "Explain eigenvalues and eigenvectors"
    test_response = """
# Eigenvalues and Eigenvectors

## Intuitive Understanding
Eigenvalues and eigenvectors represent special directions in linear transformations.

## Mathematical Definition
For a matrix $A$ and vector $v$, if $Av = \\lambda v$ where $\\lambda$ is a scalar, then $v$ is an eigenvector and $\\lambda$ is the corresponding eigenvalue.

## Applications
Eigenvalues are used in Principal Component Analysis, stability analysis, and quantum mechanics.

## Connection to Other Concepts
Eigenvalues connect to determinants, matrix diagonalization, and spectral theory.
    """
    
    print("🎨 Testing concept flowchart generation...")
    
    result = generator.create_concept_flowchart(test_question, test_response)
    
    if result:
        print(f"✅ Flowchart generated: {result}")
    else:
        print("❌ Failed to generate flowchart")
    
    # Test simple diagram
    simple_concepts = ["Linear Transformation", "Matrix", "Eigenvalue", "Eigenvector", "Diagonalization"]
    simple_result = generator.create_simple_diagram(simple_concepts, "Linear Algebra Concepts")
    
    if simple_result:
        print(f"✅ Simple diagram generated: {simple_result}")
    else:
        print("❌ Failed to generate simple diagram")

if __name__ == "__main__":
    main()