"""
Ollama-based Educational Inference Script
Uses Ollama's local API for Gemma 3 12B inference
"""
import json
import time
import requests
import logging
from typing import List, Dict, Optional
from config import config
from smart_knowledge_base import SmartKnowledgeBase
from concept_flowchart import ConceptFlowchartGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaEducationalInference:
    def __init__(self, model_name: str = None, ollama_url: str = "http://localhost:11434", 
                 vault_path: str = None, use_knowledge_base: bool = True):
        """Initialize Ollama inference client with knowledge base integration"""
        self.model_name = model_name or config.model.model_name
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/generate"
        self.use_knowledge_base = use_knowledge_base
        
        logger.info(f"Initializing Ollama Educational Inference")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Ollama URL: {ollama_url}")
        
        # Initialize knowledge base
        self.knowledge_base = None
        if use_knowledge_base:
            try:
                self.knowledge_base = SmartKnowledgeBase(vault_path)
                # Try to load or create index
                if self.knowledge_base.index_vault():
                    stats = self.knowledge_base.get_enhanced_statistics()
                    logger.info(f"🧠 Smart knowledge base loaded: {stats.get('total_documents', 0)} documents, {stats.get('concept_graph', {}).get('total_concepts', 0)} concepts")
                else:
                    logger.warning("⚠️ Knowledge base indexing failed - proceeding without RAG")
                    self.knowledge_base = None
            except Exception as e:
                logger.warning(f"⚠️ Knowledge base initialization failed: {e}")
                self.knowledge_base = None
        
        # Test connection
        self._test_connection()
        
        # Question analysis patterns for adaptive responses
        self.question_patterns = {
            'comparison': ['compare', 'vs', 'versus', 'difference between', 'contrast'],
            'summary': ['summarize', 'overview', 'what is', 'explain briefly', 'in short'],
            'process': ['how to', 'steps', 'process', 'procedure', 'guide', 'tutorial'],
            'list': ['types of', 'kinds of', 'categories', 'list', 'examples of'],
            'concept': ['explain', 'understand', 'concept', 'theory', 'principle'],
            'calculation': ['calculate', 'solve', 'find', 'compute', 'derive'],
            'analysis': ['analyze', 'evaluate', 'assess', 'critique', 'examine']
        }
        
        self.complexity_indicators = {
            'simple': ['quick', 'simple', 'basic', 'easy', 'brief'],
            'detailed': ['detailed', 'comprehensive', 'thorough', 'in-depth', 'advanced']
        }
    
    def _test_connection(self):
        """Test connection to Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                logger.info(f"✅ Connected to Ollama. Available models: {model_names}")
                
                if self.model_name not in model_names:
                    logger.warning(f"⚠️ Model {self.model_name} not found in available models")
                else:
                    logger.info(f"✅ Model {self.model_name} is available")
            else:
                logger.error(f"❌ Failed to connect to Ollama: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
    
    def generate_response(self, question: str, 
                         max_tokens: int = None,
                         temperature: float = None,
                         adaptive_format: bool = True,
                         include_knowledge_gap_context: bool = True) -> str:
        """Generate adaptive educational response using Ollama"""
        
        # Lightweight RAG: Only check for knowledge gaps, not heavy context injection
        known_concepts = []
        if include_knowledge_gap_context and self.knowledge_base:
            try:
                # Quick check for what user already knows
                smart_results = self.knowledge_base.smart_search(question, top_k=2, include_related_concepts=True)
                if smart_results:
                    known_concepts = []
                    for result in smart_results:
                        if 'matched_concepts' in result and result['matched_concepts']:
                            known_concepts.extend(result['matched_concepts'][:2])
                    known_concepts = list(set(known_concepts))[:5]  # Limit to 5 concepts
                    if known_concepts:
                        logger.info(f"🎯 Knowledge gap context: User knows {known_concepts}")
            except Exception as e:
                logger.warning(f"⚠️ Knowledge gap detection failed: {e}")
        
        # Use adaptive prompt based on question analysis
        if adaptive_format:
            prompt = self._create_adaptive_prompt(question, known_concepts)
        else:
            prompt = question
        
        # Generation parameters - enhanced for longer detailed responses
        temperature = temperature or config.model.temperature
        max_tokens = max_tokens or 3072  # Increased for detailed explanations
        
        # Prepare request
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": config.model.top_p,
                "num_predict": max_tokens,
                "repeat_penalty": 1.05,  # Slight penalty to avoid repetition while allowing detailed explanations
                "top_k": 40,  # Add top_k for better quality
            }
        }
        
        start_time = time.time()
        
        try:
            logger.info(f"🧠 Generating response for: {question[:50]}...")
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300  # 5 minute timeout for detailed responses
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                generation_time = time.time() - start_time
                logger.info(f"✅ Generated response in {generation_time:.2f}s")
                
                return generated_text.strip()
            else:
                logger.error(f"❌ Ollama API error: {response.status_code} - {response.text}")
                return f"Error: Failed to generate response (HTTP {response.status_code})"
                
        except requests.exceptions.Timeout:
            logger.error("❌ Request timed out")
            return "Error: Request timed out"
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return f"Error: {str(e)}"
    
    def _analyze_question(self, question: str) -> Dict[str, str]:
        """Analyze question to determine appropriate response format and complexity"""
        question_lower = question.lower()
        
        # Determine question type
        question_type = 'concept'  # default
        for qtype, patterns in self.question_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                question_type = qtype
                break
        
        # Determine complexity level
        complexity = 'medium'  # default
        if any(indicator in question_lower for indicator in self.complexity_indicators['simple']):
            complexity = 'simple'
        elif any(indicator in question_lower for indicator in self.complexity_indicators['detailed']):
            complexity = 'detailed'
        
        # Determine if mathematical content likely
        math_indicators = ['formula', 'equation', 'calculate', 'derivative', 'integral', 'matrix', 'probability', 'statistics', 'algorithm']
        needs_math = any(indicator in question_lower for indicator in math_indicators)
        
        return {
            'type': question_type,
            'complexity': complexity,
            'needs_math': needs_math
        }
    
    def _get_format_instructions(self, question_type: str, complexity: str, needs_math: bool) -> str:
        """Get specific formatting instructions based on question analysis"""
        
        format_templates = {
            'comparison': {
                'simple': "Present as a clear comparison with key differences highlighted. Use bullet points or a simple table format.",
                'medium': "Create a structured comparison table with categories. Include pros/cons and use cases.",
                'detailed': "Provide comprehensive comparison table with multiple dimensions. Include examples, trade-offs, and decision criteria."
            },
            'summary': {
                'simple': "Provide key points in bullet format. Keep it concise and focused.",
                'medium': "Structure as: Overview + Key Points + Brief Examples. Use clear headings.",
                'detailed': "Comprehensive summary with: Executive Summary + Detailed Sections + Examples + Implications."
            },
            'process': {
                'simple': "Number each step clearly. Keep explanations brief and actionable.",
                'medium': "Detailed step-by-step process with brief explanations for each step.",
                'detailed': "Comprehensive guide with steps, sub-steps, examples, and troubleshooting tips."
            },
            'list': {
                'simple': "Organized bullet list with brief descriptions.",
                'medium': "Categorized list with examples and key characteristics.",
                'detailed': "Comprehensive categorization with detailed descriptions, examples, and use cases."
            },
            'concept': {
                'simple': "Brief explanation with one clear example.",
                'medium': "Structured explanation: Definition + Example + Why It Matters.",
                'detailed': "Comprehensive explanation: Intuition + Definition + Examples + Applications + Connections."
            },
            'calculation': {
                'simple': "Step-by-step solution with brief explanations.",
                'medium': "Detailed solution process with reasoning for each step.",
                'detailed': "Complete solution with multiple approaches, explanations, and verification."
            },
            'analysis': {
                'simple': "Key findings with supporting points.",
                'medium': "Structured analysis with methodology, findings, and implications.",
                'detailed': "Comprehensive analysis with multiple perspectives, evidence, and detailed conclusions."
            }
        }
        
        base_instruction = format_templates.get(question_type, format_templates['concept']).get(complexity, format_templates['concept']['medium'])
        
        if needs_math:
            math_instruction = "\n\nMATH FORMATTING: Use LaTeX notation for all mathematical expressions. Use $...$ for inline math and $$...$$ for displayed equations."
            base_instruction += math_instruction
            
        return base_instruction
    
    def _create_adaptive_prompt(self, question: str, known_concepts: List[str] = None) -> str:
        """Create adaptive prompt based on question analysis and lightweight knowledge context"""
        
        # Analyze the question
        analysis = self._analyze_question(question)
        format_instructions = self._get_format_instructions(
            analysis['type'], 
            analysis['complexity'], 
            analysis['needs_math']
        )
        
        # Create knowledge context (lightweight)
        knowledge_context = ""
        if known_concepts:
            knowledge_context = f"\nBackground: User is already familiar with: {', '.join(known_concepts)}. Focus on new aspects and avoid redundant explanations of these concepts."
        
        # Create follow-up suggestion context (lightweight)
        follow_up_context = ""
        if known_concepts:
            follow_up_context = f"\n\nAt the end, suggest 1-2 related topics to explore next that build on this knowledge while avoiding: {', '.join(known_concepts)}."
        
        # Base prompt
        base_prompt = f"""You are an expert tutor. Respond in the most effective format for the question type.

{format_instructions}{knowledge_context}{follow_up_context}

Question: {question}

Provide a clear, well-formatted response:"""
        
        return base_prompt    
    
    def _create_educational_prompt(self, question: str) -> str:
        """Legacy method - now uses adaptive prompting"""
        return self._create_adaptive_prompt(question, [])
    
    def batch_generate(self, questions: List[str], **kwargs) -> List[Dict]:
        """Generate responses for multiple questions"""
        results = []
        
        logger.info(f"🚀 Starting batch generation for {len(questions)} questions")
        
        for i, question in enumerate(questions):
            logger.info(f"📝 Processing question {i+1}/{len(questions)}")
            
            result = self.generate_response(question, **kwargs)
            
            if isinstance(result, dict):
                results.append({
                    "question": question,
                    "response": result.get('response', ''),
                    "diagram_path": result.get('diagram_path'),
                    "generation_time": result.get('generation_time', 0),
                    "timestamp": time.time(),
                    "model": self.model_name,
                    "error": result.get('error', False)
                })
            else:
                # Backward compatibility for string responses
                results.append({
                    "question": question,
                    "response": str(result),
                    "diagram_path": None,
                    "generation_time": 0,
                    "timestamp": time.time(),
                    "model": self.model_name
                })
            
            # Small delay between requests to be nice to the system
            time.sleep(0.5)
        
        logger.info(f"✅ Batch generation completed!")
        return results
    
    def save_responses(self, results: List[Dict], filename: str):
        """Save responses to JSON file"""
        filepath = f"{config.data_dir}/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Responses saved to {filepath}")
        return filepath

def main():
    """Test the Ollama inference"""
    # Initialize inference
    inference = OllamaEducationalInference()
    
    # Test questions
    test_questions = [
        "Explain eigenvalues and eigenvectors",
        "What is the difference between Bayesian and frequentist probability?",
        "How do transformers work in deep learning?",
        "What are world models in AI?"
    ]
    
    logger.info("🧪 Testing with sample questions...")
    
    # Generate responses
    results = inference.batch_generate(test_questions)
    
    # Save results
    filepath = inference.save_responses(results, "ollama_baseline_responses.json")
    
    # Print sample response
    print("\n" + "="*80)
    print("SAMPLE RESPONSE:")
    print("="*80)
    print(f"Question: {results[0]['question']}")
    print(f"Model: {results[0]['model']}")
    print(f"Generation Time: {results[0]['generation_time']:.2f}s")
    print(f"Response: {results[0]['response'][:500]}...")
    print("="*80)

if __name__ == "__main__":
    main()
