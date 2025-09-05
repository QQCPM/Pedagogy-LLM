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
from adaptive_templates import AdaptiveTemplateEngine

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
        
        # Initialize adaptive template engine
        self.adaptive_templates = AdaptiveTemplateEngine()
        
        # Legacy question analysis patterns (kept for backward compatibility)
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
                         include_knowledge_gap_context: bool = True,
                         use_advanced_templates: bool = True,
                         use_ground_rules: bool = False,
                         research_mode: bool = False,
                         complexity_override: str = None) -> str:
        """Generate adaptive educational response using Ollama"""
        
        # Enhanced RAG for Llama, lightweight for others
        known_concepts = []
        if include_knowledge_gap_context and self.knowledge_base:
            try:
                # Enhanced context injection for Llama
                if "llama" in self.model_name.lower():
                    # More comprehensive search for Llama
                    smart_results = self.knowledge_base.smart_search(question, top_k=5, include_related_concepts=True)
                    if smart_results:
                        known_concepts = []
                        for result in smart_results:
                            if 'matched_concepts' in result and result['matched_concepts']:
                                known_concepts.extend(result['matched_concepts'][:3])  # More concepts for Llama
                        known_concepts = list(set(known_concepts))[:8]  # Higher limit for Llama
                        if known_concepts:
                            logger.info(f"🦙 Enhanced Llama context: User knows {known_concepts}")
                else:
                    # Standard lightweight search for other models
                    smart_results = self.knowledge_base.smart_search(question, top_k=2, include_related_concepts=True)
                    if smart_results:
                        known_concepts = []
                        for result in smart_results:
                            if 'matched_concepts' in result and result['matched_concepts']:
                                known_concepts.extend(result['matched_concepts'][:2])
                        known_concepts = list(set(known_concepts))[:5]  # Standard limit
                        if known_concepts:
                            logger.info(f"🎯 Knowledge gap context: User knows {known_concepts}")
            except Exception as e:
                logger.warning(f"⚠️ Knowledge gap detection failed: {e}")
        
        # Use adaptive prompt based on question analysis  
        if research_mode or use_ground_rules:
            # Use research-focused ground rules prompting
            from ground_rules_prompt import create_ground_rules_prompt
            prompt = create_ground_rules_prompt(question, known_concepts)
        elif adaptive_format:
            if use_advanced_templates:
                # Use new adaptive template system
                if complexity_override:
                    # Override complexity in the question for template analysis
                    complexity_indicators = {
                        'simple': 'brief simple',
                        'detailed': 'detailed comprehensive thorough'
                    }
                    modified_question = question
                    if complexity_override in complexity_indicators:
                        modified_question = f"{question} {complexity_indicators[complexity_override]}"
                    base_prompt = self.adaptive_templates.create_prompt(modified_question)
                else:
                    base_prompt = self.adaptive_templates.create_prompt(question)
                
                # Add knowledge context if available
                if known_concepts:
                    knowledge_context = f"\n\nBackground: User is already familiar with: {', '.join(known_concepts)}. Focus on new aspects and avoid redundant explanations of these concepts."
                    base_prompt += knowledge_context
                prompt = base_prompt
            else:
                # Use legacy adaptive prompting
                prompt = self._create_adaptive_prompt(question, known_concepts)
        else:
            prompt = question
        
        # Model-specific generation parameters
        if "llama" in self.model_name.lower():
            # Enhanced settings for Llama to generate longer, more detailed responses
            temperature = temperature or 0.6  # Slightly lower for more focused responses
            max_tokens = max_tokens or 6144   # Much higher token limit for verbose responses
            repeat_penalty = 1.03             # Lower penalty to allow detailed explanations
            top_p = 0.95                      # Higher top_p for more diverse vocabulary
            top_k = 60                        # Higher top_k for richer word choice
        else:
            # Enhanced settings for Gemma 3 and other models
            temperature = temperature or config.model.temperature
            
            # Extended context for ground rules or research mode (research-focused responses)
            if use_ground_rules or research_mode:
                max_tokens = max_tokens or 8192   # Maximum context for detailed research notes
                repeat_penalty = 1.02             # Lower penalty for comprehensive explanations
                top_p = 0.9                       # Higher diversity for research depth
                top_k = 50                        # More word choices for technical vocabulary
            else:
                max_tokens = max_tokens or 6144   # Extended standard length for richer responses
                repeat_penalty = 1.04             # Slightly lower penalty for better flow
                top_p = config.model.top_p
                top_k = 45                        # Increased word choices for better vocabulary
        
        # Prepare request
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "repeat_penalty": repeat_penalty,
                "top_k": top_k,
            }
        }
        
        start_time = time.time()
        
        # Set timeout based on model - Llama needs more time for longer responses
        timeout_duration = 600 if "llama" in self.model_name.lower() else 300  # 10 min for Llama, 5 min for others
        
        try:
            logger.info(f"🧠 Generating response for: {question[:50]}...")
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=timeout_duration
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
    
    def _get_llama_verbosity_instructions(self, question_type: str) -> str:
        """Get specific verbosity instructions for Llama based on question type"""
        
        verbosity_templates = {
            'comparison': """
For COMPARISON responses:
- Create detailed comparison tables with 5-7 dimensions
- Provide 2-3 concrete examples for each item being compared
- Include pros/cons with real-world scenarios
- Add decision-making criteria and use-case recommendations
- Discuss trade-offs in depth with quantitative examples where possible""",
            
            'summary': """
For SUMMARY responses:
- Start with a comprehensive executive summary (100-150 words)
- Break into 4-6 detailed sections with subheadings
- Include background/context section explaining why this topic matters
- Provide multiple examples and case studies
- Add implications and future directions section""",
            
            'process': """
For PROCESS/HOW-TO responses:
- Provide detailed step-by-step instructions with sub-steps
- Include troubleshooting tips and common pitfalls for each step
- Add alternative approaches and when to use them
- Explain the reasoning behind each step
- Include resource requirements and time estimates""",
            
            'concept': """
For CONCEPT explanations:
- Start with intuitive explanations using 2-3 different analogies
- Provide formal definitions with detailed breakdowns of each component
- Include historical development and key contributors
- Give 3-4 progressively complex examples
- Discuss variations, exceptions, and edge cases
- Connect to broader theoretical frameworks""",
            
            'calculation': """
For CALCULATION/MATH responses:
- Show multiple solution methods when possible
- Explain each step with detailed reasoning
- Include verification/checking methods
- Provide intuitive explanations of what each step accomplishes
- Add real-world applications and interpretations of results""",
            
            'list': """
For LIST responses:
- Organize into clear categories with detailed explanations
- Provide 2-3 examples for each item with context
- Include criteria for classification
- Add exceptions and edge cases
- Discuss relationships between different items"""
        }
        
        return verbosity_templates.get(question_type, verbosity_templates['concept'])
    
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
        
        # Model-specific prompt enhancement
        if "llama" in self.model_name.lower():
            # Enhanced verbose prompt for Llama
            verbosity_instructions = self._get_llama_verbosity_instructions(analysis['type'])
            base_prompt = f"""You are a comprehensive educational expert. Your goal is to provide extremely detailed, thorough explanations that leave no stone unturned.

RESPONSE REQUIREMENTS:
- Write AT LEAST 800-1200 words
- Include multiple detailed examples with step-by-step breakdowns
- Explain the "why" behind every concept, not just the "what"
- Add historical context, real-world applications, and connections to other fields
- Use analogies and metaphors to make complex ideas accessible
- Include potential misconceptions and how to avoid them

{verbosity_instructions}
{format_instructions}{knowledge_context}{follow_up_context}

Question: {question}

Provide an exceptionally comprehensive, detailed response that thoroughly explores all aspects of this topic:"""
        else:
            # Standard prompt for other models (including raw Gemma 3)
            obsidian_latex_note = ""
            if "gemma" in self.model_name.lower():
                obsidian_latex_note = "\n\nIMPORTANT: When writing mathematical formulas, use proper LaTeX notation with $...$ for inline math and $$...$$ for display math so they render correctly in Obsidian."
            
            base_prompt = f"""You are an expert tutor. Respond in the most effective format for the question type.

{format_instructions}{knowledge_context}{follow_up_context}{obsidian_latex_note}

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
