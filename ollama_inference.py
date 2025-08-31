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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaEducationalInference:
    def __init__(self, model_name: str = None, ollama_url: str = "http://localhost:11434"):
        """Initialize Ollama inference client"""
        self.model_name = model_name or config.model.model_name
        self.ollama_url = ollama_url
        self.api_url = f"{ollama_url}/api/generate"
        
        logger.info(f"Initializing Ollama Educational Inference")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Ollama URL: {ollama_url}")
        
        # Test connection
        self._test_connection()
    
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
                         educational_prompt: bool = True) -> str:
        """Generate educational response using Ollama"""
        
        # Use educational prompt template
        if educational_prompt:
            prompt = self._create_educational_prompt(question)
        else:
            prompt = question
        
        # Generation parameters
        temperature = temperature or config.model.temperature
        max_tokens = max_tokens or config.model.max_length
        
        # Prepare request
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": config.model.top_p,
                "num_predict": max_tokens,
            }
        }
        
        start_time = time.time()
        
        try:
            logger.info(f"🧠 Generating response for: {question[:50]}...")
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=180  # 3 minute timeout for complex questions
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
    
    def _create_educational_prompt(self, question: str) -> str:
        """Create educational prompt template with LaTeX formatting requirement"""
        return f"""You are an expert educational tutor. When explaining concepts, always use this structure and MUST use proper LaTeX notation for all mathematical expressions:

# [Concept Name]

## Intuitive Understanding
Start with an intuitive explanation or analogy that helps build understanding.

## Mathematical Definition
Provide the formal mathematical definition using PROPER LATEX NOTATION. All formulas must be in LaTeX format.

LATEX FORMATTING REQUIREMENTS:
- Use $...$ for inline mathematical expressions
- Use $$....$$ for displayed equations
- Use proper LaTeX commands: \\frac{{}}{{}} for fractions, \\sum for summation, \\int for integrals
- Use \\begin{{bmatrix}} for matrices, \\begin{{align}} for multi-line equations
- Use subscripts with _ and superscripts with ^
- Use \\mathbb{{}} for special number sets, \\mathcal{{}} for script letters
- Use \\partial for partial derivatives, \\nabla for gradient
- Use \\alpha, \\beta, \\gamma, \\sigma, \\lambda for Greek letters

## Step-by-step Example
Walk through a concrete example with ALL mathematical expressions in proper LaTeX format.

## Why This Matters
Explain real-world applications and importance.

## Connection to Other Concepts
Link to related mathematical concepts using LaTeX notation.

## 🌟 Explore Further
Suggest 3-5 related topics, advanced concepts, or interesting applications that build on this knowledge. Format as:
- **Topic Name**: Brief description of how it connects
- **Advanced Application**: Real-world use case or deeper concept
- **Related Field**: How this connects to other domains

Question: {question}

Answer (remember: ALL MATH MUST BE IN LATEX FORMAT):"""
    
    def batch_generate(self, questions: List[str], **kwargs) -> List[Dict]:
        """Generate responses for multiple questions"""
        results = []
        
        logger.info(f"🚀 Starting batch generation for {len(questions)} questions")
        
        for i, question in enumerate(questions):
            logger.info(f"📝 Processing question {i+1}/{len(questions)}")
            
            start_time = time.time()
            response = self.generate_response(question, **kwargs)
            generation_time = time.time() - start_time
            
            results.append({
                "question": question,
                "response": response,
                "generation_time": generation_time,
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
