"""
Gemma 3 12B Inference Script with Memory Optimization
Efficient inference for educational question answering
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import psutil
import time
from typing import List, Dict, Optional
from config import config
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GemmaEducationalInference:
    def __init__(self, model_name: str = None, load_in_4bit: bool = True):
        """Initialize Gemma model with memory optimization"""
        self.model_name = model_name or config.model.model_name
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing Gemma Educational Inference on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer with memory optimization"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=config.model.cache_dir,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization for memory efficiency
            if self.load_in_4bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    cache_dir=config.model.cache_dir,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=config.model.cache_dir,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
            
            logger.info(f"Model loaded successfully: {self.model_name}")
            self._log_memory_usage()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _log_memory_usage(self):
        """Log current memory usage"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_allocated = torch.cuda.memory_allocated(0) / 1e9
            gpu_cached = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"GPU Memory - Total: {gpu_memory:.1f}GB, Allocated: {gpu_allocated:.1f}GB, Cached: {gpu_cached:.1f}GB")
        
        ram_usage = psutil.virtual_memory().percent
        logger.info(f"RAM Usage: {ram_usage:.1f}%")
    
    def generate_response(self, question: str, 
                         max_length: int = None,
                         temperature: float = None,
                         educational_prompt: bool = True) -> str:
        """Generate educational response to a question"""
        
        # Use educational prompt template
        if educational_prompt:
            prompt = self._create_educational_prompt(question)
        else:
            prompt = question
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.model.max_length // 2  # Leave room for generation
        ).to(self.device)
        
        # Generation parameters
        max_length = max_length or config.model.max_length
        temperature = temperature or config.model.temperature
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=config.model.top_p,
                do_sample=config.model.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        generation_time = time.time() - start_time
        logger.info(f"Generated response in {generation_time:.2f}s")
        
        return response
    
    def _create_educational_prompt(self, question: str) -> str:
        """Create educational prompt template"""
        return f"""You are an expert educational tutor. When explaining concepts, always use this structure:

# [Concept Name]

## Intuitive Understanding
Start with an intuitive explanation or analogy that helps build understanding.

## Mathematical Definition
Provide the formal mathematical definition with proper notation.

## Step-by-step Example
Walk through a concrete example with clear steps.

## Why This Matters
Explain real-world applications and importance.

## Connection to Other Concepts
Link to related mathematical concepts and broader context.

Question: {question}

Answer:"""
    
    def batch_generate(self, questions: List[str], **kwargs) -> List[Dict]:
        """Generate responses for multiple questions"""
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            start_time = time.time()
            response = self.generate_response(question, **kwargs)
            generation_time = time.time() - start_time
            
            results.append({
                "question": question,
                "response": response,
                "generation_time": generation_time,
                "timestamp": time.time()
            })
            
            # Log memory after each generation
            if i % 5 == 0:
                self._log_memory_usage()
        
        return results
    
    def save_responses(self, results: List[Dict], filename: str):
        """Save responses to JSON file"""
        filepath = f"{config.data_dir}/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Responses saved to {filepath}")

def main():
    """Test the inference script"""
    # Initialize inference
    inference = GemmaEducationalInference()
    
    # Test questions
    test_questions = [
        "Explain eigenvalues and eigenvectors",
        "What is the difference between Bayesian and frequentist probability?",
        "How do transformers work in deep learning?",
        "What are world models in AI?"
    ]
    
    # Generate responses
    logger.info("Generating baseline responses...")
    results = inference.batch_generate(test_questions)
    
    # Save results
    inference.save_responses(results, "baseline_responses.json")
    
    # Print sample response
    print("\n" + "="*80)
    print("SAMPLE RESPONSE:")
    print("="*80)
    print(f"Question: {results[0]['question']}")
    print(f"Response: {results[0]['response'][:500]}...")
    print("="*80)

if __name__ == "__main__":
    main()
