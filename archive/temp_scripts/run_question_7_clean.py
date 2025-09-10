#!/usr/bin/env python3
"""
Run Question 7 evaluation for GPT-OSS 20B and Llama 3.3 70B with clean format
"""
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
from ollama_inference import OllamaEducationalInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Obsidian vault path for direct saving
OBSIDIAN_VAULT = Path("/Users/tld/Documents/Obsidian LLM")

# Models to test
TEST_MODELS = {
    "gpt-oss:20b": "GPT-OSS 20B",
    "llama3.3:70b": "Llama 3.3 70B"
}

# Question 7
QUESTION_7 = {
    "id": 7,
    "question": "How to apply the optimization method to Neural Network",
    "domain": "deep_learning"
}

def save_response_to_obsidian(result: Dict, model_id: str, approach: str) -> Path:
    """Save individual response to Obsidian vault with clean format"""
    
    # Create folder structure: Model Evaluations/[Model Name]/[Approach]/
    model_name = result['model_name'].replace(' ', '_')
    folder_path = OBSIDIAN_VAULT / "Educational" / "Model Evaluations" / model_name / approach.title()
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    question = result['question']
    safe_question = "".join(c for c in question if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_question = safe_question.replace(' ', '_')[:50]
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"Q{result['question_id']}_{date_str}_{safe_question}.md"
    
    # Create note content - clean format without template sections
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    content = f"""# {question}

*Generated: {timestamp}*  
*Model: {result['model_name']} ({result['approach']} mode)*  
*Domain: {result['domain']}*  
*Generation Time: {result['metrics']['generation_time']:.1f}s*  
*Response Length: {result['metrics']['char_count']} chars*  
*Speed: {result['metrics']['chars_per_second']:.1f} chars/sec*  

---

{result['response']}

---

## 📊 Performance Metrics
- **Generation Time:** {result['metrics']['generation_time']:.2f} seconds
- **Response Length:** {result['metrics']['char_count']:,} characters
- **Word Count:** {result['metrics']['word_count']:,} words  
- **Generation Speed:** {result['metrics']['chars_per_second']:.1f} chars/sec
- **Estimated Tokens:** {result['metrics']['token_estimate']:.0f}

## 🏷️ Metadata
- **Question ID:** {result['question_id']}
- **Domain:** {result['domain']}
- **Model:** {result['model']}
- **Approach:** {result['approach']}
- **Timestamp:** {result['timestamp']}
"""
    
    # Save file
    filepath = folder_path / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath

def test_model_with_question_7(model_id: str, use_ground_rules: bool = False) -> Dict:
    """Test a single model with question 7"""
    question = QUESTION_7["question"]
    question_id = QUESTION_7["id"]
    domain = QUESTION_7["domain"]
    approach = "ground_rules" if use_ground_rules else "raw"
    
    logger.info(f"🧪 Testing {TEST_MODELS[model_id]} ({approach}) on Q{question_id}: {question}")
    
    try:
        # Initialize inference with the specific model
        inference = OllamaEducationalInference(
            model_name=model_id,
            use_knowledge_base=False
        )
        
        start_time = time.time()
        
        # Generate response with appropriate settings
        if use_ground_rules:
            response = inference.generate_response(
                question=question,
                use_ground_rules=True,
                research_mode=True,
                adaptive_format=False,
                max_tokens=None,
                temperature=0.7
            )
        else:
            obsidian_question = f"{question}\n\nPlease format your response in Obsidian-compatible markdown with proper LaTeX notation for math ($inline$ and $$block$$)."
            response = inference.generate_response(
                question=obsidian_question,
                use_ground_rules=False,
                research_mode=False,
                adaptive_format=False,
                max_tokens=None,
                temperature=0.7
            )
        
        generation_time = time.time() - start_time
        
        # Calculate response metrics
        char_count = len(response)
        word_count = len(response.split())
        token_estimate = word_count * 1.3
        
        result = {
            "question_id": question_id,
            "question": question,
            "domain": domain,
            "model": model_id,
            "model_name": TEST_MODELS.get(model_id, model_id),
            "approach": approach,
            "response": response,
            "metrics": {
                "generation_time": generation_time,
                "char_count": char_count,
                "word_count": word_count,
                "token_estimate": token_estimate,
                "chars_per_second": char_count / generation_time if generation_time > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Generated {char_count} chars in {generation_time:.1f}s ({result['metrics']['chars_per_second']:.1f} chars/sec)")
        
        # Save to Obsidian immediately
        try:
            obsidian_path = save_response_to_obsidian(result, model_id, approach)
            logger.info(f"📝 Saved to Obsidian: {obsidian_path}")
            result['obsidian_path'] = str(obsidian_path)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save to Obsidian: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to generate response: {e}")
        return {
            "question_id": question_id,
            "question": question,
            "domain": domain,
            "model": model_id,
            "model_name": TEST_MODELS.get(model_id, model_id),
            "approach": approach,
            "response": f"ERROR: {str(e)}",
            "metrics": {"generation_time": 0, "char_count": 0, "word_count": 0, "token_estimate": 0, "chars_per_second": 0},
            "timestamp": datetime.now().isoformat(),
            "error": True
        }

def main():
    """Run Question 7 evaluation"""
    print("🧠 Question 7 Evaluation: GPT-OSS 20B vs Llama 3.3 70B")
    print("="*80)
    print(f"📄 Question: {QUESTION_7['question']}")
    print(f"🏷️  Domain: {QUESTION_7['domain']}")
    
    results = []
    
    for model_id in TEST_MODELS.keys():
        model_name = TEST_MODELS[model_id]
        
        print(f"\n🤖 Testing {model_name}")
        print("-" * 40)
        
        # Test with raw approach
        print(f"📊 {model_name} (Raw)")
        result_raw = test_model_with_question_7(model_id, use_ground_rules=False)
        results.append(result_raw)
        
        time.sleep(3)
        
        # Test with ground rules approach
        print(f"📊 {model_name} (Ground Rules)")
        result_ground_rules = test_model_with_question_7(model_id, use_ground_rules=True)
        results.append(result_ground_rules)
        
        time.sleep(5)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/question_7_evaluation_{timestamp}.json"
    
    Path("data").mkdir(exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Results saved to: {results_file}")
    
    # Print summary
    print(f"\n📊 QUESTION 7 SUMMARY")
    print("="*40)
    
    for result in results:
        if not result.get('error'):
            print(f"{result['model_name']} ({result['approach']}):")
            print(f"  • Length: {result['metrics']['char_count']} chars")
            print(f"  • Time: {result['metrics']['generation_time']:.1f}s")
            print(f"  • Speed: {result['metrics']['chars_per_second']:.1f} chars/sec")
        else:
            print(f"{result['model_name']} ({result['approach']}): ERROR")
    
    print("\n✅ Question 7 evaluation completed!")

if __name__ == "__main__":
    main()