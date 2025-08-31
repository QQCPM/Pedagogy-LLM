"""
Run Missing Domain Questions
Complete the evaluation with Deep Learning and World Models questions
"""
import json
import logging
from pathlib import Path
from ollama_inference import OllamaEducationalInference
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run questions from missing domains: deep_learning and world_models"""
    
    # Load evaluation dataset
    dataset_path = Path(config.data_dir) / "evaluation_dataset.json"
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Find questions from missing domains
    deep_learning_questions = [item for item in dataset if item["domain"] == "deep_learning"][:2]
    world_models_questions = [item for item in dataset if item["domain"] == "world_models"][:2]
    
    # Combine for final evaluation
    missing_questions = deep_learning_questions + world_models_questions
    
    logger.info(f"🎯 Running {len(missing_questions)} questions from missing domains:")
    logger.info(f"   📚 Deep Learning: {len(deep_learning_questions)} questions")
    logger.info(f"   🌍 World Models: {len(world_models_questions)} questions")
    
    # Show what questions we'll run
    print("\n📝 QUESTIONS TO RUN:")
    print("="*60)
    for i, item in enumerate(missing_questions):
        print(f"{i+1}. [{item['domain'].upper()}] {item['question']}")
    print("="*60)
    
    # Initialize Ollama inference
    inference = OllamaEducationalInference()
    
    # Generate responses
    results = []
    for i, item in enumerate(missing_questions):
        question = item["question"]
        domain = item["domain"]
        
        logger.info(f"📝 Processing question {i+1}/{len(missing_questions)}")
        logger.info(f"📖 Domain: {domain}")
        logger.info(f"❓ Question: {question[:60]}...")
        
        # Generate response
        response = inference.generate_response(question)
        
        results.append({
            "question": question,
            "domain": domain,
            "response": response,
            "model": "gemma3:12b",
            "question_number": f"missing_domain_{i+1}"
        })
        
        logger.info(f"✅ Completed {i+1}/{len(missing_questions)} questions")
    
    # Save results
    output_path = Path(config.data_dir) / "missing_domains_responses.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Missing domains responses saved to {output_path}")
    
    # Show summary
    successful_responses = [r for r in results if not r["response"].startswith("Error:")]
    
    print("\n" + "="*80)
    print("📊 MISSING DOMAINS EVALUATION SUMMARY")
    print("="*80)
    print(f"Deep Learning Questions: {len(deep_learning_questions)}")
    print(f"World Models Questions: {len(world_models_questions)}")
    print(f"Total Questions: {len(missing_questions)}")
    print(f"Successful Responses: {len(successful_responses)}")
    print(f"Success Rate: {len(successful_responses)/len(missing_questions)*100:.1f}%")
    
    # Show sample questions
    print(f"\n📚 QUESTIONS COVERED:")
    for item in missing_questions:
        print(f"  • [{item['domain'].replace('_', ' ').title()}] {item['question']}")
    
    print("="*80)

if __name__ == "__main__":
    main()
