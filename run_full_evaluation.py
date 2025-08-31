"""
Run Full Evaluation with Ollama
Generate responses for all questions in the evaluation dataset
"""
import json
import logging
from pathlib import Path
from ollama_inference import OllamaEducationalInference
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Generate responses for the full evaluation dataset"""
    
    # Load evaluation dataset
    dataset_path = Path(config.data_dir) / "evaluation_dataset.json"
    
    if not dataset_path.exists():
        logger.error(f"Evaluation dataset not found at {dataset_path}")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    questions = [item["question"] for item in dataset]
    domains = [item["domain"] for item in dataset]
    
    logger.info(f"📊 Loaded {len(questions)} questions from evaluation dataset")
    logger.info(f"📚 Domains: {list(set(domains))}")
    
    # Initialize Ollama inference
    inference = OllamaEducationalInference()
    
    # Generate responses with slightly longer timeout
    logger.info(f"🚀 Starting full evaluation with Gemma 3 12B...")
    
    # Increase timeout for complex questions
    results = []
    for i, question in enumerate(questions):
        logger.info(f"📝 Processing question {i+1}/{len(questions)}")
        logger.info(f"📖 Domain: {domains[i]}")
        logger.info(f"❓ Question: {question[:60]}...")
        
        # Generate response with extended timeout for complex questions
        response = inference.generate_response(question)
        
        results.append({
            "question": question,
            "domain": domains[i],
            "response": response,
            "model": "gemma3:12b",
            "timestamp": None  # Will be filled by batch_generate
        })
        
        # Progress update
        if (i + 1) % 5 == 0:
            logger.info(f"✅ Completed {i+1}/{len(questions)} questions")
    
    # Save results
    output_path = Path(config.data_dir) / "gemma3_full_evaluation_responses.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Full evaluation responses saved to {output_path}")
    
    # Generate summary statistics
    successful_responses = [r for r in results if not r["response"].startswith("Error:")]
    failed_responses = [r for r in results if r["response"].startswith("Error:")]
    
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Questions: {len(questions)}")
    print(f"Successful Responses: {len(successful_responses)}")
    print(f"Failed/Timeout Responses: {len(failed_responses)}")
    print(f"Success Rate: {len(successful_responses)/len(questions)*100:.1f}%")
    
    # Domain breakdown
    print(f"\n📚 Responses by Domain:")
    for domain in set(domains):
        domain_results = [r for r in results if r["domain"] == domain]
        domain_success = [r for r in domain_results if not r["response"].startswith("Error:")]
        print(f"  {domain.replace('_', ' ').title()}: {len(domain_success)}/{len(domain_results)} successful")
    
    # Show a sample response
    if successful_responses:
        sample = successful_responses[0]
        print(f"\n📝 Sample Response (Domain: {sample['domain']}):")
        print(f"Question: {sample['question']}")
        print(f"Response (first 300 chars): {sample['response'][:300]}...")
    
    print("="*80)

if __name__ == "__main__":
    main()
