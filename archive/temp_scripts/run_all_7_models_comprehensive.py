#!/usr/bin/env python3
"""
Comprehensive All 7 Models Evaluation
Test all available models with both Raw and Ground Rules approaches
"""
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Available models (from ollama tags)
MODELS = {
    "gpt-oss:120b": {"name": "GPT-OSS 120B", "size": "116.8B params, 65GB"},
    "gpt-oss:20b": {"name": "GPT-OSS 20B", "size": "20.9B params, 13GB"},
    "llama3.3:70b": {"name": "Llama 3.3 70B", "size": "70.6B params, 42GB"},
    "llama3.1:70b-instruct-q8_0": {"name": "Llama 3.1 70B", "size": "70.6B params, 74GB"},
    "deepseek-r1:70b": {"name": "DeepSeek R1 70B", "size": "70.6B params, 42GB"},
    "gemma3:27b": {"name": "Gemma 3 27B", "size": "27.4B params, 17GB"},
    "gemma3:12b": {"name": "Gemma 3 12B", "size": "12.2B params, 8GB"}
}

# Test questions (using key representative questions)
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "Explain quantum computing in 300 words",
        "domain": "quantum_computing"
    },
    {
        "id": 2, 
        "question": "How do neural networks learn and what are the key optimization algorithms?",
        "domain": "machine_learning"
    },
    {
        "id": 3,
        "question": "Explain the concept of causality in AI and how to build causal models for scientific discovery.",
        "domain": "causal_ai"
    }
]

def test_model_response(model_id: str, question: dict, approach: str) -> dict:
    """Test single model with one question and approach"""
    logger.info(f"🧪 Testing {MODELS[model_id]['name']} - {approach} on: {question['question'][:50]}...")
    
    try:
        from ollama_inference import OllamaEducationalInference
        
        # Initialize inference for this specific model
        inference = OllamaEducationalInference(
            model_name=model_id,
            use_knowledge_base=False  # Disable KB for consistent comparison
        )
        
        start_time = time.time()
        
        # Generate response based on approach
        if approach == "ground_rules":
            response = inference.generate_response(
                question=f"{question['question']}\n\nPlease format your response in Obsidian-compatible markdown.",
                use_ground_rules=True,
                research_mode=True,
                adaptive_format=False,
                max_tokens=8192,
                temperature=0.7
            )
        else:  # raw
            response = inference.generate_response(
                question=f"{question['question']}\n\nPlease format your response in Obsidian-compatible markdown.",
                use_ground_rules=False,
                research_mode=False,
                adaptive_format=False,
                max_tokens=2000,  # Shorter for raw
                temperature=0.7
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Calculate metrics
        char_count = len(response)
        word_count = len(response.split())
        chars_per_second = char_count / generation_time if generation_time > 0 else 0
        
        result = {
            "model": model_id,
            "model_name": MODELS[model_id]["name"],
            "model_size": MODELS[model_id]["size"],
            "approach": approach,
            "question_id": question["id"],
            "question": question["question"],
            "domain": question["domain"],
            "response": response,
            "metrics": {
                "generation_time": generation_time,
                "char_count": char_count,
                "word_count": word_count,
                "token_estimate": int(word_count * 1.3),
                "chars_per_second": chars_per_second
            },
            "timestamp": datetime.now().isoformat(),
            "error": False
        }
        
        logger.info(f"✅ {MODELS[model_id]['name']} ({approach}): {char_count} chars in {generation_time:.1f}s ({chars_per_second:.1f} c/s)")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed {MODELS[model_id]['name']} ({approach}): {str(e)}")
        return {
            "model": model_id,
            "model_name": MODELS[model_id]["name"],
            "model_size": MODELS[model_id]["size"],
            "approach": approach,
            "question_id": question["id"],
            "question": question["question"],
            "domain": question["domain"],
            "response": f"ERROR: {str(e)}",
            "metrics": {
                "generation_time": 0,
                "char_count": 0,
                "word_count": 0,
                "token_estimate": 0,
                "chars_per_second": 0
            },
            "timestamp": datetime.now().isoformat(),
            "error": True,
            "error_message": str(e)
        }

def run_comprehensive_evaluation():
    """Run evaluation across all models and questions"""
    all_results = []
    total_tests = len(MODELS) * len(TEST_QUESTIONS) * 2  # 2 approaches
    current_test = 0
    
    print(f"🚀 Starting Comprehensive 7-Model Evaluation")
    print(f"📊 Models: {len(MODELS)} | Questions: {len(TEST_QUESTIONS)} | Total Tests: {total_tests}")
    print(f"⏱️  Estimated time: ~{total_tests * 30 / 60:.0f} minutes")
    
    for question in TEST_QUESTIONS:
        print(f"\n📝 Question {question['id']}: {question['question'][:50]}...")
        
        for model_id in MODELS:
            print(f"\n🤖 Testing {MODELS[model_id]['name']} ({MODELS[model_id]['size']})")
            
            # Test Raw approach
            current_test += 1
            print(f"🔄 Progress: {current_test}/{total_tests} - Raw approach")
            result_raw = test_model_response(model_id, question, "raw")
            all_results.append(result_raw)
            
            # Small delay between approaches
            time.sleep(2)
            
            # Test Ground Rules approach
            current_test += 1
            print(f"🔄 Progress: {current_test}/{total_tests} - Ground Rules approach")
            result_gr = test_model_response(model_id, question, "ground_rules")
            all_results.append(result_gr)
            
            # Longer delay between models to prevent overload
            time.sleep(5)
            
            # Quick progress report
            successful = len([r for r in all_results if not r.get('error', False)])
            failed = len([r for r in all_results if r.get('error', False)])
            print(f"   ✅ Success: {successful} | ❌ Failed: {failed}")
    
    return all_results

def save_results(results: list) -> str:
    """Save results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_7_models_comprehensive_{timestamp}.json"
    filepath = Path("data") / filename
    
    filepath.parent.mkdir(exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Results saved to {filepath}")
    return str(filepath)

def generate_quick_summary(results: list):
    """Generate quick summary of results"""
    successful = [r for r in results if not r.get('error', False)]
    failed = [r for r in results if r.get('error', False)]
    
    print(f"\n" + "="*60)
    print(f"📊 COMPREHENSIVE EVALUATION SUMMARY")
    print(f"="*60)
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        # Speed ranking
        speed_ranking = sorted(successful, key=lambda x: x['metrics']['chars_per_second'], reverse=True)
        print(f"\n🏆 TOP 5 SPEED PERFORMERS:")
        for i, result in enumerate(speed_ranking[:5], 1):
            print(f"   {i}. {result['model_name']} ({result['approach']}): {result['metrics']['chars_per_second']:.1f} c/s")
        
        # Length ranking
        length_ranking = sorted(successful, key=lambda x: x['metrics']['char_count'], reverse=True)
        print(f"\n📏 TOP 5 COMPREHENSIVE RESPONSES:")
        for i, result in enumerate(length_ranking[:5], 1):
            print(f"   {i}. {result['model_name']} ({result['approach']}): {result['metrics']['char_count']} chars")
        
        # Model performance overview
        model_performance = {}
        for result in successful:
            model_key = result['model_name']
            if model_key not in model_performance:
                model_performance[model_key] = []
            model_performance[model_key].append(result)
        
        print(f"\n🤖 MODEL PERFORMANCE OVERVIEW:")
        for model, results_list in sorted(model_performance.items()):
            avg_speed = sum(r['metrics']['chars_per_second'] for r in results_list) / len(results_list)
            avg_length = sum(r['metrics']['char_count'] for r in results_list) / len(results_list)
            success_rate = len(results_list) / (len(TEST_QUESTIONS) * 2) * 100
            print(f"   • {model}: {avg_speed:.1f} c/s avg, {avg_length:.0f} chars avg, {success_rate:.0f}% success")

def main():
    """Main evaluation pipeline"""
    print("🧠 Comprehensive 7-Model Evaluation Pipeline")
    print("="*60)
    
    # Check Ollama connection
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama server not responding")
            return
        print("✅ Ollama server connected")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return
    
    # Run comprehensive evaluation
    start_time = time.time()
    results = run_comprehensive_evaluation()
    total_time = time.time() - start_time
    
    # Save results
    results_file = save_results(results)
    
    # Generate quick summary
    generate_quick_summary(results)
    
    print(f"\n⏱️  Total evaluation time: {total_time/60:.1f} minutes")
    print(f"📁 Results saved to: {results_file}")
    print(f"\n🔄 Next: Run 'python3 generate_final_report.py' for comprehensive analysis")

if __name__ == "__main__":
    main()