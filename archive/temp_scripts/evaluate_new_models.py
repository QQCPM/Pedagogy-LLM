#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Test new models (Llama 3.1 70B, DeepSeek R1 70B, GPT-OSS 120B) with questions from gemma eval.json
Compare raw vs ground rules approaches
"""
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from ollama_inference import OllamaEducationalInference
from config import config
from save_individual_responses import save_response_to_obsidian, create_folder_index_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# New models to test
TEST_MODELS = {
    "llama3.1:70b-instruct-q8_0": "Llama 3.1 70B Instruct",
    "deepseek-r1:70b": "DeepSeek R1 70B", 
    "gpt-oss:120b": "GPT-OSS 120B"
}

def load_eval_questions(file_path: str = "data/gemma eval.json") -> List[Dict]:
    """Load evaluation questions from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_model_with_question(model_id: str, question: str, question_id: int, domain: str, use_ground_rules: bool = False) -> Dict:
    """Test a single model with one question"""
    logger.info(f"🧪 Testing {model_id} on Q{question_id}: {question[:50]}...")
    
    try:
        # Initialize inference with the specific model
        inference = OllamaEducationalInference(
            model_name=model_id,
            use_knowledge_base=False  # Disable KB for consistent comparison
        )
        
        start_time = time.time()
        
        # Generate response with appropriate settings
        response = inference.generate_response(
            question=question,
            use_ground_rules=use_ground_rules,
            research_mode=use_ground_rules,  # Use research mode with ground rules
            adaptive_format=not use_ground_rules,  # Disable adaptive format for raw mode
            max_tokens=8192,  # Extended context for all models
            temperature=0.7
        )
        
        generation_time = time.time() - start_time
        
        # Calculate response metrics
        char_count = len(response)
        word_count = len(response.split())
        token_estimate = word_count * 1.3  # Rough token estimation
        
        result = {
            "question_id": question_id,
            "question": question,
            "domain": domain,
            "model": model_id,
            "model_name": TEST_MODELS.get(model_id, model_id),
            "approach": "ground_rules" if use_ground_rules else "raw",
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
        
        logger.info(f"✅ Generated {char_count} chars in {generation_time:.1f}s")
        
        # Save individual response to Obsidian immediately
        try:
            obsidian_path = save_response_to_obsidian(result)
            logger.info(f"📝 Saved to Obsidian: {obsidian_path}")
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
            "approach": "ground_rules" if use_ground_rules else "raw",
            "response": f"ERROR: {str(e)}",
            "metrics": {"generation_time": 0, "char_count": 0, "word_count": 0, "token_estimate": 0},
            "timestamp": datetime.now().isoformat(),
            "error": True
        }

def run_comprehensive_evaluation(questions: List[Dict], models_to_test: List[str] = None) -> List[Dict]:
    """Run comprehensive evaluation across models and approaches"""
    if models_to_test is None:
        models_to_test = list(TEST_MODELS.keys())
    
    results = []
    total_tests = len(questions) * len(models_to_test) * 2  # 2 approaches per model
    current_test = 0
    
    logger.info(f"🚀 Starting comprehensive evaluation: {total_tests} total tests")
    
    # Create organized folder structure in Obsidian
    try:
        create_folder_index_files("/Users/tld/Documents/Obsidian LLM")
        logger.info("📁 Created Obsidian folder structure")
    except Exception as e:
        logger.warning(f"⚠️ Failed to create folder structure: {e}")
    
    for question_data in questions:
        question = question_data["question"]
        question_id = question_data["id"]
        domain = question_data["domain"]
        
        for model_id in models_to_test:
            # Test with raw approach
            current_test += 1
            logger.info(f"📊 Progress: {current_test}/{total_tests}")
            
            result_raw = test_model_with_question(
                model_id, question, question_id, domain, use_ground_rules=False
            )
            results.append(result_raw)
            
            # Small delay between tests
            time.sleep(2)
            
            # Test with ground rules approach
            current_test += 1
            logger.info(f"📊 Progress: {current_test}/{total_tests}")
            
            result_ground_rules = test_model_with_question(
                model_id, question, question_id, domain, use_ground_rules=True
            )
            results.append(result_ground_rules)
            
            # Longer delay between models to prevent overload
            time.sleep(5)
    
    return results

def save_results(results: List[Dict], filename: str = None) -> str:
    """Save evaluation results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"new_models_evaluation_{timestamp}.json"
    
    filepath = Path("data") / filename
    
    # Ensure data directory exists
    filepath.parent.mkdir(exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Results saved to {filepath}")
    return str(filepath)

def generate_summary_stats(results: List[Dict]) -> Dict:
    """Generate summary statistics from results"""
    stats = {
        "total_tests": len(results),
        "models_tested": list(set(r["model"] for r in results)),
        "approaches": list(set(r["approach"] for r in results)),
        "domains": list(set(r["domain"] for r in results)),
        "success_rate": len([r for r in results if not r.get("error", False)]) / len(results),
        "avg_metrics_by_model": {},
        "avg_metrics_by_approach": {}
    }
    
    # Calculate averages by model
    for model in stats["models_tested"]:
        model_results = [r for r in results if r["model"] == model and not r.get("error", False)]
        if model_results:
            stats["avg_metrics_by_model"][model] = {
                "avg_generation_time": sum(r["metrics"]["generation_time"] for r in model_results) / len(model_results),
                "avg_char_count": sum(r["metrics"]["char_count"] for r in model_results) / len(model_results),
                "avg_word_count": sum(r["metrics"]["word_count"] for r in model_results) / len(model_results),
                "total_tests": len(model_results)
            }
    
    # Calculate averages by approach
    for approach in stats["approaches"]:
        approach_results = [r for r in results if r["approach"] == approach and not r.get("error", False)]
        if approach_results:
            stats["avg_metrics_by_approach"][approach] = {
                "avg_generation_time": sum(r["metrics"]["generation_time"] for r in approach_results) / len(approach_results),
                "avg_char_count": sum(r["metrics"]["char_count"] for r in approach_results) / len(approach_results),
                "avg_word_count": sum(r["metrics"]["word_count"] for r in approach_results) / len(approach_results),
                "total_tests": len(approach_results)
            }
    
    return stats

def main():
    """Main evaluation pipeline"""
    print("🧠 New Models Evaluation Pipeline")
    print("="*60)
    
    # Load evaluation questions
    try:
        questions = load_eval_questions()
        logger.info(f"📋 Loaded {len(questions)} evaluation questions")
    except FileNotFoundError:
        logger.error("❌ Could not find 'data/gemma eval.json'")
        return
    
    # Check which models are available
    available_models = []
    for model_id in TEST_MODELS.keys():
        try:
            # Quick test to see if model is available
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if model_id in model_names:
                    available_models.append(model_id)
                    logger.info(f"✅ {TEST_MODELS[model_id]} is available")
                else:
                    logger.warning(f"⚠️ {TEST_MODELS[model_id]} not found")
        except Exception as e:
            logger.error(f"❌ Failed to check model availability: {e}")
            return
    
    if not available_models:
        logger.error("❌ No test models are available")
        return
    
    print(f"\n🎯 Testing {len(available_models)} models with {len(questions)} questions")
    print(f"📊 Total tests: {len(available_models) * len(questions) * 2}")
    
    # Auto-proceed for background execution
    print("\n🚀 Starting evaluation automatically...")
    time.sleep(2)
    
    # Run evaluation
    results = run_comprehensive_evaluation(questions, available_models)
    
    # Save results
    results_file = save_results(results)
    
    # Generate and save summary statistics
    stats = generate_summary_stats(results)
    stats_file = results_file.replace('.json', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Generate Obsidian report automatically
    print("\n🔄 Generating Obsidian report...")
    try:
        import subprocess
        subprocess.run([
            "python3", "generate_obsidian_report.py", results_file, 
            "-o", f"/Users/tld/Documents/Obsidian LLM/Educational/Model Evaluations/Evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        ], check=True)
        print("✅ Obsidian report generated and saved to vault!")
    except Exception as e:
        print(f"⚠️ Failed to generate Obsidian report: {e}")
        print(f"💡 You can manually generate it with: python3 generate_obsidian_report.py {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"Total tests completed: {stats['total_tests']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Models tested: {', '.join(stats['models_tested'])}")
    print(f"Results saved to: {results_file}")
    print(f"Statistics saved to: {stats_file}")
    
    # Show performance comparison
    if "raw" in stats["avg_metrics_by_approach"] and "ground_rules" in stats["avg_metrics_by_approach"]:
        raw_stats = stats["avg_metrics_by_approach"]["raw"]
        gr_stats = stats["avg_metrics_by_approach"]["ground_rules"]
        
        print(f"\n🔍 APPROACH COMPARISON:")
        print(f"Raw approach:")
        print(f"  - Avg chars: {raw_stats['avg_char_count']:.0f}")
        print(f"  - Avg time: {raw_stats['avg_generation_time']:.1f}s")
        
        print(f"Ground rules approach:")
        print(f"  - Avg chars: {gr_stats['avg_char_count']:.0f}")
        print(f"  - Avg time: {gr_stats['avg_generation_time']:.1f}s")
        
        improvement = gr_stats['avg_char_count'] / raw_stats['avg_char_count']
        print(f"  - Length improvement: {improvement:.1f}x")
    
    print("\n✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()