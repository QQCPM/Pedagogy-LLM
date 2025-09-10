#!/usr/bin/env python3
"""
Focused Model Evaluation Script
Test GPT-OSS 20B and Llama 3.3 70B with the same 6 questions used for GPT-OSS 120B evaluation
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

# Obsidian vault path for direct saving
OBSIDIAN_VAULT = Path("/Users/tld/Documents/Obsidian LLM")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models to test
TEST_MODELS = {
    "gpt-oss:20b": "GPT-OSS 20B",
    "llama3.3:70b": "Llama 3.3 70B"
}

# First 6 questions from the evaluation set
EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "question": "Tell me history of the Earth",
        "domain": "earth_science"
    },
    {
        "id": 2,
        "question": "I wanna build a causality model for scientific discovery, what I need to prepare for the knowledge",
        "domain": "scientific_method"
    },
    {
        "id": 3,
        "question": "I wanna learn about the technology of hydrogel, its architecture, its potential research path",
        "domain": "materials_science"
    },
    {
        "id": 4,
        "question": "I wanna learn about quantum computing, are they using a lot of math, AI,...",
        "domain": "quantum_computing"
    },
    {
        "id": 5,
        "question": "Teach me step by step, so easy to understand the concept of numerical computation",
        "domain": "numerical_methods"
    },
    {
        "id": 6,
        "question": "whats the no free lunch theorem, im not too familiar with the advanced concepts of AI, can u tech me so detailed",
        "domain": "machine_learning"
    }
]

def save_response_to_obsidian(result: Dict, model_id: str, approach: str) -> Path:
    """Save individual response to Obsidian vault with organized structure"""
    
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
    
    # Create note content
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

## 📝 Personal Notes
<!-- Add your own thoughts, connections, and insights here -->

## 🔗 Related Concepts
<!-- Link to other notes in your vault -->

## ❓ Follow-up Questions
<!-- Questions this raised for future exploration -->
"""
    
    # Save file
    filepath = folder_path / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath

def test_model_with_question(model_id: str, question: str, question_id: int, domain: str, use_ground_rules: bool = False) -> Dict:
    """Test a single model with one question"""
    approach = "ground_rules" if use_ground_rules else "raw"
    logger.info(f"🧪 Testing {TEST_MODELS[model_id]} ({approach}) on Q{question_id}: {question[:50]}...")
    
    try:
        # Initialize inference with the specific model
        inference = OllamaEducationalInference(
            model_name=model_id,
            use_knowledge_base=False  # Disable KB for consistent comparison
        )
        
        start_time = time.time()
        
        # Generate response with appropriate settings
        if use_ground_rules:
            # Ground rules mode with research focus
            response = inference.generate_response(
                question=question,
                use_ground_rules=True,
                research_mode=True,
                adaptive_format=False,
                max_tokens=None,  # No token limit
                temperature=0.7
            )
        else:
            # Raw mode but with Obsidian formatting instruction
            obsidian_question = f"{question}\n\nPlease format your response in Obsidian-compatible markdown with proper LaTeX notation for math ($inline$ and $$block$$)."
            response = inference.generate_response(
                question=obsidian_question,
                use_ground_rules=False,
                research_mode=False,
                adaptive_format=False,
                max_tokens=None,  # No token limit
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

def run_focused_evaluation() -> List[Dict]:
    """Run focused evaluation on GPT-OSS 20B and Llama 3.3 70B"""
    results = []
    total_tests = len(EVALUATION_QUESTIONS) * len(TEST_MODELS) * 2  # 2 approaches per model
    current_test = 0
    
    logger.info(f"🚀 Starting focused evaluation: {total_tests} total tests")
    logger.info(f"📋 Testing models: {', '.join(TEST_MODELS.values())}")
    logger.info(f"📝 Questions: {len(EVALUATION_QUESTIONS)} questions")
    
    for question_data in EVALUATION_QUESTIONS:
        question = question_data["question"]
        question_id = question_data["id"]
        domain = question_data["domain"]
        
        logger.info(f"\n" + "="*80)
        logger.info(f"📄 QUESTION {question_id}: {question}")
        logger.info(f"🏷️  Domain: {domain}")
        logger.info("="*80)
        
        for model_id in TEST_MODELS.keys():
            model_name = TEST_MODELS[model_id]
            
            # Test with raw approach
            current_test += 1
            logger.info(f"\n📊 Progress: {current_test}/{total_tests} - {model_name} (Raw)")
            
            result_raw = test_model_with_question(
                model_id, question, question_id, domain, use_ground_rules=False
            )
            results.append(result_raw)
            
            # Small delay between tests
            time.sleep(3)
            
            # Test with ground rules approach
            current_test += 1
            logger.info(f"\n📊 Progress: {current_test}/{total_tests} - {model_name} (Ground Rules)")
            
            result_ground_rules = test_model_with_question(
                model_id, question, question_id, domain, use_ground_rules=True
            )
            results.append(result_ground_rules)
            
            # Longer delay between models to prevent overload
            time.sleep(5)
    
    return results

def save_results(results: List[Dict]) -> str:
    """Save evaluation results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gpt20b_llama33_70b_evaluation_{timestamp}.json"
    
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
        "models_tested": list(set(r["model_name"] for r in results)),
        "approaches": list(set(r["approach"] for r in results)),
        "success_rate": len([r for r in results if not r.get("error", False)]) / len(results),
        "avg_metrics_by_model": {},
        "avg_metrics_by_approach": {},
        "comparison_analysis": {}
    }
    
    # Calculate averages by model
    for model_name in stats["models_tested"]:
        model_results = [r for r in results if r["model_name"] == model_name and not r.get("error", False)]
        if model_results:
            stats["avg_metrics_by_model"][model_name] = {
                "avg_generation_time": sum(r["metrics"]["generation_time"] for r in model_results) / len(model_results),
                "avg_char_count": sum(r["metrics"]["char_count"] for r in model_results) / len(model_results),
                "avg_word_count": sum(r["metrics"]["word_count"] for r in model_results) / len(model_results),
                "avg_chars_per_second": sum(r["metrics"]["chars_per_second"] for r in model_results) / len(model_results),
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
                "avg_chars_per_second": sum(r["metrics"]["chars_per_second"] for r in approach_results) / len(approach_results),
                "total_tests": len(approach_results)
            }
    
    # Model vs Model comparison
    if len(stats["models_tested"]) >= 2:
        model_names = list(stats["models_tested"])
        model1, model2 = model_names[0], model_names[1]
        
        if model1 in stats["avg_metrics_by_model"] and model2 in stats["avg_metrics_by_model"]:
            m1_stats = stats["avg_metrics_by_model"][model1]
            m2_stats = stats["avg_metrics_by_model"][model2]
            
            stats["comparison_analysis"]["model_comparison"] = {
                f"{model1}_vs_{model2}": {
                    "char_count_ratio": m1_stats["avg_char_count"] / m2_stats["avg_char_count"],
                    "speed_ratio": m1_stats["avg_chars_per_second"] / m2_stats["avg_chars_per_second"],
                    "time_ratio": m1_stats["avg_generation_time"] / m2_stats["avg_generation_time"]
                }
            }
    
    return stats

def print_summary(stats: Dict):
    """Print comprehensive summary of results"""
    print("\n" + "="*80)
    print("📊 FOCUSED EVALUATION SUMMARY")
    print("="*80)
    print(f"Total tests completed: {stats['total_tests']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Models tested: {', '.join(stats['models_tested'])}")
    
    # Model performance comparison
    print(f"\n🏆 MODEL PERFORMANCE:")
    for model_name, metrics in stats["avg_metrics_by_model"].items():
        print(f"\n{model_name}:")
        print(f"  • Avg response length: {metrics['avg_char_count']:.0f} chars")
        print(f"  • Avg generation time: {metrics['avg_generation_time']:.1f}s")
        print(f"  • Generation speed: {metrics['avg_chars_per_second']:.1f} chars/sec")
        print(f"  • Total tests: {metrics['total_tests']}")
    
    # Approach comparison
    if "raw" in stats["avg_metrics_by_approach"] and "ground_rules" in stats["avg_metrics_by_approach"]:
        raw_stats = stats["avg_metrics_by_approach"]["raw"]
        gr_stats = stats["avg_metrics_by_approach"]["ground_rules"]
        
        print(f"\n🔍 APPROACH COMPARISON:")
        print(f"Raw approach:")
        print(f"  • Avg length: {raw_stats['avg_char_count']:.0f} chars")
        print(f"  • Avg time: {raw_stats['avg_generation_time']:.1f}s")
        print(f"  • Speed: {raw_stats['avg_chars_per_second']:.1f} chars/sec")
        
        print(f"Ground rules approach:")
        print(f"  • Avg length: {gr_stats['avg_char_count']:.0f} chars")
        print(f"  • Avg time: {gr_stats['avg_generation_time']:.1f}s")
        print(f"  • Speed: {gr_stats['avg_chars_per_second']:.1f} chars/sec")
        
        improvement = gr_stats['avg_char_count'] / raw_stats['avg_char_count']
        print(f"  • Length improvement: {improvement:.1f}x")
    
    # Model vs model analysis
    if "comparison_analysis" in stats and "model_comparison" in stats["comparison_analysis"]:
        comp_data = stats["comparison_analysis"]["model_comparison"]
        for comparison, ratios in comp_data.items():
            model1, model2 = comparison.split("_vs_")
            print(f"\n⚖️  {model1} vs {model2}:")
            print(f"  • Response length ratio: {ratios['char_count_ratio']:.2f}x")
            print(f"  • Speed ratio: {ratios['speed_ratio']:.2f}x")
            print(f"  • Time ratio: {ratios['time_ratio']:.2f}x")

def main():
    """Main evaluation pipeline"""
    print("🧠 Focused Model Evaluation: GPT-OSS 20B vs Llama 3.3 70B")
    print("="*80)
    
    # Run evaluation
    results = run_focused_evaluation()
    
    if not results:
        logger.error("❌ No results generated")
        return
    
    # Save results
    results_file = save_results(results)
    
    # Generate and save summary statistics
    stats = generate_summary_stats(results)
    stats_file = results_file.replace('.json', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print_summary(stats)
    
    print(f"\n📁 Files saved:")
    print(f"  • Results: {results_file}")
    print(f"  • Statistics: {stats_file}")
    
    print("\n✅ Focused evaluation completed successfully!")

if __name__ == "__main__":
    main()