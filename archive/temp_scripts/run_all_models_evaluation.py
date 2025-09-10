#!/usr/bin/env python3
"""
Comprehensive All Models Evaluation
Test all 7 available models: GPT-OSS 120B, GPT-OSS 20B, Llama 3.3 70B, Llama 3.1 70B, DeepSeek R1 70B, Gemma 3 27B, Gemma 3 12B
"""
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ollama_inference import OllamaEducationalInference

# All available models from ollama list
ALL_MODELS = [
    {
        'model': 'gpt-oss:120b',
        'name': 'GPT-OSS 120B',
        'size': '65GB'
    },
    {
        'model': 'gpt-oss:20b', 
        'name': 'GPT-OSS 20B',
        'size': '13GB'
    },
    {
        'model': 'llama3.3:70b',
        'name': 'Llama 3.3 70B',
        'size': '42GB'
    },
    {
        'model': 'llama3.1:70b-instruct-q8_0',
        'name': 'Llama 3.1 70B',
        'size': '74GB'
    },
    {
        'model': 'deepseek-r1:70b',
        'name': 'DeepSeek R1 70B', 
        'size': '42GB'
    },
    {
        'model': 'gemma3:27b',
        'name': 'Gemma 3 27B',
        'size': '17GB'
    },
    {
        'model': 'gemma3:12b',
        'name': 'Gemma 3 12B',
        'size': '8GB'
    }
]

# First 3 questions for efficient evaluation
EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "question": "Tell me history of the Earth",
        "domain": "earth_science"
    },
    {
        "id": 2, 
        "question": "Explain the concept of Machine Learning",
        "domain": "machine_learning"
    },
    {
        "id": 3,
        "question": "What is quantum computing and how does it work?",
        "domain": "quantum_computing"
    }
]

def test_model_response(model_config, question, approach="raw", max_retries=2):
    """Test a single model response with retries"""
    for attempt in range(max_retries + 1):
        try:
            print(f"  Attempt {attempt + 1}: Testing {model_config['name']} ({approach}) on Q{question['id']}")
            
            inference = OllamaEducationalInference()
            
            start_time = time.time()
            
            if approach == "ground_rules":
                response = inference.generate_response(
                    question=question["question"],
                    use_ground_rules=True,
                    research_mode=True,
                    adaptive_format=False,
                    max_tokens=None,
                    temperature=0.7,
                    model=model_config["model"]
                )
            else:
                # Raw mode with Obsidian formatting
                obsidian_question = f"{question['question']}\n\nPlease format your response in Obsidian-compatible markdown with proper LaTeX notation for math ($inline$ and $$block$$)."
                response = inference.generate_response(
                    question=obsidian_question,
                    use_ground_rules=False,
                    research_mode=False,
                    adaptive_format=False,
                    max_tokens=None,
                    temperature=0.7,
                    model=model_config["model"]
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Calculate metrics
            char_count = len(response)
            word_count = len(response.split())
            token_estimate = word_count * 1.3
            chars_per_second = char_count / generation_time if generation_time > 0 else 0
            
            result = {
                "question_id": question["id"],
                "question": question["question"],
                "domain": question["domain"],
                "model": model_config["model"],
                "model_name": model_config["name"],
                "model_size": model_config["size"],
                "approach": approach,
                "response": response,
                "metrics": {
                    "generation_time": generation_time,
                    "char_count": char_count,
                    "word_count": word_count,
                    "token_estimate": token_estimate,
                    "chars_per_second": chars_per_second
                },
                "timestamp": datetime.now().isoformat(),
                "error": False
            }
            
            print(f"    ✅ Success: {char_count} chars in {generation_time:.1f}s ({chars_per_second:.1f} c/s)")
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"    ❌ Attempt {attempt + 1} failed: {error_msg}")
            
            if attempt == max_retries:
                return {
                    "question_id": question["id"],
                    "question": question["question"],
                    "domain": question["domain"],
                    "model": model_config["model"],
                    "model_name": model_config["name"],
                    "model_size": model_config["size"],
                    "approach": approach,
                    "response": "",
                    "metrics": {
                        "generation_time": 0,
                        "char_count": 0,
                        "word_count": 0,
                        "token_estimate": 0,
                        "chars_per_second": 0
                    },
                    "timestamp": datetime.now().isoformat(),
                    "error": True,
                    "error_message": error_msg
                }
            
            time.sleep(5)  # Wait before retry

def save_response_to_obsidian(result):
    """Save individual response to Obsidian vault"""
    if result["error"] or not result["response"]:
        return None
    
    obsidian_base = Path("/Users/tld/Documents/Obsidian LLM/Educational/Model Evaluations")
    
    # Create folder structure: Model/Approach/
    model_folder = obsidian_base / result["model_name"].replace(" ", "_")
    approach_folder = model_folder / result["approach"].title()
    approach_folder.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    clean_question = result["question"][:50].replace(" ", "_").replace("?", "").replace("/", "_")
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"Q{result['question_id']}_{date_str}_{clean_question}.md"
    filepath = approach_folder / filename
    
    # Clean content (remove template sections)
    content = result["response"]
    template_sections = [
        "## 📝 Personal Notes",
        "<!-- Add your own thoughts, connections, and insights here -->",
        "## 🔗 Related Concepts", 
        "<!-- Link to other notes in your vault -->",
        "## ❓ Follow-up Questions",
        "<!-- Questions this raised for future exploration -->"
    ]
    
    for section in template_sections:
        content = content.replace(section, "").strip()
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(filepath)

def run_comprehensive_evaluation():
    """Run comprehensive evaluation across all models"""
    print("🚀 Starting Comprehensive All Models Evaluation")
    print(f"📊 Testing {len(ALL_MODELS)} models with {len(EVALUATION_QUESTIONS)} questions")
    print(f"🔄 Both Raw and Ground Rules approaches")
    print(f"📝 Total evaluations: {len(ALL_MODELS) * len(EVALUATION_QUESTIONS) * 2}")
    
    all_results = []
    total_evaluations = len(ALL_MODELS) * len(EVALUATION_QUESTIONS) * 2
    current_eval = 0
    
    for model_config in ALL_MODELS:
        print(f"\n🤖 Testing {model_config['name']} ({model_config['size']})")
        
        for question in EVALUATION_QUESTIONS:
            for approach in ["raw", "ground_rules"]:
                current_eval += 1
                print(f"\n📋 Evaluation {current_eval}/{total_evaluations}")
                
                # Test the model
                result = test_model_response(model_config, question, approach)
                all_results.append(result)
                
                # Save to Obsidian if successful
                if not result["error"]:
                    obsidian_path = save_response_to_obsidian(result)
                    if obsidian_path:
                        result["obsidian_path"] = obsidian_path
                
                # Progress update
                success_rate = len([r for r in all_results if not r["error"]]) / len(all_results) * 100
                print(f"📈 Progress: {current_eval}/{total_evaluations} ({current_eval/total_evaluations*100:.1f}%) | Success Rate: {success_rate:.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/all_models_evaluation_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {filename}")
    
    # Generate summary
    successful_results = [r for r in all_results if not r["error"]]
    failed_results = [r for r in all_results if r["error"]]
    
    print(f"\n📊 EVALUATION SUMMARY:")
    print(f"✅ Successful: {len(successful_results)}/{total_evaluations}")
    print(f"❌ Failed: {len(failed_results)}/{total_evaluations}")
    print(f"📈 Success Rate: {len(successful_results)/total_evaluations*100:.1f}%")
    
    if successful_results:
        avg_speed = sum(r["metrics"]["chars_per_second"] for r in successful_results) / len(successful_results)
        avg_length = sum(r["metrics"]["char_count"] for r in successful_results) / len(successful_results)
        total_time = sum(r["metrics"]["generation_time"] for r in successful_results)
        
        print(f"⚡ Average Speed: {avg_speed:.1f} chars/sec")
        print(f"📏 Average Length: {avg_length:.0f} characters")
        print(f"⏱️  Total Generation Time: {total_time:.1f} seconds")
    
    if failed_results:
        print(f"\n❌ FAILED EVALUATIONS:")
        for result in failed_results:
            print(f"  • {result['model_name']} ({result['approach']}) Q{result['question_id']}: {result.get('error_message', 'Unknown error')}")
    
    return filename

if __name__ == "__main__":
    results_file = run_comprehensive_evaluation()
    print(f"\n🎉 Comprehensive evaluation complete!")
    print(f"📁 Results: {results_file}")
    print(f"📝 Individual responses saved to Obsidian vault")