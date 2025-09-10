#!/usr/bin/env python3
"""
Quick All Models Evaluation - Efficient version
Test all available models with a single shared inference instance
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
    'gpt-oss:120b',
    'gpt-oss:20b', 
    'llama3.3:70b',
    'llama3.1:70b-instruct-q8_0',
    'deepseek-r1:70b',
    'gemma3:27b',
    'gemma3:12b'
]

MODEL_NAMES = {
    'gpt-oss:120b': 'GPT-OSS 120B',
    'gpt-oss:20b': 'GPT-OSS 20B',
    'llama3.3:70b': 'Llama 3.3 70B',
    'llama3.1:70b-instruct-q8_0': 'Llama 3.1 70B',
    'deepseek-r1:70b': 'DeepSeek R1 70B',
    'gemma3:27b': 'Gemma 3 27B',
    'gemma3:12b': 'Gemma 3 12B'
}

# Quick evaluation question
QUESTION = {
    "id": 1,
    "question": "Explain quantum computing in 300 words",
    "domain": "quantum_computing"
}

def test_model_quick(inference, model, approach="raw"):
    """Test a single model quickly"""
    try:
        print(f"  Testing {MODEL_NAMES[model]} ({approach})")
        
        start_time = time.time()
        
        if approach == "ground_rules":
            response = inference.generate_response(
                question=QUESTION["question"],
                use_ground_rules=True,
                research_mode=True,
                adaptive_format=False,
                max_tokens=500,  # Limit for quick test
                temperature=0.7,
            )
        else:
            # Raw mode 
            response = inference.generate_response(
                question=f"{QUESTION['question']}\n\nPlease format your response in Obsidian-compatible markdown.",
                use_ground_rules=False,
                research_mode=False,
                adaptive_format=False,
                max_tokens=500,  # Limit for quick test
                temperature=0.7,
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Calculate metrics
        char_count = len(response)
        word_count = len(response.split())
        chars_per_second = char_count / generation_time if generation_time > 0 else 0
        
        result = {
            "model": model,
            "model_name": MODEL_NAMES[model],
            "approach": approach,
            "question_id": QUESTION["id"],
            "question": QUESTION["question"],
            "domain": QUESTION["domain"],
            "response": response,
            "metrics": {
                "generation_time": generation_time,
                "char_count": char_count,
                "word_count": word_count,
                "chars_per_second": chars_per_second
            },
            "timestamp": datetime.now().isoformat(),
            "error": False
        }
        
        print(f"    ✅ {char_count} chars in {generation_time:.1f}s ({chars_per_second:.1f} c/s)")
        return result
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)}")
        return {
            "model": model,
            "model_name": MODEL_NAMES[model],
            "approach": approach,
            "question_id": QUESTION["id"],
            "question": QUESTION["question"],
            "domain": QUESTION["domain"],
            "response": "",
            "metrics": {
                "generation_time": 0,
                "char_count": 0,
                "word_count": 0,
                "chars_per_second": 0
            },
            "timestamp": datetime.now().isoformat(),
            "error": True,
            "error_message": str(e)
        }

def run_quick_evaluation():
    """Run quick evaluation across all models"""
    print("🚀 Quick All Models Evaluation")
    print(f"📊 Testing {len(ALL_MODELS)} models")
    
    # Initialize inference once
    print("🔧 Initializing inference system...")
    try:
        inference = OllamaEducationalInference()
        print("✅ Inference system ready")
    except Exception as e:
        print(f"❌ Failed to initialize inference: {e}")
        return
    
    all_results = []
    
    for i, model in enumerate(ALL_MODELS, 1):
        print(f"\n🤖 [{i}/{len(ALL_MODELS)}] Testing {MODEL_NAMES[model]}")
        
        # Test raw mode
        result = test_model_quick(inference, model, "raw")
        all_results.append(result)
        
        # Test ground rules mode 
        result = test_model_quick(inference, model, "ground_rules")
        all_results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/quick_all_models_evaluation_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {filename}")
    
    # Generate quick summary
    successful_results = [r for r in all_results if not r["error"]]
    failed_results = [r for r in all_results if r["error"]]
    
    print(f"\n📊 QUICK EVALUATION SUMMARY:")
    print(f"✅ Successful: {len(successful_results)}/{len(all_results)}")
    print(f"❌ Failed: {len(failed_results)}/{len(all_results)}")
    
    if successful_results:
        print(f"\n🏆 TOP PERFORMERS:")
        
        # Sort by speed
        speed_sorted = sorted(successful_results, key=lambda x: x["metrics"]["chars_per_second"], reverse=True)
        print(f"🔥 Fastest: {speed_sorted[0]['model_name']} ({speed_sorted[0]['approach']}) - {speed_sorted[0]['metrics']['chars_per_second']:.1f} chars/sec")
        
        # Sort by length
        length_sorted = sorted(successful_results, key=lambda x: x["metrics"]["char_count"], reverse=True)
        print(f"📏 Most Comprehensive: {length_sorted[0]['model_name']} ({length_sorted[0]['approach']}) - {length_sorted[0]['metrics']['char_count']} chars")
        
        # Model comparison
        print(f"\n📋 MODEL RANKINGS (by speed):")
        for i, result in enumerate(speed_sorted[:10], 1):
            print(f"  {i}. {result['model_name']} ({result['approach']}): {result['metrics']['chars_per_second']:.1f} c/s")
    
    if failed_results:
        print(f"\n❌ FAILED MODELS:")
        for result in failed_results:
            print(f"  • {result['model_name']} ({result['approach']}): {result.get('error_message', 'Unknown error')}")
    
    return filename

if __name__ == "__main__":
    results_file = run_quick_evaluation()
    print(f"\n🎉 Quick evaluation complete!")
    print(f"📁 Results: {results_file}")