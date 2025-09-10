#!/usr/bin/env python3
"""
Run next 3 questions (Q4-Q6) with special focus on DeepSeek quality
"""
import json
import time
from pathlib import Path
from datetime import datetime
from ollama_inference import OllamaEducationalInference

def format_raw_response_for_obsidian(response: str, question: str, model_name: str, metrics: dict, question_id: int, domain: str) -> str:
    """Format raw response into Obsidian-ready markdown"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add Obsidian formatting to raw response
    formatted = f"""# {model_name} - Raw Response

**Question {question_id}:** {question}  
**Domain:** {domain}  
**Generated:** {timestamp}  
**Length:** {metrics.get('char_count', 0):,} characters  
**Generation Time:** {metrics.get('generation_time', 0):.1f} seconds  
**Speed:** {metrics.get('chars_per_second', 0):.1f} chars/second  

---

## Response

{response}

---

**Approach:** Raw (direct model output)  
**Model:** {model_name}  
**Tags:** #{domain} #raw #{"_".join(model_name.lower().split())}
"""
    
    return formatted

def ask_and_save_to_obsidian(question: str, question_id: int, domain: str, model_id: str, model_name: str):
    """Ask question to model and save both raw and ground rules responses directly to Obsidian"""
    
    print(f"\n🧪 Testing {model_name} on Q{question_id}")
    print(f"📋 Question: {question}")
    print(f"📂 Domain: {domain}")
    
    # Setup paths
    obsidian_vault = Path("/Users/tld/Documents/Obsidian LLM")
    educational_folder = obsidian_vault / "Educational" / "Model Evaluations"
    
    model_clean = model_name.replace(" ", "_").replace(".", "_")
    raw_folder = educational_folder / "Raw Responses" / model_clean
    gr_folder = educational_folder / "Ground_Rules Responses" / model_clean
    
    # Create folders
    raw_folder.mkdir(parents=True, exist_ok=True)
    gr_folder.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"  🔧 Initializing {model_name}...")
        inference = OllamaEducationalInference(model_name=model_id, use_knowledge_base=False)
        
        # Test connection first
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            raise Exception("Ollama server not responding")
        
        models_available = [m["name"] for m in response.json().get("models", [])]
        if model_id not in models_available:
            print(f"  ⚠️ Model {model_id} not found, skipping...")
            return None
        
        # 1. RAW RESPONSE
        print("  📝 Generating raw response...")
        start_time = time.time()
        
        raw_response = inference.generate_response(
            question=question,
            use_ground_rules=False,
            adaptive_format=False,  # Truly raw
            max_tokens=None,  # No limit - let models generate freely
            temperature=0.7
        )
        
        raw_time = time.time() - start_time
        raw_metrics = {
            'char_count': len(raw_response),
            'word_count': len(raw_response.split()),
            'generation_time': raw_time,
            'chars_per_second': len(raw_response) / raw_time if raw_time > 0 else 0
        }
        
        # Format and save raw response
        raw_formatted = format_raw_response_for_obsidian(raw_response, question, model_name, raw_metrics, question_id, domain)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = raw_folder / f"Q{question_id}_{domain}_raw_{timestamp}.md"
        raw_file.write_text(raw_formatted, encoding='utf-8')
        
        print(f"  ✅ Raw: {raw_metrics['char_count']} chars in {raw_time:.1f}s → {raw_file.name}")
        
        # Special check for DeepSeek to ensure real content
        if "deepseek" in model_id.lower():
            if raw_metrics['char_count'] < 1000:
                print(f"  ⚠️ DeepSeek raw response seems short ({raw_metrics['char_count']} chars)")
            else:
                print(f"  ✅ DeepSeek raw response looks good ({raw_metrics['char_count']} chars)")
        
        # Small delay
        time.sleep(5)
        
        # 2. GROUND RULES RESPONSE (already Obsidian-formatted)
        print("  📜 Generating ground rules response...")
        start_time = time.time()
        
        gr_response = inference.generate_response(
            question=question,
            use_ground_rules=True,
            research_mode=True,
            max_tokens=None,  # No limit - let models generate freely
            temperature=0.7
        )
        
        gr_time = time.time() - start_time
        gr_metrics = {
            'char_count': len(gr_response),
            'word_count': len(gr_response.split()),
            'generation_time': gr_time,
            'chars_per_second': len(gr_response) / gr_time if gr_time > 0 else 0
        }
        
        # Ground rules response is already Obsidian-ready, just add minimal header
        gr_header = f"""# {model_name} - Ground Rules Response

**Question {question_id}:** {question}  
**Domain:** {domain}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Length:** {gr_metrics['char_count']:,} characters  
**Generation Time:** {gr_time:.1f} seconds  
**Speed:** {gr_metrics['chars_per_second']:.1f} chars/second  

---

"""
        
        gr_formatted = gr_header + gr_response
        
        gr_file = gr_folder / f"Q{question_id}_{domain}_ground_rules_{timestamp}.md"
        gr_file.write_text(gr_formatted, encoding='utf-8')
        
        print(f"  ✅ Ground Rules: {gr_metrics['char_count']} chars in {gr_time:.1f}s → {gr_file.name}")
        
        # Special check for DeepSeek ground rules
        if "deepseek" in model_id.lower():
            if gr_metrics['char_count'] < 2000:
                print(f"  ⚠️ DeepSeek ground rules response seems short ({gr_metrics['char_count']} chars)")
            else:
                print(f"  ✅ DeepSeek ground rules response looks comprehensive ({gr_metrics['char_count']} chars)")
        
        # Show improvement
        improvement = gr_metrics['char_count'] / raw_metrics['char_count'] if raw_metrics['char_count'] > 0 else 0
        print(f"  📊 Improvement: {improvement:.1f}x longer with ground rules")
        
        # Show speed comparison
        speed_diff = gr_metrics['chars_per_second'] / raw_metrics['chars_per_second'] if raw_metrics['chars_per_second'] > 0 else 0
        print(f"  ⚡ Speed: {speed_diff:.1f}x chars/sec ratio (GR vs Raw)")
        
        return {
            'raw': {'file': str(raw_file), 'metrics': raw_metrics},
            'ground_rules': {'file': str(gr_file), 'metrics': gr_metrics},
            'improvement': improvement
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    """Run questions 4-6"""
    
    # Load questions
    with open('data/gemma eval.json', 'r') as f:
        questions = json.load(f)
    
    # Available models - Fair comparison across different sizes
    models = [
        ("llama3.1:70b-instruct-q8_0", "Llama 3.1 70B Instruct"),
        ("deepseek-r1:70b", "DeepSeek R1 70B"),
        ("gpt-oss:120b", "GPT-OSS 120B"),
        ("gemma3:27b", "Gemma 3 27B"),
        ("gemma3:12b", "Gemma 3 12B")
    ]
    
    print("🎯 Next 3 Questions → Obsidian Pipeline")
    print("="*60)
    print("Running Questions 4-6 with special attention to DeepSeek quality")
    print()
    
    # Get questions 4-6 (indices 3-5)
    test_questions = questions[3:6]  # Q4, Q5, Q6
    
    results = []
    total_tests = len(test_questions) * len(models)
    current_test = 0
    
    for q_data in test_questions:
        question = q_data['question']
        question_id = q_data['id']
        domain = q_data['domain']
        
        print(f"\n{'='*60}")
        print(f"🔄 Question {question_id}: {question}")
        print(f"📂 Domain: {domain}")
        print(f"{'='*60}")
        
        for model_id, model_name in models:
            current_test += 1
            print(f"\n📊 Progress: {current_test}/{total_tests}")
            
            result = ask_and_save_to_obsidian(question, question_id, domain, model_id, model_name)
            if result:
                results.append({
                    'question_id': question_id,
                    'model': model_name,
                    'result': result
                })
            
            # Delay between models
            if current_test < total_tests:
                print("  ⏳ Waiting 15 seconds before next model...")
                time.sleep(15)
        
        print(f"\n✅ Question {question_id} completed for all models")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"📊 FINAL SUMMARY")
    print(f"="*60)
    print(f"Generated responses for {len(test_questions)} questions × {len(models)} models = {len(results)} complete test sets")
    print(f"All responses saved to: /Users/tld/Documents/Obsidian LLM/Educational/Model Evaluations/")
    
    if results:
        print(f"\n🎯 Model Performance Summary:")
        
        # Group by model
        model_stats = {}
        for r in results:
            model = r['model']
            if model not in model_stats:
                model_stats[model] = []
            model_stats[model].append(r['result']['improvement'])
        
        for model, improvements in model_stats.items():
            avg_improvement = sum(improvements) / len(improvements)
            print(f"  📈 {model}: {avg_improvement:.1f}x average improvement")
        
        print(f"\n🔍 DeepSeek Quality Check:")
        deepseek_results = [r for r in results if "DeepSeek" in r['model']]
        if deepseek_results:
            for r in deepseek_results:
                gr_chars = r['result']['ground_rules']['metrics']['char_count']
                raw_chars = r['result']['raw']['metrics']['char_count']
                print(f"  Q{r['question_id']}: Raw={raw_chars} chars, GR={gr_chars} chars, Ratio={gr_chars/raw_chars:.1f}x")

if __name__ == "__main__":
    main()