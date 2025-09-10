#!/usr/bin/env python3
"""
Direct Question → Model Response → Obsidian Vault
Simple and efficient approach
"""
import json
import time
from pathlib import Path
from datetime import datetime
from ollama_inference import OllamaEducationalInference

def format_raw_response_for_obsidian(response: str, question: str, model_name: str, metrics: dict) -> str:
    """Format raw response into Obsidian-ready markdown"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add Obsidian formatting to raw response
    formatted = f"""# {model_name} - Raw Response

**Question:** {question}  
**Generated:** {timestamp}  
**Length:** {metrics.get('char_count', 0):,} characters  
**Generation Time:** {metrics.get('generation_time', 0):.1f} seconds  

---

## Response

{response}

---

**Approach:** Raw (direct model output)  
**Model:** {model_name}  
"""
    
    return formatted

def ask_and_save_to_obsidian(question: str, question_id: int, domain: str, model_id: str, model_name: str):
    """Ask question to model and save both raw and ground rules responses directly to Obsidian"""
    
    print(f"\n🧪 Testing {model_name} on Q{question_id}: {question[:50]}...")
    
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
        inference = OllamaEducationalInference(model_name=model_id, use_knowledge_base=False)
        
        # 1. RAW RESPONSE
        print("  📝 Generating raw response...")
        start_time = time.time()
        
        raw_response = inference.generate_response(
            question=question,
            use_ground_rules=False,
            adaptive_format=False,  # Truly raw
            max_tokens=8192,
            temperature=0.7
        )
        
        raw_time = time.time() - start_time
        raw_metrics = {
            'char_count': len(raw_response),
            'word_count': len(raw_response.split()),
            'generation_time': raw_time
        }
        
        # Format and save raw response
        raw_formatted = format_raw_response_for_obsidian(raw_response, question, model_name, raw_metrics)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = raw_folder / f"Q{question_id}_{domain}_raw_{timestamp}.md"
        raw_file.write_text(raw_formatted, encoding='utf-8')
        
        print(f"  ✅ Raw: {raw_metrics['char_count']} chars in {raw_time:.1f}s → {raw_file}")
        
        # Small delay
        time.sleep(5)
        
        # 2. GROUND RULES RESPONSE (already Obsidian-formatted)
        print("  📜 Generating ground rules response...")
        start_time = time.time()
        
        gr_response = inference.generate_response(
            question=question,
            use_ground_rules=True,
            research_mode=True,
            max_tokens=8192,
            temperature=0.7
        )
        
        gr_time = time.time() - start_time
        gr_metrics = {
            'char_count': len(gr_response),
            'word_count': len(gr_response.split()),
            'generation_time': gr_time
        }
        
        # Ground rules response is already Obsidian-ready, just add minimal header
        gr_header = f"""# {model_name} - Ground Rules Response

**Question:** {question}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Length:** {gr_metrics['char_count']:,} characters  
**Generation Time:** {gr_time:.1f} seconds  

---

"""
        
        gr_formatted = gr_header + gr_response
        
        gr_file = gr_folder / f"Q{question_id}_{domain}_ground_rules_{timestamp}.md"
        gr_file.write_text(gr_formatted, encoding='utf-8')
        
        print(f"  ✅ Ground Rules: {gr_metrics['char_count']} chars in {gr_time:.1f}s → {gr_file}")
        
        # Show improvement
        improvement = gr_metrics['char_count'] / raw_metrics['char_count'] if raw_metrics['char_count'] > 0 else 0
        print(f"  📊 Improvement: {improvement:.1f}x longer with ground rules")
        
        return {
            'raw': {'file': str(raw_file), 'metrics': raw_metrics},
            'ground_rules': {'file': str(gr_file), 'metrics': gr_metrics},
            'improvement': improvement
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    """Direct question testing"""
    
    # Load questions
    with open('data/gemma eval.json', 'r') as f:
        questions = json.load(f)
    
    # Available models
    models = [
        ("llama3.1:70b-instruct-q8_0", "Llama 3.1 70B Instruct"),
        ("deepseek-r1:70b", "DeepSeek R1 70B"),
        ("gpt-oss:120b", "GPT-OSS 120B")
    ]
    
    print("🎯 Direct Question → Obsidian Pipeline")
    print("="*50)
    print("This will ask questions directly and save formatted responses to Obsidian")
    print()
    
    # Test with first few questions
    test_questions = questions[:3]  # Start with first 3 questions
    
    results = []
    
    for q_data in test_questions:
        question = q_data['question']
        question_id = q_data['id']
        domain = q_data['domain']
        
        print(f"\n🔄 Question {question_id}: {question}")
        print(f"📂 Domain: {domain}")
        
        for model_id, model_name in models:
            result = ask_and_save_to_obsidian(question, question_id, domain, model_id, model_name)
            if result:
                results.append({
                    'question_id': question_id,
                    'model': model_name,
                    'result': result
                })
            
            # Delay between models
            print("  ⏳ Waiting 10 seconds...")
            time.sleep(10)
        
        print(f"✅ Question {question_id} completed for all models")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"Generated responses for {len(test_questions)} questions × {len(models)} models = {len(results)} complete test sets")
    print(f"All responses saved to: /Users/tld/Documents/Obsidian LLM/Educational/Model Evaluations/")
    
    if results:
        print(f"\nAverage improvements (Ground Rules vs Raw):")
        improvements = [r['result']['improvement'] for r in results if r['result']['improvement'] > 0]
        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            print(f"  📈 {avg_improvement:.1f}x average improvement")

if __name__ == "__main__":
    main()