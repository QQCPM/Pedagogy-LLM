#!/usr/bin/env python3
"""
Generate actual model responses for the completed questions and save to Obsidian
"""
import time
from ollama_inference import OllamaEducationalInference
from save_individual_responses import save_response_to_obsidian
from datetime import datetime

def test_single_response(model_id: str, model_name: str, question: str, question_id: int, domain: str, approach: str):
    """Generate single response and save to Obsidian"""
    print(f"\n🧪 Testing {model_name} ({approach}) on Q{question_id}")
    
    try:
        inference = OllamaEducationalInference(
            model_name=model_id,
            use_knowledge_base=False
        )
        
        start_time = time.time()
        
        response = inference.generate_response(
            question=question,
            use_ground_rules=(approach == "ground_rules"),
            research_mode=(approach == "ground_rules"),
            adaptive_format=(approach != "ground_rules"),
            max_tokens=8192,
            temperature=0.7
        )
        
        generation_time = time.time() - start_time
        char_count = len(response)
        word_count = len(response.split())
        
        result = {
            "question_id": question_id,
            "question": question,
            "domain": domain,
            "model": model_id,
            "model_name": model_name,
            "approach": approach,
            "response": response,
            "metrics": {
                "generation_time": generation_time,
                "char_count": char_count,
                "word_count": word_count,
                "token_estimate": word_count * 1.3,
                "chars_per_second": char_count / generation_time if generation_time > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to Obsidian
        filepath = save_response_to_obsidian(result)
        
        print(f"✅ Generated {char_count} chars in {generation_time:.1f}s")
        print(f"📝 Saved to: {filepath}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Generate actual responses for Question 1"""
    
    # Question 1 data
    question = "Tell me history of the Earth"
    question_id = 1
    domain = "earth_science"
    
    # Models to test
    models = [
        ("llama3.1:70b-instruct-q8_0", "Llama 3.1 70B Instruct"),
        ("deepseek-r1:70b", "DeepSeek R1 70B"),
        ("gpt-oss:120b", "GPT-OSS 120B")
    ]
    
    approaches = ["raw", "ground_rules"]
    
    print("🔄 Generating ACTUAL model responses for Question 1...")
    print(f"Question: {question}")
    print(f"This will take ~15-20 minutes for all 6 responses")
    
    results = []
    total_tests = len(models) * len(approaches)
    current_test = 0
    
    for model_id, model_name in models:
        for approach in approaches:
            current_test += 1
            print(f"\n📊 Progress: {current_test}/{total_tests}")
            
            result = test_single_response(
                model_id, model_name, question, question_id, domain, approach
            )
            
            if result:
                results.append(result)
                
            # Delay between tests
            if current_test < total_tests:
                print("⏳ Waiting 10 seconds before next test...")
                time.sleep(10)
    
    print(f"\n✅ Completed! Generated {len(results)} actual responses")
    print("📝 All responses saved to organized Obsidian folders with REAL content")
    
    # Show summary
    if results:
        print(f"\n📊 Summary:")
        for result in results:
            print(f"  {result['model_name']} ({result['approach']}): {result['metrics']['char_count']} chars")

if __name__ == "__main__":
    main()