#!/usr/bin/env python3
"""Quick test of one model with one question"""
import json
from ollama_inference import OllamaEducationalInference

# Test with DeepSeek R1 70B on one question
model_id = "deepseek-r1:70b"
question = "whats the no free lunch theorem, im not too familiar with the advanced concepts of AI, can u tech me so detailed"

print(f"🧪 Testing {model_id} with question: {question[:50]}...")

try:
    # Test raw approach
    print("\n📝 Testing RAW approach...")
    inference = OllamaEducationalInference(model_name=model_id, use_knowledge_base=False)
    
    raw_response = inference.generate_response(
        question=question,
        use_ground_rules=False,
        adaptive_format=False,
        max_tokens=4096,
        temperature=0.7
    )
    
    print(f"✅ Raw response: {len(raw_response)} chars")
    print(f"Preview: {raw_response[:200]}...")
    
    # Test ground rules approach
    print("\n📜 Testing GROUND RULES approach...")
    gr_response = inference.generate_response(
        question=question,
        use_ground_rules=True,
        research_mode=True,
        max_tokens=8192,
        temperature=0.7
    )
    
    print(f"✅ Ground rules response: {len(gr_response)} chars")
    print(f"Preview: {gr_response[:200]}...")
    
    print(f"\n📊 Comparison:")
    print(f"Raw: {len(raw_response)} chars")
    print(f"Ground rules: {len(gr_response)} chars")
    print(f"Improvement: {len(gr_response)/len(raw_response):.1f}x")
    
except Exception as e:
    print(f"❌ Error: {e}")