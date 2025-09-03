#!/usr/bin/env python3
"""
Test script for the new adaptive response system
Tests different question types and formats
"""
from ollama_inference import OllamaEducationalInference

def test_adaptive_responses():
    """Test various question types to verify adaptive formatting"""
    
    # Initialize inference engine
    print("🧪 Initializing Adaptive Educational Inference System...")
    inference = OllamaEducationalInference(use_knowledge_base=False)  # Disable KB for pure testing
    
    # Test questions for different formats
    test_cases = [
        {
            'question': 'Compare machine learning vs deep learning',
            'expected_type': 'comparison',
            'expected_format': 'table/comparison'
        },
        {
            'question': 'Summarize neural networks briefly',
            'expected_type': 'summary', 
            'expected_format': 'bullet points'
        },
        {
            'question': 'How to train a neural network step by step',
            'expected_type': 'process',
            'expected_format': 'numbered steps'
        },
        {
            'question': 'What are the types of machine learning algorithms?',
            'expected_type': 'list',
            'expected_format': 'categorized list'
        },
        {
            'question': 'Explain the concept of gradient descent',
            'expected_type': 'concept',
            'expected_format': 'structured explanation'
        },
        {
            'question': 'Calculate the derivative of x^2 + 3x + 1',
            'expected_type': 'calculation',
            'expected_format': 'step-by-step solution with LaTeX'
        }
    ]
    
    print(f"\n🎯 Testing {len(test_cases)} different question types...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case['question']
        expected_type = test_case['expected_type']
        expected_format = test_case['expected_format']
        
        print(f"{'='*60}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"Question: {question}")
        print(f"Expected Type: {expected_type}")
        print(f"Expected Format: {expected_format}")
        print(f"{'='*60}")
        
        # Test the analysis system first
        analysis = inference._analyze_question(question)
        format_instructions = inference._get_format_instructions(
            analysis['type'], 
            analysis['complexity'], 
            analysis['needs_math']
        )
        
        print(f"\n🔍 ANALYSIS RESULTS:")
        print(f"   Type: {analysis['type']}")
        print(f"   Complexity: {analysis['complexity']}")
        print(f"   Needs Math: {analysis['needs_math']}")
        print(f"\n📝 FORMAT INSTRUCTIONS:")
        print(f"   {format_instructions}")
        
        # Generate the adaptive prompt (but don't call the model to save time/resources)
        prompt = inference._create_adaptive_prompt(question, [])
        print(f"\n🤖 GENERATED PROMPT:")
        print(f"{prompt[:200]}...")
        
        # Check if analysis matches expectation
        type_match = "✅" if analysis['type'] == expected_type else "❌"
        print(f"\n{type_match} Type Analysis: Expected '{expected_type}', Got '{analysis['type']}'")
        
        print(f"\n{'='*60}\n")
        
        # Optional: Uncomment to actually generate responses (requires Ollama running)
        # response = inference.generate_response(question, adaptive_format=True, include_knowledge_gap_context=False)
        # print(f"📄 RESPONSE:\n{response[:300]}...\n")

if __name__ == "__main__":
    test_adaptive_responses()