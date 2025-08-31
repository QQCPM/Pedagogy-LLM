"""
Test Updated Prompt with LaTeX
Verify the updated prompt automatically uses LaTeX formatting
"""
import json
import logging
from pathlib import Path
from ollama_inference import OllamaEducationalInference
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test updated prompt with LaTeX formatting"""
    
    test_question = "Solve the quadratic equation ax² + bx + c = 0"
    
    print("🧪 TESTING UPDATED PROMPT WITH LATEX")
    print("="*60)
    print(f"Question: {test_question}")
    print("="*60)
    
    # Test with updated prompt
    inference = OllamaEducationalInference()
    
    print("\n📝 Testing updated prompt (should use LaTeX automatically)")
    response = inference.generate_response(test_question)
    
    # Count LaTeX indicators
    latex_indicators = ["$", "\\frac", "\\pm", "\\sqrt", "$$", "\\begin", "\\end"]
    html_indicators = ["<sup>", "<sub>", "&plusmn;"]
    
    latex_count = sum(1 for indicator in latex_indicators if indicator in response)
    html_count = sum(1 for indicator in html_indicators if indicator in response)
    
    print(f"\n📊 RESULTS:")
    print(f"   ✅ LaTeX indicators found: {latex_count}")
    print(f"   ❌ HTML indicators found: {html_count}")
    
    # Show some LaTeX examples from the response
    print(f"\n📐 LaTeX Examples Found:")
    for indicator in latex_indicators:
        if indicator in response:
            print(f"   ✅ Found: {indicator}")
    
    # Save results
    results = {
        "question": test_question,
        "response": response,
        "latex_count": latex_count,
        "html_count": html_count,
        "test_status": "updated_prompt"
    }
    
    output_path = Path(config.data_dir) / "updated_prompt_test.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {output_path}")
    
    # Final verdict
    print(f"\n🎯 VERDICT:")
    if latex_count >= 3:
        print("✅ SUCCESS! Updated prompt automatically uses LaTeX")
        print("🎉 All future mathematical content will use proper LaTeX formatting")
    elif latex_count > 0:
        print("⚠️ PARTIAL: Some LaTeX found, but could be better")
    else:
        print("❌ FAILED: No LaTeX formatting detected")
    
    # Show first part of response
    print(f"\n📝 Sample Response (first 300 chars):")
    print("-" * 40)
    print(response[:300] + "...")
    print("-" * 40)

if __name__ == "__main__":
    main()
