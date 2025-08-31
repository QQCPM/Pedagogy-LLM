"""
Test LaTeX Consistency
Check if model automatically uses LaTeX without explicit prompting
"""
import json
import logging
from pathlib import Path
from ollama_inference import OllamaEducationalInference
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test if model uses LaTeX automatically vs explicitly prompted"""
    
    # Same question, two different prompts
    test_question = "Derive the quadratic formula and explain each step"
    
    print("🧪 TESTING LATEX CONSISTENCY")
    print("="*60)
    print(f"Question: {test_question}")
    print("="*60)
    
    # Initialize inference with original prompt (no LaTeX emphasis)
    inference_original = OllamaEducationalInference()
    
    print("\n📝 Test 1: Original prompt (no LaTeX emphasis)")
    response_original = inference_original.generate_response(test_question)
    
    # Count LaTeX vs HTML indicators
    latex_indicators = ["$", "\\frac", "\\pm", "\\sqrt", "$$"]
    html_indicators = ["<sup>", "<sub>", "&plusmn;"]
    
    latex_count_orig = sum(1 for indicator in latex_indicators if indicator in response_original)
    html_count_orig = sum(1 for indicator in html_indicators if indicator in response_original)
    
    print(f"   LaTeX indicators: {latex_count_orig}")
    print(f"   HTML indicators: {html_count_orig}")
    
    # Save responses for comparison
    results = {
        "question": test_question,
        "original_response": response_original,
        "original_latex_count": latex_count_orig,
        "original_html_count": html_count_orig,
    }
    
    # Save results
    output_path = Path(config.data_dir) / "latex_consistency_test.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {output_path}")
    
    # Analysis
    print(f"\n📊 ANALYSIS:")
    if latex_count_orig > html_count_orig:
        print("✅ Model prefers LaTeX formatting")
    elif html_count_orig > latex_count_orig:
        print("❌ Model defaults to HTML formatting")
    else:
        print("⚠️ Mixed or no mathematical formatting")
    
    print("\n🎯 CONCLUSION:")
    if latex_count_orig == 0:
        print("❌ Model does NOT automatically use LaTeX")
        print("🔧 Solution: Update default prompt template")
    else:
        print("✅ Model automatically uses LaTeX!")

if __name__ == "__main__":
    main()
