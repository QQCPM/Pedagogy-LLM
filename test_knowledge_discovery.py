"""
Test Knowledge Discovery Feature
Test the updated prompt with related topics suggestions
"""
import json
import logging
from pathlib import Path
from ollama_inference import OllamaEducationalInference
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test knowledge discovery suggestions"""
    
    test_question = "Explain the concept of eigenvalues and eigenvectors"
    
    print("🌟 TESTING KNOWLEDGE DISCOVERY FEATURE")
    print("="*60)
    print(f"Question: {test_question}")
    print("="*60)
    
    # Test with updated prompt that includes knowledge discovery
    inference = OllamaEducationalInference()
    
    print("\n📝 Testing updated prompt with knowledge discovery suggestions")
    response = inference.generate_response(test_question)
    
    # Look for knowledge discovery indicators
    discovery_indicators = [
        "🌟", "Explore", "related", "advanced", "application", 
        "build on", "connects", "further", "deeper"
    ]
    
    discovery_count = sum(1 for indicator in discovery_indicators if indicator.lower() in response.lower())
    
    print(f"\n📊 RESULTS:")
    print(f"   🌟 Knowledge discovery indicators: {discovery_count}")
    
    # Check if it has the "Explore Further" section
    has_explore_section = "explore further" in response.lower() or "🌟" in response
    print(f"   📖 Has exploration section: {'✅' if has_explore_section else '❌'}")
    
    # Save results
    results = {
        "question": test_question,
        "response": response,
        "discovery_count": discovery_count,
        "has_explore_section": has_explore_section,
        "test_type": "knowledge_discovery"
    }
    
    output_path = Path(config.data_dir) / "knowledge_discovery_test.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {output_path}")
    
    # Show the exploration section if present
    if has_explore_section:
        print(f"\n🌟 KNOWLEDGE DISCOVERY SECTION FOUND!")
        # Extract the explore section
        explore_start = response.lower().find("🌟") or response.lower().find("explore")
        if explore_start > 0:
            explore_section = response[explore_start:explore_start+500]
            print("-" * 40)
            print(explore_section + "...")
            print("-" * 40)
    
    print(f"\n🎯 VERDICT:")
    if has_explore_section and discovery_count >= 3:
        print("✅ SUCCESS! Knowledge discovery feature working perfectly")
        print("🎉 Students will get related topics suggestions after each answer")
    elif discovery_count >= 2:
        print("⚠️ PARTIAL: Some discovery content found, but could be improved")
    else:
        print("❌ FAILED: No knowledge discovery suggestions detected")

if __name__ == "__main__":
    main()
