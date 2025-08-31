"""
Format LaTeX Test Responses
"""
import json
from pathlib import Path
from config import config

def format_latex_responses():
    data_dir = Path(config.data_dir)
    
    # Find the latest LaTeX test file
    latex_files = list(data_dir.glob("latex_test_responses_*.json"))
    if not latex_files:
        print("❌ No LaTeX test files found")
        return
    
    latest_file = sorted(latex_files)[-1]
    print(f"📄 Processing: {latest_file.name}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create readable text with LaTeX comparison
    readable_path = data_dir / "readable_latex_test.txt"
    with open(readable_path, 'w', encoding='utf-8') as f:
        f.write("📐 LATEX-FORMATTED MATHEMATICAL RESPONSES\n")
        f.write("=" * 80 + "\n\n")
        f.write("🎯 TESTING PROPER LATEX NOTATION vs HTML FORMATTING\n")
        f.write("=" * 80 + "\n\n")
        
        for i, item in enumerate(data, 1):
            domain = item.get('domain', 'unknown').replace('_', ' ').title()
            question = item.get('question', 'No question')
            response = item.get('response', 'No response')
            focus = item.get('focus_area', 'No focus specified')
            
            f.write(f"[LaTeX Test {i}] {domain}\n")
            f.write(f"🎯 Focus: {focus}\n")
            f.write("-" * 60 + "\n")
            f.write(f"❓ QUESTION:\n{question}\n\n")
            
            if response.startswith('Error:'):
                f.write(f"❌ RESPONSE: {response}\n\n")
            else:
                f.write(f"📐 LATEX RESPONSE:\n{response}\n\n")
            
            f.write("=" * 80 + "\n\n")
    
    print(f"✅ Created LaTeX test responses: {readable_path}")

if __name__ == "__main__":
    format_latex_responses()
