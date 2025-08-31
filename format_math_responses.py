"""
Format Math-Focused Responses
"""
import json
from pathlib import Path
from config import config

def format_math_responses():
    data_dir = Path(config.data_dir)
    
    # Find the latest math responses file
    math_files = list(data_dir.glob("math_focused_responses_*.json"))
    if not math_files:
        print("❌ No math response files found")
        return
    
    latest_math_file = sorted(math_files)[-1]
    print(f"📄 Processing: {latest_math_file.name}")
    
    with open(latest_math_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create formatted JSON
    formatted_path = data_dir / "formatted_math_responses.json"
    with open(formatted_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Create readable text
    readable_path = data_dir / "readable_math_responses.txt"
    with open(readable_path, 'w', encoding='utf-8') as f:
        f.write("🧮 MATH-FOCUSED EDUCATIONAL RESPONSES\n")
        f.write("=" * 80 + "\n\n")
        
        for i, item in enumerate(data, 1):
            domain = item.get('domain', 'unknown').replace('_', ' ').title()
            question = item.get('question', 'No question')
            response = item.get('response', 'No response')
            focus = item.get('focus_area', 'No focus specified')
            
            f.write(f"[Question {i}] {domain}\n")
            f.write(f"🎯 Focus: {focus}\n")
            f.write("-" * 60 + "\n")
            f.write(f"❓ QUESTION:\n{question}\n\n")
            
            if response.startswith('Error:'):
                f.write(f"❌ RESPONSE: {response}\n\n")
            else:
                f.write(f"💬 RESPONSE:\n{response}\n\n")
            
            f.write("=" * 80 + "\n\n")
    
    print(f"✅ Created formatted math responses:")
    print(f"   📄 JSON: {formatted_path}")
    print(f"   📖 Text: {readable_path}")

if __name__ == "__main__":
    format_math_responses()
