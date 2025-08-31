"""
Format Response Files for Better Readability
Reformats JSON and creates readable text versions
"""
import json
from pathlib import Path
from config import config

def format_json_file(input_file, output_file):
    """Reformat JSON file with proper indentation"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Formatted {input_file} -> {output_file}")

def create_readable_text(json_file, text_file):
    """Create a readable text version of the responses"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write("📚 EDUCATIONAL LLM RESPONSES\n")
        f.write("=" * 80 + "\n\n")
        
        for i, item in enumerate(data, 1):
            domain = item.get('domain', 'unknown').replace('_', ' ').title()
            question = item.get('question', 'No question')
            response = item.get('response', 'No response')
            
            f.write(f"[Question {i}] {domain}\n")
            f.write("-" * 60 + "\n")
            f.write(f"❓ QUESTION:\n{question}\n\n")
            f.write(f"💬 RESPONSE:\n{response}\n\n")
            f.write("=" * 80 + "\n\n")
    
    print(f"✅ Created readable text: {text_file}")

def main():
    """Format all response files"""
    data_dir = Path(config.data_dir)
    
    # Files to format
    files_to_format = [
        "ollama_baseline_responses.json",
        "missing_domains_responses.json"
    ]
    
    print("🔧 Formatting JSON files for better readability...")
    
    for filename in files_to_format:
        input_path = data_dir / filename
        
        if input_path.exists():
            # Create formatted JSON version
            formatted_json = data_dir / f"formatted_{filename}"
            format_json_file(input_path, formatted_json)
            
            # Create readable text version
            text_filename = filename.replace('.json', '.txt')
            text_path = data_dir / f"readable_{text_filename}"
            create_readable_text(input_path, text_path)
            
        else:
            print(f"❌ File not found: {filename}")
    
    print(f"\n🎯 Summary of formatted files:")
    print(f"📁 Location: {data_dir}")
    print(f"📄 Formatted JSON: formatted_*.json (properly indented)")
    print(f"📖 Readable Text: readable_*.txt (easy to read)")
    
    # Show file sizes
    print(f"\n📊 File Information:")
    for file_path in data_dir.glob("*responses*"):
        size_kb = file_path.stat().st_size / 1024
        print(f"   📄 {file_path.name}: {size_kb:.1f} KB")

if __name__ == "__main__":
    main()
