#!/usr/bin/env python3
"""
Monitor evaluation progress and extract ground rules responses 
(which are already in Obsidian format) to the vault as they complete
"""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def check_for_new_results() -> List[Dict]:
    """Check for new evaluation results files"""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    
    # Look for the latest evaluation results
    result_files = list(data_dir.glob("new_models_evaluation_*.json"))
    if not result_files:
        return []
    
    # Get the most recent file
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_ground_rules_response_directly(result: Dict) -> str:
    """Save ground rules response directly to Obsidian (it's already formatted)"""
    
    # Only process ground rules responses (they're already Obsidian-ready)
    if result.get('approach') != 'ground_rules':
        return None
    
    # Get the response content (already Obsidian-formatted)
    response_content = result.get('response', '')
    if not response_content or len(response_content) < 100:
        return None
    
    # Create file path in Obsidian vault
    obsidian_vault = Path("/Users/tld/Documents/Obsidian LLM")
    educational_folder = obsidian_vault / "Educational"
    
    # Create organized folder: Ground Rules Responses/Model/
    model_clean = result['model_name'].replace(" ", "_").replace(".", "_")
    model_folder = educational_folder / "Model Evaluations" / "Ground Rules Responses" / model_clean
    model_folder.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    question_id = result.get('question_id', 'unknown')
    domain = result.get('domain', 'general')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"Q{question_id}_{domain}_ground_rules_{timestamp}.md"
    filepath = model_folder / filename
    
    # Add minimal header to the existing Obsidian-ready content
    header = f"""# {result['model_name']} - Ground Rules Response

**Question {question_id}:** {result.get('question', 'Unknown')}  
**Domain:** {domain}  
**Generated:** {result.get('timestamp', 'Unknown')}  
**Response Length:** {result.get('metrics', {}).get('char_count', 0):,} characters  

---

"""
    
    # Combine header with the already-formatted response
    full_content = header + response_content
    
    # Save to Obsidian vault
    filepath.write_text(full_content, encoding='utf-8')
    
    print(f"📝 Saved ground rules response: {filepath}")
    return str(filepath)

def monitor_and_extract():
    """Monitor evaluation and extract ground rules responses as they complete"""
    print("🔍 Monitoring evaluation for ground rules responses...")
    print("💡 Ground rules responses are already Obsidian-ready with LaTeX formatting")
    
    processed_results = set()
    last_check_time = time.time()
    
    while True:
        # Check for new results every 30 seconds
        current_results = check_for_new_results()
        
        if current_results:
            new_ground_rules = []
            
            for result in current_results:
                # Create unique ID for each result
                result_id = f"{result.get('question_id')}_{result.get('model')}_{result.get('approach')}"
                
                if (result_id not in processed_results and 
                    result.get('approach') == 'ground_rules' and
                    not result.get('error', False)):
                    
                    # Save this ground rules response
                    filepath = save_ground_rules_response_directly(result)
                    if filepath:
                        new_ground_rules.append(result)
                        processed_results.add(result_id)
            
            if new_ground_rules:
                print(f"✅ Extracted {len(new_ground_rules)} new ground rules responses")
                
                # Show summary
                for result in new_ground_rules:
                    print(f"  - {result['model_name']} Q{result['question_id']}: {result['metrics']['char_count']} chars")
        
        # Show status every 5 minutes
        if time.time() - last_check_time > 300:  # 5 minutes
            print(f"⏳ Still monitoring... {len(processed_results)} ground rules responses extracted so far")
            last_check_time = time.time()
        
        time.sleep(30)  # Check every 30 seconds

def main():
    """Main monitoring function"""
    print("🎯 Ground Rules Response Extractor")
    print("="*50)
    print("This will monitor the main evaluation and extract ground rules responses")
    print("(which are already in perfect Obsidian format) directly to your vault.")
    print()
    
    try:
        monitor_and_extract()
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()