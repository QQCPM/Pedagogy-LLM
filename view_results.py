"""
View Results in a Readable Format
Shows summaries and sample responses from the evaluation
"""
import json
import os
from pathlib import Path
from config import config

def view_evaluation_results():
    """Display results in a readable format"""
    
    # Check for different result files
    data_dir = Path(config.data_dir)
    result_files = [
        "gemma3_full_evaluation_responses.json",
        "ollama_baseline_responses.json"
    ]
    
    print("🔍 Looking for result files...")
    
    for filename in result_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"✅ Found: {filename}")
            display_results(filepath)
            print("\n" + "="*80 + "\n")
        else:
            print(f"❌ Not found: {filename}")
    
    # Check if evaluation is still running
    print("💡 TIP: If evaluation is still running, try 'ollama_baseline_responses.json' for completed samples")

def display_results(filepath):
    """Display results from a JSON file"""
    print(f"\n📊 RESULTS FROM: {filepath.name}")
    print("="*60)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            print("❌ No results found in file")
            return
        
        print(f"📈 Total Responses: {len(results)}")
        
        # Count by domain
        domains = {}
        successful = 0
        failed = 0
        
        for result in results:
            domain = result.get('domain', 'unknown')
            if domain not in domains:
                domains[domain] = 0
            domains[domain] += 1
            
            # Check if response is successful
            response = result.get('response', '')
            if response.startswith('Error:'):
                failed += 1
            else:
                successful += 1
        
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed/Timeout: {failed}")
        print(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
        
        print(f"\n📚 By Domain:")
        for domain, count in domains.items():
            print(f"  • {domain.replace('_', ' ').title()}: {count} responses")
        
        # Show sample responses (first 3)
        print(f"\n📝 SAMPLE RESPONSES:")
        print("-" * 60)
        
        for i, result in enumerate(results[:3]):
            if result.get('response', '').startswith('Error:'):
                continue
                
            print(f"\n[{i+1}] QUESTION ({result.get('domain', 'unknown')}):")
            print(f"❓ {result['question']}")
            
            print(f"\n💬 RESPONSE (first 200 chars):")
            response = result.get('response', '')
            print(f"{response[:200]}...")
            
            # Show structure analysis
            if analyze_structure(response):
                print("✅ Good educational structure detected")
            else:
                print("⚠️ Basic response structure")
            
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def analyze_structure(response):
    """Check if response has good educational structure"""
    indicators = [
        '# ',  # Headers
        '## ',  # Subheaders
        'intuitive',
        'mathematical',
        'example',
        'why this matters',
        'connection'
    ]
    
    response_lower = response.lower()
    found = sum(1 for indicator in indicators if indicator in response_lower)
    return found >= 3  # At least 3 indicators of good structure

def show_specific_question(question_number=None, domain=None):
    """Show a specific question response"""
    data_dir = Path(config.data_dir)
    
    # Try to find the most recent results file
    for filename in ["gemma3_full_evaluation_responses.json", "ollama_baseline_responses.json"]:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if question_number and question_number <= len(results):
                result = results[question_number - 1]
                print(f"\n🎯 QUESTION {question_number}:")
                print(f"📚 Domain: {result.get('domain', 'unknown')}")
                print(f"❓ Question: {result['question']}")
                print(f"\n💬 FULL RESPONSE:")
                print(result.get('response', 'No response'))
                return
            
            if domain:
                domain_results = [r for r in results if r.get('domain') == domain]
                if domain_results:
                    print(f"\n🎯 FIRST {domain.upper()} QUESTION:")
                    result = domain_results[0]
                    print(f"❓ Question: {result['question']}")
                    print(f"\n💬 FULL RESPONSE:")
                    print(result.get('response', 'No response'))
                    return
            
            break
    
    print("❌ No results found or invalid question number")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "question" and len(sys.argv) > 2:
            show_specific_question(question_number=int(sys.argv[2]))
        elif sys.argv[1] == "domain" and len(sys.argv) > 2:
            show_specific_question(domain=sys.argv[2])
    else:
        view_evaluation_results()
