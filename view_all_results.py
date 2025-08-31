"""
View All Results Combined
Shows all completed evaluation results
"""
import json
from pathlib import Path
from config import config

def main():
    data_dir = Path(config.data_dir)
    
    print("🔍 COMPLETE EVALUATION RESULTS")
    print("="*60)
    
    # Check all result files
    result_files = [
        ("ollama_baseline_responses.json", "Original Sample"),
        ("missing_domains_responses.json", "Missing Domains"),
    ]
    
    all_results = []
    total_successful = 0
    total_questions = 0
    
    for filename, description in result_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"\n📊 {description}: {filename}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            successful = len([r for r in results if not r.get('response', '').startswith('Error:')])
            total_successful += successful
            total_questions += len(results)
            
            print(f"   📈 Questions: {len(results)}")
            print(f"   ✅ Successful: {successful}")
            print(f"   📊 Success Rate: {successful/len(results)*100:.1f}%")
            
            # Count by domain
            domains = {}
            for result in results:
                domain = result.get('domain', 'unknown')
                domains[domain] = domains.get(domain, 0) + 1
            
            print(f"   📚 Domains: {domains}")
            
            all_results.extend(results)
            
        else:
            print(f"❌ Not found: {filename}")
    
    print("\n" + "="*60)
    print("🎯 OVERALL SUMMARY")
    print("="*60)
    print(f"📈 Total Questions: {total_questions}")
    print(f"✅ Total Successful: {total_successful}")
    print(f"📊 Overall Success Rate: {total_successful/total_questions*100:.1f}%")
    
    # Domain breakdown
    all_domains = {}
    for result in all_results:
        domain = result.get('domain', 'unknown')
        all_domains[domain] = all_domains.get(domain, 0) + 1
    
    print(f"\n📚 Complete Domain Coverage:")
    for domain, count in all_domains.items():
        print(f"   • {domain.replace('_', ' ').title()}: {count} questions")
    
    # Show some sample questions
    print(f"\n📝 SAMPLE QUESTIONS:")
    print("-" * 40)
    
    shown_domains = set()
    for result in all_results:
        domain = result.get('domain', 'unknown')
        if domain not in shown_domains and not result.get('response', '').startswith('Error:'):
            shown_domains.add(domain)
            print(f"\n[{domain.upper()}]")
            print(f"❓ {result['question']}")
            response = result.get('response', '')
            print(f"💬 {response[:150]}...")
            print("-" * 40)
            
            if len(shown_domains) >= 3:  # Show max 3 examples
                break
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete! You have responses across all domains.")
    print("🎯 Ready for next steps: manual improvement and fine-tuning!")

if __name__ == "__main__":
    main()

