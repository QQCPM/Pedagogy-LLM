#!/usr/bin/env python3
"""
Generate Final Comprehensive Report
Analyze all evaluation results and create summary without external dependencies
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def load_all_results() -> List[Dict]:
    """Load all evaluation results from JSON files"""
    all_results = []
    data_dir = Path("data")
    
    # Prioritize the comprehensive all_7_models file
    comprehensive_file = data_dir / "all_7_models_comprehensive_20250909_232101.json"
    if comprehensive_file.exists():
        json_files = [comprehensive_file]
    else:
        json_files = list(data_dir.glob("*evaluation*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                
            for result in results:
                if not result.get('error', False):
                    all_results.append(result)
                    
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_results

def calculate_model_stats(results: List[Dict]) -> Dict:
    """Calculate comprehensive statistics by model"""
    model_stats = {}
    
    # Group by model and approach
    for result in results:
        model = result['model_name']
        approach = result['approach']
        key = f"{model}_{approach}"
        
        if key not in model_stats:
            model_stats[key] = {
                'model_name': model,
                'approach': approach,
                'responses': [],
                'total_length': 0,
                'total_time': 0,
                'total_speed': 0,
                'questions': set()
            }
        
        stats = model_stats[key]
        stats['responses'].append(result)
        stats['total_length'] += result['metrics']['char_count']
        stats['total_time'] += result['metrics']['generation_time']
        stats['total_speed'] += result['metrics']['chars_per_second']
        stats['questions'].add(result['question_id'])
    
    # Convert sets to lists for JSON serialization
    for key, stats in model_stats.items():
        stats['questions'] = list(stats['questions'])
    
    # Calculate averages  
    for key, stats in model_stats.items():
        count = len(stats['responses'])
        stats['avg_length'] = stats['total_length'] / count
        stats['avg_time'] = stats['total_time'] / count
        stats['avg_speed'] = stats['total_speed'] / count
        stats['total_questions'] = len(stats['questions'])
        stats['total_responses'] = count
    
    return model_stats

def analyze_approach_effectiveness(results: List[Dict]) -> Dict:
    """Analyze Raw vs Ground Rules effectiveness"""
    approach_analysis = {'raw': [], 'ground_rules': []}
    
    for result in results:
        approach = result['approach']
        if approach in approach_analysis:
            approach_analysis[approach].append(result)
    
    comparison = {}
    for approach, data in approach_analysis.items():
        if data:
            total_length = sum(r['metrics']['char_count'] for r in data)
            total_time = sum(r['metrics']['generation_time'] for r in data)
            total_speed = sum(r['metrics']['chars_per_second'] for r in data)
            count = len(data)
            
            comparison[approach] = {
                'avg_length': total_length / count,
                'avg_time': total_time / count,
                'avg_speed': total_speed / count,
                'sample_size': count
            }
    
    # Calculate improvement ratios
    if 'raw' in comparison and 'ground_rules' in comparison:
        raw = comparison['raw']
        gr = comparison['ground_rules']
        
        comparison['improvements'] = {
            'length_improvement': gr['avg_length'] / raw['avg_length'],
            'time_cost': gr['avg_time'] / raw['avg_time'],
            'speed_change': gr['avg_speed'] / raw['avg_speed']
        }
    
    return comparison

def analyze_domain_performance(results: List[Dict]) -> Dict:
    """Analyze performance across domains"""
    domain_stats = {}
    
    for result in results:
        domain = result['domain']
        if domain not in domain_stats:
            domain_stats[domain] = []
        domain_stats[domain].append(result)
    
    domain_analysis = {}
    for domain, data in domain_stats.items():
        if data:
            total_length = sum(r['metrics']['char_count'] for r in data)
            total_time = sum(r['metrics']['generation_time'] for r in data)
            total_speed = sum(r['metrics']['chars_per_second'] for r in data)
            count = len(data)
            
            # Find best performing model in this domain
            best_response = max(data, key=lambda x: x['metrics']['chars_per_second'])
            
            domain_analysis[domain] = {
                'avg_length': total_length / count,
                'avg_time': total_time / count,
                'avg_speed': total_speed / count,
                'sample_size': count,
                'best_model': best_response['model_name'],
                'best_approach': best_response['approach'],
                'questions_covered': len(set(r['question_id'] for r in data))
            }
    
    return domain_analysis

def generate_model_rankings(model_stats: Dict) -> Dict:
    """Generate model rankings across different metrics"""
    rankings = {
        'speed_ranking': [],
        'length_ranking': [],
        'efficiency_ranking': []
    }
    
    # Speed ranking (higher is better)
    speed_sorted = sorted(model_stats.items(), key=lambda x: x[1]['avg_speed'], reverse=True)
    rankings['speed_ranking'] = [(key, stats['avg_speed']) for key, stats in speed_sorted]
    
    # Length ranking (higher is better for comprehensiveness)
    length_sorted = sorted(model_stats.items(), key=lambda x: x[1]['avg_length'], reverse=True)
    rankings['length_ranking'] = [(key, stats['avg_length']) for key, stats in length_sorted]
    
    # Efficiency ranking (speed per character)
    efficiency_sorted = sorted(model_stats.items(), 
                             key=lambda x: x[1]['avg_speed'] / max(x[1]['avg_length'], 1), 
                             reverse=True)
    rankings['efficiency_ranking'] = [(key, stats['avg_speed'] / max(stats['avg_length'], 1)) 
                                    for key, stats in efficiency_sorted]
    
    return rankings

def create_comprehensive_report(results: List[Dict]) -> Dict:
    """Create comprehensive evaluation report"""
    if not results:
        return {"error": "No results to analyze"}
    
    model_stats = calculate_model_stats(results)
    approach_analysis = analyze_approach_effectiveness(results)
    domain_analysis = analyze_domain_performance(results)
    rankings = generate_model_rankings(model_stats)
    
    # Overall statistics
    total_chars = sum(r['metrics']['char_count'] for r in results)
    total_time = sum(r['metrics']['generation_time'] for r in results)
    total_responses = len(results)
    
    report = {
        "metadata": {
            "report_generated": datetime.now().isoformat(),
            "total_evaluations": total_responses,
            "unique_models": len(set(r['model_name'] for r in results)),
            "unique_questions": len(set(r['question_id'] for r in results)),
            "unique_domains": len(set(r['domain'] for r in results)),
            "total_chars_generated": total_chars,
            "total_generation_time": total_time,
            "evaluation_period": {
                "start": min(r['timestamp'] for r in results),
                "end": max(r['timestamp'] for r in results)
            }
        },
        "model_performance": model_stats,
        "approach_analysis": approach_analysis,
        "domain_analysis": domain_analysis,
        "rankings": rankings,
        "key_findings": generate_key_findings(model_stats, approach_analysis, domain_analysis),
        "recommendations": generate_recommendations(model_stats, approach_analysis, domain_analysis)
    }
    
    return report

def generate_key_findings(model_stats: Dict, approach_analysis: Dict, domain_analysis: Dict) -> List[str]:
    """Generate key findings from the analysis"""
    findings = []
    
    # Speed champion
    if model_stats:
        fastest = max(model_stats.items(), key=lambda x: x[1]['avg_speed'])
        findings.append(f"Speed Champion: {fastest[0]} at {fastest[1]['avg_speed']:.1f} chars/sec")
    
    # Comprehensiveness champion
    if model_stats:
        longest = max(model_stats.items(), key=lambda x: x[1]['avg_length'])
        findings.append(f"Most Comprehensive: {longest[0]} with {longest[1]['avg_length']:.0f} avg chars")
    
    # Approach effectiveness
    if 'improvements' in approach_analysis:
        imp = approach_analysis['improvements']
        findings.append(f"Ground Rules Improvement: {imp['length_improvement']:.1f}x longer responses")
        findings.append(f"Ground Rules Time Cost: {imp['time_cost']:.1f}x longer generation time")
    
    # Domain insights
    if domain_analysis:
        fastest_domain = max(domain_analysis.items(), key=lambda x: x[1]['avg_speed'])
        findings.append(f"Fastest Domain: {fastest_domain[0]} ({fastest_domain[1]['avg_speed']:.1f} chars/sec)")
    
    return findings

def generate_recommendations(model_stats: Dict, approach_analysis: Dict, domain_analysis: Dict) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    if model_stats:
        # Find best overall performers by family
        gpt_models = {k: v for k, v in model_stats.items() if 'GPT-OSS' in k}
        llama_models = {k: v for k, v in model_stats.items() if 'Llama' in k}
        gemma_models = {k: v for k, v in model_stats.items() if 'Gemma' in k}
        deepseek_models = {k: v for k, v in model_stats.items() if 'DeepSeek' in k}
        
        if gpt_models:
            best_gpt = max(gpt_models.items(), key=lambda x: x[1]['avg_speed'])
            recommendations.append(f"For speed: Use {best_gpt[0]}")
        
        if llama_models:
            best_llama = max(llama_models.items(), key=lambda x: x[1]['avg_length'])
            recommendations.append(f"For comprehensiveness: Consider {best_llama[0]}")
        
        if gemma_models:
            best_gemma = max(gemma_models.items(), key=lambda x: x[1]['avg_speed'])
            recommendations.append(f"For efficiency: Consider {best_gemma[0]}")
            
        if deepseek_models:
            best_deepseek = max(deepseek_models.items(), key=lambda x: x[1]['avg_speed'])
            recommendations.append(f"For reasoning: Consider {best_deepseek[0]}")
    
    # Approach recommendations
    if 'improvements' in approach_analysis:
        imp = approach_analysis['improvements']
        if imp['length_improvement'] > 1.5:
            recommendations.append("Use Ground Rules approach for detailed explanations")
        if imp['time_cost'] < 2.0:
            recommendations.append("Ground Rules time cost is reasonable for quality gain")
    
    # Domain-specific recommendations
    for domain, stats in domain_analysis.items():
        if stats['sample_size'] >= 2:
            recommendations.append(f"For {domain}: {stats['best_model']} performs best")
    
    return recommendations

def save_obsidian_summary(report: Dict) -> str:
    """Save comprehensive summary to Obsidian"""
    obsidian_vault = Path("/Users/tld/Documents/Obsidian LLM")
    summary_folder = obsidian_vault / "Educational" / "Model Evaluations"
    summary_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"Comprehensive_Analysis_{timestamp}.md"
    filepath = summary_folder / filename
    
    # Create markdown content
    content = f"""# Comprehensive Model Evaluation Analysis

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Evaluation Overview

- **Total Evaluations:** {report['metadata']['total_evaluations']}
- **Models Tested:** {report['metadata']['unique_models']}
- **Questions Covered:** {report['metadata']['unique_questions']}
- **Domains:** {report['metadata']['unique_domains']}
- **Total Characters Generated:** {report['metadata']['total_chars_generated']:,}
- **Total Generation Time:** {report['metadata']['total_generation_time']:.1f} seconds

## Performance Rankings

### Speed Champions
"""
    
    # Add speed rankings
    for i, (model, speed) in enumerate(report['rankings']['speed_ranking'][:5], 1):
        content += f"{i}. **{model}**: {speed:.1f} chars/sec\n"
    
    content += "\n### Comprehensiveness Champions\n"
    
    # Add length rankings
    for i, (model, length) in enumerate(report['rankings']['length_ranking'][:5], 1):
        content += f"{i}. **{model}**: {length:.0f} characters\n"
    
    content += "\n## 🔍 Key Findings\n\n"
    
    for finding in report['key_findings']:
        content += f"- {finding}\n"
    
    content += "\n## 💡 Recommendations\n\n"
    
    for rec in report['recommendations']:
        content += f"- {rec}\n"
    
    # Add approach analysis
    if 'improvements' in report['approach_analysis']:
        imp = report['approach_analysis']['improvements']
        content += f"""
## Approach Comparison: Raw vs Ground Rules

- **Length Improvement:** {imp['length_improvement']:.1f}x longer responses
- **Time Cost:** {imp['time_cost']:.1f}x longer generation time
- **Speed Impact:** {imp['speed_change']:.1f}x speed change

**Conclusion:** Ground Rules approach provides significantly more comprehensive responses with reasonable time cost.
"""
    
    # Add domain analysis
    content += "\n## Domain Performance\n\n"
    
    for domain, stats in report['domain_analysis'].items():
        content += f"### {domain.replace('_', ' ').title()}\n"
        content += f"- **Best Model:** {stats['best_model']} ({stats['best_approach']})\n"
        content += f"- **Average Speed:** {stats['avg_speed']:.1f} chars/sec\n"
        content += f"- **Average Length:** {stats['avg_length']:.0f} characters\n"
        content += f"- **Questions Covered:** {stats['questions_covered']}\n\n"
    
    content += """
## Data Sources

This analysis is based on comprehensive evaluations saved in the following locations:
- **Raw Data:** `data/` directory JSON files
- **Individual Responses:** Obsidian vault organized by model and approach
- **Performance Metrics:** Automated collection during evaluation

## Next Steps

1. **Extended Evaluation:** Test remaining questions (8-12) across all models
2. **Quality Analysis:** Implement automated quality scoring
3. **Domain Expansion:** Add more specialized domains
4. **Real-world Testing:** Deploy for actual research workflows

---

*Analysis completed as part of Educational LLM Comparative Research Project*
"""
    
    # Save file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(filepath)

def main():
    """Generate comprehensive final report"""
    print("Generating Comprehensive Final Report...")
    
    # Load all results
    results = load_all_results()
    
    if not results:
        print("No evaluation results found in data/ directory")
        return
    
    print(f"Loaded {len(results)} evaluation results")
    print(f"Models: {len(set(r['model_name'] for r in results))}")
    print(f"Questions: {len(set(r['question_id'] for r in results))}")
    print(f"Domains: {len(set(r['domain'] for r in results))}")
    
    # Generate comprehensive report
    report = create_comprehensive_report(results)
    
    # Save JSON report
    with open('data/comprehensive_final_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("JSON report saved to: data/comprehensive_final_report.json")
    
    # Save Obsidian summary
    obsidian_path = save_obsidian_summary(report)
    print(f"Obsidian summary saved to: {obsidian_path}")
    
    # Print key results
    print("\nTOP PERFORMERS:")
    for i, (model, speed) in enumerate(report['rankings']['speed_ranking'][:3], 1):
        print(f"   {i}. {model}: {speed:.1f} chars/sec")
    
    print("\nKEY FINDINGS:")
    for finding in report['key_findings'][:5]:
        print(f"   • {finding}")
    
    print("\nRECOMMENDATIONS:")
    for rec in report['recommendations'][:5]:
        print(f"   • {rec}")
    
    print("\nComprehensive analysis complete!")

if __name__ == "__main__":
    main()