#!/usr/bin/env python3
"""
Comprehensive Model Analysis - Text-Based Report
Natural analysis of model performance without external dependencies
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import statistics

def load_comprehensive_data() -> Dict:
    """Load and create comprehensive evaluation data from raw files"""
    try:
        # Load from the working evaluation files
        all_results = []
        data_dir = Path("data")
        
        for json_file in data_dir.glob("*evaluation*.json"):
            if 'stats' in str(json_file):
                continue  # Skip stats files
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                for item in data:
                    if isinstance(item, dict) and not item.get('error', False):
                        all_results.append(item)
                        
            except Exception as e:
                print(f"Skipping {json_file}: {e}")
        
        if not all_results:
            return {}
        
        # Create comprehensive data structure
        return create_comprehensive_structure(all_results)
    except Exception as e:
        print(f"Error loading comprehensive data: {e}")
        return {}

def create_comprehensive_structure(results: List[Dict]) -> Dict:
    """Create comprehensive data structure from raw results"""
    # Group by model and approach
    model_stats = {}
    
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
    
    # Calculate averages
    for key, stats in model_stats.items():
        count = len(stats['responses'])
        stats['avg_length'] = stats['total_length'] / count
        stats['avg_time'] = stats['total_time'] / count
        stats['avg_speed'] = stats['total_speed'] / count
        stats['total_questions'] = len(stats['questions'])
        stats['total_responses'] = count
        # Convert set to list for JSON compatibility
        stats['questions'] = list(stats['questions'])
    
    # Create domain analysis
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
    
    # Create approach analysis
    approach_data = {'raw': [], 'ground_rules': []}
    for result in results:
        approach = result['approach']
        if approach in approach_data:
            approach_data[approach].append(result)
    
    approach_analysis = {}
    for approach, data in approach_data.items():
        if data:
            total_length = sum(r['metrics']['char_count'] for r in data)
            total_time = sum(r['metrics']['generation_time'] for r in data)
            total_speed = sum(r['metrics']['chars_per_second'] for r in data)
            count = len(data)
            
            approach_analysis[approach] = {
                'avg_length': total_length / count,
                'avg_time': total_time / count,
                'avg_speed': total_speed / count,
                'sample_size': count
            }
    
    # Calculate improvements
    if 'raw' in approach_analysis and 'ground_rules' in approach_analysis:
        raw = approach_analysis['raw']
        gr = approach_analysis['ground_rules']
        
        approach_analysis['improvements'] = {
            'length_improvement': gr['avg_length'] / raw['avg_length'],
            'time_cost': gr['avg_time'] / raw['avg_time'],
            'speed_change': gr['avg_speed'] / raw['avg_speed']
        }
    
    return {
        'metadata': {
            'report_generated': datetime.now().isoformat(),
            'total_evaluations': len(results),
            'unique_models': len(set(r['model_name'] for r in results)),
            'unique_questions': len(set(r['question_id'] for r in results)),
            'unique_domains': len(set(r['domain'] for r in results)),
            'total_chars_generated': sum(r['metrics']['char_count'] for r in results),
            'total_generation_time': sum(r['metrics']['generation_time'] for r in results),
            'evaluation_period': {
                'start': min(r['timestamp'] for r in results),
                'end': max(r['timestamp'] for r in results)
            }
        },
        'model_performance': model_stats,
        'approach_analysis': approach_analysis,
        'domain_analysis': domain_analysis
    }

def analyze_efficiency_metrics(data: Dict) -> Dict:
    """Analyze efficiency across multiple dimensions"""
    model_performance = data.get('model_performance', {})
    efficiency_analysis = {}
    
    for model_key, stats in model_performance.items():
        model_name = stats['model_name']
        approach = stats['approach']
        
        # Calculate efficiency metrics
        avg_length = stats['avg_length']
        avg_time = stats['avg_time']
        avg_speed = stats['avg_speed']
        total_questions = stats['total_questions']
        
        # Efficiency per character
        time_per_char = avg_time / avg_length if avg_length > 0 else 0
        
        # Quality-speed balance (longer responses often indicate more detail)
        quality_speed_ratio = avg_length / avg_time if avg_time > 0 else 0
        
        # Consistency across questions
        responses = stats['responses']
        speeds = [r['metrics']['chars_per_second'] for r in responses]
        lengths = [r['metrics']['char_count'] for r in responses]
        
        speed_consistency = 1 / (statistics.stdev(speeds) / statistics.mean(speeds)) if len(speeds) > 1 and statistics.mean(speeds) > 0 else 1
        length_consistency = 1 / (statistics.stdev(lengths) / statistics.mean(lengths)) if len(lengths) > 1 and statistics.mean(lengths) > 0 else 1
        
        efficiency_analysis[model_key] = {
            'model_name': model_name,
            'approach': approach,
            'speed_chars_per_sec': avg_speed,
            'time_per_char': time_per_char,
            'quality_speed_ratio': quality_speed_ratio,
            'speed_consistency': speed_consistency,
            'length_consistency': length_consistency,
            'avg_response_length': avg_length,
            'questions_covered': total_questions
        }
    
    return efficiency_analysis

def analyze_response_complexity(data: Dict) -> Dict:
    """Analyze response complexity and depth"""
    model_performance = data.get('model_performance', {})
    complexity_analysis = {}
    
    for model_key, stats in model_performance.items():
        responses = stats['responses']
        
        # Complexity indicators
        total_equations = 0
        total_tables = 0
        total_code_blocks = 0
        total_sections = 0
        technical_terms = 0
        
        for response in responses:
            content = response['response']
            
            # Count LaTeX equations
            total_equations += content.count('$$') // 2 + content.count('$') // 2
            
            # Count tables
            total_tables += content.count('|---')
            
            # Count code blocks
            total_code_blocks += content.count('```')
            
            # Count section headers
            total_sections += content.count('##') + content.count('###')
            
            # Estimate technical complexity (simple heuristic)
            technical_indicators = ['algorithm', 'function', 'method', 'parameter', 'optimization', 'gradient', 'matrix']
            for term in technical_indicators:
                technical_terms += content.lower().count(term)
        
        num_responses = len(responses)
        
        complexity_analysis[model_key] = {
            'model_name': stats['model_name'],
            'approach': stats['approach'],
            'avg_equations_per_response': total_equations / num_responses if num_responses > 0 else 0,
            'avg_tables_per_response': total_tables / num_responses if num_responses > 0 else 0,
            'avg_code_blocks_per_response': total_code_blocks / num_responses if num_responses > 0 else 0,
            'avg_sections_per_response': total_sections / num_responses if num_responses > 0 else 0,
            'technical_density': technical_terms / sum(r['metrics']['word_count'] for r in responses) if sum(r['metrics']['word_count'] for r in responses) > 0 else 0,
            'structure_complexity': (total_sections + total_tables + total_code_blocks) / num_responses if num_responses > 0 else 0
        }
    
    return complexity_analysis

def analyze_domain_performance(data: Dict) -> Dict:
    """Analyze performance across different domains"""
    domain_analysis = data.get('domain_analysis', {})
    domain_insights = {}
    
    for domain, stats in domain_analysis.items():
        domain_name = domain.replace('_', ' ').title()
        
        domain_insights[domain] = {
            'domain_name': domain_name,
            'best_model': stats['best_model'],
            'best_approach': stats['best_approach'],
            'avg_speed': stats['avg_speed'],
            'avg_length': stats['avg_length'],
            'sample_size': stats['sample_size'],
            'performance_category': categorize_domain_performance(stats['avg_speed'], stats['avg_length'])
        }
    
    return domain_insights

def categorize_domain_performance(speed: float, length: float) -> str:
    """Categorize domain performance"""
    if speed > 200 and length > 6000:
        return "High Speed, High Detail"
    elif speed > 200 and length <= 6000:
        return "High Speed, Moderate Detail"
    elif speed <= 200 and length > 6000:
        return "Moderate Speed, High Detail"
    else:
        return "Moderate Speed, Moderate Detail"

def compare_approaches(data: Dict) -> Dict:
    """Compare Raw vs Ground Rules approaches"""
    approach_analysis = data.get('approach_analysis', {})
    comparison = {}
    
    if 'raw' in approach_analysis and 'ground_rules' in approach_analysis:
        raw = approach_analysis['raw']
        ground_rules = approach_analysis['ground_rules']
        improvements = approach_analysis.get('improvements', {})
        
        comparison = {
            'raw_performance': {
                'avg_length': raw['avg_length'],
                'avg_speed': raw['avg_speed'],
                'avg_time': raw['avg_time'],
                'sample_size': raw['sample_size']
            },
            'ground_rules_performance': {
                'avg_length': ground_rules['avg_length'],
                'avg_speed': ground_rules['avg_speed'],
                'avg_time': ground_rules['avg_time'],
                'sample_size': ground_rules['sample_size']
            },
            'improvements': improvements,
            'trade_offs': analyze_trade_offs(improvements)
        }
    
    return comparison

def analyze_trade_offs(improvements: Dict) -> Dict:
    """Analyze trade-offs between approaches"""
    if not improvements:
        return {}
    
    length_gain = improvements.get('length_improvement', 1) - 1
    time_cost = improvements.get('time_cost', 1) - 1
    
    efficiency_ratio = length_gain / time_cost if time_cost > 0 else float('inf')
    
    return {
        'length_gain_percent': length_gain * 100,
        'time_cost_percent': time_cost * 100,
        'efficiency_ratio': efficiency_ratio,
        'verdict': get_trade_off_verdict(length_gain, time_cost, efficiency_ratio)
    }

def get_trade_off_verdict(length_gain: float, time_cost: float, efficiency_ratio: float) -> str:
    """Get verdict on approach trade-offs"""
    if efficiency_ratio > 1.5:
        return "Excellent trade-off: significant quality gain with reasonable time cost"
    elif efficiency_ratio > 1.0:
        return "Good trade-off: quality improvement justifies time cost"
    elif efficiency_ratio > 0.5:
        return "Acceptable trade-off: moderate quality gain for time investment"
    else:
        return "Poor trade-off: time cost outweighs quality benefits"

def generate_model_rankings(efficiency_data: Dict, complexity_data: Dict) -> Dict:
    """Generate comprehensive model rankings"""
    rankings = {
        'speed_ranking': [],
        'comprehensive_ranking': [],
        'efficiency_ranking': [],
        'complexity_ranking': []
    }
    
    # Speed ranking
    speed_sorted = sorted(efficiency_data.items(), key=lambda x: x[1]['speed_chars_per_sec'], reverse=True)
    rankings['speed_ranking'] = [(key, data['speed_chars_per_sec'], data['model_name'], data['approach']) for key, data in speed_sorted]
    
    # Comprehensive ranking (based on response length)
    comprehensive_sorted = sorted(efficiency_data.items(), key=lambda x: x[1]['avg_response_length'], reverse=True)
    rankings['comprehensive_ranking'] = [(key, data['avg_response_length'], data['model_name'], data['approach']) for key, data in comprehensive_sorted]
    
    # Efficiency ranking (quality-speed ratio)
    efficiency_sorted = sorted(efficiency_data.items(), key=lambda x: x[1]['quality_speed_ratio'], reverse=True)
    rankings['efficiency_ranking'] = [(key, data['quality_speed_ratio'], data['model_name'], data['approach']) for key, data in efficiency_sorted]
    
    # Complexity ranking (structure complexity)
    complexity_sorted = sorted(complexity_data.items(), key=lambda x: x[1]['structure_complexity'], reverse=True)
    rankings['complexity_ranking'] = [(key, data['structure_complexity'], data['model_name'], data['approach']) for key, data in complexity_sorted]
    
    return rankings

def create_comprehensive_report(data: Dict) -> str:
    """Create comprehensive text-based analysis report"""
    if not data:
        return "No data available for analysis."
    
    # Perform analyses
    efficiency_analysis = analyze_efficiency_metrics(data)
    complexity_analysis = analyze_response_complexity(data)
    domain_analysis = analyze_domain_performance(data)
    approach_comparison = compare_approaches(data)
    rankings = generate_model_rankings(efficiency_analysis, complexity_analysis)
    
    # Generate report
    report = f"""# Comprehensive Model Performance Analysis

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This analysis examines {data['metadata']['total_evaluations']} evaluations across {data['metadata']['unique_models']} models, covering {data['metadata']['unique_questions']} questions in {data['metadata']['unique_domains']} domains. The focus is on practical performance metrics including efficiency, response quality, and structural complexity.

## Performance Rankings

### Speed Performance
"""
    
    for i, (key, speed, model, approach) in enumerate(rankings['speed_ranking'], 1):
        report += f"{i}. {model} ({approach}): {speed:.1f} chars/sec\n"
    
    report += "\n### Response Comprehensiveness\n"
    for i, (key, length, model, approach) in enumerate(rankings['comprehensive_ranking'], 1):
        report += f"{i}. {model} ({approach}): {length:.0f} characters average\n"
    
    report += "\n### Efficiency (Quality per Second)\n"
    for i, (key, ratio, model, approach) in enumerate(rankings['efficiency_ranking'], 1):
        report += f"{i}. {model} ({approach}): {ratio:.1f} chars per second of generation\n"
    
    report += "\n### Structural Complexity\n"
    for i, (key, complexity, model, approach) in enumerate(rankings['complexity_ranking'], 1):
        report += f"{i}. {model} ({approach}): {complexity:.1f} structural elements per response\n"
    
    # Model-by-model analysis
    report += "\n## Detailed Model Analysis\n\n"
    
    for model_key in efficiency_analysis:
        eff = efficiency_analysis[model_key]
        comp = complexity_analysis[model_key]
        
        report += f"### {eff['model_name']} - {eff['approach'].title()} Mode\n\n"
        report += f"**Performance Metrics:**\n"
        report += f"- Generation Speed: {eff['speed_chars_per_sec']:.1f} chars/sec\n"
        report += f"- Average Response Length: {eff['avg_response_length']:.0f} characters\n"
        report += f"- Time per Character: {eff['time_per_char']:.3f} seconds\n"
        report += f"- Speed Consistency: {eff['speed_consistency']:.2f}\n"
        report += f"- Length Consistency: {eff['length_consistency']:.2f}\n\n"
        
        report += f"**Response Complexity:**\n"
        report += f"- Average Equations per Response: {comp['avg_equations_per_response']:.1f}\n"
        report += f"- Average Tables per Response: {comp['avg_tables_per_response']:.1f}\n"
        report += f"- Average Code Blocks per Response: {comp['avg_code_blocks_per_response']:.1f}\n"
        report += f"- Average Sections per Response: {comp['avg_sections_per_response']:.1f}\n"
        report += f"- Technical Term Density: {comp['technical_density']:.3f}\n"
        report += f"- Overall Structure Complexity: {comp['structure_complexity']:.1f}\n\n"
    
    # Approach comparison
    if approach_comparison:
        report += "## Approach Comparison: Raw vs Ground Rules\n\n"
        
        raw_perf = approach_comparison['raw_performance']
        gr_perf = approach_comparison['ground_rules_performance']
        trade_offs = approach_comparison.get('trade_offs', {})
        
        report += f"**Raw Approach Performance:**\n"
        report += f"- Average Length: {raw_perf['avg_length']:.0f} characters\n"
        report += f"- Average Speed: {raw_perf['avg_speed']:.1f} chars/sec\n"
        report += f"- Average Time: {raw_perf['avg_time']:.1f} seconds\n\n"
        
        report += f"**Ground Rules Approach Performance:**\n"
        report += f"- Average Length: {gr_perf['avg_length']:.0f} characters\n"
        report += f"- Average Speed: {gr_perf['avg_speed']:.1f} chars/sec\n"
        report += f"- Average Time: {gr_perf['avg_time']:.1f} seconds\n\n"
        
        if trade_offs:
            report += f"**Trade-off Analysis:**\n"
            report += f"- Quality Improvement: {trade_offs['length_gain_percent']:.1f}% longer responses\n"
            report += f"- Time Cost: {trade_offs['time_cost_percent']:.1f}% longer generation time\n"
            report += f"- Efficiency Ratio: {trade_offs['efficiency_ratio']:.1f}\n"
            report += f"- Verdict: {trade_offs['verdict']}\n\n"
    
    # Domain analysis
    report += "## Domain-Specific Performance\n\n"
    
    for domain, analysis in domain_analysis.items():
        report += f"### {analysis['domain_name']}\n"
        report += f"- Best Performer: {analysis['best_model']} ({analysis['best_approach']})\n"
        report += f"- Average Speed: {analysis['avg_speed']:.1f} chars/sec\n"
        report += f"- Average Length: {analysis['avg_length']:.0f} characters\n"
        report += f"- Performance Category: {analysis['performance_category']}\n"
        report += f"- Sample Size: {analysis['sample_size']} evaluations\n\n"
    
    # Key insights
    report += "## Key Insights\n\n"
    
    # Speed insights
    fastest_model = rankings['speed_ranking'][0]
    report += f"**Speed Champion:** {fastest_model[2]} in {fastest_model[3]} mode delivers {fastest_model[1]:.1f} chars/sec, making it the fastest option for real-time applications.\n\n"
    
    # Comprehensiveness insights
    most_comprehensive = rankings['comprehensive_ranking'][0]
    report += f"**Most Comprehensive:** {most_comprehensive[2]} in {most_comprehensive[3]} mode produces the longest responses ({most_comprehensive[1]:.0f} characters average), ideal for detailed explanations.\n\n"
    
    # Efficiency insights
    most_efficient = rankings['efficiency_ranking'][0]
    report += f"**Best Efficiency:** {most_efficient[2]} in {most_efficient[3]} mode offers the best quality-to-speed ratio ({most_efficient[1]:.1f}), balancing detail with performance.\n\n"
    
    # Complexity insights
    most_complex = rankings['complexity_ranking'][0]
    report += f"**Highest Complexity:** {most_complex[2]} in {most_complex[3]} mode provides the most structured responses ({most_complex[1]:.1f} elements per response), suitable for technical documentation.\n\n"
    
    # Practical recommendations
    report += "## Practical Recommendations\n\n"
    
    report += "**For Speed-Critical Applications:**\n"
    speed_top = rankings['speed_ranking'][0]
    report += f"Choose {speed_top[2]} in {speed_top[3]} mode for maximum throughput.\n\n"
    
    report += "**For Educational Content:**\n"
    comprehensive_top = rankings['comprehensive_ranking'][0]
    report += f"Use {comprehensive_top[2]} in {comprehensive_top[3]} mode for detailed explanations.\n\n"
    
    report += "**For Balanced Performance:**\n"
    efficiency_top = rankings['efficiency_ranking'][0]
    report += f"Deploy {efficiency_top[2]} in {efficiency_top[3]} mode for optimal quality-speed balance.\n\n"
    
    report += "**For Technical Documentation:**\n"
    complexity_top = rankings['complexity_ranking'][0]
    report += f"Utilize {complexity_top[2]} in {complexity_top[3]} mode for structured, technical responses.\n\n"
    
    if approach_comparison and approach_comparison.get('trade_offs'):
        trade_offs = approach_comparison['trade_offs']
        if trade_offs['efficiency_ratio'] > 1.0:
            report += "**Approach Recommendation:**\n"
            report += f"The Ground Rules approach provides {trade_offs['length_gain_percent']:.1f}% longer responses with only {trade_offs['time_cost_percent']:.1f}% time cost increase. This trade-off is favorable for most educational applications.\n\n"
    
    # Future considerations
    report += "## Future Research Directions\n\n"
    report += "1. **Quality Scoring:** Implement automated content quality assessment\n"
    report += "2. **Domain Expansion:** Test performance across more specialized domains\n"
    report += "3. **Real-world Validation:** Deploy top performers in actual educational settings\n"
    report += "4. **Cost Analysis:** Evaluate computational resource requirements\n"
    report += "5. **User Preference Studies:** Gather feedback on response quality and usefulness\n\n"
    
    report += "---\n\n"
    report += f"*Analysis based on {data['metadata']['total_evaluations']} evaluations*\n"
    report += f"*Data collection period: {data['metadata']['evaluation_period']['start']} to {data['metadata']['evaluation_period']['end']}*\n"
    
    return report

def save_obsidian_report(report: str) -> str:
    """Save comprehensive report to Obsidian"""
    obsidian_vault = Path("/Users/tld/Documents/Obsidian LLM")
    reports_folder = obsidian_vault / "Educational" / "Model Evaluations"
    reports_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"Comprehensive_Performance_Analysis_{timestamp}.md"
    filepath = reports_folder / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return str(filepath)

def main():
    """Generate comprehensive text-based analysis"""
    print("Generating Comprehensive Model Performance Analysis...")
    
    # Load data
    data = load_comprehensive_data()
    
    if not data:
        print("No comprehensive data found. Please ensure evaluation data exists.")
        return
    
    print(f"Loaded data: {data['metadata']['total_evaluations']} evaluations")
    print(f"Models: {data['metadata']['unique_models']}")
    print(f"Questions: {data['metadata']['unique_questions']}")
    print(f"Domains: {data['metadata']['unique_domains']}")
    
    # Generate comprehensive report
    report = create_comprehensive_report(data)
    
    # Save to file
    with open('comprehensive_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Text report saved to: comprehensive_analysis_report.txt")
    
    # Save to Obsidian
    obsidian_path = save_obsidian_report(report)
    print(f"Obsidian report saved to: {obsidian_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*60)
    
    print("\nQuick Summary:")
    efficiency_analysis = analyze_efficiency_metrics(data)
    
    # Find best performers
    fastest = max(efficiency_analysis.items(), key=lambda x: x[1]['speed_chars_per_sec'])
    most_comprehensive = max(efficiency_analysis.items(), key=lambda x: x[1]['avg_response_length'])
    most_efficient = max(efficiency_analysis.items(), key=lambda x: x[1]['quality_speed_ratio'])
    
    print(f"Fastest: {fastest[1]['model_name']} ({fastest[1]['approach']}) - {fastest[1]['speed_chars_per_sec']:.1f} chars/sec")
    print(f"Most Comprehensive: {most_comprehensive[1]['model_name']} ({most_comprehensive[1]['approach']}) - {most_comprehensive[1]['avg_response_length']:.0f} chars")
    print(f"Best Efficiency: {most_efficient[1]['model_name']} ({most_efficient[1]['approach']}) - {most_efficient[1]['quality_speed_ratio']:.1f} chars/sec")
    
    print("\nFull analysis available in generated reports.")

if __name__ == "__main__":
    main()