#!/usr/bin/env python3
"""
Update Comprehensive Analysis to include ALL models
Analyze all available evaluation data including new comprehensive results
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import statistics

def load_all_evaluation_data() -> List[Dict]:
    """Load all evaluation data from all JSON files"""
    all_results = []
    data_dir = Path("data")
    
    print("Loading evaluation data from:")
    
    for json_file in data_dir.glob("*.json"):
        if 'stats' in str(json_file) or 'comprehensive_final_report' in str(json_file):
            continue  # Skip stats and report files
            
        try:
            print(f"  📁 {json_file.name}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different data structures
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and not item.get('error', False):
                        all_results.append(item)
            elif isinstance(data, dict):
                # Single result or nested structure
                if 'results' in data:
                    for item in data['results']:
                        if not item.get('error', False):
                            all_results.append(item)
                elif not data.get('error', False):
                    all_results.append(data)
                    
        except Exception as e:
            print(f"    ⚠️  Error loading {json_file}: {e}")
    
    # Filter and normalize results
    normalized_results = []
    for result in all_results:
        # Ensure required fields exist
        if all(field in result for field in ['model_name', 'approach', 'metrics', 'question']):
            # Normalize model names
            model_name = result['model_name']
            if 'gpt-oss:120b' in str(result.get('model', '')).lower():
                model_name = 'GPT-OSS 120B'
            elif 'gpt-oss:20b' in str(result.get('model', '')).lower():
                model_name = 'GPT-OSS 20B'
            elif 'llama3.3' in str(result.get('model', '')).lower():
                model_name = 'Llama 3.3 70B'
            elif 'llama3.1' in str(result.get('model', '')).lower():
                model_name = 'Llama 3.1 70B'
            elif 'deepseek-r1' in str(result.get('model', '')).lower():
                model_name = 'DeepSeek R1 70B'
            elif 'gemma3:27b' in str(result.get('model', '')).lower():
                model_name = 'Gemma 3 27B'
            elif 'gemma3:12b' in str(result.get('model', '')).lower():
                model_name = 'Gemma 3 12B'
            
            result['model_name'] = model_name
            normalized_results.append(result)
    
    print(f"📊 Total valid results loaded: {len(normalized_results)}")
    
    # Show model distribution
    model_counts = {}
    for result in normalized_results:
        model = result['model_name']
        model_counts[model] = model_counts.get(model, 0) + 1
    
    print("🤖 Model distribution:")
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count} evaluations")
    
    return normalized_results

def analyze_all_models_performance(results: List[Dict]) -> Dict:
    """Analyze performance across all models"""
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
                'questions': set(),
                'domains': set()
            }
        
        stats = model_stats[key]
        stats['responses'].append(result)
        stats['total_length'] += result['metrics']['char_count']
        stats['total_time'] += result['metrics']['generation_time']
        stats['total_speed'] += result['metrics']['chars_per_second']
        
        if 'question_id' in result:
            stats['questions'].add(result['question_id'])
        if 'domain' in result:
            stats['domains'].add(result['domain'])
    
    # Calculate comprehensive statistics
    for key, stats in model_stats.items():
        count = len(stats['responses'])
        stats['avg_length'] = stats['total_length'] / count
        stats['avg_time'] = stats['total_time'] / count
        stats['avg_speed'] = stats['total_speed'] / count
        stats['total_questions'] = len(stats['questions'])
        stats['total_domains'] = len(stats['domains'])
        stats['total_responses'] = count
        
        # Calculate consistency metrics
        speeds = [r['metrics']['chars_per_second'] for r in stats['responses']]
        lengths = [r['metrics']['char_count'] for r in stats['responses']]
        
        if len(speeds) > 1:
            stats['speed_std'] = statistics.stdev(speeds)
            stats['speed_consistency'] = 1 / (stats['speed_std'] / stats['avg_speed']) if stats['avg_speed'] > 0 else 1
        else:
            stats['speed_std'] = 0
            stats['speed_consistency'] = 1
            
        if len(lengths) > 1:
            stats['length_std'] = statistics.stdev(lengths)
            stats['length_consistency'] = 1 / (stats['length_std'] / stats['avg_length']) if stats['avg_length'] > 0 else 1
        else:
            stats['length_std'] = 0
            stats['length_consistency'] = 1
        
        # Convert sets to lists for JSON compatibility
        stats['questions'] = list(stats['questions'])
        stats['domains'] = list(stats['domains'])
    
    return model_stats

def create_comprehensive_model_comparison(model_stats: Dict) -> str:
    """Create comprehensive model comparison report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate total statistics
    total_evaluations = sum(stats['total_responses'] for stats in model_stats.values())
    unique_models = len(set(stats['model_name'] for stats in model_stats.values()))
    
    report = f"""# Comprehensive All Models Performance Analysis

Generated: {timestamp}

## Executive Summary

This analysis examines {total_evaluations} evaluations across {unique_models} models, testing both Raw and Ground Rules approaches. The evaluation covers multiple domains including Earth Science, Machine Learning, Quantum Computing, Deep Learning, Materials Science, and Numerical Methods.

## Available Models Analysis

"""
    
    # Group by model for size comparison
    models_info = {}
    for key, stats in model_stats.items():
        model_name = stats['model_name']
        if model_name not in models_info:
            models_info[model_name] = {'raw': None, 'ground_rules': None}
        models_info[model_name][stats['approach']] = stats
    
    # Model size categories
    model_sizes = {
        'GPT-OSS 120B': '120B parameters (65GB)',
        'GPT-OSS 20B': '20B parameters (13GB)', 
        'Llama 3.3 70B': '70B parameters (42GB)',
        'Llama 3.1 70B': '70B parameters (74GB)',
        'DeepSeek R1 70B': '70B parameters (42GB)',
        'Gemma 3 27B': '27B parameters (17GB)',
        'Gemma 3 12B': '12B parameters (8GB)'
    }
    
    report += "### Model Overview\n\n"
    for model_name in sorted(models_info.keys()):
        size_info = model_sizes.get(model_name, 'Unknown size')
        raw_data = models_info[model_name].get('raw')
        gr_data = models_info[model_name].get('ground_rules')
        
        report += f"**{model_name}** ({size_info})\n"
        
        if raw_data:
            report += f"- Raw Mode: {raw_data['avg_speed']:.1f} chars/sec, {raw_data['avg_length']:.0f} chars avg\n"
        if gr_data:
            report += f"- Ground Rules: {gr_data['avg_speed']:.1f} chars/sec, {gr_data['avg_length']:.0f} chars avg\n"
        
        total_evals = (raw_data['total_responses'] if raw_data else 0) + (gr_data['total_responses'] if gr_data else 0)
        report += f"- Total Evaluations: {total_evals}\n\n"
    
    # Performance rankings
    report += "## Performance Rankings\n\n"
    
    # Speed ranking
    speed_ranking = sorted(model_stats.items(), key=lambda x: x[1]['avg_speed'], reverse=True)
    report += "### Speed Performance (chars/sec)\n"
    for i, (key, stats) in enumerate(speed_ranking, 1):
        report += f"{i}. **{stats['model_name']}** ({stats['approach']}): {stats['avg_speed']:.1f} chars/sec\n"
    
    # Comprehensiveness ranking
    length_ranking = sorted(model_stats.items(), key=lambda x: x[1]['avg_length'], reverse=True)
    report += "\n### Response Comprehensiveness (average length)\n"
    for i, (key, stats) in enumerate(length_ranking, 1):
        report += f"{i}. **{stats['model_name']}** ({stats['approach']}): {stats['avg_length']:.0f} characters\n"
    
    # Efficiency ranking (chars per second of generation time)
    efficiency_ranking = sorted(model_stats.items(), key=lambda x: x[1]['avg_length'] / x[1]['avg_time'] if x[1]['avg_time'] > 0 else 0, reverse=True)
    report += "\n### Efficiency (chars per second of generation)\n"
    for i, (key, stats) in enumerate(efficiency_ranking, 1):
        efficiency = stats['avg_length'] / stats['avg_time'] if stats['avg_time'] > 0 else 0
        report += f"{i}. **{stats['model_name']}** ({stats['approach']}): {efficiency:.1f} chars/gen_sec\n"
    
    # Model size vs performance analysis
    report += "\n## Model Size vs Performance Analysis\n\n"
    
    # Group by model size categories
    size_categories = {
        'Large Models (70B+)': ['GPT-OSS 120B', 'Llama 3.3 70B', 'Llama 3.1 70B', 'DeepSeek R1 70B'],
        'Medium Models (20-30B)': ['GPT-OSS 20B', 'Gemma 3 27B'],
        'Small Models (<20B)': ['Gemma 3 12B']
    }
    
    for category, model_list in size_categories.items():
        report += f"### {category}\n\n"
        
        category_stats = []
        for key, stats in model_stats.items():
            if stats['model_name'] in model_list:
                category_stats.append(stats)
        
        if category_stats:
            avg_speed = sum(s['avg_speed'] for s in category_stats) / len(category_stats)
            avg_length = sum(s['avg_length'] for s in category_stats) / len(category_stats)
            best_speed = max(category_stats, key=lambda x: x['avg_speed'])
            best_comprehensive = max(category_stats, key=lambda x: x['avg_length'])
            
            report += f"- Average Speed: {avg_speed:.1f} chars/sec\n"
            report += f"- Average Response Length: {avg_length:.0f} characters\n"
            report += f"- Fastest: {best_speed['model_name']} ({best_speed['approach']}) - {best_speed['avg_speed']:.1f} chars/sec\n"
            report += f"- Most Comprehensive: {best_comprehensive['model_name']} ({best_comprehensive['approach']}) - {best_comprehensive['avg_length']:.0f} chars\n\n"
    
    # Approach comparison across all models
    report += "## Raw vs Ground Rules Comparison\n\n"
    
    raw_results = [stats for stats in model_stats.values() if stats['approach'] == 'raw']
    gr_results = [stats for stats in model_stats.values() if stats['approach'] == 'ground_rules']
    
    if raw_results and gr_results:
        raw_avg_speed = sum(s['avg_speed'] for s in raw_results) / len(raw_results)
        raw_avg_length = sum(s['avg_length'] for s in raw_results) / len(raw_results)
        raw_avg_time = sum(s['avg_time'] for s in raw_results) / len(raw_results)
        
        gr_avg_speed = sum(s['avg_speed'] for s in gr_results) / len(gr_results)
        gr_avg_length = sum(s['avg_length'] for s in gr_results) / len(gr_results)
        gr_avg_time = sum(s['avg_time'] for s in gr_results) / len(gr_results)
        
        length_improvement = gr_avg_length / raw_avg_length if raw_avg_length > 0 else 1
        time_cost = gr_avg_time / raw_avg_time if raw_avg_time > 0 else 1
        speed_change = gr_avg_speed / raw_avg_speed if raw_avg_speed > 0 else 1
        
        report += f"**Raw Approach (Average across all models):**\n"
        report += f"- Speed: {raw_avg_speed:.1f} chars/sec\n"
        report += f"- Length: {raw_avg_length:.0f} characters\n"
        report += f"- Time: {raw_avg_time:.1f} seconds\n\n"
        
        report += f"**Ground Rules Approach (Average across all models):**\n"
        report += f"- Speed: {gr_avg_speed:.1f} chars/sec\n"
        report += f"- Length: {gr_avg_length:.0f} characters\n"
        report += f"- Time: {gr_avg_time:.1f} seconds\n\n"
        
        report += f"**Overall Impact of Ground Rules:**\n"
        report += f"- Response Length: {(length_improvement-1)*100:.1f}% increase\n"
        report += f"- Generation Time: {(time_cost-1)*100:.1f}% increase\n"
        report += f"- Speed Change: {(speed_change-1)*100:.1f}% change\n\n"
    
    # Key insights and recommendations
    report += "## Key Insights\n\n"
    
    fastest_overall = max(model_stats.items(), key=lambda x: x[1]['avg_speed'])
    most_comprehensive = max(model_stats.items(), key=lambda x: x[1]['avg_length'])
    most_efficient = max(efficiency_ranking[:5], key=lambda x: x[1]['avg_length'] / x[1]['avg_time'] if x[1]['avg_time'] > 0 else 0)
    
    report += f"**Speed Champion:** {fastest_overall[1]['model_name']} in {fastest_overall[1]['approach']} mode achieves {fastest_overall[1]['avg_speed']:.1f} chars/sec\n\n"
    report += f"**Most Comprehensive:** {most_comprehensive[1]['model_name']} in {most_comprehensive[1]['approach']} mode produces {most_comprehensive[1]['avg_length']:.0f} character responses\n\n"
    
    efficiency_score = most_efficient[1]['avg_length'] / most_efficient[1]['avg_time'] if most_efficient[1]['avg_time'] > 0 else 0
    report += f"**Best Efficiency:** {most_efficient[1]['model_name']} in {most_efficient[1]['approach']} mode with {efficiency_score:.1f} chars per generation second\n\n"
    
    # Model-specific insights
    report += "## Model-Specific Performance Insights\n\n"
    
    for model_name in sorted(models_info.keys()):
        raw_data = models_info[model_name].get('raw')
        gr_data = models_info[model_name].get('ground_rules')
        
        report += f"### {model_name}\n\n"
        
        if raw_data and gr_data:
            speed_improvement = gr_data['avg_speed'] / raw_data['avg_speed'] if raw_data['avg_speed'] > 0 else 1
            length_improvement = gr_data['avg_length'] / raw_data['avg_length'] if raw_data['avg_length'] > 0 else 1
            
            report += f"- **Raw Performance:** {raw_data['avg_speed']:.1f} chars/sec, {raw_data['avg_length']:.0f} chars\n"
            report += f"- **Ground Rules Performance:** {gr_data['avg_speed']:.1f} chars/sec, {gr_data['avg_length']:.0f} chars\n"
            report += f"- **Speed Change:** {(speed_improvement-1)*100:+.1f}%\n"
            report += f"- **Length Improvement:** {(length_improvement-1)*100:+.1f}%\n"
            
            # Determine model characteristics
            if gr_data['avg_speed'] > 200:
                speed_category = "Very Fast"
            elif gr_data['avg_speed'] > 100:
                speed_category = "Fast"
            elif gr_data['avg_speed'] > 50:
                speed_category = "Moderate"
            else:
                speed_category = "Slow"
                
            if gr_data['avg_length'] > 8000:
                detail_category = "Highly Detailed"
            elif gr_data['avg_length'] > 5000:
                detail_category = "Detailed"
            else:
                detail_category = "Concise"
                
            report += f"- **Characteristics:** {speed_category}, {detail_category}\n"
            
        elif raw_data or gr_data:
            data = raw_data or gr_data
            approach = "Raw" if raw_data else "Ground Rules"
            report += f"- **{approach} Only:** {data['avg_speed']:.1f} chars/sec, {data['avg_length']:.0f} chars\n"
        
        report += "\n"
    
    # Practical recommendations
    report += "## Practical Recommendations\n\n"
    
    report += "### For Different Use Cases\n\n"
    
    report += "**Speed-Critical Applications:**\n"
    top_speed = speed_ranking[0]
    report += f"- Primary: {top_speed[1]['model_name']} ({top_speed[1]['approach']}) - {top_speed[1]['avg_speed']:.1f} chars/sec\n"
    report += f"- Alternative: {speed_ranking[1][1]['model_name']} ({speed_ranking[1][1]['approach']}) - {speed_ranking[1][1]['avg_speed']:.1f} chars/sec\n\n"
    
    report += "**Comprehensive Educational Content:**\n"
    top_comprehensive = length_ranking[0]
    report += f"- Primary: {top_comprehensive[1]['model_name']} ({top_comprehensive[1]['approach']}) - {top_comprehensive[1]['avg_length']:.0f} chars\n"
    report += f"- Alternative: {length_ranking[1][1]['model_name']} ({length_ranking[1][1]['approach']}) - {length_ranking[1][1]['avg_length']:.0f} chars\n\n"
    
    report += "**Balanced Performance:**\n"
    top_efficient = efficiency_ranking[0]
    efficiency_score = top_efficient[1]['avg_length'] / top_efficient[1]['avg_time'] if top_efficient[1]['avg_time'] > 0 else 0
    report += f"- Primary: {top_efficient[1]['model_name']} ({top_efficient[1]['approach']}) - {efficiency_score:.1f} efficiency score\n\n"
    
    report += "### Resource Considerations\n\n"
    
    report += "**For Limited Resources:**\n"
    small_models = [stats for stats in model_stats.values() if 'Gemma 3 12B' in stats['model_name']]
    if small_models:
        best_small = max(small_models, key=lambda x: x['avg_speed'])
        report += f"- {best_small['model_name']} ({best_small['approach']}) offers {best_small['avg_speed']:.1f} chars/sec with minimal resource requirements\n\n"
    
    report += "**For Maximum Performance:**\n"
    large_models = [stats for stats in model_stats.values() if any(size in stats['model_name'] for size in ['120B', '70B'])]
    if large_models:
        best_large = max(large_models, key=lambda x: x['avg_speed'])
        report += f"- {best_large['model_name']} ({best_large['approach']}) provides {best_large['avg_speed']:.1f} chars/sec with high resource usage\n\n"
    
    report += "---\n\n"
    report += f"*Analysis based on {total_evaluations} evaluations across {unique_models} models*\n"
    report += f"*Generated: {timestamp}*"
    
    return report

def main():
    """Generate updated comprehensive analysis"""
    print("🔄 Updating Comprehensive Analysis with ALL Models...")
    
    # Load all evaluation data
    all_results = load_all_evaluation_data()
    
    if not all_results:
        print("❌ No evaluation data found")
        return
    
    print(f"📊 Analyzing {len(all_results)} total evaluations")
    
    # Analyze all models
    model_stats = analyze_all_models_performance(all_results)
    
    print(f"🤖 Found {len(model_stats)} model-approach combinations")
    
    # Create comprehensive report
    report = create_comprehensive_model_comparison(model_stats)
    
    # Save text report
    with open('comprehensive_all_models_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📝 Text report saved to: comprehensive_all_models_analysis.txt")
    
    # Save to Obsidian
    obsidian_vault = Path("/Users/tld/Documents/Obsidian LLM")
    analysis_folder = obsidian_vault / "Educational" / "Model Evaluations"
    analysis_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    obsidian_filename = f"All_Models_Comprehensive_Analysis_{timestamp}.md"
    obsidian_filepath = analysis_folder / obsidian_filename
    
    with open(obsidian_filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📝 Obsidian report saved to: {obsidian_filepath}")
    
    # Print summary
    unique_models = len(set(stats['model_name'] for stats in model_stats.values()))
    total_evals = sum(stats['total_responses'] for stats in model_stats.values())
    
    print(f"\n🏆 ANALYSIS SUMMARY:")
    print(f"📊 Total Evaluations: {total_evals}")
    print(f"🤖 Unique Models: {unique_models}")
    print(f"🔄 Approaches: Raw and Ground Rules")
    
    # Top performers
    fastest = max(model_stats.items(), key=lambda x: x[1]['avg_speed'])
    most_comprehensive = max(model_stats.items(), key=lambda x: x[1]['avg_length'])
    
    print(f"⚡ Fastest: {fastest[1]['model_name']} ({fastest[1]['approach']}) - {fastest[1]['avg_speed']:.1f} chars/sec")
    print(f"📏 Most Comprehensive: {most_comprehensive[1]['model_name']} ({most_comprehensive[1]['approach']}) - {most_comprehensive[1]['avg_length']:.0f} chars")
    
    print(f"\n✅ Comprehensive analysis complete!")

if __name__ == "__main__":
    main()