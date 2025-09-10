#!/usr/bin/env python3
"""
Generate Obsidian-formatted reports from model evaluation results
"""
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

def load_evaluation_results(results_file: str) -> List[Dict]:
    """Load evaluation results from JSON file"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_response_preview(response: str, max_chars: int = 200) -> str:
    """Format response preview for display"""
    if len(response) <= max_chars:
        return response
    return response[:max_chars] + "..."

def generate_model_comparison_table(results: List[Dict]) -> str:
    """Generate comparison table of models"""
    models = {}
    
    # Aggregate results by model and approach
    for result in results:
        if result.get("error"):
            continue
            
        model = result["model_name"]
        approach = result["approach"]
        key = f"{model}_{approach}"
        
        if key not in models:
            models[key] = {
                "model": model,
                "approach": approach,
                "total_chars": 0,
                "total_time": 0,
                "count": 0,
                "questions": []
            }
        
        models[key]["total_chars"] += result["metrics"]["char_count"]
        models[key]["total_time"] += result["metrics"]["generation_time"]
        models[key]["count"] += 1
        models[key]["questions"].append(result["question_id"])
    
    # Calculate averages
    for key, data in models.items():
        if data["count"] > 0:
            data["avg_chars"] = data["total_chars"] / data["count"]
            data["avg_time"] = data["total_time"] / data["count"]
            data["chars_per_second"] = data["total_chars"] / data["total_time"] if data["total_time"] > 0 else 0
    
    # Generate table
    table = """
| Model | Approach | Avg Chars | Avg Time (s) | Chars/sec | Tests |
|-------|----------|-----------|--------------|-----------|-------|
"""
    
    for key, data in sorted(models.items()):
        table += f"| {data['model']} | {data['approach']} | {data['avg_chars']:.0f} | {data['avg_time']:.1f} | {data['chars_per_second']:.0f} | {data['count']} |\n"
    
    return table

def generate_detailed_responses_section(results: List[Dict], question_id: int = None) -> str:
    """Generate detailed responses section"""
    if question_id is not None:
        filtered_results = [r for r in results if r["question_id"] == question_id and not r.get("error")]
        title = f"## Question {question_id} Detailed Responses"
        if filtered_results:
            question_text = filtered_results[0]["question"]
            title += f"\n\n**Question:** {question_text}\n"
    else:
        filtered_results = [r for r in results if not r.get("error")]
        title = "## All Detailed Responses"
    
    content = title + "\n"
    
    for result in filtered_results:
        content += f"\n### {result['model_name']} - {result['approach'].title()}\n\n"
        content += f"**Domain:** {result['domain']}  \n"
        content += f"**Generation Time:** {result['metrics']['generation_time']:.1f}s  \n"
        content += f"**Length:** {result['metrics']['char_count']} chars, {result['metrics']['word_count']} words  \n\n"
        content += f"**Response:**\n\n{result['response']}\n\n"
        content += "---\n"
    
    return content

def calculate_improvement_ratios(results: List[Dict]) -> Dict:
    """Calculate improvement ratios between approaches"""
    raw_results = [r for r in results if r["approach"] == "raw" and not r.get("error")]
    gr_results = [r for r in results if r["approach"] == "ground_rules" and not r.get("error")]
    
    if not raw_results or not gr_results:
        return {}
    
    raw_avg_chars = sum(r["metrics"]["char_count"] for r in raw_results) / len(raw_results)
    gr_avg_chars = sum(r["metrics"]["char_count"] for r in gr_results) / len(gr_results)
    
    raw_avg_time = sum(r["metrics"]["generation_time"] for r in raw_results) / len(raw_results)
    gr_avg_time = sum(r["metrics"]["generation_time"] for r in gr_results) / len(gr_results)
    
    return {
        "length_improvement": gr_avg_chars / raw_avg_chars if raw_avg_chars > 0 else 0,
        "time_ratio": gr_avg_time / raw_avg_time if raw_avg_time > 0 else 0,
        "raw_avg_chars": raw_avg_chars,
        "gr_avg_chars": gr_avg_chars,
        "raw_avg_time": raw_avg_time,
        "gr_avg_time": gr_avg_time
    }

def save_to_obsidian_vault(report_content: str, filename: str = None) -> str:
    """Save report directly to Obsidian vault Educational folder"""
    obsidian_vault = Path("/Users/tld/Documents/Obsidian LLM")
    educational_folder = obsidian_vault / "Educational"
    
    # Create Educational folder if it doesn't exist
    educational_folder.mkdir(parents=True, exist_ok=True)
    
    # Create Model Evaluations subfolder for organization
    eval_folder = educational_folder / "Model Evaluations"
    eval_folder.mkdir(exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Model_Evaluation_Report_{timestamp}.md"
    
    output_file = eval_folder / filename
    output_file.write_text(report_content, encoding='utf-8')
    
    print(f"📝 Report saved to Obsidian vault: {output_file}")
    return str(output_file)

def generate_obsidian_report(results: List[Dict], output_file: str = None) -> str:
    """Generate comprehensive Obsidian-formatted report"""
    
    # Calculate basic stats
    total_tests = len(results)
    successful_tests = len([r for r in results if not r.get("error")])
    models_tested = list(set(r["model_name"] for r in results))
    domains_covered = list(set(r["domain"] for r in results))
    
    improvement_ratios = calculate_improvement_ratios(results)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build report
    report = f"""# New Models Evaluation Report

**Generated:** {timestamp}  
**Total Tests:** {total_tests}  
**Successful:** {successful_tests}/{total_tests} ({successful_tests/total_tests:.1%})  
**Models:** {', '.join(models_tested)}  
**Domains:** {', '.join(domains_covered)}  

## Executive Summary

This evaluation compares the performance of new large language models using both raw prompting and ground rules approaches across {len(set(r["question_id"] for r in results))} diverse questions spanning multiple domains.

### Key Findings

"""
    
    if improvement_ratios:
        report += f"""- **Ground Rules Improvement:** {improvement_ratios['length_improvement']:.1f}x longer responses on average
- **Raw Approach:** {improvement_ratios['raw_avg_chars']:.0f} chars avg, {improvement_ratios['raw_avg_time']:.1f}s avg
- **Ground Rules Approach:** {improvement_ratios['gr_avg_chars']:.0f} chars avg, {improvement_ratios['gr_avg_time']:.1f}s avg
- **Time Trade-off:** {improvement_ratios['time_ratio']:.1f}x longer generation time for {improvement_ratios['length_improvement']:.1f}x more content

"""
    
    report += f"""## Model Performance Comparison
{generate_model_comparison_table(results)}

## Domain Performance Analysis

"""
    
    # Domain breakdown
    domain_stats = {}
    for result in results:
        if result.get("error"):
            continue
        domain = result["domain"]
        if domain not in domain_stats:
            domain_stats[domain] = {"raw": [], "ground_rules": []}
        domain_stats[domain][result["approach"]].append(result["metrics"]["char_count"])
    
    for domain, stats in domain_stats.items():
        raw_avg = sum(stats["raw"]) / len(stats["raw"]) if stats["raw"] else 0
        gr_avg = sum(stats["ground_rules"]) / len(stats["ground_rules"]) if stats["ground_rules"] else 0
        improvement = gr_avg / raw_avg if raw_avg > 0 else 0
        
        report += f"- **{domain.replace('_', ' ').title()}:** {improvement:.1f}x improvement ({raw_avg:.0f} → {gr_avg:.0f} chars)\n"
    
    report += "\n## Question-by-Question Analysis\n\n"
    
    # Group results by question
    questions = {}
    for result in results:
        qid = result["question_id"]
        if qid not in questions:
            questions[qid] = {
                "question": result["question"],
                "domain": result["domain"],
                "results": []
            }
        questions[qid]["results"].append(result)
    
    for qid, qdata in sorted(questions.items()):
        report += f"### Q{qid}: {qdata['question']}\n\n"
        report += f"**Domain:** {qdata['domain']}\n\n"
        
        # Create mini table for this question
        report += "| Model | Approach | Chars | Time (s) | Preview |\n"
        report += "|-------|----------|-------|----------|----------|\n"
        
        for result in qdata["results"]:
            if result.get("error"):
                preview = "ERROR"
                chars = "0"
                time_str = "0"
            else:
                preview = format_response_preview(result["response"], 100)
                chars = str(result["metrics"]["char_count"])
                time_str = f"{result['metrics']['generation_time']:.1f}"
            
            report += f"| {result['model_name']} | {result['approach']} | {chars} | {time_str} | {preview} |\n"
        
        report += "\n"
    
    report += """
## Methodology

### Testing Approach
- **Raw prompting:** Direct question without educational formatting
- **Ground rules prompting:** Research-focused prompting with flexible guidelines
- **Extended context:** 8K tokens maximum for comprehensive responses
- **Consistent parameters:** Temperature 0.7, same model settings across tests

### Evaluation Metrics
- **Response length:** Character and word count as proxy for comprehensiveness
- **Generation time:** Speed of response generation
- **Content quality:** Manual assessment of educational value (qualitative)

### Models Tested
"""
    
    for model in models_tested:
        report += f"- **{model}**\n"
    
    report += f"""
## Technical Details

**Evaluation Framework:** Custom Python evaluation pipeline  
**Base System:** Ollama inference with educational prompting system  
**Knowledge Base:** Disabled for consistent comparison  
**Timestamp:** {timestamp}  

## Conclusions

"""
    
    if improvement_ratios:
        report += f"""The ground rules approach demonstrates a consistent {improvement_ratios['length_improvement']:.1f}x improvement in response comprehensiveness across all tested models, validating the research finding that flexible prompting principles outperform rigid templates for educational content.

The {improvement_ratios['time_ratio']:.1f}x increase in generation time represents an acceptable trade-off for {improvement_ratios['length_improvement']:.1f}x more comprehensive educational content, particularly for research-level learning applications.
"""
    
    report += """
## Next Steps

1. **Qualitative Analysis:** Manual assessment of response quality and educational value
2. **Domain-Specific Optimization:** Fine-tune prompting strategies per domain
3. **User Study:** Gather feedback from actual learners on response quality
4. **Integration:** Deploy best-performing model with ground rules approach

---

*Generated by Educational LLM Evaluation Pipeline*
"""
    
    # Save to file if specified
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Obsidian report saved to: {output_file}")
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Generate Obsidian report from evaluation results")
    parser.add_argument("results_file", help="Path to evaluation results JSON file")
    parser.add_argument("-o", "--output", help="Output file path for Obsidian report")
    parser.add_argument("-q", "--question", type=int, help="Generate detailed report for specific question ID")
    
    args = parser.parse_args()
    
    try:
        results = load_evaluation_results(args.results_file)
        print(f"📊 Loaded {len(results)} evaluation results")
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            base_name = Path(args.results_file).stem
            output_file = f"data/{base_name}_obsidian_report.md"
        
        # Generate report
        if args.question:
            # Generate detailed report for specific question
            report = generate_detailed_responses_section(results, args.question)
            output_file = output_file.replace('.md', f'_q{args.question}.md')
        else:
            # Generate full report
            report = generate_obsidian_report(results, output_file)
        
        print("✅ Report generation completed!")
        
    except FileNotFoundError:
        print(f"❌ Results file not found: {args.results_file}")
    except Exception as e:
        print(f"❌ Error generating report: {e}")

if __name__ == "__main__":
    main()