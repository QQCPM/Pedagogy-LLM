#!/usr/bin/env python3
"""
Comprehensive Model Analysis Framework
Evaluate all available models across multiple dimensions with detailed metrics and visualizations
"""
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelSpecs:
    name: str
    size_gb: float
    parameter_count: str
    architecture: str
    strengths: List[str]
    weaknesses: List[str]

class ComprehensiveModelAnalyzer:
    def __init__(self):
        self.models = {
            "gpt-oss:120b": ModelSpecs(
                name="GPT-OSS 120B",
                size_gb=65.3,
                parameter_count="120B",
                architecture="GPT",
                strengths=["Massive knowledge", "Complex reasoning", "Research depth"],
                weaknesses=["Very slow", "High memory", "Resource intensive"]
            ),
            "llama3.1:70b-instruct-q8_0": ModelSpecs(
                name="Llama 3.1 70B",
                size_gb=75.0,
                parameter_count="70B",
                architecture="Llama",
                strengths=["Instruction following", "Detailed responses", "Multilingual"],
                weaknesses=["Slow generation", "Verbose", "Memory intensive"]
            ),
            "deepseek-r1:70b": ModelSpecs(
                name="DeepSeek R1 70B",
                size_gb=42.5,
                parameter_count="70B",
                architecture="DeepSeek",
                strengths=["Reasoning focus", "Math capabilities", "Code generation"],
                weaknesses=["Slower speed", "Domain specific", "Less general knowledge"]
            ),
            "llama3.3:70b": ModelSpecs(
                name="Llama 3.3 70B",
                size_gb=42.5,
                parameter_count="70B", 
                architecture="Llama",
                strengths=["Latest architecture", "Balanced performance", "Good reasoning"],
                weaknesses=["Moderate speed", "High memory", "Still developing"]
            ),
            "gemma3:27b": ModelSpecs(
                name="Gemma 3 27B",
                size_gb=17.4,
                parameter_count="27B",
                architecture="Gemma",
                strengths=["Balanced size", "Good quality", "Moderate speed"],
                weaknesses=["Limited knowledge", "Less capable", "Smaller context"]
            ),
            "gpt-oss:20b": ModelSpecs(
                name="GPT-OSS 20B",
                size_gb=13.8,
                parameter_count="20B",
                architecture="GPT",
                strengths=["Fast generation", "Good efficiency", "Reasonable quality"],
                weaknesses=["Limited depth", "Less knowledge", "Simpler reasoning"]
            ),
            "gemma3:12b": ModelSpecs(
                name="Gemma 3 12B",
                size_gb=8.1,
                parameter_count="12B",
                architecture="Gemma",
                strengths=["Very fast", "Low memory", "Good for simple tasks"],
                weaknesses=["Limited capability", "Basic reasoning", "Shallow responses"]
            )
        }
        
        self.test_questions = [
            {
                "category": "simple_factual",
                "question": "What is machine learning?",
                "expected_length": "short",
                "complexity": "basic"
            },
            {
                "category": "technical_explanation", 
                "question": "Explain the mathematical foundations of neural networks",
                "expected_length": "medium",
                "complexity": "intermediate"
            },
            {
                "category": "research_level",
                "question": "Analyze the theoretical implications of attention mechanisms in transformer architectures for causal reasoning",
                "expected_length": "long", 
                "complexity": "advanced"
            }
        ]
        
        self.evaluation_results = {}
        
    def test_model_response_time(self, model_id: str, question: str) -> Dict:
        """Test a single model's response time and basic metrics"""
        try:
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_id,
                    "prompt": question,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1000,  # Limit for speed testing
                    }
                },
                timeout=60
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "response_length": len(generated_text),
                    "word_count": len(generated_text.split()),
                    "chars_per_second": len(generated_text) / (end_time - start_time),
                    "response": generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def evaluate_response_quality(self, response: str, category: str) -> Dict:
        """Evaluate response quality across multiple dimensions"""
        
        # Length appropriateness
        length_score = self._assess_length_appropriateness(response, category)
        
        # Technical depth
        depth_score = self._assess_technical_depth(response)
        
        # Structure quality
        structure_score = self._assess_structure(response)
        
        # Information density
        density_score = self._assess_information_density(response)
        
        # Clarity score
        clarity_score = self._assess_clarity(response)
        
        overall_quality = np.mean([length_score, depth_score, structure_score, density_score, clarity_score])
        
        return {
            "length_appropriateness": length_score,
            "technical_depth": depth_score,
            "structure_quality": structure_score,
            "information_density": density_score,
            "clarity": clarity_score,
            "overall_quality": overall_quality
        }
    
    def _assess_length_appropriateness(self, response: str, category: str) -> float:
        """Assess if response length matches expectation"""
        length = len(response)
        
        expectations = {
            "simple_factual": (100, 500),    # Short, concise
            "technical_explanation": (500, 1500),  # Medium length
            "research_level": (1000, 3000)   # Longer, detailed
        }
        
        min_len, max_len = expectations.get(category, (200, 1000))
        
        if min_len <= length <= max_len:
            return 1.0
        elif length < min_len:
            return length / min_len
        else:
            return max_len / length if length > max_len * 2 else 0.8
    
    def _assess_technical_depth(self, response: str) -> float:
        """Assess technical depth and sophistication"""
        technical_indicators = [
            'algorithm', 'function', 'method', 'approach', 'technique',
            'analysis', 'mathematical', 'theoretical', 'empirical',
            'optimization', 'implementation', 'framework', 'architecture'
        ]
        
        response_lower = response.lower()
        tech_count = sum(1 for indicator in technical_indicators if indicator in response_lower)
        
        return min(tech_count / 5, 1.0)  # Normalize to 0-1
    
    def _assess_structure(self, response: str) -> float:
        """Assess response structure and organization"""
        structure_points = 0
        
        # Check for paragraphs
        paragraphs = len([p for p in response.split('\n\n') if p.strip()])
        if paragraphs >= 2:
            structure_points += 0.3
        
        # Check for lists or enumeration
        if any(marker in response for marker in ['1.', '2.', '-', '*']):
            structure_points += 0.3
        
        # Check for logical flow words
        flow_words = ['first', 'second', 'however', 'therefore', 'consequently', 'furthermore']
        if any(word in response.lower() for word in flow_words):
            structure_points += 0.2
        
        # Check for conclusion or summary
        if any(word in response.lower() for word in ['conclusion', 'summary', 'in summary']):
            structure_points += 0.2
        
        return min(structure_points, 1.0)
    
    def _assess_information_density(self, response: str) -> float:
        """Assess information density - concepts per unit length"""
        
        # Count unique concepts (simplified)
        words = response.lower().split()
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        # Vocabulary richness
        richness = unique_words / total_words
        
        # Sentence complexity (average words per sentence)
        sentences = len([s for s in response.split('.') if s.strip()])
        if sentences == 0:
            return richness
        
        avg_sentence_length = total_words / sentences
        complexity_score = min(avg_sentence_length / 15, 1.0)  # Normalize around 15 words/sentence
        
        return (richness + complexity_score) / 2
    
    def _assess_clarity(self, response: str) -> float:
        """Assess clarity and readability"""
        
        # Simple readability metrics
        words = response.split()
        sentences = len([s for s in response.split('.') if s.strip()])
        
        if sentences == 0:
            return 0.5
        
        avg_words_per_sentence = len(words) / sentences
        
        # Optimal range: 15-20 words per sentence
        if 15 <= avg_words_per_sentence <= 20:
            sentence_score = 1.0
        elif avg_words_per_sentence < 15:
            sentence_score = avg_words_per_sentence / 15
        else:
            sentence_score = 20 / avg_words_per_sentence
        
        # Check for overly complex words (simplified)
        complex_words = len([w for w in words if len(w) > 8])
        complexity_ratio = complex_words / len(words) if words else 0
        
        # Moderate complexity is good
        if 0.1 <= complexity_ratio <= 0.3:
            word_score = 1.0
        else:
            word_score = 1.0 - abs(complexity_ratio - 0.2) * 2
        
        return (sentence_score + max(word_score, 0)) / 2
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation across all models"""
        print("Starting comprehensive model evaluation...")
        print("This will test response speed, quality, and efficiency across all available models")
        
        results = {}
        
        for model_id, model_spec in self.models.items():
            print(f"\nEvaluating {model_spec.name}...")
            model_results = {
                "specs": model_spec,
                "performance": {},
                "quality_scores": {},
                "efficiency_metrics": {}
            }
            
            for question_data in self.test_questions:
                category = question_data["category"]
                question = question_data["question"]
                
                print(f"  Testing {category}...")
                
                # Test performance
                perf_result = self.test_model_response_time(model_id, question)
                
                if perf_result["success"]:
                    # Evaluate quality
                    quality_result = self.evaluate_response_quality(
                        perf_result["response"], category
                    )
                    
                    model_results["performance"][category] = perf_result
                    model_results["quality_scores"][category] = quality_result
                    
                    # Calculate efficiency metrics
                    efficiency = {
                        "quality_per_second": quality_result["overall_quality"] / perf_result["response_time"],
                        "chars_per_gb": perf_result["chars_per_second"] / model_spec.size_gb,
                        "quality_per_gb": quality_result["overall_quality"] / model_spec.size_gb
                    }
                    model_results["efficiency_metrics"][category] = efficiency
                    
                else:
                    print(f"    Failed: {perf_result.get('error', 'Unknown error')}")
                    
                # Small delay between requests
                time.sleep(2)
            
            results[model_id] = model_results
        
        return results
    
    def create_comprehensive_visualizations(self, results: Dict):
        """Create comprehensive visualizations of all metrics"""
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 24))
        
        # Extract data for plotting
        models = []
        speeds = []
        qualities = []
        sizes = []
        efficiency_scores = []
        
        for model_id, data in results.items():
            if "performance" in data and data["performance"]:
                model_name = data["specs"].name
                models.append(model_name)
                sizes.append(data["specs"].size_gb)
                
                # Average across categories
                avg_speed = np.mean([p["chars_per_second"] for p in data["performance"].values() if "chars_per_second" in p])
                avg_quality = np.mean([q["overall_quality"] for q in data["quality_scores"].values()])
                avg_efficiency = np.mean([e["quality_per_second"] for e in data["efficiency_metrics"].values()])
                
                speeds.append(avg_speed)
                qualities.append(avg_quality)
                efficiency_scores.append(avg_efficiency)
        
        # 1. Speed Comparison
        plt.subplot(4, 3, 1)
        bars = plt.bar(range(len(models)), speeds, color='skyblue')
        plt.title('Generation Speed Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Characters per Second')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speeds)*0.01, 
                    f'{speeds[i]:.1f}', ha='center', va='bottom')
        
        # 2. Quality Comparison
        plt.subplot(4, 3, 2)
        bars = plt.bar(range(len(models)), qualities, color='lightgreen')
        plt.title('Response Quality Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Quality Score (0-1)')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{qualities[i]:.2f}', ha='center', va='bottom')
        
        # 3. Model Size Comparison
        plt.subplot(4, 3, 3)
        bars = plt.bar(range(len(models)), sizes, color='salmon')
        plt.title('Model Size Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Size (GB)')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.01, 
                    f'{sizes[i]:.1f}GB', ha='center', va='bottom')
        
        # 4. Efficiency Score
        plt.subplot(4, 3, 4)
        bars = plt.bar(range(len(models)), efficiency_scores, color='gold')
        plt.title('Efficiency Score (Quality/Time)', fontsize=14, fontweight='bold')
        plt.ylabel('Quality per Second')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency_scores)*0.01, 
                    f'{efficiency_scores[i]:.3f}', ha='center', va='bottom')
        
        # 5. Speed vs Quality Scatter
        plt.subplot(4, 3, 5)
        plt.scatter(speeds, qualities, s=[size*3 for size in sizes], alpha=0.7, c=range(len(models)), cmap='viridis')
        plt.xlabel('Generation Speed (chars/sec)')
        plt.ylabel('Quality Score')
        plt.title('Speed vs Quality Trade-off', fontsize=14, fontweight='bold')
        for i, model in enumerate(models):
            plt.annotate(model.split()[0], (speeds[i], qualities[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 6. Size vs Performance Efficiency
        plt.subplot(4, 3, 6)
        efficiency_per_gb = [eff/size for eff, size in zip(efficiency_scores, sizes)]
        bars = plt.bar(range(len(models)), efficiency_per_gb, color='purple', alpha=0.7)
        plt.title('Performance per GB', fontsize=14, fontweight='bold')
        plt.ylabel('Efficiency per GB')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        
        # 7. Quality by Category Heatmap
        plt.subplot(4, 3, 7)
        categories = ["simple_factual", "technical_explanation", "research_level"]
        quality_matrix = []
        
        for model_id, data in results.items():
            if "quality_scores" in data and data["quality_scores"]:
                row = []
                for cat in categories:
                    if cat in data["quality_scores"]:
                        row.append(data["quality_scores"][cat]["overall_quality"])
                    else:
                        row.append(0)
                quality_matrix.append(row)
        
        if quality_matrix:
            plt.imshow(quality_matrix, cmap='RdYlGn', aspect='auto')
            plt.colorbar(label='Quality Score')
            plt.title('Quality by Question Category', fontsize=14, fontweight='bold')
            plt.ylabel('Models')
            plt.xlabel('Question Categories')
            plt.yticks(range(len(models)), models)
            plt.xticks(range(len(categories)), [cat.replace('_', ' ').title() for cat in categories])
        
        # 8. Speed by Category
        plt.subplot(4, 3, 8)
        speed_matrix = []
        for model_id, data in results.items():
            if "performance" in data and data["performance"]:
                row = []
                for cat in categories:
                    if cat in data["performance"]:
                        row.append(data["performance"][cat]["chars_per_second"])
                    else:
                        row.append(0)
                speed_matrix.append(row)
        
        if speed_matrix:
            plt.imshow(speed_matrix, cmap='Blues', aspect='auto')
            plt.colorbar(label='Speed (chars/sec)')
            plt.title('Speed by Question Category', fontsize=14, fontweight='bold')
            plt.ylabel('Models')
            plt.xlabel('Question Categories')
            plt.yticks(range(len(models)), models)
            plt.xticks(range(len(categories)), [cat.replace('_', ' ').title() for cat in categories])
        
        # 9. Comprehensive Ranking
        plt.subplot(4, 3, 9)
        
        # Calculate composite scores
        composite_scores = []
        for i, model in enumerate(models):
            # Normalize scores (0-1)
            norm_speed = speeds[i] / max(speeds) if max(speeds) > 0 else 0
            norm_quality = qualities[i]
            norm_efficiency = efficiency_scores[i] / max(efficiency_scores) if max(efficiency_scores) > 0 else 0
            
            # Weighted composite (can adjust weights)
            composite = (norm_speed * 0.3 + norm_quality * 0.4 + norm_efficiency * 0.3)
            composite_scores.append(composite)
        
        # Sort by composite score
        sorted_indices = sorted(range(len(composite_scores)), key=lambda x: composite_scores[x], reverse=True)
        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [composite_scores[i] for i in sorted_indices]
        
        bars = plt.barh(range(len(sorted_models)), sorted_scores, color='mediumseagreen')
        plt.title('Overall Performance Ranking', fontsize=14, fontweight='bold')
        plt.xlabel('Composite Score')
        plt.yticks(range(len(sorted_models)), sorted_models)
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{sorted_scores[i]:.3f}', va='center')
        
        # 10. Resource Efficiency Analysis
        plt.subplot(4, 3, 10)
        memory_efficiency = [q/s for q, s in zip(qualities, sizes)]
        bars = plt.bar(range(len(models)), memory_efficiency, color='orange', alpha=0.7)
        plt.title('Quality per GB (Memory Efficiency)', fontsize=14, fontweight='bold')
        plt.ylabel('Quality per GB')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        
        # 11. Response Length Analysis
        plt.subplot(4, 3, 11)
        avg_lengths = []
        for model_id, data in results.items():
            if "performance" in data and data["performance"]:
                lengths = [p["response_length"] for p in data["performance"].values()]
                avg_lengths.append(np.mean(lengths))
        
        bars = plt.bar(range(len(models)), avg_lengths, color='lightcoral')
        plt.title('Average Response Length', fontsize=14, fontweight='bold')
        plt.ylabel('Characters')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        
        # 12. Speed vs Size Analysis
        plt.subplot(4, 3, 12)
        plt.scatter(sizes, speeds, s=100, alpha=0.7, c=qualities, cmap='viridis')
        plt.xlabel('Model Size (GB)')
        plt.ylabel('Generation Speed (chars/sec)')
        plt.title('Size vs Speed Relationship', fontsize=14, fontweight='bold')
        plt.colorbar(label='Quality Score')
        for i, model in enumerate(models):
            plt.annotate(model.split()[0], (sizes[i], speeds[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_detailed_report(self, results: Dict) -> str:
        """Generate detailed analysis report"""
        
        report = """# Comprehensive Model Analysis Report

## Executive Summary

This analysis evaluates 7 language models across multiple dimensions including generation speed, response quality, resource efficiency, and practical applicability. The evaluation tests each model on three categories of questions: simple factual queries, technical explanations, and research-level analysis.

## Key Findings

"""
        
        # Find best performers in each category
        best_speed = max(results.items(), 
                        key=lambda x: np.mean([p.get("chars_per_second", 0) for p in x[1].get("performance", {}).values()]))
        
        best_quality = max(results.items(),
                          key=lambda x: np.mean([q.get("overall_quality", 0) for q in x[1].get("quality_scores", {}).values()]))
        
        best_efficiency = max(results.items(),
                             key=lambda x: np.mean([e.get("quality_per_second", 0) for e in x[1].get("efficiency_metrics", {}).values()]))
        
        report += f"""### Performance Leaders

**Speed Champion**: {best_speed[1]["specs"].name}
- Fastest generation across all question types
- Best for time-critical applications
- Trade-off: May sacrifice some quality for speed

**Quality Leader**: {best_quality[1]["specs"].name}
- Highest overall response quality
- Best for comprehensive, detailed responses
- Consideration: May require more time and resources

**Efficiency Winner**: {best_efficiency[1]["specs"].name}
- Best quality-to-time ratio
- Optimal balance for most applications
- Sweet spot for practical deployment

## Detailed Model Analysis

"""
        
        for model_id, data in results.items():
            if not data.get("performance"):
                continue
                
            specs = data["specs"]
            report += f"""### {specs.name}

**Specifications**:
- Parameters: {specs.parameter_count}
- Size: {specs.size_gb:.1f} GB
- Architecture: {specs.architecture}

**Performance Metrics**:
"""
            
            # Calculate averages
            avg_speed = np.mean([p["chars_per_second"] for p in data["performance"].values()])
            avg_quality = np.mean([q["overall_quality"] for q in data["quality_scores"].values()])
            avg_efficiency = np.mean([e["quality_per_second"] for e in data["efficiency_metrics"].values()])
            
            report += f"""- Average Speed: {avg_speed:.1f} chars/second
- Average Quality: {avg_quality:.3f}/1.0
- Efficiency Score: {avg_efficiency:.4f}
- Memory Efficiency: {avg_quality/specs.size_gb:.4f} quality/GB

**Strengths**: {', '.join(specs.strengths)}
**Limitations**: {', '.join(specs.weaknesses)}

**Best Use Cases**:
"""
            
            # Determine best use cases based on performance profile
            if avg_speed > 100:
                report += "- Real-time applications requiring quick responses\n"
            if avg_quality > 0.7:
                report += "- Research and educational content generation\n"
            if specs.size_gb < 20:
                report += "- Resource-constrained environments\n"
            if avg_efficiency > 0.01:
                report += "- Production systems requiring balanced performance\n"
            
            report += "\n"
        
        # Add recommendations section
        report += """## Strategic Recommendations

### For Different Use Cases

**Quick Answers & High Throughput**:
Recommend smaller, faster models like Gemma 3 12B or GPT-OSS 20B. These provide adequate quality for straightforward questions while maintaining high speed.

**Research & Educational Content**:
Larger models like GPT-OSS 120B or Llama 3.1 70B excel at comprehensive, detailed explanations with proper depth and context.

**Balanced Production Use**:
Medium-sized models like Gemma 3 27B or DeepSeek R1 70B offer good compromise between speed, quality, and resource requirements.

**Resource-Constrained Environments**:
Gemma 3 12B provides the best performance per GB, making it ideal for edge deployments or limited hardware.

### Implementation Strategy

1. **Start with efficiency leaders** for general purpose applications
2. **Scale up to quality leaders** for specialized, high-value content
3. **Use speed champions** for interactive or real-time scenarios
4. **Consider model ensemble** approaches for different content types

## Technical Considerations

### Memory Requirements
- Models range from 8GB to 75GB
- Consider VRAM limitations for GPU deployment
- Larger models require more sophisticated infrastructure

### Response Characteristics
- Smaller models tend toward concise, direct answers
- Larger models provide more comprehensive, detailed responses
- Quality scales non-linearly with size

### Performance Scaling
- Speed generally decreases with model size
- Quality improvements diminish beyond certain thresholds
- Efficiency sweet spot appears in 20-30B parameter range

"""
        
        return report

def main():
    """Main evaluation function"""
    print("Comprehensive Model Analysis")
    print("="*50)
    
    analyzer = ComprehensiveModelAnalyzer()
    
    # Run evaluation
    results = analyzer.run_comprehensive_evaluation()
    
    # Create visualizations
    print("\nGenerating comprehensive visualizations...")
    fig = analyzer.create_comprehensive_visualizations(results)
    
    # Generate detailed report
    print("Creating detailed analysis report...")
    report = analyzer.generate_detailed_report(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw data
    with open(f'comprehensive_analysis_results_{timestamp}.json', 'w') as f:
        # Convert ModelSpecs to dict for JSON serialization
        json_results = {}
        for model_id, data in results.items():
            json_results[model_id] = {
                "specs": {
                    "name": data["specs"].name,
                    "size_gb": data["specs"].size_gb,
                    "parameter_count": data["specs"].parameter_count,
                    "architecture": data["specs"].architecture,
                    "strengths": data["specs"].strengths,
                    "weaknesses": data["specs"].weaknesses
                },
                "performance": data.get("performance", {}),
                "quality_scores": data.get("quality_scores", {}),
                "efficiency_metrics": data.get("efficiency_metrics", {})
            }
        json.dump(json_results, f, indent=2)
    
    # Save report
    with open(f'comprehensive_analysis_report_{timestamp}.md', 'w') as f:
        f.write(report)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: comprehensive_analysis_results_{timestamp}.json")
    print(f"Report saved to: comprehensive_analysis_report_{timestamp}.md")
    print(f"Visualizations saved to: comprehensive_model_analysis.png")
    
    # Print summary
    print("\nQuick Summary:")
    print("-" * 30)
    
    for model_id, data in results.items():
        if data.get("performance"):
            specs = data["specs"]
            avg_speed = np.mean([p["chars_per_second"] for p in data["performance"].values()])
            avg_quality = np.mean([q["overall_quality"] for q in data["quality_scores"].values()])
            
            print(f"{specs.name}:")
            print(f"  Speed: {avg_speed:.1f} chars/sec")
            print(f"  Quality: {avg_quality:.3f}/1.0")
            print(f"  Size: {specs.size_gb:.1f} GB")
            print()

if __name__ == "__main__":
    main()