"""
Evaluation Framework for Educational LLM
Compare base model vs fine-tuned model performance
"""
import json
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from config import config
import logging

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question: str
    base_response: str
    finetuned_response: str = ""
    domain: str = ""
    pedagogical_structure_score: float = 0.0
    concept_clarity_score: float = 0.0
    example_quality_score: float = 0.0
    connection_building_score: float = 0.0
    overall_score: float = 0.0
    human_feedback: str = ""
    timestamp: float = 0.0

class EducationalEvaluator:
    """Framework for evaluating educational effectiveness"""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
        self.evaluation_criteria = {
            "pedagogical_structure": {
                "description": "Does the response follow a clear educational structure?",
                "scoring": {
                    1: "No structure, scattered information",
                    2: "Minimal structure, hard to follow",
                    3: "Some structure, could be clearer",
                    4: "Good structure, easy to follow",
                    5: "Excellent structure, perfect pedagogical flow"
                }
            },
            "concept_clarity": {
                "description": "How clearly are concepts explained?",
                "scoring": {
                    1: "Very confusing, incorrect explanations",
                    2: "Somewhat confusing, missing key points",
                    3: "Adequate clarity, some unclear parts",
                    4: "Clear explanations, minor issues",
                    5: "Crystal clear, perfect explanations"
                }
            },
            "example_quality": {
                "description": "Quality and relevance of examples provided",
                "scoring": {
                    1: "No examples or poor examples",
                    2: "Limited examples, not very helpful",
                    3: "Adequate examples, somewhat helpful",
                    4: "Good examples, helpful for understanding",
                    5: "Excellent examples, greatly enhance learning"
                }
            },
            "connection_building": {
                "description": "Does it connect to other concepts and broader context?",
                "scoring": {
                    1: "No connections made",
                    2: "Few connections, mostly isolated",
                    3: "Some connections, could be better",
                    4: "Good connections to related concepts",
                    5: "Excellent connections, builds comprehensive understanding"
                }
            }
        }
    
    def create_evaluation_dataset(self, domains: List[str] = None) -> List[Dict]:
        """Create evaluation dataset template for human curation"""
        domains = domains or config.evaluation.domains
        
        # Template questions for each domain
        question_templates = {
            "linear_algebra": [
                "Explain eigenvalues and eigenvectors",
                "What is the difference between rank and nullity of a matrix?",
                "How do you compute the determinant of a matrix and why is it important?",
                "Explain principal component analysis (PCA) from a linear algebra perspective",
                "What are orthogonal matrices and why are they useful?",
                "Explain the singular value decomposition (SVD)",
                "What is the difference between row space and column space?",
                "How do linear transformations relate to matrices?",
                "Explain matrix diagonalization and when it's possible",
                "What are the geometric interpretations of matrix operations?"
            ],
            "probability": [
                "Explain the difference between Bayesian and frequentist probability",
                "What is Bayes' theorem and why is it important?",
                "Explain the Central Limit Theorem and its implications",
                "What are probability distributions and how do you choose the right one?",
                "Explain conditional probability with a practical example",
                "What is the law of large numbers?",
                "How do you interpret confidence intervals?",
                "Explain maximum likelihood estimation",
                "What is the difference between correlation and causation?",
                "How do you work with joint and marginal probability distributions?"
            ],
            "deep_learning": [
                "How do neural networks learn through backpropagation?",
                "Explain the transformer architecture and attention mechanism",
                "What is the vanishing gradient problem and how do you solve it?",
                "How do convolutional neural networks work?",
                "Explain regularization techniques in deep learning",
                "What are activation functions and how do you choose them?",
                "How does batch normalization help neural network training?",
                "Explain the difference between overfitting and underfitting",
                "What are generative adversarial networks (GANs)?",
                "How do you interpret and visualize what neural networks learn?"
            ],
            "world_models": [
                "What are world models in AI and how do they work?",
                "How do world models enable model-based reinforcement learning?",
                "Explain the difference between model-free and model-based learning",
                "How do you handle uncertainty in world models?",
                "What are the challenges in learning accurate world models?",
                "How do world models relate to human cognition and mental models?",
                "Explain forward models vs inverse models",
                "How do you evaluate the quality of a world model?",
                "What role do world models play in planning and decision making?",
                "How do world models help with sample efficiency in learning?"
            ]
        }
        
        dataset = []
        for domain, questions in question_templates.items():
            if domain in domains:
                for question in questions:
                    dataset.append({
                        "question": question,
                        "domain": domain,
                        "base_response": "",  # To be filled by inference
                        "finetuned_response": "",  # To be filled after training
                        "evaluation_notes": ""
                    })
        
        return dataset
    
    def save_evaluation_dataset(self, dataset: List[Dict], filename: str = "evaluation_dataset.json"):
        """Save evaluation dataset template"""
        filepath = Path(config.data_dir) / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation dataset template saved to {filepath}")
        return filepath
    
    def load_responses(self, base_responses_file: str, finetuned_responses_file: str = None) -> List[EvaluationResult]:
        """Load responses for evaluation"""
        # Load base responses
        with open(base_responses_file, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # Load fine-tuned responses if available
        finetuned_data = []
        if finetuned_responses_file:
            try:
                with open(finetuned_responses_file, 'r', encoding='utf-8') as f:
                    finetuned_data = json.load(f)
            except FileNotFoundError:
                logger.warning(f"Fine-tuned responses file not found: {finetuned_responses_file}")
        
        # Create evaluation results
        results = []
        for i, base_item in enumerate(base_data):
            finetuned_response = ""
            if i < len(finetuned_data):
                finetuned_response = finetuned_data[i].get('response', '')
            
            result = EvaluationResult(
                question=base_item['question'],
                base_response=base_item['response'],
                finetuned_response=finetuned_response,
                timestamp=time.time()
            )
            results.append(result)
        
        return results
    
    def analyze_response_structure(self, response: str) -> Dict[str, float]:
        """Automatically analyze response structure"""
        analysis = {
            "has_sections": 0.0,
            "has_examples": 0.0,
            "has_definitions": 0.0,
            "has_applications": 0.0,
            "length_score": 0.0
        }
        
        response_lower = response.lower()
        
        # Check for section headers
        section_indicators = ['#', '##', 'definition', 'example', 'intuition', 'application', 'connection']
        if any(indicator in response_lower for indicator in section_indicators):
            analysis["has_sections"] = 1.0
        
        # Check for examples
        example_indicators = ['example', 'for instance', 'consider', 'suppose', 'let us', 'imagine']
        if any(indicator in response_lower for indicator in example_indicators):
            analysis["has_examples"] = 1.0
        
        # Check for formal definitions
        definition_indicators = ['definition', 'formally', 'mathematically', 'defined as']
        if any(indicator in response_lower for indicator in definition_indicators):
            analysis["has_definitions"] = 1.0
        
        # Check for applications
        application_indicators = ['application', 'used in', 'useful for', 'important because', 'why this matters']
        if any(indicator in response_lower for indicator in application_indicators):
            analysis["has_applications"] = 1.0
        
        # Length score (educational responses should be substantial but not too long)
        word_count = len(response.split())
        if 100 <= word_count <= 500:
            analysis["length_score"] = 1.0
        elif 50 <= word_count < 100 or 500 < word_count <= 800:
            analysis["length_score"] = 0.7
        elif word_count < 50 or word_count > 800:
            analysis["length_score"] = 0.3
        
        return analysis
    
    def generate_comparison_report(self, results: List[EvaluationResult], output_file: str = "evaluation_report.md"):
        """Generate comparison report"""
        
        # Calculate statistics
        base_scores = [r.overall_score for r in results if r.overall_score > 0]
        finetuned_scores = [r.overall_score for r in results if r.overall_score > 0 and r.finetuned_response]
        
        report_lines = [
            "# Educational LLM Evaluation Report",
            "",
            f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Questions Evaluated:** {len(results)}",
            "",
            "## Summary Statistics",
            "",
        ]
        
        if base_scores:
            report_lines.extend([
                f"**Base Model Average Score:** {np.mean(base_scores):.2f}/5.0",
                f"**Base Model Score Range:** {min(base_scores):.2f} - {max(base_scores):.2f}",
                "",
            ])
        
        if finetuned_scores:
            improvement = np.mean(finetuned_scores) - np.mean(base_scores) if base_scores else 0
            report_lines.extend([
                f"**Fine-tuned Model Average Score:** {np.mean(finetuned_scores):.2f}/5.0",
                f"**Improvement:** {improvement:+.2f} points",
                "",
            ])
        
        # Add detailed results
        report_lines.extend([
            "## Detailed Results",
            "",
            "| Question | Domain | Base Score | Fine-tuned Score | Improvement |",
            "|----------|--------|------------|------------------|-------------|",
        ])
        
        for result in results:
            ft_score = result.overall_score if result.finetuned_response else "N/A"
            improvement = f"{result.overall_score - result.overall_score:.2f}" if result.finetuned_response else "N/A"
            
            report_lines.append(
                f"| {result.question[:50]}... | {result.domain} | {result.overall_score:.1f} | {ft_score} | {improvement} |"
            )
        
        # Save report
        report_path = Path(config.output_dir) / output_file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path

def main():
    """Generate evaluation dataset"""
    evaluator = EducationalEvaluator()
    
    # Create evaluation dataset
    dataset = evaluator.create_evaluation_dataset()
    
    # Save dataset
    filepath = evaluator.save_evaluation_dataset(dataset)
    
    print(f"\nEvaluation dataset created: {filepath}")
    print(f"Total questions: {len(dataset)}")
    print(f"Domains: {list(set(item['domain'] for item in dataset))}")
    
    # Print sample questions
    print("\nSample questions:")
    for domain in ["linear_algebra", "probability"]:
        domain_questions = [item for item in dataset if item['domain'] == domain][:2]
        print(f"\n{domain.replace('_', ' ').title()}:")
        for item in domain_questions:
            print(f"  - {item['question']}")

if __name__ == "__main__":
    main()
