"""
Data utilities for Educational LLM project
Handles data loading, preprocessing, and formatting
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from config import config
import logging

logger = logging.getLogger(__name__)

@dataclass
class EducationalExample:
    """Structure for educational training examples"""
    question: str
    answer: str
    domain: str
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    concepts: List[str] = None
    has_math: bool = False
    has_code: bool = False
    
    def __post_init__(self):
        if self.concepts is None:
            self.concepts = []

class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self):
        self.data_dir = Path(config.data_dir)
        self.output_dir = Path(config.output_dir)
        
        # Educational formatting templates
        self.educational_template = """# {concept}

## Intuitive Understanding
{intuition}

## Mathematical Definition
{definition}

## Step-by-step Example
{example}

## Why This Matters
{applications}

## Connection to Other Concepts
{connections}"""
    
    def load_evaluation_responses(self, filename: str) -> List[Dict]:
        """Load evaluation responses from JSON"""
        filepath = self.data_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def format_for_training(self, examples: List[EducationalExample]) -> List[Dict]:
        """Format examples for LoRA training"""
        formatted_examples = []
        
        for example in examples:
            # Create conversational format for training
            prompt = f"""You are an expert educational tutor. When explaining concepts, always use this structure:

# [Concept Name]

## Intuitive Understanding
Start with an intuitive explanation or analogy that helps build understanding.

## Mathematical Definition
Provide the formal mathematical definition with proper notation.

## Step-by-step Example
Walk through a concrete example with clear steps.

## Why This Matters
Explain real-world applications and importance.

## Connection to Other Concepts
Link to related mathematical concepts and broader context.

Question: {example.question}"""

            formatted_examples.append({
                "input": prompt,
                "output": example.answer,
                "domain": example.domain,
                "difficulty": example.difficulty,
                "concepts": example.concepts
            })
        
        return formatted_examples
    
    def extract_structure_quality(self, response: str) -> Dict[str, bool]:
        """Extract structural quality indicators from response"""
        structure_check = {
            "has_title": bool(re.search(r'^#\s+', response, re.MULTILINE)),
            "has_subsections": bool(re.search(r'^##\s+', response, re.MULTILINE)),
            "has_intuition_section": bool(re.search(r'intuiti(on|ve)', response, re.IGNORECASE)),
            "has_definition_section": bool(re.search(r'definition|formally', response, re.IGNORECASE)),
            "has_example_section": bool(re.search(r'example|step-by-step', response, re.IGNORECASE)),
            "has_application_section": bool(re.search(r'application|why.*matter|important', response, re.IGNORECASE)),
            "has_connection_section": bool(re.search(r'connection|related.*concept', response, re.IGNORECASE)),
            "has_math_notation": bool(re.search(r'\\[a-zA-Z]|\$.*\$|\\frac|\\sum|\\int', response)),
            "proper_length": 200 <= len(response.split()) <= 800
        }
        
        return structure_check
    
    def validate_educational_format(self, response: str) -> Tuple[bool, List[str]]:
        """Validate if response follows educational format"""
        structure = self.extract_structure_quality(response)
        issues = []
        
        # Check required components
        required_components = [
            ("has_title", "Missing main title/heading"),
            ("has_subsections", "Missing subsection structure"),
            ("has_definition_section", "Missing formal definition"),
            ("has_example_section", "Missing concrete examples"),
        ]
        
        for component, error_msg in required_components:
            if not structure[component]:
                issues.append(error_msg)
        
        # Check recommended components
        recommended_components = [
            ("has_intuition_section", "Consider adding intuitive explanation"),
            ("has_application_section", "Consider adding applications/importance"),
            ("has_connection_section", "Consider adding connections to other concepts"),
        ]
        
        for component, warning_msg in recommended_components:
            if not structure[component]:
                issues.append(f"Recommendation: {warning_msg}")
        
        is_valid = len([issue for issue in issues if not issue.startswith("Recommendation:")]) == 0
        
        return is_valid, issues
    
    def create_training_dataset(self, 
                              base_responses_file: str,
                              improved_responses_file: str,
                              output_file: str = "training_dataset.json") -> str:
        """Create training dataset from improved responses"""
        
        # Load base responses
        base_responses = self.load_evaluation_responses(base_responses_file)
        
        # Load improved responses (manually curated)
        try:
            with open(self.data_dir / improved_responses_file, 'r', encoding='utf-8') as f:
                improved_responses = json.load(f)
        except FileNotFoundError:
            logger.error(f"Improved responses file not found: {improved_responses_file}")
            return None
        
        # Create training examples
        training_examples = []
        for base, improved in zip(base_responses, improved_responses):
            if improved.get('improved_answer'):
                example = EducationalExample(
                    question=base['question'],
                    answer=improved['improved_answer'],
                    domain=improved.get('domain', 'general'),
                    difficulty=improved.get('difficulty', 'intermediate'),
                    concepts=improved.get('concepts', [])
                )
                training_examples.append(example)
        
        # Format for training
        formatted_data = self.format_for_training(training_examples)
        
        # Save training dataset
        output_path = self.data_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training dataset created: {output_path}")
        logger.info(f"Total training examples: {len(formatted_data)}")
        
        return str(output_path)
    
    def analyze_dataset_quality(self, dataset_file: str) -> Dict[str, Any]:
        """Analyze quality metrics of training dataset"""
        with open(self.data_dir / dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        analysis = {
            "total_examples": len(dataset),
            "domains": {},
            "difficulty_distribution": {},
            "avg_input_length": 0,
            "avg_output_length": 0,
            "structural_quality": {
                "well_structured": 0,
                "needs_improvement": 0
            }
        }
        
        input_lengths = []
        output_lengths = []
        
        for example in dataset:
            # Domain distribution
            domain = example.get('domain', 'unknown')
            analysis["domains"][domain] = analysis["domains"].get(domain, 0) + 1
            
            # Difficulty distribution
            difficulty = example.get('difficulty', 'unknown')
            analysis["difficulty_distribution"][difficulty] = analysis["difficulty_distribution"].get(difficulty, 0) + 1
            
            # Length analysis
            input_lengths.append(len(example['input'].split()))
            output_lengths.append(len(example['output'].split()))
            
            # Structural quality
            is_valid, _ = self.validate_educational_format(example['output'])
            if is_valid:
                analysis["structural_quality"]["well_structured"] += 1
            else:
                analysis["structural_quality"]["needs_improvement"] += 1
        
        analysis["avg_input_length"] = sum(input_lengths) / len(input_lengths) if input_lengths else 0
        analysis["avg_output_length"] = sum(output_lengths) / len(output_lengths) if output_lengths else 0
        
        return analysis
    
    def create_sample_improved_template(self, base_responses_file: str) -> str:
        """Create template for manual improvement of responses"""
        base_responses = self.load_evaluation_responses(base_responses_file)
        
        template = []
        for i, response in enumerate(base_responses[:10]):  # First 10 for template
            template.append({
                "id": i,
                "question": response['question'],
                "base_answer": response['response'],
                "improved_answer": "",  # To be filled manually
                "domain": "",  # To be assigned
                "difficulty": "intermediate",  # To be assigned
                "concepts": [],  # To be assigned
                "improvement_notes": ""
            })
        
        template_file = self.data_dir / "improvement_template.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Improvement template created: {template_file}")
        return str(template_file)

def main():
    """Test data utilities"""
    processor = DataProcessor()
    
    # Test structure validation
    good_response = """# Eigenvalues and Eigenvectors

## Intuitive Understanding
Think of eigenvalues as special directions in space...

## Mathematical Definition
For a matrix A and vector v: Av = λv

## Step-by-step Example
Let's compute eigenvalues for a 2x2 matrix...

## Why This Matters
Used in PCA, neural networks, and physics simulations.

## Connection to Other Concepts
Related to diagonalization and spectral theory."""
    
    bad_response = "Eigenvalues are numbers associated with matrices. They're computed using the characteristic polynomial."
    
    # Validate responses
    good_valid, good_issues = processor.validate_educational_format(good_response)
    bad_valid, bad_issues = processor.validate_educational_format(bad_response)
    
    print("Good response validation:")
    print(f"Valid: {good_valid}, Issues: {good_issues}")
    
    print("\nBad response validation:")
    print(f"Valid: {bad_valid}, Issues: {bad_issues}")

if __name__ == "__main__":
    main()
