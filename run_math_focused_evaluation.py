"""
Math-Focused Evaluation
Test model's ability to structure and represent mathematical formulas
"""
import json
import logging
from pathlib import Path
from ollama_inference import OllamaEducationalInference
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run 10 math-heavy questions to test formula representation"""
    
    # Carefully selected questions that require heavy mathematical notation
    math_questions = [
        # Linear Algebra (3 questions)
        {
            "question": "Derive the eigenvalue decomposition formula and explain each component",
            "domain": "linear_algebra",
            "focus": "Matrix decomposition formulas, eigenvalues/eigenvectors notation"
        },
        {
            "question": "Explain the Singular Value Decomposition (SVD) with complete mathematical derivation",
            "domain": "linear_algebra", 
            "focus": "SVD formula, matrix notation, rank relationships"
        },
        {
            "question": "Derive the least squares solution formula and show the normal equations",
            "domain": "linear_algebra",
            "focus": "Optimization formulas, matrix calculus, projections"
        },
        
        # Probability (3 questions)
        {
            "question": "Derive Bayes' theorem from first principles and show its applications in conditional probability",
            "domain": "probability",
            "focus": "Conditional probability formulas, probability notation"
        },
        {
            "question": "Explain the Central Limit Theorem with mathematical proof and convergence formulas",
            "domain": "probability",
            "focus": "Limit theorems, statistical convergence, distribution notation"
        },
        {
            "question": "Derive the probability density function of the multivariate normal distribution",
            "domain": "probability",
            "focus": "Complex probability formulas, matrix notation in statistics"
        },
        
        # Information Theory (2 questions)
        {
            "question": "Derive Shannon's entropy formula and explain its relationship to information content",
            "domain": "information_theory",
            "focus": "Logarithmic formulas, entropy notation, information measures"
        },
        {
            "question": "Explain mutual information and derive its mathematical relationship to joint and marginal entropies",
            "domain": "information_theory",
            "focus": "Information theory formulas, conditional entropy relationships"
        },
        
        # Numerical Computation (2 questions)
        {
            "question": "Derive the Newton-Raphson method formula and analyze its convergence properties",
            "domain": "numerical_computation",
            "focus": "Iterative formulas, convergence analysis, function approximation"
        },
        {
            "question": "Explain the mathematical foundations of gradient descent and derive the update rule",
            "domain": "numerical_computation",
            "focus": "Optimization formulas, partial derivatives, learning rate analysis"
        }
    ]
    
    logger.info(f"🧮 Running {len(math_questions)} math-focused questions")
    logger.info(f"🎯 Focus: Mathematical formula representation and structure")
    
    # Show what questions we'll run
    print("\n📝 MATH-FOCUSED QUESTIONS:")
    print("="*80)
    for i, item in enumerate(math_questions, 1):
        print(f"{i:2d}. [{item['domain'].upper()}]")
        print(f"    ❓ {item['question']}")
        print(f"    🎯 Focus: {item['focus']}")
        print("-" * 60)
    print("="*80)
    
    # Initialize Ollama inference
    inference = OllamaEducationalInference()
    
    # Generate responses with focus on mathematical content
    results = []
    for i, item in enumerate(math_questions):
        question = item["question"]
        domain = item["domain"]
        focus = item["focus"]
        
        logger.info(f"📝 Processing question {i+1}/{len(math_questions)}")
        logger.info(f"📖 Domain: {domain}")
        logger.info(f"🎯 Focus: {focus}")
        logger.info(f"❓ Question: {question[:60]}...")
        
        # Generate response with mathematical emphasis
        response = inference.generate_response(question)
        
        results.append({
            "question": question,
            "domain": domain,
            "focus_area": focus,
            "response": response,
            "model": "gemma3:12b",
            "question_number": f"math_focused_{i+1}",
            "evaluation_notes": f"Testing: {focus}"
        })
        
        logger.info(f"✅ Completed {i+1}/{len(math_questions)} questions")
        
        # Brief pause between questions
        import time
        time.sleep(1)
    
    # Save results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(config.data_dir) / f"math_focused_responses_{timestamp}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Math-focused responses saved to {output_path}")
    
    # Analyze formula content
    formula_indicators = [
        "=", "∫", "∑", "∂", "∇", "≈", "≤", "≥", "∞", "π", "α", "β", "θ", "λ", "μ", "σ",
        "matrix", "determinant", "eigenvalue", "derivative", "integral", "logarithm"
    ]
    
    # Generate summary
    successful_responses = [r for r in results if not r["response"].startswith("Error:")]
    
    print("\n" + "="*80)
    print("🧮 MATH-FOCUSED EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Questions: {len(math_questions)}")
    print(f"Successful Responses: {len(successful_responses)}")
    print(f"Success Rate: {len(successful_responses)/len(math_questions)*100:.1f}%")
    
    # Domain breakdown
    print(f"\n📚 Questions by Domain:")
    domain_counts = {}
    for item in math_questions:
        domain = item['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    for domain, count in domain_counts.items():
        print(f"  • {domain.replace('_', ' ').title()}: {count} questions")
    
    # Formula content analysis
    print(f"\n🔢 Mathematical Content Analysis:")
    total_formula_indicators = 0
    for result in successful_responses:
        response = result['response'].lower()
        formula_count = sum(1 for indicator in formula_indicators if indicator in response)
        total_formula_indicators += formula_count
    
    avg_formula_content = total_formula_indicators / len(successful_responses) if successful_responses else 0
    print(f"  • Average mathematical indicators per response: {avg_formula_content:.1f}")
    print(f"  • Total mathematical content detected: {total_formula_indicators}")
    
    # Show sample focuses
    print(f"\n🎯 Focus Areas Tested:")
    for item in math_questions:
        print(f"  • {item['focus']}")
    
    print("="*80)
    print("📊 Ready for mathematical formula analysis!")
    print("🎯 Use format_responses.py to create readable versions")

if __name__ == "__main__":
    main()
