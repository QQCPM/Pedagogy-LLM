"""
LaTeX-Focused Mathematical Evaluation
Test model's ability to use proper LaTeX notation for formulas
"""
import json
import logging
from pathlib import Path
from ollama_inference import OllamaEducationalInference
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaTeXEducationalInference(OllamaEducationalInference):
    """Modified inference class that emphasizes LaTeX formatting"""
    
    def _create_educational_prompt(self, question: str) -> str:
        """Create educational prompt with strong LaTeX emphasis"""
        return f"""You are an expert educational tutor specializing in mathematical content. When explaining concepts, you MUST use proper LaTeX notation for all mathematical expressions. Follow this structure:

# [Concept Name]

## Intuitive Understanding
Start with an intuitive explanation or analogy that helps build understanding.

## Mathematical Definition
Provide the formal mathematical definition using PROPER LATEX NOTATION. All formulas must be in LaTeX format using $ for inline math and $$ for display math.

LATEX FORMATTING REQUIREMENTS:
- Use $...$ for inline mathematical expressions
- Use $$....$$ for displayed equations
- Use proper LaTeX commands: \\frac{{}}{{}} for fractions, \\sum for summation, \\int for integrals
- Use \\begin{{bmatrix}} for matrices, \\begin{{align}} for multi-line equations
- Use subscripts with _ and superscripts with ^
- Use \\mathbb{{}} for special number sets, \\mathcal{{}} for script letters
- Use \\partial for partial derivatives, \\nabla for gradient
- Use \\alpha, \\beta, \\gamma, \\sigma, \\lambda for Greek letters

## Step-by-step Example
Walk through a concrete example with ALL mathematical expressions in proper LaTeX format.

## Why This Matters
Explain real-world applications and importance.

## Connection to Other Concepts
Link to related mathematical concepts using LaTeX notation.

Question: {question}

Answer (remember: ALL MATH MUST BE IN LATEX FORMAT):"""

def main():
    """Test LaTeX formatting with mathematical questions"""
    
    # Select 3 math-heavy questions to test LaTeX formatting
    latex_test_questions = [
        {
            "question": "Derive the gradient descent update rule and explain the role of the learning rate",
            "domain": "optimization",
            "focus": "Partial derivatives, gradient notation, iterative formulas"
        },
        {
            "question": "Explain the multivariate normal distribution probability density function",
            "domain": "probability", 
            "focus": "Matrix notation, exponential functions, determinants"
        },
        {
            "question": "Derive the least squares solution using matrix calculus",
            "domain": "linear_algebra",
            "focus": "Matrix derivatives, optimization, projections"
        }
    ]
    
    logger.info(f"🧮 Running {len(latex_test_questions)} LaTeX-focused questions")
    logger.info(f"🎯 Focus: Proper LaTeX mathematical notation")
    
    # Show what questions we'll run
    print("\n📝 LATEX FORMATTING TEST QUESTIONS:")
    print("="*80)
    for i, item in enumerate(latex_test_questions, 1):
        print(f"{i}. [{item['domain'].upper()}]")
        print(f"   ❓ {item['question']}")
        print(f"   🎯 Focus: {item['focus']}")
        print("-" * 60)
    print("="*80)
    
    # Initialize LaTeX-focused inference
    inference = LaTeXEducationalInference()
    
    # Generate responses with LaTeX emphasis
    results = []
    for i, item in enumerate(latex_test_questions):
        question = item["question"]
        domain = item["domain"]
        focus = item["focus"]
        
        logger.info(f"📝 Processing LaTeX question {i+1}/{len(latex_test_questions)}")
        logger.info(f"📖 Domain: {domain}")
        logger.info(f"🎯 Focus: {focus}")
        logger.info(f"❓ Question: {question[:60]}...")
        
        # Generate response with LaTeX emphasis
        response = inference.generate_response(question)
        
        results.append({
            "question": question,
            "domain": domain,
            "focus_area": focus,
            "response": response,
            "model": "gemma3:12b",
            "question_number": f"latex_test_{i+1}",
            "evaluation_notes": f"Testing LaTeX formatting: {focus}"
        })
        
        logger.info(f"✅ Completed {i+1}/{len(latex_test_questions)} questions")
        
        # Brief pause between questions
        import time
        time.sleep(1)
    
    # Save results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(config.data_dir) / f"latex_test_responses_{timestamp}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 LaTeX test responses saved to {output_path}")
    
    # Analyze LaTeX content
    latex_indicators = [
        "$", "\\frac", "\\sum", "\\int", "\\partial", "\\nabla", "\\alpha", "\\beta", "\\sigma", 
        "\\lambda", "\\begin{", "\\end{", "\\mathbb", "\\mathcal", "^{", "_{", "\\cdot"
    ]
    
    # Generate summary
    successful_responses = [r for r in results if not r["response"].startswith("Error:")]
    
    print("\n" + "="*80)
    print("🧮 LATEX FORMATTING TEST SUMMARY")
    print("="*80)
    print(f"Total Questions: {len(latex_test_questions)}")
    print(f"Successful Responses: {len(successful_responses)}")
    print(f"Success Rate: {len(successful_responses)/len(latex_test_questions)*100:.1f}%")
    
    # LaTeX content analysis
    print(f"\n📐 LaTeX Content Analysis:")
    total_latex_indicators = 0
    for result in successful_responses:
        response = result['response']
        latex_count = sum(1 for indicator in latex_indicators if indicator in response)
        total_latex_indicators += latex_count
        print(f"  • Question {result['question_number']}: {latex_count} LaTeX indicators")
    
    avg_latex_content = total_latex_indicators / len(successful_responses) if successful_responses else 0
    print(f"  • Average LaTeX indicators per response: {avg_latex_content:.1f}")
    print(f"  • Total LaTeX content detected: {total_latex_indicators}")
    
    # Show focus areas tested
    print(f"\n🎯 LaTeX Focus Areas Tested:")
    for item in latex_test_questions:
        print(f"  • {item['focus']}")
    
    print("="*80)
    print("📊 Ready to compare LaTeX vs HTML formatting!")
    print("🎯 Check the responses for proper $ $ notation")

if __name__ == "__main__":
    main()
