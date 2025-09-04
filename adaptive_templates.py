#!/usr/bin/env python3
"""
Adaptive Template System for Educational Responses
Breaks the rigid mold by selecting appropriate response structures based on question analysis
"""
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class QuestionIntent(Enum):
    """Enhanced question intent classification"""
    CONCEPT_EXPLANATION = "concept_explanation"
    COMPARISON = "comparison" 
    PROCESS_HOWTO = "process_howto"
    PROBLEM_SOLVING = "problem_solving"
    ANALYSIS_EVALUATION = "analysis_evaluation"
    LIST_CATEGORIZATION = "list_categorization"
    QUICK_FACTUAL = "quick_factual"
    TROUBLESHOOTING = "troubleshooting"
    CREATIVE_BRAINSTORM = "creative_brainstorm"

class ResponseComplexity(Enum):
    """Response complexity levels"""
    BRIEF = "brief"          # 1-2 paragraphs
    STANDARD = "standard"    # 3-5 sections
    COMPREHENSIVE = "comprehensive"  # 6+ sections with deep detail

@dataclass
class SectionConfig:
    """Configuration for a response section"""
    name: str
    prompt_instruction: str
    is_required: bool = True
    math_heavy: bool = False
    example_heavy: bool = False

class AdaptiveTemplateEngine:
    """Generate adaptive response templates based on question analysis"""
    
    def __init__(self):
        # Enhanced question patterns for better classification
        self.intent_patterns = {
            QuestionIntent.CONCEPT_EXPLANATION: [
                r'\b(?:what is|explain|understand|concept of|theory of|principle of)\b',
                r'\b(?:definition|meaning|purpose)\b'
            ],
            QuestionIntent.COMPARISON: [
                r'\b(?:difference between|compare|vs\.|versus|contrast)\b',
                r'\b(?:which is better|advantages?\s+(?:and|vs)\s+disadvantages?)\b',
                r'\b(?:pros?\s+and\s+cons?|trade-?offs?)\b'
            ],
            QuestionIntent.PROCESS_HOWTO: [
                r'\b(?:how to|how do|steps to|process of|procedure|guide|tutorial)\b',
                r'\b(?:implement|build|create|set up|configure)\b'
            ],
            QuestionIntent.PROBLEM_SOLVING: [
                r'\b(?:solve|calculate|compute|find|derive|prove)\b',
                r'\b(?:solution|answer|result)\b'
            ],
            QuestionIntent.ANALYSIS_EVALUATION: [
                r'\b(?:analyze|evaluate|assess|critique|review)\b',
                r'\b(?:strengths?\s+and\s+weaknesses|pros?\s+and\s+cons?)\b'
            ],
            QuestionIntent.LIST_CATEGORIZATION: [
                r'\b(?:types? of|kinds? of|categories of|examples? of)\b',
                r'\b(?:list|enumerate|name|identify)\b'
            ],
            QuestionIntent.QUICK_FACTUAL: [
                r'\b(?:what|when|where|who|which)\b.*\?$',
                r'\b(?:briefly|quickly|in short|simple answer)\b'
            ],
            QuestionIntent.TROUBLESHOOTING: [
                r'\b(?:why (?:does|is|doesn\'t|isn\'t)|problem with|issue with|error)\b',
                r'\b(?:not working|failing|broken)\b'
            ],
            QuestionIntent.CREATIVE_BRAINSTORM: [
                r'\b(?:ideas? for|suggestions? for|ways to|approaches? to)\b',
                r'\b(?:brainstorm|creative|innovative)\b'
            ]
        }
        
        # Complexity indicators
        self.complexity_patterns = {
            ResponseComplexity.BRIEF: [
                r'\b(?:briefly|quick|short|simple|basic)\b'
            ],
            ResponseComplexity.COMPREHENSIVE: [
                r'\b(?:detailed|comprehensive|thorough|in-depth|advanced|complete)\b'
            ]
        }
        
        # Available sections for dynamic composition
        self.available_sections = {
            'overview': SectionConfig(
                "Overview", 
                "Start with a high-level overview that sets the context"
            ),
            'intuition': SectionConfig(
                "Intuitive Understanding", 
                "Explain the concept intuitively with analogies and mental models"
            ),
            'definition': SectionConfig(
                "Formal Definition", 
                "Provide precise, formal definition with proper terminology",
                math_heavy=True
            ),
            'comparison_table': SectionConfig(
                "Comparison", 
                "Create a clear comparison table or side-by-side analysis"
            ),
            'step_by_step': SectionConfig(
                "Step-by-Step Process", 
                "Break down into numbered, actionable steps",
                example_heavy=True
            ),
            'solution_approach': SectionConfig(
                "Solution Approach", 
                "Show the problem-solving methodology and solution path",
                math_heavy=True
            ),
            'examples': SectionConfig(
                "Concrete Examples", 
                "Provide worked examples with detailed explanations",
                example_heavy=True
            ),
            'pros_cons': SectionConfig(
                "Pros and Cons", 
                "Balanced analysis of advantages and disadvantages"
            ),
            'use_cases': SectionConfig(
                "Use Cases & Applications", 
                "Real-world applications and when to use this"
            ),
            'common_mistakes': SectionConfig(
                "Common Mistakes", 
                "Typical pitfalls and how to avoid them"
            ),
            'connections': SectionConfig(
                "Connections to Other Concepts", 
                "Link to related topics and broader context"
            ),
            'next_steps': SectionConfig(
                "Next Steps", 
                "What to learn or explore next"
            ),
            'troubleshooting': SectionConfig(
                "Troubleshooting", 
                "Common issues and solutions"
            ),
            'variations': SectionConfig(
                "Variations & Alternatives", 
                "Different approaches and when to use each"
            )
        }
        
        # Intent-specific section templates
        self.intent_templates = {
            QuestionIntent.CONCEPT_EXPLANATION: {
                ResponseComplexity.BRIEF: ['overview', 'examples'],
                ResponseComplexity.STANDARD: ['intuition', 'definition', 'examples', 'use_cases'],
                ResponseComplexity.COMPREHENSIVE: ['overview', 'intuition', 'definition', 'examples', 'use_cases', 'connections', 'common_mistakes']
            },
            QuestionIntent.COMPARISON: {
                ResponseComplexity.BRIEF: ['comparison_table'],
                ResponseComplexity.STANDARD: ['overview', 'comparison_table', 'use_cases'],
                ResponseComplexity.COMPREHENSIVE: ['overview', 'comparison_table', 'pros_cons', 'use_cases', 'examples']
            },
            QuestionIntent.PROCESS_HOWTO: {
                ResponseComplexity.BRIEF: ['step_by_step'],
                ResponseComplexity.STANDARD: ['overview', 'step_by_step', 'common_mistakes'],
                ResponseComplexity.COMPREHENSIVE: ['overview', 'step_by_step', 'examples', 'troubleshooting', 'variations']
            },
            QuestionIntent.PROBLEM_SOLVING: {
                ResponseComplexity.BRIEF: ['solution_approach'],
                ResponseComplexity.STANDARD: ['solution_approach', 'examples'],
                ResponseComplexity.COMPREHENSIVE: ['overview', 'solution_approach', 'examples', 'variations', 'connections']
            },
            QuestionIntent.ANALYSIS_EVALUATION: {
                ResponseComplexity.BRIEF: ['pros_cons'],
                ResponseComplexity.STANDARD: ['overview', 'pros_cons', 'use_cases'],
                ResponseComplexity.COMPREHENSIVE: ['overview', 'pros_cons', 'examples', 'use_cases', 'connections']
            },
            QuestionIntent.LIST_CATEGORIZATION: {
                ResponseComplexity.BRIEF: ['overview'],
                ResponseComplexity.STANDARD: ['overview', 'examples'],
                ResponseComplexity.COMPREHENSIVE: ['overview', 'examples', 'use_cases', 'connections']
            },
            QuestionIntent.QUICK_FACTUAL: {
                ResponseComplexity.BRIEF: ['overview'],
                ResponseComplexity.STANDARD: ['overview', 'examples'],
                ResponseComplexity.COMPREHENSIVE: ['overview', 'examples', 'connections']
            },
            QuestionIntent.TROUBLESHOOTING: {
                ResponseComplexity.BRIEF: ['troubleshooting'],
                ResponseComplexity.STANDARD: ['overview', 'troubleshooting', 'common_mistakes'],
                ResponseComplexity.COMPREHENSIVE: ['overview', 'troubleshooting', 'examples', 'variations', 'next_steps']
            },
            QuestionIntent.CREATIVE_BRAINSTORM: {
                ResponseComplexity.BRIEF: ['overview'],
                ResponseComplexity.STANDARD: ['overview', 'examples', 'variations'],
                ResponseComplexity.COMPREHENSIVE: ['overview', 'examples', 'variations', 'pros_cons', 'next_steps']
            }
        }
    
    def analyze_question(self, question: str) -> Tuple[QuestionIntent, ResponseComplexity, Dict[str, bool]]:
        """Enhanced question analysis with intent classification"""
        question_lower = question.lower().strip()
        
        # Classify intent
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, question_lower))
                score += matches
            intent_scores[intent] = score
        
        # Get best intent or default to concept explanation
        detected_intent = max(intent_scores, key=intent_scores.get) if max(intent_scores.values()) > 0 else QuestionIntent.CONCEPT_EXPLANATION
        
        # Classify complexity
        complexity = ResponseComplexity.STANDARD  # default
        for comp, patterns in self.complexity_patterns.items():
            if any(re.search(pattern, question_lower) for pattern in patterns):
                complexity = comp
                break
        
        # Detect content characteristics
        characteristics = {
            'needs_math': bool(re.search(r'\b(?:formula|equation|calculate|derive|proof|matrix|vector|integral|derivative)\b', question_lower)),
            'needs_code': bool(re.search(r'\b(?:code|program|implement|algorithm|function|class)\b', question_lower)),
            'needs_visual': bool(re.search(r'\b(?:diagram|chart|graph|visual|plot|draw)\b', question_lower)),
            'domain_specific': bool(re.search(r'\b(?:deep learning|machine learning|physics|chemistry|biology|economics)\b', question_lower))
        }
        
        return detected_intent, complexity, characteristics
    
    def generate_adaptive_template(self, question: str) -> Dict[str, str]:
        """Generate adaptive template based on question analysis"""
        intent, complexity, characteristics = self.analyze_question(question)
        
        # Get base sections for this intent and complexity
        base_sections = self.intent_templates.get(intent, {}).get(
            complexity, 
            self.intent_templates[QuestionIntent.CONCEPT_EXPLANATION][ResponseComplexity.STANDARD]
        )
        
        # Build the adaptive prompt
        template_parts = []
        template_parts.append(f"Respond to this {intent.value.replace('_', ' ')} question with {complexity.value} detail level.")
        template_parts.append("")
        
        # Add section-specific instructions
        for section_key in base_sections:
            section = self.available_sections[section_key]
            template_parts.append(f"## {section.name}")
            template_parts.append(section.prompt_instruction)
            
            # Add content-specific guidance
            if section.math_heavy and characteristics['needs_math']:
                template_parts.append("Use proper LaTeX notation for mathematical expressions.")
            if section.example_heavy:
                template_parts.append("Include concrete, detailed examples.")
            
            template_parts.append("")
        
        # Add content-specific instructions
        if characteristics['needs_code']:
            template_parts.append("Include relevant code examples with explanations.")
        if characteristics['needs_visual']:
            template_parts.append("Describe visual elements clearly or suggest diagrams where helpful.")
        
        # Add format guidance
        template_parts.append("## Format Guidelines")
        template_parts.append("- Use clear headings for organization")
        template_parts.append("- Include specific examples relevant to the question")
        template_parts.append("- Keep explanations at the appropriate complexity level")
        template_parts.append("- End with actionable next steps if applicable")
        
        return {
            'intent': intent.value,
            'complexity': complexity.value,
            'characteristics': characteristics,
            'sections': base_sections,
            'template': '\n'.join(template_parts)
        }
    
    def create_prompt(self, question: str) -> str:
        """Create the full adaptive prompt for the question"""
        template_info = self.generate_adaptive_template(question)
        
        prompt = f"""You are an expert educational tutor. Analyze this question and respond using the most appropriate format.

{template_info['template']}

Question: {question}

Provide your response following the structure above:"""
        
        return prompt

def main():
    """Test the adaptive template system"""
    engine = AdaptiveTemplateEngine()
    
    test_questions = [
        "What is the difference between supervised and unsupervised learning?",
        "How do I implement a neural network from scratch?",
        "Calculate the eigenvalues of the matrix [[2, 1], [1, 2]]",
        "What are the types of machine learning algorithms?",
        "Explain transformers in deep learning",
        "Why is my gradient descent not converging?",
        "Ideas for improving model performance"
    ]
    
    print("🧠 Testing Adaptive Template System")
    print("="*80)
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        
        # Analyze question
        intent, complexity, characteristics = engine.analyze_question(question)
        print(f"   Intent: {intent.value}")
        print(f"   Complexity: {complexity.value}")
        print(f"   Characteristics: {characteristics}")
        
        # Generate template
        template_info = engine.generate_adaptive_template(question)
        print(f"   Sections: {', '.join(template_info['sections'])}")
        print("-" * 40)

if __name__ == "__main__":
    main()
