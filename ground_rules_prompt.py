#!/usr/bin/env python3
"""
Ground Rules System for Educational Responses
Minimal guiding principles instead of rigid templates
"""

def create_ground_rules_prompt(question: str, known_concepts=None) -> str:
    """Create a personalized prompt optimized for research-level AI learning"""
    
    # Refined ground rules for research-focused AI learning
    ground_rules = """You are an expert AI Research Assistant and Mentor.

- **Primary Focus:** World Models, Causal AI, Deep Learning, and Reinforcement Learning.
- **Audience:** An advanced undergraduate pursuing research.
- **Style:** Provide deep, comprehensive answers with an analytical, rigorous tone that is intellectually honest about complexity. Briefly define key prerequisite concepts as needed, then build logically to the advanced topic.
- **Length:** Create detailed, thorough explanations suitable for research notes. Aim for comprehensive coverage rather than brevity.
- **Format:** Use Obsidian-ready markdown. Use LaTeX for all math ($inline$ and $$block$$).
- **Structure:** When helpful, organize complex information with markdown tables or Mermaid code for graphs."""

    # Add context for known concepts if available
    context = ""
    if known_concepts and len(known_concepts) > 0:
        context = f"\n\nCONTEXT: The learner is already familiar with: {', '.join(known_concepts)}. Build on this knowledge rather than re-explaining basics."
    
    # Final prompt - minimal but principled
    prompt = f"""{ground_rules}{context}

QUESTION: {question}

Provide an educational response that follows these principles while maintaining your natural teaching style:"""
    
    return prompt


def create_focused_rules_prompt(question: str, focus_mode: str = "standard") -> str:
    """Create ground rules with specific focus (brief, detailed, etc.)"""
    
    focus_guidance = {
        "brief": "• Keep response concise but complete - focus on core concepts\n• Prioritize clarity over comprehensive coverage",
        "detailed": "• Provide comprehensive coverage with multiple examples\n• Include advanced connections and implications\n• Show detailed derivations or step-by-step processes",
        "intuitive": "• Emphasize analogies and intuitive explanations\n• Minimize formalism unless essential\n• Focus on conceptual understanding over technical precision",
        "technical": "• Include precise definitions and formal notation\n• Show mathematical derivations when relevant\n• Emphasize accuracy and technical depth"
    }
    
    base_rules = """You are an educational AI. Follow these core principles:
• Build understanding, not just provide information
• Use examples and connections to aid learning  
• Explain the significance and "why" behind concepts"""
    
    specific_focus = focus_guidance.get(focus_mode, "")
    
    prompt = f"""{base_rules}
{specific_focus}

QUESTION: {question}

Provide an educational response following these principles:"""
    
    return prompt