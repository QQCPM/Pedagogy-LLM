# Educational LLM Research: Adaptive Prompting vs Ground Rules Analysis

🎯 **Goal**: Research and optimize prompting strategies for educational AI systems targeting research-level AI/ML content. This project compares adaptive template-based approaches, ground rules prompting, and raw model performance across Gemma 3 12B and Llama 3.1.

## Quick Start

1. **Setup Environment**:
```bash
pip install -r requirements.txt
# Ensure Ollama is running: ollama serve
```

2. **Basic Usage**:
```bash
# Interactive mode (choose model each time)
python ask.py "Your question here"

# Compare different prompting approaches
python ask.py "Explain eigenvalues" --ground-rules --quick --ai  # Ground rules approach
python ask.py "Explain eigenvalues" --quick --ai                # Adaptive templates
python ask.py "Explain eigenvalues" --raw --no-save             # Raw model baseline
```

3. **Model & Strategy Comparison** (See RESEARCH_REPORT.md for detailed analysis):
```bash
python ask.py "question" --compare        # Compare Gemma vs Llama
python ask.py "question" --ground-rules   # Research-focused prompting (recommended)
python ask.py "question" --detailed       # Comprehensive adaptive templates
python ask.py "question" --no-kb          # Without knowledge base integration
```

## Project Structure

```
├── ask.py                     # Main CLI interface  
├── config.py                  # Project configuration
├── ollama_inference.py        # Ollama inference engine with Obsidian LaTeX
├── smart_knowledge_base.py    # RAG system with concept graphs
├── adaptive_templates.py      # Dynamic response formatting
├── concept_flowchart.py       # Concept visualization
├── obsidian_knowledge_base.py # Base knowledge system
├── requirements.txt           # Dependencies
├── CLI_IMPROVEMENTS.md        # Workflow optimization specs
├── OBSIDIAN_WORKFLOW.md       # Obsidian integration guide
└── data/                      # Interaction logs and knowledge index
```

## Research Findings & Current Status

### ✅ Completed Research
- [x] **Comparative Analysis**: Adaptive templates vs Ground rules vs Raw baseline
- [x] **Model Comparison**: Gemma 3 12B vs Llama 3.1 performance analysis
- [x] **Context Optimization**: Extended 8K token context for comprehensive responses
- [x] **A/B Testing Framework**: Systematic evaluation of prompting strategies
- [x] **Knowledge Base Integration**: RAG system with concept graphs
- [x] **Research Report**: Comprehensive analysis documented in RESEARCH_REPORT.md

### 🔬 Key Research Results
- **Ground Rules Approach**: 2.7x longer responses, superior research-level depth
- **Llama 3.1**: Better for verbose, comprehensive explanations (6K+ tokens)
- **Gemma 3 12B**: More focused, efficient responses with excellent LaTeX support
- **Extended Context**: 8K tokens enabled qualitatively better mathematical derivations
- **Personalized Prompting**: Research-focused approach outperformed generic templates

### 📊 Proven Results
Comparison of different prompting strategies:

**Raw Baseline (Gemma 3)**:
> "Eigenvalues are scalars λ such that Av = λv for some non-zero vector v..." (~800 chars)

**Adaptive Templates**:
```markdown
## Intuitive Understanding
Eigenvalues represent scaling factors...
## Mathematical Definition  
For matrix A, eigenvalue λ satisfies Av = λv...
## Examples
[Structured examples with calculations]
```
*Result: ~5,500 chars, good structure but sometimes forced*

**Ground Rules Approach (Recommended)**:
```markdown
# Eigenvalues and Eigenvectors: A Research Perspective

Eigenvalues fundamentally capture how linear transformations 
scale space along specific directions. In the context of 
world models and representation learning...

$$Av = \lambda v$$

[Comprehensive derivations, research connections, 15K+ chars]
```
*Result: 2.7x more comprehensive, research-appropriate depth*

## Key Features

### 🧠 Prompting Strategies (Research-Validated)
- **Ground Rules Prompting**: Research-focused, flexible guidance (recommended)
- **Adaptive Templates**: Dynamic structure based on question analysis
- **Raw Baseline**: Direct model responses for comparison
- **Model-Specific Optimization**: Llama (verbose) vs Gemma (focused) tuning

### 🔬 Research Infrastructure
- **A/B Testing Framework**: Systematic strategy comparison
- **Extended Context**: 8K tokens for comprehensive technical explanations
- **Knowledge Base Integration**: RAG with concept graphs and Obsidian
- **Performance Analytics**: Response length, generation time, quality metrics
- **Multi-Model Support**: Gemma 3 12B and Llama 3.1 with optimized parameters

## Configuration

Edit `config.py` to customize:
- Model settings (memory, generation parameters)
- Training hyperparameters (LoRA, learning rate)
- Evaluation domains and metrics
- File paths and directories

## Hardware Requirements

**Minimum**:
- 16GB RAM
- 8GB GPU memory (with 4-bit quantization)
- 50GB disk space

**Recommended**:
- 32GB RAM  
- 24GB GPU memory
- 100GB disk space
- CUDA-compatible GPU

## Research Impact & Usage

**Current Status**: Research complete, system deployed for daily use

1. **Proven Approach**: Ground rules prompting established as superior for research content
2. **Optimized Parameters**: Extended context (8K tokens) + research-focused prompting
3. **Integration**: Seamless Obsidian workflow for research note-taking
4. **Future Work**: Multi-user studies, objective pedagogical metrics, model fine-tuning

## Architecture Overview

### Core Components
- `ollama_inference.py`: Multi-model inference engine (Gemma 3, Llama 3.1)
- `ground_rules_prompt.py`: Research-focused prompting system (recommended)
- `adaptive_templates.py`: Dynamic template generation system
- `smart_knowledge_base.py`: RAG system with concept graphs
- `ask.py`: CLI interface with A/B testing capabilities

### Research Pipeline
1. **Question Analysis** → Intent classification & complexity detection
2. **Strategy Selection** → Ground rules vs Templates vs Raw baseline
3. **Model Selection** → Gemma 3 (focused) vs Llama 3.1 (verbose)
4. **Context Enhancement** → Knowledge base integration (RAG)
5. **Response Generation** → Optimized parameters per strategy
6. **Evaluation & Storage** → Performance metrics & Obsidian integration

## Prompting Strategy Comparison

### Ground Rules Approach (Recommended for Research)
```python
# Flexible, principles-based guidance
ground_rules = """
You are an expert AI Research Assistant and Mentor.
- Primary Focus: World Models, Causal AI, Deep Learning, RL
- Audience: Advanced undergraduate pursuing research  
- Style: Analytical, rigorous, intellectually honest
- Length: Comprehensive coverage for research notes
- Format: Obsidian-ready markdown with LaTeX math
"""
```
**Result**: Natural organization, research-appropriate depth, 2.7x longer responses

### Adaptive Templates Approach
```python
# Dynamic structure based on question analysis
if intent == "concept_explanation":
    sections = ["intuition", "definition", "examples", "applications"]
elif intent == "comparison":
    sections = ["comparison_table", "pros_cons", "use_cases"]
```
**Result**: Consistent structure, good for basic concepts, can be limiting for advanced topics

### Model-Specific Optimizations
- **Llama 3.1**: 6K+ tokens, verbose explanations, comprehensive coverage
- **Gemma 3 12B**: Focused responses, excellent LaTeX support, efficient generation

---

## Research Publication

See **RESEARCH_REPORT.md** for the complete comparative analysis:
- Quantitative metrics (response length, generation time, context utilization)
- Qualitative assessment of pedagogical effectiveness
- A/B testing methodology and results
- Implications for educational AI development

**Key Finding**: Ground rules-based prompting significantly outperforms rigid templates for research-level educational content, challenging conventional wisdom about structured prompting necessity.

---

**Research-validated AI assistant optimized for World Models and Causal AI learning!** 🧠🔬
