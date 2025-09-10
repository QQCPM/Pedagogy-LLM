# Educational LLM Research: Comprehensive Model Evaluation

**Goal**: Research and optimize prompting strategies for educational AI systems. This project provides a comprehensive evaluation of seven large language models across different prompting approaches, delivering insights for educational AI deployment.

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

3. **Model & Strategy Comparison**:
```bash
python ask.py "question" --compare        # Compare different models
python ask.py "question" --ground-rules   # Research-focused prompting (recommended)
python ask.py "question" --detailed       # Comprehensive adaptive templates
python ask.py "question" --no-kb          # Without knowledge base integration
```

## Project Structure

```
├── ask.py                     # Main CLI interface  
├── config.py                  # Project configuration
├── ollama_inference.py        # Ollama inference engine
├── smart_knowledge_base.py    # RAG system with concept graphs
├── adaptive_templates.py      # Dynamic response formatting
├── concept_flowchart.py       # Concept visualization
├── obsidian_knowledge_base.py # Base knowledge system
├── ground_rules_prompt.py     # Research-focused prompting
├── generate_final_report.py   # Evaluation report generation
├── requirements.txt           # Python dependencies
└── data/                      # Evaluation results and knowledge index
```

## Research Findings

### Completed Research
- **Comprehensive Model Evaluation**: 7 models tested across 3 key domains
- **Prompting Strategy Analysis**: Ground rules vs Adaptive templates vs Raw baseline
- **Performance Benchmarking**: Speed, comprehensiveness, and efficiency metrics
- **Resource Optimization**: Parameter efficiency and deployment recommendations
- **Knowledge Base Integration**: RAG system with concept graphs

### Key Research Results

**Model Performance Rankings (Speed)**:
1. GPT-OSS 20B (Ground Rules): 285.1 chars/sec
2. Gemma 3 12B (Ground Rules): 201.1 chars/sec  
3. GPT-OSS 20B (Raw): 193.5 chars/sec

**Model Performance Rankings (Comprehensiveness)**:
1. GPT-OSS 120B (Ground Rules): 9,852 chars avg
2. GPT-OSS 20B (Ground Rules): 7,937 chars avg
3. DeepSeek R1 70B (Ground Rules): 6,894 chars avg

**Ground Rules Approach Benefits**:
- 1.5x longer responses with higher educational value
- Often faster generation than raw prompting
- Superior research-appropriate depth and structure
- Consistent performance across all model families

### Proven Results

**Raw Baseline Example**:
> "Eigenvalues are scalars λ such that Av = λv for some non-zero vector v..." (~800 chars)

**Ground Rules Approach Example**:
```markdown
# Eigenvalues and Eigenvectors: A Research Perspective

Eigenvalues fundamentally capture how linear transformations 
scale space along specific directions. In the context of 
world models and representation learning...

$$Av = \lambda v$$

[Comprehensive derivations, research connections, detailed analysis]
```
Result: 2.7x more comprehensive, research-appropriate depth

## Model Families Evaluated

### GPT-OSS Family
- **GPT-OSS 120B (116.8B params)**: Highest comprehensiveness, moderate speed
- **GPT-OSS 20B (20.9B params)**: Optimal speed-quality balance, top performer

### Gemma Family  
- **Gemma 3 27B (27.4B params)**: Balanced performance
- **Gemma 3 12B (12.2B params)**: Exceptional efficiency, best performance per parameter

### Llama Family
- **Llama 3.3 70B (70.6B params)**: Consistent quality, moderate speed
- **Llama 3.1 70B (70.6B params)**: High precision, slower generation

### DeepSeek Family
- **DeepSeek R1 70B (70.6B params)**: Specialized reasoning capabilities

## Key Features

### Prompting Strategies
- **Ground Rules Prompting**: Research-focused, flexible guidance (recommended)
- **Adaptive Templates**: Dynamic structure based on question analysis
- **Raw Baseline**: Direct model responses for comparison

### Research Infrastructure
- **Comprehensive Evaluation**: 42 tests across 7 models, 100% success rate
- **Performance Analytics**: Speed, comprehensiveness, efficiency metrics
- **Knowledge Base Integration**: RAG with concept graphs and Obsidian
- **Automated Reporting**: JSON and Markdown report generation

## Configuration

Edit `config.py` to customize:
- Model settings and generation parameters
- Knowledge base configuration
- File paths and directories
- Evaluation domains and metrics

## Hardware Requirements

**Minimum**:
- 16GB RAM
- 8GB GPU memory (for smallest models)
- 10GB disk space

**Recommended**:
- 32GB RAM  
- 24GB GPU memory
- 50GB disk space for multiple models

**Model-Specific Requirements**:
- Gemma 3 12B: 8GB memory
- GPT-OSS 20B: 13GB memory
- Llama/DeepSeek 70B: 42-74GB memory
- GPT-OSS 120B: 65GB memory

## Research Impact & Usage

**Current Status**: Comprehensive evaluation complete with actionable insights

1. **Proven Approach**: Ground rules prompting established as superior across all models
2. **Efficiency Insights**: Smaller models often outperform larger ones in speed
3. **Resource Optimization**: Clear guidance for cost-effective deployment
4. **Integration Ready**: Seamless Obsidian workflow for research note-taking

## Architecture Overview

### Core Components
- `ollama_inference.py`: Multi-model inference engine supporting 7+ models
- `ground_rules_prompt.py`: Research-focused prompting system
- `adaptive_templates.py`: Dynamic template generation system
- `smart_knowledge_base.py`: RAG system with concept graphs
- `ask.py`: CLI interface with comprehensive model support

### Research Pipeline
1. **Model Selection** → Choose from 7 evaluated models based on requirements
2. **Prompting Strategy** → Ground rules vs Templates vs Raw baseline
3. **Context Enhancement** → Knowledge base integration (RAG)
4. **Response Generation** → Optimized parameters per model and strategy
5. **Performance Tracking** → Comprehensive metrics and analytics

## Recommendations

### For Speed-Optimized Deployment
- **Primary**: GPT-OSS 20B with Ground Rules prompting
- **Alternative**: Gemma 3 12B for resource-constrained environments

### For Comprehensive Responses
- **Primary**: GPT-OSS 120B with Ground Rules prompting
- **Alternative**: DeepSeek R1 70B for reasoning-heavy tasks

### For Resource-Constrained Deployment
- **Recommended**: Gemma 3 12B (8GB memory, excellent efficiency)

## Documentation

- **COMPREHENSIVE_MODEL_EVALUATION_REPORT.md**: Complete research findings and analysis
- **RESEARCH_REPORT.md**: Previous research findings and methodology
- **CLI_IMPROVEMENTS.md**: Workflow optimization specifications
- **OBSIDIAN_WORKFLOW.md**: Integration guide for research note-taking

## Key Finding

Ground rules-based prompting significantly outperforms structured templates across all seven models, providing superior educational value while often maintaining or improving generation speed. Model size does not directly correlate with performance quality, with smaller, efficient models often outperforming larger alternatives.

Research-validated AI assistant optimized for educational applications with comprehensive model evaluation and deployment guidance.