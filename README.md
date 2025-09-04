# Educational LLM Specialization Project

🎯 **Goal**: Fine-tune Gemma 3 12B to provide structured, pedagogically effective explanations for technical subjects (math, AI/ML, physics).

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

# Quick workflow examples  
python ask.py "Explain eigenvalues" --quick --math
python ask.py "Compare ML algorithms" --compare --ai
python ask.py "Raw Gemma test" --raw --no-save
```

3. **Advanced Usage** (See CLI_IMPROVEMENTS.md for full spec):
```bash
python ask.py "question" --no-kb     # No knowledge base
python ask.py "question" --compare   # Compare both models  
python ask.py "question" --detailed  # Comprehensive answers
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

## Phase 1: Foundation (Current)

### ✅ Completed
- [x] Environment setup and dependencies
- [x] Project structure and configuration
- [x] Gemma 3 12B inference script with memory optimization
- [x] Baseline evaluation framework
- [x] Data loading and preprocessing utilities
- [x] System monitoring and logging

### 🎯 Your Tasks (Human)
- [ ] Define your specific learning domains (Linear Algebra, World Models, etc.)
- [ ] Create 50-question evaluation dataset manually
- [ ] Test baseline Gemma responses and document weaknesses  
- [ ] Define what "educational effectiveness" means for your use case
- [ ] Set up hardware/cloud GPU environment
- [ ] Document your learning style preferences

### 📊 Success Metrics
Compare base model response vs target educational format:

**Base Model**:
> "Eigenvalues are scalars associated with linear transformations..."

**Target Educational Format**:
```markdown
# Eigenvalues and Eigenvectors

## Intuitive Understanding
Think of eigenvalues as 'special directions'...

## Mathematical Definition
Av=λv where...

## Step-by-step Example
[Detailed walkthrough]

## Why This Matters
[Applications in PCA, neural networks, etc.]

## Connection to Other Concepts
[Links to matrix diagonalization, spectral theory...]
```

## Key Features

### 🤖 Automated (Claude Handles)
- **Memory-Optimized Inference**: 4-bit quantization for Gemma 3 12B
- **Evaluation Framework**: Structured comparison tools
- **Data Processing**: Format validation and quality checking
- **System Monitoring**: GPU/memory tracking during training
- **Training Pipeline**: LoRA fine-tuning setup (Phase 2)

### 👤 Manual (You Handle)
- **Content Curation**: High-quality educational examples
- **Domain Expertise**: Subject matter validation
- **Learning Style**: Personal preferences and effectiveness assessment
- **Quality Control**: Final evaluation of model improvements

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

## Next Steps

1. **Complete Phase 1**: Run setup and baseline evaluation
2. **Phase 2**: Create training dataset with improved responses
3. **Phase 3**: Fine-tune with LoRA and evaluate improvements

## Files Overview

### Core Scripts
- `gemma_inference.py`: Main inference engine with educational prompting
- `evaluation_framework.py`: A/B testing and quality metrics
- `data_utils.py`: Data processing and validation
- `monitor.py`: System resource monitoring
- `setup.py`: One-click environment setup

### Data Flow
1. **Questions** → `evaluation_framework.py` → evaluation dataset
2. **Dataset** → `gemma_inference.py` → baseline responses  
3. **Responses** → manual improvement → training data
4. **Training** → LoRA fine-tuning → specialized model
5. **Evaluation** → comparison reports → insights

## Educational Format Template

The target format for educational responses:

```markdown
# [Concept Name]

## Intuitive Understanding
- Start with analogies and intuitive explanations
- Build conceptual understanding before formalism

## Mathematical Definition  
- Formal definition with proper notation
- Clear variable explanations

## Step-by-step Example
- Concrete walkthrough with calculations
- Show each step clearly

## Why This Matters
- Real-world applications
- Importance in the field

## Connection to Other Concepts
- Links to related topics
- Broader context and implications
```

---

**Ready to build an LLM that actually teaches instead of just answering!** 🚀
