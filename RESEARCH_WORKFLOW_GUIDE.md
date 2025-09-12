# Research Workflow Guide: World Models & GPT OSS 120B

## Quick Start Commands

### 🔬 Research Mode (Recommended)
```bash
# Optimized for AI/ML research - interactive model selection + ground rules + auto-save to AI-ML folder
python ask.py "Explain World Models architecture and dreaming mechanism" --research

# Research mode with specific folder
python ask.py "Compare GPT OSS 120B with other MoE architectures" --research --cs

# Note: Research mode now prompts you to choose from all 7 available models!
```

### 🤖 Interactive Model Selection
```bash
# Choose from 7 available models interactively
python ask.py "Your question here"

# Available models:
# 1. Gemma 3 12B - Efficient, excellent LaTeX support (8GB)
# 2. GPT-OSS 20B - Speed champion, 285.1 chars/sec (13GB)  
# 3. GPT-OSS 120B - Most comprehensive, 9,852 chars avg (65GB)
# 4. Llama 3.1 70B - Large model, comprehensive (74GB)
# 5. Llama 3.3 70B - Latest Llama, good performance (42GB)
# 6. DeepSeek R1 70B - Reasoning specialist (42GB)
# 7. Gemma 3 27B - Larger Gemma variant (17GB)
```

### 🎯 Specific Model Research
```bash
# Force specific model with ground rules
python ask.py "World Models VAE compression analysis" --ground-rules --ai

# Compare models side by side
python ask.py "GPT OSS 120B vs DeepSeek R1 reasoning capabilities" --compare --ai
```

## Research Question Templates

### World Models Research
```bash
# Architecture Deep Dive
python ask.py "Explain World Models VAE-MDN-RNN-Controller architecture in detail" --research

# Training Methodology
python ask.py "How does World Models dreaming mechanism work and why is it effective?" --research

# Modern Applications
python ask.py "How do World Models concepts relate to modern foundation models and LLMs?" --research

# Implementation Details
python ask.py "What are the key hyperparameters and design choices in World Models training?" --research
```

### GPT OSS 120B Research
```bash
# Architecture Analysis
python ask.py "Explain GPT OSS 120B MoE architecture and MXFP4 quantization" --research

# Performance Analysis
python ask.py "Compare GPT OSS 120B performance with other 100B+ parameter models" --research

# Capabilities Assessment
python ask.py "What are GPT OSS 120B's agentic capabilities and reasoning modes?" --research

# Integration Opportunities
python ask.py "How can GPT OSS 120B enhance educational AI systems?" --research
```

### Synthesis Questions
```bash
# Connections
python ask.py "What are the conceptual connections between World Models and GPT OSS 120B?" --research

# Future Directions
python ask.py "How could World Models concepts improve LLM reasoning and planning?" --research

# Educational Applications
python ask.py "Applications of World Models and GPT OSS 120B in educational AI" --research
```

## Workflow Options

### Research Modes
- `--research`: Ground rules + Gemma 3 + auto-save to AI-ML
- `--ground-rules`: Research-focused prompting with any model
- `--raw`: Pure model output, no educational formatting

### Model Selection
- No flags: Interactive selection from 7 models
- `--default`: Use Gemma 3 12B automatically
- `--compare`: Side-by-side comparison of Gemma 3 vs Llama 3.1

### Save Options
- `--quick`: Auto-save without feedback
- `--no-save`: Don't save response
- `--ai`, `--math`, `--cs`, etc.: Auto-save to specific folders

### Response Style
- `--brief`: Concise, focused answers
- `--detailed`: Comprehensive explanations
- `--no-kb`: Disable knowledge base integration

## Example Research Session

```bash
# Start with World Models overview
python ask.py "Provide comprehensive overview of World Models paper by Ha and Schmidhuber" --research

# Deep dive into architecture
python ask.py "Explain VAE component in World Models - compression and reconstruction" --research

# Explore training methodology  
python ask.py "How does MDN-RNN predict future states in World Models?" --research

# Test with GPT OSS 120B specifically
python ask.py "Analyze GPT OSS 120B MoE efficiency compared to dense models" --ground-rules --ai

# Compare models on complex topic
python ask.py "Compare World Models dreaming with modern LLM chain-of-thought reasoning" --compare --ai
```

## Integration with Your Knowledge Base

Your system automatically:
- Builds concept graphs from responses
- Creates semantic relationships
- Enables smart search across research notes
- Integrates with Obsidian vault structure

## Performance Expectations

Based on your evaluation results:

| Model | Speed (chars/sec) | Avg Length | Memory | Best Use Case |
|-------|------------------|------------|---------|---------------|
| Gemma 3 12B | 193.9 | 5,585 | 8GB | Efficient research |
| GPT-OSS 20B | 239.3 | 6,396 | 13GB | Speed + quality |
| GPT-OSS 120B | 131.9 | 7,151 | 65GB | Comprehensive analysis |
| DeepSeek R1 70B | 47.8 | 6,508 | 42GB | Complex reasoning |

## Next Steps

1. **Test the enhanced model selection**:
   ```bash
   python ask.py "Test question"
   ```

2. **Use research mode for World Models**:
   ```bash
   python ask.py "World Models architecture overview" --research
   ```

3. **Compare models on GPT OSS 120B**:
   ```bash
   python ask.py "GPT OSS 120B capabilities analysis" --compare
   ```

Your research infrastructure is now optimized for exhaustive World Models and GPT OSS 120B investigation!
