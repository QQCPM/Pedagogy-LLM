# CLI Workflow Improvements Specification

## Current System
- 2 models: Gemma 3 12B (preferred) + Llama 3.1 8B (alternative)  
- Manual model selection on each run
- Interactive feedback loops
- Manual Obsidian folder selection

## Efficiency Improvements Needed

### 🎯 Core Workflow Flags

**Model Selection:**
```bash
--compare     # Both models → choose preferred → Obsidian  
--raw         # Raw Gemma 3 (no educational prompts)
--no-kb       # No RAG, but keep educational formatting
--default     # Skip model selection, use Gemma 3 directly
```

**Save Mode Control:**
```bash
--quick       # Skip feedback loop, auto-save to Obsidian
--no-save     # Quick question, don't save anywhere
```

**Folder Shortcuts (auto-save to specific Obsidian folders):**
```bash
--math        # Auto-save to Mathematics folder
--ai          # Auto-save to AI-ML folder  
--physics     # Auto-save to Physics folder
--cs          # Auto-save to Computer-Science folder
--general     # Auto-save to General folder
```

**Response Control:**
```bash
--brief       # Force brief, focused answers
--detailed    # Force comprehensive explanations
```

**Session Management:**
```bash
--session     # Multiple questions without restarting
```

### 🚀 Optimal Workflow Examples

**Most Common Use (90% of cases):**
```bash
python ask.py "question" --quick --math
# → Gemma 3 → Mathematics folder → No friction
```

**Compare Mode:**
```bash
python ask.py "question" --compare --ai
# → Both models → Choose preferred → AI-ML folder
```

**Raw Exploration:**
```bash
python ask.py "question" --raw --no-save
# → Pure Gemma 3 → No educational formatting → Don't save
```

**Deep Learning Session:**
```bash
python ask.py "question" --detailed --session --ai
# → Comprehensive answer → Follow-up questions → AI-ML folder
```

**Quick Check (no save):**
```bash
python ask.py "what's 2+2" --brief --no-save
# → Quick answer → No save → No friction
```

## Implementation Priority

### Phase 1 - Biggest Impact:
1. `--default` flag (skip model selection)
2. `--quick` flag (skip feedback, auto-save)
3. Folder shortcuts (`--math`, `--ai`, etc.)

### Phase 2 - Nice to Have:
1. `--raw` flag (pure Gemma)
2. `--brief`/`--detailed` flags
3. `--session` mode

### Phase 3 - Optional:
1. Config file for personal preferences
2. History/recent questions
3. Favorite questions bookmark

## Personal Usage Patterns to Optimize For

1. **Quick Math Questions** → `--quick --math`
2. **AI/ML Learning** → `--detailed --ai --session` 
3. **Compare Perspectives** → `--compare --cs`
4. **Raw Model Testing** → `--raw --no-save`
5. **Physics Problems** → `--quick --physics`

## Implementation Notes

- Keep current functionality intact
- Add flags as optional enhancements
- Default behavior unchanged (backwards compatible)
- Focus on reducing friction for personal workflow
- Maintain quality while increasing speed

---

*Last Updated: September 2025*  
*Purpose: Personal learning workflow optimization*