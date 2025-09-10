#!/usr/bin/env python3
"""
Educational LLM with Interactive Obsidian Save & Enhanced Feedback
Usage: python ask.py "Your question here"
"""
import sys
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from ollama_inference import OllamaEducationalInference
from config import config

# UPDATE THIS PATH TO YOUR OBSIDIAN VAULT
OBSIDIAN_VAULT = Path("/Users/tld/Documents/Obsidian LLM")

# Folder structure for research notes
FOLDERS = {
    "1": "Mathematics",
    "2": "AI-ML", 
    "3": "Physics",
    "4": "Computer-Science",
    "5": "General"
}

# Available models for selection (streamlined to 2 models)
AVAILABLE_MODELS = {
    "1": {"name": "Gemma 3 12B", "model_id": "gemma3:12b", "description": "Your preferred model"},
    "2": {"name": "Llama 3.1 8B", "model_id": "llama3.1:8b", "description": "Alternative perspective"}
}

# Default model for --default flag
DEFAULT_MODEL = "gemma3:12b"

# Folder shortcuts mapping
FOLDER_SHORTCUTS = {
    "--math": "Mathematics",
    "--ai": "AI-ML",
    "--physics": "Physics", 
    "--cs": "Computer-Science",
    "--general": "General"
}

def check_ollama_connection():
    """Check if Ollama server is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def choose_model():
    """Interactive model selection"""
    print("\n🤖 Choose your model:")
    for key, model_info in AVAILABLE_MODELS.items():
        print(f"  {key}. {model_info['name']} - {model_info['description']}")
    
    while True:
        choice = input("\nSelect model (1-2): ").strip()
        if choice in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[choice]['model_id']
        print("Please enter 1 or 2")

def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py 'Your question here' [OPTIONS]")
        print("Example: python ask.py 'Explain eigenvalues' --quick --math")
        print("")
        print("Core Options:")
        print("  --research      Optimized research mode (ground rules + extended context + quick save)")
        print("  --default       Use Gemma 3 without model selection")
        print("  --raw           Raw Gemma 3 (no educational formatting)")
        print("  --ground-rules  Use research-focused ground rules (recommended)")
        print("  --compare       Compare both models → choose preferred → save")
        print("  --no-kb         Disable knowledge base (but keep educational formatting)")
        print("")
        print("Workflow Options:")
        print("  --quick     Skip feedback loop, auto-save to Obsidian")
        print("  --no-save   Don't save anywhere (quick questions)")
        print("  --brief     Force brief, focused answers")
        print("  --detailed  Force comprehensive explanations")
        print("")
        print("Folder Shortcuts (auto-save to Obsidian):")
        print("  --math      Save to Mathematics folder")
        print("  --ai        Save to AI-ML folder")
        print("  --physics   Save to Physics folder")
        print("  --cs        Save to Computer-Science folder")
        print("  --general   Save to General folder")
        print("")
        print("Examples:")
        print("  python ask.py 'Explain world models' --research")
        print("  python ask.py 'Explain eigenvalues' --quick --math")
        print("  python ask.py 'Compare ML algorithms' --compare --ai")
        print("  python ask.py 'Raw test' --raw --no-save")
        return

    # Parse command line arguments
    args = sys.argv[1:]
    
    # Initialize flags
    compare_mode = False
    use_kb = True
    default_model = False
    raw_mode = False
    ground_rules_mode = False
    research_mode = False
    quick_mode = False
    no_save = False
    force_brief = False
    force_detailed = False
    folder_shortcut = None
    
    # Parse all flags
    if '--compare' in args:
        compare_mode = True
        args.remove('--compare')
    
    if '--no-kb' in args:
        use_kb = False
        args.remove('--no-kb')
        print("📚 Knowledge base disabled for this session")
    
    if '--default' in args:
        default_model = True
        args.remove('--default')
        print("🎯 Using default model (Gemma 3)")
    
    if '--raw' in args:
        raw_mode = True
        default_model = True  # Raw mode implies Gemma 3
        args.remove('--raw')
        print("🔥 Raw mode: Pure Gemma 3, no educational formatting")
    
    if '--research' in args:
        research_mode = True
        ground_rules_mode = True  # Research mode includes ground rules
        default_model = True      # Research mode uses Gemma 3
        quick_mode = True         # Research mode auto-saves (can be overridden by --no-save)
        folder_shortcut = "AI-ML" # Default to AI-ML folder
        args.remove('--research')
        print("🔬 Research mode: Optimized for AI/ML research (ground rules + extended context + quick save)")
    
    if '--ground-rules' in args:
        ground_rules_mode = True
        default_model = True  # Ground rules mode implies Gemma 3
        args.remove('--ground-rules')
        print("🎯 Ground rules mode: Research-focused prompting")
    
    if '--quick' in args:
        quick_mode = True
        args.remove('--quick')
        print("⚡ Quick mode: Auto-save, no feedback loop")
    
    if '--no-save' in args:
        no_save = True
        args.remove('--no-save')
        print("🚫 No-save mode: Won't save response")
    
    if '--brief' in args:
        force_brief = True
        args.remove('--brief')
        print("📝 Brief mode: Short, focused answers")
    
    if '--detailed' in args:
        force_detailed = True
        args.remove('--detailed')
        print("📚 Detailed mode: Comprehensive explanations")
    
    # Check for folder shortcuts
    for shortcut, folder_name in FOLDER_SHORTCUTS.items():
        if shortcut in args:
            folder_shortcut = folder_name
            args.remove(shortcut)
            print(f"📁 Auto-save to {folder_name} folder")
            break
    
    # Validation: conflicting flags
    if raw_mode and compare_mode:
        print("❌ Error: --raw and --compare cannot be used together")
        return
    
    if raw_mode and (ground_rules_mode or research_mode):
        print("❌ Error: --raw cannot be used with --ground-rules or --research")
        return
    
    if force_brief and force_detailed:
        print("❌ Error: --brief and --detailed cannot be used together")
        return
    
    # Allow --no-save to override automatic quick_mode from research mode
    if no_save and research_mode:
        quick_mode = False  # Override research mode's automatic quick save
    elif quick_mode and no_save:
        print("❌ Error: --quick and --no-save cannot be used together")
        return
    
    question = " ".join(args)
    session_id = str(uuid.uuid4())[:8]  # Short session ID for tracking
    
    # Check Ollama connection first
    if not check_ollama_connection():
        print("❌ Ollama server is not running!")
        print("💡 Start it with: ollama serve")
        return
    
    # Model selection logic
    selected_model = None
    if compare_mode:
        print(f"⚔️ Model Comparison Mode: Gemma 3 vs Llama 3.1")
    elif default_model or raw_mode or ground_rules_mode or research_mode:
        selected_model = DEFAULT_MODEL
        if raw_mode:
            print(f"🔥 Raw mode with {selected_model}")
        elif research_mode:
            print(f"🔬 Research mode with {selected_model}")
        elif ground_rules_mode:
            print(f"🎯 Ground rules mode with {selected_model}")
        else:
            print(f"🎯 Default model: {selected_model}")
    else:
        # Interactive model selection
        selected_model = choose_model()
        print(f"🤖 Selected model: {selected_model}")
    
    print(f"🧠 Asking: {question}")
    print(f"📋 Session: {session_id}")
    
    # Initialize inference system(s)
    try:
        if compare_mode:
            # Initialize both models for comparison
            gemma_inference = OllamaEducationalInference(
                model_name="gemma3:12b",
                vault_path=str(OBSIDIAN_VAULT),
                use_knowledge_base=use_kb
            )
            llama_inference = OllamaEducationalInference(
                model_name="llama3.1:8b",
                vault_path=str(OBSIDIAN_VAULT),
                use_knowledge_base=use_kb
            )
        else:
            # Single model mode with user selection
            inference = OllamaEducationalInference(
                model_name=selected_model,
                vault_path=str(OBSIDIAN_VAULT),
                use_knowledge_base=use_kb
            )
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return
    
    if compare_mode:
        # Model comparison workflow
        run_model_comparison(question, session_id, gemma_inference, llama_inference, use_kb, 
                           folder_shortcut, quick_mode)
    else:
        # Single model interaction loop
        run_single_model_interaction(question, session_id, inference, raw_mode, ground_rules_mode,
                                   research_mode, quick_mode, no_save, force_brief, force_detailed, folder_shortcut)

def collect_feedback(question, response, attempt):
    """Collect comprehensive feedback from user"""
    print("\n" + "="*60)
    print("📊 FEEDBACK & NEXT STEPS")
    print("="*60)
    
    # Quality rating
    while True:
        try:
            rating = input("⭐ Rate this response (1-5, 5=excellent): ").strip()
            rating = int(rating)
            if 1 <= rating <= 5:
                break
            else:
                print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    # Specific feedback categories
    print("\n🎯 What worked well? (select all that apply)")
    print("  1. Clear explanations")
    print("  2. Good examples")
    print("  3. Math notation")
    print("  4. Structure/organization")
    print("  5. Depth of content")
    print("  6. Connections to other topics")
    
    strengths = input("Enter numbers (e.g., '1,3,5'): ").strip().split(',')
    strengths = [s.strip() for s in strengths if s.strip().isdigit()]
    
    print("\n🎯 What could be improved?")
    print("  1. Too complex/advanced")
    print("  2. Too simple/basic")
    print("  3. Poor examples")
    print("  4. Math notation issues")
    print("  5. Missing connections")
    print("  6. Poor structure")
    print("  7. Too long")
    print("  8. Too short")
    
    improvements = input("Enter numbers (e.g., '2,4'): ").strip().split(',')
    improvements = [i.strip() for i in improvements if i.strip().isdigit()]
    
    # Open feedback
    specific_feedback = input("\n💭 Any specific feedback? (optional): ").strip()
    
    # Next action
    print("\n🚀 What would you like to do?")
    print("  1. Save to Obsidian")
    print("  2. Regenerate with feedback")
    print("  3. Exit without saving")
    
    while True:
        choice = input("Choose (1/2/3): ").strip()
        if choice == '1':
            action = 'save'
            break
        elif choice == '2':
            action = 'regenerate'
            break
        elif choice == '3':
            action = 'exit'
            break
        else:
            print("Please enter 1, 2, or 3")
    
    return {
        'rating': rating,
        'strengths': strengths,
        'improvements': improvements,
        'specific_feedback': specific_feedback,
        'action': action,
        'attempt': attempt,
        'timestamp': datetime.now().isoformat()
    }

def enhance_question_with_feedback(question, feedback):
    """Enhance the question based on user feedback"""
    improvements_map = {
        '1': "Make this less complex and more beginner-friendly.",
        '2': "Make this more advanced and in-depth.",
        '3': "Provide better, more concrete examples.",
        '4': "Fix the mathematical notation and use proper LaTeX.",
        '5': "Add more connections to related concepts.",
        '6': "Improve the structure and organization.",
        '7': "Make this more concise.",
        '8': "Provide more detailed explanation."
    }
    
    if feedback['improvements']:
        improvement_text = " ".join([improvements_map.get(i, "") for i in feedback['improvements']])
        enhanced_question = f"{question}\n\nPLEASE IMPROVE: {improvement_text}"
        
        if feedback['specific_feedback']:
            enhanced_question += f" Also: {feedback['specific_feedback']}"
        
        return enhanced_question
    
    return question

def save_interaction_data(session_id, question, response, feedback, attempt):
    """Save interaction data for analysis and training"""
    interaction_data = {
        'session_id': session_id,
        'question': question,
        'response': response,
        'feedback': feedback,
        'attempt': attempt,
        'timestamp': datetime.now().isoformat(),
        'model': 'gemma3:12b'
    }
    
    # Save to interactions log
    interactions_dir = Path(config.data_dir) / "interactions"
    interactions_dir.mkdir(exist_ok=True)
    
    interaction_file = interactions_dir / f"{datetime.now().strftime('%Y-%m-%d')}_interactions.jsonl"
    
    with open(interaction_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(interaction_data, ensure_ascii=False) + '\n')
    
    # Also save high-rated responses for training data
    if feedback['rating'] >= 4:
        training_dir = Path(config.data_dir) / "training_data"
        training_dir.mkdir(exist_ok=True)
        
        training_file = training_dir / "high_quality_responses.jsonl"
        
        training_data = {
            'question': question,
            'response': response,
            'rating': feedback['rating'],
            'strengths': feedback['strengths'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(training_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(training_data, ensure_ascii=False) + '\n')

def save_to_obsidian_direct(question, response, folder_name, feedback=None, model_name=None):
    """Direct save to Obsidian vault with specified folder (no interaction)"""
    
    # Check vault exists
    if not OBSIDIAN_VAULT.exists():
        print(f"❌ Obsidian vault not found at: {OBSIDIAN_VAULT}")
        print("💡 Update OBSIDIAN_VAULT path at top of script")
        return
    
    # Create folder if it doesn't exist
    folder_path = OBSIDIAN_VAULT / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    # Clean question for filename (first 40 chars, safe characters only)
    safe_question = "".join(c for c in question if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_question = safe_question.replace(' ', '_')[:40]
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}_{safe_question}.md"
    
    # Create note content with proper metadata and feedback
    content = create_note_content(question, response, folder_name, feedback, model_name)
    
    # Save file
    filepath = folder_path / filename
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Saved to: {folder_name}/{filename}")
        print(f"📁 Full path: {filepath}")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")

def save_to_obsidian(question, response, feedback=None, model_name=None):
    """Interactive save to Obsidian vault"""
    
    # Check vault exists
    if not OBSIDIAN_VAULT.exists():
        print(f"❌ Obsidian vault not found at: {OBSIDIAN_VAULT}")
        print("💡 Update OBSIDIAN_VAULT path at top of script")
        return
    
    # Show folder options
    print(f"\n📁 Choose folder:")
    for key, folder in FOLDERS.items():
        print(f"  {key}. {folder}")
    print(f"  c. Custom folder name")
    
    choice = input("Enter choice: ").strip()
    
    # Determine folder
    if choice in FOLDERS:
        folder_name = FOLDERS[choice]
    elif choice.lower() == 'c':
        folder_name = input("Enter folder name: ").strip()
        if not folder_name:
            folder_name = "General"
    else:
        print("Invalid choice, using General")
        folder_name = "General"
    
    # Create folder if it doesn't exist
    folder_path = OBSIDIAN_VAULT / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    # Clean question for filename (first 40 chars, safe characters only)
    safe_question = "".join(c for c in question if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_question = safe_question.replace(' ', '_')[:40]
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}_{safe_question}.md"
    
    # Create note content with proper metadata and feedback
    content = create_note_content(question, response, folder_name, feedback, model_name)
    
    # Save file
    filepath = folder_path / filename
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Saved to: {folder_name}/{filename}")
        print(f"📁 Full path: {filepath}")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")

def create_note_content(question, response, folder, feedback=None, model_source=None):
    """Create properly formatted Obsidian note with feedback metadata"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Extract potential tags from response content
    tags = ["#ai-generated", f"#{folder.lower().replace('-', '_')}"]
    
    # Add quality tag based on rating
    if feedback and feedback.get('rating'):
        if feedback['rating'] >= 4:
            tags.append("#high-quality")
        elif feedback['rating'] <= 2:
            tags.append("#needs-review")
    
    # Determine source model
    source = model_source or "Educational LLM (Gemma 3)"
    
    # Build metadata section
    metadata_lines = [
        f"*Generated: {timestamp}*",
        f"*Source: {source}*",
        f"*Tags: {' '.join(tags)}*"
    ]
    
    if feedback:
        # Handle different feedback types (single model vs comparison)
        if 'rating' in feedback:
            metadata_lines.append(f"*Quality Rating: {'⭐' * feedback['rating']} ({feedback['rating']}/5)*")
        elif 'preferred_response' in feedback:
            metadata_lines.append(f"*Model Comparison: Preferred Response {feedback['preferred_response']}*")
            if feedback.get('preference_reasons'):
                reasons = ', '.join(feedback['preference_reasons'])
                metadata_lines.append(f"*Preference Reasons: {reasons}*")
        
        if feedback.get('specific_feedback'):
            metadata_lines.append(f"*Feedback: {feedback['specific_feedback']}*")
    
    content = f"""# {question}

{chr(10).join(metadata_lines)}

---

{response}

---

## 📝 Personal Notes
<!-- Add your own thoughts, connections, and insights here -->

## 🔗 Related Concepts
<!-- Link to other notes in your vault -->

## ❓ Follow-up Questions
<!-- Questions this raised for future exploration -->
"""
    
    return content

def run_model_comparison(question, session_id, gemma_inference, llama_inference, use_kb, 
                        folder_shortcut=None, quick_mode=False):
    """Run sequential model comparison workflow with new options"""
    
    models = [
        ("Gemma 3 12B", gemma_inference, "gemma3:12b"),
        ("Llama 3.1 8B", llama_inference, "llama3.1:8b")
    ]
    
    responses = []
    
    # Generate responses from both models
    for model_name, inference_engine, model_id in models:
        print(f"\n⏳ Generating response from {model_name}...")
        
        try:
            result = inference_engine.generate_response(question, include_knowledge_gap_context=True)
            
            if isinstance(result, dict):
                responses.append({
                    'model_name': model_name,
                    'model_id': model_id,
                    'response': result.get('response', ''),
                    'diagram_path': result.get('diagram_path'),
                    'generation_time': result.get('generation_time', 0),
                    'error': result.get('error', False)
                })
            else:
                # Backward compatibility
                responses.append({
                    'model_name': model_name,
                    'model_id': model_id,
                    'response': str(result),
                    'diagram_path': None,
                    'generation_time': 0
                })
            
            print(f"✅ {model_name} response generated")
            if isinstance(result, dict) and result.get('diagram_path'):
                print(f"🎨 Concept diagram created: {result['diagram_path']}")
            
        except Exception as e:
            print(f"❌ Error generating response from {model_name}: {e}")
            responses.append({
                'model_name': model_name,
                'model_id': model_id,
                'response': f"Error: {str(e)}",
                'diagram_path': None,
                'generation_time': 0,
                'error': True
            })
    
    # Display responses sequentially
    for i, resp in enumerate(responses, 1):
        if not resp.get('error'):
            print(f"\n" + "="*80)
            print(f"RESPONSE {i}: {resp['model_name']} ({resp['generation_time']:.2f}s)")
            if resp.get('diagram_path'):
                print(f"Concept Diagram: {resp['diagram_path']}")
            print("="*80)
            print(resp['response'])
            print("="*80)
        else:
            print(f"\n❌ {resp['model_name']} failed: {resp['response']}")
    
    # Collect comparison feedback
    comparison_feedback = collect_comparison_feedback(responses)
    
    # Save comparison data
    comparison_data = {
        'session_id': session_id,
        'question': question,
        'responses': responses,
        'feedback': comparison_feedback,
        'timestamp': datetime.now().isoformat(),
        'comparison_type': 'sequential'
    }
    
    save_comparison_data(comparison_data)
    
    # Handle save option for preferred response
    if comparison_feedback.get('preferred_response') and comparison_feedback.get('save_preferred'):
        preferred_idx = comparison_feedback['preferred_response'] - 1
        if 0 <= preferred_idx < len(responses) and not responses[preferred_idx].get('error'):
            preferred_response = responses[preferred_idx]['response']
            diagram_path = responses[preferred_idx].get('diagram_path')
            
            # Include diagram info in feedback if available
            if diagram_path:
                comparison_feedback['diagram_path'] = diagram_path
            
            # Get the model name for metadata
            preferred_model = responses[preferred_idx]['model_name']
            
            # Use folder shortcut if available, otherwise interactive save
            if folder_shortcut:
                save_to_obsidian_direct(question, preferred_response, folder_shortcut, 
                                      comparison_feedback, preferred_model)
            else:
                save_to_obsidian(question, preferred_response, comparison_feedback, preferred_model)

def run_single_model_interaction(question, session_id, inference, raw_mode=False, ground_rules_mode=False, 
                                 research_mode=False, quick_mode=False, no_save=False, force_brief=False, force_detailed=False, folder_shortcut=None):
    """Run single model interaction loop with new workflow options"""
    attempt = 1
    feedback_history = []
    
    while True:
        print(f"\n⏳ Generating response (attempt {attempt})...")
        
        try:
            # Prepare generation parameters based on flags
            adaptive_format = not raw_mode  # Raw mode disables educational formatting
            
            # Set complexity preference
            complexity_override = None
            if force_brief:
                complexity_override = 'simple'
            elif force_detailed:
                complexity_override = 'detailed'
            
            # Generate response with appropriate settings
            if raw_mode:
                # Raw mode: minimal parameters, no educational formatting
                result = inference.generate_response(question, 
                                                   adaptive_format=False,
                                                   include_knowledge_gap_context=False,
                                                   use_advanced_templates=False)
            elif research_mode:
                # Research mode: optimized research workflow  
                result = inference.generate_response(question, 
                                                   adaptive_format=False,
                                                   include_knowledge_gap_context=True,
                                                   research_mode=True)
            elif ground_rules_mode:
                # Ground rules mode: research-focused prompting
                result = inference.generate_response(question, 
                                                   adaptive_format=False,
                                                   include_knowledge_gap_context=True,
                                                   use_ground_rules=True)
            else:
                # Normal educational mode
                result = inference.generate_response(question, 
                                                   adaptive_format=adaptive_format,
                                                   include_knowledge_gap_context=True,
                                                   complexity_override=complexity_override)
            
            if isinstance(result, dict):
                response = result.get('response', '')
                diagram_path = result.get('diagram_path')
                generation_time = result.get('generation_time', 0)
                
                if result.get('error'):
                    print(f"❌ Error generating response: {response}")
                    return
            else:
                response = str(result)
                diagram_path = None
                generation_time = 0
                
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return
        
        # Display response
        print("\n" + "="*80)
        print(f"RESPONSE ({generation_time:.2f}s):")
        if diagram_path:
            print(f"Concept Diagram: {diagram_path}")
        print("="*80)
        print(response)
        print("="*80)
        
        # Handle workflow modes
        if no_save:
            # No-save mode: just display and exit
            print("🚫 No-save mode: Response complete, not saving")
            break
        elif quick_mode:
            # Quick mode: auto-save without feedback
            print("⚡ Quick mode: Auto-saving...")
            if folder_shortcut:
                # Use folder shortcut
                save_to_obsidian_direct(question, response, folder_shortcut)
            else:
                # Use default General folder
                save_to_obsidian_direct(question, response, "General")
            break
        else:
            # Normal interactive mode
            feedback = collect_feedback(question, response, attempt)
            feedback_history.append(feedback)
            
            # Save interaction data
            save_interaction_data(session_id, question, response, feedback, attempt)
            
            # Handle user choices
            if feedback['action'] == 'save':
                if folder_shortcut:
                    # Use folder shortcut, skip folder selection
                    save_to_obsidian_direct(question, response, folder_shortcut, feedback)
                else:
                    # Normal interactive save
                    save_to_obsidian(question, response, feedback)
                break
            elif feedback['action'] == 'regenerate':
                # Add feedback context for next generation
                print(f"\n🔄 Regenerating with your feedback...")
                question = enhance_question_with_feedback(question, feedback)
                attempt += 1
                continue
            elif feedback['action'] == 'exit':
                print("👋 Session ended")
                break

def collect_comparison_feedback(responses):
    """Collect feedback for model comparison"""
    print("\n" + "="*60)
    print("⚔️ MODEL COMPARISON FEEDBACK")
    print("="*60)
    
    # Filter out error responses
    valid_responses = [r for r in responses if not r.get('error')]
    
    if len(valid_responses) < 2:
        print("⚠️ Not enough valid responses for comparison")
        return {'error': 'insufficient_responses'}
    
    # Show model options
    print("🎯 Which response do you prefer?")
    for i, resp in enumerate(responses, 1):
        status = "[FAILED]" if resp.get('error') else "[SUCCESS]"
        print(f"  {i}. {resp['model_name']} {status}")
    print("  0. Neither (both have issues)")
    
    # Get preference
    while True:
        try:
            choice = input(f"\nSelect preferred response (0-{len(responses)}): ").strip()
            choice = int(choice)
            if 0 <= choice <= len(responses):
                break
            else:
                print(f"Please enter a number between 0 and {len(responses)}")
        except ValueError:
            print("Please enter a valid number")
    
    feedback = {
        'preferred_response': choice if choice > 0 else None,
        'timestamp': datetime.now().isoformat()
    }
    
    if choice > 0:
        preferred_model = responses[choice - 1]['model_name']
        print(f"\n✅ You preferred: {preferred_model}")
        
        # Ask why they preferred it
        print("\n💭 Why did you prefer this response? (select all that apply)")
        print("  1. Better explanations")
        print("  2. More detailed")
        print("  3. Better examples")
        print("  4. Clearer structure")
        print("  5. Better math notation")
        print("  6. More relevant to my knowledge")
        print("  7. Faster generation")
        print("  8. Other")
        
        reasons = input("Enter numbers (e.g., '1,3,5'): ").strip().split(',')
        reasons = [r.strip() for r in reasons if r.strip().isdigit()]
        feedback['preference_reasons'] = reasons
        
        if '8' in reasons:
            other_reason = input("Please specify other reason: ").strip()
            feedback['other_reason'] = other_reason
        
        # Ask about saving
        save_choice = input(f"\n💾 Save the preferred response to Obsidian? (y/n): ").lower().strip()
        feedback['save_preferred'] = save_choice in ['y', 'yes']
        
    else:
        print("\n⚠️ You found issues with both responses")
        
        issues = input("\n💭 What were the main issues? (optional): ").strip()
        feedback['issues_with_both'] = issues
        
        feedback['save_preferred'] = False
    
    return feedback

def save_comparison_data(comparison_data):
    """Save model comparison data for analysis"""
    comparisons_dir = Path(config.data_dir) / "model_comparisons"
    comparisons_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d')
    comparison_file = comparisons_dir / f"{timestamp}_comparisons.jsonl"
    
    with open(comparison_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(comparison_data, ensure_ascii=False) + '\n')
    
    print(f"\n📊 Comparison data saved for analysis")

if __name__ == "__main__":
    main()