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
OBSIDIAN_VAULT = Path("/Users/quangnguyen/Downloads/hello")

# Folder structure for research notes
FOLDERS = {
    "1": "Mathematics",
    "2": "AI-ML", 
    "3": "Physics",
    "4": "Computer-Science",
    "5": "General"
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py 'Your question here' [--no-kb] [--compare]")
        print("Example: python ask.py 'Explain eigenvalues and eigenvectors'")
        print("Options:")
        print("  --no-kb    Disable knowledge base integration for this question")
        print("  --compare  Compare responses from Gemma 3 and Llama 3.1")
        return

    # Parse command line arguments
    args = sys.argv[1:]
    compare_mode = False
    use_kb = True
    
    if '--compare' in args:
        compare_mode = True
        args.remove('--compare')
    
    if '--no-kb' in args:
        use_kb = False
        args.remove('--no-kb')
        print("📚 Knowledge base disabled for this session")
    
    question = " ".join(args)
    session_id = str(uuid.uuid4())[:8]  # Short session ID for tracking
    
    if compare_mode:
        print(f"⚔️ Model Comparison Mode: Gemma 3 vs Llama 3.1")
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
            # Single model mode (default Gemma 3)
            inference = OllamaEducationalInference(
                vault_path=str(OBSIDIAN_VAULT),
                use_knowledge_base=use_kb
            )
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return
    
    if compare_mode:
        # Model comparison workflow
        run_model_comparison(question, session_id, gemma_inference, llama_inference, use_kb)
    else:
        # Single model interaction loop
        run_single_model_interaction(question, session_id, inference)

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

def run_model_comparison(question, session_id, gemma_inference, llama_inference, use_kb):
    """Run sequential model comparison workflow"""
    
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
            save_to_obsidian(question, preferred_response, comparison_feedback, preferred_model)

def run_single_model_interaction(question, session_id, inference):
    """Run single model interaction loop"""
    attempt = 1
    feedback_history = []
    
    while True:
        print(f"\n⏳ Generating response (attempt {attempt})...")
        
        try:
            result = inference.generate_response(question, include_knowledge_gap_context=True)
            
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
        
        # Enhanced feedback collection
        feedback = collect_feedback(question, response, attempt)
        feedback_history.append(feedback)
        
        # Save interaction data
        save_interaction_data(session_id, question, response, feedback, attempt)
        
        # Handle user choices
        if feedback['action'] == 'save':
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