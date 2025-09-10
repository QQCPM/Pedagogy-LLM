#!/bin/bash
# Codebase Cleanup Script
# Removes temporary evaluation and analysis scripts

echo "🧹 Starting codebase cleanup..."

# Create backup directory
mkdir -p archive/temp_scripts
mkdir -p archive/old_data

# Archive temporary scripts before removal
echo "📦 Archiving temporary scripts..."

# Evaluation runner scripts
mv run_all_7_models_comprehensive.py archive/temp_scripts/ 2>/dev/null
mv run_all_models_evaluation.py archive/temp_scripts/ 2>/dev/null
mv run_focused_evaluation.py archive/temp_scripts/ 2>/dev/null
mv run_next_3_questions.py archive/temp_scripts/ 2>/dev/null
mv run_one_more_question.py archive/temp_scripts/ 2>/dev/null
mv run_question_7_clean.py archive/temp_scripts/ 2>/dev/null
mv run_quick_all_models.py archive/temp_scripts/ 2>/dev/null

# Analysis scripts
mv comprehensive_model_analysis.py archive/temp_scripts/ 2>/dev/null
mv comprehensive_text_analysis.py archive/temp_scripts/ 2>/dev/null
mv update_comprehensive_analysis.py archive/temp_scripts/ 2>/dev/null
mv evaluation_metrics_framework.py archive/temp_scripts/ 2>/dev/null

# Data processing scripts
mv extract_ground_rules_responses.py archive/temp_scripts/ 2>/dev/null
mv save_completed_responses.py archive/temp_scripts/ 2>/dev/null
mv save_individual_responses.py archive/temp_scripts/ 2>/dev/null
mv get_actual_responses.py archive/temp_scripts/ 2>/dev/null
mv direct_to_obsidian.py archive/temp_scripts/ 2>/dev/null

# Dashboard and setup scripts
mv create_evaluation_dashboard.py archive/temp_scripts/ 2>/dev/null
mv create_simple_report.py archive/temp_scripts/ 2>/dev/null
mv setup_obsidian_integration.py archive/temp_scripts/ 2>/dev/null
mv generate_obsidian_report.py archive/temp_scripts/ 2>/dev/null

# Test scripts
mv quick_test.py archive/temp_scripts/ 2>/dev/null
mv evaluate_new_models.py archive/temp_scripts/ 2>/dev/null

# Archive old data files
echo "📦 Archiving redundant data files..."
mv data/gemma_eval_results_*.json archive/old_data/ 2>/dev/null
mv data/gpt20b_llama33_70b_evaluation_*_stats.json archive/old_data/ 2>/dev/null
mv data/question_7_evaluation_*.json archive/old_data/ 2>/dev/null
mv data/quick_all_models_evaluation_*.json archive/old_data/ 2>/dev/null

# Remove any .pyc files and __pycache__ directories
echo "🗑️ Cleaning Python cache files..."
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# Count remaining files
echo "✅ Cleanup complete!"
echo "📊 Core Python files remaining: $(find . -name "*.py" -type f | wc -l)"
echo "📊 Archived temporary scripts: $(find archive/temp_scripts -name "*.py" | wc -l)"

echo ""
echo "🔧 CORE FILES PRESERVED:"
echo "   • ask.py - Main CLI interface"
echo "   • ollama_inference.py - Inference engine"  
echo "   • config.py - Configuration"
echo "   • smart_knowledge_base.py - RAG system"
echo "   • obsidian_knowledge_base.py - Knowledge base"
echo "   • ground_rules_prompt.py - Research prompting"
echo "   • adaptive_templates.py - Dynamic templates"
echo "   • concept_flowchart.py - Visualization"
echo "   • generate_final_report.py - Report generation"

echo ""
echo "📁 KEY DATA PRESERVED:"
echo "   • data/all_7_models_comprehensive_20250909_232101.json"
echo "   • data/comprehensive_final_report.json"
echo "   • data/gemma eval.json"
echo "   • COMPREHENSIVE_MODEL_EVALUATION_REPORT.md"

echo ""
echo "🗂️ Temporary files archived in archive/ directory"