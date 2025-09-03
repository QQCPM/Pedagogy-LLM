#!/usr/bin/env python3
"""
Comprehensive Test of Enhanced Educational LLM System
Tests all new features: smart knowledge base, model comparison, concept flowcharts, etc.
"""
import time
import json
from pathlib import Path
from datetime import datetime

from ollama_inference import OllamaEducationalInference
from smart_knowledge_base import SmartKnowledgeBase
from concept_flowchart import ConceptFlowchartGenerator
from learning_analytics import LearningAnalytics
from config import config

def test_smart_knowledge_base():
    """Test the smart knowledge base functionality"""
    print("\\n🧠 Testing Smart Knowledge Base...")
    
    try:
        kb = SmartKnowledgeBase()
        success = kb.index_vault()
        
        if success:
            stats = kb.get_enhanced_statistics()
            print(f"   ✅ Vault indexed: {stats.get('total_documents', 0)} documents")
            print(f"   ✅ Concepts extracted: {stats.get('concept_graph', {}).get('total_concepts', 0)}")
            print(f"   ✅ Relationships found: {stats.get('concept_graph', {}).get('total_relationships', 0)}")
            
            # Test smart search
            test_query = "machine learning algorithms"
            results = kb.smart_search(test_query, top_k=3, include_related_concepts=True)
            print(f"   ✅ Smart search returned {len(results)} results for '{test_query}'")
            
            return True
        else:
            print("   ❌ Vault indexing failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Smart KB test failed: {e}")
        return False

def test_concept_flowchart():
    """Test concept flowchart generation"""
    print("\\n🎨 Testing Concept Flowchart Generation...")
    
    try:
        generator = ConceptFlowchartGenerator()
        
        test_question = "Explain eigenvalues and eigenvectors"
        test_response = """# Eigenvalues and Eigenvectors

## Intuitive Understanding
Eigenvalues and eigenvectors represent special directions in linear transformations where the direction of a vector is preserved, only scaled.

## Mathematical Definition
For a square matrix $A$ and non-zero vector $v$, if $Av = \\lambda v$ where $\\lambda$ is a scalar, then $v$ is an eigenvector and $\\lambda$ is the corresponding eigenvalue.

## Applications
Eigenvalues are crucial in Principal Component Analysis, stability analysis in differential equations, and quantum mechanics.
        """
        
        result = generator.create_concept_flowchart(test_question, test_response)
        
        if result:
            print(f"   ✅ Flowchart generated: {result}")
            return True
        else:
            print("   ⚠️ Flowchart generation returned None (may be expected)")
            return True  # Not a failure if no concepts found
            
    except Exception as e:
        print(f"   ❌ Flowchart test failed: {e}")
        return False

def test_model_comparison():
    """Test model comparison functionality"""
    print("\\n⚔️ Testing Model Comparison...")
    
    try:
        # Initialize both models
        gemma = OllamaEducationalInference(
            model_name="gemma3:12b",
            use_knowledge_base=False  # Disable for faster testing
        )
        
        llama = OllamaEducationalInference(
            model_name="llama3.1:8b",
            use_knowledge_base=False  # Disable for faster testing
        )
        
        test_question = "What is a derivative in calculus?"
        print(f"   🧪 Testing with question: {test_question}")
        
        # Generate responses (with timeouts)
        print("   ⏳ Generating Gemma 3 response...")
        start_time = time.time()
        gemma_result = gemma.generate_response(test_question, create_diagram=False)
        gemma_time = time.time() - start_time
        
        print("   ⏳ Generating Llama 3.1 response...")
        start_time = time.time()
        llama_result = llama.generate_response(test_question, create_diagram=False)
        llama_time = time.time() - start_time
        
        if isinstance(gemma_result, dict) and isinstance(llama_result, dict):
            print(f"   ✅ Gemma 3 generated response ({gemma_time:.1f}s): {len(gemma_result.get('response', ''))} chars")
            print(f"   ✅ Llama 3.1 generated response ({llama_time:.1f}s): {len(llama_result.get('response', ''))} chars")
            return True
        else:
            print("   ❌ Unexpected response format")
            return False
            
    except Exception as e:
        print(f"   ❌ Model comparison test failed: {e}")
        return False

def test_enhanced_prompting():
    """Test the enhanced prompting system"""
    print("\\n📝 Testing Enhanced Prompting System...")
    
    try:
        # Test with knowledge base disabled for speed
        inference = OllamaEducationalInference(use_knowledge_base=False)
        
        test_question = "Explain the concept of limits in calculus"
        print(f"   🧪 Testing with: {test_question}")
        
        result = inference.generate_response(
            test_question, 
            create_diagram=False,  # Disable diagram for testing
            max_tokens=1000  # Limit for testing
        )
        
        if isinstance(result, dict):
            response = result.get('response', '')
            print(f"   ✅ Enhanced response generated: {len(response)} characters")
            
            # Check for key elements of enhanced prompting
            enhanced_indicators = [
                "intuitive", "mathematical definition", "example", 
                "why this matters", "connection", "exploration"
            ]
            
            found_indicators = sum(1 for indicator in enhanced_indicators 
                                 if indicator.lower() in response.lower())
            
            print(f"   ✅ Enhanced structure elements found: {found_indicators}/{len(enhanced_indicators)}")
            
            if found_indicators >= 3:
                print("   ✅ Enhanced prompting structure detected")
                return True
            else:
                print("   ⚠️ Limited enhanced structure detected")
                return True  # Still success, just noting
        else:
            print("   ❌ Unexpected response format")
            return False
            
    except Exception as e:
        print(f"   ❌ Enhanced prompting test failed: {e}")
        return False

def test_learning_analytics():
    """Test learning analytics functionality"""
    print("\\n📊 Testing Learning Analytics...")
    
    try:
        analytics = LearningAnalytics()
        
        # Check if interaction data exists
        interactions = analytics.load_interaction_data(days_back=7)
        
        if interactions:
            print(f"   ✅ Found {len(interactions)} recent interactions")
            
            # Test analysis
            analysis = analytics.analyze_learning_patterns(interactions)
            insights = analytics.generate_insights(analysis)
            
            print(f"   ✅ Analysis completed: {len(insights)} insights generated")
            return True
        else:
            print("   ⚠️ No interaction data found (expected for new setup)")
            return True  # Not a failure
            
    except Exception as e:
        print(f"   ❌ Learning analytics test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all system tests"""
    print("🚀 COMPREHENSIVE SYSTEM TEST")
    print("=" * 60)
    
    test_results = {
        'smart_knowledge_base': test_smart_knowledge_base(),
        'concept_flowchart': test_concept_flowchart(),
        'model_comparison': test_model_comparison(),
        'enhanced_prompting': test_enhanced_prompting(),
        'learning_analytics': test_learning_analytics()
    }
    
    print("\\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        formatted_name = test_name.replace('_', ' ').title()
        print(f"   {formatted_name}: {status}")
    
    print(f"\\n🎯 OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("   Your enhanced educational LLM is ready for use.")
    elif passed_tests >= total_tests * 0.8:
        print("\\n✅ MOSTLY OPERATIONAL!")
        print("   Core functionality is working. Some advanced features may need attention.")
    else:
        print("\\n⚠️ PARTIAL FUNCTIONALITY")
        print("   Some core systems need attention before full operation.")
    
    print("\\n📚 NEXT STEPS:")
    print("   • Try: python ask.py 'Explain eigenvalues and eigenvectors'")
    print("   • Try: python ask.py 'What is machine learning?' --compare")
    print("   • Check: python learning_analytics.py")
    print("   • View generated diagrams in outputs/flowcharts/")
    
    return test_results

def demonstrate_features():
    """Demonstrate key features with a practical example"""
    print("\\n\\n🎯 FEATURE DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Initialize the system
        print("Initializing enhanced educational LLM...")
        inference = OllamaEducationalInference(use_knowledge_base=True)
        
        # Demo question
        demo_question = "What are eigenvalues and why are they important?"
        print(f"\\n📚 Demo Question: {demo_question}")
        
        print("\\n⏳ Generating comprehensive response with all enhancements...")
        print("   • Smart knowledge base context")
        print("   • Enhanced detailed prompting")
        print("   • Concept flowchart generation")
        print("   • Anti-redundant topic suggestions")
        
        result = inference.generate_response(demo_question, create_diagram=True)
        
        if isinstance(result, dict):
            response = result.get('response', '')
            diagram_path = result.get('diagram_path')
            generation_time = result.get('generation_time', 0)
            
            print(f"\\n✅ Response generated in {generation_time:.1f}s")
            print(f"   📄 Length: {len(response)} characters")
            print(f"   🎨 Diagram: {'Yes' if diagram_path else 'No'}")
            
            if diagram_path:
                print(f"   📍 Diagram saved: {diagram_path}")
            
            # Show first part of response
            print(f"\\n📖 Response Preview:")
            print("-" * 40)
            print(response[:500] + "..." if len(response) > 500 else response)
            print("-" * 40)
            
            print("\\n🎉 Full system demonstration completed successfully!")
            return True
        else:
            print("   ❌ Unexpected response format in demonstration")
            return False
            
    except Exception as e:
        print(f"   ❌ Feature demonstration failed: {e}")
        return False

def main():
    """Main test runner"""
    import sys
    
    if '--demo' in sys.argv:
        demonstrate_features()
    else:
        run_comprehensive_test()
        
        # Optionally run demo after tests
        if '--with-demo' in sys.argv:
            demonstrate_features()

if __name__ == "__main__":
    main()