#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics Framework
Advanced analysis of model performance, quality, and educational effectiveness
"""
import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class QualityMetrics:
    """Quality assessment metrics for educational responses"""
    mathematical_notation_score: float  # LaTeX usage and correctness
    structural_organization_score: float  # Headers, formatting, flow
    concept_coverage_score: float  # Depth and breadth of explanation
    research_appropriateness_score: float  # Academic rigor level
    obsidian_compliance_score: float  # Markdown formatting quality
    overall_quality_score: float  # Weighted average

@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation"""
    generation_speed: float  # chars/sec
    response_length: int  # character count
    token_efficiency: float  # estimated tokens/sec
    consistency_score: float  # variance in performance
    scalability_score: float  # performance across complexity levels

class EducationalResponseAnalyzer:
    """Analyze educational response quality and effectiveness"""
    
    def __init__(self):
        self.latex_patterns = [
            r'\$[^$]+\$',  # inline math
            r'\$\$[^$]+\$\$',  # block math  
            r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}',  # environments
            r'\\[a-zA-Z]+\{[^}]*\}',  # commands
        ]
        
        self.structure_patterns = {
            'headers': r'^#+\s+.+$',
            'bullet_points': r'^\s*[-*+]\s+.+$',
            'numbered_lists': r'^\s*\d+\.\s+.+$',
            'code_blocks': r'```[^`]*```',
            'emphasis': r'\*\*[^*]+\*\*|\*[^*]+\*',
            'tables': r'\|[^|]*\|',
        }
        
        self.research_indicators = [
            'theorem', 'lemma', 'proof', 'definition', 'hypothesis',
            'research', 'study', 'analysis', 'framework', 'methodology',
            'literature', 'empirical', 'theoretical', 'experimental',
            'model', 'algorithm', 'optimization', 'mathematical', 'statistical'
        ]
        
    def analyze_mathematical_notation(self, text: str) -> float:
        """Analyze mathematical notation quality and usage"""
        total_math = 0
        correct_latex = 0
        
        for pattern in self.latex_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            total_math += len(matches)
            
            # Check for correct LaTeX syntax (basic validation)
            for match in matches:
                if self._is_valid_latex(match):
                    correct_latex += 1
        
        if total_math == 0:
            # No math found - check if question likely needs math
            if any(term in text.lower() for term in ['formula', 'equation', 'calculate', 'mathematical']):
                return 0.3  # Expected math but none found
            else:
                return 0.8  # No math needed for this question
        
        accuracy_score = correct_latex / total_math
        usage_score = min(total_math / 10, 1.0)  # Normalize to reasonable usage
        
        return (accuracy_score * 0.7 + usage_score * 0.3)
    
    def _is_valid_latex(self, latex_text: str) -> bool:
        """Basic LaTeX syntax validation"""
        # Remove outer $ symbols
        content = latex_text.strip('$').strip()
        
        # Check for balanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        
        if open_braces != close_braces:
            return False
        
        # Check for common LaTeX commands
        common_commands = ['frac', 'sqrt', 'sum', 'int', 'alpha', 'beta', 'gamma', 
                          'theta', 'lambda', 'mu', 'sigma', 'pi', 'partial']
        
        has_valid_commands = any(f'\\{cmd}' in content for cmd in common_commands)
        has_basic_math = any(char in content for char in ['=', '+', '-', '*', '/', '^', '_'])
        
        return has_valid_commands or has_basic_math
    
    def analyze_structural_organization(self, text: str) -> float:
        """Analyze structural organization and formatting"""
        scores = {}
        lines = text.split('\n')
        
        # Header hierarchy
        headers = [line for line in lines if re.match(self.structure_patterns['headers'], line)]
        scores['headers'] = min(len(headers) / 5, 1.0) * 0.9  # Prefer some headers
        
        # Lists and organization
        bullets = len([line for line in lines if re.match(self.structure_patterns['bullet_points'], line)])
        numbers = len([line for line in lines if re.match(self.structure_patterns['numbered_lists'], line)])
        scores['lists'] = min((bullets + numbers) / 10, 1.0) * 0.8
        
        # Code examples (for technical content)
        code_blocks = len(re.findall(self.structure_patterns['code_blocks'], text, re.DOTALL))
        scores['code'] = min(code_blocks / 3, 1.0) * 0.7
        
        # Emphasis and formatting
        emphasis = len(re.findall(self.structure_patterns['emphasis'], text))
        scores['emphasis'] = min(emphasis / 8, 1.0) * 0.6
        
        # Tables for comparisons
        tables = len([line for line in lines if re.match(self.structure_patterns['tables'], line)])
        scores['tables'] = min(tables / 5, 1.0) * 0.8 if tables > 0 else 0.0
        
        # Overall flow (paragraph structure)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_score = min(len(paragraphs) / 8, 1.0) * 0.7
        
        return np.mean([scores['headers'], scores['lists'], scores['code'], 
                       scores['emphasis'], scores['tables'], paragraph_score])
    
    def analyze_concept_coverage(self, text: str, domain: str) -> float:
        """Analyze depth and breadth of concept coverage"""
        
        # Domain-specific keywords
        domain_keywords = {
            'deep_learning': ['neural', 'network', 'layer', 'activation', 'gradient', 'backpropagation', 
                             'optimizer', 'loss', 'training', 'learning rate', 'regularization'],
            'machine_learning': ['algorithm', 'model', 'training', 'validation', 'overfitting', 
                                'feature', 'classification', 'regression', 'clustering'],
            'quantum_computing': ['qubit', 'superposition', 'entanglement', 'quantum', 'gate', 
                                'measurement', 'algorithm', 'circuit'],
            'materials_science': ['polymer', 'structure', 'property', 'synthesis', 'characterization',
                                'molecular', 'material', 'hydrogel'],
            'numerical_methods': ['computation', 'algorithm', 'approximation', 'error', 'convergence',
                                 'iteration', 'numerical', 'discrete'],
            'scientific_method': ['hypothesis', 'experiment', 'data', 'analysis', 'method', 'research',
                                 'causality', 'inference', 'discovery'],
            'earth_science': ['geological', 'evolution', 'formation', 'earth', 'time', 'process',
                            'history', 'planet', 'atmosphere']
        }
        
        # Get relevant keywords for domain
        keywords = domain_keywords.get(domain, [])
        text_lower = text.lower()
        
        if not keywords:
            # Generic concept coverage for unknown domains
            concept_density = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)) / len(text.split())
            return min(concept_density * 10, 1.0)
        
        # Calculate keyword coverage
        keyword_coverage = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
        
        # Calculate concept explanation depth
        explanatory_phrases = ['defined as', 'refers to', 'means that', 'involves', 'consists of',
                              'characterized by', 'example of', 'such as', 'including', 'namely']
        
        explanation_count = sum(1 for phrase in explanatory_phrases if phrase in text_lower)
        explanation_score = min(explanation_count / 5, 1.0)
        
        # Calculate technical depth
        technical_terms = len(re.findall(r'\b[a-z]+(?:tion|ity|ism|ment|ness|ing|ed)\b', text_lower))
        technical_score = min(technical_terms / 20, 1.0)
        
        return (keyword_coverage * 0.4 + explanation_score * 0.3 + technical_score * 0.3)
    
    def analyze_research_appropriateness(self, text: str) -> float:
        """Analyze research-level appropriateness"""
        text_lower = text.lower()
        
        # Research vocabulary usage
        research_vocab_count = sum(1 for term in self.research_indicators if term in text_lower)
        vocab_score = min(research_vocab_count / 10, 1.0)
        
        # Citation-like patterns
        citation_patterns = [
            r'\([A-Za-z]+\s+et\s+al\.,?\s+\d{4}\)',  # (Author et al., 2020)
            r'\([A-Za-z]+\s+\&\s+[A-Za-z]+,?\s+\d{4}\)',  # (Smith & Jones, 2020)
            r'according to [A-Z][a-z]+',  # according to Smith
            r'research shows',
            r'studies indicate',
            r'literature suggests'
        ]
        
        citation_count = sum(len(re.findall(pattern, text)) for pattern in citation_patterns)
        citation_score = min(citation_count / 3, 1.0)
        
        # Formal academic language
        formal_indicators = ['furthermore', 'moreover', 'consequently', 'therefore', 'nevertheless',
                           'however', 'specifically', 'particularly', 'significantly', 'notably']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        formal_score = min(formal_count / 5, 1.0)
        
        # Technical precision (longer sentences, complex structures)
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()])
        complexity_score = min((avg_sentence_length - 10) / 15, 1.0) if avg_sentence_length > 10 else 0
        
        return (vocab_score * 0.3 + citation_score * 0.2 + formal_score * 0.2 + complexity_score * 0.3)
    
    def analyze_obsidian_compliance(self, text: str) -> float:
        """Analyze Obsidian markdown compliance"""
        scores = []
        
        # Header structure (should use # for headers)
        proper_headers = len(re.findall(r'^#+\s+[^\n]+$', text, re.MULTILINE))
        improper_headers = len(re.findall(r'^[A-Z][^a-z\n]*:?\s*$', text, re.MULTILINE))
        header_score = proper_headers / max(proper_headers + improper_headers, 1)
        scores.append(header_score)
        
        # Math notation (should use $ delimiters)
        proper_math = len(re.findall(r'\$[^$]+\$', text))
        improper_math = len(re.findall(r'(?<!\$)[a-zA-Z]_[a-zA-Z0-9]|[a-zA-Z]\^[a-zA-Z0-9](?!\$)', text))
        math_score = proper_math / max(proper_math + improper_math, 1) if proper_math + improper_math > 0 else 1.0
        scores.append(math_score)
        
        # List formatting
        proper_lists = len(re.findall(r'^[-*+]\s+', text, re.MULTILINE))
        improper_lists = len(re.findall(r'^\s*[•·]\s+', text, re.MULTILINE))
        list_score = proper_lists / max(proper_lists + improper_lists, 1) if proper_lists + improper_lists > 0 else 1.0
        scores.append(list_score)
        
        # Code blocks
        proper_code = len(re.findall(r'```[^`]*```', text, re.DOTALL))
        improper_code = len(re.findall(r'^\s{4,}[^\n]*$', text, re.MULTILINE))
        code_score = 1.0 if improper_code == 0 else proper_code / max(proper_code + improper_code, 1)
        scores.append(code_score)
        
        # Links and references (Obsidian-style [[links]])
        obsidian_links = len(re.findall(r'\[\[[^\]]+\]\]', text))
        md_links = len(re.findall(r'\[[^\]]*\]\([^)]+\)', text))
        link_score = 1.0 if obsidian_links + md_links == 0 else 1.0  # Both are acceptable
        scores.append(link_score)
        
        return np.mean(scores)
    
    def calculate_quality_metrics(self, text: str, domain: str) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        
        math_score = self.analyze_mathematical_notation(text)
        structure_score = self.analyze_structural_organization(text)
        coverage_score = self.analyze_concept_coverage(text, domain)
        research_score = self.analyze_research_appropriateness(text)
        obsidian_score = self.analyze_obsidian_compliance(text)
        
        # Weighted overall score
        weights = {
            'math': 0.20,
            'structure': 0.25,
            'coverage': 0.25,
            'research': 0.20,
            'obsidian': 0.10
        }
        
        overall = (math_score * weights['math'] +
                  structure_score * weights['structure'] +
                  coverage_score * weights['coverage'] +
                  research_score * weights['research'] +
                  obsidian_score * weights['obsidian'])
        
        return QualityMetrics(
            mathematical_notation_score=math_score,
            structural_organization_score=structure_score,
            concept_coverage_score=coverage_score,
            research_appropriateness_score=research_score,
            obsidian_compliance_score=obsidian_score,
            overall_quality_score=overall
        )

class ComprehensiveEvaluationFramework:
    """Complete evaluation framework for educational LLM comparison"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.analyzer = EducationalResponseAnalyzer()
        self.results_df = self.load_evaluation_data()
        
    def load_evaluation_data(self) -> pd.DataFrame:
        """Load all evaluation results and enhance with quality metrics"""
        all_results = []
        
        # Load from JSON files
        for json_file in self.data_dir.glob("*evaluation*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    
                for result in results:
                    if not result.get('error', False):
                        # Calculate quality metrics
                        quality = self.analyzer.calculate_quality_metrics(
                            result['response'], result['domain']
                        )
                        
                        # Create enhanced record
                        enhanced_result = {
                            'source_file': str(json_file),
                            'question_id': result['question_id'],
                            'question': result['question'],
                            'domain': result['domain'],
                            'model': result['model'],
                            'model_name': result['model_name'],
                            'approach': result['approach'],
                            'response': result['response'],
                            'response_length': len(result['response']),
                            'generation_time': result['metrics']['generation_time'],
                            'char_count': result['metrics']['char_count'],
                            'word_count': result['metrics']['word_count'],
                            'chars_per_second': result['metrics']['chars_per_second'],
                            'timestamp': result['timestamp'],
                            # Quality metrics
                            'math_notation_score': quality.mathematical_notation_score,
                            'structure_score': quality.structural_organization_score,
                            'coverage_score': quality.concept_coverage_score,
                            'research_score': quality.research_appropriateness_score,
                            'obsidian_score': quality.obsidian_compliance_score,
                            'overall_quality_score': quality.overall_quality_score,
                        }
                        all_results.append(enhanced_result)
                        
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        if self.results_df.empty:
            return {"error": "No data available"}
        
        report = {
            "metadata": {
                "report_generated": datetime.now().isoformat(),
                "total_evaluations": len(self.results_df),
                "unique_models": self.results_df['model_name'].nunique(),
                "unique_questions": self.results_df['question_id'].nunique(),
                "unique_domains": self.results_df['domain'].nunique(),
                "evaluation_period": {
                    "start": self.results_df['timestamp'].min(),
                    "end": self.results_df['timestamp'].max()
                }
            },
            "model_rankings": self.rank_models(),
            "approach_analysis": self.analyze_approaches(),
            "domain_performance": self.analyze_domain_performance(),
            "quality_analysis": self.analyze_quality_metrics(),
            "statistical_significance": self.test_statistical_significance(),
            "recommendations": self.generate_recommendations()
        }
        
        return report
    
    def rank_models(self) -> Dict:
        """Rank models across different metrics"""
        rankings = {}
        
        # Performance rankings
        perf_metrics = ['chars_per_second', 'generation_time', 'char_count']
        for metric in perf_metrics:
            if metric == 'generation_time':
                # Lower is better for generation time
                rankings[f'{metric}_ranking'] = (
                    self.results_df.groupby('model_name')[metric]
                    .mean().sort_values().to_dict()
                )
            else:
                # Higher is better for other metrics
                rankings[f'{metric}_ranking'] = (
                    self.results_df.groupby('model_name')[metric]
                    .mean().sort_values(ascending=False).to_dict()
                )
        
        # Quality rankings
        quality_metrics = ['overall_quality_score', 'research_score', 'coverage_score']
        for metric in quality_metrics:
            rankings[f'{metric}_ranking'] = (
                self.results_df.groupby('model_name')[metric]
                .mean().sort_values(ascending=False).to_dict()
            )
        
        # Overall composite score
        composite_scores = {}
        for model in self.results_df['model_name'].unique():
            model_data = self.results_df[self.results_df['model_name'] == model]
            
            # Normalize and combine metrics (higher is better)
            speed_norm = model_data['chars_per_second'].mean() / self.results_df['chars_per_second'].max()
            quality_norm = model_data['overall_quality_score'].mean()
            length_norm = model_data['char_count'].mean() / self.results_df['char_count'].max()
            
            composite_scores[model] = (speed_norm * 0.3 + quality_norm * 0.5 + length_norm * 0.2)
        
        rankings['composite_ranking'] = dict(sorted(composite_scores.items(), 
                                                   key=lambda x: x[1], reverse=True))
        
        return rankings
    
    def analyze_approaches(self) -> Dict:
        """Analyze Raw vs Ground Rules approaches"""
        approach_analysis = {}
        
        for approach in self.results_df['approach'].unique():
            approach_data = self.results_df[self.results_df['approach'] == approach]
            
            approach_analysis[approach] = {
                'avg_response_length': approach_data['char_count'].mean(),
                'avg_generation_time': approach_data['generation_time'].mean(),
                'avg_quality_score': approach_data['overall_quality_score'].mean(),
                'avg_research_score': approach_data['research_score'].mean(),
                'avg_coverage_score': approach_data['coverage_score'].mean(),
                'consistency': approach_data['overall_quality_score'].std(),
                'sample_size': len(approach_data)
            }
        
        # Compare approaches if both exist
        if 'raw' in approach_analysis and 'ground_rules' in approach_analysis:
            raw = approach_analysis['raw']
            gr = approach_analysis['ground_rules']
            
            approach_analysis['comparison'] = {
                'length_improvement': gr['avg_response_length'] / raw['avg_response_length'],
                'quality_improvement': gr['avg_quality_score'] / raw['avg_quality_score'],
                'time_cost': gr['avg_generation_time'] / raw['avg_generation_time'],
                'research_improvement': gr['avg_research_score'] / raw['avg_research_score'],
            }
        
        return approach_analysis
    
    def analyze_domain_performance(self) -> Dict:
        """Analyze performance across different domains"""
        domain_analysis = {}
        
        for domain in self.results_df['domain'].unique():
            domain_data = self.results_df[self.results_df['domain'] == domain]
            
            domain_analysis[domain] = {
                'avg_quality_score': domain_data['overall_quality_score'].mean(),
                'avg_response_length': domain_data['char_count'].mean(),
                'avg_generation_speed': domain_data['chars_per_second'].mean(),
                'best_model': domain_data.loc[domain_data['overall_quality_score'].idxmax(), 'model_name'],
                'fastest_model': domain_data.loc[domain_data['chars_per_second'].idxmax(), 'model_name'],
                'sample_size': len(domain_data),
                'quality_variance': domain_data['overall_quality_score'].std()
            }
        
        return domain_analysis
    
    def analyze_quality_metrics(self) -> Dict:
        """Detailed analysis of quality metrics"""
        quality_metrics = ['math_notation_score', 'structure_score', 'coverage_score', 
                          'research_score', 'obsidian_score']
        
        analysis = {}
        
        for metric in quality_metrics:
            analysis[metric] = {
                'overall_mean': self.results_df[metric].mean(),
                'overall_std': self.results_df[metric].std(),
                'by_model': self.results_df.groupby('model_name')[metric].agg(['mean', 'std']).to_dict(),
                'by_approach': self.results_df.groupby('approach')[metric].agg(['mean', 'std']).to_dict(),
                'by_domain': self.results_df.groupby('domain')[metric].agg(['mean', 'std']).to_dict()
            }
        
        # Correlation analysis
        correlation_matrix = self.results_df[quality_metrics + ['chars_per_second', 'char_count']].corr()
        analysis['correlations'] = correlation_matrix.to_dict()
        
        return analysis
    
    def test_statistical_significance(self) -> Dict:
        """Test statistical significance of differences"""
        significance_tests = {}
        
        # Test approach differences
        if len(self.results_df['approach'].unique()) >= 2:
            raw_data = self.results_df[self.results_df['approach'] == 'raw']
            gr_data = self.results_df[self.results_df['approach'] == 'ground_rules']
            
            if len(raw_data) > 0 and len(gr_data) > 0:
                # T-test for quality scores
                t_stat, p_value = stats.ttest_ind(
                    gr_data['overall_quality_score'], 
                    raw_data['overall_quality_score']
                )
                
                significance_tests['approach_quality_difference'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': (gr_data['overall_quality_score'].mean() - 
                                  raw_data['overall_quality_score'].mean()) / 
                                 np.sqrt((gr_data['overall_quality_score'].var() + 
                                        raw_data['overall_quality_score'].var()) / 2)
                }
        
        # Test model differences
        if len(self.results_df['model_name'].unique()) >= 2:
            model_names = self.results_df['model_name'].unique()
            
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    model1_data = self.results_df[self.results_df['model_name'] == model1]
                    model2_data = self.results_df[self.results_df['model_name'] == model2]
                    
                    if len(model1_data) > 0 and len(model2_data) > 0:
                        t_stat, p_value = stats.ttest_ind(
                            model1_data['overall_quality_score'],
                            model2_data['overall_quality_score']
                        )
                        
                        significance_tests[f'{model1}_vs_{model2}'] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
        
        return significance_tests
    
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if self.results_df.empty:
            return ["No data available for analysis"]
        
        # Speed recommendations
        fastest_model = self.results_df.loc[self.results_df['chars_per_second'].idxmax(), 'model_name']
        recommendations.append(f"For speed-critical applications: Use {fastest_model}")
        
        # Quality recommendations
        highest_quality = self.results_df.loc[self.results_df['overall_quality_score'].idxmax(), 'model_name']
        recommendations.append(f"For highest quality responses: Use {highest_quality}")
        
        # Approach recommendations
        if len(self.results_df['approach'].unique()) >= 2:
            approach_comparison = self.analyze_approaches()
            if 'comparison' in approach_comparison:
                comp = approach_comparison['comparison']
                if comp['quality_improvement'] > 1.1:
                    recommendations.append(f"Ground rules approach provides {comp['quality_improvement']:.1f}x quality improvement")
                if comp['length_improvement'] > 1.5:
                    recommendations.append(f"Use ground rules for comprehensive responses ({comp['length_improvement']:.1f}x longer)")
        
        # Domain-specific recommendations
        domain_perf = self.analyze_domain_performance()
        for domain, metrics in domain_perf.items():
            if metrics['sample_size'] >= 2:
                recommendations.append(f"For {domain}: {metrics['best_model']} performs best")
        
        # Quality improvement recommendations
        avg_quality = self.results_df['overall_quality_score'].mean()
        if avg_quality < 0.7:
            recommendations.append("Consider implementing additional quality enhancement techniques")
        
        return recommendations
    
    def save_comprehensive_report(self, filename: str = "comprehensive_evaluation_report.json"):
        """Save comprehensive evaluation report"""
        report = self.generate_comprehensive_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 Comprehensive report saved to: {filename}")
        return report

def main():
    """Generate comprehensive evaluation analysis"""
    print("🔬 Generating Comprehensive Evaluation Analysis...")
    
    framework = ComprehensiveEvaluationFramework()
    
    if framework.results_df.empty:
        print("❌ No evaluation data found")
        return
    
    print(f"📊 Analyzing {len(framework.results_df)} evaluations")
    print(f"🤖 Models: {', '.join(framework.results_df['model_name'].unique())}")
    print(f"🏷️  Domains: {', '.join(framework.results_df['domain'].unique())}")
    
    # Generate comprehensive report
    report = framework.save_comprehensive_report()
    
    # Print key findings
    print("\n🏆 TOP PERFORMERS:")
    if 'model_rankings' in report:
        composite = report['model_rankings']['composite_ranking']
        for i, (model, score) in enumerate(list(composite.items())[:3], 1):
            print(f"   {i}. {model} (score: {score:.3f})")
    
    print("\n💡 KEY RECOMMENDATIONS:")
    for i, rec in enumerate(report.get('recommendations', [])[:5], 1):
        print(f"   {i}. {rec}")
    
    # Save enhanced dataset
    framework.results_df.to_csv('data/enhanced_evaluation_results.csv', index=False)
    print(f"\n📁 Enhanced dataset saved to: data/enhanced_evaluation_results.csv")
    
    print("\n✅ Comprehensive analysis completed!")

if __name__ == "__main__":
    main()