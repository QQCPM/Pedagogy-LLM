#!/usr/bin/env python3
"""
Create Interactive Evaluation Dashboard
Visualize model performance, compare approaches, and analyze results
"""
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluationDashboard:
    """Interactive dashboard for model evaluation results"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.results = self.load_all_results()
        
    def load_all_results(self) -> pd.DataFrame:
        """Load all evaluation results from JSON files"""
        all_results = []
        
        # Load evaluation files
        for json_file in self.data_dir.glob("*evaluation*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    
                for result in results:
                    if not result.get('error', False):
                        # Flatten metrics into main record
                        flattened = {
                            'file_source': str(json_file),
                            'question_id': result['question_id'],
                            'question': result['question'],
                            'domain': result['domain'],
                            'model': result['model'],
                            'model_name': result['model_name'],
                            'approach': result['approach'],
                            'response_length': len(result['response']),
                            'generation_time': result['metrics']['generation_time'],
                            'char_count': result['metrics']['char_count'],
                            'word_count': result['metrics']['word_count'],
                            'chars_per_second': result['metrics']['chars_per_second'],
                            'timestamp': result['timestamp']
                        }
                        all_results.append(flattened)
                        
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        if not all_results:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'question_id', 'question', 'domain', 'model', 'model_name', 
                'approach', 'response_length', 'generation_time', 'char_count',
                'word_count', 'chars_per_second', 'timestamp'
            ])
            
        return pd.DataFrame(all_results)
    
    def create_performance_comparison(self) -> go.Figure:
        """Create performance comparison visualization"""
        if self.results.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Generation Speed by Model', 'Response Length by Approach',
                'Speed vs Length Trade-off', 'Performance by Domain'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # 1. Generation Speed by Model
        speed_data = self.results.groupby(['model_name', 'approach'])['chars_per_second'].mean().reset_index()
        
        for approach in speed_data['approach'].unique():
            data = speed_data[speed_data['approach'] == approach]
            fig.add_trace(
                go.Bar(
                    x=data['model_name'],
                    y=data['chars_per_second'],
                    name=f'{approach.title()} Speed',
                    text=[f'{x:.0f}' for x in data['chars_per_second']],
                    textposition='auto',
                ),
                row=1, col=1
            )
        
        # 2. Response Length by Approach
        length_data = self.results.groupby(['model_name', 'approach'])['char_count'].mean().reset_index()
        
        for approach in length_data['approach'].unique():
            data = length_data[length_data['approach'] == approach]
            fig.add_trace(
                go.Bar(
                    x=data['model_name'],
                    y=data['char_count'],
                    name=f'{approach.title()} Length',
                    text=[f'{x:.0f}' for x in data['char_count']],
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Speed vs Length Trade-off (scatter)
        for model in self.results['model_name'].unique():
            model_data = self.results[self.results['model_name'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['char_count'],
                    y=model_data['chars_per_second'],
                    mode='markers',
                    name=f'{model} Trade-off',
                    text=model_data['approach'],
                    marker=dict(size=8),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Performance by Domain
        domain_perf = self.results.groupby('domain')['chars_per_second'].mean().sort_values(ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=domain_perf.values,
                y=domain_perf.index,
                orientation='h',
                name='Domain Performance',
                text=[f'{x:.0f}' for x in domain_perf.values],
                textposition='auto',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Model Performance Analysis Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Chars/Second", row=1, col=1)
        
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_yaxes(title_text="Response Length (chars)", row=1, col=2)
        
        fig.update_xaxes(title_text="Response Length (chars)", row=2, col=1)
        fig.update_yaxes(title_text="Generation Speed (chars/sec)", row=2, col=1)
        
        fig.update_xaxes(title_text="Chars/Second", row=2, col=2)
        fig.update_yaxes(title_text="Domain", row=2, col=2)
        
        return fig
    
    def create_approach_comparison(self) -> go.Figure:
        """Compare Raw vs Ground Rules approaches"""
        if self.results.empty:
            return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
        
        # Calculate improvement metrics
        approach_comparison = []
        
        for model in self.results['model_name'].unique():
            model_data = self.results[self.results['model_name'] == model]
            
            if len(model_data['approach'].unique()) >= 2:
                raw_data = model_data[model_data['approach'] == 'raw']
                gr_data = model_data[model_data['approach'] == 'ground_rules']
                
                if not raw_data.empty and not gr_data.empty:
                    length_improvement = gr_data['char_count'].mean() / raw_data['char_count'].mean()
                    speed_ratio = gr_data['chars_per_second'].mean() / raw_data['chars_per_second'].mean()
                    time_ratio = gr_data['generation_time'].mean() / raw_data['generation_time'].mean()
                    
                    approach_comparison.append({
                        'model': model,
                        'length_improvement': length_improvement,
                        'speed_ratio': speed_ratio,
                        'time_ratio': time_ratio,
                        'raw_avg_length': raw_data['char_count'].mean(),
                        'gr_avg_length': gr_data['char_count'].mean(),
                        'raw_avg_speed': raw_data['chars_per_second'].mean(),
                        'gr_avg_speed': gr_data['chars_per_second'].mean()
                    })
        
        if not approach_comparison:
            return go.Figure().add_annotation(text="No comparison data available", x=0.5, y=0.5)
        
        comp_df = pd.DataFrame(approach_comparison)
        
        # Create comparison visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Length Improvement (Ground Rules vs Raw)',
                'Speed Comparison',
                'Approach Performance by Model',
                'Time vs Quality Trade-off'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]],
            vertical_spacing=0.15
        )
        
        # Length improvement
        fig.add_trace(
            go.Bar(
                x=comp_df['model'],
                y=comp_df['length_improvement'],
                text=[f'{x:.1f}x' for x in comp_df['length_improvement']],
                textposition='auto',
                name='Length Improvement',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Speed comparison
        fig.add_trace(
            go.Bar(
                x=comp_df['model'],
                y=comp_df['raw_avg_speed'],
                name='Raw Speed',
                marker_color='salmon'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=comp_df['model'],
                y=comp_df['gr_avg_speed'],
                name='Ground Rules Speed',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # Performance comparison (grouped bar)
        models = comp_df['model']
        fig.add_trace(
            go.Bar(
                x=models,
                y=comp_df['raw_avg_length'],
                name='Raw Length',
                marker_color='coral',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=comp_df['gr_avg_length'],
                name='Ground Rules Length',
                marker_color='skyblue',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Time vs Quality scatter
        fig.add_trace(
            go.Scatter(
                x=comp_df['time_ratio'],
                y=comp_df['length_improvement'],
                mode='markers+text',
                text=comp_df['model'],
                textposition='top center',
                marker=dict(size=12),
                name='Time-Quality Trade-off',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Prompting Strategy Comparison: Raw vs Ground Rules",
            title_x=0.5,
            height=800
        )
        
        return fig
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary statistics"""
        if self.results.empty:
            return {"error": "No data available for analysis"}
        
        summary = {
            "overview": {
                "total_evaluations": len(self.results),
                "unique_models": self.results['model_name'].nunique(),
                "unique_questions": self.results['question_id'].nunique(),
                "unique_domains": self.results['domain'].nunique(),
                "evaluation_period": {
                    "start": self.results['timestamp'].min(),
                    "end": self.results['timestamp'].max()
                }
            },
            "performance_summary": {},
            "approach_analysis": {},
            "domain_analysis": {},
            "recommendations": []
        }
        
        # Performance by model
        for model in self.results['model_name'].unique():
            model_data = self.results[self.results['model_name'] == model]
            summary["performance_summary"][model] = {
                "avg_generation_time": model_data['generation_time'].mean(),
                "avg_response_length": model_data['char_count'].mean(),
                "avg_generation_speed": model_data['chars_per_second'].mean(),
                "total_evaluations": len(model_data),
                "domains_tested": model_data['domain'].nunique()
            }
        
        # Approach analysis
        for approach in self.results['approach'].unique():
            approach_data = self.results[self.results['approach'] == approach]
            summary["approach_analysis"][approach] = {
                "avg_response_length": approach_data['char_count'].mean(),
                "avg_generation_time": approach_data['generation_time'].mean(),
                "avg_speed": approach_data['chars_per_second'].mean(),
                "usage_count": len(approach_data)
            }
        
        # Domain analysis
        for domain in self.results['domain'].unique():
            domain_data = self.results[self.results['domain'] == domain]
            summary["domain_analysis"][domain] = {
                "avg_response_length": domain_data['char_count'].mean(),
                "avg_generation_time": domain_data['generation_time'].mean(),
                "models_tested": domain_data['model_name'].nunique(),
                "best_performing_model": domain_data.loc[domain_data['chars_per_second'].idxmax(), 'model_name']
            }
        
        # Generate recommendations
        if len(summary["performance_summary"]) > 1:
            # Find fastest model
            fastest_model = max(summary["performance_summary"].keys(),
                              key=lambda x: summary["performance_summary"][x]["avg_generation_speed"])
            summary["recommendations"].append(f"For speed: Use {fastest_model}")
            
            # Find most comprehensive model
            most_comprehensive = max(summary["performance_summary"].keys(),
                                   key=lambda x: summary["performance_summary"][x]["avg_response_length"])
            summary["recommendations"].append(f"For comprehensiveness: Use {most_comprehensive}")
        
        if len(summary["approach_analysis"]) > 1:
            approaches = summary["approach_analysis"]
            if approaches.get("ground_rules", {}).get("avg_response_length", 0) > approaches.get("raw", {}).get("avg_response_length", 0):
                improvement = approaches["ground_rules"]["avg_response_length"] / approaches["raw"]["avg_response_length"]
                summary["recommendations"].append(f"Ground rules approach provides {improvement:.1f}x longer responses")
        
        return summary
    
    def save_dashboard_html(self, filename: str = "evaluation_dashboard.html"):
        """Save interactive dashboard as HTML"""
        # Create performance comparison
        perf_fig = self.create_performance_comparison()
        approach_fig = self.create_approach_comparison()
        
        # Generate summary report
        summary = self.generate_summary_report()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: white; border-radius: 3px; }}
                h1, h2 {{ color: #333; }}
                .dashboard {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🧠 Model Evaluation Dashboard</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>📊 Evaluation Overview</h2>
                <div class="metric"><strong>Total Evaluations:</strong> {summary.get('overview', {}).get('total_evaluations', 'N/A')}</div>
                <div class="metric"><strong>Models Tested:</strong> {summary.get('overview', {}).get('unique_models', 'N/A')}</div>
                <div class="metric"><strong>Questions:</strong> {summary.get('overview', {}).get('unique_questions', 'N/A')}</div>
                <div class="metric"><strong>Domains:</strong> {summary.get('overview', {}).get('unique_domains', 'N/A')}</div>
            </div>
            
            <div class="dashboard">
                <h2>🏆 Performance Comparison</h2>
                <div id="performance-chart"></div>
            </div>
            
            <div class="dashboard">
                <h2>🔍 Approach Analysis</h2>
                <div id="approach-chart"></div>
            </div>
            
            <div class="summary">
                <h2>💡 Key Recommendations</h2>
                <ul>
        """
        
        for rec in summary.get('recommendations', []):
            html_content += f"<li>{rec}</li>"
        
        html_content += f"""
                </ul>
            </div>
            
            <script>
                var perfData = {perf_fig.to_json()};
                Plotly.newPlot('performance-chart', perfData.data, perfData.layout);
                
                var approachData = {approach_fig.to_json()};
                Plotly.newPlot('approach-chart', approachData.data, approachData.layout);
            </script>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Dashboard saved to: {filename}")
        return filename

def main():
    """Generate evaluation dashboard"""
    print("🧠 Creating Model Evaluation Dashboard...")
    
    dashboard = ModelEvaluationDashboard()
    
    if dashboard.results.empty:
        print("❌ No evaluation data found in data/ directory")
        print("💡 Run evaluations first to generate dashboard")
        return
    
    print(f"📊 Loaded {len(dashboard.results)} evaluation results")
    print(f"🤖 Models: {', '.join(dashboard.results['model_name'].unique())}")
    print(f"📝 Questions: {dashboard.results['question_id'].nunique()}")
    print(f"🏷️  Domains: {dashboard.results['domain'].nunique()}")
    
    # Generate summary report
    summary = dashboard.generate_summary_report()
    
    # Save summary as JSON
    with open('data/evaluation_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("📁 Summary saved to: data/evaluation_summary.json")
    
    # Create and save dashboard
    dashboard_file = dashboard.save_dashboard_html()
    
    print(f"✅ Dashboard created successfully!")
    print(f"🌐 Open {dashboard_file} in your browser to view interactive dashboard")
    
    # Print key metrics
    if summary.get('recommendations'):
        print(f"\n💡 Key Recommendations:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"   {i}. {rec}")

if __name__ == "__main__":
    main()