#!/usr/bin/env python3
"""
Learning Analytics Dashboard
Analyze your interactions and learning progress with the Educational LLM
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from config import config

class LearningAnalytics:
    """Analyze learning patterns and generate insights"""
    
    def __init__(self):
        self.data_dir = Path(config.data_dir)
        self.interactions_dir = self.data_dir / "interactions"
        self.training_dir = self.data_dir / "training_data"
        self.output_dir = Path(config.output_dir)
        
        # Ensure directories exist
        for dir_path in [self.interactions_dir, self.training_dir, self.output_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def load_interaction_data(self, days_back=30):
        """Load interaction data from the last N days"""
        interactions = []
        
        # Load all interaction files
        for interaction_file in self.interactions_dir.glob("*.jsonl"):
            try:
                with open(interaction_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            interaction = json.loads(line)
                            interactions.append(interaction)
            except Exception as e:
                print(f"⚠️ Error reading {interaction_file}: {e}")
        
        # Filter by date if specified
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            interactions = [
                i for i in interactions 
                if datetime.fromisoformat(i['timestamp']) >= cutoff_date
            ]
        
        return interactions
    
    def analyze_learning_patterns(self, interactions):
        """Analyze learning patterns and preferences"""
        if not interactions:
            return {"error": "No interaction data found"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(interactions)
        
        # Basic statistics
        total_sessions = len(df['session_id'].unique())
        total_questions = len(df)
        avg_rating = df['feedback'].apply(lambda x: x['rating']).mean()
        
        # Rating distribution
        ratings = df['feedback'].apply(lambda x: x['rating'])
        rating_dist = ratings.value_counts().sort_index()
        
        # Topics/domains analysis (extract from questions)
        topics = self._extract_topics_from_questions(df['question'].tolist())
        
        # Time patterns
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        
        # Feedback patterns
        strengths_counter = Counter()
        improvements_counter = Counter()
        
        for _, row in df.iterrows():
            feedback = row['feedback']
            strengths_counter.update(feedback.get('strengths', []))
            improvements_counter.update(feedback.get('improvements', []))
        
        # Success metrics
        high_quality_responses = len(df[ratings >= 4])
        needs_improvement = len(df[ratings <= 2])
        regeneration_rate = len(df[df['attempt'] > 1]) / total_questions
        
        return {
            'summary': {
                'total_sessions': total_sessions,
                'total_questions': total_questions,
                'average_rating': round(avg_rating, 2),
                'high_quality_rate': round(high_quality_responses / total_questions, 2),
                'regeneration_rate': round(regeneration_rate, 2)
            },
            'rating_distribution': rating_dist.to_dict(),
            'topics': dict(topics.most_common(10)),
            'time_patterns': {
                'by_hour': df['hour'].value_counts().to_dict(),
                'by_day': df['day_of_week'].value_counts().to_dict()
            },
            'feedback_patterns': {
                'strengths': dict(strengths_counter.most_common()),
                'improvements': dict(improvements_counter.most_common())
            },
            'learning_trends': self._analyze_learning_trends(df)
        }
    
    def _extract_topics_from_questions(self, questions):
        """Extract topics/domains from questions using keyword matching"""
        topic_keywords = {
            'linear_algebra': ['matrix', 'eigenvalue', 'eigenvector', 'determinant', 'vector', 'linear', 'svd', 'pca'],
            'probability': ['probability', 'bayes', 'distribution', 'random', 'statistics', 'variance', 'mean'],
            'deep_learning': ['neural', 'network', 'transformer', 'attention', 'backprop', 'gradient', 'cnn', 'rnn'],
            'calculus': ['derivative', 'integral', 'limit', 'differential', 'function', 'optimization'],
            'world_models': ['world model', 'reinforcement', 'planning', 'environment', 'agent', 'model-based'],
            'physics': ['quantum', 'mechanics', 'thermodynamics', 'electromagnetic', 'relativity'],
            'algorithms': ['algorithm', 'complexity', 'data structure', 'sorting', 'search', 'graph'],
            'machine_learning': ['learning', 'training', 'model', 'supervised', 'unsupervised', 'classification']
        }
        
        topic_counter = Counter()
        
        for question in questions:
            question_lower = question.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    topic_counter[topic] += 1
        
        return topic_counter
    
    def _analyze_learning_trends(self, df):
        """Analyze learning trends over time"""
        if len(df) < 5:  # Need minimum data points
            return {"error": "Insufficient data for trend analysis"}
        
        # Sort by timestamp
        df_sorted = df.sort_values('datetime')
        
        # Rolling average of ratings
        ratings = df_sorted['feedback'].apply(lambda x: x['rating'])
        if len(ratings) >= 3:
            rolling_avg = ratings.rolling(window=3, min_periods=1).mean()
            
            # Check if learning is improving
            recent_avg = rolling_avg.tail(5).mean()
            early_avg = rolling_avg.head(5).mean()
            improvement = recent_avg - early_avg
            
            return {
                'rating_trend': rolling_avg.tolist(),
                'improvement_score': round(improvement, 2),
                'is_improving': improvement > 0.2,
                'recent_performance': round(recent_avg, 2)
            }
        
        return {"error": "Insufficient data for trend analysis"}
    
    def generate_insights(self, analysis):
        """Generate actionable insights from analysis"""
        insights = []
        
        if analysis.get('summary'):
            summary = analysis['summary']
            
            # Quality insights
            if summary['high_quality_rate'] >= 0.7:
                insights.append("🎉 Excellent! 70%+ of responses are high quality (4-5 stars)")
            elif summary['high_quality_rate'] >= 0.5:
                insights.append("👍 Good progress! 50%+ responses are high quality")
            else:
                insights.append("🎯 Focus area: Try being more specific in your questions for better responses")
            
            # Learning efficiency
            if summary['regeneration_rate'] <= 0.2:
                insights.append("⚡ Efficient learning! You rarely need regeneration")
            elif summary['regeneration_rate'] >= 0.4:
                insights.append("💡 Tip: Try more specific questions to reduce regeneration needs")
        
        # Topic insights
        if analysis.get('topics'):
            top_topic = max(analysis['topics'], key=analysis['topics'].get)
            insights.append(f"📚 Your most explored topic: {top_topic.replace('_', ' ').title()}")
            
            if len(analysis['topics']) >= 3:
                insights.append("🌟 Great diversity in learning topics!")
            else:
                insights.append("🔄 Consider exploring more diverse topics for well-rounded learning")
        
        # Feedback patterns
        if analysis.get('feedback_patterns'):
            strengths = analysis['feedback_patterns']['strengths']
            improvements = analysis['feedback_patterns']['improvements']
            
            if strengths:
                top_strength = max(strengths, key=strengths.get)
                insights.append(f"💪 The model excels at: {self._feedback_code_to_text(top_strength, 'strength')}")
            
            if improvements:
                top_improvement = max(improvements, key=improvements.get)
                insights.append(f"🎯 Most common improvement need: {self._feedback_code_to_text(top_improvement, 'improvement')}")
        
        # Learning trends
        if analysis.get('learning_trends') and not analysis['learning_trends'].get('error'):
            trends = analysis['learning_trends']
            if trends.get('is_improving'):
                insights.append(f"📈 Your learning is improving! Recent average: {trends['recent_performance']}/5")
            else:
                insights.append("📊 Learning curve is stable - consider trying more challenging topics")
        
        return insights
    
    def _feedback_code_to_text(self, code, category):
        """Convert feedback codes to readable text"""
        strength_map = {
            '1': 'Clear explanations',
            '2': 'Good examples', 
            '3': 'Math notation',
            '4': 'Structure/organization',
            '5': 'Depth of content',
            '6': 'Connections to other topics'
        }
        
        improvement_map = {
            '1': 'Too complex/advanced',
            '2': 'Too simple/basic',
            '3': 'Poor examples',
            '4': 'Math notation issues',
            '5': 'Missing connections',
            '6': 'Poor structure',
            '7': 'Too long',
            '8': 'Too short'
        }
        
        if category == 'strength':
            return strength_map.get(code, f"Category {code}")
        else:
            return improvement_map.get(code, f"Category {code}")
    
    def create_visualizations(self, analysis):
        """Create visualizations of learning patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Learning Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # Rating distribution
        if analysis.get('rating_distribution'):
            ratings = list(analysis['rating_distribution'].keys())
            counts = list(analysis['rating_distribution'].values())
            
            axes[0, 0].bar(ratings, counts, color='skyblue', edgecolor='navy', alpha=0.7)
            axes[0, 0].set_title('Response Quality Distribution')
            axes[0, 0].set_xlabel('Rating (1-5)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_ylim(0, max(counts) * 1.1 if counts else 1)
        
        # Topic distribution
        if analysis.get('topics'):
            topics = list(analysis['topics'].keys())[:8]  # Top 8 topics
            counts = [analysis['topics'][topic] for topic in topics]
            
            axes[0, 1].barh(topics, counts, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
            axes[0, 1].set_title('Most Explored Topics')
            axes[0, 1].set_xlabel('Number of Questions')
        
        # Time patterns (by hour)
        if analysis.get('time_patterns', {}).get('by_hour'):
            hours = list(analysis['time_patterns']['by_hour'].keys())
            counts = list(analysis['time_patterns']['by_hour'].values())
            
            axes[1, 0].plot(hours, counts, marker='o', color='orange', linewidth=2)
            axes[1, 0].set_title('Learning Activity by Hour')
            axes[1, 0].set_xlabel('Hour of Day')
            axes[1, 0].set_ylabel('Number of Questions')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning trend
        if analysis.get('learning_trends') and not analysis['learning_trends'].get('error'):
            trend_data = analysis['learning_trends']['rating_trend']
            axes[1, 1].plot(range(len(trend_data)), trend_data, marker='o', color='purple', linewidth=2)
            axes[1, 1].set_title('Learning Quality Trend')
            axes[1, 1].set_xlabel('Question Number')
            axes[1, 1].set_ylabel('Rolling Average Rating')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(1, 5)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / f"learning_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        
        print(f"📊 Visualization saved: {viz_path}")
        return viz_path
    
    def generate_report(self, days_back=30):
        """Generate comprehensive learning analytics report"""
        print("🔍 Loading interaction data...")
        interactions = self.load_interaction_data(days_back)
        
        if not interactions:
            print("❌ No interaction data found. Start using `python ask.py` to generate data!")
            return
        
        print(f"📊 Analyzing {len(interactions)} interactions...")
        analysis = self.analyze_learning_patterns(interactions)
        
        # Generate insights
        insights = self.generate_insights(analysis)
        
        # Create visualizations
        viz_path = self.create_visualizations(analysis)
        
        # Create text report
        report_lines = [
            "# 🎯 Personal Learning Analytics Report",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Analysis Period: Last {days_back} days*",
            "",
            "## 📈 Summary Statistics",
        ]
        
        if analysis.get('summary'):
            summary = analysis['summary']
            report_lines.extend([
                f"- **Total Learning Sessions**: {summary['total_sessions']}",
                f"- **Total Questions Asked**: {summary['total_questions']}",  
                f"- **Average Response Quality**: {summary['average_rating']}/5 ⭐",
                f"- **High Quality Response Rate**: {summary['high_quality_rate']*100:.1f}%",
                f"- **Regeneration Rate**: {summary['regeneration_rate']*100:.1f}%",
                ""
            ])
        
        # Add insights
        report_lines.extend([
            "## 💡 Key Insights",
            ""
        ])
        
        for insight in insights:
            report_lines.append(f"- {insight}")
        
        report_lines.append("")
        
        # Add topic breakdown
        if analysis.get('topics'):
            report_lines.extend([
                "## 📚 Learning Topics",
                ""
            ])
            
            for topic, count in list(analysis['topics'].items())[:10]:
                topic_name = topic.replace('_', ' ').title()
                report_lines.append(f"- **{topic_name}**: {count} questions")
            
            report_lines.append("")
        
        # Save report
        report_path = self.output_dir / f"learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Display summary
        print("\n" + "="*80)
        print("🎯 LEARNING ANALYTICS SUMMARY")
        print("="*80)
        
        if analysis.get('summary'):
            summary = analysis['summary']
            print(f"📊 Sessions: {summary['total_sessions']} | Questions: {summary['total_questions']}")
            print(f"⭐ Avg Quality: {summary['average_rating']}/5")
            print(f"🎯 High Quality Rate: {summary['high_quality_rate']*100:.1f}%")
        
        print(f"\n💡 TOP INSIGHTS:")
        for insight in insights[:3]:  # Show top 3 insights
            print(f"   {insight}")
        
        print(f"\n📄 Full report: {report_path}")
        print(f"📊 Visualization: {viz_path}")
        print("="*80)
        
        return {
            'analysis': analysis,
            'insights': insights,
            'report_path': report_path,
            'visualization_path': viz_path
        }

def main():
    """Generate learning analytics report"""
    analytics = LearningAnalytics()
    
    # Check for command line arguments
    import sys
    days_back = 30
    
    if len(sys.argv) > 1:
        try:
            days_back = int(sys.argv[1])
        except ValueError:
            print("Usage: python learning_analytics.py [days_back]")
            print("Example: python learning_analytics.py 7  # Last 7 days")
            return
    
    print(f"🔍 Generating learning analytics for the last {days_back} days...")
    analytics.generate_report(days_back)

if __name__ == "__main__":
    main()