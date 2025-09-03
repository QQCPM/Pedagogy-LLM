#!/usr/bin/env python3
"""
Generate Preference Pairs for RLHF Training
Creates comparison pairs from user feedback data for future fine-tuning
"""
import json
import itertools
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreferencePairGenerator:
    """Generate preference pairs from interaction data"""
    
    def __init__(self):
        self.data_dir = Path(config.data_dir)
        self.interactions_dir = self.data_dir / "interactions"
        self.training_dir = self.data_dir / "training_data"
        
        # Ensure training directory exists
        self.training_dir.mkdir(exist_ok=True)
    
    def load_interactions(self):
        """Load all interaction data"""
        interactions = []
        
        for interaction_file in self.interactions_dir.glob("*.jsonl"):
            try:
                with open(interaction_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            interaction = json.loads(line)
                            interactions.append(interaction)
            except Exception as e:
                logger.error(f"Error reading {interaction_file}: {e}")
        
        logger.info(f"Loaded {len(interactions)} interactions")
        return interactions
    
    def group_by_similarity(self, interactions):
        """Group interactions by similar questions for comparison"""
        # Simple grouping by question similarity (exact match for now)
        # In a more advanced version, you could use semantic similarity
        
        question_groups = defaultdict(list)
        
        for interaction in interactions:
            # Extract the original question (without feedback modifications)
            question = interaction['question']
            
            # Remove feedback modifications to get base question
            base_question = question.split('\n\nPLEASE IMPROVE:')[0]
            base_question = base_question.split('\n\nAlso:')[0]
            
            question_groups[base_question].append(interaction)
        
        # Filter groups that have multiple responses
        viable_groups = {
            q: responses for q, responses in question_groups.items() 
            if len(responses) >= 2
        }
        
        logger.info(f"Found {len(viable_groups)} question groups with multiple responses")
        return viable_groups
    
    def create_preference_pairs(self, question_groups):
        """Create preference pairs from grouped interactions"""
        preference_pairs = []
        
        for question, interactions in question_groups.items():
            # Sort by rating for easier comparison
            interactions.sort(key=lambda x: x['feedback']['rating'], reverse=True)
            
            # Create pairs comparing higher-rated vs lower-rated responses
            for i in range(len(interactions)):
                for j in range(i + 1, len(interactions)):
                    higher_rated = interactions[i]
                    lower_rated = interactions[j]
                    
                    # Only create pairs if there's a clear rating difference
                    if higher_rated['feedback']['rating'] > lower_rated['feedback']['rating']:
                        
                        # Calculate confidence based on rating difference
                        rating_diff = higher_rated['feedback']['rating'] - lower_rated['feedback']['rating']
                        confidence = min(rating_diff / 4.0, 1.0)  # Normalize to 0-1
                        
                        pair = {
                            'question': question,
                            'chosen_response': higher_rated['response'],
                            'rejected_response': lower_rated['response'],
                            'chosen_rating': higher_rated['feedback']['rating'],
                            'rejected_rating': lower_rated['feedback']['rating'],
                            'confidence': confidence,
                            'chosen_feedback': {
                                'strengths': higher_rated['feedback'].get('strengths', []),
                                'improvements': higher_rated['feedback'].get('improvements', []),
                                'specific_feedback': higher_rated['feedback'].get('specific_feedback', '')
                            },
                            'rejected_feedback': {
                                'strengths': lower_rated['feedback'].get('strengths', []),
                                'improvements': lower_rated['feedback'].get('improvements', []),
                                'specific_feedback': lower_rated['feedback'].get('specific_feedback', '')
                            },
                            'metadata': {
                                'chosen_session': higher_rated['session_id'],
                                'rejected_session': lower_rated['session_id'],
                                'chosen_timestamp': higher_rated['timestamp'],
                                'rejected_timestamp': lower_rated['timestamp'],
                                'pair_created': datetime.now().isoformat()
                            }
                        }
                        
                        preference_pairs.append(pair)
        
        logger.info(f"Generated {len(preference_pairs)} preference pairs")
        return preference_pairs
    
    def create_cross_question_pairs(self, interactions, min_rating_diff=2):
        """Create preference pairs across different questions (same rating patterns)"""
        # Group by rating to find high vs low quality responses across different questions
        high_quality = [i for i in interactions if i['feedback']['rating'] >= 4]
        low_quality = [i for i in interactions if i['feedback']['rating'] <= 2]
        
        cross_pairs = []
        
        # Limit pairs to avoid explosion
        max_pairs_per_category = 50
        pair_count = 0
        
        for high in high_quality[:max_pairs_per_category]:
            for low in low_quality[:max_pairs_per_category]:
                if pair_count >= max_pairs_per_category:
                    break
                
                # Don't pair responses to the same question
                if high['question'].split('\n\nPLEASE IMPROVE:')[0] == low['question'].split('\n\nPLEASE IMPROVE:')[0]:
                    continue
                
                pair = {
                    'comparison_type': 'cross_question',
                    'high_quality_example': {
                        'question': high['question'],
                        'response': high['response'],
                        'rating': high['feedback']['rating'],
                        'feedback': high['feedback']
                    },
                    'low_quality_example': {
                        'question': low['question'],
                        'response': low['response'],
                        'rating': low['feedback']['rating'],
                        'feedback': low['feedback']
                    },
                    'rating_difference': high['feedback']['rating'] - low['feedback']['rating'],
                    'metadata': {
                        'created': datetime.now().isoformat()
                    }
                }
                
                cross_pairs.append(pair)
                pair_count += 1
        
        logger.info(f"Generated {len(cross_pairs)} cross-question comparison pairs")
        return cross_pairs
    
    def create_improvement_pairs(self, interactions):
        """Create pairs showing improvement within the same session"""
        improvement_pairs = []
        
        # Group by session to find improvement within sessions
        session_groups = defaultdict(list)
        for interaction in interactions:
            session_groups[interaction['session_id']].append(interaction)
        
        for session_id, session_interactions in session_groups.items():
            if len(session_interactions) < 2:
                continue
            
            # Sort by attempt number
            session_interactions.sort(key=lambda x: x['attempt'])
            
            # Look for improvements (later attempts with higher ratings)
            for i in range(len(session_interactions) - 1):
                current = session_interactions[i]
                next_attempt = session_interactions[i + 1]
                
                if next_attempt['feedback']['rating'] > current['feedback']['rating']:
                    pair = {
                        'type': 'iterative_improvement',
                        'question': current['question'],
                        'improved_response': next_attempt['response'],
                        'original_response': current['response'],
                        'improvement_rating': next_attempt['feedback']['rating'],
                        'original_rating': current['feedback']['rating'],
                        'improvement_delta': next_attempt['feedback']['rating'] - current['feedback']['rating'],
                        'feedback_given': current['feedback'],
                        'session_id': session_id,
                        'metadata': {
                            'original_attempt': current['attempt'],
                            'improved_attempt': next_attempt['attempt'],
                            'created': datetime.now().isoformat()
                        }
                    }
                    
                    improvement_pairs.append(pair)
        
        logger.info(f"Generated {len(improvement_pairs)} improvement pairs")
        return improvement_pairs
    
    def save_preference_data(self, preference_pairs, cross_pairs, improvement_pairs):
        """Save all preference data for training"""
        
        # Save main preference pairs
        if preference_pairs:
            pairs_file = self.training_dir / "preference_pairs.jsonl"
            with open(pairs_file, 'w', encoding='utf-8') as f:
                for pair in preference_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(preference_pairs)} preference pairs to {pairs_file}")
        
        # Save cross-question pairs
        if cross_pairs:
            cross_file = self.training_dir / "cross_question_pairs.jsonl"
            with open(cross_file, 'w', encoding='utf-8') as f:
                for pair in cross_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(cross_pairs)} cross-question pairs to {cross_file}")
        
        # Save improvement pairs
        if improvement_pairs:
            improvement_file = self.training_dir / "improvement_pairs.jsonl"
            with open(improvement_file, 'w', encoding='utf-8') as f:
                for pair in improvement_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(improvement_pairs)} improvement pairs to {improvement_file}")
        
        # Create summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'total_preference_pairs': len(preference_pairs),
            'total_cross_question_pairs': len(cross_pairs),
            'total_improvement_pairs': len(improvement_pairs),
            'total_training_examples': len(preference_pairs) + len(cross_pairs) + len(improvement_pairs),
            'data_quality': {
                'high_confidence_pairs': len([p for p in preference_pairs if p.get('confidence', 0) >= 0.5]),
                'significant_improvements': len([p for p in improvement_pairs if p.get('improvement_delta', 0) >= 2])
            }
        }
        
        summary_file = self.training_dir / "preference_data_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def generate_preference_dataset(self):
        """Main function to generate complete preference dataset"""
        logger.info("🚀 Generating preference pairs for RLHF training...")
        
        # Load interactions
        interactions = self.load_interactions()
        
        if len(interactions) < 2:
            logger.warning("❌ Need at least 2 interactions to generate preference pairs")
            return None
        
        # Generate different types of preference pairs
        logger.info("📊 Creating same-question preference pairs...")
        question_groups = self.group_by_similarity(interactions)
        preference_pairs = self.create_preference_pairs(question_groups)
        
        logger.info("🔄 Creating cross-question comparison pairs...")
        cross_pairs = self.create_cross_question_pairs(interactions)
        
        logger.info("📈 Creating iterative improvement pairs...")
        improvement_pairs = self.create_improvement_pairs(interactions)
        
        # Save all data
        summary = self.save_preference_data(preference_pairs, cross_pairs, improvement_pairs)
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 PREFERENCE PAIR GENERATION SUMMARY")
        print("="*80)
        print(f"📊 Total Training Examples: {summary['total_training_examples']}")
        print(f"   • Preference Pairs: {summary['total_preference_pairs']}")
        print(f"   • Cross-Question Pairs: {summary['total_cross_question_pairs']}")
        print(f"   • Improvement Pairs: {summary['total_improvement_pairs']}")
        print(f"🎯 High Confidence Pairs: {summary['data_quality']['high_confidence_pairs']}")
        print(f"📈 Significant Improvements: {summary['data_quality']['significant_improvements']}")
        print(f"💾 Data saved in: {self.training_dir}")
        print("="*80)
        
        if summary['total_training_examples'] >= 10:
            print("✅ Sufficient data for initial RLHF training!")
        elif summary['total_training_examples'] >= 5:
            print("⚠️  Limited data - consider collecting more interactions")
        else:
            print("❌ Insufficient data - need more interactions for training")
        
        return summary

def main():
    """Generate preference pairs from interaction data"""
    generator = PreferencePairGenerator()
    generator.generate_preference_dataset()

if __name__ == "__main__":
    main()