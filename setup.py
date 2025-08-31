"""
Setup script for Educational LLM project
Handles environment setup, model downloading, and validation
"""
import subprocess
import sys
import os
import torch
from pathlib import Path
import logging
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectSetup:
    """Handles complete project setup"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            raise RuntimeError("Python 3.8 or higher is required")
        logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    def install_dependencies(self):
        """Install required dependencies"""
        logger.info("Installing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ])
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise
    
    def check_gpu_availability(self):
        """Check GPU availability and CUDA setup"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            logger.info(f"GPU available: {gpu_name}")
            logger.info(f"GPU memory: {gpu_memory:.1f}GB")
            logger.info(f"Number of GPUs: {gpu_count}")
            
            # Check if we have enough memory for Gemma 3 12B
            if gpu_memory < 12:
                logger.warning("GPU memory < 12GB. Consider using 4-bit quantization.")
            
            return True
        else:
            logger.warning("No GPU available. Training will be very slow on CPU.")
            return False
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            config.output_dir,
            config.data_dir,
            config.logs_dir,
            config.model.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def test_model_loading(self):
        """Test if we can load the model"""
        logger.info("Testing model loading...")
        try:
            from gemma_inference import GemmaEducationalInference
            
            # Try to initialize (will download model if needed)
            inference = GemmaEducationalInference()
            
            # Test a simple generation
            test_response = inference.generate_response("What is 2+2?", max_length=100)
            logger.info(f"Model test successful. Sample response: {test_response[:100]}...")
            
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def create_baseline_evaluation(self):
        """Create baseline evaluation dataset"""
        logger.info("Creating baseline evaluation dataset...")
        try:
            from evaluation_framework import EducationalEvaluator
            
            evaluator = EducationalEvaluator()
            dataset = evaluator.create_evaluation_dataset()
            filepath = evaluator.save_evaluation_dataset(dataset)
            
            logger.info(f"Evaluation dataset created: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to create evaluation dataset: {e}")
            return False
    
    def run_baseline_inference(self):
        """Run baseline inference on evaluation dataset"""
        logger.info("Running baseline inference...")
        try:
            from gemma_inference import GemmaEducationalInference
            import json
            
            # Load evaluation dataset
            eval_file = Path(config.data_dir) / "evaluation_dataset.json"
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
            
            # Get questions
            questions = [item['question'] for item in eval_data]
            
            # Generate responses
            inference = GemmaEducationalInference()
            results = inference.batch_generate(questions[:5])  # Start with 5 questions
            
            # Save results
            inference.save_responses(results, "baseline_responses_sample.json")
            
            logger.info("Baseline inference completed")
            return True
        except Exception as e:
            logger.error(f"Baseline inference failed: {e}")
            return False
    
    def setup_complete(self):
        """Run complete setup process"""
        logger.info("Starting Educational LLM project setup...")
        
        steps = [
            ("Check Python version", self.check_python_version),
            ("Install dependencies", self.install_dependencies),
            ("Check GPU availability", self.check_gpu_availability),
            ("Setup directories", self.setup_directories),
            ("Create evaluation dataset", self.create_baseline_evaluation),
            ("Test model loading", self.test_model_loading),
            ("Run baseline inference", self.run_baseline_inference),
        ]
        
        for step_name, step_func in steps:
            try:
                logger.info(f"Step: {step_name}")
                step_func()
                logger.info(f"✓ {step_name} completed")
            except Exception as e:
                logger.error(f"✗ {step_name} failed: {e}")
                return False
        
        logger.info("🎉 Setup completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Review the evaluation dataset in data/evaluation_dataset.json")
        logger.info("2. Run full baseline evaluation: python evaluation_framework.py")
        logger.info("3. Manually improve some responses for training data")
        logger.info("4. Start Phase 2: Training data preparation")
        
        return True

def main():
    """Main setup function"""
    setup = ProjectSetup()
    success = setup.setup_complete()
    
    if not success:
        logger.error("Setup failed. Please check the errors above.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🚀 EDUCATIONAL LLM PROJECT READY!")
    print("="*60)
    print("Key files created:")
    print("- config.py: Project configuration")
    print("- gemma_inference.py: Model inference")
    print("- evaluation_framework.py: Evaluation tools")
    print("- data_utils.py: Data processing")
    print("- data/evaluation_dataset.json: Your evaluation questions")
    print("- data/baseline_responses_sample.json: Initial responses")
    print("\nNext: Review baseline responses and plan improvements!")

if __name__ == "__main__":
    main()
