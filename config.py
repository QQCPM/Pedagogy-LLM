"""Configuration settings for Educational LLM project"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_name: str = "gemma3:12b"  # Using Gemma 3 12B via Ollama  
    cache_dir: str = "./models"
    max_length: int = 6144
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    load_in_4bit: bool = True  # For memory efficiency
    
@dataclass
class TrainingConfig:
    """LoRA training configuration"""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 500
    evaluation_steps: int = 250
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

@dataclass
class EvaluationConfig:
    """Evaluation settings"""
    eval_dataset_size: int = 50
    domains: List[str] = None
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = ["linear_algebra", "probability", "deep_learning", "world_models"]
        if self.metrics is None:
            self.metrics = ["pedagogical_structure", "concept_clarity", "example_quality", "connection_building"]

@dataclass
class ProjectConfig:
    """Main project configuration"""
    project_name: str = "educational-llm"
    output_dir: str = "./outputs"
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    wandb_project: str = "educational-llm-finetune"
    
    model: ModelConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        
        # Create directories
        for dir_path in [self.output_dir, self.data_dir, self.logs_dir, self.model.cache_dir]:
            os.makedirs(dir_path, exist_ok=True)

# Global config instance
config = ProjectConfig()
