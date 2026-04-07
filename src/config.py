"""
Configuration classes for training and evaluation.
"""

import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    """Training configuration."""
    
    dataset_name: str
    model_name: str = "vinai/phobert-base"
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 3
    seed: int = 42
    output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize device and output directory."""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.output_dir is None:
            # Extract model short name (e.g., "phobert-base" from "vinai/phobert-base")
            model_short = self.model_name.split("/")[-1]
            # Create timestamp: YYYYMMDD_HHMMSS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Format: models/{dataset}_{model}_{timestamp}
            self.output_dir = Path("models") / f"{self.dataset_name}_{model_short}_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "patience": self.patience,
            "seed": self.seed,
            "device": self.device,
            "output_dir": str(self.output_dir),
        }
