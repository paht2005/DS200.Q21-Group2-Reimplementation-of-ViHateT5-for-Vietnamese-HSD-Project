"""
Vietnamese Hate Speech Detection Package
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import TrainConfig
from .data_loader import load_dataset_by_name, TextDataset
from .model import build_model
from .utils import set_seed, evaluate, train_epoch

__all__ = [
    "TrainConfig",
    "load_dataset_by_name",
    "TextDataset",
    "build_model",
    "set_seed",
    "evaluate",
    "train_epoch",
]
