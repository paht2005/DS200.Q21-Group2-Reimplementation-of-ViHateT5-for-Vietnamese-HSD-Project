"""
Model building utilities.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple


def build_model(model_name: str, num_labels: int, device: str = "cuda") -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Build model and tokenizer.
    
    Args:
        model_name: Name of the pretrained model
        num_labels: Number of classification labels
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    
    model.to(device)
    
    return model, tokenizer


def load_trained_model(model_path: str, device: str = "cuda") -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model directory
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer
