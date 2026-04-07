"""
Utility functions for training and evaluation.
"""

import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from typing import Tuple


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
    
    Returns:
        Tuple of (predictions, labels, average_loss)
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / max(len(dataloader), 1)
    return np.array(all_preds), np.array(all_labels), avg_loss


def train_epoch(model, dataloader, optimizer, scheduler, device: str) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    
    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / max(len(dataloader), 1)


def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
    
    Returns:
        Dictionary of metrics
    """
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }
