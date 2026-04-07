"""
Auto-labeling script for VOZ-HSD dataset using trained hate speech detection model.

Usage:
    python src/label_dataset.py --model_path models/your_model --batch_idx 0 --total_batches 10
"""

import argparse
import os
import pandas as pd
import torch
from datasets import load_dataset
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model import load_trained_model


class SimpleTextDataset(Dataset):
    """Simple dataset for inference only."""
    
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def predict_labels(model, dataloader, device):
    """Predict labels for a dataset."""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
    
    return all_preds


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Auto-label VOZ-HSD dataset")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "validation", "test"],
                       help="Dataset split to label")
    parser.add_argument("--batch_idx", type=int, default=0,
                       help="Batch index for parallel processing (0-indexed)")
    parser.add_argument("--total_batches", type=int, default=1,
                       help="Total number of batches to split dataset into")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Inference batch size")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="labeled_data",
                       help="Output directory for labeled data")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to process (for testing)")
    
    return parser.parse_args()


def main():
    """Main labeling function."""
    args = parse_args()
    
    # Validate batch parameters
    if args.batch_idx < 0 or args.batch_idx >= args.total_batches:
        raise ValueError(f"batch_idx must be in range [0, {args.total_batches})")
    
    print("=" * 80)
    print(f"Auto-Labeling VOZ-HSD Dataset")
    print("=" * 80)
    print(f"  Model: {args.model_path}")
    print(f"  Split: {args.split}")
    print(f"  Batch: {args.batch_idx + 1}/{args.total_batches}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 80)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Device: {device}")
    
    # Load model
    print(f"\n🤖 Loading model from {args.model_path}...")
    model, tokenizer = load_trained_model(args.model_path, device)
    
    # Load dataset
    print(f"\n📚 Loading VOZ-HSD dataset ({args.split} split)...")
    dataset = load_dataset("tarudesu/VOZ-HSD", split=args.split)
    df = dataset.to_pandas()
    
    print(f"  Total samples: {len(df):,}")
    
    # Apply max_samples limit if specified
    if args.max_samples is not None:
        df = df.head(args.max_samples)
        print(f"  Limited to: {len(df):,} samples (for testing)")
    
    # Split into batches for parallel processing
    total_samples = len(df)
    batch_size_split = total_samples // args.total_batches
    start_idx = args.batch_idx * batch_size_split
    
    if args.batch_idx == args.total_batches - 1:
        # Last batch gets remaining samples
        end_idx = total_samples
    else:
        end_idx = start_idx + batch_size_split
    
    df_batch = df.iloc[start_idx:end_idx].copy()
    
    print(f"\n📦 Processing batch {args.batch_idx + 1}/{args.total_batches}")
    print(f"  Samples: {len(df_batch):,} (indices {start_idx:,} to {end_idx:,})")
    
    # Debug: Print all available columns
    print(f"  Available columns: {list(df_batch.columns)}")
    
    # Determine text and label columns
    # VOZ-HSD actual columns based on inspection
    if "comment" in df_batch.columns:
        text_col = "comment"
    elif "text" in df_batch.columns:
        text_col = "text"
    elif "Comment" in df_batch.columns:
        text_col = "Comment"
    else:
        # Fallback: use first string column
        text_col = df_batch.columns[0]
        print(f"  ⚠️  Warning: Using first column as text: {text_col}")
    
    if "label" in df_batch.columns:
        label_col = "label"
    elif "toxicity" in df_batch.columns:
        label_col = "toxicity"
    elif "Toxicity" in df_batch.columns:
        label_col = "Toxicity"
    else:
        # Fallback: use last column
        label_col = df_batch.columns[-1]
        print(f"  ⚠️  Warning: Using last column as label: {label_col}")
    
    print(f"  Text column: {text_col}")
    print(f"  Label column: {label_col}")
    
    # Create dataset and dataloader
    print("\n🔨 Creating inference dataloader...")
    inference_dataset = SimpleTextDataset(
        df_batch[text_col].tolist(),
        tokenizer,
        args.max_length
    )
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Predict labels
    print(f"\n🚀 Starting prediction...")
    predicted_labels = predict_labels(model, inference_loader, device)
    
    # Add predictions to dataframe
    df_batch["predicted_label"] = predicted_labels
    
    # Convert original labels to integers (VOZ-HSD might have float labels)
    # Also handle any NaN values
    df_batch["original_label"] = df_batch[label_col].fillna(0).astype(int)
    
    # Calculate metrics
    print("\n📊 Evaluation Metrics:")
    print("=" * 80)
    
    original = df_batch["original_label"].values
    predicted = df_batch["predicted_label"].values
    
    accuracy = accuracy_score(original, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        original, predicted, average="binary", pos_label=1
    )
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (HATE): {precision:.4f}")
    print(f"  Recall (HATE): {recall:.4f}")
    print(f"  F1 (HATE): {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(original, predicted)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted CLEAN  Predicted HATE")
    print(f"  Actual CLEAN       {cm[0][0]:>10,}    {cm[0][1]:>10,}")
    print(f"  Actual HATE        {cm[1][0]:>10,}    {cm[1][1]:>10,}")
    
    # Agreement statistics
    agreement = (original == predicted).sum()
    disagreement = (original != predicted).sum()
    print(f"\n  Agreement: {agreement:,} samples ({agreement/len(df_batch)*100:.2f}%)")
    print(f"  Disagreement: {disagreement:,} samples ({disagreement/len(df_batch)*100:.2f}%)")
    
    # Save results to temp directory (will be merged and cleaned up later)
    output_dir = Path(args.output_dir)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save labeled data to temp
    output_file = temp_dir / f"{args.split}_batch_{args.batch_idx}_{args.total_batches}.csv"
    df_batch.to_csv(output_file, index=False)
    print(f"\n💾 Saved labeled data to: {output_file}")
    
    # Save batch metrics to temp
    metrics = {
        "batch_idx": args.batch_idx,
        "total_batches": args.total_batches,
        "split": args.split,
        "samples": len(df_batch),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "agreement": agreement,
        "disagreement": disagreement,
        "model_path": args.model_path,
    }
    
    metrics_file = temp_dir / f"metrics_batch_{args.batch_idx}_{args.total_batches}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    print(f"  Saved metrics to: {metrics_file}")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
