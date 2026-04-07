"""
Inference script for hate speech detection - encoder models (PhoBERT, ViSoBERT, etc).

Usage (single text):
    python src/inference.py --model_path models/ViHSD_processed_visobert_reproduction --text "This is a bad comment"

Usage (HuggingFace model):
    python src/inference.py --model_name tarudesu/ViSoBERT-HSD --text "This is a bad comment"

Usage (test set from dataset):
    python src/inference.py --model_path models/ViHSD_processed_visobert_reproduction --dataset ViHSD_processed --output_csv predictions.csv

Usage (local CSV):
    python src/inference.py --model_path models/ViHSD_processed_visobert_reproduction --input_csv data.csv --text_column text --output_csv predictions.csv
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def load_model_and_tokenizer(model_path=None, model_name=None, device="cuda"):
    """Load pretrained model and tokenizer from local path or HuggingFace."""
    if model_path:
        print(f"Loading model from {model_path}...")
        source = model_path
    elif model_name:
        print(f"Loading model from HuggingFace: {model_name}...")
        source = model_name
    else:
        raise ValueError("Either --model_path or --model_name must be provided")
    
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tokenizer


def predict_single(text, model, tokenizer, device, max_length=128, label_map=None):
    """Predict label for single text."""
    label_map = label_map or {0: "none", 1: "hate"}
    
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(logits, dim=1).item()
        pred_prob = probs[0, pred_idx].item()
    
    return {
        "label": label_map[pred_idx],
        "label_idx": pred_idx,
        "probability": pred_prob,
        "all_probs": probs[0].cpu().numpy()
    }


def predict_batch(texts, model, tokenizer, device, max_length=128, label_map=None):
    """Predict labels for batch of texts."""
    label_map = label_map or {0: "none", 1: "hate"}
    predictions = []
    
    for text in tqdm(texts, desc="Predicting"):
        pred = predict_single(text, model, tokenizer, device, max_length, label_map)
        predictions.append(pred)
    
    return predictions


def load_dataset_test(dataset_name):
    """Load test split from HuggingFace dataset."""
    print(f"Loading test set from {dataset_name}...")
    ds = load_dataset(dataset_name)
    
    # Check available splits
    available_splits = list(ds.keys())
    print(f"  Available splits: {available_splits}")
    
    # Use test split if available, otherwise validation
    split_name = "test" if "test" in available_splits else "validation"
    print(f"  Using split: {split_name}")
    
    split_data = ds[split_name]
    print(f"  Available columns: {split_data.column_names}")
    
    # Try common column names for text
    text_col = None
    for col in ["text", "free_text", "content", "sentence"]:
        if col in split_data.column_names:
            text_col = col
            break
    
    if not text_col:
        # If no common name found, use first string column
        for col in split_data.column_names:
            if split_data.features[col].dtype == "string":
                text_col = col
                break
    
    if not text_col:
        raise ValueError(f"Could not find text column. Available: {split_data.column_names}")
    
    print(f"  Text column: {text_col}")
    return split_data[text_col], split_data.column_names, text_col


def main():
    parser = argparse.ArgumentParser(description="Inference with trained encoder model")
    
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to trained model directory")
    parser.add_argument("--model_name", type=str, default=None,
                       help="HuggingFace model name (e.g., tarudesu/ViSoBERT-HSD)")
    parser.add_argument("--text", type=str, default=None,
                       help="Single text to predict")
    parser.add_argument("--dataset", type=str, default=None,
                       help="HuggingFace dataset name - will use test split")
    parser.add_argument("--input_csv", type=str, default=None,
                       help="CSV file with texts to predict")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Column name in CSV/dataset containing text")
    parser.add_argument("--output_csv", type=str, default=None,
                       help="Output CSV with predictions")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Max sequence length")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not (args.model_path or args.model_name):
        parser.error("Either --model_path or --model_name must be provided")
    
    if not (args.text or args.dataset or args.input_csv):
        parser.error("Either --text, --dataset, or --input_csv must be provided")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.model_name, device)
    label_map = {0: "none", 1: "hate"}
    
    # Single text inference
    if args.text:
        print(f"\n Text: {args.text}")
        result = predict_single(args.text, model, tokenizer, device, args.max_length, label_map)
        print(f"  Label: {result['label']} (confidence: {result['probability']:.4f})")
        print(f"  Probabilities: none={result['all_probs'][0]:.4f}, hate={result['all_probs'][1]:.4f}\n")
    
    # Dataset test split inference
    if args.dataset:
        print(f"\n Loading test set from {args.dataset}...")
        texts, columns, text_col = load_dataset_test(args.dataset)
        print(f"  Found {len(texts)} test samples\n")
        
        # Check if true labels exist
        split_data = load_dataset(args.dataset)
        split_name = "test" if "test" in split_data else "validation"
        split = split_data[split_name]
        
        true_labels = None
        label_col = None
        for col in ["label", "target", "class"]:
            if col in split.column_names:
                true_labels = split[col]
                label_col = col
                break
        
        print(f"Predicting {len(texts)} texts...")
        predictions = predict_batch(texts, model, tokenizer, device, args.max_length, label_map)
        
        # Create output dataframe
        df = pd.DataFrame({
            "text": texts,
            "pred_label": [p["label"] for p in predictions],
            "pred_label_idx": [p["label_idx"] for p in predictions],
            "pred_prob": [p["probability"] for p in predictions],
            "pred_prob_none": [p["all_probs"][0] for p in predictions],
            "pred_prob_hate": [p["all_probs"][1] for p in predictions],
        })
        
        # Add true labels if available
        if true_labels is not None:
            # Convert true labels to label names
            true_label_names = []
            for lbl in true_labels:
                if isinstance(lbl, str):
                    true_label_names.append(lbl.lower())
                else:
                    true_label_names.append(label_map.get(lbl, str(lbl)))
            df["true_label"] = true_label_names
        
        # Save if output specified
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"\n Predictions saved to {args.output_csv}")
        
        # Print summary
        print(f"\n Prediction Summary:")
        print(df[["pred_label", "pred_prob"]].describe())
        print(f"\nLabel distribution:")
        print(df["pred_label"].value_counts())
        
        # Classification report if true labels available
        if true_labels is not None:
            print(f"\n Classification Report:")
            print(classification_report(df["true_label"], df["pred_label"], digits=4))
            
            # Metrics
            acc = accuracy_score(df["true_label"], df["pred_label"])
            f1 = f1_score(df["true_label"], df["pred_label"], average="macro")
            print(f"\nAccuracy: {acc:.4f}")
            print(f"Macro F1: {f1:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(df["true_label"], df["pred_label"])
            print(f"\nConfusion Matrix:")
            print(cm)
    
    # CSV inference
    if args.input_csv:
        print(f"\n Loading texts from {args.input_csv}...")
        df = pd.read_csv(args.input_csv)
        texts = df[args.text_column].tolist()
        print(f"  Found {len(texts)} texts\n")
        
        print(f"Predicting {len(texts)} texts...")
        predictions = predict_batch(texts, model, tokenizer, device, args.max_length, label_map)
        
        # Add predictions to dataframe
        df["pred_label"] = [p["label"] for p in predictions]
        df["pred_label_idx"] = [p["label_idx"] for p in predictions]
        df["pred_prob"] = [p["probability"] for p in predictions]
        df["pred_prob_none"] = [p["all_probs"][0] for p in predictions]
        df["pred_prob_hate"] = [p["all_probs"][1] for p in predictions]
        
        # Save if output specified
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"\n Predictions saved to {args.output_csv}")
        
        # Print summary
        print(f"\n Prediction Summary:")
        print(df[["pred_label", "pred_prob"]].describe())
        print(f"\nLabel distribution:")
        print(df["pred_label"].value_counts())
        
        # Classification report if true label column exists
        for label_col in ["label", "true_label", "target", "class"]:
            if label_col in df.columns:
                print(f"\n Classification Report (using '{label_col}' as ground truth):")
                # Convert to string labels if numeric
                true_labels = []
                for lbl in df[label_col]:
                    if isinstance(lbl, str):
                        true_labels.append(lbl.lower())
                    else:
                        true_labels.append(label_map.get(lbl, str(lbl)))
                
                print(classification_report(true_labels, df["pred_label"], digits=4))
                
                # Metrics
                acc = accuracy_score(true_labels, df["pred_label"])
                f1 = f1_score(true_labels, df["pred_label"], average="macro")
                print(f"\nAccuracy: {acc:.4f}")
                print(f"Macro F1: {f1:.4f}")
                
                # Confusion matrix
                cm = confusion_matrix(true_labels, df["pred_label"])
                print(f"\nConfusion Matrix:")
                print(cm)
                break


if __name__ == "__main__":
    main()
