"""
Phase 1: Run error analysis on ViHateT5 model predictions.
Generates confusion matrices, per-class F1, bootstrap CI for REQ-03.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer
from data_loader import load_vihsd, load_victsd
from error_analysis import run_full_error_analysis

# --- Config ---
MODEL_ID = "models/vit5_finetune_balanced"  # Local model path
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # Smaller for MPS memory

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_ID}")


def predict_t5_batch(model, tokenizer, texts, prefix, batch_size=BATCH_SIZE):
    """Generate T5 predictions in batches."""
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {t}" for t in texts[i:i + batch_size]]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=256
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64, num_beams=1, do_sample=False)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(decoded)
        if (i // batch_size) % 10 == 0:
            print(f"  Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
    return preds


def main():
    # Load model
    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE).eval()
    print(f"  Model loaded on {DEVICE}")

    # Load test datasets
    print("\n[2/4] Loading test datasets...")
    _, _, vihsd_test, _ = load_vihsd()
    _, _, victsd_test, _ = load_victsd()
    print(f"  ViHSD test: {len(vihsd_test)} samples")
    print(f"  ViCTSD test: {len(victsd_test)} samples")

    # Generate predictions
    print("\n[3/4] Generating predictions...")
    vihsd_texts = vihsd_test["free_text"].tolist()
    print("  ViHSD predictions...")
    vihsd_raw_preds = predict_t5_batch(model, tokenizer, vihsd_texts, "hate-speech-detection")

    victsd_texts = victsd_test["Comment"].tolist()
    print("  ViCTSD predictions...")
    victsd_raw_preds = predict_t5_batch(model, tokenizer, victsd_texts, "toxic-speech-detection")

    # Map to numeric labels
    label_map_vihsd = {"CLEAN": 0, "clean": 0, "OFFENSIVE": 1, "offensive": 1, "HATE": 2, "hate": 2}
    vihsd_preds = [label_map_vihsd.get(p.strip(), 2) for p in vihsd_raw_preds]
    vihsd_true = vihsd_test["label_id"].tolist()

    label_map_victsd = {"NONE": 0, "none": 0, "TOXIC": 1, "toxic": 1}
    victsd_preds = [label_map_victsd.get(p.strip(), 0) for p in victsd_raw_preds]
    victsd_true = victsd_test["Toxicity"].tolist()

    # Save raw predictions for reuse in other phases
    os.makedirs("results/analysis", exist_ok=True)
    pd.DataFrame({
        "text": vihsd_texts,
        "true_label": vihsd_true,
        "pred_label": vihsd_preds,
        "raw_pred": vihsd_raw_preds,
    }).to_csv("results/analysis/vihsd_predictions.csv", index=False)

    pd.DataFrame({
        "text": victsd_texts,
        "true_label": victsd_true,
        "pred_label": victsd_preds,
        "raw_pred": victsd_raw_preds,
    }).to_csv("results/analysis/victsd_predictions.csv", index=False)
    print("  Saved raw predictions to results/analysis/")

    # Run full error analysis
    print("\n[4/4] Running error analysis...")
    run_full_error_analysis(
        vihsd_true=vihsd_true,
        vihsd_pred=vihsd_preds,
        vihsd_texts=vihsd_texts,
        victsd_true=victsd_true,
        victsd_pred=victsd_preds,
        victsd_texts=victsd_texts,
    )

    # Print summary
    from sklearn.metrics import f1_score, accuracy_score
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE — Error Analysis Results")
    print("=" * 60)
    print(f"ViHSD  — Macro F1: {f1_score(vihsd_true, vihsd_preds, average='macro'):.4f}, Acc: {accuracy_score(vihsd_true, vihsd_preds):.4f}")
    print(f"ViCTSD — Macro F1: {f1_score(victsd_true, victsd_preds, average='macro'):.4f}, Acc: {accuracy_score(victsd_true, victsd_preds):.4f}")

    print("\nGenerated files:")
    for d in ["results/analysis", "results/images"]:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                fpath = os.path.join(d, f)
                size = os.path.getsize(fpath)
                print(f"  {d}/{f} ({size:,} bytes)")


if __name__ == "__main__":
    main()
