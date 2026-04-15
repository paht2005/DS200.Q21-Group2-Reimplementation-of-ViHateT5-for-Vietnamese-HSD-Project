"""
Error analysis module for Vietnamese hate speech detection models.

Generates confusion matrices, per-class F1 breakdown, failure case analysis,
and statistical significance tests for comprehensive model evaluation (REQ-03).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    precision_score, recall_score, accuracy_score
)


RESULTS_DIR = "results/images"
ANALYSIS_DIR = "results/analysis"


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, labels, class_names, title, filename):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"{title} — Counts")

    # Normalized (%)
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"{title} — Normalized")

    fig.tight_layout()
    filepath = os.path.join(RESULTS_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")

    return cm, cm_normalized


def per_class_f1_report(y_true, y_pred, class_names, task_name):
    """Generate per-class precision, recall, F1 and return as DataFrame."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).transpose()
    df = df.round(4)
    print(f"\n  {task_name} — Per-Class Metrics:")
    print(df.to_string())
    return df


def plot_per_class_f1(vihsd_report, victsd_report, filename="per_class_f1_breakdown.png"):
    """Bar chart showing per-class F1 for ViHSD and ViCTSD."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ViHSD
    vihsd_classes = ["CLEAN", "OFFENSIVE", "HATE"]
    vihsd_f1 = [vihsd_report.loc[c, "f1-score"] for c in vihsd_classes]
    colors_vihsd = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars1 = axes[0].bar(vihsd_classes, vihsd_f1, color=colors_vihsd, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars1, vihsd_f1):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_title("ViHSD — Per-Class F1")
    axes[0].set_ylim(0, 1.0)
    axes[0].axhline(y=vihsd_report.loc["macro avg", "f1-score"], color="gray",
                    linestyle="--", label=f'Macro F1 = {vihsd_report.loc["macro avg", "f1-score"]:.4f}')
    axes[0].legend()

    # ViCTSD
    victsd_classes = ["NONE", "TOXIC"]
    victsd_f1 = [victsd_report.loc[c, "f1-score"] for c in victsd_classes]
    colors_victsd = ["#3498db", "#e74c3c"]
    bars2 = axes[1].bar(victsd_classes, victsd_f1, color=colors_victsd, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars2, victsd_f1):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("F1-Score")
    axes[1].set_title("ViCTSD — Per-Class F1")
    axes[1].set_ylim(0, 1.0)
    axes[1].axhline(y=victsd_report.loc["macro avg", "f1-score"], color="gray",
                    linestyle="--", label=f'Macro F1 = {victsd_report.loc["macro avg", "f1-score"]:.4f}')
    axes[1].legend()

    fig.suptitle("Per-Class F1 Breakdown — ViHateT5", fontsize=13, fontweight="bold")
    fig.tight_layout()
    filepath = os.path.join(RESULTS_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def analyze_misclassifications(texts, y_true, y_pred, class_names, task_name, top_n=10):
    """Identify and categorize misclassification patterns."""
    mismatches = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            mismatches.append({
                "text": texts[i][:200],  # Truncate long texts
                "true_label": class_names[true],
                "predicted_label": class_names[pred],
                "error_type": f"{class_names[true]} → {class_names[pred]}",
            })

    df = pd.DataFrame(mismatches)
    if df.empty:
        print(f"  {task_name}: No misclassifications found!")
        return df

    # Error type distribution
    error_dist = df["error_type"].value_counts()
    print(f"\n  {task_name} — Misclassification Patterns:")
    print(f"    Total errors: {len(df)} / {len(y_true)} ({len(df)/len(y_true)*100:.1f}%)")
    for err_type, count in error_dist.items():
        print(f"    {err_type}: {count} ({count/len(df)*100:.1f}%)")

    # Save top-N failure cases
    top_failures = df.head(top_n)
    filepath = os.path.join(ANALYSIS_DIR, f"{task_name.lower()}_failure_cases.csv")
    df.to_csv(filepath, index=False, encoding="utf-8")
    print(f"  Saved all {len(df)} failure cases: {filepath}")

    return df


def plot_error_distribution(vihsd_errors, victsd_errors, filename="error_distribution.png"):
    """Visualize misclassification pattern distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if not vihsd_errors.empty:
        error_counts = vihsd_errors["error_type"].value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(error_counts)))
        axes[0].barh(error_counts.index, error_counts.values, color=colors)
        for i, (val, name) in enumerate(zip(error_counts.values, error_counts.index)):
            axes[0].text(val + 0.5, i, str(val), va="center", fontsize=9)
        axes[0].set_xlabel("Count")
        axes[0].set_title("ViHSD — Error Distribution")

    if not victsd_errors.empty:
        error_counts = victsd_errors["error_type"].value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(error_counts)))
        axes[1].barh(error_counts.index, error_counts.values, color=colors)
        for i, (val, name) in enumerate(zip(error_counts.values, error_counts.index)):
            axes[1].text(val + 0.5, i, str(val), va="center", fontsize=9)
        axes[1].set_xlabel("Count")
        axes[1].set_title("ViCTSD — Error Distribution")

    fig.suptitle("Misclassification Error Distribution — ViHateT5", fontsize=13, fontweight="bold")
    fig.tight_layout()
    filepath = os.path.join(RESULTS_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def bootstrap_confidence_interval(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95, seed=42):
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            scores.append(score)
        except Exception:
            continue

    scores = np.array(scores)
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    mean = np.mean(scores)

    return mean, lower, upper


def statistical_significance_report(results_dict, filename="statistical_significance.csv"):
    """Generate bootstrap CI report for all tasks and save."""
    rows = []
    for task_name, (y_true, y_pred) in results_dict.items():
        # Macro F1
        mean_f1, lower_f1, upper_f1 = bootstrap_confidence_interval(
            y_true, y_pred,
            lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
        )
        # Accuracy
        mean_acc, lower_acc, upper_acc = bootstrap_confidence_interval(
            y_true, y_pred, accuracy_score
        )

        rows.append({
            "Task": task_name,
            "Metric": "Macro F1",
            "Mean": round(mean_f1, 4),
            "95% CI Lower": round(lower_f1, 4),
            "95% CI Upper": round(upper_f1, 4),
        })
        rows.append({
            "Task": task_name,
            "Metric": "Accuracy",
            "Mean": round(mean_acc, 4),
            "95% CI Lower": round(lower_acc, 4),
            "95% CI Upper": round(upper_acc, 4),
        })

    df = pd.DataFrame(rows)
    filepath = os.path.join(ANALYSIS_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"\n  Saved: {filepath}")
    print(df.to_string(index=False))
    return df


def run_full_error_analysis(
    vihsd_true, vihsd_pred, vihsd_texts,
    victsd_true, victsd_pred, victsd_texts,
):
    """Run complete error analysis pipeline."""
    ensure_dirs()

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS — ViHateT5")
    print("=" * 80)

    # --- ViHSD ---
    vihsd_classes = ["CLEAN", "OFFENSIVE", "HATE"]
    print("\n[1/6] ViHSD Confusion Matrix")
    plot_confusion_matrix(
        vihsd_true, vihsd_pred,
        labels=[0, 1, 2], class_names=vihsd_classes,
        title="ViHSD — Hate Speech Detection",
        filename="confusion_matrix_vihsd.png"
    )

    print("\n[2/6] ViCTSD Confusion Matrix")
    victsd_classes = ["NONE", "TOXIC"]
    plot_confusion_matrix(
        victsd_true, victsd_pred,
        labels=[0, 1], class_names=victsd_classes,
        title="ViCTSD — Toxic Speech Detection",
        filename="confusion_matrix_victsd.png"
    )

    print("\n[3/6] Per-Class F1 Reports")
    vihsd_report = per_class_f1_report(vihsd_true, vihsd_pred, vihsd_classes, "ViHSD")
    victsd_report = per_class_f1_report(victsd_true, victsd_pred, victsd_classes, "ViCTSD")

    print("\n[4/6] Per-Class F1 Charts")
    plot_per_class_f1(vihsd_report, victsd_report)

    print("\n[5/6] Misclassification Analysis")
    vihsd_errors = analyze_misclassifications(
        vihsd_texts, vihsd_true, vihsd_pred, vihsd_classes, "ViHSD"
    )
    victsd_errors = analyze_misclassifications(
        victsd_texts, victsd_true, victsd_pred, victsd_classes, "ViCTSD"
    )
    plot_error_distribution(vihsd_errors, victsd_errors)

    print("\n[6/6] Statistical Significance (Bootstrap 95% CI)")
    statistical_significance_report({
        "ViHSD": (vihsd_true, vihsd_pred),
        "ViCTSD": (victsd_true, victsd_pred),
    })

    # Save per-class reports as CSV
    vihsd_report.to_csv(os.path.join(ANALYSIS_DIR, "vihsd_per_class_report.csv"))
    victsd_report.to_csv(os.path.join(ANALYSIS_DIR, "victsd_per_class_report.csv"))

    print("\n" + "=" * 80)
    print("Error analysis complete! Results saved to results/analysis/ and results/images/")
    print("=" * 80)
