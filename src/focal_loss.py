"""
Focal Loss implementation for T5 multi-task training.

Addresses class imbalance in Vietnamese hate speech datasets,
especially ViHSD where CLEAN >> OFFENSIVE >> HATE.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer


class FocalLoss(nn.Module):
    """
    Focal Loss for token-level classification in Seq2Seq models.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Higher gamma = more focus on hard examples.
               gamma=0 is equivalent to standard cross-entropy.
        alpha: Optional class weighting (not used for token-level).
        ignore_index: Index to ignore in loss computation (default: -100).
    """

    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len) with ignore_index for padding
        """
        vocab_size = logits.size(-1)

        # Flatten
        logits_flat = logits.view(-1, vocab_size)  # (B*T, V)
        targets_flat = targets.view(-1)  # (B*T,)

        # Mask out ignored indices
        mask = targets_flat != self.ignore_index
        logits_flat = logits_flat[mask]
        targets_flat = targets_flat[mask]

        if logits_flat.numel() == 0:
            return logits_flat.sum()  # Return 0 if no valid targets

        # Compute log probabilities
        log_probs = F.log_softmax(logits_flat, dim=-1)
        probs = torch.exp(log_probs)

        # Gather the probabilities for the target classes
        targets_flat = targets_flat.clamp(0, vocab_size - 1)
        log_pt = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Compute loss
        loss = -focal_weight * log_pt

        return loss.mean()


class LabelSmoothingFocalLoss(nn.Module):
    """
    Focal Loss with label smoothing for better generalization.

    Combines focal loss focusing with label smoothing regularization.

    Args:
        gamma: Focusing parameter for focal loss.
        smoothing: Label smoothing factor (0.0 = no smoothing).
        ignore_index: Index to ignore in loss computation.
    """

    def __init__(self, gamma=2.0, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        vocab_size = logits.size(-1)

        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        mask = targets_flat != self.ignore_index
        logits_flat = logits_flat[mask]
        targets_flat = targets_flat[mask]

        if logits_flat.numel() == 0:
            return logits_flat.sum()

        log_probs = F.log_softmax(logits_flat, dim=-1)
        probs = torch.exp(log_probs)

        targets_flat = targets_flat.clamp(0, vocab_size - 1)
        log_pt = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        # Focal component
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = -focal_weight * log_pt

        # Label smoothing component
        smooth_loss = -log_probs.mean(dim=-1)

        # Combined loss
        loss = (1 - self.smoothing) * focal_loss + self.smoothing * smooth_loss

        return loss.mean()


class FocalLossSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer that uses Focal Loss instead of standard CrossEntropy.

    Usage:
        trainer = FocalLossSeq2SeqTrainer(
            model=model,
            args=training_args,
            focal_gamma=2.0,
            label_smoothing=0.1,
            ...
        )
    """

    def __init__(self, *args, focal_gamma=2.0, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if label_smoothing > 0:
            self.focal_loss_fn = LabelSmoothingFocalLoss(
                gamma=focal_gamma, smoothing=label_smoothing
            )
        else:
            self.focal_loss_fn = FocalLoss(gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.focal_loss_fn(logits, labels)

        # Put labels back for other methods
        inputs["labels"] = labels

        return (loss, outputs) if return_outputs else loss
