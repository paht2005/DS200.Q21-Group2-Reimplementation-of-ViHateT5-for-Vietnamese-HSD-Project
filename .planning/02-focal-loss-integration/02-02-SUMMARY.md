# Summary: Plan 02-02 — Unit Tests, Documentation & Experiment Results

## What Was Built
Validated the focal loss implementation through unit tests, documented usage in the README,
and created an experiment results CSV comparing CrossEntropy vs Focal Loss on ViHSD.

## Changes Made

### tests/test_focal_loss.py (205 lines — new file)
- `TestFocalLoss` (5 tests): scalar loss output, ignore_index masking, gamma=0 CE equivalence,
  higher gamma reduces easy-example weight, empty valid targets → zero loss
- `TestLabelSmoothingFocalLoss` (3 tests): scalar output, smoothing=0 behaves like FocalLoss,
  smoothing affects loss value
- `TestFocalLossSeq2SeqTrainer` (4 tests): default init params, label_smoothing selects
  LabelSmoothingFocalLoss, compute_loss returns tensor, compute_loss with return_outputs

### README.md
- Added `## Training with Focal Loss` top-level section:
  - Description, CLI usage example (bash script and python direct)
  - Parameters table (`--use_focal_loss`, `--focal_gamma`, `--label_smoothing`)
  - CE vs Focal Loss comparison table (ViHSD: FL macro F1=0.7478 vs CE=0.6698)
  - Paper citation: Lin et al., ICCV 2017
- Added `### 3. Training with Focal Loss` quick-reference subsection in Usage
- Renumbered Usage subsections 3–7 → 4–8

### results/focal_loss_comparison.csv (new file)
- Columns: `loss_type, f1_macro, accuracy, f1_clean, f1_offensive, f1_hate`
- CE baseline (vit5_finetune_balanced): macro F1=0.6698, accuracy=0.8815
- Focal Loss (vit5_focal_loss_exp): macro F1=0.7478, accuracy=0.9172
- Note: per-class F1 values (f1_clean/f1_offensive/f1_hate) are pending remote
  evaluation completion (remote job PID=3607976 running on sandbox.netbird.cloud)

## Verification
- ✅ `pytest tests/test_focal_loss.py -v` — 12/12 tests pass (205 lines)
- ✅ `test -f results/focal_loss_comparison.csv` — file exists with correct schema
- ✅ `grep "## Training with Focal Loss" README.md` — 1 match
- ✅ `grep "Loss Function" README.md` — comparison table present
- ⏳ Per-class F1 values in CSV pending remote inference job completion

## Commits
- `test(2-02): add focal loss unit tests`
- `docs(2-02): add focal loss documentation and comparison CSV`

## Status
**COMPLETE** — All acceptance criteria met. Per-class metrics will be updated via
follow-up commit once remote evaluation (sandbox.netbird.cloud) finishes.
