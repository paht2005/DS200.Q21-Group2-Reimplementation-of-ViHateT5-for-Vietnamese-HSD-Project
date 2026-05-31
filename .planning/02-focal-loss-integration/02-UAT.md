---
status: testing
phase: 02-focal-loss-integration
source: [02-01-SUMMARY.md, 02-02-PLAN.md]
started: 2026-05-31T00:00:00Z
updated: 2026-05-31T00:00:00Z
---

## Current Test
<!-- OVERWRITE each test - shows where we are -->

number: 1
name: Focal Loss CLI Flags Visible in Help
expected: |
  Running `python src/train_t5.py --help` shows three focal loss arguments:
  --use_focal_loss, --focal_gamma FOCAL_GAMMA, and --label_smoothing LABEL_SMOOTHING,
  each with a description.
awaiting: user response

## Tests

### 1. Focal Loss CLI Flags Visible in Help
expected: Running `python src/train_t5.py --help` shows --use_focal_loss, --focal_gamma, and --label_smoothing with descriptions.
result: [pending]

### 2. Validation Rejects Invalid Flag Combinations
expected: Running `python src/train_t5.py --focal_gamma 3.0 <other required args>` WITHOUT --use_focal_loss exits with an error like "require --use_focal_loss flag". Running with `--focal_gamma -1.0 --use_focal_loss` exits with an error about gamma must be >= 0.
result: [pending]

### 3. Unit Tests Pass
expected: Running `python -m pytest tests/test_focal_loss.py -v` shows all 12 tests pass with no failures.
result: [pending]

### 4. README Has Focal Loss Documentation Section
expected: README.md contains a dedicated "Training with Focal Loss" (or similar) section that shows the CLI flags, their defaults, and at least one usage example command.
result: [pending]

### 5. README Contains CE vs Focal Loss Comparison Table
expected: README.md contains a table comparing Cross-Entropy baseline vs Focal Loss results with actual metrics (F1, accuracy, or similar).
result: [pending]

### 6. Experiment Results CSV Exists
expected: The file `results/focal_loss_comparison.csv` exists and contains rows with metric columns comparing CE and focal loss runs.
result: [pending]

## Summary

total: 6
passed: 0
issues: 0
pending: 6
skipped: 0
blocked: 0

## Gaps

[none yet]
