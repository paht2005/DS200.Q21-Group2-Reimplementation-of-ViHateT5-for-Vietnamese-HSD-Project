# Testing Guide

Guide for running unit tests, creating branches, and opening pull requests.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Run Pytest Locally](#step-1-run-pytest-locally)
- [Step 2: Create a New Branch](#step-2-create-a-new-branch)
- [Step 3: Commit and Push](#step-3-commit-and-push)
- [Step 4: Create a Pull Request on GitHub](#step-4-create-a-pull-request-on-github)
- [Workflow Summary](#workflow-summary)
- [Test Files Overview](#test-files-overview)

---

## Prerequisites

Make sure all dependencies are installed:

```bash
# Create virtual environment (if not already created)
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Step 1: Run Pytest Locally

> **Important**: Always ensure all tests pass before creating a branch or pull request.

### Run the full test suite

```bash
# Run from the project root directory
python -m pytest tests/ -v
```

### Run with short output

```bash
python -m pytest tests/ --tb=short
```

### Run a specific test file

```bash
python -m pytest tests/test_config.py -v
python -m pytest tests/test_utils.py -v
```

### Run a specific test case

```bash
python -m pytest tests/test_config.py::TestTrainConfigDefaults::test_default_model_name -v
```

### Expected output

```
================== 123 passed, 1 skipped, 0 failed ==================
```

If any test fails, read the error message and fix the code before continuing.

---

## Step 2: Create a New Branch

After all tests pass, create a new branch for your changes:

```bash
# Make sure you are on main and up to date
git checkout main
git pull origin main

# Create a new branch with a descriptive name
git checkout -b <branch-name>
```

**Branch naming conventions:**

| Type | Format | Example |
|:-----|:-------|:--------|
| New feature | `feature/<description>` | `feature/add-focal-loss` |
| Bug fix | `fix/<description>` | `fix/data-loader-crash` |
| Improvement | `improve/<description>` | `improve/training-speed` |
| Documentation | `docs/<description>` | `docs/update-readme` |

---

## Step 3: Commit and Push

```bash
# Stage all changed files
git add .

# Review what will be committed
git status

# Commit with a clear message
git commit -m "Short description of changes"

# Push the branch to remote
git push origin <branch-name>
```

---

## Step 4: Create a Pull Request on GitHub

1. Go to the repository on GitHub
2. GitHub will show a **"Compare & pull request"** banner — click it
3. Fill in the details:
   - **Title**: Short description of changes
   - **Description**: Explain what was changed and why
   - **Reviewers**: Add team members for code review
4. Click **"Create pull request"**
5. Wait for review and merge

---

## Workflow Summary

```
1. Run pytest → All tests pass ✓
2. Create a new branch
3. Commit & push changes
4. Create a pull request on GitHub
5. Review & merge
```

---

## Test Files Overview

| File | Description | Tests |
|:-----|:------------|:-----:|
| `test_config.py` | TrainConfig dataclass (defaults, serialization) | 12 |
| `test_data_loader.py` | TextDataset and dataset routing functions | 10 |
| `test_evaluate.py` | T5 evaluation helper functions | 15 |
| `test_inference.py` | Encoder model inference pipeline | 5 |
| `test_model.py` | Model building utilities | 5 |
| `test_utils.py` | compute_metrics and set_seed | 8 |
| `test_t5_collator.py` | T5 span corruption data collator | 8 |
| `test_project_structure.py` | Project directory structure validation | 22 |
| `test_scripts_guard.py` | Shell script guards (shebang, hardcoded paths) | 15 |
| `test_quality_gates.py` | Quality gates (secrets, consistency, README) | 23 |
