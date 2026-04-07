"""Validate project directory structure and essential files.

Ensures the project has all expected directories and files.
No external dependencies required — pure path checks.
"""

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDirectoryStructure:
    """Verify all expected directories exist."""

    @pytest.mark.parametrize(
        "directory",
        [
            "src",
            "scripts",
            "notebooks",
            "docs",
        ],
    )
    def test_directory_exists(self, project_root, directory):
        d = project_root / directory
        assert d.is_dir(), f"Expected directory '{directory}/' not found at {d}"

    def test_models_dir_exists(self, project_root):
        """models/ may be gitignored but should exist locally if weights are present."""
        d = project_root / "models"
        # Not required for CI, just informational
        if not d.is_dir():
            pytest.skip("models/ directory not present (gitignored)")


class TestEssentialFiles:
    """Verify critical project files exist."""

    @pytest.mark.parametrize(
        "filename",
        [
            "README.md",
            "requirements.txt",
            "setup.py",
            ".gitignore",
        ],
    )
    def test_root_file_exists(self, project_root, filename):
        f = project_root / filename
        assert f.is_file(), f"Expected file '{filename}' not found at {f}"

    def test_readme_not_empty(self, readme_file):
        content = readme_file.read_text()
        assert len(content) > 100, "README.md appears to be empty or too short"


class TestSourceModules:
    """Verify all expected Python source modules exist under src/."""

    @pytest.mark.parametrize(
        "module",
        [
            "__init__.py",
            "config.py",
            "data_loader.py",
            "model.py",
            "utils.py",
            "train_bert.py",
            "train_t5.py",
            "pre_train_t5.py",
            "t5_data_collator.py",
            "evaluate.py",
            "inference.py",
            "label_dataset.py",
        ],
    )
    def test_source_module_exists(self, src_dir, module):
        f = src_dir / module
        assert f.is_file(), f"Expected source module 'src/{module}' not found"


class TestShellScripts:
    """Verify all expected shell scripts exist under scripts/."""

    @pytest.mark.parametrize(
        "script",
        [
            "run_pretrain_t5.sh",
            "run_train_t5.sh",
            "run_train_bert.sh",
        ],
    )
    def test_script_exists(self, scripts_dir, script):
        f = scripts_dir / script
        assert f.is_file(), f"Expected script 'scripts/{script}' not found"


class TestNotebooks:
    """Verify notebooks."""

    def test_demo_notebook_exists(self, notebooks_dir):
        nb = notebooks_dir / "demo.ipynb"
        assert nb.is_file(), "Expected notebooks/demo.ipynb not found"
