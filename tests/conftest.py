"""Shared pytest fixtures for DS200.Q21 Vietnamese Hate Speech Detection tests."""

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Auto-skip markers for GPU-dependent tests
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="Requires CUDA GPU",
)

requires_transformers = pytest.mark.skipif(
    "transformers" not in {m.split(".")[0] for m in sys.modules}
    and __import__("importlib").util.find_spec("transformers") is None,
    reason="Requires transformers library",
)


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def src_dir(project_root):
    return project_root / "src"


@pytest.fixture
def scripts_dir(project_root):
    return project_root / "scripts"


@pytest.fixture
def notebooks_dir(project_root):
    return project_root / "notebooks"


@pytest.fixture
def models_dir(project_root):
    return project_root / "models"


@pytest.fixture
def requirements_file(project_root):
    return project_root / "requirements.txt"


@pytest.fixture
def gitignore_file(project_root):
    return project_root / ".gitignore"


@pytest.fixture
def readme_file(project_root):
    return project_root / "README.md"
