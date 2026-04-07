"""Quality gate tests — project-level consistency checks.

Validates:
- requirements.txt has no duplicate packages
- .gitignore covers models/ and common artifacts
- README references match actual source files
- No secrets or credentials in tracked files

No external dependencies — pure file scanning.
"""

import re
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRequirementsConsistency:
    """Validate requirements.txt quality."""

    def test_requirements_exists(self, requirements_file):
        assert requirements_file.is_file()

    def test_no_duplicate_packages(self, requirements_file):
        lines = requirements_file.read_text().splitlines()
        packages = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Extract package name (before ==, >=, <=, ~=, etc.)
            pkg = re.split(r"[><=~!\[]", line)[0].strip().lower()
            if pkg:
                packages.append(pkg)
        duplicates = [p for p in set(packages) if packages.count(p) > 1]
        assert not duplicates, f"Duplicate packages in requirements.txt: {duplicates}"

    def test_no_pinned_file_paths(self, requirements_file):
        """requirements.txt should not reference absolute file paths."""
        content = requirements_file.read_text()
        assert "/home/" not in content and "/Users/" not in content, (
            "requirements.txt contains hardcoded file paths"
        )


class TestGitignoreConsistency:
    """Validate .gitignore covers large/sensitive files."""

    def test_gitignore_exists(self, gitignore_file):
        assert gitignore_file.is_file()

    def test_models_dir_ignored(self, gitignore_file):
        content = gitignore_file.read_text()
        assert "models/" in content, ".gitignore should exclude models/ directory"

    def test_env_file_ignored(self, gitignore_file):
        content = gitignore_file.read_text()
        assert ".env" in content, ".gitignore should exclude .env file"

    def test_pycache_ignored(self, gitignore_file):
        content = gitignore_file.read_text()
        assert "__pycache__" in content, ".gitignore should exclude __pycache__/"

    def test_checkpoint_files_ignored(self, gitignore_file):
        content = gitignore_file.read_text()
        assert ".ckpt" in content or "*.ckpt" in content, (
            ".gitignore should exclude checkpoint files"
        )


class TestReadmeConsistency:
    """Validate README references match actual project files."""

    def test_readme_references_existing_source_files(self, readme_file, src_dir):
        """All src/ files mentioned in README should actually exist."""
        content = readme_file.read_text()
        # Find all src/*.py references
        referenced = re.findall(r"src/(\w+\.py)", content)
        for module in set(referenced):
            f = src_dir / module
            assert f.is_file(), (
                f"README references 'src/{module}' but file does not exist"
            )

    def test_readme_references_existing_scripts(self, readme_file, scripts_dir):
        """All scripts/ files mentioned in README should actually exist."""
        content = readme_file.read_text()
        referenced = re.findall(r"scripts/(\w+\.sh)", content)
        for script in set(referenced):
            f = scripts_dir / script
            assert f.is_file(), (
                f"README references 'scripts/{script}' but file does not exist"
            )

    def test_readme_has_table_of_contents(self, readme_file):
        content = readme_file.read_text()
        assert "## " in content, "README should have section headers"

    def test_readme_has_installation_section(self, readme_file):
        content = readme_file.read_text()
        assert "Installation" in content or "install" in content.lower()

    def test_readme_has_usage_section(self, readme_file):
        content = readme_file.read_text()
        assert "Usage" in content or "usage" in content.lower()


class TestNoSecrets:
    """Ensure no secrets or credentials are committed."""

    @pytest.mark.parametrize(
        "pattern,description",
        [
            (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key"),
            (r"hf_[a-zA-Z0-9]{20,}", "HuggingFace token"),
            (r"ghp_[a-zA-Z0-9]{20,}", "GitHub personal access token"),
            (r"AKIA[A-Z0-9]{16}", "AWS access key"),
        ],
    )
    def test_no_secrets_in_source(self, src_dir, pattern, description):
        for py_file in src_dir.glob("*.py"):
            content = py_file.read_text()
            matches = re.findall(pattern, content)
            assert not matches, (
                f"Possible {description} found in {py_file.name}: {matches[:3]}"
            )

    def test_no_secrets_in_scripts(self, scripts_dir):
        patterns = [
            (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key"),
            (r"hf_[a-zA-Z0-9]{20,}", "HuggingFace token"),
        ]
        for sh_file in scripts_dir.glob("*.sh"):
            content = sh_file.read_text()
            for pattern, desc in patterns:
                matches = re.findall(pattern, content)
                assert not matches, (
                    f"Possible {desc} found in {sh_file.name}"
                )


class TestSourceCodeQuality:
    """Basic source code quality checks."""

    def test_all_python_files_have_docstrings(self, src_dir):
        """Each Python module should start with a docstring or descriptive comment."""
        for py_file in src_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            content = py_file.read_text().strip()
            has_docstring = content.startswith('"""') or content.startswith("'''")
            has_comment = content.startswith("#")
            assert has_docstring or has_comment, (
                f"src/{py_file.name}: missing module-level docstring or comment header"
            )

    def test_no_print_debugging(self, src_dir):
        """Source files should not contain obvious debug prints."""
        debug_patterns = [
            r"\bprint\s*\(\s*[\"']DEBUG",
            r"\bprint\s*\(\s*[\"']TODO",
            r"\bbreakpoint\s*\(\s*\)",
            r"\bpdb\.set_trace\s*\(\s*\)",
        ]
        for py_file in src_dir.glob("*.py"):
            content = py_file.read_text()
            for pattern in debug_patterns:
                matches = re.findall(pattern, content)
                assert not matches, (
                    f"src/{py_file.name}: debug statement found: {matches[0]}"
                )
