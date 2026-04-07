"""Validate shell script safety guards and conventions.

Every .sh file in scripts/ must:
- Start with #!/bin/bash shebang
- Contain a usage comment
- Define default values before parsing arguments

No external dependencies — pure file inspection.
"""

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"


def _get_shell_scripts():
    if not _SCRIPTS_DIR.is_dir():
        return []
    return sorted(_SCRIPTS_DIR.glob("*.sh"))


class TestShellScriptGuards:
    """Verify shell scripts follow project conventions."""

    @pytest.mark.parametrize(
        "script",
        [pytest.param(p, id=p.name) for p in _get_shell_scripts()],
    )
    def test_has_bash_shebang(self, script):
        first_line = script.read_text().splitlines()[0] if script.read_text().strip() else ""
        assert first_line == "#!/bin/bash", (
            f"{script.name}: first line must be '#!/bin/bash', got '{first_line}'"
        )

    @pytest.mark.parametrize(
        "script",
        [pytest.param(p, id=p.name) for p in _get_shell_scripts()],
    )
    def test_has_usage_comment(self, script):
        content = script.read_text()
        assert "Usage:" in content or "usage:" in content, (
            f"{script.name}: should contain a 'Usage:' comment"
        )

    @pytest.mark.parametrize(
        "script",
        [pytest.param(p, id=p.name) for p in _get_shell_scripts()],
    )
    def test_has_default_values(self, script):
        content = script.read_text()
        assert "Default values" in content or "default" in content.lower(), (
            f"{script.name}: should define default values section"
        )

    @pytest.mark.parametrize(
        "script",
        [pytest.param(p, id=p.name) for p in _get_shell_scripts()],
    )
    def test_parses_command_line_arguments(self, script):
        content = script.read_text()
        assert "while" in content and "case" in content, (
            f"{script.name}: should have argument parsing (while/case block)"
        )

    @pytest.mark.parametrize(
        "script",
        [pytest.param(p, id=p.name) for p in _get_shell_scripts()],
    )
    def test_no_hardcoded_absolute_paths(self, script):
        """Scripts should not hardcode machine-specific paths."""
        content = script.read_text()
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "/home/" not in stripped and "/Users/" not in stripped, (
                f"{script.name}:{i}: hardcoded absolute path found: {stripped}"
            )
