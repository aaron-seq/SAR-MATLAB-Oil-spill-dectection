"""Guards that the repository ships everything the package needs.

These tests exist because of a real failure: an unanchored ``data/`` rule in
`.gitignore` (intended for datasets) also matched ``src/sar_oil_spill/data/``,
so the synthetic generator and dataset loader were never committed. Every local
test passed -- the files were present on disk -- while a fresh clone could not
import ``sar_oil_spill.data`` at all.

Nothing that only inspects the working tree can catch that class of bug, so
these tests compare the working tree against what git actually tracks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "sar_oil_spill"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _tracked_files() -> set[Path] | None:
    """Every path git tracks, or ``None`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    return {
        (REPO_ROOT / name).resolve()
        for name in result.stdout.decode().split("\0")
        if name
    }


@pytest.fixture(scope="module")
def tracked() -> set[Path]:
    files = _tracked_files()
    if files is None:
        pytest.skip("Not a git checkout, or git is unavailable.")
    return files


class TestEverySourceFileIsCommitted:
    def test_all_package_modules_are_tracked_by_git(self, tracked):
        """A source file that exists locally but is not committed is invisible
        to CI, to Docker and to anyone who clones the repository."""
        on_disk = {p.resolve() for p in PACKAGE_ROOT.rglob("*.py")}
        untracked = sorted(str(p.relative_to(REPO_ROOT)) for p in on_disk - tracked)

        assert not untracked, (
            "These package modules exist on disk but are not tracked by git, so "
            "they will be missing from a fresh clone:\n  "
            + "\n  ".join(untracked)
            + "\nCheck .gitignore for an unanchored pattern matching them."
        )

    def test_no_package_directory_is_gitignored(self):
        """`git check-ignore` is the authority on what would be skipped."""
        directories = [PACKAGE_ROOT, *(p for p in PACKAGE_ROOT.rglob("*") if p.is_dir())]
        directories = [d for d in directories if "__pycache__" not in d.parts]

        result = subprocess.run(
            ["git", "check-ignore", *[str(d) for d in directories]],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=30,
        )

        # check-ignore exits 0 and prints a path only when something IS ignored.
        ignored = result.stdout.decode().strip()
        assert not ignored, f"These package directories are gitignored:\n{ignored}"

    def test_every_subpackage_is_importable(self):
        """Each directory with an ``__init__.py`` must actually import."""
        import importlib

        subpackages = [
            f"sar_oil_spill.{path.parent.name}"
            for path in PACKAGE_ROOT.glob("*/__init__.py")
        ]

        assert subpackages, "Expected to find subpackages under the package root."
        for name in subpackages:
            importlib.import_module(name)


class TestDeclaredPackagingIsComplete:
    def test_wheel_target_covers_the_package(self):
        """`pyproject.toml` must point the wheel build at the real package."""
        content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'packages = ["src/sar_oil_spill"]' in content

    def test_console_script_entry_point_resolves(self):
        """The `sar-oil-spill` entry point must import and be callable."""
        from sar_oil_spill.cli import main

        assert callable(main)

    def test_gitignore_dataset_rules_stay_anchored(self, tracked):
        """The rule that caused the outage must remain root-anchored."""
        lines = [
            line.strip()
            for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        assert "data/" not in lines, (
            "An unanchored 'data/' rule also matches src/sar_oil_spill/data/. "
            "Use '/data/' so it only applies at the repository root."
        )
        _ = tracked
