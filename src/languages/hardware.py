"""Shared source-discovery rules for hardware language handlers and plugins."""

from __future__ import annotations

import os
from pathlib import Path


CHISEL_EXTENSIONS = frozenset({".scala", ".sc"})
VERILOG_EXTENSIONS = frozenset({".v", ".sv", ".svh"})

SOURCE_SCAN_EXCLUDED_DIRECTORY_NAMES = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
    "fm_agent",
    "target",
    "build",
    "out",
    "dist",
})

_WORK_DIRECTORY_MARKERS = (
    "extracted_functions",
    "phases.json",
    "plugin_context.json",
    "spec_prompts",
)


def is_excluded_source_directory(name: str) -> bool:
    """Return whether a nested directory is outside hardware source scans."""
    return (
        name.startswith(".")
        or name in SOURCE_SCAN_EXCLUDED_DIRECTORY_NAMES
    )


def resolve_hardware_project_paths(
    proj_dir: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Return the hardware project root and its Pipeline work directory.

    Hardware language services normally receive the project root, but some
    artifact readers also accept its generated ``fm_agent`` directory. A path
    basename alone cannot distinguish those forms when the target repository
    itself is named ``fm_agent``. Git and Pipeline markers provide the required
    structural evidence while retaining support for lightweight test fixtures.
    """
    root = Path(os.path.abspath(os.fspath(proj_dir)))
    is_git_project = (root / ".git").exists()
    is_pipeline_work_dir = (
        root.name == "fm_agent"
        and not is_git_project
        and (
            (root.parent / ".git").exists()
            or any((root / marker).exists() for marker in _WORK_DIRECTORY_MARKERS)
        )
    )
    if is_pipeline_work_dir:
        return root.parent, root
    return root, root / "fm_agent"
