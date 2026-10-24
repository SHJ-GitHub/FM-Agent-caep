"""Preserve complete analysis workspaces when starting or restoring a run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path


ARCHIVE_SUFFIX = ".fm_agent_runs"
RUN_METADATA = "run.json"


def _new_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex


def archive_root(project_dir: str | Path) -> Path:
    """Keep archives beside the project, outside its source/agent scan scope."""
    project = Path(project_dir).resolve()
    return project.with_name(project.name + ARCHIVE_SUFFIX)


def archive_workspace(work_dir: str | Path, *, project_dir: str | Path | None = None) -> Path | None:
    """Move the entire workspace to a unique project archive, or fail intact.

    Require a rename on the same filesystem; never fall back to copy-and-delete
    if it fails. No existing archive is ever pruned or overwritten. project_dir
    allows a stale entry-run copy to be preserved outside that disposable copy.
    """
    work_dir = Path(work_dir).absolute()
    if work_dir.is_symlink():
        raise ValueError(f"Refusing to archive a symlink workspace: {work_dir}")
    if not work_dir.exists():
        return None
    if not work_dir.is_dir():
        raise ValueError(f"Workspace is not a directory: {work_dir}")
    # Preflight scaffolding is not a previous run. Remove only a truly empty
    # directory, so publication can also promote its staged workspace here.
    if not any(work_dir.iterdir()):
        work_dir.rmdir()
        return None
    archive_project = project_dir if project_dir else work_dir.parent
    root = archive_root(archive_project)
    if root.is_symlink():
        raise ValueError(f"Archive directory must not be a symlink: {root}")
    root.mkdir(exist_ok=True)
    destination = root / _new_id()
    while destination.exists() or destination.is_symlink():
        destination = root / _new_id()
    work_dir.rename(destination)
    print(f"[Pipeline] Previous run archived at: {destination}")
    return destination


def validate_resume_source(source: str | Path, work_dir: str | Path) -> Path:
    source = Path(os.path.abspath(os.path.expanduser(source)))
    target = Path(work_dir).absolute()
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"--resume-from must name an existing workspace directory: {source}")
    # Resolve parent symlinks too, to reject overlapping copies.
    resolved_source, resolved_target = source.resolve(), target.resolve()
    if (resolved_source == resolved_target
            or resolved_source in resolved_target.parents
            or resolved_target in resolved_source.parents):
        raise ValueError("--resume-from must be separate from the current fm_agent/; use --resume for it")
    markers = (RUN_METADATA, "phases.json", "trace", "extracted_functions", "spec_prompts")
    if not any((source / marker).exists() for marker in markers):
        raise ValueError(f"--resume-from is not an FM-Agent workspace: {source}")
    return source


def publish_workspace(source: str | Path, work_dir: str | Path) -> None:
    """Install a complete copy, archiving the current workspace first.

    A failed copy leaves the current workspace untouched. A failed promotion
    restores it. The source (archive or isolated run) always remains available.
    """
    source, work_dir = Path(source).absolute(), Path(work_dir).absolute()
    validate_resume_source(source, work_dir)
    with tempfile.TemporaryDirectory(prefix=".fm_agent_publish_", dir=work_dir.parent) as staging:
        staged = Path(staging) / "workspace"
        shutil.copytree(source, staged, symlinks=True)
        archived = archive_workspace(work_dir)
        try:
            staged.rename(work_dir)
        except BaseException:
            if archived is not None:
                archived.rename(work_dir)
            raise
    print(f"[Pipeline] Current run: {work_dir}")


def restore_workspace(source: str | Path, work_dir: str | Path) -> None:
    source = validate_resume_source(source, work_dir)
    publish_workspace(source, work_dir)
    print(f"[Pipeline] Restored run from: {source} (archive retained)")


def record_run_metadata(work_dir: str | Path, project_dir: str | Path) -> dict:
    """Record identity and source revision before analysis, including partial runs.

    Resume retains the original identity. For an older workspace, version.log
    remains the source of its completed baseline; the new metadata describes
    the checkout at the time it is first recorded.
    """
    work_dir, project_dir = Path(work_dir), Path(project_dir).absolute()
    path = work_dir / RUN_METADATA
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(project_dir), *args],
            capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=normal", "--", ".",
                 ":(exclude)fm_agent")
    metadata = {
        "schema_version": 1,
        "run_id": _new_id(),
        "recorded_at": datetime.now(UTC).isoformat(),
        "project_dir": str(project_dir),
        "analysis_dir": str(work_dir.absolute().parent),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_status": status,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return metadata
