"""Regression coverage for #243; all analysis/LLM stages are stubbed."""

import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import main
import report
from plugins.entry_reasoning import plugin as entry_plugin
from src.file_utils import _iter_project_source_files
from src.git import frozen_worktree
from src.run_estimate import collect_history_samples, read_history
from src.run_workspace import (
    ARCHIVE_DIRECTORY,
    archive_workspace,
    publish_workspace,
    record_run_metadata,
    restore_workspace,
    validate_resume_source,
)


class InterruptedAnalysis(RuntimeError):
    pass


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def artifacts(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    }


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.work = self.project / "fm_agent"
        write(self.project / "main.c", "int main(void) { return 0; }\n")
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def seed(self, workspace=None):
        workspace = workspace or self.work
        for path, content in {
            "phases.json": '{"phases": []}',
            "report.html": "<p>previous report</p>",
            "trace/events.jsonl": '{"event": "previous"}\n',
            "trace/payloads/result.bin": "payload\x00data",
            "extracted_functions/main-c/main.c": "int main(void) { return 0; }",
            "extracted_functions/main-c/main.spec.json": '{"pre_condition": "true"}',
            "extracted_functions/main-c/main.info.json": '{"callees": []}',
            "logic_verification_results/main.json": '{"verdict": "MATCH"}',
            "bug_validation/bug.md": "Evidence",
            "bug_validation/probe.sh": "#!/bin/sh\nexit 0\n",
            "bug_validation/bug.json": '{"status": "confirmed"}',
            "history.jsonl": '{"run_id": "older-run", "function_count": 1}\n',
            ".checkpoint": "partial progress",
        }.items():
            write(workspace / path, content)
        (workspace / "bug_validation/probe.sh").chmod(0o755)
        (workspace / "latest-report").symlink_to("report.html")
        return artifacts(workspace)

    def archives(self):
        return sorted((self.project / ARCHIVE_DIRECTORY).glob("*"))

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.project), *args], check=True,
            capture_output=True, text=True,
        ).stdout.strip()

    def init_git(self):
        self.git("init")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("add", "main.c")
        self.git("commit", "-m", "fixture")

    def test_archive_preserves_all_bytes_permissions_and_symlinks(self):
        before = self.seed()
        archive = archive_workspace(self.work)
        self.assertFalse(self.work.exists())
        self.assertEqual(artifacts(archive), before)
        self.assertEqual((archive / "bug_validation/probe.sh").stat().st_mode & 0o777, 0o755)
        self.assertEqual(os.readlink(archive / "latest-report"), "report.html")
        self.seed()
        second = archive_workspace(self.work)
        self.assertNotEqual(archive, second)
        self.assertEqual(artifacts(archive), before)

    def test_archive_failure_leaves_original_untouched(self):
        before = self.seed()
        with patch.object(Path, "rename", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                main.run_pipeline(str(self.project))
        self.assertEqual(artifacts(self.work), before)
        self.assertEqual(self.archives(), [])

    def test_restore_keeps_both_runs_and_original_archive(self):
        original = self.seed()
        archive = archive_workspace(self.work)
        write(self.work / "trace/new.jsonl", "new run")
        current = artifacts(self.work)
        restore_workspace(archive, self.work)
        self.assertEqual(artifacts(self.work), original)
        self.assertEqual(artifacts(archive), original)
        saved_current = [p for p in self.archives() if p != archive]
        self.assertEqual(len(saved_current), 1)
        self.assertEqual(artifacts(saved_current[0]), current)

    def test_copy_failure_never_replaces_current(self):
        before = self.seed()
        source = self.root / "source"
        self.seed(source)
        with patch("src.run_workspace.shutil.copytree", side_effect=OSError("copy failed")):
            with self.assertRaises(OSError):
                publish_workspace(source, self.work)
        self.assertEqual(artifacts(self.work), before)
        self.assertEqual(self.archives(), [])

    def test_publish_archive_failure_never_replaces_current(self):
        before = self.seed()
        source = self.root / "source"
        self.seed(source)
        with patch("src.run_workspace.archive_workspace", side_effect=OSError("archive failed")):
            with self.assertRaises(OSError):
                publish_workspace(source, self.work)
        self.assertEqual(artifacts(self.work), before)
        self.assertEqual(list(self.project.glob(".fm_agent_publish_*")), [])

    def test_failed_promotion_rolls_back_current(self):
        before = self.seed()
        source = self.root / "source"
        self.seed(source)
        rename = Path.rename

        def fail_promotion(path, target):
            if path.name == "workspace":
                raise OSError("promotion failed")
            return rename(path, target)

        with patch.object(Path, "rename", fail_promotion):
            with self.assertRaises(OSError):
                publish_workspace(source, self.work)
        self.assertEqual(artifacts(self.work), before)
        self.assertEqual(artifacts(source), before)

    def test_resume_rejects_invalid_and_overlapping_sources(self):
        self.seed()
        for source in (self.work, self.project, self.work / "trace", self.root / "missing"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                validate_resume_source(source, self.work)
        link = self.root / "linked"
        link.symlink_to(self.work, target_is_directory=True)
        with self.assertRaises(ValueError):
            validate_resume_source(link, self.work)
        self.assertEqual(self.archives(), [])

    def test_fresh_pipeline_archives_before_first_stage_and_carries_history(self):
        before = self.seed()
        with patch.object(main, "_run_generate_phases", side_effect=InterruptedAnalysis):
            with self.assertRaises(InterruptedAnalysis):
                main.run_pipeline(str(self.project))
        self.assertEqual(artifacts(self.archives()[0]), before)
        self.assertFalse((self.work / "extracted_functions").exists())
        self.assertEqual(read_history(self.work)[0]["run_id"], "older-run")
        self.assertTrue((self.work / "run.json").is_file())
        self.assertIn(str(self.archives()[0]), self.output.getvalue())
        self.assertIn(str(self.work), self.output.getvalue())

    def test_resume_preserves_identity_and_checkpoints(self):
        self.seed()
        metadata = record_run_metadata(self.work, self.project)
        with patch.object(main, "_run_generate_phases", side_effect=InterruptedAnalysis) as stage:
            with self.assertRaises(InterruptedAnalysis):
                main.run_pipeline(str(self.project), resume=True)
        self.assertTrue(stage.call_args.kwargs["resume"])
        self.assertEqual(self.archives(), [])
        self.assertTrue((self.work / ".checkpoint").is_file())
        self.assertEqual(json.loads((self.work / "run.json").read_text()), metadata)

    def test_metadata_records_revision_before_interruption(self):
        self.init_git()
        self.seed()
        metadata = record_run_metadata(self.work, self.project)
        self.assertEqual(metadata["git_head"], self.git("rev-parse", "HEAD"))
        self.assertFalse(metadata["git_dirty"])
        write(self.project / "main.c", "int main(void) { return 1; }\n")
        archive_workspace(self.work)
        self.work.mkdir()
        newer = record_run_metadata(self.work, self.project)
        self.assertTrue(newer["git_dirty"])
        self.assertNotEqual(newer["run_id"], metadata["run_id"])

    def test_archives_feed_estimates_but_are_not_source_code(self):
        self.seed()
        archive = archive_workspace(self.work)
        write(archive / "run_summary.json", '{"run_id": "completed-run", "function_count": 2}')
        self.assertEqual({r["run_id"] for r in collect_history_samples(str(self.project), self.work)},
                         {"older-run", "completed-run"})
        self.assertEqual(list(_iter_project_source_files(str(self.project))), ["main.c"])

    def test_archived_report_can_be_regenerated_and_displays_identity(self):
        self.seed()
        metadata = record_run_metadata(self.work, self.project)
        archive = archive_workspace(self.work)
        self.assertEqual(report._locate_workdir(str(archive)), str(archive))
        generated = Path(report.generate_report(str(archive))).read_text()
        self.assertIn(metadata["run_id"], generated)
        # Very early interrupted workspaces may have only run.json.
        self.work.mkdir()
        record_run_metadata(self.work, self.project)
        early_archive = archive_workspace(self.work)
        self.assertEqual(report._locate_workdir(str(early_archive)), str(early_archive))

    def test_entry_result_copy_preserves_previous_workspace(self):
        before = self.seed()
        run = self.root / "entry-run"
        self.seed(run / "fm_agent")
        write(run / "fm_agent/.entry_scope_ready", "ready")
        write(run / "fm_agent/trace/new.jsonl", "entry result")
        main._finalize_failed_entry_run(str(self.project), str(run))
        self.assertEqual(artifacts(self.archives()[0]), before)
        self.assertEqual((self.work / "trace/new.jsonl").read_text(), "entry result")
        self.assertFalse((self.work / ".entry_scope_ready").exists())
        self.assertFalse(run.exists())

    def test_failed_entry_copy_retains_recoverable_run(self):
        before = self.seed()
        run = self.root / "entry-run"
        self.seed(run / "fm_agent")
        write(run / "fm_agent/.entry_scope_ready", "ready")
        with patch("src.run_workspace.shutil.copytree", side_effect=OSError("copy failed")):
            with self.assertRaises(OSError):
                main._finalize_failed_entry_run(str(self.project), str(run))
        self.assertEqual(artifacts(self.work), before)
        self.assertTrue((run / "fm_agent/.entry_scope_ready").exists())

    def test_early_entry_failure_preserves_progress_without_ready_marker(self):
        before = self.seed()
        run = self.root / "entry-run"
        self.seed(run / "fm_agent")
        write(run / "fm_agent/trace/early.jsonl", "early progress")
        main._finalize_failed_entry_run(str(self.project), str(run))
        self.assertEqual(artifacts(self.archives()[0]), before)
        self.assertTrue((self.work / "trace/early.jsonl").exists())

    def test_successful_entry_publication_preserves_previous_run(self):
        before = self.seed()
        run = self.root / "entry-run"
        self.seed(run / "fm_agent")
        write(run / "fm_agent/.entry_scope_ready", "ready")
        with patch.object(entry_plugin, "_entry_run_dir", str(run)), \
             patch.object(entry_plugin, "_original_proj_dir", str(self.project)), \
             patch.object(entry_plugin, "_entry_funcs", ["main-c/main"]):
            entry_plugin.publish_entry_results(str(run))
        self.assertEqual(artifacts(self.archives()[0]), before)
        self.assertFalse(run.exists())
        self.assertFalse((self.work / ".entry_scope_ready").exists())

    def test_recreating_entry_copy_archives_leftovers_from_hard_interruption(self):
        run = self.root / "entry-run"
        previous = self.seed(run / "fm_agent")
        entry_plugin._make_run_copy(str(self.project), str(run))
        self.assertEqual(artifacts(self.archives()[0]), previous)
        self.assertFalse((run / ARCHIVE_DIRECTORY).exists())

    def test_project_copies_exclude_archive_tree(self):
        self.init_git()
        self.seed()
        archive_workspace(self.work)
        copy = self.root / "entry-copy"
        entry_plugin._make_run_copy(str(self.project), str(copy))
        self.assertFalse((copy / ARCHIVE_DIRECTORY).exists())
        snapshot_base = self.root / "snapshot-base"
        snapshot_base.mkdir()
        mkdtemp = tempfile.mkdtemp
        def make_temp(*args, **kwargs):
            return str(snapshot_base) if kwargs.get("prefix", "").startswith("fm_agent_wt_") else mkdtemp(*args, **kwargs)
        with patch("src.git.tempfile.mkdtemp", side_effect=make_temp):
            with frozen_worktree(str(self.project)) as snapshot:
                self.assertFalse((Path(snapshot) / ARCHIVE_DIRECTORY).exists())
                self.assertTrue((Path(snapshot) / "main.c").is_file())

    def run_cli_until_interrupted(self, *options):
        with patch("sys.argv", ["main.py", str(self.project), *options]), \
             patch.dict(os.environ, {"FM_AGENT_RESUME": "0"}), \
             patch("src.env_check.run", return_value=True), \
             patch("src.pipeline_setup._run_generate_phases", side_effect=InterruptedAnalysis):
            with self.assertRaises(InterruptedAnalysis):
                runpy.run_path(str(Path(main.__file__)), run_name="__main__")

    def test_cli_resume_from_preserves_archive_and_resumes_selected_run(self):
        self.init_git()
        self.seed()
        metadata = record_run_metadata(self.work, self.project)
        archive = archive_workspace(self.work)
        before = artifacts(archive)
        write(self.work / "trace/new.jsonl", "accidental new run")
        self.run_cli_until_interrupted("--resume-from", str(archive))
        self.assertEqual(artifacts(archive), before)
        self.assertEqual(json.loads((self.work / "run.json").read_text())["run_id"], metadata["run_id"])
        self.assertTrue((self.work / ".checkpoint").exists())
        self.assertFalse((self.work / "trace/new.jsonl").exists())
        self.assertEqual(len(self.archives()), 2)

    def test_isolated_cli_failure_archives_original_and_publishes_partial_run(self):
        self.init_git()
        before = self.seed()
        snapshot_base = self.root / "snapshot-base"
        snapshot_base.mkdir()
        mkdtemp = tempfile.mkdtemp
        def make_temp(*args, **kwargs):
            return str(snapshot_base) if kwargs.get("prefix", "").startswith("fm_agent_wt_") else mkdtemp(*args, **kwargs)
        with patch("src.git.tempfile.mkdtemp", side_effect=make_temp):
            self.run_cli_until_interrupted("--isolate")
        self.assertEqual(len(self.archives()), 1)
        self.assertEqual(artifacts(self.archives()[0]), before)
        self.assertTrue((self.work / "run.json").is_file())
        self.assertFalse((self.work / ".checkpoint").exists())
        self.assertEqual(read_history(self.work)[0]["run_id"], "older-run")

    def test_isolated_estimate_includes_history_from_archives(self):
        self.init_git()
        archived = self.project / ARCHIVE_DIRECTORY / "older"
        write(archived / "run_summary.json", '{"run_id": "archived-only", "function_count": 1}')
        snapshot_base = self.root / "snapshot-base"
        snapshot_base.mkdir()
        mkdtemp = tempfile.mkdtemp
        def make_temp(*args, **kwargs):
            return str(snapshot_base) if kwargs.get("prefix", "").startswith("fm_agent_wt_") else mkdtemp(*args, **kwargs)
        with patch("src.git.tempfile.mkdtemp", side_effect=make_temp):
            self.run_cli_until_interrupted("--isolate")
        self.assertEqual(read_history(self.work)[0]["run_id"], "archived-only")

    def test_completed_cli_runs_keep_separate_reports_and_versions(self):
        self.init_git()

        def plan(project, work_dir, *args, **kwargs):
            write(Path(work_dir) / "phases.json", '{"phases": []}')

        def extract(project, work_dir, **kwargs):
            write(Path(work_dir) / "extracted_functions/main-c/main.c", "int main(void) { return 0; }")

        with patch("sys.argv", ["main.py", str(self.project)]), \
             patch.dict(os.environ, {"FM_AGENT_RESUME": "0"}), \
             patch("src.env_check.run", return_value=True), \
             patch("src.pipeline_setup._run_generate_phases", side_effect=plan), \
             patch("src.pipeline_setup._post_process_phases", return_value=False), \
             patch("src.pipeline_setup._run_generate_domain_context"), \
             patch("src.languages.codegraph.try_codegraph_init"), \
             patch("src.extract.run_extraction", side_effect=extract), \
             patch("src.generate_topdown_layers.generate_topdown_layers"), \
             patch("src.spec_generation_and_verification.run_spec_generation_and_verification"):
            runpy.run_path(str(Path(main.__file__)), run_name="__main__")
            first = artifacts(self.work)
            first_id = json.loads(first["run.json"])["run_id"]
            self.assertIn(first_id.encode(), first["report.html"])
            write(self.project / "main.c", "int main(void) { return 2; }\n")
            self.git("add", "main.c")
            self.git("commit", "-m", "next revision")
            runpy.run_path(str(Path(main.__file__)), run_name="__main__")
        self.assertEqual(len(self.archives()), 1)
        self.assertEqual(artifacts(self.archives()[0]), first)
        current_id = json.loads((self.work / "run.json").read_text())["run_id"]
        self.assertNotEqual(first_id, current_id)
        self.assertIn(current_id, (self.work / "report.html").read_text())
        self.assertNotEqual(first["version.log"], (self.work / "version.log").read_bytes())

    def test_cli_rejects_incompatible_resume_before_touching_current(self):
        self.seed()
        archive = archive_workspace(self.work)
        write(archive / "logic_verification_results/main.bug-001.json", "{}")
        write(self.work / "trace/new.jsonl", "current")
        before = artifacts(self.work)
        with patch("sys.argv", ["main.py", str(self.project), "--resume-from", str(archive)]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(str(Path(main.__file__)), run_name="__main__")
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(artifacts(self.work), before)
        self.assertEqual(len(self.archives()), 1)


if __name__ == "__main__":
    unittest.main()
