"""Unit tests for `ci_detect_docs_only.py` (chelis#419).

The contract under test: a CI run is "docs only" iff the changed-file
set is non-empty and EVERY path is documentation/prose. The conservative
direction (empty list, any code file -> not docs-only) is what keeps a
code change from skipping the heavy required gate.
"""

import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "ci_detect_docs_only", here / "ci_detect_docs_only.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load_module()


class IsDocPathTests(unittest.TestCase):
    def test_markdown_is_doc(self):
        self.assertTrue(m.is_doc_path("README.md"))
        self.assertTrue(m.is_doc_path("spec/00-overview.md"))
        self.assertTrue(m.is_doc_path("CHANGELOG.markdown"))

    def test_exact_names_are_doc(self):
        for name in ("LICENSE", "CODEOWNERS", "NOTICE", ".gitignore"):
            self.assertTrue(m.is_doc_path(name), name)

    def test_docs_dir_prefix_is_doc(self):
        self.assertTrue(m.is_doc_path("docs/book/src/intro.md"))
        # Even a non-markdown file under docs/ (e.g. an image or a
        # mdBook theme asset) is prose-adjacent; the Docs job still runs.
        self.assertTrue(m.is_doc_path("docs/book/theme/custom.css"))

    def test_openspec_changes_prefix_is_doc(self):
        self.assertTrue(
            m.is_doc_path("openspec/changes/add-thing/proposal.md")
        )
        # `openspec new change` always writes this non-Markdown metadata
        # file; without the prefix rule it alone forces the full matrix.
        self.assertTrue(
            m.is_doc_path("openspec/changes/add-thing/.openspec.yaml")
        )

    def test_code_paths_are_not_doc(self):
        for p in (
            "crates/chelis-types/src/infer.rs",
            "Cargo.toml",
            "scripts/gate.py",
            ".github/workflows/ci.yml",
            "packages/chelis-std/tests/foo.ch",
            "docsignore",  # sibling, not under docs/
            "readme.md.rs",  # .rs wins; not a doc
            "openspec/config.yaml",  # tool config, not one change
            "openspec/changesets/x.yaml",  # sibling, not under changes/
        ):
            self.assertFalse(m.is_doc_path(p), p)

    def test_quoted_path_is_unquoted(self):
        # git core.quotePath wraps non-ASCII paths in quotes.
        self.assertTrue(m.is_doc_path('"docs/café.md"'))


class IsDocsOnlyTests(unittest.TestCase):
    def test_all_docs_is_docs_only(self):
        self.assertTrue(
            m.is_docs_only(["README.md", "docs/book/src/x.md", "LICENSE"])
        )

    def test_any_code_file_forces_full_suite(self):
        self.assertFalse(
            m.is_docs_only(["README.md", "crates/chelis-ir/src/lib.rs"])
        )

    def test_empty_list_is_not_docs_only(self):
        # The conservative direction: no detected changes -> full suite.
        self.assertFalse(m.is_docs_only([]))
        self.assertFalse(m.is_docs_only(["", "  ", "\n"]))

    def test_single_code_file_is_not_docs_only(self):
        self.assertFalse(m.is_docs_only(["Cargo.toml"]))

    def test_executable_oracle_doc_forces_full_suite(self):
        for oracle_doc in (
            "spec/design/faithful_observation.md",
            "spec/02-surf-syntax.md",
            "spec/03-deep-syntax.md",
            "spec/04-type-system.md",
            "spec/05-risc-primitives.md",
            "spec/11-ffi.md",
            "spec/design/capability_table.md",
            "spec/design/compiled_value_ownership.md",
            "spec/design/dtype_semantics.md",
            "spec/design/loud_unsupported.md",
            "spec/design/spec_provenance.md",
            "spec/design/remediation_roadmap.md",
            "docs/investigations/remediation_status_2026_08_04.md",
        ):
            with self.subTest(oracle_doc=oracle_doc):
                self.assertFalse(m.is_docs_only([oracle_doc]))
                self.assertFalse(m.is_docs_only(["README.md", oracle_doc]))


class DiagnosticKindChangeTests(unittest.TestCase):
    def test_every_oracle_owner_and_control_triggers_the_mutation_job(self):
        for path in (
            ".github/workflows/ci.yml",
            "crates/chelis-compiler-api/src/context.rs",
            "crates/chelis-compiler-api/src/lib.rs",
            "crates/chelis-compiler-api/src/schema.rs",
            "crates/chelis-compiler-api/tests/diagnostic_kind_pipeline.rs",
            "crates/chelis-vocab/src/lib.rs",
            "scripts/ci_detect_docs_only.py",
            "scripts/diagnostic_kind_oracle.py",
            "scripts/test_ci_detect_docs_only.py",
            "scripts/test_diagnostic_kind_oracle.py",
            "scripts/test_gate.py",
        ):
            self.assertTrue(m.diagnostic_kind_changed([path]), path)

    def test_unrelated_paths_do_not_trigger_the_mutation_job(self):
        self.assertFalse(
            m.diagnostic_kind_changed(
                ["README.md", "crates/chelis-ir/src/lower.rs"]
            )
        )

    def test_empty_change_set_fails_safe(self):
        self.assertTrue(m.diagnostic_kind_changed([]))


class CiContractChangeTests(unittest.TestCase):
    def test_workflow_scripts_and_policy_docs_trigger_preflight(self):
        for path in (
            ".github/workflows/ci.yml",
            ".github/actions/free-disk-space/action.yml",
            ".config/ci-test-targets.toml",
            "agent-skills/redteam-exec/SKILL.md",
            "scripts/gate.py",
            "scripts/ci_change_owned.py",
            "scripts/test_ci_candidate_lifecycle.py",
            "scripts/test_pr_workflow_routing.py",
            "scripts/test_change_owned_workflow.py",
            "scripts/test_hosted_validation.py",
            "scripts/test_gate.py",
            "AGENTS.md",
            "docs/ci_validation.md",
            "docs/guard_changes_for_pr_authors.md",
            "spec/design/guard_artifact_proposal_assessment.md",
        ):
            with self.subTest(path=path):
                self.assertTrue(m.ci_contract_changed([path]))

    def test_unrelated_code_and_prose_do_not_trigger_preflight(self):
        self.assertFalse(
            m.ci_contract_changed(
                ["README.md", "crates/chelis-ir/src/lower.rs"]
            )
        )

    def test_empty_change_set_fails_safe(self):
        self.assertTrue(m.ci_contract_changed([]))


class EmitTests(unittest.TestCase):
    def test_emit_writes_github_output_and_stdout(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "gh_output")
            old = os.environ.get("GITHUB_OUTPUT")
            os.environ["GITHUB_OUTPUT"] = out_path
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    m._emit(True, False, True)
                self.assertEqual(
                    buf.getvalue().splitlines(),
                    [
                        "docs_only=true",
                        "diagnostic_kind_changed=false",
                        "ci_contract_changed=true",
                    ],
                )
                with open(out_path, encoding="utf-8") as fh:
                    self.assertEqual(
                        fh.read().splitlines(),
                        [
                            "docs_only=true",
                            "diagnostic_kind_changed=false",
                            "ci_contract_changed=true",
                        ],
                    )
            finally:
                if old is None:
                    os.environ.pop("GITHUB_OUTPUT", None)
                else:
                    os.environ["GITHUB_OUTPUT"] = old

    def test_emit_stdout_only_when_no_github_output(self):
        old = os.environ.pop("GITHUB_OUTPUT", None)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                m._emit(False, True, False)
            self.assertEqual(
                buf.getvalue().splitlines(),
                [
                    "docs_only=false",
                    "diagnostic_kind_changed=true",
                    "ci_contract_changed=false",
                ],
            )
        finally:
            if old is not None:
                os.environ["GITHUB_OUTPUT"] = old


class MainTests(unittest.TestCase):
    def test_main_reads_stdin(self):
        old_stdin = sys.stdin
        old_out = os.environ.pop("GITHUB_OUTPUT", None)
        try:
            sys.stdin = io.StringIO("docs/a.md\nREADME.md\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = m.main([])
            self.assertEqual(rc, 0)
            self.assertEqual(
                buf.getvalue().splitlines(),
                [
                    "docs_only=true",
                    "diagnostic_kind_changed=false",
                    "ci_contract_changed=false",
                ],
            )
        finally:
            sys.stdin = old_stdin
            if old_out is not None:
                os.environ["GITHUB_OUTPUT"] = old_out


if __name__ == "__main__":
    unittest.main()
