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


class EmitTests(unittest.TestCase):
    def test_emit_writes_github_output_and_stdout(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "gh_output")
            old = os.environ.get("GITHUB_OUTPUT")
            os.environ["GITHUB_OUTPUT"] = out_path
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    m._emit(True)
                self.assertEqual(buf.getvalue().strip(), "docs_only=true")
                with open(out_path, encoding="utf-8") as fh:
                    self.assertIn("docs_only=true", fh.read())
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
                m._emit(False)
            self.assertEqual(buf.getvalue().strip(), "docs_only=false")
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
            self.assertEqual(buf.getvalue().strip(), "docs_only=true")
        finally:
            sys.stdin = old_stdin
            if old_out is not None:
                os.environ["GITHUB_OUTPUT"] = old_out


if __name__ == "__main__":
    unittest.main()
