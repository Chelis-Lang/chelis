"""The named `openspec_adoption` suite (Phase 0 adoption oracle inputs).

Run via: `python3 scripts/test_openspec_adoption.py` from repo root.

Owning change: `openspec/changes/adopt-openspec-governance` (task 2.4).
What is locked here:

  (a) configuration: the canonical `openspec/` root exists, uses the
      built-in spec-driven schema (no project schema override), and the
      adoption lifecycle carries proposal, deltas, design, and tasks;
  (b) workflow: `.github/workflows/openspec-governance.yml` is read-only,
      full-history, credential-free, pins the accepted checkout and
      shared-action SHAs, and passes no inputs or secrets;
  (c) instructions and routing: agent instructions and the PR template
      carry the governed-change classification and the exact
      `OpenSpec-Change: <change-id>` citation format;
  (d) docs-only boundaries: `spec/**` is never treated as docs-only
      exempt by `scripts/ci_detect_docs_only.py`;
  (e) Phase 0 messaging: `spec/design/spec_provenance.md` describes
      adoption as pending until the named oracles are green.

Tests for (c) stay red until tasks 4.1-4.2 land; that is the intended
spec-first state, not a defect.
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGE_DIR = REPO_ROOT / "openspec" / "changes" / "adopt-openspec-governance"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "openspec-governance.yml"

ACCEPTED_ACTION_SHA = "2906e03880a2ea7d553b0959c246991b1b058990"
ACCEPTED_CHECKOUT_SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"
CITATION_PREFIX = "OpenSpec-Change: "


class TestConfiguration(unittest.TestCase):
    def test_canonical_root_exists(self):
        self.assertTrue((REPO_ROOT / "openspec" / "config.yaml").is_file())

    def test_schema_is_builtin_spec_driven(self):
        text = (REPO_ROOT / "openspec" / "config.yaml").read_text(encoding="utf-8")
        declarations = re.findall(r"^schema\s*:\s*(\S+)\s*$", text, re.MULTILINE)
        for value in declarations:
            self.assertEqual("spec-driven", value)

    def test_adoption_lifecycle_is_complete(self):
        for artifact in (".openspec.yaml", "proposal.md", "design.md", "tasks.md"):
            self.assertTrue((CHANGE_DIR / artifact).is_file(), artifact)
        deltas = sorted((CHANGE_DIR / "specs").glob("*/spec.md"))
        self.assertEqual(2, len(deltas))


class TestWorkflow(unittest.TestCase):
    def _text(self):
        return WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_exists(self):
        self.assertTrue(WORKFLOW.is_file())

    def test_read_only_permissions(self):
        self.assertIn("permissions:\n  contents: read", self._text())
        self.assertNotIn("write", self._text())

    def test_full_history_credential_free_checkout(self):
        text = self._text()
        self.assertIn(f"actions/checkout@{ACCEPTED_CHECKOUT_SHA}", text)
        self.assertIn("fetch-depth: 0", text)
        self.assertIn("persist-credentials: false", text)

    def test_pinned_shared_action(self):
        self.assertIn(
            f"Chelis-Lang/ci/actions/openspec-governance@{ACCEPTED_ACTION_SHA}",
            self._text(),
        )

    def test_no_mutable_action_reference(self):
        for line in self._text().splitlines():
            if "uses:" in line:
                ref = line.split("@", 1)[1].split()[0]
                self.assertRegex(ref, r"^[0-9a-f]{40}$", line)

    def test_no_secrets_or_action_inputs(self):
        text = self._text()
        self.assertNotIn("secrets", text)
        action_step = text.split("openspec-governance@")[1]
        self.assertNotIn("with:", action_step)

    def test_no_duplicated_action_provisioning(self):
        # The shared action owns Node setup, the npm-locked OpenSpec CLI,
        # and the launcher; the consumer workflow must not duplicate them.
        text = self._text().lower()
        for fragment in ("setup-node", "npm", "setup-python", "fission-ai"):
            self.assertNotIn(fragment, text, fragment)

    def test_supported_triggers_only(self):
        text = self._text()
        for trigger in ("pull_request", "push", "workflow_dispatch"):
            self.assertIn(trigger, text)
        self.assertNotIn("schedule", text)


class TestInstructionsAndRouting(unittest.TestCase):
    def test_agent_instructions_name_governed_changes(self):
        text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("OpenSpec", text)
        self.assertIn(CITATION_PREFIX.strip(), text)

    def test_pr_template_carries_citation_field(self):
        candidates = [
            REPO_ROOT / ".github" / "pull_request_template.md",
            REPO_ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
        ]
        existing = [p for p in candidates if p.is_file()]
        self.assertTrue(existing, "no pull request template found")
        text = existing[0].read_text(encoding="utf-8")
        self.assertIn(CITATION_PREFIX, text)


def _load_docs_only_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ci_detect_docs_only",
        REPO_ROOT / "scripts" / "ci_detect_docs_only.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestDocsOnlyBoundary(unittest.TestCase):
    def test_spec_tree_is_never_docs_only_exempt(self):
        mod = _load_docs_only_module()
        self.assertFalse(mod.is_docs_only(["spec/01-nomenclature.md"]))

    def test_plain_docs_remain_docs_only(self):
        mod = _load_docs_only_module()
        self.assertTrue(mod.is_docs_only(["docs/book/src/install.md"]))


class TestPhaseZeroMessaging(unittest.TestCase):
    def test_provenance_doc_marks_phase_zero_pending(self):
        text = (REPO_ROOT / "spec" / "design" / "spec_provenance.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("OpenSpec", text)
        self.assertRegex(
            text,
            re.compile(
                r"does not activate|becomes active only|inactive|pending", re.I
            ),
        )


if __name__ == "__main__":
    unittest.main()
