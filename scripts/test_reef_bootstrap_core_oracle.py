"""Keep bootstrap acceptance identities bound to their executable CLI tests."""

from copy import deepcopy
import unittest

from ci_change_owned import test_functions
from no_std_core_support import ROOT, OracleFailure, _selected_test_names
import reef_bootstrap_core_oracle as oracle


class BootstrapInventoryTests(unittest.TestCase):
    def inventory(self):
        source = ROOT / "crates/chelis-cli/tests/reef_install_bootstrap.rs"
        names = test_functions(source.read_text(encoding="utf-8"))
        return {
            "rust-suites": {
                "bootstrap": {
                    "package-name": "chelis-cli",
                    "binary-name": "reef_install_bootstrap",
                    "testcases": {
                        name: {
                            "kind": "test",
                            "ignored": False,
                            "filter-match": {"status": "matches"},
                        }
                        for name in names
                        if name != "phaseA_item7_real_bootstrap_manual_gate"
                    },
                },
            },
        }

    def test_required_cli_cases_exist_in_the_executable_suite(self):
        selected = _selected_test_names(self.inventory(), oracle.CLI_TESTS)
        self.assertTrue(selected)

    def test_both_release_discovery_budget_controls_are_required(self):
        for name in (
            "bootstrap_defaults_keep_the_shared_release_page_budget",
            "bootstrap_defaults_charge_tags_only_for_window_releases",
        ):
            with self.subTest(name=name):
                self.assertIn(name, oracle.CLI_TESTS)

    def test_omitted_or_filtered_required_case_fails_with_other_cases_present(self):
        inventory = self.inventory()
        for name in oracle.CLI_TESTS:
            for omitted in (True, False):
                changed = deepcopy(inventory)
                cases = changed["rust-suites"]["bootstrap"]["testcases"]
                if name not in cases:
                    self.fail(f"required CLI case has no executable declaration: {name}")
                if omitted:
                    del cases[name]
                else:
                    cases[name]["filter-match"]["status"] = "mismatch"
                with self.subTest(name=name, omitted=omitted):
                    with self.assertRaisesRegex(OracleFailure, "omitted or filtered: " + name):
                        _selected_test_names(changed, oracle.CLI_TESTS)


if __name__ == "__main__":
    unittest.main()
