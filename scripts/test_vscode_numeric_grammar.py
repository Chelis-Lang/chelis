"""Checks numeric scopes in the maintained VS Code Surf and Deep grammars."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_DIR = ROOT / "editors" / "vscode" / "syntaxes"


def _patterns(grammar_name: str) -> list[tuple[str, re.Pattern[str]]]:
    grammar = json.loads((GRAMMAR_DIR / grammar_name).read_text(encoding="utf-8"))
    return [
        (pattern["name"], re.compile(pattern["match"]))
        for pattern in grammar["repository"]["numbers"]["patterns"]
    ]


def _scopes(grammar_name: str, token: str) -> list[str]:
    scopes = []
    for name, pattern in _patterns(grammar_name):
        match = pattern.match(token)
        if match is not None and match.end() == len(token):
            scopes.append(name)
    return scopes


def _surf_keyword_scopes(token: str) -> list[str]:
    grammar = json.loads(
        (GRAMMAR_DIR / "chelis.tmLanguage.json").read_text(encoding="utf-8")
    )
    scopes = []
    for pattern in grammar["repository"]["keywords"]["patterns"]:
        match = re.compile(pattern["match"]).match(token)
        if match is not None and match.end() == len(token):
            scopes.append(pattern["name"])
    return scopes


class NumericGrammarTests(unittest.TestCase):
    def test_surf_float_suffixes_on_integer_bodies_are_float_scoped(self) -> None:
        for token in ("42f32", "42f64", "42bf16", "42f16"):
            with self.subTest(token=token):
                self.assertEqual(
                    _scopes("chelis.tmLanguage.json", token),
                    ["constant.numeric.float.chelis"],
                )

    def test_surf_integer_suffixes_and_hex_values_remain_integer_scoped(self) -> None:
        for token in ("42", "42i32", "0x2a", "0b101"):
            with self.subTest(token=token):
                self.assertEqual(
                    _scopes("chelis.tmLanguage.json", token),
                    ["constant.numeric.integer.chelis"],
                )

    def test_deep_signed_float_suffixes_on_integer_bodies_are_float_scoped(self) -> None:
        for token in ("42f32", "-42f32", "42bf16", "-42f16"):
            with self.subTest(token=token):
                self.assertEqual(
                    _scopes("chelis-deep.tmLanguage.json", token),
                    ["constant.numeric.float.chelis-deep"],
                )

    def test_deep_integer_suffixes_remain_integer_scoped(self) -> None:
        for token in ("42", "-42", "42i32", "0x2a", "0b101"):
            with self.subTest(token=token):
                self.assertEqual(
                    _scopes("chelis-deep.tmLanguage.json", token),
                    ["constant.numeric.integer.chelis-deep"],
                )

    def test_malformed_suffixes_do_not_receive_a_numeric_scope(self) -> None:
        for grammar_name in (
            "chelis.tmLanguage.json",
            "chelis-deep.tmLanguage.json",
        ):
            for token in ("42f32x", "42i32f32"):
                with self.subTest(grammar=grammar_name, token=token):
                    self.assertEqual(_scopes(grammar_name, token), [])

    def test_surf_property_declaration_keywords_remain_control_scoped(self) -> None:
        for token in ("property", "forall", "where"):
            with self.subTest(token=token):
                self.assertEqual(
                    _surf_keyword_scopes(token),
                    ["keyword.control.chelis"],
                )


if __name__ == "__main__":
    unittest.main()
