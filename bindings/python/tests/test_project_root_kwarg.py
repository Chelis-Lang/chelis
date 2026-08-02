"""Gated coverage for the `project_root=` wrapper branching (issue #816).

The Python-side decision tree (the `project_root=True` rejection, the
`force_bare` derivation from `project_root=False`, and `Path` -> `str`
conversion) previously ran in no gate: CI's unittest discovery only picks up
`test_*.py`, and the reef oracle is manual. These tests exercise exactly the
wrapper layer with no compiler, no reef project, and no native artifacts,
following `test_dtype_ingress.py`'s pattern: the native module is a stub, so
the suite runs in CI where the compiled extension does not exist. Discovery
order means `chelis` may already be imported with that stub installed; either
way the stubs below are set per-test and restored.
"""

import importlib
import json
import sys
import types
import unittest
from pathlib import Path


class _ChelisError(Exception):
    pass


def _load_bindings_module():
    python_root = Path(__file__).resolve().parents[1]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

    if "chelis._native" not in sys.modules:
        native = types.ModuleType("chelis._native")
        native.ChelisError = _ChelisError
        sys.modules["chelis._native"] = native

    if "safetensors.numpy" not in sys.modules:
        safetensors = types.ModuleType("safetensors")
        safetensors_numpy = types.ModuleType("safetensors.numpy")
        safetensors_numpy.load_file = lambda _path: {}
        safetensors_numpy.save_file = lambda _tensors, _path: None
        safetensors.numpy = safetensors_numpy
        sys.modules["safetensors"] = safetensors
        sys.modules["safetensors.numpy"] = safetensors_numpy

    return importlib.import_module("chelis")


chelis = _load_bindings_module()

_MISSING = object()


class _CapturedCall(Exception):
    """Raised by the native stubs to prove the wrapper called through."""

    def __init__(self, args, kwargs):
        super().__init__("captured")
        self.args = args
        self.kwargs = kwargs


def _stub_capture(name):
    """Temporarily replace `chelis._native.<name>` with a capturing stub."""

    module = chelis._native
    original = getattr(module, name, _MISSING)

    def stub(*args, **kwargs):
        raise _CapturedCall(args, kwargs)

    setattr(module, name, stub)

    def restore():
        if original is _MISSING:
            delattr(module, name)
        else:
            setattr(module, name, original)

    return restore


class ProjectRootKwargTests(unittest.TestCase):
    def test_compile_and_load_rejects_project_root_true(self):
        with self.assertRaises(ValueError) as ctx:
            chelis.compile_and_load("model.ch", project_root=True)
        self.assertIn("project_root=True", str(ctx.exception))

    def test_eval_rejects_project_root_true(self):
        with self.assertRaises(ValueError) as ctx:
            chelis.eval("x = 1.0", project_root=True)
        self.assertIn("project_root=True", str(ctx.exception))

    def test_project_root_false_forces_bare(self):
        restore = _stub_capture("compile_and_load")
        try:
            with self.assertRaises(_CapturedCall) as ctx:
                chelis.compile_and_load("model.ch", project_root=False)
        finally:
            restore()
        self.assertIsNone(ctx.exception.kwargs["project_root"])
        self.assertTrue(ctx.exception.kwargs["force_bare"])

    def test_project_root_path_is_stringified(self):
        restore = _stub_capture("compile_and_load")
        try:
            with self.assertRaises(_CapturedCall) as ctx:
                chelis.compile_and_load(
                    "model.ch", project_root=Path("/some/reef/project")
                )
        finally:
            restore()
        self.assertEqual(
            ctx.exception.kwargs["project_root"], str(Path("/some/reef/project"))
        )
        self.assertFalse(ctx.exception.kwargs["force_bare"])

    def test_eval_project_root_path_is_stringified(self):
        restore = _stub_capture("eval_json")
        try:
            with self.assertRaises(_CapturedCall) as ctx:
                chelis.eval("x = 1.0", project_root=Path("/some/reef/project"))
        finally:
            restore()
        self.assertEqual(
            ctx.exception.kwargs["project_root"], str(Path("/some/reef/project"))
        )
        # The bindings payload still serializes (empty mapping).
        self.assertEqual(json.loads(ctx.exception.args[1]), {})


if __name__ == "__main__":
    unittest.main()
