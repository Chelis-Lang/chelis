from __future__ import annotations

from pathlib import Path
import tempfile
import warnings

import chelis
import numpy as np
import torch


PROGRAM = """def relu4(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)
"""

# Issue #817: a helper def alongside the entry def. The callable interface
# must scope to `solve` (params `a`, `b`), NOT the merged `helper`+`solve`
# parameter set.
MULTI_DEF_PROGRAM = """def helper(x: tensor[1, f32]) -> tensor[1, f32] = mul(x, x)
def solve(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[1, f32] = add(helper(a), helper(b))
"""

# Issue #818: a single-def block body that reuses a parameter across
# statements and uses `concat`. The callable interface must be `(a, b)` with
# one output, not the empty manifest the host-lane early-return produced.
CONCAT_PROGRAM = """def main(a: tensor[1, f32], b: tensor[1, f32]) -> tensor[2, f32] = {
  x = mul(copy(a), b)
  y = add(a, b)
  concat([x, y], cast(0, i32))
}
"""


def _eval_root(program: str, bindings: dict[str, np.ndarray], root_name: str) -> np.ndarray:
    result = chelis.eval(program, {k: v for k, v in bindings.items()})
    root = next(root for root in result.roots if root.name == root_name)
    return np.asarray(root.value.data, dtype=np.float32)


def check_entry_scoped_metadata(root: Path) -> None:
    # #817: multi-def file, entry_name selects `solve`.
    multi_path = root / "multi_def.ch"
    multi_path.write_text(MULTI_DEF_PROGRAM)
    solve = chelis.compile_and_load(
        multi_path,
        entry_name="solve",
        artifact_dir=root / "artifacts_multi",
    )
    assert solve.input_names == ("a", "b"), solve.input_names
    assert len(solve.output_names) == 1, solve.output_names

    a = np.array([3.0], dtype=np.float32)
    b = np.array([4.0], dtype=np.float32)
    compiled_out = np.from_dlpack(solve(a=a, b=b))
    # Ground truth: helper(a) + helper(b) = a*a + b*b = 9 + 16 = 25.
    assert np.allclose(compiled_out, np.array([25.0], dtype=np.float32)), compiled_out
    # Eval-vs-compiled agreement (binding `x` too lets the `helper` root eval).
    eval_out = _eval_root(
        MULTI_DEF_PROGRAM,
        {"a": a, "b": b, "x": np.array([0.0], dtype=np.float32)},
        "solve",
    )
    assert np.allclose(compiled_out, eval_out, atol=1e-6), (compiled_out, eval_out)

    # #818: single-def concat body.
    concat_path = root / "concat.ch"
    concat_path.write_text(CONCAT_PROGRAM)
    concat = chelis.compile_and_load(
        concat_path,
        artifact_dir=root / "artifacts_concat",
    )
    assert concat.input_names == ("a", "b"), concat.input_names
    assert len(concat.output_names) == 1, concat.output_names

    compiled_concat = np.from_dlpack(concat(a=a, b=b))
    # Ground truth: concat(mul(a, b), add(a, b)) = [12.0, 7.0].
    assert np.allclose(
        compiled_concat, np.array([12.0, 7.0], dtype=np.float32)
    ), compiled_concat
    # Eval-vs-compiled note: `chelis.eval` currently produces NO root for a def
    # whose body uses a host-runtime builtin such as `concat` (the def is
    # excluded from the pure tensor-root set — the same host-lane exclusion
    # that caused #818's empty manifest), even when its parameters are bound.
    # That is a separate eval-path gap tracked as chelis#820, not part of the
    # compile_and_load metadata contract this test guards, so the #818 compiled
    # result is validated against the hand-computed ground truth above rather
    # than against `chelis.eval`.
    concat_eval = chelis.eval(CONCAT_PROGRAM, {"a": a, "b": b})
    assert all(root.name != "main" for root in concat_eval.roots), (
        "if chelis.eval starts exposing the concat def root, tighten this into "
        "a direct eval-vs-compiled agreement assertion"
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as meta_tmpdir:
        check_entry_scoped_metadata(Path(meta_tmpdir))

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source_path = root / "model.ch"
        source_path.write_text(PROGRAM)

        compiled = chelis.compile_and_load(
            source_path,
            artifact_dir=root / "artifacts",
        )
        assert isinstance(compiled, chelis.CompiledModel)
        assert compiled.target == "c"
        assert compiled.path.endswith("model.so")
        assert compiled.input_names == ("x",)
        assert compiled.output_names == ("root0",)

        x_torch = torch.tensor([-1.0, 2.0, -3.0, 4.0], dtype=torch.float32)
        out_torch = torch.from_dlpack(compiled(x=x_torch))
        assert torch.equal(out_torch, torch.tensor([0.0, 2.0, 0.0, 4.0]))

        x_numpy = np.arange(4, dtype=np.float32)
        compiled_numpy = compiled(x=x_numpy)
        assert np.array_equal(np.from_dlpack(compiled_numpy), np.arange(4, dtype=np.float32))

        base = np.arange(8, dtype=np.float32).reshape(2, 4)
        strided = base[1]
        wrapped = chelis.from_dlpack(strided)
        roundtrip = np.from_dlpack(wrapped)
        assert roundtrip.__array_interface__["data"][0] == strided.__array_interface__["data"][0]
        assert np.array_equal(roundtrip, strided)

        try:
            compiled(x=torch.ones(4, dtype=torch.float64))
        except ValueError as err:
            assert "float32" in str(err)
        else:
            raise AssertionError("expected ValueError for unsupported dtype")

        try:
            compiled(x=torch.ones(5, dtype=torch.float32))
        except ValueError as err:
            assert "axis 0 expected 4" in str(err)
        else:
            raise AssertionError("expected ValueError for shape mismatch")

        if torch.version.hip and torch.cuda.is_available():
            hip_compiled = chelis.compile_and_load(
                source_path,
                target="hip",
                artifact_dir=root / "artifacts_hip",
            )
            x_gpu = x_torch.to("cuda")
            hip_out = torch.from_dlpack(hip_compiled(x=x_gpu)).cpu()
            assert torch.equal(hip_out, torch.tensor([0.0, 2.0, 0.0, 4.0]))

        source_path.write_text(
            "def relu4(x: tensor[4, f32]) -> tensor[4, f32] = neg(x)\n"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = chelis.load(compiled.path)
            assert isinstance(loaded, chelis.CompiledModel)
            assert any("source has changed since this artifact was compiled" in str(w.message) for w in caught)
            loaded_out = torch.from_dlpack(loaded(x=x_torch))
            assert torch.equal(loaded_out, torch.tensor([0.0, 2.0, 0.0, 4.0]))


if __name__ == "__main__":
    main()
