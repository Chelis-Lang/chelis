"""Manual acceptance oracle for reef-context dependency resolution (issue #816).

Proves the Python bindings resolve reef-declared dependencies for both
`chelis.eval(..., project_root=)` and `chelis.compile_and_load(..., project_root=)`,
including the in-context entry-DAG trap (an entry that calls a library function must
still produce a callable, correctly-scoped compiled kernel).

This test builds its own temp reef project depending on the bundled `chelis-std`
package plus a sibling library module, so it is dev-compiler-clean and needs no
network. See the module docstring in `crates/chelis-python/tests/manual_reef_context.rs`
for the environment prerequisites and the Shoals-specific note (the published
Shoals 0.23.1 artifact fails HEAD's `with seed(...)` int64 rule — tracked as
chelis#825).

Run (from the repo root, with the bindings installed into `.venv`):

    export DYLD_LIBRARY_PATH="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
    export CHELIS_RUNTIME_DIR="$PWD/target/agents/<name>/debug"   # dir with libchelis_runtime.a
    export CHELIS_TOOLCHAIN=0.16.1
    .venv/bin/python bindings/python/tests/manual_reef_context.py
"""

import math
import os
import tempfile
import traceback

import numpy as np

import chelis


# The temp project's compiler pin must equal the dev workspace version the
# bindings were built from (chelis-reef rejects any other pin). The Rust
# driver (crates/chelis-python/tests/manual_reef_context.rs) sets this env
# var from CARGO_PKG_VERSION; export it manually when invoking this file
# directly. A hardcoded pin here goes stale at every version bump — that
# failure mode is exactly what broke this oracle at 0.18.1.
_PIN = os.environ.get("CHELIS_ORACLE_COMPILER_PIN")
if not _PIN:
    raise SystemExit(
        "CHELIS_ORACLE_COMPILER_PIN is not set; export it as `=<workspace "
        "version>` (the cargo test driver sets it automatically)"
    )

REEF_TOML = f"""\
[package]
name = "acctproj"
version = "0.1.0"
compiler = "{_PIN}"
module_prefix = "Acctproj"

[dependencies]
chelis-std = {{ version = "0.4.0" }}
"""

# A sibling library module: a pure-tensor function (callable-compilable) and a
# scalar function (host-lane only). The entry imports these across the module
# boundary, so resolving them requires the reef context — the #816 scenario.
LIB_CH = """\
module Acctproj.Lib
import Std.Scalar (abs)
export (scale_sq, dist_scalar)
def scale_sq(x: tensor[1, f32]) -> tensor[1, f32] = mul(copy(x), x)
def dist_scalar(a: f32, b: f32) -> f32 = abs(sub(a, b))
"""

# tensor-in/tensor-out entry that CALLS the library function scale_sq: this is
# the case the in-context entry-DAG trap fix must handle (compiled.checked has
# only new code; scale_sq lives in the context's library).
TENSOR_ENTRY_CH = """\
module Acctproj.Tp
import Acctproj.Lib (scale_sq)
def main(x: tensor[1, f32]) -> tensor[1, f32] = scale_sq(x)
"""

# scalar-signature entry: must be rejected by compile_and_load with wrap guidance,
# but must work through eval.
SCALAR_ENTRY_CH = """\
module Acctproj.Sc
import Acctproj.Lib (dist_scalar)
def main(a: f32, b: f32) -> f32 = dist_scalar(a, b)
"""

# self-contained source with zero reef imports: the no-root regression path.
SELF_CONTAINED = """\
def main(x: tensor[1, f32]) -> tensor[1, f32] = mul(copy(x), x)
"""


def _numpy(out):
    return np.from_dlpack(out) if hasattr(out, "__dlpack__") else np.asarray(out)


def _scalar(root_value):
    """Extract a Python float from an EvaluatedRoot.value (a scalar float or a
    TensorValue with a `.data` tuple)."""
    data = getattr(root_value, "data", None)
    if data is not None:
        return float(data[0])
    return float(root_value)


def main():
    failures = []

    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src")
        os.makedirs(src)
        with open(os.path.join(root, "reef.toml"), "w") as fh:
            fh.write(REEF_TOML)
        with open(os.path.join(src, "lib.ch"), "w") as fh:
            fh.write(LIB_CH)
        tensor_entry = os.path.join(src, "tp.ch")
        with open(tensor_entry, "w") as fh:
            fh.write(TENSOR_ENTRY_CH)
        scalar_entry = os.path.join(src, "sc.ch")
        with open(scalar_entry, "w") as fh:
            fh.write(SCALAR_ENTRY_CH)

        # --- Test 1: eval scalar form with project_root (dist_scalar via reef import).
        try:
            src_eval_scalar = (
                "module Acctproj.EvS\n"
                "import Acctproj.Lib (dist_scalar)\n"
                "d = dist_scalar(cast(10.0, f32), cast(3.5, f32))\n"
            )
            r = chelis.eval(src_eval_scalar, project_root=root)
            vals = [_scalar(x.value) for x in r.roots]
            assert vals and math.isclose(vals[0], 6.5, abs_tol=1e-6), vals
            print(f"[1] eval scalar (project_root)   -> dist_scalar(10, 3.5) = {vals[0]}  OK")
        except Exception as e:
            failures.append(("1 eval scalar", e))
            print(f"[1] eval scalar FAILED: {e!r}")

        # --- Test 2: compile_and_load tensor-wrapped entry (calls library scale_sq).
        eval_compiled = None
        try:
            m = chelis.compile_and_load(tensor_entry, project_root=root)
            out = _numpy(m(np.array([3.0], dtype=np.float32)))
            eval_compiled = float(out.reshape(-1)[0])
            assert math.isclose(eval_compiled, 9.0, abs_tol=1e-6), out
            print(
                f"[2] compile_and_load (project_root) -> main([3]) = {eval_compiled}  "
                f"inputs={m.input_names} outputs={m.output_names}  OK"
            )
        except Exception as e:
            failures.append(("2 compile tensor", e))
            print(f"[2] compile_and_load FAILED: {e!r}")

        # --- Test 2b: eval-vs-compiled agreement on the same program.
        try:
            r = chelis.eval(TENSOR_ENTRY_CH, {"x": np.array([3.0], dtype=np.float32)}, project_root=root)
            eval_val = _scalar(r.roots[0].value)
            assert eval_compiled is not None and math.isclose(eval_val, eval_compiled, abs_tol=1e-6)
            print(f"[2b] eval main([3]) = {eval_val}  agrees with compiled {eval_compiled}  OK")
        except Exception as e:
            failures.append(("2b agreement", e))
            print(f"[2b] agreement FAILED: {e!r}")

        # --- Test 3: no-root regression. Self-contained source, no kwarg.
        with tempfile.TemporaryDirectory() as bare:
            bare_src = os.path.join(bare, "model.ch")
            with open(bare_src, "w") as fh:
                fh.write(SELF_CONTAINED)
            try:
                m = chelis.compile_and_load(bare_src)  # no project_root
                out = _numpy(m(np.array([4.0], dtype=np.float32)))
                assert math.isclose(float(out.reshape(-1)[0]), 16.0, abs_tol=1e-6), out
                print(f"[3] no-root self-contained         -> main([4]) = {out.reshape(-1)[0]}  OK")
            except Exception as e:
                failures.append(("3 no-root", e))
                print(f"[3] no-root FAILED: {e!r}")

        # --- Test 4: auto-discovery. Same as (2) but no project_root; file in tree.
        try:
            m = chelis.compile_and_load(tensor_entry)  # discovers reef.toml by walking up
            out = _numpy(m(np.array([5.0], dtype=np.float32)))
            assert math.isclose(float(out.reshape(-1)[0]), 25.0, abs_tol=1e-6), out
            print(f"[4] auto-discovery (no project_root)-> main([5]) = {out.reshape(-1)[0]}  OK")
        except Exception as e:
            failures.append(("4 auto-discovery", e))
            print(f"[4] auto-discovery FAILED: {e!r}")

        # --- Test 5: scalar-signature entry -> compile_and_load rejects with wrap guidance.
        try:
            chelis.compile_and_load(scalar_entry, project_root=root)
            failures.append(("5 scalar reject", "expected rejection, got a model"))
            print("[5] scalar reject FAILED: expected rejection")
        except chelis.ChelisError as e:
            msg = str(e)
            assert "tensor[1, f32]" in msg or "no callable interface" in msg, msg
            print(f"[5] scalar entry rejected with wrap guidance  OK")
        except Exception as e:
            failures.append(("5 scalar reject", e))
            print(f"[5] scalar reject FAILED (wrong error): {e!r}")

        # --- Test 5b: the same scalar program works through eval.
        try:
            src_sc = (
                "module Acctproj.EvSc\n"
                "import Acctproj.Lib (dist_scalar)\n"
                "d = dist_scalar(cast(10.0, f32), cast(3.5, f32))\n"
            )
            r = chelis.eval(src_sc, project_root=root)
            assert math.isclose(_scalar(r.roots[0].value), 6.5, abs_tol=1e-6)
            print("[5b] scalar program via eval works           OK")
        except Exception as e:
            failures.append(("5b scalar eval", e))
            print(f"[5b] scalar eval FAILED: {e!r}")

    if failures:
        print("\nFAILURES:")
        for name, err in failures:
            print(f"  - {name}: {err!r}")
        raise SystemExit(1)
    print("\nAll reef-context acceptance checks passed.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
