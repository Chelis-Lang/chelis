"""Check native and generated Lean behavior of the pinned extraction models."""

import argparse
from difflib import unified_diff
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from extract import run

MODELS = {
    "loop_continuations": "LoopContinuations",
    "slice_iterators": "SliceIterators",
    "btree_maps": "BtreeMaps",
    "associated_types": "AssociatedTypes",
    "string_models": "StringModels",
    "production_sequences": "ProductionSequences",
    "btree_sets": "BtreeSets",
    "tuple_models": "TupleModels",
    "production_scalar_helpers": "ProductionScalarHelpers",
    "production_formatting": "ProductionFormatting",
    "disjoint_mutable": "DisjointMutable",
    "projection_iterator_state": "ProjectionIteratorState",
    "constant_typing": "ConstantTyping",
    "associated_iterators": "AssociatedIterators",
    "iterator_sources": "IteratorSources",
    "iterator_combinators": "IteratorCombinators",
    "flat_map_iterators": "FlatMapIterators",
    "result_iterators": "ResultIterators",
    "ended_shared_payload": "EndedSharedPayload",
    "float_operations": "FloatOperations",
    "shared_slice_metadata": "SharedSliceMetadata",
    "join_shared_projection": "JoinSharedProjection",
    "uninhabited_variants": "UninhabitedVariants",
    "inherited_dispatch": "InheritedDispatch",
    "move_cleanup": "MoveCleanup",
}

# Retain the post-move discriminant checks produced by drop elaboration.
MODEL_MIR = {"move_cleanup": "elaborated"}

# Match the production edition for let-chain temporary lifetimes.
MODEL_EDITIONS = {"join_shared_projection": "2024"}

# These regressions require a restricted root, rather than all local items.
MODEL_ROOTS = {
    "inherited_dispatch": ("inherited_dispatch::count_visible",),
}


def check_float_model(lean_library: Path, root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="chelis-float-model-") as directory:
        native = Path(directory) / "native"
        run([
            "model-rustc", "--edition=2021", str(root / "verification" / "float_model_oracle.rs"),
            "-o", str(native),
        ], root)
        expected = run([str(native)], root, capture=True)
        actual = run([
            "lake", "env", "lean", str(root / "verification" / "FloatModelChecks.lean"),
        ], lean_library, capture=True)
        if actual != expected:
            sys.stderr.writelines(unified_diff(
                expected.splitlines(keepends=True), actual.splitlines(keepends=True),
                fromfile="native Rust", tofile="Lean model",
            ))
            raise RuntimeError("Floating model disagrees with the native boundary oracle")
        print(f"Floating model smoke: {len(expected.splitlines())} native/Lean observations agree")


def check_model(name: str, lean_library: Path, root: Path) -> None:
    module = MODELS[name]
    edition = f"--edition={MODEL_EDITIONS.get(name, '2021')}"
    source = root / "verification" / f"{name}.rs"
    checks = root / "verification" / f"{module}Checks.lean"
    with tempfile.TemporaryDirectory(prefix=f"chelis-{name}-") as directory:
        output = Path(directory)
        native = output / "native"
        run(["model-rustc", edition, "--test", str(source), "-o", str(native)], root)
        run([str(native)], root)
        llbc = output / f"{name}.llbc"
        run([
            "charon", "rustc", "--preset=aeneas", f"--mir={MODEL_MIR.get(name, 'promoted')}",
            "--monomorphize-mut=except-types", "--remove-adt-clauses",
            "--lift-associated-types=*",
            "--error-on-warnings", "--abort-on-error", "--dest-file", str(llbc),
            *[
                argument
                for adapter in (
                    "map::Map", "filter_map::FilterMap", "filter::Filter",
                    "copied::Copied", "cloned::Cloned", "enumerate::Enumerate",
                    "take::Take", "rev::Rev",
                )
                for argument in ("--include", f"core::iter::adapters::{adapter}")
            ],
            "--include", "core::option::{core::option::Option}",
            "--include", "core::result::{core::result::Result}",
            "--include", "core::option::{core::ops::try_trait::Try}",
            "--include", "core::option::{core::ops::try_trait::FromResidual}",
            "--include", "core::result::{core::ops::try_trait::Try}",
            "--include", "core::result::{core::ops::try_trait::FromResidual}",
            "--include", "core::ops::deref::{core::ops::deref::Deref}",
            "--include", "core::ops::deref::{core::ops::deref::DerefMut}",
            "--include", "core::ops::arith::{core::ops::arith::Add}",
            "--include", "core::cmp",
            "--include", "{core::cmp::PartialEq}::ne",
            "--include", "{core::cmp::PartialOrd}",
            "--include", "{core::cmp::Ord}",
            "--include", "core::f64::consts::FRAC_PI_2",
            "--include", "core::iter::traits::iterator::Iterator::by_ref",
            "--include", "{core::iter::traits::iterator::Iterator}::by_ref",
            *[
                argument
                for entry in MODEL_ROOTS.get(name, ())
                for argument in ("--start-from", entry)
            ],
            "--", edition, "--crate-type=lib", str(source),
        ], root)
        run([
            "aeneas", "-backend", "lean", "-checks", "-abort-on-error",
            "-warnings-as-errors", "-strict-joins", "-sequential", "-no-progress-bar",
            "-dest", str(output), str(llbc),
        ], root)
        model = output / f"{module}.lean"
        lean_path = os.pathsep.join(filter(None, [str(output), os.environ.get("LEAN_PATH")]))
        lean = ["env", f"LEAN_PATH={lean_path}", "lake", "env", "lean"]
        run([
            *lean, f"--root={output}", "-o", str(model.with_suffix(".olean")), str(model),
        ], lean_library)
        run([*lean, f"--root={checks.parent}", str(checks)], lean_library)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aeneas-lean-dir",
        type=Path,
        required=True,
        help="Built backends/lean directory from the pinned patched Aeneas source",
    )
    parser.add_argument(
        "--model", choices=(*MODELS, "float_model"), action="append",
        help="Check this model (repeatable); by default check every model",
    )
    args = parser.parse_args()
    lean_library = args.aeneas_lean_dir.resolve()
    root = Path(__file__).resolve().parent.parent
    for name in args.model or ["float_model", *MODELS]:
        if name == "float_model":
            check_float_model(lean_library, root)
        else:
            check_model(name, lean_library, root)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
