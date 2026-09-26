"""Check native and generated Lean behavior of the pinned extraction models."""

import argparse
from difflib import unified_diff
import json
import os
from pathlib import Path
import shlex
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
    "flatten_iterators": "FlattenIterators",
    "double_ended_adapters": "DoubleEndedAdapters",
    "str_split": "StrSplit",
    "option_collectors": "OptionCollectors",
    "range_inclusive_collectors": "RangeInclusiveCollectors",
    "result_iterators": "ResultIterators",
    "ended_shared_payload": "EndedSharedPayload",
    "float_operations": "FloatOperations",
    "shared_slice_metadata": "SharedSliceMetadata",
    "join_shared_projection": "JoinSharedProjection",
    "uninhabited_variants": "UninhabitedVariants",
    "inherited_dispatch": "InheritedDispatch",
    "move_cleanup": "MoveCleanup",
    "capacity_std_models": "CapacityStdModels",
    "generic_collectors": "GenericCollectors",
    "range_product_models": "RangeProductModels",
    "lowering_runtime_models": "LoweringRuntimeModels",
    "start_from_roots": "StartFromRoots",
    "owned_zip": "OwnedZip",
    "kernel_std_models": "KernelStdModels",
    "text_std_models": "TextStdModels",
    "closure_mut_args": "ClosureMutArgs",
    "primitive_order_models": "PrimitiveOrderModels",
    "checked_place_updates": "CheckedPlaceUpdates",
    "eval_std_models": "EvalStdModels",
    "float_std_models": "FloatStdModels",
    "misc_std_models": "MiscStdModels",
}

# Fixtures of crate-local native primitives compile under the production crate
# name, so that the models keyed on production paths replace their primitives.
MODEL_CRATES = {"lowering_runtime_models": "chelis_ir"}

# The native shell of a primitives fixture: the items of
# verification/native_shell.json under this prefix, hard-opaque as in the
# production extraction.
MODEL_SHELLS = {"lowering_runtime_models": "chelis_ir::lowering_runtime::"}

# Retain the post-move discriminant checks produced by drop elaboration.
MODEL_MIR = {"move_cleanup": "elaborated"}

# Match the production edition for let-chain temporary lifetimes.
MODEL_EDITIONS = {"join_shared_projection": "2024"}

# These regressions require a restricted root, rather than all local items.
MODEL_ROOTS = {
    "inherited_dispatch": ("inherited_dispatch::count_visible",),
    # Charon's canonical patterns, as in the whole-core chunks, and a comma-joined list.
    "start_from_roots": (
        "start_from_roots::plain,start_from_roots::other",
        "start_from_roots::{impl start_from_roots::Counter}::bump",
        "start_from_roots::{impl start_from_roots::Scale<u32> for start_from_roots::Counter}::scale",
        "start_from_roots::offsets::{impl core::ops::function::Fn<Tuple<u32>>"
        " for start_from_roots::offsets::closure#10}::call",
        "start_from_roots::_#1::{impl start_from_roots::Counter}::reset",
    ),
    "btree_maps": tuple(f"btree_maps::{entry}" for entry in (
        "ordered_insert", "equivalent_replacement", "occupied_restoration",
        "vacant_restoration", "dropped_entries", "remove_and_lookup",
        "missing_lookup", "shared_exhaustion", "owned_exhaustion", "owned_collect",
        "shared_collect", "duplicate_from_iterator", "empty_from_iterator",
        "wrapped_entry_restoration", "zipped_iterator_callbacks", "keys_collected",
        "keys_and_values", "maps_equal", "delegated_collection_state", "borrowed_for_loop",
        "borrowed_generic_total", "generic_loop_totals",
    )),
    "btree_sets": (
        "btree_sets::duplicate_from_iterator", "btree_sets::borrowed_lookup",
        "btree_sets::delegated_collection_state",
    ),
    "generic_collectors": tuple(f"generic_collectors::{entry}" for entry in (
        "map_collected", "set_collected", "map_extended", "set_extended", "custom_source",
        "custom_vec",
    )),
    "lowering_runtime_models": tuple(f"chelis_ir::{entry}" for entry in (
        "speculation_flags", "summary_probes", "preflight_frames", "host_inference_vars",
        "profile_declarations", "profile_unissued_mark", "scoped_push", "grown_push",
        "speculative_rejection", "rejection_after_inner_catch", "fatal_reraise",
        "uncaught_rejection", "caught_with_state", "host_scopes",
        "lowering_runtime::{chelis_ir::lowering_runtime::LoweringFailure}::resume_at_boundary",
    )),
    "associated_iterators": tuple(f"associated_iterators::{entry}" for entry in (
        "copied_signature", "cloned_signature", "copied_resume", "cloned_resume",
        "copied_any_resumes", "cloned_any_resumes", "cloned_first", "copied_count",
        "cloned_count", "copied_count_override", "cloned_count_override",
        "filter_resume", "filter_count", "filter_panics", "owned_array_resume",
        "owned_array_any_resume", "borrowed_fold_override", "copied_nth_override",
        "mapped_fold_override", "cloned_fold_override", "map_nth_state",
        "filter_find_override", "filter_map_find_override", "nested_filter_find_dispatch",
        "all_state", "find_state", "reversed_find_override", "reversed_rfind_override",
        "find_map_state", "position_state", "borrowed_filter_map_collect",
    )),
}

# These native behaviors require collection hooks absent from the value-only
# models. Strict extraction must reject them, including through generic delegates.
REJECTED_ROOTS = {
    "collection_contracts": tuple(f"collection_contracts::{entry}" for entry in (
        "reject_vec_hint", "reject_empty_hint", "reject_map_hint", "reject_set_hint",
        "reject_extend_hint", "reject_wrapped_hint", "reject_result_hint",
        "reject_generic_collector", "reject_zip_consumer",
        "reject_borrowed_map_hint", "reject_mapped_vec_zip", "reject_drop_vec_zip",
    )),
    # A native-shell scope on a borrow of a borrow: its instance is not the model at
    # an extended state (a staged backward function, see ModelStateABI.threaded_fun_body).
    "lowering_runtime_models": ("chelis_ir::reject_nested_borrow_state",),
}

# The admission diagnostic that each fixture's rejected roots must report.
REJECTED_DIAGNOSTICS = {
    "collection_contracts": "E_UNSUPPORTED_COLLECTION",
    "lowering_runtime_models": "Model at a borrowing state",
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


# Models of external crates: native oracle binaries of the standalone
# `crate_model_oracle` package (pinned crate versions) against Lean check files.
CRATE_MODELS = {
    "half_model": ("half_model_oracle", "HalfModelChecks.lean"),
    "biguint_model": ("biguint_model_oracle", "BigUintModelChecks.lean"),
    "sha256_model": ("sha256_model_oracle", "Sha256ModelChecks.lean"),
}


def crate_oracle_command(binary: str, target: Path, root: Path) -> list[str]:
    return [
        "cargo", "run", "--offline", "--locked", "--release", "--quiet",
        "--manifest-path", str(root / "verification" / "crate_model_oracle" / "Cargo.toml"),
        "--target-dir", str(target), "--bin", binary,
    ]


def check_crate_model(name: str, lean_library: Path, root: Path) -> None:
    binary, checks = CRATE_MODELS[name]
    with tempfile.TemporaryDirectory(prefix=f"chelis-{name}-") as directory:
        target = Path(directory)
        if name == "half_model":
            # Exhaustive: host conversions (FCVT on aarch64+fp16) equal the crate's
            # software algorithms that the Lean models transliterate.
            run(crate_oracle_command("half_hw_sw", target, root), root)
        expected = run(crate_oracle_command(binary, target, root), root, capture=True)
    actual = run([
        "lake", "env", "lean", str(root / "verification" / checks),
    ], lean_library, capture=True)
    if actual != expected:
        sys.stderr.writelines(unified_diff(
            expected.splitlines(keepends=True), actual.splitlines(keepends=True),
            fromfile="native Rust", tofile="Lean model",
        ))
        raise RuntimeError(f"{name} disagrees with the native crate oracle")
    print(f"{name} smoke: {len(expected.splitlines())} native/Lean observations agree")


def charon_command(
    name: str, source: Path, llbc: Path, roots: tuple[str, ...],
) -> list[str]:
    return [
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
        *[argument for entry in roots for argument in ("--start-from", entry)],
        *[argument for entry in shell_patterns(name) for argument in ("--opaque-hard", entry)],
        "--", f"--edition={MODEL_EDITIONS.get(name, '2021')}", "--crate-type=lib",
        *crate_arguments(name), str(source),
    ]


def crate_arguments(name: str) -> list[str]:
    """The crate name of a fixture."""
    return [f"--crate-name={MODEL_CRATES[name]}"] if name in MODEL_CRATES else []


def shell_patterns(name: str) -> tuple[str, ...]:
    """The native shell of a fixture (see MODEL_SHELLS)."""
    prefix = MODEL_SHELLS.get(name)
    if prefix is None:
        return ()
    shell = json.loads((Path(__file__).resolve().parent / "native_shell.json").read_text())
    return tuple(
        item["pattern"] for item in shell["items"] if item["pattern"].startswith(prefix)
    )



def aeneas_command(llbc: Path, output: Path) -> list[str]:
    return [
        "aeneas", "-backend", "lean", "-checks", "-abort-on-error",
        "-warnings-as-errors", "-strict-joins", "-sequential", "-no-progress-bar",
        "-dest", str(output), str(llbc),
    ]


def check_rejected_roots(name: str, source: Path, output: Path, root: Path) -> None:
    for entry in REJECTED_ROOTS.get(name, ()):
        directory = output / entry.rsplit("::", 1)[1]
        directory.mkdir()
        llbc = directory / f"{name}.llbc"
        run(charon_command(name, source, llbc, (entry,)), root)
        command = aeneas_command(llbc, directory)
        print("+ " + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        diagnostic = REJECTED_DIAGNOSTICS[name]
        if result.returncode == 0:
            raise RuntimeError(f"Unsupported root was accepted: {entry}")
        if diagnostic not in result.stdout + result.stderr:
            raise RuntimeError(f"{entry} failed without its admission diagnostic {diagnostic}")
        print(f"Admission: {entry} rejected with {diagnostic}")


def check_collection_contracts(root: Path) -> None:
    source = root / "verification" / "collection_contracts.rs"
    with tempfile.TemporaryDirectory(prefix="chelis-collection-contracts-") as directory:
        output = Path(directory)
        native = output / "native"
        run(["model-rustc", "--edition=2021", "--test", str(source), "-o", str(native)], root)
        run([str(native)], root)
        check_rejected_roots("collection_contracts", source, output, root)


def check_model(name: str, lean_library: Path, root: Path) -> None:
    module = MODELS[name]
    edition = f"--edition={MODEL_EDITIONS.get(name, '2021')}"
    source = root / "verification" / f"{name}.rs"
    checks = root / "verification" / f"{module}Checks.lean"
    with tempfile.TemporaryDirectory(prefix=f"chelis-{name}-") as directory:
        output = Path(directory)
        native = output / "native"
        run([
            "model-rustc", edition, *crate_arguments(name), "--test", str(source), "-o", str(native),
        ], root)
        run([str(native)], root)
        llbc = output / f"{name}.llbc"
        run(charon_command(name, source, llbc, MODEL_ROOTS.get(name, ())), root)
        run(aeneas_command(llbc, output), root)
        model = output / f"{module}.lean"
        lean_path = os.pathsep.join(filter(None, [str(output), os.environ.get("LEAN_PATH")]))
        lean = ["env", f"LEAN_PATH={lean_path}", "lake", "env", "lean"]
        run([
            *lean, f"--root={output}", "-o", str(model.with_suffix(".olean")), str(model),
        ], lean_library)
        run([*lean, f"--root={checks.parent}", str(checks)], lean_library)
        check_rejected_roots(name, source, output, root)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aeneas-lean-dir",
        type=Path,
        required=True,
        help="Built backends/lean directory from the pinned patched Aeneas source",
    )
    parser.add_argument(
        "--model", choices=(*MODELS, "float_model", *CRATE_MODELS, "collection_contracts"),
        action="append",
        help="Check this model (repeatable); by default check every model",
    )
    args = parser.parse_args()
    lean_library = args.aeneas_lean_dir.resolve()
    root = Path(__file__).resolve().parent.parent
    for name in args.model or ["float_model", *CRATE_MODELS, *MODELS, "collection_contracts"]:
        if name == "float_model":
            check_float_model(lean_library, root)
        elif name in CRATE_MODELS:
            check_crate_model(name, lean_library, root)
        elif name == "collection_contracts":
            check_collection_contracts(root)
        else:
            check_model(name, lean_library, root)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
