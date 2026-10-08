"""Check native and generated Lean behavior of the pinned extraction models."""

import argparse
from difflib import unified_diff
import json
import os
from pathlib import Path
import re
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
    "str_char_patterns": "StrCharPatterns",
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
    "sort_std_models": "SortStdModels",
    "preflight_std_models": "PreflightStdModels",
    "retain_closure_state": "RetainClosureState",
    "str_pattern_models": "StrPatternModels",
    "iterator_consumer_models": "IteratorConsumerModels",
    "thiserror_models": "ThiserrorModels",
    "owned_rev_models": "OwnedRevModels",
    "provided_method_models": "ProvidedMethodModels",
    "collection_std_models": "CollectionStdModels",
    "int_method_models": "IntMethodModels",
    "slice_str_models": "SliceStrModels",
    "hash_models": "HashModels",
    "char_case_models": "CharCaseModels",
    "slice_search_models": "SliceSearchModels",
    "scalar_ref_models": "ScalarRefModels",
    "collection_misc_models": "CollectionMiscModels",
    "iter_order_models": "IterOrderModels",
    "collection_convert_models": "CollectionConvertModels",
    "str_convert_models": "StrConvertModels",
    "fmt_hex_char_models": "FmtHexCharModels",
    "nonzero_models": "NonzeroModels",
    "reverse_models": "ReverseModels",
    "string_write_models": "StringWriteModels",
    "cow_asref_models": "CowAsrefModels",
    "small_std_models": "SmallStdModels",
    "btree_from_array_models": "BtreeFromArrayModels",
    "assert_unwind_safe_models": "AssertUnwindSafeModels",
    "hash_slice_models": "HashSliceModels",
    "native_services_models": "NativeServicesModels",
}

# Fixtures of crate-local native primitives compile under the production crate
# name, so that the models keyed on production paths replace their primitives.
MODEL_CRATES = {
    "lowering_runtime_models": "chelis_ir",
    "native_services_models": "chelis_types",
    "thiserror_models": "thiserror",
}

# The native shell of a primitives fixture: the items of
# verification/native_shell.json under this prefix, hard-opaque as in the
# production extraction.
MODEL_SHELLS = {
    "lowering_runtime_models": "chelis_ir::lowering_runtime::",
    "native_services_models": "chelis_types::native::",
}

# Crates of the workspace that a fixture's native test links (`--extern`), as its production
# crate does. Cargo builds them from the workspace lock file, so that native test compiles with
# the workspace compiler (rust-toolchain.toml) instead of model-rustc. Charon compiles the
# library without them: only hard-opaque shell items may use them.
MODEL_NATIVE_CRATES = {"native_services_models": ("stacker",)}

# Fixtures of a dependency crate's items (compiled under the dependency's crate name, see
# MODEL_CRATES) copy those items verbatim; they are hard-opaque, as the items of a
# dependency are in production, so that its library models replace them.
MODEL_DEPENDENCIES = {"thiserror_models": ("thiserror::display",)}

# Retain the post-move discriminant checks produced by drop elaboration.
MODEL_MIR = {"move_cleanup": "elaborated"}

# Match the production edition for let-chain temporary lifetimes.
MODEL_EDITIONS = {"join_shared_projection": "2024"}

# These regressions require a restricted root, rather than all local items.
MODEL_ROOTS = {
    # The provided-method copies are roots, as in the whole-core chunks (every item of a crate).
    "provided_method_models": (
        "provided_method_models::widen",
        "provided_method_models::{impl core::clone::Clone for provided_method_models::Span}",
        "provided_method_models::{impl core::clone::Clone for provided_method_models::Span}::clone_from",
        "provided_method_models::{impl core::error::Error for provided_method_models::ParseError}",
        "provided_method_models::{impl core::error::Error for provided_method_models::ParseError}::source",
    ),
    "retain_closure_state": (
        "retain_closure_state::{impl retain_closure_state::Subst}::take_operand_gates_on",
        "retain_closure_state::cancel_shape_checks",
        "retain_closure_state::drop_odds",
        "retain_closure_state::first_occurrences",
    ),
    "production_sequences": tuple(f"production_sequences::{entry}" for entry in (
        "remove_at",
        "clear_then_extend",
        "append_resumes",
        "extend_callback_state",
        "extend_failure",
        "stable_duplicate_keys",
        "sort_values",
        "sort_key_failure",
        "collect_ad_callback_order",
        "collect_result_short_circuit",
    )),
    "inherited_dispatch": ("inherited_dispatch::count_visible",),
    # Every item but the rejected root (see REJECTED_ROOTS).
    "iter_order_models": tuple(f"iter_order_models::{entry}" for entry in (
        "representative", "least_index", "counted_min", "earliest_even_half",
        "counted_filter_min", "is_identity", "axes_differ", "quotients_equal",
        "quotients_differ_from_odd", "module_segments", "counted_skip", "first_two_kept",
    )),
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
        "borrowed_generic_total", "generic_loop_totals", "std_entry_methods",
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
        "uncaught_rejection", "caught_with_state", "host_scopes", "fatal_before_first_catch",
        "lowering_runtime::{chelis_ir::lowering_runtime::LoweringFailure}::resume_at_boundary",
    )),
    "native_services_models": tuple(f"chelis_types::scenarios::{entry}" for entry in (
        "frame_exhausted", "descend", "ascend", "enter_outermost_scope", "grown_entry", "polls",
        "profiled",
    )),
    "sort_std_models": tuple(f"sort_std_models::{entry}" for entry in (
        "sort_usizes", "sort_pairs", "sort_strings", "sort_strs", "sort_flags", "sort_ids",
        "sort_slots", "sort_descending", "sort_ascending_by", "sort_ids_descending",
        "sort_by_first", "sort_by_names", "sort_by_name_descending", "sort_by_copied",
        "sort_stable_ids", "sort_stable_strings", "sort_stable_options", "sort_stable_in_place",
        "sort_capacities",
        "sort_by_key_field", "sort_by_key_fields", "sort_by_key_reverse", "sort_by_key_str",
        "sort_by_key_captured", "sort_by_key_fn_item", "sort_by_key_local_closure",
        "sort_by_key_cloned", "sort_by_key_in_place",
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
    # Iterator::eq over a closure capturing a mutable borrow: its model returns no state
    # (see ModelStateABI.value_only_model).
    "iter_order_models": ("iter_order_models::reject_stateful_eq",),
    # A sort_by_key whose key function has captured mutable state (see
    # llbc/SortContracts.ml): its final state depends on std's evaluation sequence.
    "production_sequences": ("production_sequences::reject_stateful_sort_key",),
    "collection_contracts": tuple(f"collection_contracts::{entry}" for entry in (
        "reject_vec_hint", "reject_empty_hint", "reject_map_hint", "reject_set_hint",
        "reject_extend_hint", "reject_wrapped_hint", "reject_result_hint",
        "reject_generic_collector", "reject_zip_consumer",
        "reject_borrowed_map_hint", "reject_mapped_vec_zip", "reject_drop_vec_zip",
        "reject_dispatch_method", "reject_dispatch_impl",
        "reject_mapped_nested_zip", "reject_borrowed_zip_left", "reject_cloned_zip_left",
        "reject_deque_hint", "reject_take_mapped_skip", "reject_skip_in_place",
    )),
    # A native-shell scope on a borrow of a borrow: its instance is not the model at
    # an extended state (a staged backward function, see ModelStateABI.threaded_fun_body).
    "lowering_runtime_models": ("chelis_ir::reject_nested_borrow_state",),
    # Sorts whose Equal may hold between distinct elements, stable sorts whose
    # comparator is not Ord::cmp of one key of both elements, key functions that
    # mutate their captures, and types whose Ord or PartialOrd is hand-written
    # (see llbc/SortContracts.ml).
    "sort_std_models": tuple(f"sort_std_models::{entry}" for entry in (
        "reject_custom_ord", "reject_partial_comparator", "reject_stateful_comparator",
        "reject_custom_ord_comparator", "reject_key", "reject_by_crossed_keys",
        "reject_by_computed_key", "reject_by_stateful", "reject_by_float",
        "reject_by_custom_ord", "reject_sort_custom_ord", "reject_sort_custom_lt",
        "reject_unstable_custom_lt", "reject_by_key_stateful", "reject_by_key_custom_ord",
    )),
}

# The admission diagnostic that each fixture's rejected roots must report.
REJECTED_DIAGNOSTICS = {
    "iter_order_models": "takes its inputs by value and returns no state",
    "production_sequences": "E_UNSUPPORTED_SORT",
    "collection_contracts": "E_UNSUPPORTED_COLLECTION",
    "lowering_runtime_models": "Model at a borrowing state",
    "sort_std_models": "E_UNSUPPORTED_SORT",
}


# Fixtures that repeat production code verbatim, up to recorded differences. A region starts
# at `start` and ends with the block of the item that starts at `end` (both texts occur in
# both files). The guard compares token sequences without comments, blank space, trailing
# commas and visibility (`pub`, `pub(crate)`, ...), so formatting does not matter. Each recorded difference (production text, fixture text) must
# occur exactly once in the production region; after these substitutions the regions must be
# equal. So a change of a production primitive without the same fixture change fails the check.
FIXTURE_COPIES = {
    "lowering_runtime_models": [{
        "production": "crates/chelis-ir/src/lowering_runtime.rs",
        "fixture": "verification/lowering_runtime_models.rs",
        "start": "fn resume_native_unwind",
        "end": "fn unrepresentable_diagnostic",
        "differences": (
            # `stacker` is not linkable into the single-file fixture.
            ("stacker::maybe_grow(red_zone, stack_size, || f(state))",
             "let _ = (red_zone, stack_size); f(state)"),
            ("stacker::grow(LOWERING_GROW_SEGMENT_BYTES, move || f(runtime, state))",
             "f(runtime, state)"),
            ("const LOWERING_GROW_SEGMENT_BYTES: usize = 512 * 1024 * 1024;", ""),
            # The fixture's `LowerDiagnostic` is reduced to a code and the `fatal` flag.
            ("|message| { LowerDiagnostic::new("
             "message.unwrap_or_else(|| \"internal lowering error\".to_string()), None, None, ) }",
             "|message| LowerDiagnostic::new(if message.is_some() { 2 } else { 3 })"),
            ("LowerDiagnostic::new(\"program uses a form that is not supported by IR evaluation "
             "yet; use `chelis build --target c` instead\", None, None, )",
             "LowerDiagnostic::new(1)"),
        ),
    }],
    "native_services_models": [{
        "production": "crates/chelis-types/src/native.rs",
        "fixture": "verification/native_services_models.rs",
        "start": "pub struct NativeServices",
        "end": "impl NativeServices {",
        "differences": (),
    }],
    "sha256_model": [{
        "production": "crates/chelis-types/src/native.rs",
        "fixture": "verification/crate_model_oracle/src/bin/sha256_model_oracle.rs",
        "start": "fn sha256_concat",
        "end": "fn sha256_concat",
        "differences": (),
    }],
    "unwind_replay": [
        {
            "production": "crates/chelis-ir/src/lowering_runtime.rs",
            "fixture": "verification/unwind_replay_oracle.rs",
            "start": item,
            "end": item,
            "differences": (
                ("stacker::grow(LOWERING_GROW_SEGMENT_BYTES, move || f(runtime, state))",
                 "f(runtime, state)"),
            ) if item == "fn catch_native_unwind" else (),
        }
        for item in (
            "enum LoweringFailure",
            "fn into_diagnostic",
            "fn resume_at_boundary",
            "fn resume_native_unwind",
            "fn scoped",
            "struct LoweringRuntime",
            "fn unrepresentable_panic_suppressed",
            "fn set_unrepresentable_panic_suppressed",
            "fn set_output_suppressed",
            "fn with_suppress_unrepresentable_panic",
            "fn raise_lowering_diagnostic",
            "fn raise_unrepresentable",
            "fn observe_raise",
            "fn raise_fatal_lowering_diagnostic",
            "fn catch_lowering<",
            "fn catch_lowering_with",
            "fn catch_native_unwind",
            "fn with_native_lowering_runtime",
            "fn unrepresentable_diagnostic",
        )
    ] + [
        {
            "production": "crates/chelis-ir/src/lowering_runtime.rs",
            "fixture": "verification/unwind_replay_oracle.rs",
            "start": "static CHELIS_PANIC_HOOK_INSTALLED",
            "end": "fn install_chelis_panic_hook",
            "differences": (),
        },
        {
            "production": "crates/chelis-ir/src/lowering_runtime.rs",
            "fixture": "verification/unwind_replay_oracle.rs",
            "start": "struct NativeLoweringState",
            "end": "struct NativeLoweringState",
            "differences": (
                ("struct NativeLoweringState { suppress_unrepresentable: Cell<bool>, "
                 "suppress_output: Cell<bool>, summary_probe_builds: Cell<u64>, "
                 "next_preflight_frame: Cell<u64>, }",
                 "struct NativeLoweringState { suppress_unrepresentable: Cell<bool>, "
                 "suppress_output: Cell<bool>, }"),
            ),
        },
        {
            "production": "crates/chelis-ir/src/lowering_runtime.rs",
            "fixture": "verification/unwind_replay_oracle.rs",
            "start": "static NATIVE_LOWERING_STATE",
            "end": "static NATIVE_LOWERING_STATE",
            "differences": (
                ("NativeLoweringState { suppress_unrepresentable: Cell::new(false), "
                 "suppress_output: Cell::new(false), summary_probe_builds: Cell::new(0), "
                 "next_preflight_frame: Cell::new(0), }",
                 "NativeLoweringState { suppress_unrepresentable: Cell::new(false), "
                 "suppress_output: Cell::new(false), }"),
            ),
        },
        {
            "production": "crates/chelis-ir/src/lower.rs",
            "fixture": "verification/unwind_replay_oracle.rs",
            "start": "fn fatal(mut self) -> Self",
            "end": "fn fatal(mut self) -> Self",
            "differences": (),
        },
    ],
}

RUST_TOKEN = re.compile(
    r'"(?:[^"\\]|\\.)*"'  # string literal
    r"|'(?:[^'\\]|\\.)'"  # char literal
    r"|[A-Za-z_][A-Za-z0-9_]*|\d[\w.]*|::|->|=>|&&|\|\||\S")
VISIBILITY = re.compile(r"\bpub(\((crate|super|self)\))?\s+")


def rust_tokens(text: str) -> list[str]:
    """Tokens of Rust source without comments, blank space, trailing commas and visibility."""
    kept, index = [], 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index)
            index = len(text) if end < 0 else end
        elif text.startswith("/*", index):
            index = text.index("*/", index) + 2
        else:
            # A string or char literal is kept whole, so `//` or `"` inside it is not syntax.
            match = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)\'|.', re.S).match(text, index)
            kept.append(match.group(0))
            index = match.end()
    tokens = RUST_TOKEN.findall(VISIBILITY.sub("", "".join(kept)))
    return [token for token, following in zip(tokens, [*tokens[1:], ""])
            if not (token == "," and following in (")", "]", "}"))]


def copy_region(text: str, start: str, end: str, path: str) -> str:
    """From `start` to the end of the block of the item that starts at `end`."""
    first = text.find(start)
    last = text.find(end, first) if first >= 0 else -1
    if first < 0 or last < 0:
        raise RuntimeError(f"{path}: no copied region from {start!r} to {end!r}")
    depth = 0
    for index in range(text.index("{", last), len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[first:index + 1]
    raise RuntimeError(f"{path}: unbalanced block at {end!r}")


def check_fixture_copies(name: str, root: Path) -> None:
    for copy in FIXTURE_COPIES.get(name, ()):
        regions = {
            side: rust_tokens(copy_region(
                (root / copy[side]).read_text(), copy["start"], copy["end"], copy[side]))
            for side in ("production", "fixture")
        }
        expected = regions["production"]
        for production, fixture in copy["differences"]:
            old, new = rust_tokens(production), rust_tokens(fixture)
            found = [index for index in range(len(expected) - len(old) + 1)
                     if expected[index:index + len(old)] == old]
            if len(found) != 1:
                raise RuntimeError(
                    f"{name}: recorded difference {production!r} occurs {len(found)} times in "
                    f"{copy['production']}")
            expected = expected[:found[0]] + new + expected[found[0] + len(old):]
        if expected != regions["fixture"]:
            lines = {side: " ".join(tokens).replace("; ", ";\n").replace("{ ", "{\n").replace(
                "} ", "}\n").splitlines(keepends=True)
                for side, tokens in (("production", expected), ("fixture", regions["fixture"]))}
            sys.stderr.writelines(unified_diff(
                lines["production"], lines["fixture"],
                fromfile=f"{copy['production']} (recorded differences applied)",
                tofile=copy["fixture"]))
            raise RuntimeError(
                f"{name}: {copy['fixture']} no longer repeats {copy['production']} "
                f"(from {copy['start']!r} to the end of {copy['end']!r})")
        print(f"{name}: {copy['fixture']} repeats {copy['production']} "
              f"with {len(copy['differences'])} recorded differences")


def native_crate_arguments(name: str, directory: Path, root: Path) -> tuple[str, list[str]]:
    """The compiler and `--extern` arguments of a fixture's native test (MODEL_NATIVE_CRATES)."""
    crates = MODEL_NATIVE_CRATES.get(name, ())
    if not crates:
        return "model-rustc", []
    target = directory / "native-crates"
    output = run([
        "cargo", "build", "--offline", "--locked", "--quiet", "--message-format=json",
        "--manifest-path", str(root / "Cargo.toml"), "--target-dir", str(target),
        *[argument for crate in crates for argument in ("-p", crate)],
    ], root, capture=True)
    libraries = {}
    for line in output.splitlines():
        message = json.loads(line)
        if message.get("reason") == "compiler-artifact" and message["target"]["name"] in crates:
            libraries[message["target"]["name"]] = next(
                path for path in message["filenames"] if path.endswith(".rlib"))
    missing = sorted(set(crates) - set(libraries))
    if missing:
        raise RuntimeError(f"{name}: cargo built no library for {missing}")
    return "rustc", [
        "-L", f"dependency={target / 'debug' / 'deps'}",
        *[f"--extern={crate}={library}" for crate, library in sorted(libraries.items())],
    ]


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


def check_unwind_replay(lean_library: Path, root: Path) -> None:
    """The native unwind segments against their Lean replays (Runtime/Unwind.lean)."""
    check_fixture_copies("unwind_replay", root)
    with tempfile.TemporaryDirectory(prefix="chelis-unwind-replay-") as directory:
        output = Path(directory)
        native = output / "native"
        run([
            "model-rustc", "--edition=2021", str(root / "verification" / "unwind_replay_oracle.rs"),
            "-o", str(native),
        ], root)
        expected = run([str(native)], root, capture=True)
        bridge = root / "verification" / "lean"
        olean = output / "ChelisBridge" / "Runtime" / "Unwind.olean"
        olean.parent.mkdir(parents=True)
        lean_path = os.pathsep.join(filter(None, [str(output), os.environ.get("LEAN_PATH")]))
        lean = ["env", f"LEAN_PATH={lean_path}", "lake", "env", "lean"]
        run([
            *lean, f"--root={bridge}", "-o", str(olean),
            str(bridge / "ChelisBridge" / "Runtime" / "Unwind.lean"),
        ], lean_library)
        actual = run([
            *lean, str(root / "verification" / "UnwindReplayChecks.lean"),
        ], lean_library, capture=True)
    if actual != expected:
        sys.stderr.writelines(unified_diff(
            expected.splitlines(keepends=True), actual.splitlines(keepends=True),
            fromfile="native Rust", tofile="Lean replay",
        ))
        raise RuntimeError("The Lean unwind replays disagree with the native unwind oracle")
    print(f"unwind_replay: {len(expected.splitlines())} native/Lean segments agree")


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
    check_fixture_copies(name, root)
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
                "take::Take", "rev::Rev", "skip_while::SkipWhile",
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
        "--include",
        "alloc::collections::btree::map::entry::{alloc::collections::btree::map::entry::Entry}",
        *[argument for entry in roots for argument in ("--start-from", entry)],
        *[argument for entry in shell_patterns(name) for argument in ("--opaque-hard", entry)],
        *[argument for entry in MODEL_DEPENDENCIES.get(name, ())
          for argument in ("--opaque-hard", entry)],
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
    check_fixture_copies(name, root)
    with tempfile.TemporaryDirectory(prefix=f"chelis-{name}-") as directory:
        output = Path(directory)
        native = output / "native"
        compiler, externs = native_crate_arguments(name, output, root)
        run([
            compiler, edition, *crate_arguments(name), *externs, "--test", str(source),
            "-o", str(native),
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
        "--model",
        choices=(*MODELS, "float_model", "unwind_replay", *CRATE_MODELS, "collection_contracts"),
        action="append",
        help="Check this model (repeatable); by default check every model",
    )
    args = parser.parse_args()
    lean_library = args.aeneas_lean_dir.resolve()
    root = Path(__file__).resolve().parent.parent
    for name in args.model or [
        "float_model", "unwind_replay", *CRATE_MODELS, *MODELS, "collection_contracts",
    ]:
        if name == "float_model":
            check_float_model(lean_library, root)
        elif name == "unwind_replay":
            check_unwind_replay(lean_library, root)
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
