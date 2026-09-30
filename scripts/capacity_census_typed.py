#!/usr/bin/env python3
"""Typed rustdoc-JSON enumerators for chelis#729's C6 entry legs.

The wire leg executes the complete covered-root graph, carrier admission and
codec proof before issuing final authority. The binding leg joins live registered
PyO3 functions to their Rust signatures. A supplied artifact can be used only
for binding infrastructure; it never grants wire authority.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

# Nested Cargo work must avoid nextest's ambient target. The wire verifier
# cleans and seals artifacts in its target; the concurrent binding worker has
# a sibling target for every Cargo, rustdoc and compiled-probe invocation.
WIRE_RUSTDOC_TARGET_DIR = Path("target/agents/729-capacity-rustdoc")
BINDING_RUSTDOC_TARGET_DIR = Path("target/agents/729-capacity-binding-proof")

NUMERIC_PRIMITIVES = {
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "isize",
    "u8",
    "u16",
    "u32",
    "u64",
    "u128",
    "usize",
}
FLOAT_PRIMITIVES = {"f32", "f64"}
INTEGER_PRIMITIVES = NUMERIC_PRIMITIVES - FLOAT_PRIMITIVES


class CensusError(RuntimeError):
    """A typed artifact could not satisfy the census contract."""


def _index(document: dict[str, Any], item_id: int | str) -> dict[str, Any]:
    return document["index"][str(item_id)]


def _angle_args(args: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not args:
        return []
    return args.get("angle_bracketed", {}).get("args", [])


def canonical_type(ty: Any) -> str:
    """Render one rustdoc JSON type into a stable, reviewable identity."""
    if ty is None:
        return "()"
    if not isinstance(ty, dict):
        return json.dumps(ty, sort_keys=True, separators=(",", ":"))
    if "primitive" in ty:
        return str(ty["primitive"])
    if "generic" in ty:
        return str(ty["generic"])
    if "resolved_path" in ty:
        path = ty["resolved_path"]
        rendered_args: list[str] = []
        for arg in _angle_args(path.get("args")):
            if "type" in arg:
                rendered_args.append(canonical_type(arg["type"]))
            elif "lifetime" in arg:
                rendered_args.append(str(arg["lifetime"]))
            elif "const" in arg:
                rendered_args.append(str(arg["const"]))
            else:
                rendered_args.append(
                    json.dumps(arg, sort_keys=True, separators=(",", ":"))
                )
        suffix = f"<{', '.join(rendered_args)}>" if rendered_args else ""
        return f"{path['path']}{suffix}"
    if "borrowed_ref" in ty:
        reference = ty["borrowed_ref"]
        lifetime = f"{reference['lifetime']} " if reference.get("lifetime") else ""
        mutable = "mut " if reference.get("is_mutable") else ""
        return f"&{lifetime}{mutable}{canonical_type(reference['type'])}"
    if "raw_pointer" in ty:
        pointer = ty["raw_pointer"]
        mutability = "mut" if pointer.get("is_mutable") else "const"
        return f"*{mutability} {canonical_type(pointer['type'])}"
    if "slice" in ty:
        return f"[{canonical_type(ty['slice'])}]"
    if "array" in ty:
        array = ty["array"]
        return f"[{canonical_type(array['type'])}; {array['len']}]"
    if "tuple" in ty:
        members = [canonical_type(member) for member in ty["tuple"]]
        suffix = "," if len(members) == 1 else ""
        return f"({', '.join(members)}{suffix})"
    if "function_pointer" in ty:
        pointer = ty["function_pointer"]
        declaration = pointer["sig"]
        inputs = ", ".join(
            canonical_type(input_ty) for _, input_ty in declaration["inputs"]
        )
        return f"fn({inputs}) -> {canonical_type(declaration.get('output'))}"
    return json.dumps(ty, sort_keys=True, separators=(",", ":"))


def numeric_primitives(ty: Any) -> set[str]:
    """Return numeric Rust primitives appearing anywhere in a typed shape."""
    found: set[str] = set()
    if isinstance(ty, dict):
        primitive = ty.get("primitive")
        if primitive in NUMERIC_PRIMITIVES:
            found.add(str(primitive))
        for value in ty.values():
            found.update(numeric_primitives(value))
    elif isinstance(ty, list):
        for value in ty:
            found.update(numeric_primitives(value))
    return found


def binding_rows(
    document: dict[str, Any], registered_names: Iterable[str], registered_methods: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """Join live registered PyO3 names to typed rustdoc function signatures."""
    registered = sorted(set(registered_names))
    paths_by_name: dict[str, str] = {}
    for item_id, path_record in document.get("paths", {}).items():
        path = path_record.get("path", [])
        if len(path) == 2 and path[0] == "chelis_python" and path_record.get("kind") == "function":
            paths_by_name[path[1]] = item_id

    rows: list[dict[str, Any]] = []
    for name in registered:
        item_id = paths_by_name.get(name)
        if item_id is None:
            raise CensusError(
                f"registered PyO3 function `{name}` has no top-level rustdoc JSON signature; "
                "registered callables may not bypass the signature-derived census"
            )
        item = _index(document, item_id)
        signature = item.get("inner", {}).get("function", {}).get("sig")
        if signature is None:
            raise CensusError(f"rustdoc item for registered PyO3 function `{name}` is not a function")
        rendered_inputs: list[str] = []
        flags: set[str] = set()
        for input_name, input_ty in signature.get("inputs", []):
            rendered_inputs.append(f"{input_name}: {canonical_type(input_ty)}")
            primitives = numeric_primitives(input_ty)
            if primitives & FLOAT_PRIMITIVES:
                flags.add("float-carrier")
            if "dtype" in input_name.lower() and primitives & INTEGER_PRIMITIVES:
                flags.add("raw-dtype-int")
            elif primitives:
                flags.add("numeric-param")
        output = canonical_type(signature.get("output"))
        rows.append(
            {
                "kind": "binding-pyfunction",
                "id": f"chelis_python::{name}({', '.join(rendered_inputs)}) -> {output}",
                "flags": sorted(flags),
            }
        )

    method_items: dict[tuple[str, str], dict[str, Any]] = {}
    for item_id, path_record in document.get("paths", {}).items():
        path = path_record.get("path", [])
        if len(path) != 2 or path[0] != "chelis_python" or path_record.get("kind") != "struct":
            continue
        owner = path[1]
        item = _index(document, item_id)
        for impl_id in item.get("inner", {}).get("struct", {}).get("impls", []):
            impl = _index(document, impl_id).get("inner", {}).get("impl", {})
            if impl.get("trait") is not None:
                continue
            for method_id in impl.get("items", []):
                method = _index(document, method_id)
                if "function" not in method.get("inner", {}):
                    continue
                method_items[(owner, str(method.get("name")))] = method

    for registered_method in sorted(set(registered_methods)):
        try:
            owner, name = registered_method.split("::", 1)
        except ValueError as error:
            raise CensusError(
                f"registered PyO3 method identity `{registered_method}` is not Class::method"
            ) from error
        rust_name = "new" if name == "__new__" else name
        rust_owner = owner
        item = method_items.get((rust_owner, rust_name))
        if item is None:
            rust_owner = f"Native{owner}"
            item = method_items.get((rust_owner, rust_name))
        if item is None and name == "__new__":
            # PyO3 exposes a non-instantiable default descriptor for a pyclass
            # without a `#[new]` method. A real `#[new] fn new(...)` is found
            # above and remains part of the signature census.
            continue
        if item is None:
            raise CensusError(
                f"registered PyO3 method `{registered_method}` has no rustdoc JSON signature; "
                "registered callables may not bypass the signature-derived census"
            )
        signature = item["inner"]["function"]["sig"]
        rendered_inputs = []
        flags: set[str] = set()
        for input_name, input_ty in signature.get("inputs", []):
            rendered_inputs.append(f"{input_name}: {canonical_type(input_ty)}")
            primitives = numeric_primitives(input_ty)
            if primitives & FLOAT_PRIMITIVES:
                flags.add("float-carrier")
            if "dtype" in input_name.lower() and primitives & INTEGER_PRIMITIVES:
                flags.add("raw-dtype-int")
            elif primitives:
                flags.add("numeric-param")
        rows.append(
            {
                "kind": "binding-pymethod",
                "id": (
                    f"chelis_python::{owner}::{name}({', '.join(rendered_inputs)}) -> "
                    f"{canonical_type(signature.get('output'))}"
                ),
                "flags": sorted(flags),
            }
        )
    return sorted(rows, key=lambda row: (row["kind"], row["id"]))


def generate_rustdoc_json(
    *, root: Path, package: str, crate_name: str, target_dir: Path
) -> dict[str, Any]:
    """Build private-item rustdoc JSON in an isolated target directory."""
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["RUSTC_BOOTSTRAP"] = "1"
    result = subprocess.run(
        [
            "cargo",
            "rustdoc",
            "-p",
            package,
            "--lib",
            "--output-format",
            "json",
            "-Z",
            "unstable-options",
            "--",
            "--document-private-items",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CensusError(
            f"rustdoc JSON generation failed for {package}:\n{result.stdout}\n{result.stderr}"
        )
    path = target_dir / "doc" / f"{crate_name}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CensusError(f"cannot read generated rustdoc JSON {path}: {error}") from error


def build_parser() -> argparse.ArgumentParser:
    # `allow_abbrev=False` so `--target-dir` has exactly one spelling. With
    # argparse's default, `--t` is an unambiguous prefix and silently splits
    # the legs back onto separate directories in a form the call-site drift
    # guard in test_capacity_census_typed.py cannot see.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("mode", choices=("wire", "bindings", "bindings-discovery"))
    # No checked-in caller supplies a target. An ad-hoc binding target stays
    # separate from the wire target, and neither target supplies authority.
    parser.add_argument("--target-dir", type=Path, default=None)
    parser.add_argument("--registered", action="append", default=[])
    parser.add_argument("--registered-method", action="append", default=[])
    parser.add_argument("--registered-class", action="append", default=[])
    parser.add_argument("--registered-provenance", type=Path)
    parser.add_argument("--rustdoc-json", type=Path)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def resolve_target_dir(root: Path, requested: Path | None, mode: str) -> Path:
    """Select a worktree-local nested Cargo target for one census mode."""
    if requested is None:
        return root / (WIRE_RUSTDOC_TARGET_DIR if mode == "wire" else BINDING_RUSTDOC_TARGET_DIR)
    return requested if requested.is_absolute() else root / requested


def _binding_worker(root: Path, target: Path):
    from capacity_census_compiler_json import collect_compiler_json_bindings
    from capacity_census_native_authority import verify_native_bindings
    from ci_timing import span

    # These phases share binding Cargo artifacts and remain serial with each
    # other. Neither phase reads wire classifications.
    with span("binding.compiler_json", "census-stage"):
        work = collect_compiler_json_bindings(root, target)
    with span("binding.native", "census-stage"):
        native = verify_native_bindings(root, target)
    return work, native


def verify_binding_proofs(root: Path, wire_target: Path, binding_target: Path):
    """Join live wire and binding execution before issuing compiler JSON authority."""
    from capacity_census_compiler_json import _CompilerJsonWork, finalize_compiler_json_bindings
    from capacity_census_native_authority import VerifiedNativeBindings
    from capacity_census_wire_adapters import source_identity
    from capacity_census_wire_verifier import VerifiedWireCensus, verify_wire_census

    root = root.resolve()
    wire_target, binding_target = wire_target.resolve(), binding_target.resolve()
    target_root = root / "target"
    if (
        not wire_target.is_relative_to(target_root)
        or not binding_target.is_relative_to(target_root)
        or wire_target.is_relative_to(binding_target)
        or binding_target.is_relative_to(wire_target)
    ):
        raise CensusError("wire and binding proof require disjoint worktree-local targets")
    before = source_identity(root)
    wire_error = binding_error = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="binding-proof") as pool:
        binding = pool.submit(_binding_worker, root, binding_target)
        try:
            wire = verify_wire_census(root, wire_target)
        except Exception as error:
            wire_error = error
        try:
            work, native = binding.result()
        except Exception as error:
            binding_error = error
    if wire_error is not None and binding_error is not None:
        raise CensusError(
            f"wire verification failed: {wire_error}; binding verification failed: {binding_error}"
        ) from wire_error
    if wire_error is not None:
        raise wire_error
    if binding_error is not None:
        raise binding_error
    if type(wire) is not VerifiedWireCensus:
        raise CensusError("binding join requires live wire verification")
    if type(native) is not VerifiedNativeBindings:
        raise CensusError("binding join requires live native verification")
    if type(work) is not _CompilerJsonWork:
        raise CensusError("binding join requires current compiler JSON collection")
    if not (
        wire.source_sha256 == work.source_sha256 == native.source_sha256
        == before == source_identity(root)
    ):
        raise CensusError("source changed between binding and wire proof")
    wire.validate()
    native.validate()
    compiler_json = finalize_compiler_json_bindings(root, binding_target, work, wire)
    compiler_json.validate()
    return compiler_json, native


def main() -> int:
    from capacity_census_graph import GraphError

    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    target_dir = resolve_target_dir(root, args.target_dir, args.mode)
    try:
        if args.mode == "wire":
            if args.rustdoc_json or args.registered or args.registered_method or args.registered_class or args.registered_provenance:
                raise CensusError("wire authority requires actual artifact and codec execution")
            from capacity_census_wire_verifier import verify_wire_census

            report = verify_wire_census(root, target_dir).execution_report()
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        compiler_json = None
        native = None
        if args.mode == "bindings-discovery":
            if args.rustdoc_json:
                raise CensusError("binding authority requires current compiled and wire execution")
            if not args.registered_provenance:
                raise CensusError("binding discovery requires registration provenance")
            provenance = json.loads(args.registered_provenance.read_text())
            # A missing registration is already a rejection. This is only an
            # early failure check; live descriptor/source/typed ownership still
            # runs below and no metadata can issue a transport witness.
            declared = {row.get("python_name") for row in provenance.get("registrations", [])
                        if row.get("owner") is None}
            for name in args.registered:
                if name not in declared:
                    raise CensusError(f"{name}: missing registration provenance")
            compiler_json, native = verify_binding_proofs(
                root, resolve_target_dir(root, None, "wire"), target_dir
            )
        if compiler_json is not None:
            document = compiler_json.graph.documents["chelis_python"]
        elif args.rustdoc_json:
            document = json.loads(args.rustdoc_json.read_text())
        else:
            document = generate_rustdoc_json(
                root=root,
                package="chelis-python",
                crate_name="chelis_python",
                target_dir=target_dir,
            )
        if args.mode == "bindings-discovery":
            from capacity_census_bindings import discover_bindings

            classes = {}
            for value in args.registered_class:
                name, separator, identity = value.partition("=")
                if not separator or not name or not identity or name in classes:
                    raise CensusError("invalid or duplicate registered class identity")
                classes[name] = identity
            rows = discover_bindings(
                compiler_json.graph.documents.values(), args.registered,
                args.registered_method, classes,
                provenance=provenance,
                compiler_json=compiler_json,
                native=native,
            )
            legacy = {row["id"]: row["flags"] for row in binding_rows(
                document, args.registered, args.registered_method
            )}
            for row in rows:
                row["legacy_flags"] = legacy.get(row["id"], row["flags"])
            receipt = root / "target/capacity-census-compiler-json-execution.json"
            report = {
                "version": 2,
                "rows": rows,
                "compiler_json": compiler_json.execution_report(),
                "native": native.execution_report(),
            }
            receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        else:
            rows = binding_rows(document, args.registered, args.registered_method)
    except (CensusError, GraphError, OSError, json.JSONDecodeError) as error:
        print(f"capacity census typed enumerator failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    from ci_timing import subprocesses
    with subprocesses():
        sys.exit(main())
