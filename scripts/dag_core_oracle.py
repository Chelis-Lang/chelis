#!/usr/bin/env python3
"""DAG acceptance with real target graphs and host execution."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import tomllib
from typing import Iterator

import no_std_core_support as support
from no_std_core_support import ROOT, TARGET, OracleFailure, compile_rejected, consumer, portable, run
from runtime_bundle_oracle import (
    RECEIPT_FILE_NAME,
    read_receipt_document,
    read_staging_receipt,
)


CRATE = "chelis-dag-core"
TARGET_CHELIS = frozenset({CRATE, "chelis-numeric-core", "chelis-crmath", "chelis-vocab", "chelis-unord", "chelis-pred", "chelis-no-std-consumer"})
EVIDENCE = frozenset({"thumb-default", "thumb-no-default", "standalone-alloc", "allocator-eval-ad",
    "ownership-fusion-eval", "graph-contracts", "checked-names", "wire-baseline", "transform-baseline",
    "native-c", "c-hip-metal-generation", "admission-diagnostics", "span-comments",
    "dependency-features", "negative-controls", "scale-allocations", "current-host-suites", "python-controls", "source-owner-controls", "moved-host-tests"})
BASELINE_REVISION = "3ca99faa7a46158eb6a0d90c2c812a7e3a5f44c1"
PRODUCTION_BASELINE_REVISION = "137ba12e2cf9bcdcac4b4d3ed3662832a93e44a4"
FORBIDDEN = frozenset({
    "chelis-types", "chelis-type-core", "chelis-deep", "chelis-surf", "chelis-ir",
    "chelis-effects", "chelis-pipeline-core", "chelis-compiler-api",
    "chelis-backend-c", "chelis-backend-hip", "chelis-backend-metal",
    "chelis-runtime", "chelis-native", "stacker", "chelis-abi",
    "pyo3", "tokio", "reqwest", "ureq", "getrandom", "rand", "std",
})
SOURCE_MODULES = (
    "dag", "load_store_name", "analysis", "axis_sources", "verify", "tier2",
    "tier2_ordered", "grad", "fuse", "optimize", "span_merge",
)

PORTABLE_SOURCE = r'''
#![no_std]
extern crate alloc;
use alloc::vec;
use chelis_dag_core::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_dag_core::{analysis, fuse, grad, ownership, tier2, verify};
pub fn square_loss() -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    dag.declare("fixture");
    let mut ty = TensorType::scalar_f32();
    ty.dims = vec![DimInfo::Lit(4)];
    let x = dag.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::Load { name: "x".into() }, vec![], ty.clone(), Some("x".into()));
    let square = dag.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::Mul, vec![x, x], ty.clone(), Some("square".into()));
    let loss = dag.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::sum_default(0, ty.precision).unwrap(), vec![square],
        TensorType::scalar_f32(), Some("loss".into()));
    dag.add_root(loss);
    (dag, x, loss)
}
pub fn portable_passes() -> Dag {
    let (dag, x, loss) = square_loss();
    assert!(verify::verify(&dag).is_empty());
    let gradient = grad::grad_dag_checked(&dag, loss, &[x]).unwrap();
    let fused = fuse::fuse_with_remap(&gradient.dag);
    let normalized = ownership::normalize_ownership(&fused.dag);
    assert!(verify::verify(&normalized.dag).is_empty());
    let cost = analysis::analyze_copy_costs(&normalized.dag,
        &[("gradient".into(), normalized.dag.roots()[0])]);
    assert_eq!(cost.total_copy_count, 0);
    let mut relu = Dag::new();
    relu.declare("fixture");
    let x = relu.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::Load { name: "x".into() }, vec![], TensorType::scalar_f32(), None);
    let out = tier2::lower_relu(chelis_dag_core::dag::Owner::unconditional(chelis_dag_core::dag::DeclId(0)), &mut relu, x, &TensorType::scalar_f32(), Some("relu"));
    relu.add_root(out);
    assert!(verify::verify(&relu).is_empty());
    normalized.dag
}
'''

HOST_SOURCE = PORTABLE_SOURCE + r'''
extern crate std;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
const EXPECT_SYMBOLIC_PRODUCT_PANIC: bool = false;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
struct CountedAllocator;
unsafe impl GlobalAlloc for CountedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountedAllocator = CountedAllocator;
fn value(dag: &Dag, root: NodeId) -> alloc::vec::Vec<f64> {
    let values = eval_tensor_roots_with_strict(dag, &[root], |name| {
        assert_eq!(name, "x");
        Some(TensorValue::from_vec(vec![4], vec![1.0, -2.0, 3.5, 0.25]))
    }).unwrap();
    values[&root].to_f64_lossy_vec()
}
fn host_checks() {
    let _ = portable_passes();
    let (dag, x, loss) = square_loss();
    assert_eq!(value(&dag, loss), vec![17.3125]);
    let gradient = grad::grad_dag_checked(&dag, loss, &[x]).unwrap();
    let root = gradient.grad_nodes[&x];
    assert_eq!(value(&gradient.dag, root), vec![2.0, -4.0, 7.0, 0.5]);
    let fused = fuse::fuse_with_remap(&gradient.dag);
    assert_eq!(value(&fused.dag, fused.old_to_new[&root]), value(&gradient.dag, root));
    let normalized = ownership::normalize_ownership(&fused.dag);
    let mapped = normalized.old_to_new[&fused.old_to_new[&root]];
    assert_eq!(value(&normalized.dag, mapped), vec![2.0, -4.0, 7.0, 0.5]);
    let mut fanout = Dag::new();
    fanout.declare("fixture");
    let x = fanout.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::Load { name: "x".into() }, vec![],
        TensorType { dims: vec![DimInfo::Lit(4)], ..TensorType::scalar_f32() }, Some("x".into()));
    let ty = fanout.get(x).unwrap().output_type.clone();
    let a = fanout.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::Realize, vec![x], ty.clone(), Some("a".into()));
    let b = fanout.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::Realize, vec![x], ty, Some("b".into()));
    fanout.set_roots(vec![b, a]);
    let normalized = ownership::normalize_ownership(&fanout);
    for old in [b, a] {
        assert_eq!(value(&fanout, old), value(&normalized.dag, normalized.old_to_new[&old]));
    }
    assert_eq!(analysis::analyze_copy_costs_for_roots(&normalized.dag,
        &[("fanout".into(), normalized.dag.roots().to_vec())]).total_bytes_copied, Some(16));
    let mut trap = Dag::new();
    trap.declare("fixture");
    let integer_ty = TensorType { dims: vec![], precision: chelis_numeric_core::prim::Prim::Int8 };
    let x = trap.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::synth_const(integer_ty.precision, 127.0), vec![], integer_ty.clone(), None);
    let out = trap.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::Add, vec![x, x], integer_ty, Some("overflow".into()));
    trap.add_root(out);
    let before = eval_tensor_roots_with_strict(&trap, &[out], |_| None).unwrap_err();
    chelis_dag_core::optimize::constant_fold(&mut trap);
    assert!(matches!(trap.get(out).unwrap().op, RiscOp::Add));
    let after = eval_tensor_roots_with_strict(&trap, &[out], |_| None).unwrap_err();
    assert_eq!(before, after);
    assert_eq!(after, "numeric trap: overflow in add at i8");
    let mut symbolic = Dag::new();
    symbolic.declare("fixture");
    let input = symbolic.add_node(chelis_dag_core::dag::DeclId(0), 
        RiscOp::Load { name: "x".into() }, vec![],
        TensorType { dims: vec![DimInfo::Named("batch".into(), None)], ..TensorType::scalar_f32() },
        Some("input".into()),
    );
    let product = symbolic.add_node(chelis_dag_core::dag::DeclId(0), 
        RiscOp::ProdReduce { axis: 0 }, vec![input], TensorType::scalar_f32(), Some("product".into()),
    );
    symbolic.add_root(product);
    if EXPECT_SYMBOLIC_PRODUCT_PANIC {
        assert!(std::panic::catch_unwind(|| grad::grad_dag_checked(&symbolic, product, &[input])).is_err());
        assert!(std::panic::catch_unwind(|| grad::grad_dag(&symbolic, product, &[input])).is_err());
        std::println!("DAG_SYMBOLIC_BASELINE: checked and optional AD retain the original symbolic-product panic");
    } else {
        match grad::grad_dag_checked(&symbolic, product, &[input]) {
            Err(grad::AdError::NotSupported { op: "prod_reduce", reason: grad::AdRejectionReason::Other(message) }) => {
                assert!(message.contains("prod_reduce") && message.contains("batch"), "{message}");
            }
            _ => panic!("symbolic product must return a diagnostic"),
        }
        assert!(grad::grad_dag(&symbolic, product, &[input]).is_none());
        std::println!("DAG_SYMBOLIC: checked and optional AD reject a symbolic product without a panic");
    }
    for scale in [64_usize, 256, 1024] {
        let mut graph = Dag::new();
    graph.declare("fixture");
        let x = graph.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::Load { name: "x".into() }, vec![], TensorType::scalar_f32(), Some("x".into()));
        let mut root = x;
        for _ in 1..scale {
            root = graph.add_node(chelis_dag_core::dag::DeclId(0), RiscOp::Neg, vec![root], TensorType::scalar_f32(), Some("neg".into()));
        }
        graph.add_root(root);
        let allocations = ALLOCATIONS.load(Ordering::Relaxed);
        let bytes = BYTES.load(Ordering::Relaxed);
        let start = Instant::now();
        let backward = grad::grad_dag_checked(&graph, root, &[x]).unwrap();
        let ad_allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations;
        let ad_bytes = BYTES.load(Ordering::Relaxed) - bytes;
        let fused = fuse::fuse_with_remap(&backward.dag);
        let fusion_allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations - ad_allocations;
        let fusion_bytes = BYTES.load(Ordering::Relaxed) - bytes - ad_bytes;
        let normalized = ownership::normalize_ownership(&fused.dag);
        let elapsed = start.elapsed().as_nanos();
        let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocations;
        let bytes = BYTES.load(Ordering::Relaxed) - bytes;
        assert!(verify::verify(&normalized.dag).is_empty());
        assert!(backward.dag.len() <= 3 * scale);
        assert!(fused.dag.len() <= backward.dag.len());
        assert!(normalized.dag.len() <= 2 * fused.dag.len());
        assert!(allocations < 1024 * scale);
        assert!(bytes < 1024 * 1024 * scale);
        std::println!("DAG_COST {{\"nodes\":{},\"ad_nodes\":{},\"fused_nodes\":{},\"ownership_nodes\":{},\"allocations\":{},\"allocated_bytes\":{},\"nanoseconds\":{},\"stack_bytes\":8388608,\"ad_allocations\":{},\"ad_bytes\":{},\"fusion_allocations\":{},\"fusion_bytes\":{},\"ownership_allocations\":{},\"ownership_bytes\":{}}}",
            scale, backward.dag.len(), fused.dag.len(), normalized.dag.len(), allocations, bytes, elapsed, ad_allocations, ad_bytes, fusion_allocations, fusion_bytes,
            allocations - ad_allocations - fusion_allocations, bytes - ad_bytes - fusion_bytes);
    }
}
fn main() {
    std::thread::Builder::new().stack_size(8 * 1024 * 1024).spawn(host_checks).unwrap().join().unwrap();
}
'''


def inspect_target_graph(metadata: dict) -> set[str]:
    """Reject host dependencies and std features on normal target paths."""
    try:
        packages = {package["id"]: package for package in metadata["packages"]}
        nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
        pending = [metadata["resolve"]["root"]]
        visited: set[str] = set()
        names: set[str] = set()
        std_packages: set[str] = set()
        while pending:
            identity = pending.pop()
            if identity in visited:
                continue
            visited.add(identity)
            package = packages[identity]
            node = nodes[identity]
            name = package["name"]
            names.add(name)
            if "std" in node["features"]:
                std_packages.add(name)
            for dependency in node["deps"]:
                if not any(kind["kind"] is None for kind in dependency["dep_kinds"]):
                    continue
                target = packages[dependency["pkg"]]
                if any("proc-macro" in item["kind"] for item in target["targets"]):
                    continue
                pending.append(dependency["pkg"])
        forbidden = sorted(name for name in names if name in FORBIDDEN or (name.startswith("chelis-") and name not in TARGET_CHELIS))
        if forbidden:
            raise OracleFailure("forbidden target dependency: " + ", ".join(forbidden))
        if std_packages:
            raise OracleFailure("target std feature: " + ", ".join(name + "/std" for name in sorted(std_packages)))
        if CRATE not in names:
            raise OracleFailure("target graph omits the real DAG crate")
        return names
    except (KeyError, TypeError) as error:
        raise OracleFailure(f"incomplete target metadata: {error}") from error


@contextmanager
def isolated_graph(*, default_features: bool = False, extra: str = "") -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="chelis-dag-boundary-") as temporary:
        directory = Path(temporary)
        (directory / "src").mkdir()
        dependency = (
            f'chelis-dag-core = {{ path = {json.dumps(str(ROOT / "crates" / CRATE))}, '
            f'default-features = {str(default_features).lower()} }}\n'
        )
        (directory / "Cargo.toml").write_text(
            '[package]\nname = "dag-boundary-consumer"\nversion = "0.0.0"\nedition = "2024"\n'
            '[workspace]\n[dependencies]\n' + dependency + extra,
            encoding="utf-8",
        )
        (directory / "src/lib.rs").write_text(PORTABLE_SOURCE, encoding="utf-8")
        yield directory


def target_metadata(directory: Path) -> dict:
    result = run(["cargo", "metadata", "--format-version", "1", "--filter-platform", TARGET], cwd=directory)
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise OracleFailure(f"invalid target metadata: {error}") from error
    return metadata


def assert_boundary_rejected(extra: str, reason: str) -> None:
    with isolated_graph(extra=extra) as directory:
        metadata = target_metadata(directory)
        try:
            inspect_target_graph(metadata)
        except OracleFailure as error:
            if reason not in str(error):
                raise OracleFailure(f"boundary control failed for another reason: {error}") from error
        else:
            raise OracleFailure("boundary negative control unexpectedly passed")


def inspect_costs(stdout: str) -> list[dict]:
    records = []
    for line in stdout.splitlines():
        if line.startswith("DAG_COST "):
            try:
                records.append(json.loads(line.removeprefix("DAG_COST ")))
            except json.JSONDecodeError as error:
                raise OracleFailure(f"invalid graph cost record: {error}") from error
    if any(not isinstance(record, dict) for record in records):
        raise OracleFailure("invalid graph cost record")
    if [record.get("nodes") for record in records] != [64, 256, 1024]:
        raise OracleFailure("missing representative graph cost evidence")
    for record in records:
        if not isinstance(record, dict):
            raise OracleFailure("invalid graph cost measurement")
        if any(not isinstance(record.get(key), int) or record[key] <= 0 for key in (
            "nodes", "ad_nodes", "fused_nodes", "ownership_nodes", "allocations", "allocated_bytes", "nanoseconds", "stack_bytes", "ad_allocations", "ad_bytes", "fusion_allocations", "fusion_bytes", "ownership_allocations", "ownership_bytes",
        )):
            raise OracleFailure("invalid graph cost measurement")
    for previous, current in zip(records, records[1:]):
        if current["allocations"] > 6 * previous["allocations"] or current["allocated_bytes"] > 8 * previous["allocated_bytes"]:
            raise OracleFailure("superlinear transform allocation growth")
    return records


def workspace_environment(workspace: Path) -> dict[str, str]:
    environment = support._environment()
    leaf = "candidate" if workspace.resolve() == ROOT.resolve() else "baseline"
    environment["CARGO_TARGET_DIR"] = str(ROOT / "target" / "agents" / "dag-core-oracle" / leaf)
    environment["CHELIS_STYLE_GATE_DISABLE"] = "1"
    return environment


def baseline_clone(directory: Path, revision: str) -> Path:
    baseline = (directory / revision).resolve()
    run(["git", "clone", "--quiet", "--no-local", "--no-hardlinks", "--no-checkout", str(ROOT), str(baseline)])
    exists = support._execute(["git", "cat-file", "-e", revision + "^{commit}"], cwd=baseline)
    if exists.returncode:
        run(["git", "fetch", "--quiet", "--depth", "1", "https://github.com/Chelis-Lang/chelis.git", revision], cwd=baseline)
    run(["git", "checkout", "--quiet", "--detach", revision], cwd=baseline)
    if run(["git", "rev-parse", "HEAD"], cwd=baseline).stdout.strip() != revision:
        raise OracleFailure("immutable DAG baseline identity differs")
    return baseline


def snapshot_record(stdout: str, prefix: str) -> dict:
    records = [line[len(prefix) + 1:] for line in stdout.splitlines() if line.startswith(prefix + " ")]
    if len(records) != 1:
        raise OracleFailure(f"expected one {prefix} record")
    try:
        record = json.loads(records[0])
    except json.JSONDecodeError as error:
        raise OracleFailure(f"invalid {prefix} record: {error}") from error
    if not isinstance(record, dict) or not record:
        raise OracleFailure(f"empty {prefix} record")
    return record


def baseline_ownership(workspace: Path) -> str:
    """Expose exact private baseline algorithms inside the temporary consumer."""
    source = (workspace / "crates/chelis-ir/src/lower.rs").read_text(encoding="utf-8")
    start = source.index("fn insert_copy_nodes_for_consuming_fanout(")
    end = source.index("\n}\n", source.index("fn strip_drop_nodes(", start)) + 3
    body = source[start:end]
    fallback_start = source.index("    fn default_type() -> TensorType {")
    fallback_end = source.index("\n    }", fallback_start) + len("\n    }")
    return ("\nmod ownership {\nuse super::*;\nuse alloc::vec::Vec;\nuse chelis_unord::{UnordMap, UnordSet};\n"
        "use chelis_numeric_core::prim::Prim;\nstruct LowerCtx;\nimpl LowerCtx {\n"
        + source[fallback_start:fallback_end] + "\n}\n" + body + "\n"
        "pub struct OwnershipResult { pub dag: Dag, pub old_to_new: UnordMap<NodeId, NodeId> }\n"
        "pub fn normalize_ownership(dag: &Dag) -> OwnershipResult {\n"
        "let (dag, old_to_new) = insert_copy_nodes_for_consuming_fanout(dag);\n"
        "OwnershipResult { dag: insert_drop_nodes_for_unconsumed_values(dag), old_to_new }\n}\n}\n")


def fixture_dependencies(workspace: Path, *, host: bool = False) -> str:
    extra = ('bincode = "=1.3.3"\nserde_json = { version = "1", features = ["float_roundtrip"] }\n'
        f'chelis-numeric-core = {{ path = {json.dumps(str(workspace / "crates/chelis-numeric-core"))} }}\n')
    if host:
        extra += (f'chelis-ir = {{ path = {json.dumps(str(workspace / "crates/chelis-ir"))} }}\n'
            f'chelis-unord = {{ path = {json.dumps(str(workspace / "crates/chelis-unord"))} }}\n')
    return extra


def wire_snapshot(workspace: Path, *, baseline: bool) -> dict:
    source = (ROOT / "crates/chelis-dag-core/tests/fixtures/wire_snapshot.rs").read_text(encoding="utf-8")
    if baseline:
        source = source.replace("chelis_dag_core", "chelis_ir")
    result = consumer(workspace / "crates" / ("chelis-ir" if baseline else CRATE), source,
        default_features=baseline, dependencies=fixture_dependencies(workspace),
        env=workspace_environment(workspace))
    value = snapshot_record(result.stdout, "DAG_SNAPSHOT")
    fields = {"graph_json", "graph_binary", "op_json", "op_binary", "payload_json", "payload_binary",
        "diagnostics", "copy_costs", "dce", "dce_map", "fusion", "fusion_map", "ad", "gradients"}
    if set(value) != fields or any(not value[key] for key in fields):
        raise OracleFailure("wire and transform baseline evidence is incomplete")
    return value


def host_snapshot(workspace: Path, *, baseline: bool) -> list[dict]:
    source = HOST_SOURCE
    if baseline:
        source = source.replace("use chelis_dag_core::{Dag, DimInfo, NodeId, RiscOp, TensorType};",
            "use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};")
        source = source.replace("use chelis_dag_core::{analysis, fuse, grad, ownership, tier2, verify};",
            "use chelis_ir::{analysis, fuse, grad, tier2, verify};")
        source = source.replace("chelis_dag_core::", "chelis_ir::")
        source += baseline_ownership(workspace)
        source = source.replace("const EXPECT_SYMBOLIC_PRODUCT_PANIC: bool = false;",
            "const EXPECT_SYMBOLIC_PRODUCT_PANIC: bool = true;")
    dependencies = fixture_dependencies(workspace, host=True)
    if baseline:
        dependencies = "\n".join(line for line in dependencies.splitlines() if not line.startswith("chelis-ir =")) + "\n"
    result = consumer(workspace / "crates" / ("chelis-ir" if baseline else CRATE), source,
        default_features=baseline, dependencies=dependencies, env=workspace_environment(workspace))
    return inspect_costs(result.stdout)


def compare_costs(expected: list[dict], actual: list[dict]) -> None:
    if len(expected) != len(actual):
        raise OracleFailure("missing transform cost comparison")
    for old, new in zip(expected, actual, strict=True):
        for key in ("nodes", "ad_nodes", "fused_nodes", "ownership_nodes", "stack_bytes"):
            if old[key] != new[key]:
                raise OracleFailure(f"graph scale differs at {key}")
        for key in ("allocations", "allocated_bytes", "ad_allocations", "ad_bytes",
            "fusion_allocations", "fusion_bytes", "ownership_allocations", "ownership_bytes"):
            if new[key] > old[key]:
                raise OracleFailure(f"transform allocation regression at {new['nodes']} nodes: {key}")
    print(json.dumps({"baseline_revision": BASELINE_REVISION, "baseline_costs": expected,
        "candidate_costs": actual, "public_maps": "chelis_unord", "internal_lookup": "fixed-hash tables"}, sort_keys=True), flush=True)


def native_toolchain(workspace: Path) -> dict:
    source = r'''
fn main() {
    let t = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements { wants_openmp: true, needs_blas: false }).unwrap();
    println!("TOOLCHAIN {}", serde_json::json!({"compiler": t.compiler,
        "compile_flags": t.compile_flags, "link_flags": t.link_flags}));
}
'''
    result = consumer(workspace / "crates/chelis-backend-c", source, default_features=True,
        dependencies='serde_json = "1"\n', env=workspace_environment(workspace))
    return snapshot_record(result.stdout, "TOOLCHAIN")


def production_snapshot(workspace: Path, path: Path, directory: Path) -> dict:
    """Capture sources, headers, native output, and each backend admission lane."""
    environment = workspace_environment(workspace)
    native = native_toolchain(workspace)
    snapshots = {"toolchain": native}
    run(["cargo", "build", "--locked", "--quiet", "-p", "chelis-cli", "--bin", "chelis"], cwd=workspace, env=environment)
    binary = Path(environment["CARGO_TARGET_DIR"]) / "debug" / "chelis"
    for backend, stem, extension in (("c", "portable_graph", ".c"), ("hip", "portable_graph_hip", ".cpp"), ("metal", "portable_graph_metal", ".mm")):
        destination = directory / backend
        run([str(binary), "build", str(path), "--target", backend, "--emit-c",
            "--output", str(destination)], cwd=workspace, env=environment)
        generated = {output.name: output.read_text(encoding="utf-8")
            for output in destination.iterdir() if output.suffix in (extension, ".h")}
        if not generated.get(stem + extension) or not generated.get(stem + ".h"):
            raise OracleFailure(f"{backend} omits source or header evidence")
        # Archive hashes and build paths are not semantic output.
        snapshots[backend] = {"generated": generated}
        if backend == "c":
            native_binary = destination / "dag_native"
            receipt_path = destination / RECEIPT_FILE_NAME
            archive = destination / str(read_receipt_document(receipt_path)["archive"])
            receipt = read_staging_receipt(receipt_path, archive, headers_dir=destination)
            print(json.dumps({"staged_runtime": str(archive),
                "archive_sha256": receipt["archive_sha256"]}, sort_keys=True), flush=True)
            run([native["compiler"], "-O2", *native["compile_flags"], str(destination / (stem + extension)),
                str(archive), *native["link_flags"], "-o", str(native_binary)], cwd=destination, env=environment)
            executed = run([str(native_binary)], cwd=destination, env=environment)
            if "out = " not in executed.stdout:
                raise OracleFailure("native C fixture has no observable result")
            snapshots["native-c"] = {"stdout": executed.stdout, "stderr": executed.stderr}
    rejected_source = directory / "rejected_graph.ch"
    rejected_source.write_text("def f(x: tensor[4, f64]) -> tensor[4, f64] = neg(x)\n", encoding="utf-8")
    rejection = support._execute([str(binary), "build", str(rejected_source), "--target", "metal",
        "--emit-c", "--output", str(directory / "rejected-output")], cwd=workspace, env=environment)
    if rejection.returncode == 0 or not any(word in rejection.stderr for word in ("f64", "FP64")):
        raise OracleFailure("Metal f64 control fails for another reason")
    snapshots["metal-rejection"] = {"status": rejection.returncode, "stdout": rejection.stdout, "stderr": rejection.stderr}
    fenced = directory / "transcendental_graph.ch"
    fenced.write_text("def f(x: tensor[4, f32]) -> tensor[4, f32] = exp(x)\n", encoding="utf-8")
    for backend in ("c", "hip", "metal"):
        admission = support._execute([str(binary), "build", str(fenced), "--target", backend,
            "--emit-c", "--output", str(directory / (backend + "-admission"))], cwd=workspace, env=environment)
        if backend == "c":
            if admission.returncode:
                raise OracleFailure("C rejects its supported transcendental fixture")
        elif admission.returncode == 0 or "[05-OP-46]" not in admission.stderr or "exp" not in admission.stderr:
            raise OracleFailure(f"{backend} transcendental fence fails for another reason")
        snapshots[backend + "-admission"] = {"status": admission.returncode}
        if admission.returncode:
            snapshots[backend + "-admission"].update(
                stdout=admission.stdout, stderr=admission.stderr)
    return snapshots


def immutable_parity() -> None:
    source = ("def f(x: tensor[4, f32]) -> tensor[4, f32] = neg(x)\n"
        "out = f(to_tensor([1.0, -2.0, 3.5, 0.25], f32))\n")
    with tempfile.TemporaryDirectory(prefix="chelis-dag-production-") as temporary:
        directory = Path(temporary).resolve()
        baseline = baseline_clone(directory, BASELINE_REVISION)
        if wire_snapshot(baseline, baseline=True) != wire_snapshot(ROOT, baseline=False):
            raise OracleFailure("current-source wire or transform snapshot differs")
        compare_costs(host_snapshot(baseline, baseline=True), host_snapshot(ROOT, baseline=False))
        path = directory / "portable_graph.ch"
        path.write_text(source, encoding="utf-8")
        outputs = directory / "output"
        outputs.mkdir()
        historical = production_snapshot(baseline, path, outputs)
        shutil.rmtree(outputs)
        outputs.mkdir()
        production_baseline = baseline_clone(directory, PRODUCTION_BASELINE_REVISION)
        expected = production_snapshot(production_baseline, path, outputs)
        shutil.rmtree(outputs)
        outputs.mkdir()
        actual = production_snapshot(ROOT, path, outputs)
        generated_backends = {"c", "hip", "metal"}
        historical_behavior = {key: value for key, value in historical.items() if key not in generated_backends}
        actual_behavior = {key: value for key, value in actual.items() if key not in generated_backends}
        if actual_behavior != historical_behavior:
            raise OracleFailure("historical production behavior differs")
        if actual != expected:
            differing = sorted(key for key in expected if actual.get(key) != expected[key])
            raise OracleFailure("production baseline mismatch: " + ", ".join(differing))
        preserved_suites(baseline)
        print(f"DAG graph and behavior baseline: {BASELINE_REVISION}", flush=True)
        print(f"DAG generated-source baseline: {PRODUCTION_BASELINE_REVISION}", flush=True)


SUITES = (
    ("chelis-ir", "fusion_correctness"), ("chelis-ir", "fusion_adversarial"),
    ("chelis-ir", "per_pass_span_propagation"), ("chelis-ir", "issue_318_grad_expand_shape_derived"),
    ("chelis-ir", "issue_291_grad_shrink_stride"), ("chelis-ir", "dim_expr_evaluation"),
    ("chelis-ir", "runtime_extent_slice_b_sources"), ("chelis-ir", "runtime_extent_slice_b_classes"),
    ("chelis-ir", "ownership_lowering"), ("chelis-ir", "implicit_copy_fanout_v3"),
    ("chelis-ir", "issue_1948_same_shape_result_claim_sources"), ("chelis-ir", "issue_2110_local_ascription_claim"),
    ("chelis-ir", "key_operation_activations"), ("chelis-ir", "key_operations_ir"),
    ("chelis-ir", "issue_3997_extrema_result_types"), ("chelis-ir", "issue_1125_host_classifier_parity"),
    ("chelis-ir", "activation_surface_parity"), ("chelis-ir", "issue_1313_relu_adjoint"),
    ("chelis-compiler-api", "pipeline_contract"), ("chelis-compiler-api", "invariant_decode"),
    ("chelis-compiler-api", "diagnostic_kind_pipeline"), ("chelis-compiler-api", "wire_dag_vocabulary"),
    ("chelis-compiler-api", "wire_dag_v7_runtime_extents"), ("chelis-compiler-api", "rule_d_entered_lanes"),
    ("chelis-compiler-api", "local_ascription_activation"), ("chelis-compiler-api", "runtime_extent_authored_claims"),
    ("chelis-cli", "issue_3995_classification_producers"), ("chelis-cli", "issue_3997_rank_row_extrema"),
    ("chelis-backend-c", "float_classification"), ("chelis-backend-c", "fused_in_place_exec"),
    ("chelis-backend-c", "ad_contraction_storage"),
    ("chelis-backend-hip", "issue_2957_device_transcendental_fence"),
    ("chelis-backend-metal", "issue_2957_device_transcendental_fence"),
    ("chelis-backend-hip", "codegen_adversarial"), ("chelis-backend-metal", "codegen_adversarial"),
    *((f"chelis-backend-{backend}", "span_comments") for backend in ("c", "hip", "metal")),
)


def preserved_suites(baseline: Path) -> None:
    """Require every selected immutable-source test, not just a nonempty suite."""
    for package, binary in SUITES:
        arguments_by_workspace = {}
        for workspace in (baseline, ROOT):
            arguments = ["--locked", "-p", package, "--test", binary]
            manifest = tomllib.loads((workspace / "crates" / package / "Cargo.toml").read_text(encoding="utf-8"))
            features = next((target.get("required-features", ()) for target in manifest.get("test", ())
                if target["name"] == binary), ())
            if features:
                arguments.extend(("--features", ",".join(features)))
            arguments_by_workspace[workspace] = arguments
        inventory = run(["cargo", "nextest", "list", *arguments_by_workspace[baseline],
            "--run-ignored", "all", "--message-format", "json"],
            cwd=baseline, env=workspace_environment(baseline))
        try:
            required = support._selected_test_names(json.loads(inventory.stdout))
        except json.JSONDecodeError as error:
            raise OracleFailure(f"invalid baseline suite inventory: {error}") from error
        run(["cargo", "nextest", "run", *arguments_by_workspace[ROOT]], required_tests=required,
            env=workspace_environment(ROOT))


def inspect_source_owners(value: dict) -> None:
    fields = {"canonical", "missing", "empty", "duplicate"}
    if not isinstance(value, dict) or set(value) != fields:
        raise OracleFailure("incomplete canonical source owner evidence")
    if any(value[field] != list(SOURCE_MODULES) for field in fields):
        raise OracleFailure("canonical source owner controls omit a destination or failure lane")


def source_owner_controls() -> None:
    import runtime_representation_oracle as source_guard

    expected = tuple(
        (f"crates/chelis-ir/src/{module}.rs", f"crates/chelis-dag-core/src/{module}.rs")
        for module in SOURCE_MODULES
    )
    if source_guard.DAG_SOURCE_RELOCATIONS != expected:
        raise OracleFailure("canonical DAG source relocation inventory differs")
    try:
        rows = source_guard.validate_dag_source_relocations(ROOT)
    except source_guard.OracleFailure as error:
        raise OracleFailure(f"canonical DAG source guard failed: {error.code}: {error}") from error
    source = (ROOT / "crates/chelis-dag-core/tests/fixtures/source_owners.rs").read_text(encoding="utf-8")
    source = source.replace('"__DAG_ROOT__"', json.dumps(str(ROOT)))
    result = consumer(CRATE, source, dependencies=(
        'syn = { version = "2", features = ["full"] }\nserde_json = "1"\n'
    ), env=workspace_environment(ROOT))
    inspect_source_owners(snapshot_record(result.stdout, "DAG_SOURCE_OWNERS"))
    print(json.dumps({"canonical_source_modules": list(SOURCE_MODULES),
        "parsed_representation_owners": [row.identity for row in rows]}, sort_keys=True), flush=True)


CURRENT_HOST_SUITES = {
    "dag_grad_host": (
        "bitwise_coefficients_are_forward_values_and_selected_bitwise_paths_reject",
        "an_expand_extent_does_not_select_a_bitwise_coefficient",
        "bitwise_read_only_through_control_slots_is_a_forward_value",
        "grad_add",
        "grad_mul",
        "grad_div_lhs",
        "grad_pow_matches_finite_differences",
        "grad_pow_selects_exact_zeros_at_zero_exponent_and_zero_base",
        "grad_pow_cotangents_at_an_infinite_base",
        "grad_pow_keeps_the_odd_power_sign_where_y_minus_one_rounds",
        "grad_pow_base_cotangent_survives_forward_overflow_and_underflow",
        "grad_pow_base_matches_finite_differences_across_normal_ranges",
        "grad_pow_negative_base_exponent_cotangent_is_nan",
        "grad_div_rhs",
        "grad_recip",
        "grad_neg",
        "grad_exp",
        "grad_log",
        "grad_sin",
        "grad_sqrt",
        "grad_x_squared",
        "grad_relu_positive",
        "grad_relu_negative",
        "grad_chain_exp_neg",
        "grad_multi_input",
        "grad_accumulation_3x",
        "grad_accumulation_uses_forward_consumer_order_and_a_balanced_tree",
        "balanced_adjoint_sum_tracks_runtime_wildcard_shape",
        "grad_conversion_read_only_through_control_slots_differentiates",
        "grad_predicate_only_parameter_is_positive_zero",
        "grad_inactive_conversion_is_a_constant",
        "grad_sum_expand",
        "grad_second_order",
        "grad_store_passthrough",
        "grad_cast_passthrough",
        "grad_cast_float_to_integer_off_the_gradient_path_is_accepted",
        "grad_cast_integer_source_is_accepted_and_contributes_zero",
        "grad_reshape_roundtrip",
        "grad_permute_transpose",
        "scatter_add_adjoint_preserves_target_and_gathers_update_cotangents",
        "grad_expand_sum_roundtrip",
        "grad_same_rank_expand_sums_over_the_broadcast_axis",
        "grad_mul_by_const",
        "grad_log_chain",
        "grad_sin_chain",
        "grad_max_elem_symmetric",
        "an_untaken_arms_contribution_is_selected_away_by_its_activation",
        "branch_broadcast_masks_rows_before_reducing_a_scalar_capture",
        "runtime_recurrence_callback_vjp_uses_explicit_tensor_seed",
        "adv_wrt_non_leaf_node",
        "adv_exp_at_2",
        "adv_log_at_0_5",
        "adv_sin_at_1",
        "adv_sqrt_at_4",
        "adv_max_x_gt_y",
        "adv_max_x_lt_y",
        "adv_max_equal_inputs",
        "adv_exp_log_identity",
        "adv_x_times_exp_x",
        "adv_sin_x_squared",
        "adv_sigmoid_gradient",
        "adv_all_zero_grad_out",
        "adv_node_used_5_times",
        "adv_deep_neg_chain_even",
        "adv_deep_neg_chain_odd",
        "adv_max_reduce_gradient",
        "adv_max_reduce_with_ties",
        "adv_max_reduce_axis1",
        "adv_log_at_negative_input",
        "adv_sqrt_at_zero",
        "adv_div_gradient",
        "adv_pad_shrink_gradient",
        "adv_shrink_gradient",
        "adv_max_elem_spec_compliance",
        "adv_log_second_order",
        "adv_stride_gradient_scatters_into_strided_slots",
        "adv_gradient_at_nontrivial_values",
        "adv_min_reduce_gradient",
        "adv_prod_reduce_gradient_nonzero",
        "adv_prod_reduce_gradient_with_zero_element",
        "grad_cos",
        "grad_tan",
        "grad_atan",
        "grad_abs_positive",
        "grad_abs_negative",
        "adv_grad_abs_at_zero_is_zero",
        "adv_grad_abs_at_neg2_7_is_neg1",
        "adv_grad_atan_at_1_5",
        "grad_erf_and_erfc_at_0_5",
        "adv_grad_tan_at_0_3_exact",
        "adv_grad_cos_of_add_composition",
        "adv_grad_cos_at_0_5_exact_value",
        "stride_adjoint_symbolic_strided_axis_builds_runtime_trim",
    ),
    "dag_optimize_host": (
        "cse_keeps_each_key_operation_and_its_draw",
    ),
    "dag_tier2_ordered_host": (
        "matmul_shaped_float_contractions_have_the_contraction_form",
        "every_other_einsum_has_no_contraction_form",
        "the_contraction_evaluates_to_the_einsum_graph_bits",
        "every_flattened_row_plans_no_trap_for_operands_that_exist",
        "every_accepted_contraction_stores_no_more_than_its_operands_or_result",
    ),
    "dag_core_axis_sources": (
        "source_lowering_preserves_shared_extent_entry_derivations",
    ),
}


def current_host_suites() -> None:
    for binary, required in CURRENT_HOST_SUITES.items():
        print(json.dumps({"current_host_binary": binary, "required_tests": list(required)}, sort_keys=True), flush=True)
        run(["cargo", "nextest", "run", "--locked", "-p", "chelis-ir", "--test", binary],
            required_tests=required, env=workspace_environment(ROOT))


def run_oracle() -> None:
    completed: set[str] = set()
    os.environ["CARGO_TARGET_DIR"] = str(ROOT / "target" / "agents" / "dag-core-oracle" / "candidate")
    source_owner_controls()
    completed.add("source-owner-controls")
    print(f"DAG platform: {platform.platform()} {platform.machine()}", flush=True)
    run(["rustc", "-vV"])
    for default_features in (False, True):
        command = ["cargo", "build", "--locked", "-p", CRATE, "--lib", "--target", TARGET]
        if not default_features:
            command.append("--no-default-features")
        run(command)
        completed.add("thumb-default" if default_features else "thumb-no-default")
        portable(CRATE, PORTABLE_SOURCE, default_features=default_features, forbidden=FORBIDDEN)
        with isolated_graph(default_features=default_features) as directory:
            inspect_target_graph(target_metadata(directory))
            run(["cargo", "tree", "--target", TARGET, "-e", "features"], cwd=directory)
    completed.update({"standalone-alloc", "dependency-features"})
    assert_boundary_rejected(
        f'chelis-ir = {{ path = {json.dumps(str(ROOT / "crates/chelis-ir"))} }}\n',
        "forbidden target dependency")
    assert_boundary_rejected('serde = { version = "1", features = ["std"] }\n', "target std feature")
    dependencies = fixture_dependencies(ROOT, host=True)
    for path, name in (("chelis_ir::Dag", "Dag"), ("chelis_ir::dag::Dag", "dag")):
        compile_rejected(CRATE, f"use {path};\n", dependencies=dependencies, expected=("E0432", name))
    compile_rejected(CRATE,
        'pub fn forge() -> chelis_dag_core::LoadStoreName { chelis_dag_core::LoadStoreName { label: "x".into(), origin: panic!() } }\n',
        expected=("E0451", "private"))
    completed.add("negative-controls")
    fold_environment = support._environment()
    fold_environment["CHELIS_DAG_FOLD_BITS_OUTPUT"] = str(
        Path(fold_environment["CARGO_TARGET_DIR"]) / "fold-final-stored-bits.jsonl"
    )
    run(["cargo", "nextest", "run", "--locked", "-p", CRATE], env=fold_environment, required_tests=(
        "checked_names_reject_constructor_and_codec_inputs", "invalid_topology_returns_deterministic_diagnostics",
        "every_rebuild_preserves_claims_owners_and_activation_edges", "malformed_declarations_activations_and_claim_edges_fail",
        "runtime_bounds_use_absolute_integer_input_slots", "symbolic_sum_adjoint_keeps_a_shape_only_load_and_scalar_seed",
        "node_json_and_positional_binary_fields_stay_stable", "top_level_names_keep_binding_identity_and_checked_wire_labels",
        "wide_image_folds_preserve_specified_stored_bits"))
    completed.update({"graph-contracts", "checked-names"})
    current_host_suites()
    completed.add("moved-host-tests")
    run(["cargo", "nextest", "run", "--locked", "-p", "chelis-cli", "--test", "parity", "-E", "test(parity_tensor_structural_ops)"],
        required_tests=("parity_tensor_structural_ops",))
    run(["cargo", "nextest", "run", "--locked", "-p", "chelis-cli", "--test", "cli", "-E",
        "test(build_c_tuple_return_header_supports_driver_extraction) | test(build_c_scalar_grad_builds_and_is_numerically_correct)"],
        required_tests=("build_c_tuple_return_header_supports_driver_extraction", "build_c_scalar_grad_builds_and_is_numerically_correct"))
    run(["cargo", "nextest", "run", "--locked", "-p", "chelis-cli", "--test", "metal_dtype_emission_and_bool_add", "-E",
        "test(metal_kernels_are_honestly_typed_per_dtype) | test(metal_rejects_f64_with_a_specific_diagnostic)"],
        required_tests=("metal_kernels_are_honestly_typed_per_dtype", "metal_rejects_f64_with_a_specific_diagnostic"))
    immutable_parity()
    completed.update({"allocator-eval-ad", "ownership-fusion-eval", "wire-baseline", "transform-baseline",
        "native-c", "c-hip-metal-generation", "admission-diagnostics", "scale-allocations", "current-host-suites", "span-comments"})
    run([sys.executable, "scripts/test_dag_core_oracle.py"])
    completed.add("python-controls")
    if completed != EVIDENCE:
        raise OracleFailure(f"incomplete DAG evidence: missing={sorted(EVIDENCE - completed)}")
    print(json.dumps({"platform": platform.platform(), "machine": platform.machine(), "baseline": BASELINE_REVISION,
        "production_baseline": PRODUCTION_BASELINE_REVISION, "evidence": sorted(completed)}, sort_keys=True), flush=True)
    print("DAG core oracle: PASS")
    print("Target compilation is not microcontroller execution. Host execution uses an explicit allocator and an 8 MiB stack.")
    print("HIP and Metal evidence covers source generation and admission, not GPU device execution.")
    print("Each hosted Linux and macOS run must execute this complete command.")


def main() -> int:
    try:
        run_oracle()
    except OracleFailure as error:
        print(f"DAG core oracle: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
