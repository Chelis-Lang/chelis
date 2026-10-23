#!/usr/bin/env python3
"""Check portable solver use and hosted checker authority as separate obligations."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import no_std_core_support as support
from no_std_core_support import (
    ROOT, TARGET, OracleFailure, compile_rejected, consumer, portable, run,
)

CRATE = "chelis-type-core"
BASELINE = "3ca99faa7a46158eb6a0d90c2c812a7e3a5f44c1"
BASELINE_REPOSITORY = "https://github.com/Chelis-Lang/chelis.git"
SURF_CASES = {
    "accepted_generic_calls", "accepted_shape_operations", "accepted_collections",
    "accepted_records", "accepted_patterns", "accepted_transforms",
    "rejected_numeric_restrictions", "rejected_invalid_shapes", "rejected_collection_callbacks",
    "rejected_records", "rejected_patterns", "rejected_transforms",
}
HOST_CASES = {
    "generic", "tuple", "tensor", "named_rank", "anchored_rank", "nominal",
    "unbound", "precision", "branch", "bad_nominal",
}
HOST_CASES |= SURF_CASES
FORBIDDEN = (
    "chelis-types", "chelis-surf", "chelis-syntax-host", "chelis-compiler-api",
    "chelis-pipeline-core", "chelis-runtime", "chelis-prove", "chelis-stdlib",
    "stacker", "psm", "tokio", "rayon", "parking_lot", "getrandom",
    "chelis-dag-core",
)

PORTABLE_SOURCE = r'''
#![no_std]
extern crate alloc;
use alloc::vec;
use core::convert::Infallible;
use chelis_type_core::{Dim, RankVar, Scheme, Subst, TensorPrec, Type, TypeVar,
    VarGen, instantiate, unify};
pub fn portable_solver() {
    let mut scheme: Scheme<Infallible> = Scheme::mono(
        Type::Tensor(vec![Dim::Rank(RankVar(0))], TensorPrec::Var(TypeVar(0))));
    scheme.tvars = vec![TypeVar(0)];
    scheme.rvars = vec![RankVar(0)];
    let mut subst: Subst<Infallible> = Subst::new();
    let instance = instantiate(&scheme, &mut VarGen::default(), &subst);
    unify(&instance, &instance, &mut subst).unwrap();
    assert_eq!(subst.apply(&instance), instance);
}
'''

SOLVER_SOURCE = r'''
extern crate alloc;
use alloc::vec;
use core::convert::Infallible;
use chelis_numeric_core::prim::Prim;
use chelis_type_core::{Dim, DimVar, RankVar, Scheme, Subst, TensorPrec, Type,
    TypeErrorKind, TypeVar, VarGen, instantiate, unify};

fn tensor(dims: alloc::vec::Vec<Dim>, precision: Prim) -> Type<Infallible> {
    Type::Tensor(dims, TensorPrec::Concrete(precision))
}
fn name(value: &str) -> Dim { Dim::Name(value.into()) }

fn rank_scheme() {
    let mut scheme: Scheme<Infallible> = Scheme::mono(
        Type::Tensor(vec![Dim::Rank(RankVar(0))], TensorPrec::Var(TypeVar(0))));
    scheme.tvars = vec![TypeVar(0)];
    scheme.rvars = vec![RankVar(0)];
    let mut vars = VarGen::default();
    let mut subst: Subst<Infallible> = Subst::new();
    let first = instantiate(&scheme, &mut vars, &subst);
    let second = instantiate(&scheme, &mut vars, &subst);
    assert_ne!(first, second);
    let expected = tensor(vec![name("batch"), name("seq"), name("hidden")], Prim::F32);
    let other = tensor(vec![name("channel")], Prim::F64);
    unify(&first, &expected, &mut subst).unwrap();
    unify(&second, &other, &mut subst).unwrap();
    assert_eq!(subst.apply(&first), expected);
    assert_eq!(subst.apply(&second), other);
    assert_eq!(scheme.body, Type::Tensor(
        vec![Dim::Rank(RankVar(0))], TensorPrec::Var(TypeVar(0))));

    let pre = RankVar(100);
    let post = RankVar(101);
    let anchored = tensor(vec![Dim::Rank(pre), name("seq"), Dim::Rank(post)], Prim::F32);
    let mut anchored_subst: Subst<Infallible> = Subst::new();
    unify(&anchored, &expected, &mut anchored_subst).unwrap();
    assert_eq!(anchored_subst.apply(&anchored), expected);
    assert_eq!(anchored_subst.apply(&tensor(vec![Dim::Rank(pre), Dim::Rank(post)], Prim::F32)),
        tensor(vec![name("batch"), name("hidden")], Prim::F32));
}

fn solver_failures() {
    let variable: Type<Infallible> = Type::Var(TypeVar(0));
    let mut subst: Subst<Infallible> = Subst::new();
    let error = unify(&variable, &Type::reference(variable.clone()), &mut subst).unwrap_err();
    assert!(matches!(error.kind, TypeErrorKind::OccursCheck));
    assert!(!error.message.is_empty());

    let left = tensor(vec![name("batch")], Prim::F32);
    let right = tensor(vec![name("batch")], Prim::F64);
    let error = unify(&left, &right, &mut Subst::new()).unwrap_err();
    assert!(matches!(error.kind, TypeErrorKind::PrecisionMismatch));

    let ambiguous = tensor(vec![Dim::Rank(RankVar(0)), name("seq"), Dim::Rank(RankVar(1))], Prim::F32);
    let repeated_anchor = tensor(vec![name("seq"), name("x"), name("seq")], Prim::F32);
    let error = unify(&ambiguous, &repeated_anchor, &mut Subst::new()).unwrap_err();
    assert!(matches!(error.kind, TypeErrorKind::DimensionMismatch));
    assert!(error.message.contains("ambiguous"));

    let adjacent = tensor(vec![Dim::Rank(RankVar(0)), Dim::Rank(RankVar(1))], Prim::F32);
    let error = unify(&adjacent, &tensor(vec![name("x"), name("y")], Prim::F32),
        &mut Subst::new()).unwrap_err();
    assert!(error.message.contains("two adjacent rank spreads"));
    let error = unify(&tensor(vec![Dim::Lit(2)], Prim::F32),
        &tensor(vec![Dim::Lit(3)], Prim::F32), &mut Subst::new()).unwrap_err();
    assert!(matches!(error.kind, TypeErrorKind::DimensionMismatch));
}

fn model_surface() {
    let models: alloc::vec::Vec<Type<Infallible>> = vec![
        Type::Unit, Type::Prim(Prim::Int32),
        Type::function(vec![Type::Prim(Prim::Bool)], Type::Unit),
        Type::reference(Type::Unit),
        Type::adt("Option".into(), vec![Type::Prim(Prim::Int32)]),
        Type::tuple(vec![Type::Unit, Type::Prim(Prim::String)]),
    ];
    for model in models {
        let mut subst = Subst::new();
        unify(&model, &model, &mut subst).unwrap();
        assert_eq!(subst.apply(&model), model);
    }
    for precision in [Prim::F32, Prim::F64, Prim::F16, Prim::Bf16,
        Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64, Prim::Bool] {
        let pattern = tensor(vec![Dim::Wildcard, Dim::Var(DimVar(0))], precision);
        let ground = tensor(vec![Dim::Lit(2), name("width")], precision);
        let mut subst = Subst::new();
        unify(&pattern, &ground, &mut subst).unwrap();
        assert_eq!(subst.apply_dim(&Dim::Var(DimVar(0))), name("width"));
        assert_eq!(subst.apply(&ground), ground);
    }
}

fn compression() {
    const LENGTH: u32 = 4096;
    let mut subst: Subst<Infallible> = Subst::new();
    for id in 0..LENGTH {
        subst.insert_type(TypeVar(id), Type::Var(TypeVar(id + 1))).unwrap();
        subst.insert_dim(DimVar(id), Dim::Var(DimVar(id + 1)));
        subst.insert_rank(RankVar(id), vec![Dim::Rank(RankVar(id + 1))]);
    }
    subst.insert_type(TypeVar(LENGTH), Type::Prim(Prim::Int32)).unwrap();
    subst.insert_dim(DimVar(LENGTH), name("width"));
    subst.insert_rank(RankVar(LENGTH), vec![name("batch"), name("width")]);
    let ranked = tensor(vec![Dim::Rank(RankVar(0))], Prim::F32);
    for _ in 0..2 {
        assert_eq!(subst.apply(&Type::Var(TypeVar(0))), Type::Prim(Prim::Int32));
        assert_eq!(subst.apply_dim(&Dim::Var(DimVar(0))), name("width"));
        assert_eq!(subst.apply(&ranked), tensor(vec![name("batch"), name("width")], Prim::F32));
    }
    #[cfg(not(target_os = "none"))]
    let start = std::time::Instant::now();
    for id in 0..LENGTH {
        assert_eq!(subst.apply(&Type::Var(TypeVar(id))), Type::Prim(Prim::Int32));
        assert_eq!(subst.apply_dim(&Dim::Var(DimVar(id))), name("width"));
        let ranked = tensor(vec![Dim::Rank(RankVar(id))], Prim::F32);
        assert_eq!(subst.apply(&ranked), tensor(vec![name("batch"), name("width")], Prim::F32));
    }
    #[cfg(not(target_os = "none"))]
    std::println!("COMPRESSED_LOOKUPS: {} lookups in {} nanoseconds", 3 * LENGTH, start.elapsed().as_nanos());
}
'''

# This bounded bump allocator supplies real storage. It never reuses freed blocks.
ALLOCATOR_SOURCE = r'''
use core::alloc::{GlobalAlloc, Layout};
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};
struct Bump {
    memory: UnsafeCell<[u8; 65536]>,
    next: AtomicUsize,
}
unsafe impl Sync for Bump {}
unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let base = self.memory.get().cast::<u8>();
        let mut used = self.next.load(Ordering::Relaxed);
        loop {
            let Some(address) = (base as usize).checked_add(used)
                .and_then(|value| value.checked_add(layout.align() - 1)) else {
                return core::ptr::null_mut();
            };
            let aligned = address & !(layout.align() - 1);
            let offset = aligned - base as usize;
            let Some(end) = offset.checked_add(layout.size()) else {
                return core::ptr::null_mut();
            };
            if end > 65536 { return core::ptr::null_mut(); }
            match self.next.compare_exchange_weak(used, end, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return unsafe { base.add(offset) },
                Err(current) => used = current,
            }
        }
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}
#[global_allocator]
static ALLOCATOR: Bump = Bump {
    memory: UnsafeCell::new([0; 65536]), next: AtomicUsize::new(0),
};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop { core::hint::spin_loop(); }
}
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    rank_scheme();
    solver_failures();
    model_surface();
    loop { core::hint::spin_loop(); }
}
'''


def dependency(crate: str) -> str:
    path = json.dumps(str(ROOT / "crates" / crate))
    return f'{crate} = {{ path = {path}, default-features = false }}\n'


def portable_graphs() -> None:
    portable(CRATE, PORTABLE_SOURCE, forbidden=FORBIDDEN)
    portable(CRATE, PORTABLE_SOURCE, features=("serde",), forbidden=FORBIDDEN)


def core_feature_isolation() -> None:
    # Portable Deep uses serde itself. It must not enable core model serde.
    compile_rejected(CRATE,
        "pub fn no_core_serde() { let _ = bincode::serialize(&chelis_type_core::Type::<()>::Unit); }\n",
        dependencies='bincode = "=1.3.3"\n', expected=("Serialize", "Type"))
    consumer(CRATE, r'''
use chelis_type_core::{Scheme, Subst, Type};
fn main() {
    let scheme: Scheme<()> = Scheme::mono(Type::Unit);
    let bytes = bincode::serialize(&scheme).unwrap();
    let decoded: Scheme<()> = bincode::deserialize(&bytes).unwrap();
    assert_eq!(decoded.body, scheme.body);
    let subst: Subst<()> = Subst::new();
    let bytes = bincode::serialize(&subst).unwrap();
    let decoded: Subst<()> = bincode::deserialize(&bytes).unwrap();
    assert_eq!(decoded.apply(&Type::Unit), Type::Unit);
}
''', features=("serde",), dependencies='bincode = "=1.3.3"\n')


def target_consumer() -> None:
    """Compile a target object with explicit allocation and panic policy."""
    with tempfile.TemporaryDirectory(prefix="chelis-type-target-") as temporary:
        directory = Path(temporary)
        (directory / "src").mkdir()
        (directory / "Cargo.toml").write_text(
            '[package]\nname = "chelis-type-target"\nversion = "0.0.0"\nedition = "2024"\n'
            '[workspace]\n[profile.dev]\npanic = "abort"\n[dependencies]\n'
            + dependency(CRATE) + dependency("chelis-numeric-core"),
            encoding="utf-8",
        )
        (directory / "src" / "main.rs").write_text(
            "#![no_std]\n#![no_main]\n" + SOLVER_SOURCE + ALLOCATOR_SOURCE,
            encoding="utf-8",
        )
        run(("cargo", "rustc", "--bin", "chelis-type-target", "--target", TARGET,
             "--", "--emit=obj"), cwd=directory)


def solver_consumer() -> None:
    consumer(
        CRATE,
        SOLVER_SOURCE + r'''
fn main() {
    let (send, receive) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        rank_scheme();
        solver_failures();
        model_surface();
        compression();
        send.send(()).unwrap();
    });
    receive.recv_timeout(std::time::Duration::from_secs(30))
        .expect("standalone solver failed or deadlocked");
}
''',
        dependencies=dependency("chelis-numeric-core"),
    )


def authority_fixtures() -> None:
    """Require real Rust failures at each independent authority boundary."""
    core_dependency = dependency(CRATE)
    cases = (
        (
            'use chelis_type_core::Type; use chelis_types::errors::ErrorWitness;\n'
            'pub fn bad() -> Type<ErrorWitness> { Type::Error }\n',
            ("mismatched types",),
        ),
        (
            'use chelis_types::errors::ErrorWitness;\n'
            'pub fn bad() -> ErrorWitness { ErrorWitness(()) }\n',
            ("E0423", "private fields"),
        ),
        (
            'use chelis_types::errors::{CheckError, CheckErrorKind, report};\n'
            'pub fn bad() { let mut errors = Vec::new();\n'
            'report(&mut errors, CheckError::new(CheckErrorKind::Other, "bad".into(), vec![])); }\n',
            ("mismatched types", "DiagnosticSink"),
        ),
        (
            'use chelis_types::errors::DiagnosticSink;\n'
            'pub fn bad() { let mut errors = Vec::new(); let _ = DiagnosticSink { errors: &mut errors }; }\n',
            ("DiagnosticSink", "private fields"),
        ),
        (
            'use chelis_types::errors::DiagnosticSink;\n'
            'pub fn bad(sink: &mut DiagnosticSink<\'_>) { sink.errors.clear(); }\n',
            ("field `errors`", "private"),
        ),
        (
            'use chelis_type_core::{Subst, Type};\n'
            'use chelis_types::{env::Env, errors::ErrorWitness};\n'
            'pub fn bad() { let foreign: Type<()> = Type::Error(());\n'
            'Env::new().generalize(&foreign, &Subst::<ErrorWitness>::new()); }\n',
            ("mismatched types", "ErrorWitness"),
        ),
    )
    for source, expected in cases:
        compile_rejected("chelis-types", source, expected=expected, dependencies=core_dependency)
    compile_rejected(
        "chelis-types",
        'use chelis_types::{DiagnosticCheckpoint, checkpoint_iter_compile_probe};\n'
        'use chelis_types::errors::DiagnosticSink;\n'
        'pub fn bad(sink: &DiagnosticSink<\'_>) { checkpoint_iter_compile_probe(sink, 0usize); }\n',
        features=("checkpoint-compile-probe",),
        expected=("mismatched types", "DiagnosticCheckpoint", "usize"),
    )
    compile_rejected(
        "chelis-types",
        'use chelis_types::DiagnosticCheckpoint;\n'
        'pub fn bad() { let _ = DiagnosticCheckpoint(0usize); }\n',
        features=("checkpoint-compile-probe",),
        expected=("E0423", "private fields"),
    )
    for module, names in (
        ("types", ("Type", "Scheme", "TypeVar", "DimVar", "RankVar", "Dim", "TensorPrec",
                   "VarGen", "Effect", "EffectSet", "Prim", "NominalArg", "PrimSet",
                   "TypeVarRestriction", "CollectionConstraint", "ResultConstraint")),
        ("unify", ("Subst", "TypeError", "TypeErrorKind", "unify", "unify_dim", "unify_tensor_prec",
                   "DeferredExpandConstraint", "DeferredOpaqueUse")),
        ("env", ("free_tvars", "free_dvars", "free_rvars", "collect_dims")),
    ):
        for name in names:
            compile_rejected(
                "chelis-types", f"use chelis_types::{module}::{name};\n",
                expected=("private", name) if module == "env" else ("E0432", module),
            )
    for name in ("Type", "Scheme", "Subst", "TypeVar", "DimVar", "RankVar", "Dim", "TensorPrec",
                 "VarGen", "Effect", "EffectSet", "NominalArg"):
        compile_rejected("chelis-types", f"use chelis_types::{name};\n",
                         expected=("E0432", name))
    for name in ("Type", "Scheme", "Subst", "NominalArg", "CollectionConstraint"):
        compile_rejected(
            CRATE, f"pub fn bad(_: chelis_type_core::{name}) {{}}\n",
            expected=("missing generics", name),
        )


def checkout_baseline(directory: Path) -> Path:
    run(("git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(directory)))
    present = support._execute(("git", "cat-file", "-e", f"{BASELINE}^{{commit}}"), cwd=directory)
    if present.returncode != 0:
        run(("git", "fetch", "--quiet", "--no-tags", "--depth=1",
             BASELINE_REPOSITORY, BASELINE), cwd=directory)
    run(("git", "checkout", "--quiet", "--detach", BASELINE), cwd=directory)
    if run(("git", "rev-parse", "HEAD"), cwd=directory).stdout.strip() != BASELINE:
        raise OracleFailure("the immutable type baseline differs")
    for path in ("Cargo.lock", "crates/chelis-types/src/types.rs",
                 "crates/chelis-types/src/unify.rs", "crates/chelis-types/src/env.rs"):
        if not (directory / path).is_file():
            raise OracleFailure(f"the original type baseline omits {path}")
    return directory


def host_snapshot(workspace: Path) -> dict:
    source = (ROOT / "scripts/fixtures/type_unification_core/host.rs").read_text(encoding="utf-8")
    corpus = json.loads((ROOT / "scripts/fixtures/type_unification_core_corpus.json").read_text(encoding="utf-8"))
    if len(corpus) != len(SURF_CASES) or {row[0] for row in corpus} != SURF_CASES:
        raise OracleFailure("the complete current Surf parity corpus differs")
    for name, accepted, case_source in corpus:
        if accepted != name.startswith("accepted_") or not case_source:
            raise OracleFailure("the Surf parity source or disposition differs")
    text = json.dumps(corpus, ensure_ascii=False)
    if '"####' in text:
        raise OracleFailure("the Surf corpus contains its Rust delimiter")
    source += '\nconst SURF_CORPUS_JSON: &str = r####"' + text + '"####;\n'
    dependencies = 'bincode = "=1.3.3"\nserde_json = { version = "1", features = ["float_roundtrip"] }\n'
    for crate in ("chelis-deep", "chelis-syntax-host", "chelis-surf", "chelis-macros"):
        dependencies += f'{crate} = {{ path = {json.dumps(str(workspace / "crates" / crate))} }}\n'
    environment = support._environment()
    environment["CARGO_TARGET_DIR"] = str(
        ROOT / "target/agents/type-core-current" if workspace == ROOT else workspace.parent / "target")
    result = consumer(workspace / "crates/chelis-types", source,
                      default_features=True, dependencies=dependencies, env=environment)
    records = [line.removeprefix("TYPE_HOST_PARITY ") for line in result.stdout.splitlines()
               if line.startswith("TYPE_HOST_PARITY ")]
    if len(records) != 1:
        raise OracleFailure("the host parity record is absent or repeated")
    try:
        return json.loads(records[0])
    except json.JSONDecodeError as error:
        raise OracleFailure(f"the host parity record is invalid: {error}") from error


def validate_snapshot(snapshot: dict) -> None:
    if not isinstance(snapshot, dict) or set(snapshot) != {"cases", "context"}:
        raise OracleFailure("the host snapshot is incomplete")
    cases = snapshot["cases"]
    if not isinstance(cases, dict) or set(cases) != HOST_CASES:
        raise OracleFailure("the host case inventory differs")
    rejected = {"unbound", "precision", "branch", "bad_nominal"}
    for name, record in cases.items():
        validate_result(record, accepted=name not in rejected and not name.startswith("rejected_"))
    context = snapshot["context"]
    if not isinstance(context, dict) or set(context) != {"bytes", "checks", "threads"}:
        raise OracleFailure("the shared context evidence is incomplete")
    if not context["bytes"] or context["threads"] != 8 or len(context["checks"]) != 3:
        raise OracleFailure("the shared context evidence differs")
    for index, result in enumerate(context["checks"]):
        validate_result(result, accepted=index < 2)


def validate_result(record: dict, *, accepted: bool) -> None:
    common = {"status", "typed_nodes", "total_nodes"}
    fields = {"checked_bytes", "tree", "types"} if accepted else {"diagnostics"}
    if not isinstance(record, dict) or set(record) != common | fields:
        raise OracleFailure("the checked result evidence is incomplete")
    if record["status"] != ("accepted" if accepted else "rejected"):
        raise OracleFailure("the host fixture disposition differs")
    if not isinstance(record["typed_nodes"], int) or not isinstance(record["total_nodes"], int):
        raise OracleFailure("the checker statistics are invalid")
    if not 0 <= record["typed_nodes"] <= record["total_nodes"]:
        raise OracleFailure("the checker statistics are invalid")
    if accepted:
        if not record["checked_bytes"] or not record["tree"]:
            raise OracleFailure("the successful checked output is empty")
    else:
        errors = record["diagnostics"]
        if not isinstance(errors, list) or not errors:
            raise OracleFailure("the rejected output omits diagnostics")
        fields = {"kind", "identifier", "message", "suggestions", "severity_bits",
                  "expected", "got", "span_offset", "span_id"}
        for error in errors:
            if not isinstance(error, dict) or set(error) != fields or not error["kind"] or not error["message"]:
                raise OracleFailure("the ordered diagnostic evidence is incomplete")


def compare_snapshots(baseline: dict, candidate: dict) -> None:
    validate_snapshot(baseline)
    validate_snapshot(candidate)
    if baseline != candidate:
        raise OracleFailure("the checked bytes, metadata, statistics, cache, or ordered diagnostics differ from the immutable base")


def immutable_parity() -> None:
    parent = ROOT / "target/agents"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="type-core-baseline-", dir=parent) as temporary:
        baseline = checkout_baseline(Path(temporary) / "source")
        previous = host_snapshot(baseline)
        current = host_snapshot(ROOT)
        compare_snapshots(previous, current)
        print("TYPE_COMPARISON " + json.dumps({
            "baseline": BASELINE, "before": previous, "after": current,
        }, sort_keys=True), flush=True)


def python_controls() -> None:
    run((sys.executable, "scripts/test_type_unification_core_oracle.py"))


def host_parity() -> None:
    # Private cache and interruption tests are mandatory core unit tests.
    run(("cargo", "nextest", "run", "--no-fail-fast", "-p", CRATE, "--features", "serde"),
        required_tests=(
            "long_alias_resolution_populates_only_transient_caches",
            "interrupted_evidence_copy_rejects_later_store_access",
            "interrupted_composition_invalidates_the_target_store",
            "evidence_serialization_can_access_its_substitution",
            "current_deferred_ledgers_clone_without_shared_mutable_state",
            "current_cache_layout_retains_logical_fields_and_omits_transient_state",
            "concurrent_resolution_and_clone_state_remain_independent",
            "shared_resolution_preserves_quantifier_alias_boundaries",
            "shared_scheme_instances_are_independent",
        ))
    run(("cargo", "nextest", "run", "--no-fail-fast", "-p", "chelis-unord", "--all-targets"))
    run(("cargo", "nextest", "run", "--no-fail-fast", "-p", "chelis-unord",
         "--all-targets", "--no-default-features"))
    # Exclude only the pre-existing census regeneration helper, not behavior tests.
    run(("cargo", "nextest", "run", "--no-fail-fast", "-p", "chelis-types", "--all-targets",
         "-E", "not (binary(unresolved_operand_census) & test(=dump_census_rows))"))
    run(("cargo", "nextest", "run", "--no-fail-fast", "-p", "chelis-types",
         "--lib", "--features", "generalize-sweep-oracle"))
    run(("cargo", "test", "-p", "chelis-types", "--doc"))
    run(("cargo", "check", "-p", "chelis-cli", "-p", "chelis-effects",
         "-p", "chelis-ir", "-p", "chelis-prove", "-p", "chelis-compiler-api",
         "-p", "chelis-reef", "--all-targets"))


def run_oracle() -> None:
    portable_graphs()
    core_feature_isolation()
    target_consumer()
    solver_consumer()
    authority_fixtures()
    immutable_parity()
    host_parity()
    python_controls()


def main() -> int:
    try:
        if sys.platform not in ("linux", "darwin"):
            raise OracleFailure("hosted type acceptance needs Linux or macOS")
        run_oracle()
    except OracleFailure as error:
        print(f"type unification core oracle: FAIL: {error}", file=sys.stderr)
        return 1
    print(f"type unification core oracle: PASS ({sys.platform})")
    print(f"Target evidence: isolated normal and serde graphs, explicit allocator, panic handler, {TARGET}.")
    print(f"Host evidence ({sys.platform}): direct Infallible solver, authority failures, checker parity and stress fixtures.")
    print("No device execution or whole-pipeline portability claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())