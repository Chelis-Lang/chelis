//! Issue #189: scalar-f32 evaluator vs C backend divergence.
//!
//! The C backend `emit_const` F32 arm narrows the IR's f64 source value
//! to f32 (intentional) and then formats the resulting f32 via a `%.8`
//! format string (lossy). For values in the `1e-7` magnitude range
//! that second step drops most of the significant digits, so the
//! generated C reproduces a value up to ~3% off the closest-f32 value
//! the evaluator's `Vec<f64>` storage would yield after `as f32`. The
//! same lossy pattern lives on the F64 arm (`%.17` format, which
//! treats `.17` as decimal places after the point, not significant
//! digits, and zeroes out values below `1e-17`).
//!
//! Acceptance: the C backend emits bit-identical reproductions of the
//! narrowed f32 (and the source f64) by emitting their bit patterns
//! and bit-casting at runtime. The pre-fix emitter cannot satisfy
//! these tests because its format strings discard information; these
//! fixtures are written before the fix lands to lock the contract.

use chelis_backend_c::emit::CEmitter;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn scalar(p: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision: p,
    }
}

/// The reproducer from issue #189: `cast(0.000000123456789, f32)`.
/// Pre-fix the F32 arm emits `chelis_fill_f32(t0, 0.00000012f);` which
/// parses to f32 bits `0x3400d959` (about `1.2e-7`). The closest-f32
/// to the source value is `f32::from_bits(0x34048f8b)` (about
/// `1.2345679e-7`). Post-fix the emitter must produce the latter bit
/// pattern verbatim.
#[test]
fn issue_189_f32_const_emits_exact_bit_pattern() {
    let mut dag = Dag::new();
    dag.add_node(
        RiscOp::Const {
            value: 0.000000123456789_f64,
        },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let src = CEmitter::emit_dag(&dag, "test_fn");
    let want_bits = (0.000000123456789_f64 as f32).to_bits();
    let needle = format!("0x{want_bits:08x}");
    assert!(
        src.contains(&needle),
        "F32 const must round-trip via exact bit pattern `{needle}`; emitted source:\n{src}"
    );
    // Pre-fix emission used a lossy decimal format and the
    // `chelis_fill_f32` symbol. Post-fix uses a dedicated bit-pattern
    // helper so the lossy format string can never re-emerge through a
    // future refactor that hand-edits the `%.8` literal back in.
    assert!(
        src.contains("chelis_fill_f32_bits"),
        "F32 const must dispatch through `chelis_fill_f32_bits`; emitted source:\n{src}"
    );
}

/// Same shape for f64. Pre-fix the F64 arm emits
/// `chelis_fill_f64(t0, 0.00000000000000000)` for values like `1e-300`
/// because `{:.17}` formats 17 digits after the decimal point. Post-fix
/// the bit pattern of the source f64 must appear verbatim.
#[test]
fn issue_189_f64_const_emits_exact_bit_pattern() {
    let mut dag = Dag::new();
    let v: f64 = 1.0e-300;
    dag.add_node(
        RiscOp::Const { value: v },
        vec![],
        scalar(Prim::F64),
        None,
    );
    let src = CEmitter::emit_dag(&dag, "test_fn");
    let want_bits = v.to_bits();
    let needle = format!("0x{want_bits:016x}");
    assert!(
        src.contains(&needle),
        "F64 const must round-trip via exact bit pattern `{needle}`; emitted source:\n{src}"
    );
    assert!(
        src.contains("chelis_fill_f64_bits"),
        "F64 const must dispatch through `chelis_fill_f64_bits`; emitted source:\n{src}"
    );
}

/// Negative-parity guard. The pre-fix lossy literals must never
/// re-emerge in F32 const emission. Stamp the canonical reproducer
/// from issue #189 plus a denormal and the smallest positive normal so
/// a future format-string refactor that re-introduces `%.8` is caught
/// here directly.
#[test]
fn issue_189_f32_const_does_not_use_lossy_format() {
    let values: &[f64] = &[
        0.000000123456789_f64,
        f32::MIN_POSITIVE as f64,
        // Smallest positive denormal.
        f32::from_bits(0x0000_0001).into(),
        0.1_f64,
    ];
    for &v in values {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Const { value: v },
            vec![],
            scalar(Prim::F32),
            None,
        );
        let src = CEmitter::emit_dag(&dag, "test_fn");
        let v32 = v as f32;
        let want_bits = v32.to_bits();
        // The bit pattern must be present.
        assert!(
            src.contains(&format!("0x{want_bits:08x}")),
            "F32 const for v={v}: expected bit pattern `0x{want_bits:08x}`; \
             emitted source:\n{src}"
        );
    }
}

/// f32 denormal: the smallest positive subnormal f32
/// (`f32::from_bits(0x0000_0001)`, about `1.4e-45`). `{:.8}` flattens
/// this to `0.00000000` and parsing back yields zero. The bit-pattern
/// emitter must preserve the denormal pattern.
#[test]
fn issue_189_f32_const_smallest_denormal_round_trips() {
    let v = f32::from_bits(0x0000_0001);
    let mut dag = Dag::new();
    dag.add_node(
        RiscOp::Const { value: v as f64 },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let src = CEmitter::emit_dag(&dag, "test_fn");
    assert!(
        src.contains("0x00000001"),
        "f32 denormal must round-trip via bit pattern `0x00000001`; emitted source:\n{src}"
    );
}

/// f64 small-magnitude sibling: `0.1f64` has bits `0x3fb999999999999a`.
/// `{:.17}` happens to round-trip 0.1 itself, but the closely related
/// `0.1_f64.next_up()` differs by one ULP and the lossy format may
/// collapse the pair. Pin both with bit-pattern verification.
#[test]
fn issue_189_f64_const_one_ulp_pair_round_trips() {
    let v1: f64 = 0.1;
    let v2: f64 = f64::from_bits(v1.to_bits() + 1);
    for v in [v1, v2] {
        let mut dag = Dag::new();
        dag.add_node(
            RiscOp::Const { value: v },
            vec![],
            scalar(Prim::F64),
            None,
        );
        let src = CEmitter::emit_dag(&dag, "test_fn");
        let bits = v.to_bits();
        assert!(
            src.contains(&format!("0x{bits:016x}")),
            "f64 one-ULP pair member v={v}: expected `0x{bits:016x}`; \
             emitted source:\n{src}"
        );
    }
}
