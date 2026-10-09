//! [05-OP-79]: built programs compute `pow` with the correctly rounded kernel they
//! carry, at every float dtype and on every C emission route `pow` can take.
//!
//! The routes are a contiguous elementwise kernel, a strided one (both operands read
//! through a permuted view), and, at f32 and f64, a fused elementwise chain. Every
//! element must give the bits `chelis-crmath` gives for the same operands, which are
//! the IR evaluator's bits too: the evaluator computes `pow` with those functions. The
//! operand pairs cover the IEEE 754 special cases, a negative base with integer and
//! non-integer exponents, overflow, a subnormal result, and a signaling NaN, which
//! [05-OP-79] decides at the operand's own dtype (so an f16 or bf16 signaling NaN
//! gives NaN even beside a zero exponent). On the fused route the base is
//! `neg(neg(x))`, whose NaN [04-NUM-2] has already made the canonical quiet NaN.

mod common;
mod support;

use std::fmt::Write as _;
use std::fs;
use std::process::Command;

use chelis_backend_c::toolchain::{CodegenRequirements, strict_reference_toolchain};
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;
use half::{bf16, f16};

#[derive(Clone, Copy, Debug)]
enum Operand {
    Value(f64),
    SignalingNan,
}

use Operand::{SignalingNan, Value};

/// `(base, exponent)` pairs.
const PAIRS: [(Operand, Operand); 22] = [
    (Value(-3.0), Value(3.0)),
    (Value(-3.0), Value(2.0)),
    (Value(-2.0), Value(0.5)),
    (Value(6.0), Value(2.5)),
    (Value(0.0), Value(0.0)),
    (Value(-0.0), Value(-3.0)),
    (Value(-0.0), Value(3.0)),
    (Value(0.0), Value(-0.5)),
    (Value(f64::NAN), Value(0.0)),
    (Value(1.0), Value(f64::NAN)),
    (Value(f64::NAN), Value(2.0)),
    (Value(-1.0), Value(f64::INFINITY)),
    (Value(0.5), Value(f64::NEG_INFINITY)),
    (Value(f64::NEG_INFINITY), Value(3.0)),
    (Value(f64::NEG_INFINITY), Value(-2.0)),
    (Value(2.0), Value(1.0e6)),
    (Value(-2.0), Value(1.0e6 + 1.0)),
    (Value(0.5), Value(1.0e6)),
    (Value(2.0), Value(-140.0)),
    (Value(1.1), Value(7.3)),
    (SignalingNan, Value(0.0)),
    (Value(1.0), SignalingNan),
];

const PRECISIONS: [Prim; 4] = [Prim::F32, Prim::F64, Prim::F16, Prim::Bf16];

fn bits(operand: Operand, precision: Prim) -> u64 {
    match (operand, precision) {
        (SignalingNan, Prim::F32) => 0x7f80_0001,
        (SignalingNan, Prim::F64) => 0x7ff0_0000_0000_0001,
        (SignalingNan, Prim::F16) => 0x7c01,
        (SignalingNan, Prim::Bf16) => 0x7f81,
        #[allow(clippy::cast_possible_truncation)]
        (Value(v), Prim::F32) => u64::from((v as f32).to_bits()),
        (Value(v), Prim::F64) => v.to_bits(),
        (Value(v), Prim::F16) => u64::from(f16::from_f64(v).to_bits()),
        (Value(v), Prim::Bf16) => u64::from(bf16::from_f64(v).to_bits()),
        (_, other) => unreachable!("pow dtype {other:?}"),
    }
}

/// The correctly rounded result bits ([05-OP-79]) for stored operand bits.
fn expected(x: u64, y: u64, precision: Prim) -> u64 {
    let narrow32 = |b: u64| u32::try_from(b).expect("32-bit operand");
    let narrow16 = |b: u64| u16::try_from(b).expect("16-bit operand");
    match precision {
        Prim::F32 => u64::from(
            chelis_crmath::pow_f32(f32::from_bits(narrow32(x)), f32::from_bits(narrow32(y)))
                .to_bits(),
        ),
        Prim::F64 => chelis_crmath::pow_f64(f64::from_bits(x), f64::from_bits(y)).to_bits(),
        Prim::F16 => u64::from(
            chelis_crmath::pow_f16(f16::from_bits(narrow16(x)), f16::from_bits(narrow16(y)))
                .to_bits(),
        ),
        Prim::Bf16 => u64::from(
            chelis_crmath::pow_bf16(bf16::from_bits(narrow16(x)), bf16::from_bits(narrow16(y)))
                .to_bits(),
        ),
        other => unreachable!("pow dtype {other:?}"),
    }
}

fn is_nan(bits: u64, precision: Prim) -> bool {
    match precision {
        Prim::F32 => f32::from_bits(u32::try_from(bits).unwrap()).is_nan(),
        Prim::F64 => f64::from_bits(bits).is_nan(),
        other => unreachable!("fused pow dtype {other:?}"),
    }
}

fn ty(dims: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().map(|&d| DimInfo::Lit(d)).collect(),
        precision,
    }
}

fn fuses(precision: Prim) -> bool {
    matches!(precision, Prim::F32 | Prim::F64)
}

/// Loads `x`, `y` (`[k]`) and `mx`, `my` (`[k, 2]`, each row the operand twice), and
/// stores `pow(x, y)` as `direct`, `pow` of the permuted views as `strided`, and, where
/// the dtype fuses, `pow(neg(neg(x)), y)` as `fused`.
fn route_dag(precision: Prim) -> Dag {
    let k = PAIRS.len();
    let mut dag = Dag::new();
    let decl = dag.declare("pow_routes");
    let vector = ty(&[k], precision);
    let load = |dag: &mut Dag, name: &str, t: TensorType| {
        dag.add_node(decl, RiscOp::Load { name: name.into() }, vec![], t, None)
    };
    let x = load(&mut dag, "x", vector.clone());
    let y = load(&mut dag, "y", vector.clone());
    let mx = load(&mut dag, "mx", ty(&[k, 2], precision));
    let my = load(&mut dag, "my", ty(&[k, 2], precision));
    let store = |dag: &mut Dag, name: &str, value, t: TensorType| {
        dag.add_node(decl, RiscOp::Store { name: name.into() }, vec![value], t, None);
    };
    let direct = dag.add_node(decl, RiscOp::Pow, vec![x, y], vector.clone(), None);
    store(&mut dag, "direct", direct, vector.clone());
    let permute = |dag: &mut Dag, m| {
        dag.add_node(
            decl,
            RiscOp::Permute { axes: vec![1, 0] },
            vec![m],
            ty(&[2, k], precision),
            None,
        )
    };
    let (px, py) = (permute(&mut dag, mx), permute(&mut dag, my));
    let strided = dag.add_node(decl, RiscOp::Pow, vec![px, py], ty(&[2, k], precision), None);
    store(&mut dag, "strided", strided, ty(&[2, k], precision));
    if fuses(precision) {
        let once = dag.add_node(decl, RiscOp::Neg, vec![x], vector.clone(), None);
        let twice = dag.add_node(decl, RiscOp::Neg, vec![once], vector.clone(), None);
        let fused = dag.add_node(decl, RiscOp::Pow, vec![twice, y], vector.clone(), None);
        store(&mut dag, "fused", fused, vector);
    }
    fuse(&dag)
}

fn storage(precision: Prim) -> (&'static str, &'static str, &'static str, usize) {
    match precision {
        Prim::F32 => ("uint32_t", "CHELIS_DTYPE_F32", "%08x", 8),
        Prim::F64 => ("uint64_t", "CHELIS_DTYPE_F64", "%016llx", 16),
        Prim::F16 => ("uint16_t", "CHELIS_DTYPE_F16", "%04x", 4),
        Prim::Bf16 => ("uint16_t", "CHELIS_DTYPE_BF16", "%04x", 4),
        other => unreachable!("pow dtype {other:?}"),
    }
}

/// A `main` that binds each input label to its stored bits, runs the kernel, and
/// prints `label index bits` for every output element.
fn harness(entry: &str, precision: Prim, input_labels: &[String], output_labels: &[String]) -> String {
    let (word, dtype, format, _) = storage(precision);
    let xs: Vec<u64> = PAIRS.iter().map(|(x, _)| bits(*x, precision)).collect();
    let ys: Vec<u64> = PAIRS.iter().map(|(_, y)| bits(*y, precision)).collect();
    let k = PAIRS.len();
    let mut body = String::new();
    for (slot, label) in input_labels.iter().enumerate() {
        let (shape, values): (Vec<usize>, Vec<u64>) = match label.as_str() {
            "x" => (vec![k], xs.clone()),
            "y" => (vec![k], ys.clone()),
            "mx" => (vec![k, 2], xs.iter().flat_map(|b| [*b, *b]).collect()),
            "my" => (vec![k, 2], ys.iter().flat_map(|b| [*b, *b]).collect()),
            other => panic!("unexpected input label `{other}`"),
        };
        let data = values
            .iter()
            .map(|b| format!("0x{b:x}u"))
            .collect::<Vec<_>>()
            .join(", ");
        let dims = shape.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
        writeln!(
            body,
            "    static {word} in{slot}[{len}] = {{{data}}};\n    int64_t shape{slot}[{rank}] = {{{dims}}};\n    inputs[{slot}] = chelis_tensor_entry_borrow({rank}, shape{slot}, {dtype}, in{slot}, sizeof in{slot});",
            len = values.len(),
            rank = shape.len(),
        )
        .unwrap();
    }
    let labels = output_labels
        .iter()
        .map(|label| format!("\"{label}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"

extern void {entry}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

int main(void) {{
    chelis_tensor *inputs[{n_in}];
{body}    chelis_tensor *outputs[{n_out}] = {{0}};
    const char *labels[{n_out}] = {{{labels}}};
    {entry}(inputs, {n_in}, outputs, {n_out});
    for (int o = 0; o < {n_out}; o++) {{
        const {word} *v = (const {word} *)chelis_tensor_read_view(outputs[o]).data;
        for (int64_t i = 0; i < chelis_tensor_numel(outputs[o]); i++) {{
            printf("%s %lld {format}\n", labels[o], (long long)i, ({cast})v[i]);
        }}
    }}
    return 0;
}}
"#,
        n_in = input_labels.len(),
        n_out = output_labels.len(),
        cast = if word == "uint64_t" { "unsigned long long" } else { "unsigned" },
    )
}

fn compile_and_run(label: &str, kernel: &str, main: &str) -> String {
    let probe = common::probe_dir(label);
    let dir = probe.path();
    fs::write(dir.join("kernel.c"), kernel).unwrap();
    fs::write(dir.join("main.c"), main).unwrap();
    let staged = chelis_runtime_bundle::stage(dir)
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    let toolchain = strict_reference_toolchain(
        chelis_backend_c::toolchain::c_compiler(),
        CodegenRequirements::default(),
    );
    let bin = dir.join("pow_routes");
    let compiled = Command::new(&toolchain.compiler)
        .args(&toolchain.compile_flags)
        .arg("-std=c11")
        .arg("-I")
        .arg(dir)
        .arg(dir.join("kernel.c"))
        .arg(dir.join("main.c"))
        .arg(&staged.archive)
        .args(&toolchain.link_flags)
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("invoke the C compiler");
    assert!(
        compiled.status.success(),
        "{label}: compile failed:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&bin).output().expect("run the routes");
    assert!(
        run.status.success(),
        "{label}: run failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).expect("utf-8 output")
}

fn assert_routes_are_correctly_rounded(precision: Prim) {
    let dag = route_dag(precision);
    assert_eq!(
        dag.nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::FusedElem { .. })),
        fuses(precision),
        "the fused route must fuse exactly where the dtype fuses"
    );
    let entry = format!("pow_routes_{}", precision.name());
    let result = support::codegen(&dag, &entry).expect("pow codegen");
    assert!(
        result.c_source.contains("chelis_cr_powf(") || result.c_source.contains("chelis_cr_pow("),
        "the unit must call the carried pow kernel"
    );
    let main = harness(&entry, precision, &result.input_labels, &result.output_labels);
    let stdout = compile_and_run(&entry, &result.c_source, &main);
    let mut routes = vec!["direct", "strided"];
    if fuses(precision) {
        routes.push("fused");
    }
    let mut mismatches = Vec::new();
    let mut checked = 0;
    for route in routes {
        let lines: Vec<&str> = stdout
            .lines()
            .filter(|line| line.split(' ').next() == Some(route))
            .collect();
        let repeat = if route == "strided" { 2 } else { 1 };
        assert_eq!(lines.len(), PAIRS.len() * repeat, "{route}:\n{stdout}");
        for (index, line) in lines.iter().enumerate() {
            let (x, y) = PAIRS[index % PAIRS.len()];
            let (mut xb, yb) = (bits(x, precision), bits(y, precision));
            if route == "fused" && is_nan(xb, precision) {
                // [04-NUM-2]: `neg` finalizes a NaN, a signaling one included,
                // to the canonical quiet NaN before `pow` reads it.
                xb = bits(Value(f64::NAN), precision);
            }
            let got = u64::from_str_radix(line.rsplit(' ').next().unwrap(), 16).unwrap();
            let want = expected(xb, yb, precision);
            checked += 1;
            if got != want {
                mismatches.push(format!(
                    "pow({x:?}, {y:?}) via {route}: got {got:#x}, correctly rounded {want:#x}"
                ));
            }
        }
    }
    assert!(checked >= PAIRS.len() * 3, "only {checked} elements checked");
    assert!(
        mismatches.is_empty(),
        "{} {} results differ:\n{}",
        mismatches.len(),
        precision.name(),
        mismatches.join("\n")
    );
}

#[test]
fn pow_is_correctly_rounded_on_every_c_route_at_every_float_dtype() {
    for precision in PRECISIONS {
        assert_routes_are_correctly_rounded(precision);
    }
}

/// The oracle is not vacuous: the pairs reach the cases a libm-free rewrite gets
/// wrong, the signed results of a negative base, and the signaling-NaN rule.
#[test]
fn pow_route_oracle_covers_the_decided_cases() {
    let at = |x: f64, y: f64| expected(bits(Value(x), Prim::F64), bits(Value(y), Prim::F64), Prim::F64);
    assert_eq!(f64::from_bits(at(-3.0, 3.0)), -27.0);
    assert_eq!(f64::from_bits(at(0.0, 0.0)), 1.0);
    assert_eq!(at(-0.0, -3.0), f64::NEG_INFINITY.to_bits());
    assert!(f64::from_bits(at(-2.0, 0.5)).is_nan());
    for precision in PRECISIONS {
        let snan = bits(SignalingNan, precision);
        let zero = bits(Value(0.0), precision);
        let one = bits(Value(1.0), precision);
        let quiet = bits(Value(f64::NAN), precision);
        assert_ne!(expected(snan, zero, precision), one, "{}", precision.name());
        assert_eq!(expected(quiet, zero, precision), one, "{}", precision.name());
    }
}
