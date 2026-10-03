//! chelis#2957: built programs compute every transcendental with the correctly
//! rounded kernels they carry (spec/design/correctly_rounded_math.md section 4.2,
//! tests 6 and 8).
//!
//! The canaries run the #2957 witnesses through each C emission route a
//! transcendental can take: a scalar literal, a run-time rank-1 tensor
//! (contiguous), a permuted view (the strided path), a rank-0 tensor, a fused
//! elementwise chain, and (at f32) a fused chain inlined into a reduction. Every
//! route must give the bits `chelis-crmath` gives, which are the correctly rounded
//! results [05-OP-46] defines; the expectation needs no evaluator.
//!
//! The structural scan checks the emitted C itself: no host math library
//! identifier (a rounded libm function other than `sqrt`, vForce, Sleef) appears, and every
//! `chelis_cr_*` entry the unit calls is defined in the unit with internal linkage.
//! A planted `expf(` call is the scan's negative control.

mod common;
mod support;

use std::fmt::Write as _;
use std::fs;
use std::process::Command;

use chelis_backend_c::toolchain::{CodegenRequirements, strict_reference_toolchain};
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;

/// The #2957 witnesses that are leaves of a C emission route: inputs where a
/// libm, vForce, or constant-folded result was observed to differ from the
/// correctly rounded one.
const WITNESSES: [f64; 6] = [
    -0.00018575789,
    1.0632437,
    -7755.11767578125,
    -0.05580474063754082,
    -1.3359944820404053,
    5.531991004943848,
];

#[derive(Clone, Copy)]
enum Function {
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Atan,
}

const FUNCTIONS: [Function; 6] = [
    Function::Exp,
    Function::Log,
    Function::Sin,
    Function::Cos,
    Function::Tan,
    Function::Atan,
];

impl Function {
    fn name(self) -> &'static str {
        match self {
            Function::Exp => "exp",
            Function::Log => "log",
            Function::Sin => "sin",
            Function::Cos => "cos",
            Function::Tan => "tan",
            Function::Atan => "atan",
        }
    }

    fn op(self) -> RiscOp {
        match self {
            Function::Exp => RiscOp::Exp,
            Function::Log => RiscOp::Log,
            Function::Sin => RiscOp::Sin,
            Function::Cos => RiscOp::Cos,
            Function::Tan => RiscOp::Tan,
            Function::Atan => RiscOp::Atan,
        }
    }

    fn f32(self, x: f32) -> f32 {
        match self {
            Function::Exp => chelis_crmath::exp_f32(x),
            Function::Log => chelis_crmath::log_f32(x),
            Function::Sin => chelis_crmath::sin_f32(x),
            Function::Cos => chelis_crmath::cos_f32(x),
            Function::Tan => chelis_crmath::tan_f32(x),
            Function::Atan => chelis_crmath::atan_f32(x),
        }
    }

    fn f64(self, x: f64) -> f64 {
        match self {
            Function::Exp => chelis_crmath::exp_f64(x),
            Function::Log => chelis_crmath::log_f64(x),
            Function::Sin => chelis_crmath::sin_f64(x),
            Function::Cos => chelis_crmath::cos_f64(x),
            Function::Tan => chelis_crmath::tan_f64(x),
            Function::Atan => chelis_crmath::atan_f64(x),
        }
    }
}

fn ty(dims: Vec<usize>, precision: Prim) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision,
    }
}

/// The witness at `precision`, as raw bits widened to u64 (f32 bits occupy the
/// low 32).
fn witness_bits(w: f64, precision: Prim) -> u64 {
    match precision {
        Prim::F32 => u64::from((w as f32).to_bits()),
        Prim::F64 => w.to_bits(),
        other => unreachable!("canary dtype {other:?}"),
    }
}

fn expected_bits(function: Function, x_bits: u64, precision: Prim) -> u64 {
    match precision {
        Prim::F32 => {
            let x = f32::from_bits(u32::try_from(x_bits).expect("f32 bits"));
            u64::from(function.f32(x).to_bits())
        }
        Prim::F64 => function.f64(f64::from_bits(x_bits)).to_bits(),
        other => unreachable!("canary dtype {other:?}"),
    }
}

/// One DAG output: its Store name and the input bits each element was computed from.
struct Route {
    label: String,
    function: Function,
    inputs: Vec<u64>,
}

/// Every route for every function at `precision`, as one DAG with one Store per
/// route. Inputs: `x` (the witnesses), `m` (each witness twice, `[k, 2]`, read
/// through a permutation), and `s<k>` (each witness as a rank-0 tensor).
fn canary_dag(precision: Prim) -> (Dag, Vec<Route>) {
    let k = WITNESSES.len();
    let bits: Vec<u64> = WITNESSES
        .iter()
        .map(|w| witness_bits(*w, precision))
        .collect();
    let mut dag = Dag::new();
    let decl = dag.declare("canary");
    let vector = ty(vec![k], precision);
    let scalar = ty(vec![], precision);
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vector.clone(),
        None,
    );
    let m = dag.add_node(
        decl,
        RiscOp::Load { name: "m".into() },
        vec![],
        ty(vec![k, 2], precision),
        None,
    );
    let permuted = dag.add_node(
        decl,
        RiscOp::Permute { axes: vec![1, 0] },
        vec![m],
        ty(vec![2, k], precision),
        None,
    );
    let rank0: Vec<NodeId> = (0..k)
        .map(|i| {
            dag.add_node(
                decl,
                RiscOp::Load {
                    name: format!("s{i}").into(),
                },
                vec![],
                scalar.clone(),
                None,
            )
        })
        .collect();

    let mut routes = Vec::new();
    let store = |dag: &mut Dag, label: String, value: NodeId, out: TensorType| {
        dag.add_node(
            decl,
            RiscOp::Store { name: label.into() },
            vec![value],
            out,
            None,
        );
    };
    for function in FUNCTIONS {
        let name = function.name();
        for (i, w) in WITNESSES.iter().enumerate() {
            let literal = dag.add_node(
                decl,
                RiscOp::synth_const(precision, *w),
                vec![],
                scalar.clone(),
                None,
            );
            let value = dag.add_node(decl, function.op(), vec![literal], scalar.clone(), None);
            let label = format!("{name}_literal_{i}");
            store(&mut dag, label.clone(), value, scalar.clone());
            routes.push(Route {
                label,
                function,
                inputs: vec![bits[i]],
            });

            let value = dag.add_node(decl, function.op(), vec![rank0[i]], scalar.clone(), None);
            let label = format!("{name}_rank0_{i}");
            store(&mut dag, label.clone(), value, scalar.clone());
            routes.push(Route {
                label,
                function,
                inputs: vec![bits[i]],
            });
        }

        let value = dag.add_node(decl, function.op(), vec![x], vector.clone(), None);
        let label = format!("{name}_contiguous");
        store(&mut dag, label.clone(), value, vector.clone());
        routes.push(Route {
            label,
            function,
            inputs: bits.clone(),
        });

        let value = dag.add_node(
            decl,
            function.op(),
            vec![permuted],
            ty(vec![2, k], precision),
            None,
        );
        let label = format!("{name}_permuted");
        store(&mut dag, label.clone(), value, ty(vec![2, k], precision));
        routes.push(Route {
            label,
            function,
            inputs: bits.iter().chain(bits.iter()).copied().collect(),
        });

        // neg(neg(x)) is x bit for bit, so the fused chain's leaf sees the witness.
        let once = dag.add_node(decl, RiscOp::Neg, vec![x], vector.clone(), None);
        let twice = dag.add_node(decl, RiscOp::Neg, vec![once], vector.clone(), None);
        let value = dag.add_node(decl, function.op(), vec![twice], vector.clone(), None);
        let label = format!("{name}_fused");
        store(&mut dag, label.clone(), value, vector.clone());
        routes.push(Route {
            label,
            function,
            inputs: bits.clone(),
        });
    }
    (fuse(&dag), routes)
}

fn c_float_type(precision: Prim) -> &'static str {
    match precision {
        Prim::F32 => "float",
        Prim::F64 => "double",
        other => unreachable!("canary dtype {other:?}"),
    }
}

fn dtype_macro(precision: Prim) -> &'static str {
    match precision {
        Prim::F32 => "CHELIS_DTYPE_F32",
        Prim::F64 => "CHELIS_DTYPE_F64",
        other => unreachable!("canary dtype {other:?}"),
    }
}

fn bits_literal(bits: u64, precision: Prim) -> String {
    match precision {
        Prim::F32 => format!("chelis_f32_from_bits(UINT32_C(0x{bits:08x}))"),
        _ => format!("chelis_f64_from_bits(UINT64_C(0x{bits:016x}))"),
    }
}

/// A `main` that binds each input label to its witness data, runs the kernel,
/// and prints `label index bits` for every output element.
fn harness(
    entry: &str,
    precision: Prim,
    input_labels: &[String],
    output_labels: &[String],
) -> String {
    let c_type = c_float_type(precision);
    let dtype = dtype_macro(precision);
    let k = WITNESSES.len();
    let bits: Vec<u64> = WITNESSES
        .iter()
        .map(|w| witness_bits(*w, precision))
        .collect();
    let mut body = String::new();
    for (slot, label) in input_labels.iter().enumerate() {
        let (shape, values): (Vec<usize>, Vec<u64>) = if label == "x" {
            (vec![k], bits.clone())
        } else if label == "m" {
            (vec![k, 2], bits.iter().flat_map(|b| [*b, *b]).collect())
        } else if let Some(index) = label.strip_prefix('s') {
            (
                vec![],
                vec![bits[index.parse::<usize>().expect("rank-0 label")]],
            )
        } else {
            panic!("unexpected input label `{label}`");
        };
        let data = values
            .iter()
            .map(|b| bits_literal(*b, precision))
            .collect::<Vec<_>>()
            .join(", ");
        let shape_init = if shape.is_empty() {
            "0".to_string()
        } else {
            shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(
            body,
            "    static {c_type} in{slot}[{len}];\n    {{ {c_type} init[{len}] = {{{data}}}; memcpy(in{slot}, init, sizeof init); }}\n    int64_t shape{slot}[{rank_len}] = {{{shape_init}}};\n    inputs[{slot}] = chelis_tensor_entry_borrow({rank}, shape{slot}, {dtype}, in{slot}, sizeof in{slot});",
            len = values.len(),
            rank_len = shape.len().max(1),
            rank = shape.len(),
        )
        .unwrap();
    }
    let print = match precision {
        Prim::F32 => {
            "uint32_t b; memcpy(&b, &v[i], sizeof b); printf(\"%s %lld %08x\\n\", labels[o], (long long)i, (unsigned)b);"
        }
        _ => {
            "uint64_t b; memcpy(&b, &v[i], sizeof b); printf(\"%s %lld %016llx\\n\", labels[o], (long long)i, (unsigned long long)b);"
        }
    };
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
        const {c_type} *v = (const {c_type} *)chelis_tensor_read_view(outputs[o]).data;
        for (int64_t i = 0; i < chelis_tensor_numel(outputs[o]); i++) {{ {print} }}
    }}
    return 0;
}}
"#,
        n_in = input_labels.len(),
        n_out = output_labels.len(),
    )
}

/// Compile `kernel` and `main` under the strict profile and return stdout.
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
    let bin = dir.join("canary");
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
    let run = Command::new(&bin).output().expect("run the canary");
    assert!(
        run.status.success(),
        "{label}: run failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).expect("utf-8 canary output")
}

fn assert_routes_are_correctly_rounded(precision: Prim) {
    let (dag, routes) = canary_dag(precision);
    assert!(
        dag.nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::FusedElem { .. })),
        "the canary must exercise a fused elementwise chain"
    );
    let entry = format!("canary_{}", precision.name());
    let result = support::codegen(&dag, &entry).expect("canary codegen");
    let main = harness(
        &entry,
        precision,
        &result.input_labels,
        &result.output_labels,
    );
    let stdout = compile_and_run(&entry, &result.c_source, &main);

    let mut checked = 0usize;
    let mut mismatches = Vec::new();
    for route in &routes {
        let lines: Vec<&str> = stdout
            .lines()
            .filter(|line| line.split(' ').next() == Some(route.label.as_str()))
            .collect();
        assert_eq!(
            lines.len(),
            route.inputs.len(),
            "{}: expected {} elements in:\n{stdout}",
            route.label,
            route.inputs.len()
        );
        for (line, input) in lines.iter().zip(&route.inputs) {
            let got = u64::from_str_radix(line.rsplit(' ').next().unwrap(), 16).unwrap();
            let want = expected_bits(route.function, *input, precision);
            checked += 1;
            if got != want {
                mismatches.push(format!(
                    "{}({input:#x}) via {}: got {got:#x}, correctly rounded {want:#x}",
                    route.function.name(),
                    route.label
                ));
            }
        }
    }
    assert!(checked > 0, "no route was checked");
    assert!(
        mismatches.is_empty(),
        "{} of {checked} {} results are not correctly rounded:\n{}",
        mismatches.len(),
        precision.name(),
        mismatches.join("\n")
    );
}

#[test]
fn f32_witnesses_are_correctly_rounded_on_every_c_route() {
    assert_routes_are_correctly_rounded(Prim::F32);
}

#[test]
fn f64_witnesses_are_correctly_rounded_on_every_c_route() {
    assert_routes_are_correctly_rounded(Prim::F64);
}

/// The f32-only fused-reduce emitter (`emit_fused_reduce`) inlines the chain
/// into the reduction. A max over the witnesses' kernel values is exact, so it
/// must equal the max of the `chelis-crmath` values. `log` is left out: two of
/// its witnesses are negative and give NaN.
#[test]
fn f32_fused_reduction_routes_are_correctly_rounded() {
    let k = WITNESSES.len();
    let bits: Vec<u64> = WITNESSES
        .iter()
        .map(|w| witness_bits(*w, Prim::F32))
        .collect();
    let mut dag = Dag::new();
    let decl = dag.declare("canary");
    let vector = ty(vec![k], Prim::F32);
    let scalar = ty(vec![], Prim::F32);
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vector.clone(),
        None,
    );
    let mut expected = Vec::new();
    for function in FUNCTIONS
        .into_iter()
        .filter(|function| !matches!(function, Function::Log))
    {
        let once = dag.add_node(decl, RiscOp::Neg, vec![x], vector.clone(), None);
        let twice = dag.add_node(decl, RiscOp::Neg, vec![once], vector.clone(), None);
        let value = dag.add_node(decl, function.op(), vec![twice], vector.clone(), None);
        let max = dag.add_node(
            decl,
            RiscOp::MaxReduce { axis: 0 },
            vec![value],
            scalar.clone(),
            None,
        );
        let label = format!("{}_reduced", function.name());
        dag.add_node(
            decl,
            RiscOp::Store {
                name: label.clone().into(),
            },
            vec![max],
            scalar.clone(),
            None,
        );
        let want = bits
            .iter()
            .map(|b| function.f32(f32::from_bits(u32::try_from(*b).unwrap())))
            .fold(f32::NEG_INFINITY, f32::max);
        expected.push((label, u64::from(want.to_bits())));
    }
    let dag = fuse(&dag);
    let entry = "canary_reduce";
    let result = support::codegen(&dag, entry).expect("canary codegen");
    assert!(
        result.c_source.contains("chelis_cr_expf(") && !result.c_source.contains("chelis_cr_logf("),
        "the reduction canary carries exactly the kernels it calls"
    );
    let main = harness(
        entry,
        Prim::F32,
        &result.input_labels,
        &result.output_labels,
    );
    let stdout = compile_and_run(entry, &result.c_source, &main);
    for (label, want) in expected {
        let line = stdout
            .lines()
            .find(|line| line.split(' ').next() == Some(label.as_str()))
            .unwrap_or_else(|| panic!("{label} missing from:\n{stdout}"));
        let got = u64::from_str_radix(line.rsplit(' ').next().unwrap(), 16).unwrap();
        assert_eq!(
            got, want,
            "{label}: got {got:#x}, correctly rounded {want:#x}"
        );
    }
}

/// The C library `<math.h>` functions whose result is not an exact operation:
/// the transcendentals plus `sqrt`, `cbrt`, and `hypot` (C11 7.12). The exact ones
/// (`floor`, `fma`, `fmax`, `roundeven`, ...) are absent: they have one correct
/// result, so the library cannot change it.
const LIBM_ROUNDED_FUNCTIONS: &[&str] = &[
    "acos", "asin", "atan", "atan2", "cos", "sin", "tan", "acosh", "asinh", "atanh", "cosh",
    "sinh", "tanh", "exp", "exp2", "expm1", "log", "log10", "log1p", "log2", "pow", "sqrt",
    "cbrt", "hypot", "erf", "erfc", "lgamma", "tgamma",
];

/// The one rounded libm function generated code may call: C Annex F (F.3)
/// requires `sqrt` to be IEEE 754 correctly rounded, so libm's result is the
/// [05-OP-46] result (spec/design/correctly_rounded_math.md section 4.2).
const LIBM_KEPT: &[&str] = &["sqrt"];

/// Vendor vector-library and Sleef identifiers, none of which is correctly rounded.
const VENDOR_MATH_IDENTIFIERS: &[&str] = &[
    "vvexpf",
    "vvlogf",
    "vvsinf",
    "vvsqrtf",
    "vvtanhf",
    "CHELIS_EXPF8",
    "CHELIS_LOGF8",
    "CHELIS_SINF8",
    "CHELIS_MATH_USE_VFORCE",
];

/// Whether `word` names a host math library routine generated code must not
/// call: a rounded libm function other than [`LIBM_KEPT`] at any precision
/// suffix, or a vendor identifier. Each is matched as a whole C identifier, so
/// `chelis_cr_expf` does not match `expf`.
fn is_host_math_identifier(word: &str) -> bool {
    let libm = LIBM_ROUNDED_FUNCTIONS
        .iter()
        .filter(|name| !LIBM_KEPT.contains(name))
        .any(|name| ["", "f", "l"].iter().any(|suffix| word == format!("{name}{suffix}")));
    libm || VENDOR_MATH_IDENTIFIERS.contains(&word) || word.starts_with("Sleef_")
}

/// `source` with comments and string and character literals blanked, so
/// prose such as a kernel's "natural exp(x)" comment is not read as code.
fn code_only(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i..].starts_with(b"/*") {
            let end = source[i + 2..]
                .find("*/")
                .map_or(bytes.len(), |at| i + 2 + at + 2);
            out.push(' ');
            i = end;
        } else if bytes[i..].starts_with(b"//") {
            i = source[i..].find('\n').map_or(bytes.len(), |at| i + at);
        } else if bytes[i] == b'"' || bytes[i] == b'\'' {
            let quote = bytes[i];
            i += 1;
            while i < bytes.len() && bytes[i] != quote {
                i += if bytes[i] == b'\\' { 2 } else { 1 };
            }
            i += 1;
            out.push_str("\"\"");
        } else {
            let ch = source[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// The host math identifiers `source` calls or names, and every `chelis_cr_*`
/// call without a `static` definition in the unit.
fn host_math_findings(source: &str) -> Vec<String> {
    let mut findings = Vec::new();
    let includes_math_header = source.lines().any(|line| {
        line.trim_start().starts_with("#include")
            && ["Accelerate", "chelis_math.h", "sleef.h"]
                .iter()
                .any(|header| line.contains(header))
    });
    if includes_math_header {
        findings.push("a math library header".to_string());
    }
    let code = code_only(source);
    let source = code.as_str();
    let words: Vec<&str> = source
        .split(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
        .filter(|word| !word.is_empty())
        .collect();
    for word in &words {
        if is_host_math_identifier(word) {
            findings.push(format!("host math identifier `{word}`"));
        }
    }
    for kernel in chelis_crmath::c_source::Kernel::ALL {
        let entry = kernel.entry();
        if words.contains(&entry) {
            let float = if entry.ends_with('f') {
                "float"
            } else {
                "double"
            };
            let definition = format!("static {float} {entry}({float} x)");
            if source.matches(&definition).count() != 1 {
                findings.push(format!(
                    "`{entry}` is called but not defined once as `{definition}`"
                ));
            }
        }
    }
    findings.sort();
    findings.dedup();
    findings
}

/// Every C emission route of every transcendental, at f32 and f64.
fn transcendental_corpus() -> Vec<(String, String)> {
    let mut corpus = Vec::new();
    for precision in [Prim::F32, Prim::F64] {
        let (dag, _) = canary_dag(precision);
        let entry = format!("scan_{}", precision.name());
        let result = support::codegen(&dag, &entry).expect("corpus codegen");
        corpus.push((entry, result.c_source));
    }
    for precision in [Prim::F16, Prim::Bf16] {
        let mut dag = Dag::new();
        let decl = dag.declare("scan");
        let vector = ty(vec![3], precision);
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vector.clone(),
            None,
        );
        for function in FUNCTIONS {
            let value = dag.add_node(decl, function.op(), vec![x], vector.clone(), None);
            dag.add_node(
                decl,
                RiscOp::Store {
                    name: function.name().into(),
                },
                vec![value],
                vector.clone(),
                None,
            );
        }
        let entry = format!("scan_{}", precision.name());
        let result = support::codegen(&dag, &entry).expect("corpus codegen");
        corpus.push((entry, result.c_source));
    }
    corpus
}

#[test]
fn emitted_c_names_no_host_math_library() {
    for (entry, source) in transcendental_corpus() {
        for function in FUNCTIONS {
            assert!(
                source.contains(&format!("chelis_cr_{}", function.name())),
                "{entry} must call the {} kernel",
                function.name()
            );
        }
        let findings = host_math_findings(&source);
        assert!(findings.is_empty(), "{entry}: {findings:?}");
    }
}

/// Negative control: the scan reports a planted libm call and a called but
/// undefined kernel, so a clean scan above is evidence rather than a blind spot.
#[test]
fn host_math_scan_reports_planted_libm_call_and_missing_kernel() {
    let (_, source) = transcendental_corpus().remove(0);
    let planted = source.replacen("chelis_cr_expf(__in", "expf(__in", 1);
    assert_ne!(planted, source, "the plant must change the unit");
    assert!(
        host_math_findings(&planted).contains(&"host math identifier `expf`".to_string()),
        "the scan must report a planted `expf` call"
    );
    let undefined = source.replace(
        "static float chelis_cr_sinf(float x)",
        "static float renamed_sinf(float x)",
    );
    assert!(
        host_math_findings(&undefined)
            .iter()
            .any(|finding| finding.starts_with("`chelis_cr_sinf` is called")),
        "the scan must report a called kernel the unit does not define"
    );
}

/// The scan's exemption is `sqrt` alone: a planted `sqrtf` or `sqrt` passes,
/// and every other rounded libm name, `cbrt` and `hypot` included, is reported.
#[test]
fn host_math_scan_exempts_only_sqrt() {
    let (_, source) = transcendental_corpus().remove(0);
    for kept in ["sqrtf", "sqrt", "sqrtl"] {
        let planted = source.replacen("chelis_cr_expf(__in", &format!("{kept}(__in"), 1);
        assert_ne!(planted, source, "the plant must change the unit");
        assert!(host_math_findings(&planted).is_empty(), "{kept} is kept");
    }
    for name in LIBM_ROUNDED_FUNCTIONS
        .iter()
        .filter(|name| !LIBM_KEPT.contains(name))
    {
        for suffix in ["", "f", "l"] {
            let call = format!("{name}{suffix}");
            let planted = source.replacen("chelis_cr_expf(__in", &format!("{call}(__in"), 1);
            assert!(
                host_math_findings(&planted).contains(&format!("host math identifier `{call}`")),
                "the scan must report a planted `{call}` call"
            );
        }
    }
}
