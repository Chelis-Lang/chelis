//! chelis#3700 and chelis#3734: every per-output reduction the C lane emits
//! streams [05-OP-30]'s adjacent-pair tree through local carries.
//!
//! Two oracles, both executed. Bits: each kernel's output equals the DAG
//! evaluator's, byte for byte, across leaf counts that exercise every carry
//! pattern (empty, one leaf, powers of two and their neighbours), with data
//! whose sums depend on association. Allocations: every `chelis_alloc` the
//! kernel makes is counted, and the count must not grow with the number of
//! outputs.

mod common;
mod support;

use chelis_ir::dag::{Dag, DimInfo, NodeId, ReduceWindowKind, RiscOp, RtAxis, RtDim, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::tier2;
use chelis_types::dtype_semantics::{StorageView, TensorStorage};
use chelis_types::types::Prim;
use chelis_types::{RawTensor, finalize_tensor};
use chelis_unord::UnordMap;
use std::fs;
use std::process::Command;

fn lit(shape: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: shape.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}

fn named(names: &[&str], precision: Prim) -> TensorType {
    TensorType {
        dims: names
            .iter()
            .map(|name| DimInfo::Named((*name).into(), None))
            .collect(),
        precision,
    }
}

/// Deterministic xorshift stream, so a failure reproduces exactly.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Finite values spanning `2^-span..2^span` with both signs, so the
    /// rounded sum depends on which leaves meet first. `special` mixes in
    /// signed zeros, infinities and NaN.
    fn float(&mut self, span: i32, special: bool) -> f64 {
        let draw = self.next();
        if special {
            match draw % 11 {
                0 => return -0.0,
                1 => return 0.0,
                2 => return f64::INFINITY,
                3 => return f64::NEG_INFINITY,
                4 => return f64::NAN,
                _ => {}
            }
        }
        let exponent = (draw >> 8) as i32 % (2 * span + 1) - span;
        let mantissa = 1.0 + ((draw >> 24) % 1024) as f64 / 1024.0;
        let sign = if draw & 1 == 0 { 1.0 } else { -1.0 };
        sign * mantissa * 2f64.powi(exponent)
    }
}

fn tensor(shape: Vec<usize>, prim: Prim, stream: &mut Stream, special: bool) -> TensorValue {
    let count = shape.iter().product::<usize>();
    let raw = if prim == Prim::Bool {
        RawTensor::Int((0..count).map(|_| (stream.next() % 2) as i64).collect())
    } else if prim.is_integer() {
        let bound = if prim == Prim::Int8 { 40 } else { 1000 };
        RawTensor::Int(
            (0..count)
                .map(|_| (stream.next() % (2 * bound + 1)) as i64 - bound as i64)
                .collect(),
        )
    } else {
        let span = if prim == Prim::F16 { 12 } else { 24 };
        RawTensor::Float((0..count).map(|_| stream.float(span, special)).collect())
    };
    TensorValue::from_storage(shape, finalize_tensor("carry_tree", prim, raw).unwrap())
}

fn bytes(storage: &TensorStorage) -> Vec<u8> {
    match storage.view() {
        StorageView::F64(v) => v.iter().flat_map(|n| n.to_bits().to_le_bytes()).collect(),
        StorageView::F32(v) => v.iter().flat_map(|n| n.to_bits().to_le_bytes()).collect(),
        StorageView::F16(v) => v.iter().flat_map(|n| n.to_bits().to_le_bytes()).collect(),
        StorageView::Bf16(v) => v.iter().flat_map(|n| n.to_bits().to_le_bytes()).collect(),
        StorageView::I64(v) => v.iter().flat_map(|n| n.to_le_bytes()).collect(),
        StorageView::I32(v) => v.iter().flat_map(|n| n.to_le_bytes()).collect(),
        StorageView::I16(v) => v.iter().flat_map(|n| n.to_le_bytes()).collect(),
        StorageView::I8(v) => v.iter().flat_map(|n| n.to_le_bytes()).collect(),
        StorageView::Bool(v) => v.to_vec(),
        StorageView::Key(_) => panic!("no key reductions here"),
    }
}

fn literal(data: &[u8]) -> String {
    if data.is_empty() {
        "0".into()
    } else {
        data.iter().map(u8::to_string).collect::<Vec<_>>().join(",")
    }
}

/// One generated kernel and the inputs it runs on.
struct Kernel {
    label: String,
    dag: Dag,
    root: NodeId,
    runs: Vec<UnordMap<String, TensorValue>>,
}

/// Compile every kernel with a counting `chelis_alloc`, run each input set,
/// require the evaluator's exact bytes, and return each run's allocation
/// count, per kernel.
fn execute(name: &str, kernels: &[Kernel]) -> Vec<Vec<usize>> {
    let probe = common::probe_dir(&format!("carry_tree_{name}"));
    let dir = probe.path();
    let staged = chelis_runtime_bundle::stage(dir).unwrap();
    let mut driver = String::from(
        "#include \"chelis_runtime.h\"\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n\
         static size_t calls;\n\
         chelis_tensor *probe_alloc(int32_t rank, const int64_t *shape, chelis_dtype dtype) {\n\
         calls++; return chelis_alloc(rank, shape, dtype);\n}\n",
    );
    let mut sources = Vec::new();
    for (index, kernel) in kernels.iter().enumerate() {
        let function = format!("carry_kernel_{index}");
        let code = support::codegen(&kernel.dag, &function)
            .unwrap_or_else(|error| panic!("{}: codegen: {error}", kernel.label));
        assert_eq!(code.output_labels.len(), 1, "{}", kernel.label);
        let source = code
            .c_source
            .replace("chelis_alloc(", "probe_alloc(")
            .replace(
                "#include \"chelis_runtime.h\"",
                "#include \"chelis_runtime.h\"\nchelis_tensor *probe_alloc(int32_t, const int64_t *, chelis_dtype);",
            );
        let path = dir.join(format!("{function}.c"));
        fs::write(&path, source).unwrap();
        sources.push(path);
        driver.push_str(&format!(
            "void {function}(chelis_tensor **, int, chelis_tensor **, int);\n"
        ));
        for (run, inputs) in kernel.runs.iter().enumerate() {
            let expected = eval_tensor(&kernel.dag, inputs)
                .unwrap_or_else(|error| panic!("{} run {run}: evaluator: {error}", kernel.label));
            let expected = bytes(expected[&kernel.root].storage());
            driver.push_str(&format!("static void run_{index}_{run}(void) {{\n"));
            for (slot, label) in code.input_labels.iter().enumerate() {
                let value = &inputs[label];
                let data = bytes(value.storage());
                let prim = kernel
                    .dag
                    .nodes()
                    .iter()
                    .find(
                        |node| matches!(&node.op, RiscOp::Load { name } if name.as_str() == label),
                    )
                    .unwrap()
                    .output_type
                    .precision;
                driver.push_str(&format!(
                    "_Alignas(8) static const unsigned char input_{slot}[] = {{ {} }};\n\
                     int64_t shape_{slot}[] = {{ {} }};\n\
                     chelis_tensor *in_{slot} = chelis_tensor_entry_borrow({}, shape_{slot}, {}, input_{slot}, {});\n",
                    literal(&data),
                    if value.shape.is_empty() {
                        "0".into()
                    } else {
                        value.shape.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
                    },
                    value.shape.len(),
                    prim.runtime_dtype().unwrap().c_macro(),
                    data.len()
                ));
            }
            let inputs_list = (0..code.input_labels.len())
                .map(|slot| format!("in_{slot}"))
                .collect::<Vec<_>>()
                .join(",");
            driver.push_str(&format!(
                "chelis_tensor *inputs[] = {{ {inputs_list} }};\nchelis_tensor *outputs[1] = {{0}};\ncalls = 0;\n\
                 {function}(inputs, {}, outputs, 1);\n\
                 static const unsigned char expected[] = {{ {} }};\n\
                 if (chelis_tensor_byte_count(outputs[0]) != {len} || ({len} && memcmp(chelis_tensor_read_view(outputs[0]).data, expected, {len}) != 0)) {{\n\
                 fprintf(stderr, \"{label} run {run}: bits differ from the evaluator\\n\");\n\
                 const unsigned char *got = chelis_tensor_read_view(outputs[0]).data;\n\
                 for (int64_t i = 0; i < chelis_tensor_byte_count(outputs[0]); i++) fprintf(stderr, \"%02x\", got[i]);\n\
                 fprintf(stderr, \"\\n\"); exit(3);\n}}\n\
                 printf(\"{index} {run} %zu\\n\", calls);\n\
                 chelis_tensor_release(outputs[0]);\n",
                code.input_labels.len(),
                literal(&expected),
                len = expected.len(),
                label = kernel.label,
            ));
            for slot in 0..code.input_labels.len() {
                driver.push_str(&format!("chelis_tensor_release(in_{slot});\n"));
            }
            driver.push_str("}\n");
        }
    }
    driver.push_str("int main(void) {\nsetbuf(stdout, NULL);\n");
    for (index, kernel) in kernels.iter().enumerate() {
        for run in 0..kernel.runs.len() {
            driver.push_str(&format!("run_{index}_{run}();\n"));
        }
    }
    driver.push_str("return 0;\n}\n");
    let main = dir.join("driver.c");
    fs::write(&main, driver).unwrap();
    let binary = dir.join("probe");
    let compiled = Command::new("clang")
        .args([
            "-O2",
            "-std=c11",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "-I",
        ])
        .arg(dir)
        .args(&sources)
        .arg(&main)
        .arg(&staged.archive)
        .args(["-lm", "-o"])
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&binary)
        .env("ASAN_OPTIONS", "detect_leaks=0")
        .output()
        .unwrap();
    if !run.status.success() {
        let preserved = probe.keep();
        panic!(
            "carry-tree probe {} failed: {}\n{}",
            preserved.display(),
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
    let mut counts = kernels
        .iter()
        .map(|kernel| vec![usize::MAX; kernel.runs.len()])
        .collect::<Vec<_>>();
    for line in String::from_utf8_lossy(&run.stdout).lines() {
        let fields = line
            .split_whitespace()
            .map(|field| field.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        counts[fields[0]][fields[1]] = fields[2];
    }
    for (kernel, counts) in kernels.iter().zip(&counts) {
        assert!(
            counts.iter().all(|count| *count != usize::MAX),
            "{}: a run reported nothing",
            kernel.label
        );
    }
    counts
}

/// Leaf counts covering every carry shape up to five levels, and the
/// boundaries around larger powers of two.
/// The subset for kernels whose extents are static.
const STATIC_LEAF_COUNTS: &[usize] = &[0, 1, 2, 3, 5, 7, 8, 9, 13, 16, 17, 64, 65];

const LEAF_COUNTS: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128,
    129, 257,
];

/// Runs over `[rows, n]` operands for each leaf count, plus a special-value
/// run, and an allocation pair that changes only the output count.
fn row_runs(
    loads: &[(&str, Prim)],
    shape: impl Fn(&str, usize, usize) -> Vec<usize>,
    counts: &[usize],
) -> Vec<UnordMap<String, TensorValue>> {
    let mut stream = Stream(0x9e37_79b9_7f4a_7c15);
    let mut runs = Vec::new();
    let mut push = |rows: usize, n: usize, special: bool, stream: &mut Stream| {
        runs.push(
            loads
                .iter()
                .map(|(name, prim)| {
                    (
                        (*name).to_string(),
                        tensor(shape(name, rows, n), *prim, stream, special),
                    )
                })
                .collect::<UnordMap<_, _>>(),
        );
    };
    // The first two runs differ only in their output count.
    push(1, 5, false, &mut stream);
    push(9, 5, false, &mut stream);
    for &n in counts {
        push(3, n, false, &mut stream);
    }
    push(4, 9, true, &mut stream);
    runs
}

fn assert_flat_allocations(label: &str, counts: &[usize]) {
    assert_eq!(
        counts[0], counts[1],
        "{label}: allocations grew with the output count (1 output: {}, 9 outputs: {})",
        counts[0], counts[1]
    );
}

fn load(dag: &mut Dag, owner: chelis_ir::dag::DeclId, name: &str, ty: TensorType) -> NodeId {
    dag.add_node(owner, RiscOp::Load { name: name.into() }, vec![], ty, None)
}

#[test]
fn materialized_sums_match_the_evaluator_and_allocate_per_kernel_not_per_output() {
    let mut kernels = Vec::new();
    for (input, accumulator) in [
        (Prim::F32, Prim::F32),
        (Prim::F32, Prim::F64),
        (Prim::F64, Prim::F64),
        (
            Prim::F16,
            RiscOp::default_reduce_sum_accumulator(Prim::F16).unwrap(),
        ),
        (
            Prim::Bf16,
            RiscOp::default_reduce_sum_accumulator(Prim::Bf16).unwrap(),
        ),
        (
            Prim::Int8,
            RiscOp::default_reduce_sum_accumulator(Prim::Int8).unwrap(),
        ),
        (
            Prim::Int32,
            RiscOp::default_reduce_sum_accumulator(Prim::Int32).unwrap(),
        ),
        (
            Prim::Int64,
            RiscOp::default_reduce_sum_accumulator(Prim::Int64).unwrap(),
        ),
    ] {
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let x = load(&mut dag, owner, "x", named(&["rows", "n"], input));
        let sum = dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 1,
                accumulator,
            },
            vec![x],
            named(&["rows"], accumulator),
            None,
        );
        dag.add_root(sum);
        kernels.push(Kernel {
            label: format!("sum {} -> {}", input.name(), accumulator.name()),
            dag,
            root: sum,
            runs: row_runs(&[("x", input)], |_, rows, n| vec![rows, n], LEAF_COUNTS),
        });
    }
    // A fused elementwise chain inside the reduction. Its plan is static,
    // so each shape is its own kernel.
    let mut fused_pair = Vec::new();
    for (index, (rows, n)) in [(1, 5), (9, 5)]
        .into_iter()
        .chain(STATIC_LEAF_COUNTS.iter().map(|n| (3, *n)))
        .enumerate()
    {
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let x = load(&mut dag, owner, "x", lit(&[rows, n], Prim::F32));
        let exp = dag.add_node(
            owner,
            RiscOp::Exp,
            vec![x],
            lit(&[rows, n], Prim::F32),
            None,
        );
        let exp = dag.add_node(
            owner,
            RiscOp::Neg,
            vec![exp],
            lit(&[rows, n], Prim::F32),
            None,
        );
        let sum = dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 1,
                accumulator: Prim::F32,
            },
            vec![exp],
            lit(&[rows], Prim::F32),
            None,
        );
        dag.add_root(sum);
        let dag = chelis_ir::fuse::fuse(&dag);
        let sum = dag.roots()[0];
        let fused = support::codegen(&dag, "fused").unwrap().c_source;
        assert!(
            fused.contains("float v0 = "),
            "the fused path must be selected: {fused}"
        );
        let mut stream = Stream(0x1234_5678 ^ (index as u64) << 20);
        // `exp` of the small-magnitude half of the stream stays finite.
        let values = (0..rows * n).map(|_| stream.float(3, index == 2)).collect();
        let storage = finalize_tensor("carry_tree", Prim::F32, RawTensor::Float(values)).unwrap();
        if index < 2 {
            fused_pair.push(kernels.len());
        }
        kernels.push(Kernel {
            label: format!("fused sum [{rows}, {n}]"),
            dag,
            root: sum,
            runs: vec![UnordMap::from([(
                "x".to_string(),
                TensorValue::from_storage(vec![rows, n], storage),
            )])],
        });
    }
    // A deferred movement view.
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let x = load(&mut dag, owner, "x", named(&["n", "rows"], Prim::F32));
    let moved = dag.add_node(
        owner,
        RiscOp::Permute { axes: vec![1, 0] },
        vec![x],
        named(&["rows", "n"], Prim::F32),
        None,
    );
    let sum = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F64,
        },
        vec![moved],
        named(&["rows"], Prim::F64),
        None,
    );
    dag.add_root(sum);
    kernels.push(Kernel {
        label: "deferred sum".into(),
        dag,
        root: sum,
        runs: row_runs(&[("x", Prim::F32)], |_, rows, n| vec![n, rows], LEAF_COUNTS),
    });
    let counts = execute("sums", &kernels);
    for (kernel, counts) in kernels.iter().zip(&counts) {
        if kernel.runs.len() > 1 {
            assert_flat_allocations(&kernel.label, counts);
        }
    }
    assert_eq!(
        counts[fused_pair[0]][0], counts[fused_pair[1]][0],
        "fused sum: allocations grew with the output count"
    );
}

#[test]
fn inlined_contractions_match_the_evaluator_and_allocate_per_kernel_not_per_output() {
    let mut kernels = Vec::new();
    for precision in [Prim::F32, Prim::F64, Prim::F16, Prim::Bf16] {
        let accumulator = RiscOp::default_reduce_sum_accumulator(precision).unwrap();
        // `[m, k] x [k, n]`: the eight-column blocked kernel.
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let (at, bt) = (named(&["m", "k"], precision), named(&["k", "n"], precision));
        let a = load(&mut dag, owner, "a", at.clone());
        let b = load(&mut dag, owner, "b", bt.clone());
        let y = tier2::lower_matmul_with_accumulator(
            owner.into(),
            &mut dag,
            a,
            b,
            &at,
            &bt,
            accumulator,
            None,
        );
        dag.add_root(y);
        let source = support::codegen(&dag, "blocked").unwrap().c_source;
        assert!(
            precision != Prim::F32 || source.contains("__sum_columns_"),
            "the f32 matmul must select the blocked kernel: {source}"
        );
        // Columns cover partial and whole eight-lane groups.
        let shape = |name: &str, rows: usize, n: usize| match name {
            "a" => vec![rows, n],
            _ => vec![n, 2 * rows + 3],
        };
        kernels.push(Kernel {
            label: format!("blocked contraction {}", precision.name()),
            dag,
            root: y,
            runs: row_runs(&[("a", precision), ("b", precision)], shape, LEAF_COUNTS),
        });
        // `sum(expand(a) * expand(b), 1)` over `[rows, n]`: one lane per
        // output.
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let a = load(&mut dag, owner, "a", named(&["n"], precision));
        let b = load(&mut dag, owner, "b", named(&["rows"], precision));
        let grid = named(&["rows", "n"], precision);
        let axis = |tensor_axis| RtDim::InputAxis {
            tensor: 1,
            axis: RtAxis::Lit(tensor_axis),
        };
        let lhs = dag.add_node(
            owner,
            RiscOp::Expand {
                axis: 0,
                size: axis(0),
            },
            vec![a, b],
            grid.clone(),
            None,
        );
        let rhs = dag.add_node(
            owner,
            RiscOp::Expand {
                axis: 1,
                size: axis(0),
            },
            vec![b, a],
            grid.clone(),
            None,
        );
        let product = dag.add_node(owner, RiscOp::Mul, vec![lhs, rhs], grid, None);
        let y = dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 1,
                accumulator,
            },
            vec![product],
            named(&["rows"], accumulator),
            None,
        );
        dag.add_root(y);
        let source = support::codegen(&dag, "vector").unwrap().c_source;
        assert!(
            source.contains("__lhs_") && !source.contains("__sum_columns_"),
            "the single-lane inlined product must be selected: {source}"
        );
        let shape = |name: &str, rows: usize, n: usize| match name {
            "a" => vec![n],
            _ => vec![rows],
        };
        kernels.push(Kernel {
            label: format!("vector contraction {}", precision.name()),
            dag,
            root: y,
            runs: row_runs(&[("a", precision), ("b", precision)], shape, LEAF_COUNTS),
        });
    }
    let counts = execute("contractions", &kernels);
    for (kernel, counts) in kernels.iter().zip(&counts) {
        assert_flat_allocations(&kernel.label, counts);
    }
}

#[test]
fn counts_and_ordered_adjoint_sums_match_the_evaluator_without_per_output_scratch() {
    let mut kernels = Vec::new();
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let x = load(&mut dag, owner, "x", named(&["rows", "n"], Prim::Bool));
    let count = dag.add_node(
        owner,
        RiscOp::Count { axes: vec![1] },
        vec![x],
        named(&["rows"], Prim::Int64),
        None,
    );
    dag.add_root(count);
    kernels.push(Kernel {
        label: "count".into(),
        dag,
        root: count,
        runs: row_runs(
            &[("x", Prim::Bool)],
            |_, rows, n| vec![rows, n],
            LEAF_COUNTS,
        ),
    });
    // One output per call; its leaves interleave two columns, then a third.
    for precision in [Prim::F32, Prim::F64] {
        let mut dag = Dag::new();
        let owner = dag.declare("test");
        let columns =
            ["p", "q", "r"].map(|name| load(&mut dag, owner, name, named(&["n"], precision)));
        let sum = dag.add_node(
            owner,
            RiscOp::OrderedAdjointSum { groups: vec![2, 1] },
            columns.to_vec(),
            lit(&[], precision),
            None,
        );
        dag.add_root(sum);
        kernels.push(Kernel {
            label: format!("ordered adjoint sum {}", precision.name()),
            dag,
            root: sum,
            runs: row_runs(
                &[("p", precision), ("q", precision), ("r", precision)],
                |_, _, n| vec![n],
                LEAF_COUNTS,
            )
            .split_off(2),
        });
    }
    let counts = execute("count_adjoint", &kernels);
    assert_flat_allocations("count", &counts[0]);
    for (kernel, counts) in kernels.iter().zip(&counts).skip(1) {
        // Every call allocates only its output, whatever its leaf count.
        assert!(
            counts.iter().all(|count| *count == counts[0]),
            "{}: allocations vary with the leaf count: {counts:?}",
            kernel.label
        );
    }
}

#[test]
fn window_sums_and_their_adjoints_match_the_evaluator_without_per_output_scratch() {
    let mut kernels = Vec::new();
    // Windowed extents are static, so each output count is its own kernel;
    // a pair of kernels that differ only in length compares allocations.
    let mut pairs = Vec::new();
    for (reducer, precision, window, stride) in [
        (ReduceWindowKind::Sum, Prim::F32, 5, 1),
        (ReduceWindowKind::Sum, Prim::F32, 13, 2),
        (ReduceWindowKind::Mean, Prim::F64, 7, 3),
        (ReduceWindowKind::Sum, Prim::Int32, 6, 1),
        (ReduceWindowKind::Sum, Prim::Bf16, 3, 1),
    ] {
        let mut pair = Vec::new();
        for length in [window, window + 8 * stride] {
            let outputs = (length - window) / stride + 1;
            let mut dag = Dag::new();
            let owner = dag.declare("test");
            let x = load(&mut dag, owner, "x", lit(&[length], precision));
            let op = RiscOp::ReduceWindow {
                reducer,
                window_shape: vec![window],
                strides: vec![stride],
            };
            let reduced = dag.add_node(owner, op, vec![x], lit(&[outputs], precision), None);
            dag.add_root(reduced);
            let mut stream = Stream(0x2545_f491_4f6c_dd1d ^ length as u64);
            let runs = [false, true]
                .into_iter()
                .filter(|special| !*special || precision.is_float())
                .map(|special| {
                    UnordMap::from([(
                        "x".to_string(),
                        tensor(vec![length], precision, &mut stream, special),
                    )])
                })
                .collect();
            pair.push(kernels.len());
            kernels.push(Kernel {
                label: format!(
                    "window {reducer:?} {} w{window} s{stride} l{length}",
                    precision.name()
                ),
                dag,
                root: reduced,
                runs,
            });
        }
        pairs.push(pair);
        if precision.is_integer() {
            continue;
        }
        let mut pair = Vec::new();
        for length in [window, window + 8 * stride] {
            let outputs = (length - window) / stride + 1;
            let mut dag = Dag::new();
            let owner = dag.declare("test");
            let x = load(&mut dag, owner, "x", lit(&[length], precision));
            let g = load(&mut dag, owner, "g", lit(&[outputs], precision));
            let op = RiscOp::ReduceWindowGrad {
                reducer,
                window_shape: vec![window],
                strides: vec![stride],
            };
            let adjoint = dag.add_node(owner, op, vec![x, g], lit(&[length], precision), None);
            dag.add_root(adjoint);
            let mut stream = Stream(0x8510_7c3d_3a5f_02e1 ^ length as u64);
            let runs = vec![UnordMap::from([
                (
                    "x".to_string(),
                    tensor(vec![length], precision, &mut stream, false),
                ),
                (
                    "g".to_string(),
                    tensor(vec![outputs], precision, &mut stream, false),
                ),
            ])];
            pair.push(kernels.len());
            kernels.push(Kernel {
                label: format!(
                    "window adjoint {reducer:?} {} w{window} s{stride} l{length}",
                    precision.name()
                ),
                dag,
                root: adjoint,
                runs,
            });
        }
        pairs.push(pair);
    }
    let counts = execute("windows", &kernels);
    for pair in pairs {
        assert_eq!(
            counts[pair[0]][0], counts[pair[1]][0],
            "{} and {}: allocations grew with the output count",
            kernels[pair[0]].label, kernels[pair[1]].label
        );
    }
}

/// The tree shape itself, independent of data: the carry schedule the
/// emitter spells and the pass-by-pass adjacent-pair fold build the same
/// expression, operand order included, for every leaf count up to 1024 and
/// around the next powers of two.
#[test]
fn the_carry_schedule_builds_the_adjacent_pair_tree() {
    fn pass_by_pass(count: usize) -> Option<String> {
        let mut level = (0..count).map(|leaf| leaf.to_string()).collect::<Vec<_>>();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| match pair {
                    [left, right] => format!("({left}+{right})"),
                    [carried] => carried.clone(),
                    _ => unreachable!(),
                })
                .collect();
        }
        level.pop()
    }
    fn carried(count: usize) -> Option<String> {
        let mut levels: Vec<Option<String>> = vec![None; 64];
        for leaf in 0..count {
            // Leaf `i` merges once per trailing one bit of `i`.
            let mut carry = leaf.to_string();
            let mut depth = 0;
            let mut pending = leaf;
            while pending & 1 == 1 {
                carry = format!("({}+{carry})", levels[depth].take().unwrap());
                pending >>= 1;
                depth += 1;
            }
            levels[depth] = Some(carry);
        }
        // The waiting subtrees, lowest level (rightmost) first.
        let mut total: Option<String> = None;
        let mut remaining = count;
        for level in levels {
            if remaining & 1 == 1 {
                let subtree = level.unwrap();
                total = Some(match total {
                    Some(total) => format!("({subtree}+{total})"),
                    None => subtree,
                });
            }
            remaining >>= 1;
        }
        total
    }
    for count in (0..=1024).chain([2047, 2048, 2049, 4095, 4096, 4097]) {
        assert_eq!(carried(count), pass_by_pass(count), "{count} leaves");
    }
}

/// The checked additions keep their trap: an overflowing row stops the
/// kernel with the evaluator's diagnostic, and the boundary row does not.
#[test]
fn integer_sum_overflow_traps_like_the_evaluator() {
    for (prim, max) in [(Prim::Int32, i32::MAX as i64), (Prim::Int64, i64::MAX)] {
        for (values, overflows) in [
            (vec![1, 2, 3, max - 7, 1], false),
            (vec![1, 2, 3, max - 6, 1], true),
            (vec![max, 0, 0, 0, 1], true),
        ] {
            let mut dag = Dag::new();
            let owner = dag.declare("test");
            let x = load(&mut dag, owner, "x", named(&["rows", "n"], prim));
            let sum = dag.add_node(
                owner,
                RiscOp::Sum {
                    axis: 1,
                    accumulator: prim,
                },
                vec![x],
                named(&["rows"], prim),
                None,
            );
            dag.add_root(sum);
            let mut rows = vec![0, 0, 0, 0, 0];
            rows.extend(&values);
            let storage = finalize_tensor("carry_tree", prim, RawTensor::Int(rows)).unwrap();
            let inputs = UnordMap::from([(
                "x".to_string(),
                TensorValue::from_storage(vec![2, 5], storage),
            )]);
            let label = format!("{} overflow={overflows}", prim.name());
            match eval_tensor(&dag, &inputs) {
                Ok(_) => {
                    assert!(!overflows, "{label}: the evaluator must trap");
                    execute(
                        &format!("overflow_{}", prim.name()),
                        &[Kernel {
                            label,
                            dag,
                            root: sum,
                            runs: vec![inputs],
                        }],
                    );
                }
                Err(expected) => {
                    assert!(overflows, "{label}: unexpected evaluator trap {expected}");
                    trap_matches(&dag, &inputs, &expected, &label);
                }
            }
        }
    }
}

fn trap_matches(dag: &Dag, inputs: &UnordMap<String, TensorValue>, expected: &str, label: &str) {
    let probe = common::probe_dir("carry_tree_trap");
    let dir = probe.path();
    let staged = chelis_runtime_bundle::stage(dir).unwrap();
    let code = support::codegen(dag, "trapping").unwrap();
    let kernel = dir.join("kernel.c");
    fs::write(&kernel, code.c_source).unwrap();
    let value = &inputs["x"];
    let data = bytes(value.storage());
    let prim = value.storage().prim();
    let driver = format!(
        "#include \"chelis_runtime.h\"\nvoid trapping(chelis_tensor **, int, chelis_tensor **, int);\nint main(void) {{\n\
         _Alignas(8) static const unsigned char data[] = {{ {} }};\nint64_t shape[] = {{ 2, 5 }};\n\
         chelis_tensor *in = chelis_tensor_entry_borrow(2, shape, {}, data, {});\n\
         chelis_tensor *inputs[] = {{ in }};\nchelis_tensor *outputs[1] = {{0}};\ntrapping(inputs, 1, outputs, 1);\nreturn 0;\n}}\n",
        literal(&data),
        prim.runtime_dtype().unwrap().c_macro(),
        data.len()
    );
    let main = dir.join("main.c");
    fs::write(&main, driver).unwrap();
    let binary = dir.join("trapping");
    let compiled = Command::new("clang")
        .args(["-O2", "-std=c11", "-ffp-contract=off", "-I"])
        .arg(dir)
        .arg(&kernel)
        .arg(&main)
        .arg(&staged.archive)
        .args(["-lm", "-o"])
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(&binary).output().unwrap();
    let error = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success(),
        "{label}: expected the trap {expected}"
    );
    assert!(
        error.contains(expected.trim()),
        "{label}: expected {expected}; got {error}"
    );
}

/// The allocation oracle must see a per-output scratch tensor: the same
/// counting driver over a kernel that allocates inside its output loop
/// reports a count that grows with the outputs.
#[test]
fn the_allocation_oracle_detects_a_per_output_scratch_tensor() {
    let mut dag = Dag::new();
    let owner = dag.declare("test");
    let x = load(&mut dag, owner, "x", named(&["rows", "n"], Prim::F32));
    let sum = dag.add_node(
        owner,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![x],
        named(&["rows"], Prim::F32),
        None,
    );
    dag.add_root(sum);
    let source = support::codegen(&dag, "scratch").unwrap().c_source;
    let open = "for (int64_t __reduce_i = 0;";
    assert_eq!(source.matches(open).count(), 1, "{source}");
    // Reintroduce the historical per-output scratch: one allocation and
    // release around each output's leaf loop.
    let reintroduced = source.replace(
        open,
        "chelis_tensor_release(chelis_alloc(1, (int64_t[]){ 1 }, CHELIS_DTYPE_F32));\nfor (int64_t __reduce_i = 0;",
    );
    let probe = common::probe_dir("carry_tree_negative_control");
    let dir = probe.path();
    let staged = chelis_runtime_bundle::stage(dir).unwrap();
    let mut counts = Vec::new();
    for (tag, source) in [("streamed", &source), ("scratch", &reintroduced)] {
        let kernel = dir.join(format!("{tag}.c"));
        fs::write(
            &kernel,
            source.replace("chelis_alloc(", "probe_alloc(").replace(
                "#include \"chelis_runtime.h\"",
                "#include \"chelis_runtime.h\"\nchelis_tensor *probe_alloc(int32_t, const int64_t *, chelis_dtype);",
            ),
        )
        .unwrap();
        let main = dir.join(format!("{tag}_main.c"));
        fs::write(
            &main,
            "#include \"chelis_runtime.h\"\n#include <stdio.h>\nstatic size_t calls;\n\
             chelis_tensor *probe_alloc(int32_t rank, const int64_t *shape, chelis_dtype dtype) { calls++; return chelis_alloc(rank, shape, dtype); }\n\
             void scratch(chelis_tensor **, int, chelis_tensor **, int);\n\
             static size_t run(int64_t rows) {\n\
             static float data[9 * 5];\nint64_t shape[] = { rows, 5 };\n\
             chelis_tensor *in = chelis_tensor_entry_borrow(2, shape, CHELIS_DTYPE_F32, data, (int64_t)(rows * 5 * sizeof(float)));\n\
             chelis_tensor *inputs[] = { in }; chelis_tensor *outputs[1] = {0};\n\
             calls = 0; scratch(inputs, 1, outputs, 1);\n\
             chelis_tensor_release(outputs[0]); chelis_tensor_release(in); return calls;\n}\n\
             int main(void) { printf(\"%zu %zu\\n\", run(1), run(9)); return 0; }\n",
        )
        .unwrap();
        let binary = dir.join(tag);
        let compiled = Command::new("clang")
            .args(["-O2", "-std=c11", "-I"])
            .arg(dir)
            .arg(&kernel)
            .arg(&main)
            .arg(&staged.archive)
            .args(["-lm", "-o"])
            .arg(&binary)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let run = Command::new(&binary).output().unwrap();
        assert!(run.status.success());
        let line = String::from_utf8_lossy(&run.stdout).trim().to_string();
        let [one, nine] = line
            .split_whitespace()
            .map(|count| count.parse::<usize>().unwrap())
            .collect::<Vec<_>>()[..]
        else {
            panic!("{tag}: {line}");
        };
        counts.push((one, nine));
    }
    assert_eq!(counts[0].0, counts[0].1, "streamed: {counts:?}");
    assert_eq!(
        counts[1].1 - counts[1].0,
        8,
        "the reintroduced scratch must cost one allocation per output: {counts:?}"
    );
}
