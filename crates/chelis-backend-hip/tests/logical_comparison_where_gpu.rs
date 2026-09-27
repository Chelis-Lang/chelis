//! Real ignored HIP matrix for direct comparison/logical/where (#1284).
//!
//! Run only through:
//! `scripts/hip_test.py -p chelis-backend-hip --test logical_comparison_where_gpu -- --ignored --test-threads=1`

mod support;

use chelis_backend_hip::HipCodegenResult;
use chelis_ir::dag::{ComparisonKind, Dag, DimInfo, LogicalKind, RiscOp, RtDim, TensorType};
use chelis_types::types::Prim;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use support::codegen_hip;

const N: usize = 8;

#[derive(Clone)]
struct RawInput {
    precision: Prim,
    bits: Vec<u64>,
}

fn ty(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(4)],
        precision,
    }
}

fn dtype_macro(precision: Prim) -> &'static str {
    match precision {
        Prim::F16 => "CHELIS_DTYPE_F16",
        Prim::Bf16 => "CHELIS_DTYPE_BF16",
        Prim::F32 => "CHELIS_DTYPE_F32",
        Prim::F64 => "CHELIS_DTYPE_F64",
        Prim::Int8 => "CHELIS_DTYPE_I8",
        Prim::Int16 => "CHELIS_DTYPE_I16",
        Prim::Int32 => "CHELIS_DTYPE_I32",
        Prim::Int64 => "CHELIS_DTYPE_I64",
        Prim::Bool => "CHELIS_DTYPE_BOOL",
        other => panic!("unsupported raw HIP test dtype {}", other.name()),
    }
}

fn carrier(precision: Prim) -> &'static str {
    match precision {
        Prim::F16 | Prim::Bf16 => "uint16_t",
        Prim::F32 | Prim::Int32 => "uint32_t",
        Prim::F64 | Prim::Int64 => "uint64_t",
        Prim::Int8 | Prim::Bool => "uint8_t",
        Prim::Int16 => "uint16_t",
        other => panic!("unsupported raw HIP test dtype {}", other.name()),
    }
}

fn width_mask(precision: Prim) -> u64 {
    match precision {
        Prim::F16 | Prim::Bf16 | Prim::Int16 => 0xffff,
        Prim::F32 | Prim::Int32 => 0xffff_ffff,
        Prim::F64 | Prim::Int64 => u64::MAX,
        Prim::Int8 | Prim::Bool => 0xff,
        other => panic!("unsupported raw HIP test dtype {}", other.name()),
    }
}

fn runtime_library_path() -> PathBuf {
    let mut candidates = Vec::new();
    if let Ok(target) = env::var("CARGO_TARGET_DIR") {
        let target = PathBuf::from(target);
        candidates.push(target.join("debug/deps"));
        candidates.push(target.join("release/deps"));
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest.join("../../target/debug/deps"));
    candidates.push(manifest.join("../../target/release/deps"));
    candidates
        .into_iter()
        .filter_map(|directory| fs::read_dir(directory).ok())
        .flat_map(|entries| entries.flatten().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
        })
        .max_by_key(|path| {
            fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .ok()
        })
        .expect("could not locate libchelis_runtime.a")
}

fn stage_runtime(directory: &Path) {
    support::stage_device_runtime(directory);
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let hip_header = manifest.join("runtime/chelis_hip_runtime.h");
    fs::copy(hip_header, directory.join("chelis_hip_runtime.h")).expect("stage HIP header");
    let include = manifest.join("../chelis-runtime/include");
    for name in [
        "chelis_runtime.h",
        "chelis_runtime_views.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        fs::copy(include.join(name), directory.join(name))
            .unwrap_or_else(|error| panic!("stage {name}: {error}"));
    }
    fs::copy(
        runtime_library_path(),
        directory.join("libchelis_runtime.a"),
    )
    .expect("stage runtime library");
}

fn build_main(
    function: &str,
    result: &HipCodegenResult,
    inputs: &BTreeMap<String, RawInput>,
    outputs: &[(Prim, usize)],
) -> String {
    let mut body = Vec::new();
    body.push("    int64_t shape[2] = { 2, 4 };".to_string());
    body.push(format!(
        "    chelis_tensor *inputs[{}] = {{0}};",
        result.input_labels.len()
    ));
    for (slot, label) in result.input_labels.iter().enumerate() {
        let input = &inputs[label];
        assert_eq!(input.bits.len(), N);
        body.push(format!(
            "    inputs[{slot}] = chelis_alloc(2, shape, {});",
            dtype_macro(input.precision)
        ));
        body.push(format!(
            "    chelis_tensor_write *guard_{slot} = chelis_tensor_begin_write(inputs[{slot}]);"
        ));
        body.push(format!(
            "    chelis_write_view view_{slot} = chelis_tensor_write_view(guard_{slot});"
        ));
        for (index, bits) in input.bits.iter().enumerate() {
            body.push(format!(
                "    (({carrier} *)view_{slot}.data)[{index}] = ({carrier})0x{bits:x}ULL;",
                carrier = carrier(input.precision),
                bits = bits & width_mask(input.precision),
            ));
        }
        body.push(format!("    chelis_tensor_end_write(guard_{slot});"));
    }
    body.push(format!(
        "    chelis_tensor *outputs[{}] = {{0}};",
        outputs.len()
    ));
    body.push(format!(
        "    {function}(inputs, {}, outputs, {});",
        result.input_labels.len(),
        outputs.len()
    ));
    for (slot, (precision, count)) in outputs.iter().enumerate() {
        body.push(format!(
            "    chelis_read_view output_{slot} = chelis_tensor_read_view(outputs[{slot}]);"
        ));
        body.push(format!("    printf(\"OUT {slot}\");"));
        body.push(format!(
            "    for (int i = 0; i < {count}; ++i) printf(\" %llx\", (unsigned long long)((const {carrier} *)output_{slot}.data)[i]);",
            carrier = carrier(*precision),
        ));
        body.push("    printf(\"\\n\");".to_string());
        body.push(format!("    chelis_tensor_release(outputs[{slot}]);"));
    }
    for slot in 0..result.input_labels.len() {
        body.push(format!("    chelis_tensor_release(inputs[{slot}]);"));
    }
    format!(
        "#include \"chelis_runtime.h\"\n#include <stdint.h>\n#include <stdio.h>\n\
         extern \"C\" void {function}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);\n\
         int main(void) {{\n{}\n    return 0;\n}}\n",
        body.join("\n")
    )
}

fn compile_and_run(
    dag: &Dag,
    function: &str,
    inputs: &BTreeMap<String, RawInput>,
    outputs: &[Prim],
) -> Vec<Vec<u64>> {
    let shaped_outputs = outputs
        .iter()
        .copied()
        .map(|precision| (precision, N))
        .collect::<Vec<_>>();
    compile_and_run_shaped(dag, function, inputs, &shaped_outputs)
}

fn compile_and_run_shaped(
    dag: &Dag,
    function: &str,
    inputs: &BTreeMap<String, RawInput>,
    outputs: &[(Prim, usize)],
) -> Vec<Vec<u64>> {
    let probe = Command::new("hipcc")
        .arg("--version")
        .output()
        .expect("probe hipcc");
    assert!(
        probe.status.success(),
        "hipcc is required for the manual HIP gate"
    );
    let result = codegen_hip(dag, function).expect("HIP codegen");
    assert_eq!(result.output_labels.len(), outputs.len());
    let temp = tempfile::tempdir().expect("tempdir");
    stage_runtime(temp.path());
    fs::write(temp.path().join("model.cpp"), &result.c_source).expect("write model");
    fs::write(
        temp.path().join("main.cpp"),
        build_main(function, &result, inputs, outputs),
    )
    .expect("write harness");
    let binary = temp.path().join("matrix");
    let mut command = Command::new("hipcc");
    command
        .arg("-O2")
        .args(&result.compile_flags)
        .arg(temp.path().join("main.cpp"))
        .arg(temp.path().join("model.cpp"))
        .arg(temp.path().join("chelis_device_owner.cpp"))
        .arg(format!("-L{}", temp.path().display()))
        .arg("-lchelis_runtime")
        .arg("-lpthread")
        .arg("-ldl")
        .args(&result.link_flags)
        .arg("-o")
        .arg(&binary);
    let compile = command.output().expect("run hipcc");
    assert!(
        compile.status.success(),
        "hipcc failed:\n{}\n{}",
        String::from_utf8_lossy(&compile.stderr),
        result.c_source
    );
    let run = Command::new(binary).output().expect("run HIP matrix");
    assert!(
        run.status.success(),
        "HIP matrix failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let mut parsed = vec![Vec::new(); outputs.len()];
    for line in String::from_utf8(run.stdout).expect("UTF-8 output").lines() {
        let mut fields = line.split_whitespace();
        if fields.next() != Some("OUT") {
            continue;
        }
        let slot = fields
            .next()
            .expect("output slot")
            .parse::<usize>()
            .expect("slot");
        parsed[slot] = fields
            .map(|field| u64::from_str_radix(field, 16).expect("hex output"))
            .collect();
    }
    assert!(
        parsed
            .iter()
            .zip(outputs)
            .all(|(row, (_, count))| row.len() == *count),
        "{parsed:?}"
    );
    parsed
}

fn direct_numeric_dag(precision: Prim) -> (Dag, Vec<Prim>) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let lhs = dag.add_node(
        decl,
        RiscOp::Load { name: "lhs".into() },
        vec![],
        ty(precision),
        None,
    );
    let rhs = dag.add_node(
        decl,
        RiscOp::Load { name: "rhs".into() },
        vec![],
        ty(precision),
        None,
    );
    let cond = dag.add_node(
        decl,
        RiscOp::Load {
            name: "cond".into(),
        },
        vec![],
        ty(Prim::Bool),
        None,
    );
    let mut outputs = Vec::new();
    for kind in [
        ComparisonKind::CmpLt,
        ComparisonKind::Lt,
        ComparisonKind::Eq,
        ComparisonKind::Neq,
        ComparisonKind::Gt,
        ComparisonKind::Gte,
        ComparisonKind::Lte,
    ] {
        let node = dag.add_node(
            decl,
            RiscOp::Compare(kind),
            vec![lhs, rhs],
            ty(Prim::Bool),
            None,
        );
        dag.add_root(node);
        outputs.push(Prim::Bool);
    }
    let selected = dag.add_node(
        decl,
        RiscOp::Where,
        vec![cond, lhs, rhs],
        ty(precision),
        None,
    );
    dag.add_root(selected);
    outputs.push(precision);
    (dag, outputs)
}

fn float_inputs(precision: Prim) -> (Vec<u64>, Vec<u64>) {
    match precision {
        Prim::F16 => (
            vec![
                0x7e11, 0x3c00, 0x8000, 0x7c00, 0xfc00, 0x4000, 0xc000, 0xfe22,
            ],
            vec![
                0x3c00, 0x7e33, 0x0000, 0x7c00, 0x7c00, 0x3c00, 0xc200, 0x7e44,
            ],
        ),
        Prim::Bf16 => (
            vec![
                0x7fc1, 0x3f80, 0x8000, 0x7f80, 0xff80, 0x4000, 0xc000, 0xffc2,
            ],
            vec![
                0x3f80, 0x7fc3, 0x0000, 0x7f80, 0x7f80, 0x3f80, 0xc040, 0x7fc4,
            ],
        ),
        Prim::F32 => (
            vec![
                0x7fc1_2345,
                0x3f80_0000,
                0x8000_0000,
                0x7f80_0000,
                0xff80_0000,
                0x4000_0000,
                0xc000_0000,
                0xffc5_4321,
            ],
            vec![
                0x3f80_0000,
                0x7fc2_3456,
                0,
                0x7f80_0000,
                0x7f80_0000,
                0x3f80_0000,
                0xc040_0000,
                0x7fc6_5432,
            ],
        ),
        Prim::F64 => (
            vec![
                0x7ff8_1234_5678_9abc,
                0x3ff0_0000_0000_0000,
                0x8000_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x4000_0000_0000_0000,
                0xc000_0000_0000_0000,
                0xfff8_abcd_1234_5678,
            ],
            vec![
                0x3ff0_0000_0000_0000,
                0x7ff8_2222_3333_4444,
                0,
                0x7ff0_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0x3ff0_0000_0000_0000,
                0xc008_0000_0000_0000,
                0x7ff8_dead_beef_cafe,
            ],
        ),
        _ => unreachable!(),
    }
}

fn encode_signed(values: &[i64], precision: Prim) -> Vec<u64> {
    values
        .iter()
        .map(|value| (*value as u64) & width_mask(precision))
        .collect()
}

#[test]
#[ignore = "requires a real HIP GPU; run through scripts/hip_test.py"]
fn real_hip_direct_nonnumeric_matrix_is_bit_exact() {
    let cond = vec![1, 0, 1, 0, 1, 0, 1, 0];
    let float_expected = [
        vec![0, 0, 0, 0, 1, 0, 0, 0],
        vec![0, 0, 0, 0, 1, 0, 0, 0],
        vec![0, 0, 1, 1, 0, 0, 0, 0],
        vec![1, 1, 0, 0, 1, 1, 1, 1],
        vec![0, 0, 0, 0, 0, 1, 1, 0],
        vec![0, 0, 1, 1, 0, 1, 1, 0],
        vec![0, 0, 1, 1, 1, 0, 0, 0],
    ];
    for precision in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        let (lhs, rhs) = float_inputs(precision);
        let (dag, outputs) = direct_numeric_dag(precision);
        let inputs = BTreeMap::from([
            (
                "cond".into(),
                RawInput {
                    precision: Prim::Bool,
                    bits: cond.clone(),
                },
            ),
            (
                "lhs".into(),
                RawInput {
                    precision,
                    bits: lhs.clone(),
                },
            ),
            (
                "rhs".into(),
                RawInput {
                    precision,
                    bits: rhs.clone(),
                },
            ),
        ]);
        let actual = compile_and_run(
            &dag,
            &format!("matrix_{}", precision.name()),
            &inputs,
            &outputs,
        );
        assert_eq!(&actual[..7], &float_expected, "{}", precision.name());
        let selected = (0..N)
            .map(|index| {
                if cond[index] != 0 {
                    lhs[index]
                } else {
                    rhs[index]
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(actual[7], selected, "{} where bits", precision.name());
    }

    let lhs_values = [-2, -1, 0, 1, -7, 7, 5, 5];
    let rhs_values = [-1, -1, 0, -1, 7, -7, 5, 6];
    for precision in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        let lhs = encode_signed(&lhs_values, precision);
        let rhs = encode_signed(&rhs_values, precision);
        let predicates = [
            lhs_values
                .iter()
                .zip(rhs_values)
                .map(|(a, b)| u64::from(*a < b))
                .collect::<Vec<_>>(),
            lhs_values
                .iter()
                .zip(rhs_values)
                .map(|(a, b)| u64::from(*a < b))
                .collect::<Vec<_>>(),
            lhs_values
                .iter()
                .zip(rhs_values)
                .map(|(a, b)| u64::from(*a == b))
                .collect::<Vec<_>>(),
            lhs_values
                .iter()
                .zip(rhs_values)
                .map(|(a, b)| u64::from(*a != b))
                .collect::<Vec<_>>(),
            lhs_values
                .iter()
                .zip(rhs_values)
                .map(|(a, b)| u64::from(*a > b))
                .collect::<Vec<_>>(),
            lhs_values
                .iter()
                .zip(rhs_values)
                .map(|(a, b)| u64::from(*a >= b))
                .collect::<Vec<_>>(),
            lhs_values
                .iter()
                .zip(rhs_values)
                .map(|(a, b)| u64::from(*a <= b))
                .collect::<Vec<_>>(),
        ];
        let (dag, outputs) = direct_numeric_dag(precision);
        let inputs = BTreeMap::from([
            (
                "cond".into(),
                RawInput {
                    precision: Prim::Bool,
                    bits: cond.clone(),
                },
            ),
            (
                "lhs".into(),
                RawInput {
                    precision,
                    bits: lhs.clone(),
                },
            ),
            (
                "rhs".into(),
                RawInput {
                    precision,
                    bits: rhs.clone(),
                },
            ),
        ]);
        let actual = compile_and_run(
            &dag,
            &format!("matrix_{}", precision.name()),
            &inputs,
            &outputs,
        );
        assert_eq!(&actual[..7], &predicates, "{}", precision.name());
        let selected = (0..N)
            .map(|index| {
                if cond[index] != 0 {
                    lhs[index]
                } else {
                    rhs[index]
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(actual[7], selected, "{} where bits", precision.name());
    }
}

#[test]
#[ignore = "requires a real HIP GPU; run through scripts/hip_test.py"]
fn real_hip_bool_logic_equality_and_where_matrix_is_exact() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let lhs = dag.add_node(
        decl,
        RiscOp::Load { name: "lhs".into() },
        vec![],
        ty(Prim::Bool),
        None,
    );
    let rhs = dag.add_node(
        decl,
        RiscOp::Load { name: "rhs".into() },
        vec![],
        ty(Prim::Bool),
        None,
    );
    let cond = dag.add_node(
        decl,
        RiscOp::Load {
            name: "cond".into(),
        },
        vec![],
        ty(Prim::Bool),
        None,
    );
    for op in [
        RiscOp::Compare(ComparisonKind::Eq),
        RiscOp::Compare(ComparisonKind::Neq),
        RiscOp::Logical(LogicalKind::And),
        RiscOp::Logical(LogicalKind::Or),
    ] {
        let out = dag.add_node(decl, op, vec![lhs, rhs], ty(Prim::Bool), None);
        dag.add_root(out);
    }
    let not = dag.add_node(
        decl,
        RiscOp::Logical(LogicalKind::Not),
        vec![lhs],
        ty(Prim::Bool),
        None,
    );
    dag.add_root(not);
    let selected = dag.add_node(
        decl,
        RiscOp::Where,
        vec![cond, lhs, rhs],
        ty(Prim::Bool),
        None,
    );
    dag.add_root(selected);
    let lhs_bits = vec![0, 0, 1, 1, 0, 1, 0, 1];
    let rhs_bits = vec![0, 1, 0, 1, 1, 0, 1, 0];
    let cond_bits = vec![1, 1, 1, 1, 0, 0, 0, 0];
    let inputs = BTreeMap::from([
        (
            "cond".into(),
            RawInput {
                precision: Prim::Bool,
                bits: cond_bits.clone(),
            },
        ),
        (
            "lhs".into(),
            RawInput {
                precision: Prim::Bool,
                bits: lhs_bits.clone(),
            },
        ),
        (
            "rhs".into(),
            RawInput {
                precision: Prim::Bool,
                bits: rhs_bits.clone(),
            },
        ),
    ]);
    let outputs = vec![Prim::Bool; 6];
    let actual = compile_and_run(&dag, "matrix_bool", &inputs, &outputs);
    assert_eq!(
        actual[0],
        lhs_bits
            .iter()
            .zip(&rhs_bits)
            .map(|(a, b)| u64::from(a == b))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual[1],
        lhs_bits
            .iter()
            .zip(&rhs_bits)
            .map(|(a, b)| u64::from(a != b))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual[2],
        lhs_bits
            .iter()
            .zip(&rhs_bits)
            .map(|(a, b)| u64::from(*a != 0 && *b != 0))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual[3],
        lhs_bits
            .iter()
            .zip(&rhs_bits)
            .map(|(a, b)| u64::from(*a != 0 || *b != 0))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual[4],
        lhs_bits
            .iter()
            .map(|value| u64::from(*value == 0))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual[5],
        (0..N)
            .map(|index| if cond_bits[index] != 0 {
                lhs_bits[index]
            } else {
                rhs_bits[index]
            })
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires a real HIP GPU; run through scripts/hip_test.py"]
fn real_hip_permuted_stepped_nonnumeric_views_are_exact() {
    let matrix = |rows, cols, precision| TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision,
    };
    let input_f32 = matrix(2, 4, Prim::F32);
    let input_bool = matrix(2, 4, Prim::Bool);
    let permuted_f32 = matrix(4, 2, Prim::F32);
    let permuted_bool = matrix(4, 2, Prim::Bool);
    let stepped_f32 = matrix(2, 2, Prim::F32);
    let stepped_bool = matrix(2, 2, Prim::Bool);
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let view = |dag: &mut Dag,
                name: &str,
                input: &TensorType,
                permuted: &TensorType,
                stepped: &TensorType| {
        let load = dag.add_node(
            decl,
            RiscOp::Load { name: name.into() },
            vec![],
            input.clone(),
            None,
        );
        let permute = dag.add_node(
            decl,
            RiscOp::Permute { axes: vec![1, 0] },
            vec![load],
            permuted.clone(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2), RtDim::Lit(1)],
            },
            vec![permute],
            stepped.clone(),
            None,
        )
    };
    let lhs = view(&mut dag, "lhs", &input_f32, &permuted_f32, &stepped_f32);
    let rhs = view(&mut dag, "rhs", &input_f32, &permuted_f32, &stepped_f32);
    let logical_lhs = view(
        &mut dag,
        "logical_lhs",
        &input_bool,
        &permuted_bool,
        &stepped_bool,
    );
    let logical_rhs = view(
        &mut dag,
        "logical_rhs",
        &input_bool,
        &permuted_bool,
        &stepped_bool,
    );
    let condition = view(
        &mut dag,
        "condition",
        &input_bool,
        &permuted_bool,
        &stepped_bool,
    );
    for kind in [
        ComparisonKind::Eq,
        ComparisonKind::Neq,
        ComparisonKind::Gte,
        ComparisonKind::Lte,
    ] {
        let output = dag.add_node(
            decl,
            RiscOp::Compare(kind),
            vec![lhs, rhs],
            stepped_bool.clone(),
            None,
        );
        dag.add_root(output);
    }
    for kind in [LogicalKind::And, LogicalKind::Or] {
        let output = dag.add_node(
            decl,
            RiscOp::Logical(kind),
            vec![logical_lhs, logical_rhs],
            stepped_bool.clone(),
            None,
        );
        dag.add_root(output);
    }
    let not = dag.add_node(
        decl,
        RiscOp::Logical(LogicalKind::Not),
        vec![logical_lhs],
        stepped_bool,
        None,
    );
    dag.add_root(not);
    let selected = dag.add_node(
        decl,
        RiscOp::Where,
        vec![condition, lhs, rhs],
        stepped_f32,
        None,
    );
    dag.add_root(selected);

    let lhs_bits = vec![
        0x7fc1_2345,
        0x4130_0000,
        0x8000_0000,
        0x4150_0000,
        0x40a0_0000,
        0x4170_0000,
        0x0000_0000,
        0x4188_0000,
    ];
    let rhs_bits = vec![
        0x0000_0000,
        0x41a8_0000,
        0x0000_0000,
        0x41b8_0000,
        0x4080_0000,
        0x41c8_0000,
        0x8000_0000,
        0x41d8_0000,
    ];
    let logical_lhs_bits = vec![1, 1, 0, 1, 0, 1, 1, 0];
    let logical_rhs_bits = vec![1, 0, 0, 1, 1, 0, 0, 1];
    let condition_bits = vec![1, 0, 1, 0, 0, 1, 0, 1];
    let inputs = BTreeMap::from([
        (
            "lhs".into(),
            RawInput {
                precision: Prim::F32,
                bits: lhs_bits.clone(),
            },
        ),
        (
            "rhs".into(),
            RawInput {
                precision: Prim::F32,
                bits: rhs_bits.clone(),
            },
        ),
        (
            "logical_lhs".into(),
            RawInput {
                precision: Prim::Bool,
                bits: logical_lhs_bits,
            },
        ),
        (
            "logical_rhs".into(),
            RawInput {
                precision: Prim::Bool,
                bits: logical_rhs_bits,
            },
        ),
        (
            "condition".into(),
            RawInput {
                precision: Prim::Bool,
                bits: condition_bits,
            },
        ),
    ]);
    let outputs = [
        (Prim::Bool, 4),
        (Prim::Bool, 4),
        (Prim::Bool, 4),
        (Prim::Bool, 4),
        (Prim::Bool, 4),
        (Prim::Bool, 4),
        (Prim::Bool, 4),
        (Prim::F32, 4),
    ];
    let actual = compile_and_run_shaped(&dag, "matrix_permuted_stepped", &inputs, &outputs);
    assert_eq!(actual[0], vec![0, 0, 1, 1]);
    assert_eq!(actual[1], vec![1, 1, 0, 0]);
    assert_eq!(actual[2], vec![0, 1, 1, 1]);
    assert_eq!(actual[3], vec![0, 0, 1, 1]);
    assert_eq!(actual[4], vec![1, 0, 0, 0]);
    assert_eq!(actual[5], vec![1, 1, 0, 1]);
    assert_eq!(actual[6], vec![0, 1, 1, 0]);
    assert_eq!(
        actual[7],
        vec![lhs_bits[0], rhs_bits[4], lhs_bits[2], rhs_bits[6]],
        "where must retain the selected NaN payload and negative-zero images"
    );
}
