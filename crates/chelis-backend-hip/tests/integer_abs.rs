//! Exact integer unary/Grad device emission and compiled kernel execution.
mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn gradient_dag() -> Dag {
    let source = "def loss(x: tensor[4, f32], w: tensor[4, i64]) -> tensor[f32] = sum(mul(x, cast(abs(w), f32)), 0i32)\ndef main(x: tensor[4, f32], w: tensor[4, i64]) -> tensor[4, f32] = (grad(loss, wrt=x))(x, w)\n";
    let declarations = chelis_surf::parser::parse_str(source).expect("parse");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("desugar");
    let checked =
        chelis_types::check_typed_program(&deep).unwrap_or_else(|e| panic!("{:?}", e.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects");
    let checked = chelis_types::check_linearity(&checked).expect("linearity");
    let mut dag =
        chelis_ir::host::lower_named_tensor_entry_dag(&checked, "main").expect("tensor entry");
    chelis_ir::optimize::constant_fold(&mut dag);
    chelis_ir::optimize::dead_code_eliminate(&dag)
}

#[test]
fn lowered_integer_abs_gradient_emits_checked_hip_kernel_and_typed_cast() {
    let dag = gradient_dag();
    let dag = chelis_ir::fuse::fuse(&chelis_ir::specialize::specialize_for_blas(&dag));
    assert!(!dag.nodes().iter().any(|node| matches!(&node.op, RiscOp::FusedElem { ops } if ops.iter().any(|step| step.op == chelis_ir::dag::FusedStepOp::Abs))));
    let result = support::codegen_hip(&dag, "integer_gradient").expect("typed gradient emission");
    assert!(
        result
            .c_source
            .contains("numeric trap: overflow in abs at i64")
    );
    assert!(result.c_source.contains("chelis_cast_bits"));
}

#[test]
fn integer_constants_stay_exact_before_abs() {
    for precision in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision,
        };
        let value = if precision == Prim::Int64 {
            9_007_199_254_740_993
        } else {
            100
        };
        let scalar =
            chelis_types::dtype_semantics::scalar_from_i64("test", precision, -value).unwrap();
        let constant = dag.add_node(
            decl,
            RiscOp::ConstTensor {
                data: chelis_types::dtype_semantics::tensor_from_scalars(
                    precision,
                    &[scalar, scalar],
                ),
            },
            vec![],
            ty.clone(),
            None,
        );
        let abs = dag.add_node(decl, RiscOp::Abs, vec![constant], ty, None);
        dag.set_roots(vec![abs]);
        let result = support::codegen_hip(&dag, "constant_abs").expect("typed constant abs");
        assert!(result.c_source.contains(&format!("INT64_C({value})")));
    }
}

#[test]
fn emitted_hip_integer_kernels_execute_exactly_with_device_intrinsic_shims() {
    use std::{fs, process::Command};
    for (precision, values, minimum) in [
        (Prim::Int8, vec![-127i64, 0, 1, 126], -128),
        (Prim::Int16, vec![-32767, 0, 1, 32766], -32768),
        (Prim::Int32, vec![-16777217, 0, 1, 2147483646], -2147483648),
        (
            Prim::Int64,
            vec![-9007199254740993, 0, 1, i64::MAX - 1],
            i64::MIN,
        ),
    ] {
        let source = chelis_backend_hip::kernels::unary_checked_abs_integer(
            1,
            "checked_abs",
            precision,
            None,
        );
        let ty = match precision {
            Prim::Int8 => "signed char",
            Prim::Int16 => "short",
            Prim::Int32 => "int",
            Prim::Int64 => "long long",
            _ => unreachable!(),
        };
        let values_text = values
            .iter()
            .map(|v| format!("({v}LL)"))
            .collect::<Vec<_>>()
            .join(",");
        let minimum = if minimum == i64::MIN {
            "(-9223372036854775807LL - 1LL)".to_string()
        } else {
            minimum.to_string()
        };
        let expected = values
            .iter()
            .map(|v| format!("{}LL", v.abs()))
            .collect::<Vec<_>>()
            .join(",");
        let code = format!(
            r#"
#include <stdint.h>
#include <stdio.h>
#define __device__
#define __global__
struct Index {{ long long x; }} blockIdx = {{0}}, blockDim = {{1}}, threadIdx = {{0}};
unsigned long long atomicCAS(unsigned long long *p, unsigned long long expected, unsigned long long desired) {{ auto old=*p; if (old==expected) *p=desired; return old; }}
unsigned int atomicExch(unsigned int *p, unsigned int value) {{ auto old=*p; *p=value; return old; }}
{source}
int main() {{
    {ty} input[] = {{ {values_text} }}, output[4] = {{0}};
    long long expected[] = {{ {expected} }};
    for (int i=0; i<4; ++i) {{ threadIdx.x=i; checked_abs(input,1,1,4,output,4,1,4); }}
    if (chelis_numeric_failure_flag != 0) return 1;
    for (int i=0; i<4; ++i) if (output[i] != expected[i]) return 2;
    input[3] = input[2] = {minimum};
    // Reverse thread visitation: first row-major error must still be 2.
    for (int i=3; i>=0; --i) {{ threadIdx.x=i; checked_abs(input,1,1,4,output,4,1,4); }}
    if (chelis_numeric_failure_flag != 1 || chelis_numeric_failure_index != 2) return 3;
    puts("exact");
}}
"#
        );
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("kernel.cpp"), code).unwrap();
        let compile = Command::new("c++")
            .args(["-std=c++17", "-O2", "-fsanitize=undefined"])
            .arg(dir.path().join("kernel.cpp"))
            .arg("-o")
            .arg(dir.path().join("kernel"))
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "{}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let run = Command::new(dir.path().join("kernel")).output().unwrap();
        assert!(
            run.status.success(),
            "{precision:?}: {:?}: {}",
            run.status,
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(
            run.stderr.is_empty(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(run.stdout, b"exact\n");
    }
}

fn element_size(precision: Prim) -> usize {
    match precision {
        Prim::Int8 => 1,
        Prim::Int16 => 2,
        Prim::Int32 | Prim::F32 => 4,
        Prim::Int64 => 8,
        _ => panic!("fixture dtype"),
    }
}

fn run_hip(dag: &Dag, inputs: &[(&str, Prim, &[i64])]) -> std::process::Output {
    use std::{fs, process::Command};
    let result = support::codegen_hip(dag, "integer_device_test").expect("HIP codegen");
    let dir = tempfile::tempdir().unwrap();
    let runtime = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime");
    fs::copy(
        runtime.join("chelis_hip_runtime.h"),
        dir.path().join("chelis_hip_runtime.h"),
    )
    .unwrap();
    support::stage_device_runtime(dir.path());
    let staged = chelis_runtime_bundle::stage(dir.path()).expect("bundled runtime");
    fs::write(dir.path().join("model.cpp"), &result.c_source).unwrap();
    let mut body = format!("chelis_tensor *inputs[{}] = {{0}};", inputs.len().max(1));
    for (slot, name) in result.input_labels.iter().enumerate() {
        let (_, prim, values) = inputs
            .iter()
            .find(|(label, _, _)| label == name)
            .expect("input fixture");
        let (tag, ty) = match prim {
            Prim::Int8 => ("I8", "int8_t"),
            Prim::Int16 => ("I16", "int16_t"),
            Prim::Int32 => ("I32", "int32_t"),
            Prim::Int64 => ("I64", "int64_t"),
            Prim::F32 => ("F32", "float"),
            _ => unreachable!(),
        };
        body += &format!(
            "int64_t shape_{slot}[] = {{ {} }}; inputs[{slot}] = chelis_alloc(1, shape_{slot}, CHELIS_DTYPE_{tag}); chelis_tensor_write *guard_{slot} = chelis_tensor_begin_write(inputs[{slot}]); {ty} *data_{slot} = ({ty}*)chelis_tensor_write_view(guard_{slot}).data;",
            values.len()
        );
        for (index, value) in values.iter().enumerate() {
            let literal = if *value == i64::MIN {
                "INT64_MIN".to_string()
            } else {
                format!("({value}LL)")
            };
            body += &format!("data_{slot}[{index}] = {literal};");
        }
        body += &format!("chelis_tensor_end_write(guard_{slot});");
    }
    body += &format!(
        "chelis_tensor *outputs[1] = {{0}}; integer_device_test(inputs, {}, outputs, 1); chelis_read_view view = chelis_tensor_read_view(outputs[0]);",
        inputs.len()
    );
    let root = dag.get(dag.roots()[0]).unwrap();
    let count: usize = root
        .output_type
        .dims
        .iter()
        .map(|d| match d {
            DimInfo::Lit(n) => *n,
            _ => panic!("static fixture"),
        })
        .product();
    let bytes = count * element_size(root.output_type.precision);
    body += &format!(
        "for (size_t i=0; i<{bytes}; ++i) printf(\"%02x\", ((const unsigned char*)view.data)[i]); printf(\"\\n\"); chelis_tensor_release(outputs[0]);"
    );
    for slot in 0..inputs.len() {
        body += &format!("chelis_tensor_release(inputs[{slot}]);");
    }
    let driver = format!(
        "#include <stdint.h>\n#include <stdio.h>\n#include \"chelis_runtime.h\"\nextern \"C\" void integer_device_test(chelis_tensor**, int, chelis_tensor**, int);\nint main() {{ {body} return 0; }}"
    );
    fs::write(dir.path().join("driver.cpp"), driver).unwrap();
    let binary = dir.path().join("test");
    let compile = Command::new("hipcc")
        .args(["-O2"])
        .args(&result.compile_flags)
        .arg(dir.path().join("driver.cpp"))
        .arg(dir.path().join("model.cpp"))
        .arg(format!("-I{}", dir.path().display()))
        .arg(dir.path().join("chelis_device_owner.cpp"))
        .arg(&staged.archive)
        .args(["-lpthread", "-ldl"])
        .args(&result.link_flags)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&compile.stderr),
        result.c_source
    );
    Command::new(binary).output().unwrap()
}

#[test]
#[ignore = "manual gate: HIP GPU and hipcc; run via scripts/hip_test.py"]
fn hip_integer_abs_and_gradient_execute_exact_values_and_minimum_traps() {
    for (precision, values, minimum) in [
        (Prim::Int8, vec![-127, 0, 1, 126], -128),
        (Prim::Int16, vec![-32767, 0, 1, 32766], -32768),
        (Prim::Int32, vec![-16777217, 0, 1, 2147483646], -2147483648),
        (
            Prim::Int64,
            vec![-9007199254740993, 0, 1, i64::MAX - 1],
            i64::MIN,
        ),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision,
        };
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "w".into() },
            vec![],
            ty.clone(),
            None,
        );
        let abs = dag.add_node(decl, RiscOp::Abs, vec![input], ty, None);
        dag.set_roots(vec![abs]);
        let result = run_hip(&dag, &[("w", precision, &values)]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let width = element_size(precision);
        let expected = values
            .iter()
            .flat_map(|v| v.abs().to_le_bytes()[..width].to_vec())
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), expected);
        let trap = run_hip(&dag, &[("w", precision, &[0, 1, minimum, minimum])]);
        assert!(!trap.status.success());
        assert_eq!(
            String::from_utf8(trap.stderr).unwrap().trim(),
            format!("numeric trap: overflow in abs at {}", precision.name())
        );
    }
    let dag = gradient_dag();
    let result = run_hip(
        &dag,
        &[
            ("x", Prim::F32, &[1, 2, 3, 4]),
            ("w", Prim::Int64, &[-100, 200, -300, 400]),
        ],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let expected = [100f32, 200., 300., 400.]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), expected);
    let trap = run_hip(
        &dag,
        &[
            ("x", Prim::F32, &[1, 2, 3, 4]),
            ("w", Prim::Int64, &[0, 1, i64::MIN, 2]),
        ],
    );
    assert!(!trap.status.success());
    assert_eq!(
        String::from_utf8(trap.stderr).unwrap().trim(),
        "numeric trap: overflow in abs at i64"
    );
}

#[test]
fn uncanonicalized_integer_rounding_is_rejected_at_the_ir_boundary() {
    for precision in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        for op in [RiscOp::Floor, RiscOp::Ceil, RiscOp::Round] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let ty = TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision,
            };
            let input = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty.clone(),
                None,
            );
            let out = dag.add_node(decl, op, vec![input], ty, None);
            dag.set_roots(vec![out]);
            let error = chelis_ir::ownership::lower_dag_ownership(dag)
                .expect_err("integer rounding must already be canonicalized to identity");
            assert!(error.to_string().contains("requires float input"));
        }
    }
}
