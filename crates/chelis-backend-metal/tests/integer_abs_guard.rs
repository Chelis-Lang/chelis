mod support;
use chelis_ir::dag::{Dag, DimInfo, FusedInput, FusedStep, FusedStepOp, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::try_codegen_metal;

fn vec_i64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int64,
    }
}

#[test]
fn integer_abs_uses_checked_kernel_and_fused_abs_stays_rejected() {
    let ty = vec_i64(1);

    let mut direct = Dag::new();
    let direct_decl = direct.declare("test");
    let x = direct.add_node(
        direct_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let out = direct.add_node(direct_decl, RiscOp::Abs, vec![x], ty.clone(), None);
    direct.set_roots(vec![out]);
    let code = try_codegen_metal(&direct, "integer_abs").expect("typed integer abs");
    assert!(
        code.mm_source
            .contains("numeric trap: overflow in abs at i64")
    );
    assert!(code.mm_source.contains("atomic_fetch_or_explicit"));
    assert!(!code.mm_source.contains("fabs("));

    let mut fused = Dag::new();
    let fused_decl = fused.declare("test");
    let x = fused.add_node(
        fused_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let out = fused.add_node(
        fused_decl,
        RiscOp::FusedElem {
            ops: vec![FusedStep {
                op: FusedStepOp::Abs,
                input_indices: vec![FusedInput::External(0)],
            }],
        },
        vec![x],
        ty,
        None,
    );
    fused.set_roots(vec![out]);
    let error = try_codegen_metal(&fused, "fused_integer_abs").unwrap_err();
    assert!(error.to_string().contains("unsupported: op `Abs`"));
}

#[test]
fn integer_abs_then_float_cast_has_distinct_typed_kernels_at_every_width() {
    for precision in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        for target in [Prim::F16, Prim::Bf16, Prim::F32] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let ty = TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision,
            };
            let x = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty.clone(),
                None,
            );
            let abs = dag.add_node(decl, RiscOp::Abs, vec![x], ty, None);
            let cast = dag.add_node(
                decl,
                RiscOp::Cast {
                    new_precision: target,
                },
                vec![abs],
                TensorType {
                    dims: vec![DimInfo::Lit(4)],
                    precision: target,
                },
                None,
            );
            dag.set_roots(vec![cast]);
            let result = try_codegen_metal(&dag, "abs_cast").expect("typed abs then explicit cast");
            assert!(result.mm_source.contains("chelis_cast_bits"));
            assert!(!result.mm_source.contains("fabs("));
            assert!(result.mm_source.contains(&format!(
                "numeric trap: overflow in abs at {}",
                precision.name()
            )));
        }
    }
}

#[test]
fn lowered_integer_abs_gradient_reaches_typed_metal_kernels() {
    let dag = gradient_dag();
    let result = try_codegen_metal(&dag, "integer_gradient")
        .unwrap_or_else(|error| panic!("{error}\n{:#?}", dag.nodes()));
    assert!(
        result
            .mm_source
            .contains("numeric trap: overflow in abs at i64")
    );
}

#[test]
fn source_grad_and_vmap_bitwise_coefficients_reach_metal_kernels() {
    let grad = lower_gradient_source(
        "def loss(x: tensor[4, f32], w: tensor[4, i64], n: tensor[4, i64]) -> tensor[f32] = sum(mul(x, cast(bitxor(w, n), f32)), 0i32)\n\
         def main(x: tensor[4, f32], w: tensor[4, i64], n: tensor[4, i64]) -> tensor[4, f32] = (grad(loss, wrt=x))(x, w, n)\n",
    );
    assert!(grad.nodes().iter().any(|node| matches!(node.op, RiscOp::Bitwise(chelis_types::BitwiseKind::Xor))));
    let grad_source = try_codegen_metal(&grad, "bitwise_grad").expect("source grad device lowering").mm_source;
    assert!(grad_source.contains("k_bitwise_bitxor_i64_"));

    let mapped = lower_gradient_source(
        "def row(w: tensor[i64], n: tensor[i64]) -> tensor[i64] = shl(w, n)\n\
         def main(w: tensor[4, i64], n: tensor[4, i64]) -> tensor[4, i64] = vmap(row)(w, n)\n",
    );
    assert!(mapped.nodes().iter().any(|node| matches!(node.op, RiscOp::Bitwise(chelis_types::BitwiseKind::ShiftLeft))));
    let mapped_source = try_codegen_metal(&mapped, "bitwise_vmap").expect("source vmap device lowering").mm_source;
    assert!(mapped_source.contains("k_bitwise_shl_i64_"));

    let control = lower_gradient_source(
        "def row(x: tensor[f32], y: tensor[f32]) -> tensor[f32] = mul(x, y)\n\
         def main(x: tensor[4, f32], y: tensor[4, f32]) -> tensor[4, f32] = vmap(row)(x, y)\n",
    );
    assert!(!control.nodes().iter().any(|node| matches!(node.op, RiscOp::Bitwise(_))));
    try_codegen_metal(&control, "non_bitwise_vmap").expect("non-bitwise control");
}

fn gradient_dag() -> Dag {
    let source = "def loss(x: tensor[4, f32], w: tensor[4, i64]) -> tensor[f32] = sum(mul(x, cast(abs(w), f32)), 0i32)\ndef main(x: tensor[4, f32], w: tensor[4, i64]) -> tensor[4, f32] = (grad(loss, wrt=x))(x, w)\n";
    lower_gradient_source(source)
}

fn literal_gradient_dag(op: &str) -> Dag {
    lower_gradient_source(&format!(
        "def loss(x: tensor[4, f32]) -> tensor[f32] = {{\n w = cast({op}(to_tensor([-100i64, 200i64, -300i64, 400i64])), f32)\n sum(mul(x, w), 0i32)\n}}\ndef main(x: tensor[4, f32]) -> tensor[4, f32] = (grad(loss, wrt=x))(x)\n"
    ))
}

fn lower_gradient_source(source: &str) -> Dag {
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

#[cfg(target_os = "macos")]
fn run_metal(dag: &Dag, inputs: &[(&str, Prim, &[i64])]) -> std::process::Output {
    use std::{fs, process::Command};
    let result = try_codegen_metal(dag, "integer_device_test").expect("Metal codegen");
    let dir = tempfile::tempdir().unwrap();
    let runtime = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("runtime");
    fs::copy(
        runtime.join("chelis_metal_runtime.h"),
        dir.path().join("chelis_metal_runtime.h"),
    )
    .unwrap();
    let staged = chelis_runtime_bundle::stage(dir.path()).expect("bundled runtime");
    fs::write(dir.path().join("model.mm"), &result.mm_source).unwrap();
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
    body += &format!(
        "if (view.count != {count} || chelis_tensor_rank(outputs[0]) != {}) return 90; if (view.dtype != {}) return 91;",
        root.output_type.dims.len(),
        chelis_backend_metal::dtype::runtime_dtype_tag(root.output_type.precision)
    );
    for (axis, dim) in root.output_type.dims.iter().enumerate() {
        let DimInfo::Lit(extent) = dim else {
            unreachable!()
        };
        body += &format!("if (chelis_tensor_shape(outputs[0], {axis}) != {extent}) return 92;");
    }
    let bytes = count * chelis_backend_metal::dtype::metal_elem_size(root.output_type.precision);
    body += &format!(
        "for (size_t i=0; i<{bytes}; ++i) printf(\"%02x\", ((const unsigned char*)view.data)[i]); printf(\"\\n\"); chelis_tensor_release(outputs[0]);"
    );
    for slot in 0..inputs.len() {
        body += &format!("chelis_tensor_release(inputs[{slot}]);");
    }
    let driver = format!(
        "#import <Foundation/Foundation.h>\n#include <stdint.h>\n#include <stdio.h>\n#include \"chelis_runtime.h\"\nextern \"C\" void integer_device_test(chelis_tensor**, int, chelis_tensor**, int);\nint main() {{ @autoreleasepool {{ {body} }} return 0; }}"
    );
    fs::write(dir.path().join("driver.mm"), driver).unwrap();
    let binary = dir.path().join("test");
    let compile = Command::new("xcrun")
        .args(["-sdk", "macosx", "clang++", "-O2"])
        .args(&result.compile_flags)
        .arg(dir.path().join("driver.mm"))
        .arg(dir.path().join("model.mm"))
        .arg(format!("-I{}", dir.path().display()))
        .arg(&staged.archive)
        .args(&result.link_flags)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&compile.stderr),
        result.mm_source
    );
    Command::new(binary).output().unwrap()
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "manual gate: Apple Silicon Metal device and Xcode command-line tools"]
fn metal_integer_abs_and_gradient_execute_exact_values_and_minimum_traps() {
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
        let result = run_metal(&dag, &[("w", precision, &values)]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let width = chelis_backend_metal::dtype::metal_elem_size(precision);
        let expected = values
            .iter()
            .flat_map(|v| v.abs().to_le_bytes()[..width].to_vec())
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), expected);
        let trap = run_metal(&dag, &[("w", precision, &[0, 1, minimum, minimum])]);
        assert!(!trap.status.success());
        assert_eq!(
            String::from_utf8(trap.stderr).unwrap().trim(),
            format!("numeric trap: overflow in abs at {}", precision.name())
        );
    }
    let dag = gradient_dag();
    let result = run_metal(
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
    let trap = run_metal(
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
fn literal_integer_unary_gradients_lower_without_zero_placeholders() {
    for op in ["abs", "floor", "ceil", "round"] {
        let dag = literal_gradient_dag(op);
        try_codegen_metal(&dag, "literal_gradient").expect("literal integer gradient");
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "manual gate: Apple Silicon Metal device and Xcode command-line tools"]
fn metal_literal_gradients_and_integer_float_rounding_execute() {
    use chelis_types::{
        ElementRef,
        dtype_semantics::{cast_scalar, scalar_from_i64},
    };
    for op in ["abs", "floor", "ceil", "round"] {
        let result = run_metal(
            &literal_gradient_dag(op),
            &[("x", Prim::F32, &[1, 2, 3, 4])],
        );
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let values = if op == "abs" {
            [100f32, 200., 300., 400.]
        } else {
            [-100f32, 200., -300., 400.]
        };
        let expected = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(
            String::from_utf8(result.stdout).unwrap().trim(),
            expected,
            "{op}"
        );
    }
    let values = [
        4_629_700_416_936_869_889i64,
        9_007_199_254_740_993,
        i64::MIN,
        65520,
    ];
    for precision in [Prim::F16, Prim::Bf16, Prim::F32] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "w".into() },
            vec![],
            vec_i64(4),
            None,
        );
        let cast = dag.add_node(
            decl,
            RiscOp::Cast {
                new_precision: precision,
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision,
            },
            None,
        );
        dag.set_roots(vec![cast]);
        let result = run_metal(&dag, &[("w", Prim::Int64, &values)]);
        assert!(
            result.status.success(),
            "{precision:?}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let expected = values
            .iter()
            .flat_map(|value| {
                match cast_scalar(
                    "cast",
                    scalar_from_i64("test", Prim::Int64, *value).unwrap(),
                    precision,
                )
                .unwrap()
                .element_ref()
                {
                    ElementRef::F16(v) => v.to_bits().to_le_bytes().to_vec(),
                    ElementRef::Bf16(v) => v.to_bits().to_le_bytes().to_vec(),
                    ElementRef::F32(v) => v.to_bits().to_le_bytes().to_vec(),
                    _ => unreachable!(),
                }
            })
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(
            String::from_utf8(result.stdout).unwrap().trim(),
            expected,
            "{precision:?}"
        );
    }
}

#[test]
fn activated_integer_abs_is_refused_before_emission_without_a_gate() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let active = dag.add_node(
        decl,
        RiscOp::Load {
            name: "active".into(),
        },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        },
        None,
    );
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "w".into() },
        vec![],
        vec_i64(4),
        None,
    );
    let abs = dag.add_node(decl, RiscOp::Abs, vec![input], vec_i64(4), None);
    dag.node_mut(abs).unwrap().owner.activation = Some(active);
    dag.set_roots(vec![abs]);
    let error = try_codegen_metal(&dag, "gated_abs").unwrap_err();
    assert_eq!(
        error.authority.issue().map(|issue| issue.number()),
        Some(693)
    );
    assert!(
        error
            .to_string()
            .contains("Metal has no operand activation gate")
    );
}

#[test]
fn exact_integer_constants_and_abs_cover_scalar_empty_and_rank_two_shapes() {
    for dims in [
        vec![],
        vec![DimInfo::Lit(0)],
        vec![DimInfo::Lit(2), DimInfo::Lit(2)],
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims,
            precision: Prim::Int64,
        };
        let constant = dag.add_node(
            decl,
            RiscOp::Const {
                value: chelis_types::dtype_semantics::scalar_from_i64(
                    "test",
                    Prim::Int64,
                    -9_007_199_254_740_993,
                )
                .unwrap(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let abs = dag.add_node(decl, RiscOp::Abs, vec![constant], ty, None);
        dag.set_roots(vec![abs]);
        let result = try_codegen_metal(&dag, "constant_abs").expect("exact constant abs");
        assert!(result.mm_source.contains("-9007199254740993LL"));
        assert!(!result.mm_source.contains("-9007199254740992"));
    }
}

fn constant_operation(shape: &[usize], value: i64, op: RiscOp) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let input_type = TensorType {
        dims: shape.iter().copied().map(DimInfo::Lit).collect(),
        precision: Prim::Int64,
    };
    let input = dag.add_node(
        decl,
        RiscOp::Const {
            value: chelis_types::scalar_from_i64("test", Prim::Int64, value).unwrap(),
        },
        vec![],
        input_type.clone(),
        None,
    );
    let mut output_type = input_type;
    match &op {
        RiscOp::Cast { new_precision } => output_type.precision = *new_precision,
        RiscOp::Expand {
            axis,
            size: chelis_ir::dag::RtDim::Lit(size),
        } => {
            output_type.dims.insert(*axis, DimInfo::Lit(*size));
        }
        RiscOp::Abs => {}
        _ => panic!("unexpected fixture operation"),
    }
    let output = dag.add_node(decl, op, vec![input], output_type, None);
    dag.set_roots(vec![output]);
    dag
}

#[test]
fn empty_pointwise_dispatches_are_omitted_without_omitting_output_allocations() {
    for op in [
        RiscOp::Abs,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
    ] {
        let empty =
            try_codegen_metal(&constant_operation(&[0], i64::MIN, op.clone()), "empty").unwrap();
        assert!(!empty.mm_source.contains("chelis_metal_launch("), "{op:?}");
        assert!(
            empty.mm_source.contains("buf_1 = chelis_metal_alloc(0u *"),
            "{op:?}"
        );
        let nonempty =
            try_codegen_metal(&constant_operation(&[1], 1, op.clone()), "nonempty").unwrap();
        assert!(
            nonempty.mm_source.contains("chelis_metal_launch("),
            "{op:?}"
        );
    }
}

#[cfg(target_os = "macos")]
fn assert_metal_bytes(dag: &Dag, expected: &[u8]) {
    let result = run_metal(dag, &[]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let expected = expected
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), expected);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "manual gate: Apple Silicon Metal device and Xcode command-line tools"]
fn metal_empty_abs_preserves_shape_without_trapping_and_nonempty_min_traps() {
    for shape in [&[0][..], &[2, 0][..]] {
        // MIN is not an element of an empty constant; there is no numeric trap.
        assert_metal_bytes(&constant_operation(shape, i64::MIN, RiscOp::Abs), &[]);
    }
    assert_metal_bytes(
        &constant_operation(&[], -9_007_199_254_740_993, RiscOp::Abs),
        &9_007_199_254_740_993i64.to_le_bytes(),
    );
    let trap = run_metal(&constant_operation(&[1], i64::MIN, RiscOp::Abs), &[]);
    assert!(!trap.status.success());
    assert_eq!(
        String::from_utf8(trap.stderr).unwrap().trim(),
        "numeric trap: overflow in abs at i64"
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "manual gate: Apple Silicon Metal device and Xcode command-line tools"]
fn metal_empty_integer_float_cast_preserves_shape_and_nonempty_conversion_executes() {
    let op = RiscOp::Cast {
        new_precision: Prim::F32,
    };
    for shape in [&[0][..], &[0, 2][..]] {
        assert_metal_bytes(&constant_operation(shape, i64::MIN, op.clone()), &[]);
    }
    // Empty input still requires a reduction dispatch to produce its scalar
    // identity. Launch planning uses kernel work, not the input element count.
    let mut reduction = constant_operation(&[0], i64::MIN, op.clone());
    let input = reduction.roots()[0];
    let decl = reduction.declare("reduction");
    let output = reduction.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![input],
        TensorType::scalar_f32(),
        None,
    );
    reduction.set_roots(vec![output]);
    assert_metal_bytes(&reduction, &0f32.to_le_bytes());
    assert_metal_bytes(
        &constant_operation(&[1], -123, op),
        &(-123f32).to_le_bytes(),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "manual gate: Apple Silicon Metal device and Xcode command-line tools"]
fn metal_empty_expand_preserves_inserted_axis_and_nonempty_replication_executes() {
    let op = RiscOp::Expand {
        axis: 0,
        size: chelis_ir::dag::RtDim::Lit(2),
    };
    assert_metal_bytes(&constant_operation(&[0], 1, op.clone()), &[]);
    // A zero inserted axis also produces empty work from a nonempty source.
    assert_metal_bytes(
        &constant_operation(
            &[2],
            1,
            RiscOp::Expand {
                axis: 1,
                size: chelis_ir::dag::RtDim::Lit(0),
            },
        ),
        &[],
    );
    assert_metal_bytes(
        &constant_operation(&[1], -9_007_199_254_740_993, op),
        &(-9_007_199_254_740_993i64).to_le_bytes().repeat(2),
    );
}
