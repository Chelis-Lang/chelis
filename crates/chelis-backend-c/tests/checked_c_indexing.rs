//! Bounded shared-indexing adoption control; native execution lives in
//! exec_compile::checked_c_indexing_* and the existing dtype/cast/reuse suites.

const DAG_METHODS: [&str; 18] = [
    "emit_binary",
    "emit_floor_div",
    "emit_floor_div_reduced_f",
    "emit_binary_reduced_f",
    "emit_binary_func",
    "emit_cmplt",
    "emit_unary",
    "emit_integer_abs",
    "emit_recip",
    "emit_unary_func",
    "emit_unary_reduced_f",
    "emit_recip_reduced_f",
    "emit_binary_func_reduced_f",
    "emit_extrema_adjoint",
    "emit_unary_func_reduced_f",
    "emit_fused_elem",
    "emit_realize",
    "emit_cast",
];
const HOST_ARMS: [&str; 4] = [
    "emit_binary_elementwise_arm",
    "emit_binary_func_elementwise_arm",
    "emit_unary_elementwise_arm",
    "emit_unary_func_elementwise_arm",
];

fn run_projection_methods(dag: &str, host: &str) -> std::process::Output {
    use std::{fs, process::Command};
    let probe = tempfile::tempdir().unwrap();
    let source = format!(
        r#"
use std::collections::BTreeSet;
struct NodeId(usize);
struct TensorType;
#[derive(Default)]
struct Emitter {{ lines: Vec<String>, indent: String }}
impl Emitter {{
    fn line(&mut self, line: &str) {{ self.lines.push(line.to_string()); }}
    fn ndim(_: &TensorType) -> usize {{ 2 }}
    fn tagged_shape_literal(_: &TensorType) -> String {{ "tagged_shape".into() }}
    {dag}
    {host}
}}
fn main() {{
    let mut emitter = Emitter::default();
    let condition = emitter.emit_elementwise_index_steps(7, &[NodeId(2), NodeId(1), NodeId(2)], &TensorType);
    assert_eq!(emitter.lines, [
        "const int64_t t7_input1_step = chelis_tensor_elementwise_index_step_for_shape(t1, chelis_scalar_from_bits(CHELIS_DTYPE_I64, 2), tagged_shape);",
        "const int64_t t7_input2_step = chelis_tensor_elementwise_index_step_for_shape(t2, chelis_scalar_from_bits(CHELIS_DTYPE_I64, 2), tagged_shape);",
    ], "checked DAG projection authority");
    assert_eq!(condition, "t7_size <= 1 || (t7_input1_step == 1 && t7_input2_step == 1)", "identity guard authority");
    emitter.lines.clear();
    emitter.emit_elementwise_index_step("out", "rhs", "right", "left");
    assert_eq!(emitter.lines, ["const int64_t out_rhs_step = chelis_tensor_elementwise_index_step(right, left);"], "checked host projection authority");
}}
"#
    );
    let input = probe.path().join("projection.rs");
    let binary = probe.path().join("projection");
    fs::write(&input, source).unwrap();
    let compiled = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args(["--edition=2024", "-O"])
        .arg(input)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    Command::new(binary).output().unwrap()
}

#[test]
fn final_projection_owners_reject_raw_calculations_constant_steps_and_weak_identity() {
    let dag = method(
        include_str!("../src/emit.rs"),
        "emit_elementwise_index_steps",
    );
    let host = method(
        include_str!("../src/host_emit.rs"),
        "emit_elementwise_index_step",
    );
    let baseline = run_projection_methods(dag, host);
    assert!(
        baseline.status.success(),
        "{}",
        String::from_utf8_lossy(&baseline.stderr)
    );
    for (from, to, message) in [
        (
            "chelis_tensor_elementwise_index_step_for_shape(t{input}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {shape})",
            "(t{input}_rank == 0 ? 0 : 1) /* {rank} {shape} */",
            "checked DAG projection authority",
        ),
        (
            "t{id}_size <= 1 || ({})",
            "1 || ({}) /* {id} */",
            "identity guard authority",
        ),
    ] {
        assert_eq!(dag.matches(from).count(), 1);
        let run = run_projection_methods(&dag.replace(from, to), host);
        assert!(!run.status.success());
        assert!(String::from_utf8_lossy(&run.stderr).contains(message));
    }
    let from = "chelis_tensor_elementwise_index_step({input}, {domain})";
    assert_eq!(host.matches(from).count(), 1);
    let run = run_projection_methods(dag, &host.replace(from, "1 /* {input} {domain} */"));
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("checked host projection authority"));
}

fn method<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(&format!("    fn {name}(")).expect(name);
    let rest = &source[start..];
    let end = rest.find("\n    }\n").expect("method end") + 7;
    &rest[..end]
}

fn check_dag(source: &str) -> Result<(), String> {
    for name in DAG_METHODS {
        let body = method(source, name);
        for retired in [
            "chelis_flat_to_indices(",
            "chelis_indices_to_flat(",
            "int64_t indices[",
        ] {
            if body.contains(retired) {
                return Err(format!("{name}: retired coordinate arithmetic"));
            }
        }
        let projection = body
            .find("self.emit_elementwise_index_steps(id, inputs, ty)")
            .ok_or_else(|| format!("{name}: missing checked projection"))?;
        let allocation = [
            "self.emit_slot_wrapper(",
            "self.emit_fused_in_place_wrapper(",
        ]
        .into_iter()
        .filter_map(|anchor| body.find(anchor))
        .min()
        .expect("allocation");
        if projection >= allocation {
            return Err(format!("{name}: projection after repurpose/allocation"));
        }
        for line in body.lines().filter(|line| {
            line.contains("\"if (chelis_is_contiguous(")
                || line.contains("\"if (({contiguity_cond})")
        }) {
            if !line.contains("{identity}") {
                return Err(format!("{name}: scalar admitted to fast path"));
            }
        }
        if !body.contains("i < t{id}_size") {
            return Err(format!("{name}: missing checked loop bound"));
        }
        if !body.contains("_step") {
            return Err(format!("{name}: missing projected input index"));
        }
    }
    Ok(())
}

#[test]
fn shared_indexing_cohort_has_checked_projections_before_storage_changes() {
    let source = include_str!("../src/emit.rs");
    check_dag(source).unwrap();
    let host = include_str!("../src/host_emit.rs");
    assert!(!host.contains("chelis_host_flat_to_indices("));
    assert!(!host.contains("chelis_host_indices_to_flat("));
    assert!(
        host.contains("chelis_host_tensor_stride("),
        "BLAS stride observer remains needed"
    );
    for name in HOST_ARMS {
        let body = method(host, name);
        assert!(body.contains("_step"), "{name} projects input indices");
        assert!(
            body.contains("{target_view}.count"),
            "{name} uses checked output count"
        );
    }
    for name in [
        "assign_checked_tensor_cast",
        "assign_tensor_binary_elementwise",
        "assign_tensor_binary_func_elementwise",
        "assign_tensor_unary_elementwise",
        "assign_tensor_unary_func_elementwise",
    ] {
        let body = method(host, name);
        assert!(
            body.find("self.emit_elementwise_index_step(").expect(name)
                < body.find("chelis_host_alloc_like(").expect(name),
            "{name} validates before allocation"
        );
    }
}

#[test]
fn cohort_control_rejects_raw_helpers_late_validation_and_unchecked_loop_bounds() {
    let source = include_str!("../src/emit.rs");
    check_dag(source).unwrap();
    for (from, to, reason) in [
        (
            "int64_t idx_a = i * t{id}_input{a}_step;",
            "int64_t idx_a = chelis_indices_to_flat(indices, t{a}_strides, t{a}_rank);",
            "retired coordinate arithmetic",
        ),
        (
            "let identity = self.emit_elementwise_index_steps(id, inputs, ty);",
            "let identity = String::from(\"1\");",
            "missing checked projection",
        ),
        (
            "i < t{id}_size",
            "i < t{a}_size",
            "missing checked loop bound",
        ),
        (
            "\"if (chelis_is_contiguous(t{a}) && ({identity})",
            "\"if (chelis_is_contiguous(t{a})",
            "scalar admitted to fast path",
        ),
    ] {
        assert!(source.contains(from), "production mutation anchor: {from}");
        assert!(
            check_dag(&source.replace(from, to))
                .unwrap_err()
                .contains(reason)
        );
    }
    let early = "let identity = self.emit_elementwise_index_steps(id, inputs, ty);\n        self.emit_slot_wrapper(id, ty);";
    let late = "self.emit_slot_wrapper(id, ty);\n        let identity = self.emit_elementwise_index_steps(id, inputs, ty);";
    assert!(source.contains(early));
    assert!(
        check_dag(&source.replacen(early, late, 1))
            .unwrap_err()
            .contains("projection after repurpose/allocation")
    );
    let guard = "if (({contiguity_cond}) && ({identity}))";
    assert!(source.contains(guard));
    assert!(
        check_dag(&source.replacen(guard, "if (({contiguity_cond}))", 1))
            .unwrap_err()
            .contains("scalar admitted to fast path")
    );
}
