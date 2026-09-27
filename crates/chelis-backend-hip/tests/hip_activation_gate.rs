//! The HIP activation gate (spec/10 section 3.2; #2413): a node whose
//! activation is false is computed, since a `Where` needs its value, but
//! checks nothing. A checking node under an activation (the operations
//! `DagNode::inactive_operand` names) launches a gated kernel: each thread
//! reads the activation once, before any operand, and reads every operand
//! through it, taking the value its checks accept where the element is
//! inactive, exactly the values the evaluator and the C lane substitute. The
//! check and the computation are not wrapped; only what they read is.
//!
//! These are emission tests: `chelis build --target hip` cannot run where
//! this suite runs, so each asserts the generated kernel and launch text.
//! Sources are lowered through the front end and the HIP pipeline's
//! specialization and fusion, as `compile` does for `--target hip`.

mod support;

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use chelis_types::unsupported::Unsupported;

/// Lower `source`'s `main` as `chelis build --target hip` lowers an entry
/// def before codegen.
fn lowered(source: &str) -> Dag {
    let declarations = chelis_surf::parser::parse_str(source).expect("parse source");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("desugar source");
    let checked = chelis_types::check_typed_program(&deep)
        .unwrap_or_else(|errors| panic!("check source: {:?}", errors.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects");
    let checked = chelis_types::check_linearity(&checked).expect("linearity");
    let dag = chelis_ir::host::lower_named_tensor_entry_dag(&checked, "main")
        .expect("`main` lowers as a tensor entry");
    let dag = chelis_ir::optimize::dead_code_eliminate(&dag);
    chelis_ir::fuse::fuse(&chelis_ir::specialize::specialize_for_blas(&dag))
}

fn emit(source: &str) -> Result<String, Unsupported> {
    support::codegen_hip(&lowered(source), "gated").map(|result| result.c_source)
}

/// The kernel source string declared as `{name}_src`, unescaped.
fn kernel(host: &str, name: &str) -> String {
    let start = host
        .find(&format!("const char *{name}_src =\n"))
        .unwrap_or_else(|| panic!("no kernel `{name}` in:\n{host}"));
    host[start..]
        .lines()
        .skip(1)
        .map_while(|line| line.trim_start().strip_prefix('"'))
        .map(|line| {
            line.trim_end_matches(';')
                .trim_end_matches('"')
                .trim_end_matches("\\n")
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The launch block of the kernel `name`: from its launch metadata to the
/// launch itself.
fn launch(host: &str, name: &str) -> String {
    let call = host
        .find(&format!("chelis_launch_kernel(mod_{name}, \"{name}\""))
        .unwrap_or_else(|| panic!("no launch of `{name}` in:\n{host}"));
    let start = host[..call]
        .rfind("void *args[]")
        .expect("launch arguments");
    let block = host[..start].rfind("{\n").expect("launch block");
    host[block..call].to_string()
}

/// The source line of `kernel` holding `needle`.
fn line_with<'a>(kernel: &'a str, needle: &str) -> &'a str {
    kernel
        .lines()
        .find(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("no line with `{needle}` in:\n{kernel}"))
}

const ACTIVATION_READ: &str = "const bool chelis_active = chelis_activation_at(chelis_act, chelis_act_sh, chelis_act_s, chelis_act_ndim, chelis_act_count, chelis_act_row, i);";

/// The gated launch's trailing arguments for node `id` reading activation
/// `act`, in a program specialized to `rank`.
fn gate_arguments(id: usize, act: usize, rank: usize) -> String {
    let refs = |suffix: &str| {
        (0..rank)
            .map(|axis| format!("&t{id}_act_{suffix}{axis}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "&t{id}_size, &p_t{act}, {}, {}, &t{id}_act_ndim, &t{id}_act_count, &t{id}_act_row }};",
        refs("sh"),
        refs("s")
    )
}

/// An `if` over a runtime scalar whose `then` arm applies `op` to `x` and
/// `d`; the `lt` condition keeps the `if` a `Where`, so both arms run.
fn arm(op: &str, element: &str) -> String {
    format!(
        "def main(x: tensor[4, {element}], d: tensor[4, {element}], s: tensor[f32]) -> tensor[4, {element}] = if lt(tensor_to_scalar(s), 0.0f32) then {op}(x, d) else x\n"
    )
}

/// The same operation every execution runs.
fn unconditional(op: &str, element: &str) -> String {
    format!(
        "def main(x: tensor[4, {element}], d: tensor[4, {element}]) -> tensor[4, {element}] = {op}(x, d)\n"
    )
}

/// A rank-0 activation gates an untaken arm's integer `floor_div`: the
/// thread reads the activation once, after its bounds check and before
/// either operand; an inactive element divides 0 by 1; the division itself
/// is not under the test, and the launch passes the activation's bytes, its
/// layout, its count and the elements per row (the whole output, for rank
/// 0).
///
/// HIP's integer `floor_div` kernel has no zero-divisor guard, so a taken
/// arm's division by zero does not trap on this lane at all; the gate still
/// keeps an untaken arm's division well defined.
///
/// Evidentiary status: REGRESSION TEST. At 224414e1f the kernel is the
/// ungated `kernel_floor_div_i32`, whose untaken arm divides by the zero.
#[test]
fn an_untaken_arms_integer_floor_div_reads_its_operands_through_the_activation() {
    let host = emit(&arm("floor_div", "i32")).expect("HIP compiles the arm");
    let name = "kernel_floor_div_i32_gated_0_1";
    let source = kernel(&host, name);
    assert!(
        source.contains("chelis_device_metadata out_size,\n    const unsigned char *chelis_act, chelis_device_metadata chelis_act_sh0, chelis_device_metadata chelis_act_s0, chelis_device_metadata chelis_act_ndim, chelis_device_metadata chelis_act_count, chelis_device_metadata chelis_act_row) {"),
        "{source}"
    );
    let bounds = source.find("if (i >= out_size) return;").unwrap();
    let read = source.find(ACTIVATION_READ).expect("one activation read");
    let first_operand = source.find("a[idx_a]").unwrap();
    assert!(bounds < read && read < first_operand, "{source}");
    assert_eq!(
        source.matches("chelis_activation_at(").count(),
        2,
        "{source}"
    );
    assert_eq!(
        line_with(&source, "an ="),
        "  int32_t an = (chelis_active ? a[idx_a] : (int32_t)(0LL));"
    );
    assert_eq!(
        line_with(&source, "bn ="),
        "  int32_t bn = (chelis_active ? b[idx_b] : (int32_t)(1LL));"
    );
    assert_eq!(line_with(&source, "q = an / bn"), "  int32_t q = an / bn;");
    let launch = launch(&host, name);
    assert!(launch.contains("_act_count = d_t"), "{launch}");
    assert!(
        launch.contains("_act_row = t") && launch.contains("_act_count > 0 ?"),
        "{launch}"
    );
    assert!(launch.contains("_act_count, &t"), "{launch}");
    assert!(
        !host.contains("\"kernel_floor_div_i32\""),
        "the arm launches only the gated kernel:\n{host}"
    );
}

/// The taken-arm twin: a `floor_div` every execution runs has no
/// activation, so its kernel reads its operands directly.
///
/// Evidentiary status: DISPOSITION LOCK (green at 224414e1f).
#[test]
fn an_unconditional_floor_div_reads_its_operands_directly() {
    let host = emit(&unconditional("floor_div", "i32")).expect("HIP compiles the kernel");
    let source = kernel(&host, "kernel_floor_div_i32");
    assert!(!source.contains("chelis_act"), "{source}");
    assert_eq!(line_with(&source, "an ="), "  int32_t an = a[idx_a];");
    assert_eq!(line_with(&source, "bn ="), "  int32_t bn = b[idx_b];");
    assert!(!host.contains("_gated"), "{host}");
}

/// HIP's one checking integer kernel, `sub`'s overflow guard: under an
/// untaken arm's activation the guard reads the gated operands, so an
/// inactive element subtracts 0 from 0 and records nothing, while the
/// guard and the subtraction themselves are unchanged.
///
/// Evidentiary status: REGRESSION TEST. At 224414e1f the kernel is the
/// ungated `kernel_sub_i32`, whose untaken arm records an overflow.
#[test]
fn an_untaken_arms_integer_sub_guard_reads_its_operands_through_the_activation() {
    let host = emit(&arm("sub", "i32")).expect("HIP compiles the arm");
    let name = "kernel_sub_i32_gated_0_0";
    let source = kernel(&host, name);
    assert_eq!(source.matches(ACTIVATION_READ).count(), 1, "{source}");
    assert_eq!(
        line_with(&source, "av ="),
        "  int32_t av = (chelis_active ? a[idx_a] : (int32_t)(0LL));"
    );
    assert_eq!(
        line_with(&source, "bv ="),
        "  int32_t bv = (chelis_active ? b[idx_b] : (int32_t)(0LL));"
    );
    assert!(
        source.contains(
            "  if (overflow) {\n    chelis_record_numeric_failure((unsigned long long)i);"
        ),
        "{source}"
    );
    assert_eq!(
        line_with(&source, "(av - bv)"),
        "  out[i] = (int32_t)(av - bv);"
    );
    let launch = launch(&host, name);
    assert!(launch.contains("_act_count, &t"), "{launch}");
}

/// The taken-arm twin of the `sub` guard: unconditional, ungated.
///
/// Evidentiary status: DISPOSITION LOCK (green at 224414e1f).
#[test]
fn an_unconditional_integer_sub_guard_reads_its_operands_directly() {
    let host = emit(&unconditional("sub", "i32")).expect("HIP compiles the kernel");
    let source = kernel(&host, "kernel_sub_i32");
    assert!(!source.contains("chelis_act"), "{source}");
    assert_eq!(line_with(&source, "av ="), "  int32_t av = a[idx_a];");
    assert!(source.contains("chelis_record_numeric_failure"), "{source}");
}

/// A cast checks its operand only when its target is an integer or `bool`
/// width (`DagNode::runtime_check`), and HIP compiles no such cast, gated or
/// not: its cast kernels cover exactly `f32` and `f64` (chelis#689), and a
/// cast into `bool` needs the one-byte family HIP lacks (chelis#1364). So no
/// cast that can trap reaches a HIP kernel, and each is refused in an arm as
/// it is unconditionally. The one cast family HIP compiles, `f32` to and from
/// `f64`, checks nothing, so an arm launches the same ungated kernel as the
/// unconditional cast (a node with nothing to check takes no gate).
///
/// Evidentiary status: DISPOSITION LOCK throughout. By code read of
/// 224414e1f, which has the same `f32`/`f64` and `bool` rejections and no
/// HIP activation gate, every row holds there too.
#[test]
fn no_cast_hip_compiles_can_trap_so_an_arms_cast_launches_ungated() {
    for (target, refusal) in [("i32", "chelis#689"), ("bool", "chelis#1364")] {
        let in_arm = format!(
            "def main(x: tensor[4, f32], s: tensor[f32], y: tensor[4, {target}]) -> tensor[4, {target}] = if lt(tensor_to_scalar(s), 0.0f32) then cast(x, {target}) else y\n"
        );
        let unconditional =
            format!("def main(x: tensor[4, f32]) -> tensor[4, {target}] = cast(x, {target})\n");
        for source in [in_arm, unconditional] {
            let error = emit(&source).expect_err("HIP refuses a cast that can trap");
            let error = error.to_string();
            assert!(
                error.contains(&format!("dtype `{target}`")) && error.contains(refusal),
                "{error}"
            );
        }
    }
    let widening_arm = "def main(x: tensor[4, f32], s: tensor[f32], y: tensor[4, f64]) -> tensor[4, f64] = if lt(tensor_to_scalar(s), 0.0f32) then cast(x, f64) else y\n";
    let host = emit(widening_arm).expect("HIP compiles the widening arm");
    let source = kernel(&host, "kernel_cast_f32_to_f64");
    assert!(!source.contains("chelis_act"), "{source}");
    assert_eq!(line_with(&source, "out[i] ="), "  out[i] = (double)a[idx];");
    assert!(!host.contains("_gated"), "{host}");
    let host = emit("def main(x: tensor[4, f32]) -> tensor[4, f64] = cast(x, f64)\n")
        .expect("HIP compiles the widening cast");
    let source = kernel(&host, "kernel_cast_f32_to_f64");
    assert_eq!(line_with(&source, "out[i] ="), "  out[i] = (double)a[idx];");
}

/// A per-row activation under `vmap`: the arm's condition is one bool per
/// row, so the launch binds the row width (the node's elements per
/// activation element, 3 here) and each thread reads the byte of its own
/// row, one mask entry per row and never one per element.
///
/// Evidentiary status: REGRESSION TEST. At 224414e1f the batched `sub`
/// launches the ungated kernel and every row's guard checks.
#[test]
fn a_vmapped_arm_reads_one_activation_byte_per_row() {
    let source = "def row(x: tensor[3, i32], d: tensor[3, i32], s: tensor[f32]) -> tensor[3, i32] = if lt(tensor_to_scalar(s), 0.0f32) then sub(x, d) else x\ndef main(xs: tensor[4, 3, i32], ds: tensor[4, 3, i32], ss: tensor[4, f32]) -> tensor[4, 3, i32] = vmap(row)(xs, ds, ss)\n";
    let dag = lowered(source);
    let gated = dag
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Sub))
        .expect("the batched sub");
    let activation = gated.owner.activation.expect("the arm's activation");
    assert_eq!(
        dag.get(activation).unwrap().output_type,
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Bool,
        },
        "one activation byte per row"
    );
    let host = support::codegen_hip(&dag, "gated")
        .expect("HIP compiles the vmapped arm")
        .c_source;
    let name = "kernel_sub_i32_gated_0_0";
    let source = kernel(&host, name);
    assert_eq!(source.matches(ACTIVATION_READ).count(), 1, "{source}");
    assert_eq!(
        line_with(&source, "bv ="),
        "  int32_t bv = (chelis_active ? b[idx_b] : (int32_t)(0LL));"
    );
    let launch = launch(&host, name);
    let id = gated.id.0;
    let act = activation.0;
    assert!(
        launch.contains(&format!(
            "chelis_device_metadata t{id}_act_count = d_t{act}->count;"
        )),
        "{launch}"
    );
    assert!(
        launch.contains(&format!(
            "chelis_device_metadata t{id}_act_row = t{id}_act_count > 0 ? t{id}_size / t{id}_act_count : 0;"
        )),
        "{launch}"
    );
    assert!(launch.contains(&gate_arguments(id, act, 2)), "{launch}");
    assert!(
        host.contains(
            "if (row > 0) return act[chelis_logical_offset(i / row, shape, strides, ndim)] != 0;"
        ),
        "the device helper indexes the row through the activation's strides:\n{host}"
    );
}

/// A `vmap` applied in an untaken arm: the vmapped body runs under the
/// call site's activation, which the batching expands to every row, so the
/// batched `sub`'s activation is a stride-0 view of the arm's one bool. The
/// launch passes the view's shape and strides, and the kernel reads its
/// row's element through them, so every row reads the arm's one byte.
///
/// Evidentiary status: REGRESSION TEST. At 224414e1f the batched `sub`
/// launches the ungated kernel and every row's guard checks.
#[test]
fn a_vmap_in_an_untaken_arm_reads_the_expanded_activation_through_its_strides() {
    let source = "def row(x: tensor[3, i32], d: tensor[3, i32]) -> tensor[3, i32] = sub(x, d)\ndef main(xs: tensor[4, 3, i32], ds: tensor[4, 3, i32], s: tensor[f32]) -> tensor[4, 3, i32] = if lt(tensor_to_scalar(s), 0.0f32) then vmap(row)(xs, ds) else xs\n";
    let dag = lowered(source);
    let gated = dag
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::Sub))
        .expect("the batched sub");
    let activation = gated.owner.activation.expect("the call site's activation");
    let expanded = dag.get(activation).unwrap();
    assert!(
        matches!(expanded.op, RiscOp::Expand { axis: 0, .. }),
        "the call site's activation expanded to every row: {:?}",
        expanded.op
    );
    assert_eq!(
        expanded.output_type,
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Bool,
        }
    );
    let host = support::codegen_hip(&dag, "gated")
        .expect("HIP compiles the arm")
        .c_source;
    let name = "kernel_sub_i32_gated_0_0";
    let source = kernel(&host, name);
    assert_eq!(source.matches(ACTIVATION_READ).count(), 1, "{source}");
    let launch = launch(&host, name);
    let (id, act) = (gated.id.0, activation.0);
    for axis in 0..2 {
        assert!(
            launch.contains(&format!(
                "chelis_device_metadata t{id}_act_s{axis} = ({axis} < d_t{act}->rank) ? d_t{act}->strides[{axis}] : 0;"
            )),
            "{launch}"
        );
    }
    assert!(launch.contains(&gate_arguments(id, act, 2)), "{launch}");
}

/// Fail-closed for the kinds whose check HIP cannot gate (decisions section
/// 11; `TrapSeeds::is_activation_gated`): an integer reduction's overflow check
/// (HIP's one integer `sum`, the `i16` sum promoted to `i32`) and an extreme reduction's empty-axis check under an untaken arm's
/// activation are refused, with the gate's `unimplemented chelis#2413`
/// rejection, rather than launched with a check that would run where the
/// activation is false. The same reduction every execution runs has no
/// activation and compiles.
///
/// Evidentiary status: REGRESSION TEST for the refusals. At 224414e1f HIP
/// launches both arms' reductions ungated. The unconditional twins are a
/// disposition lock (they compile at 224414e1f).
#[test]
fn an_untaken_arms_reduction_check_is_refused_and_its_unconditional_twin_compiles() {
    for (kind, arm, unconditional) in [
        (
            "integer sum",
            "def main(x: tensor[4, i16], y: tensor[i32], s: tensor[f32]) -> tensor[i32] = if lt(tensor_to_scalar(s), 0.0f32) then sum(x, 0i32) else y\n",
            "def main(x: tensor[4, i16]) -> tensor[i32] = sum(x, 0i32)\n",
        ),
        (
            "empty-axis max_reduce",
            "def main(e: tensor[0, f32], y: tensor[f32], s: tensor[f32]) -> tensor[f32] = if lt(tensor_to_scalar(s), 0.0f32) then max_reduce(e, 0i32) else y\n",
            "def main(e: tensor[0, f32]) -> tensor[f32] = max_reduce(e, 0i32)\n",
        ),
    ] {
        let dag = lowered(arm);
        let seeds = dag.trap_seeds();
        let gated = dag
            .nodes()
            .iter()
            .filter(|node| seeds.is_activation_gated(node))
            .map(|node| chelis_ir::grad::risc_op_name(&node.op))
            .collect::<Vec<_>>();
        assert!(
            gated
                .iter()
                .any(|op| op.contains("sum") || op.contains("max")),
            "{kind}: the arm's reduction is gated: {gated:?}"
        );
        let error = support::codegen_hip(&dag, "gated")
            .map(|result| result.c_source)
            .expect_err(kind);
        let message = error.to_string();
        assert!(
            message.contains("a checking operation under an activation")
                && message.contains("2413"),
            "{kind}: {message}"
        );
        emit(unconditional)
            .unwrap_or_else(|error| panic!("{kind}: the unconditional twin: {error}"));
    }
}

/// A restamp (chelis#2512) under an activation is refused: where no row of
/// its activation holds, the evaluator and the C lane give it zeros of its
/// declared type without reading its operand, and this lane has no such
/// value. `x: [n]` is restamped `[m]` under `a` by `neg(x)` or `add(x, x)`
/// and added to `z: [m]`, so its class guards the restamp. The float `neg`
/// checks nothing else; the integer `add` also checks its operand values,
/// which this lane could otherwise gate.
///
/// Evidentiary status: REGRESSION TEST. At c18888f34 both compile: the
/// float `neg` is not gated, and the integer `add` takes the operand gate.
#[test]
fn an_untaken_arms_restamp_is_refused() {
    for (op, precision) in [(RiscOp::Neg, Prim::F32), (RiscOp::Add, Prim::Int32)] {
        let named = |name: &str| TensorType {
            dims: vec![DimInfo::Named(name.into(), None)],
            precision,
        };
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let load = |dag: &mut Dag, name: &str, ty: TensorType| {
            dag.add_node(decl, RiscOp::Load { name: name.into() }, vec![], ty, None)
        };
        let x = load(&mut dag, "x", named("n"));
        let z = load(&mut dag, "z", named("m"));
        let a = load(
            &mut dag,
            "a",
            TensorType {
                dims: Vec::new(),
                precision: Prim::Bool,
            },
        );
        let under_a = chelis_ir::dag::Owner::new(decl, Some(a));
        let operands = if matches!(op, RiscOp::Neg) {
            vec![x]
        } else {
            vec![x, x]
        };
        let restamped = dag.add_node(under_a, op, operands, named("m"), None);
        let plus = dag.add_node(under_a, RiscOp::Add, vec![restamped, z], named("m"), None);
        dag.add_root(plus);
        let message = support::codegen_hip(&dag, "gated")
            .map(|result| result.c_source)
            .expect_err("a restamp under an activation")
            .to_string();
        assert!(
            message.contains("a checking operation under an activation")
                && message.contains(&format!("DAG node {}", restamped.0))
                && message.contains("2413"),
            "{precision:?}: {message}"
        );
    }
}
