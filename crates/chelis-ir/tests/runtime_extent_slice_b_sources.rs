//! Slice B executable contract for chelis#1277: one checked source per
//! realized output axis (`spec/design/runtime_extents.md` C4).

use chelis_ir::axis_sources::{
    AxisSource, ExtentOrigin, check_axis_sources, dim_extent_origins, output_axis_sources,
    resolve_axis_extent, unresolved_dim_names,
};
use chelis_ir::dag::{
    Dag, DimExpr, DimInfo, ExtremaKind, ExtremaOperand, FusedInput, FusedStep, FusedStepOp, NodeId,
    ReduceWindowKind, RiscOp, RtAxis, RtDim, TensorType,
};
use chelis_types::types::Prim;
use chelis_types::unsupported::Stage;

fn ty(dims: Vec<DimInfo>, precision: Prim) -> TensorType {
    TensorType { dims, precision }
}

fn scalar(precision: Prim) -> TensorType {
    ty(vec![], precision)
}

fn named(name: &str) -> DimInfo {
    DimInfo::Named(name.to_string(), None)
}

fn load(dag: &mut Dag, name: &str, dims: Vec<DimInfo>) -> NodeId {
    dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        ty(dims, Prim::F32),
        None,
    )
}

fn input_axis(input: usize, axis: i32) -> AxisSource {
    AxisSource::InputAxis {
        input,
        axis: RtAxis::Lit(axis),
    }
}

/// C4.2: rank-increasing `Expand` maps axes before the insertion unchanged,
/// the inserted axis to its size, and every later output axis to input axis
/// `output_axis - 1`.
#[test]
fn expand_insert_maps_later_output_axes_to_input_minus_one() {
    let mut dag = Dag::new();
    let operand = load(&mut dag, "x", vec![DimInfo::Lit(2), DimInfo::Lit(3)]);
    let inserted = dag.add_node(
        RiscOp::Expand {
            axis: 1,
            size: RtDim::Lit(4),
        },
        vec![operand],
        ty(
            vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(3)],
            Prim::F32,
        ),
        None,
    );

    assert_eq!(
        output_axis_sources(&dag, inserted),
        vec![
            input_axis(0, 0),
            AxisSource::Literal { value: 4 },
            input_axis(0, 1),
        ],
        "the inserted axis takes the size and axis 2 reads input axis 1"
    );

    // Negative parity: a same-rank `Expand` replaces one axis and leaves the
    // others on their own input axis, with no shift.
    let replaced = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(5),
        },
        vec![operand],
        ty(vec![DimInfo::Lit(5), DimInfo::Lit(3)], Prim::F32),
        None,
    );
    assert_eq!(
        output_axis_sources(&dag, replaced),
        vec![AxisSource::Literal { value: 5 }, input_axis(0, 1)],
    );
}

/// C4.2: only `Stride` with a literal step of one and `Pad` with literal
/// zero padding pass an input axis through.
#[test]
fn identity_stride_one_and_zero_pad_pass_the_input_axis_through() {
    let mut dag = Dag::new();
    let operand = load(&mut dag, "x", vec![DimInfo::Lit(6), DimInfo::Lit(4)]);

    let identity_stride = dag.add_node(
        RiscOp::Stride {
            strides: vec![RtDim::Lit(1), RtDim::Lit(1)],
        },
        vec![operand],
        ty(vec![DimInfo::Lit(6), DimInfo::Lit(4)], Prim::F32),
        None,
    );
    assert_eq!(
        output_axis_sources(&dag, identity_stride),
        vec![input_axis(0, 0), input_axis(0, 1)],
    );

    let zero_pad = dag.add_node(
        RiscOp::zero_pad(
            Prim::F32,
            vec![
                (RtDim::Lit(0), RtDim::Lit(0)),
                (RtDim::Lit(0), RtDim::Lit(0)),
            ],
        ),
        vec![operand],
        ty(vec![DimInfo::Lit(6), DimInfo::Lit(4)], Prim::F32),
        None,
    );
    assert_eq!(
        output_axis_sources(&dag, zero_pad),
        vec![input_axis(0, 0), input_axis(0, 1)],
    );

    // Negative parity: a step of two and a nonzero pad are fresh extents,
    // never the input axis's runtime dim.
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2), RtDim::Lit(1)],
        },
        vec![operand],
        ty(vec![DimInfo::Lit(3), DimInfo::Lit(4)], Prim::F32),
        None,
    );
    assert_eq!(
        output_axis_sources(&dag, strided),
        vec![
            AxisSource::OpComputed {
                op: strided,
                axis: 0
            },
            input_axis(0, 1),
        ],
    );

    let padded = dag.add_node(
        RiscOp::zero_pad(
            Prim::F32,
            vec![
                (RtDim::Lit(1), RtDim::Lit(0)),
                (RtDim::Lit(0), RtDim::Lit(0)),
            ],
        ),
        vec![operand],
        ty(vec![DimInfo::Lit(7), DimInfo::Lit(4)], Prim::F32),
        None,
    );
    assert_eq!(
        output_axis_sources(&dag, padded),
        vec![
            AxisSource::OpComputed {
                op: padded,
                axis: 0
            },
            input_axis(0, 1),
        ],
    );
}

/// C4.2: every symbolic `Shrink` output axis is `OpComputed` with a fresh
/// extent, a full-axis `(Lit(0), ToEnd)` slice included. The identity is
/// decided by typed proof, never by the bound's shape.
#[test]
fn full_axis_symbolic_shrink_is_op_computed_not_pass_through() {
    let mut dag = Dag::new();
    let operand = load(&mut dag, "x", vec![named("n")]);
    let full_axis = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(RtDim::Lit(0), RtDim::ToEnd)],
        },
        vec![operand],
        ty(vec![named("m")], Prim::F32),
        None,
    );

    assert_eq!(
        output_axis_sources(&dag, full_axis),
        vec![AxisSource::OpComputed {
            op: full_axis,
            axis: 0
        }],
        "a full-axis slice mints a fresh extent, so it is not the input's dim"
    );

    // Negative parity: the input's own axis is external, so the two axes do
    // not share a source even though they have equal extent at run time.
    assert_eq!(
        output_axis_sources(&dag, operand),
        vec![AxisSource::ExternalAxis {
            load: operand,
            axis: 0
        }],
    );
}

/// C4.2 / chelis#665: an op-declared runtime axis on an `Expand` input flows
/// through the kept output axis with the insertion shift, instead of being
/// dropped because no `Load` carries its name.
#[test]
fn op_declared_axis_on_an_expand_input_flows_through_the_kept_output_axis() {
    let mut dag = Dag::new();
    let source = load(&mut dag, "x", vec![named("n")]);
    // A strided axis is op-declared: its extent is `ceil(n / 2)`, which no
    // `Load` declares.
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![source],
        ty(vec![named("m")], Prim::F32),
        None,
    );
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(3),
        },
        vec![strided],
        ty(vec![DimInfo::Lit(3), named("m")], Prim::F32),
        None,
    );

    let sources = output_axis_sources(&dag, expanded);
    assert_eq!(
        sources,
        vec![AxisSource::Literal { value: 3 }, input_axis(0, 0)],
        "the kept output axis 1 reads input axis 0, the op-declared extent"
    );
    // The kept axis resolves to the strided node's own computed extent, not
    // to a `Load` carrying a matching string.
    assert_eq!(
        output_axis_sources(&dag, strided),
        vec![AxisSource::OpComputed {
            op: strided,
            axis: 0
        }],
    );
    assert!(check_axis_sources(&dag, Stage::Lowering).is_ok());
}

/// C4.2: a reduction removes its axis, and `Count` removes every axis it
/// names, so a kept output axis maps back to its own input axis.
#[test]
fn reduction_and_count_shift_kept_output_axes_back_to_their_input_axis() {
    let mut dag = Dag::new();
    let operand = load(
        &mut dag,
        "x",
        vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
    );
    let summed = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![operand],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(4)], Prim::F32),
        None,
    );
    assert_eq!(
        output_axis_sources(&dag, summed),
        vec![input_axis(0, 0), input_axis(0, 2)],
    );

    let counted = dag.add_node(
        RiscOp::Count { axes: vec![2, 0] },
        vec![operand],
        ty(vec![DimInfo::Lit(3)], Prim::Int64),
        None,
    );
    assert_eq!(output_axis_sources(&dag, counted), vec![input_axis(0, 1)],);

    // Negative parity: reducing axis 0 keeps axes 1 and 2, so no output axis
    // maps to input axis 0.
    let leading = dag.add_node(
        RiscOp::MaxReduce { axis: 0 },
        vec![operand],
        ty(vec![DimInfo::Lit(3), DimInfo::Lit(4)], Prim::F32),
        None,
    );
    assert_eq!(
        output_axis_sources(&dag, leading),
        vec![input_axis(0, 1), input_axis(0, 2)],
    );
}

/// C4.1: the result must have exactly the output rank, with no omitted or
/// duplicated axis, and the check runs on every production path rather than
/// only in `verify`. Both directions are reached through real operations
/// here; the module's own unit tests present the synthetic vectors a correct
/// exhaustive match can never produce.
#[test]
fn omitted_or_duplicated_output_axis_source_fails_before_emission() {
    // Omitted: an input-less uniform fill whose only extent is anonymous has
    // nothing that can supply it, so it yields zero sources for rank 1.
    let mut omitted = Dag::new();
    let fill = omitted.add_node(
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        ty(vec![named("*")], Prim::F32),
        None,
    );
    omitted.add_root(fill);
    assert_eq!(output_axis_sources(&omitted, fill).len(), 0);
    let error = check_axis_sources(&omitted, Stage::Lowering)
        .expect_err("an omitted axis source must fail");
    assert!(
        error
            .to_string()
            .contains("0 extent source(s) for 1 output axis(es)"),
        "{error}"
    );

    // Duplicated: a `Permute` that names three source axes for a rank 2
    // output declares one source too many.
    let mut duplicated = Dag::new();
    let operand = load(&mut duplicated, "x", vec![DimInfo::Lit(2), DimInfo::Lit(3)]);
    let permuted = duplicated.add_node(
        RiscOp::Permute {
            axes: vec![0, 1, 0],
        },
        vec![operand],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32),
        None,
    );
    duplicated.add_root(permuted);
    assert_eq!(output_axis_sources(&duplicated, permuted).len(), 3);
    let error = check_axis_sources(&duplicated, Stage::Lowering)
        .expect_err("a duplicated axis source must fail");
    assert!(
        error
            .to_string()
            .contains("3 extent source(s) for 2 output axis(es)"),
        "{error}"
    );
    assert!(error.to_string().contains("chelis#1277"), "{error}");

    // Positive control: the same permutation over its real rank passes.
    let mut exact = Dag::new();
    let operand = load(&mut exact, "x", vec![DimInfo::Lit(2), DimInfo::Lit(3)]);
    let permuted = exact.add_node(
        RiscOp::Permute { axes: vec![1, 0] },
        vec![operand],
        ty(vec![DimInfo::Lit(3), DimInfo::Lit(2)], Prim::F32),
        None,
    );
    exact.add_root(permuted);
    assert!(check_axis_sources(&exact, Stage::Lowering).is_ok());
}

/// C4.1: `ExternalAxis` names the exact external `Load` by `NodeId`. No
/// source is located by searching for a `Load` that carries a string.
#[test]
fn external_axis_names_the_exact_load_not_a_string_match() {
    let mut dag = Dag::new();
    let first = load(&mut dag, "left", vec![named("n")]);
    let second = load(&mut dag, "right", vec![named("n")]);

    assert_eq!(
        output_axis_sources(&dag, first),
        vec![AxisSource::ExternalAxis {
            load: first,
            axis: 0
        }],
    );
    assert_eq!(
        output_axis_sources(&dag, second),
        vec![AxisSource::ExternalAxis {
            load: second,
            axis: 0
        }],
        "two Loads sharing a dimension name still name themselves, not each other"
    );
    assert_ne!(
        output_axis_sources(&dag, first),
        output_axis_sources(&dag, second),
        "the backward search returns whichever Load it reaches first; a source names one node"
    );

    // Negative parity: a node that is not a `Load` never produces an
    // `ExternalAxis`, so its axis is traced to its operand instead.
    let negated = dag.add_node(
        RiscOp::Neg,
        vec![first],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    assert_eq!(output_axis_sources(&dag, negated), vec![input_axis(0, 0)]);
    assert!(check_axis_sources(&dag, Stage::Lowering).is_ok());
}

/// C4.1 / C2.1: `InputAxis` validates the tensor slot and its normalized
/// `int32` axis; `ScalarInput` validates the rank-0 exact-`int64` contract.
#[test]
fn input_axis_and_scalar_input_sources_validate_their_slots() {
    let mut dag = Dag::new();
    let value = dag.add_node(
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let witness = load(&mut dag, "witness", vec![named("n")]);
    let size = dag.add_node(
        RiscOp::Shape { axis: 0 },
        vec![witness],
        scalar(Prim::Int64),
        None,
    );
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![value, size],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    dag.add_root(expanded);
    assert_eq!(
        output_axis_sources(&dag, expanded),
        vec![AxisSource::ScalarInput { input: 1 }],
    );
    assert!(check_axis_sources(&dag, Stage::Lowering).is_ok());

    // A folded tensor-axis read validates the tensor slot and its axis.
    let mut folded = Dag::new();
    let value = folded.add_node(
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let witness = load(&mut folded, "witness", vec![named("n")]);
    let expanded = folded.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![value, witness],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    folded.add_root(expanded);
    assert_eq!(
        output_axis_sources(&folded, expanded),
        vec![input_axis(1, 0)],
    );
    assert!(check_axis_sources(&folded, Stage::Lowering).is_ok());

    // Negative parity: an extent slot past the node's inputs.
    let mut absent_slot = Dag::new();
    let value = absent_slot.add_node(
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let expanded = absent_slot.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(4),
        },
        vec![value],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    absent_slot.add_root(expanded);
    let error = check_axis_sources(&absent_slot, Stage::Lowering)
        .expect_err("an absent extent slot must fail");
    assert!(error.to_string().contains("input slot 4"), "{error}");

    // Negative parity: a rank-0 non-int64 scalar is not an admissible
    // extent source under C2.1.
    let mut wrong_dtype = Dag::new();
    let value = wrong_dtype.add_node(
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let int32_size = wrong_dtype.add_node(
        RiscOp::Load {
            name: "size".into(),
        },
        vec![],
        scalar(Prim::Int32),
        None,
    );
    let expanded = wrong_dtype.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Node(1),
        },
        vec![value, int32_size],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    wrong_dtype.add_root(expanded);
    let error = check_axis_sources(&wrong_dtype, Stage::Lowering)
        .expect_err("a non-int64 extent scalar is not an admissible ScalarInput");
    assert!(error.to_string().contains("must be int64"), "{error}");
}

/// C4.3: a well-typed but currently unsupported mapping yields the
/// registered chelis#730 typed receipt. It never reaches the occurrence-pass
/// ICE and never substitutes an input extent.
#[test]
fn unsupported_but_well_typed_mapping_yields_the_registered_receipt_not_an_ice() {
    // chelis#1482's remaining class: an elementwise lowering-synthesized
    // uniform-fill `Const` whose declared extent is anonymous at runtime. No
    // literal, input, or shape dependency supplies it. Chelis#1313 removed
    // this shape from ReLU specifically, but sigmoid and sibling composite
    // lowerings still exercise the class.
    let mut dag = Dag::new();
    let operand = load(&mut dag, "x", vec![named("n")]);
    let shrunk = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(RtDim::Lit(0), RtDim::Lit(2))],
        },
        vec![operand],
        ty(vec![named("*")], Prim::F32),
        None,
    );
    let fill = dag.add_node(
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        ty(vec![named("*")], Prim::F32),
        None,
    );
    let maxed = dag.add_node(
        RiscOp::MaxElem,
        vec![shrunk, fill],
        ty(vec![named("*")], Prim::F32),
        None,
    );
    dag.add_root(maxed);

    assert_eq!(
        output_axis_sources(&dag, fill),
        vec![],
        "an anonymous extent on an input-less uniform fill has no source"
    );
    let error = check_axis_sources(&dag, Stage::Codegen("c"))
        .expect_err("a sourceless output axis must be a typed receipt");
    let rendered = error.to_string();
    assert!(rendered.starts_with("unsupported: "), "{rendered}");
    assert!(rendered.contains("chelis#1482"), "{rendered}");
    assert!(rendered.contains("codegen:c"), "{rendered}");

    // Positive control: the same fill with a literal extent has a source and
    // the check passes, so the receipt is about the missing source and not
    // about `Const` in an elementwise position.
    let mut sized = Dag::new();
    let operand = load(&mut sized, "x", vec![DimInfo::Lit(2)]);
    let fill = sized.add_node(
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        ty(vec![DimInfo::Lit(2)], Prim::F32),
        None,
    );
    let maxed = sized.add_node(
        RiscOp::MaxElem,
        vec![operand, fill],
        ty(vec![DimInfo::Lit(2)], Prim::F32),
        None,
    );
    sized.add_root(maxed);
    assert_eq!(
        output_axis_sources(&sized, fill),
        vec![AxisSource::Literal { value: 2 }],
    );
    assert!(check_axis_sources(&sized, Stage::Codegen("c")).is_ok());

    // A named extent that an equality class supplies is likewise not a
    // sourceless axis: a direct max/zero composition over `tensor[n, f32]`
    // builds and must keep building.
    let mut symbolic = Dag::new();
    let operand = load(&mut symbolic, "x", vec![named("n")]);
    let fill = symbolic.add_node(
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    let maxed = symbolic.add_node(
        RiscOp::MaxElem,
        vec![operand, fill],
        ty(vec![named("n")], Prim::F32),
        None,
    );
    symbolic.add_root(maxed);
    assert!(check_axis_sources(&symbolic, Stage::Codegen("c")).is_ok());
}

/// The number of `RiscOp` variants the table below must construct. Bumping
/// it without adding a row makes the coverage assertion fail.
const RISC_OP_VARIANTS: usize = 63;

/// Adding a `RiscOp` variant breaks this match, which is what forces the
/// table in `every_risc_op_yields_exactly_one_source_per_output_axis` to
/// grow with the vocabulary. Exhaustiveness in `output_axis_sources` proves
/// only that an ARM exists for each variant; a constructed node is what
/// proves the arm yields one source per output axis.
fn variant_index(op: &RiscOp) -> usize {
    match op {
        RiscOp::Add => 0,
        RiscOp::Sub => 1,
        RiscOp::Mul => 2,
        RiscOp::Div => 3,
        RiscOp::FloorDiv => 4,
        RiscOp::TruncDiv => 5,
        RiscOp::CmpLt => 6,
        RiscOp::MaxElem => 7,
        RiscOp::MinElem => 8,
        RiscOp::ExtremaAdjoint { .. } => 9,
        RiscOp::Neg => 10,
        RiscOp::Exp => 11,
        RiscOp::Log => 12,
        RiscOp::Sin => 13,
        RiscOp::Sqrt => 14,
        RiscOp::Cos => 15,
        RiscOp::Tan => 16,
        RiscOp::Atan => 17,
        RiscOp::Abs => 18,
        RiscOp::Floor => 19,
        RiscOp::Ceil => 20,
        RiscOp::Round => 21,
        RiscOp::Recip => 22,
        RiscOp::UniformLike { .. } => 23,
        RiscOp::Dropout { .. } => 24,
        RiscOp::Sum { .. } => 25,
        RiscOp::Count { .. } => 26,
        RiscOp::MaxReduce { .. } => 27,
        RiscOp::MinReduce { .. } => 28,
        RiscOp::ProdReduce { .. } => 29,
        RiscOp::ReduceWindow { .. } => 30,
        RiscOp::ReduceWindowGrad { .. } => 31,
        RiscOp::Argmax { .. } => 32,
        RiscOp::Argmin { .. } => 33,
        RiscOp::Reshape { .. } => 34,
        RiscOp::Permute { .. } => 35,
        RiscOp::Expand { .. } => 36,
        RiscOp::OneHot { .. } => 37,
        RiscOp::Pad { .. } => 38,
        RiscOp::Shrink { .. } => 39,
        RiscOp::Stride { .. } => 40,
        RiscOp::Shape { .. } => 41,
        RiscOp::Const { .. } => 42,
        RiscOp::ConstTensor { .. } => 43,
        RiscOp::Load { .. } => 44,
        RiscOp::Store { .. } => 45,
        RiscOp::Copy => 46,
        RiscOp::Drop => 47,
        RiscOp::Realize => 48,
        RiscOp::Cast { .. } => 49,
        RiscOp::CastTrunc { .. } => 50,
        RiscOp::FusedElem { .. } => 51,
        RiscOp::BlasMatmul { .. } => 52,
        RiscOp::Gather { .. } => 53,
        RiscOp::ScatterAdd { .. } => 54,
        RiscOp::Scatter { .. } => 55,
        RiscOp::ScatterElements { .. } => 56,
        RiscOp::Relu => 57,
        RiscOp::ReluAdjoint => 58,
        RiscOp::ExtentWitness { .. } => 59,
        RiscOp::CheckedReshapeExtent { .. } => 60,
        RiscOp::CheckedUnitAxis { .. } => 61,
        RiscOp::Mod => 62,
    }
}

/// C4.1: `output_axis_sources` yields exactly one source per output axis for
/// every operation in the vocabulary, and every source it yields refers to a
/// real edge.
///
/// One constructed node per `RiscOp` variant. An arm that returns the wrong
/// count, or names a slot or axis that does not exist, fails here rather
/// than surfacing later as a missing guard in a lane.
#[test]
fn every_risc_op_yields_exactly_one_source_per_output_axis() {
    let f32_23 = || ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F32);
    let mut dag = Dag::new();

    let f = load(&mut dag, "f", vec![DimInfo::Lit(2), DimInfo::Lit(3)]);
    let g = load(&mut dag, "g", vec![DimInfo::Lit(2), DimInfo::Lit(3)]);
    let ints = dag.add_node(
        RiscOp::Load { name: "i".into() },
        vec![],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::Int32),
        None,
    );
    let more_ints = dag.add_node(
        RiscOp::Load { name: "j".into() },
        vec![],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::Int32),
        None,
    );
    let flags = dag.add_node(
        RiscOp::Load { name: "p".into() },
        vec![],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::Bool),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load { name: "k".into() },
        vec![],
        ty(vec![DimInfo::Lit(4)], Prim::Int64),
        None,
    );
    let cell_indices = dag.add_node(
        RiscOp::Load { name: "c".into() },
        vec![],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::Int64),
        None,
    );
    let values = load(
        &mut dag,
        "v",
        vec![DimInfo::Lit(2), DimInfo::Lit(5), DimInfo::Lit(7)],
    );
    let updates = load(
        &mut dag,
        "u",
        vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(7)],
    );
    let cotangent = load(&mut dag, "w", vec![DimInfo::Lit(2), DimInfo::Lit(2)]);
    let lhs = load(
        &mut dag,
        "a",
        vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(3)],
    );
    let rhs = load(
        &mut dag,
        "b",
        vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(5)],
    );

    let mut nodes = vec![f];
    let add = |dag: &mut Dag, op: RiscOp, inputs: Vec<NodeId>, out: TensorType| {
        dag.add_node(op, inputs, out, None)
    };

    // Binary elementwise.
    for op in [
        RiscOp::Add,
        RiscOp::Sub,
        RiscOp::Mul,
        RiscOp::Div,
        RiscOp::FloorDiv,
        RiscOp::MaxElem,
        RiscOp::MinElem,
    ] {
        nodes.push(add(&mut dag, op, vec![f, g], f32_23()));
    }
    nodes.push(add(
        &mut dag,
        RiscOp::TruncDiv,
        vec![ints, more_ints],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::Int32),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::Mod,
        vec![ints, more_ints],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::Int32),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::CmpLt,
        vec![f, g],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::Bool),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::ExtremaAdjoint {
            kind: ExtremaKind::Max,
            operand: ExtremaOperand::Left,
        },
        vec![f, g, cotangent],
        f32_23(),
    ));
    nodes.push(add(&mut dag, RiscOp::Relu, vec![f], f32_23()));
    nodes.push(add(
        &mut dag,
        RiscOp::ReluAdjoint,
        vec![f, cotangent],
        f32_23(),
    ));

    // Unary elementwise.
    for op in [
        RiscOp::Neg,
        RiscOp::Exp,
        RiscOp::Log,
        RiscOp::Sin,
        RiscOp::Sqrt,
        RiscOp::Cos,
        RiscOp::Tan,
        RiscOp::Atan,
        RiscOp::Abs,
        RiscOp::Floor,
        RiscOp::Ceil,
        RiscOp::Round,
        RiscOp::Recip,
        RiscOp::Copy,
        RiscOp::Drop,
        RiscOp::Realize,
        RiscOp::UniformLike {
            low: 0.0,
            high: 1.0,
            seed: 7,
        },
        RiscOp::Dropout {
            rate: 0.5,
            seed: 11,
        },
        RiscOp::Store { name: "out".into() },
    ] {
        nodes.push(add(&mut dag, op, vec![f], f32_23()));
    }
    nodes.push(add(
        &mut dag,
        RiscOp::Cast {
            new_precision: Prim::F64,
        },
        vec![f],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::F64),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::CastTrunc {
            new_precision: Prim::Int32,
        },
        vec![f],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(3)], Prim::Int32),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::FusedElem {
            ops: vec![FusedStep {
                op: FusedStepOp::Add,
                input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
            }],
        },
        vec![f, g],
        f32_23(),
    ));

    // Reductions: axis 0 of a rank 2 operand leaves rank 1.
    for op in [
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        RiscOp::MaxReduce { axis: 0 },
        RiscOp::MinReduce { axis: 0 },
        RiscOp::ProdReduce { axis: 0 },
    ] {
        nodes.push(add(
            &mut dag,
            op,
            vec![f],
            ty(vec![DimInfo::Lit(3)], Prim::F32),
        ));
    }
    for op in [RiscOp::Argmax { axis: 0 }, RiscOp::Argmin { axis: 0 }] {
        nodes.push(add(
            &mut dag,
            op,
            vec![f],
            ty(vec![DimInfo::Lit(3)], Prim::Int64),
        ));
    }
    nodes.push(add(
        &mut dag,
        RiscOp::Count { axes: vec![0] },
        vec![flags],
        ty(vec![DimInfo::Lit(3)], Prim::Int64),
    ));

    // Windowed: the leading axis passes through, the windowed axis has
    // extent floor((3 - 2) / 1) + 1 = 2.
    nodes.push(add(
        &mut dag,
        RiscOp::ReduceWindow {
            reducer: ReduceWindowKind::Sum,
            window_shape: vec![2],
            strides: vec![1],
        },
        vec![f],
        ty(vec![DimInfo::Lit(2), DimInfo::Lit(2)], Prim::F32),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::ReduceWindowGrad {
            reducer: ReduceWindowKind::Sum,
            window_shape: vec![2],
            strides: vec![1],
        },
        vec![f, cotangent],
        f32_23(),
    ));

    // Movement.
    nodes.push(add(
        &mut dag,
        RiscOp::Reshape {
            new_shape: vec![RtDim::Lit(6)],
        },
        vec![f],
        ty(vec![DimInfo::Lit(6)], Prim::F32),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::Permute { axes: vec![1, 0] },
        vec![f],
        ty(vec![DimInfo::Lit(3), DimInfo::Lit(2)], Prim::F32),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(4),
        },
        vec![f],
        ty(
            vec![DimInfo::Lit(4), DimInfo::Lit(2), DimInfo::Lit(3)],
            Prim::F32,
        ),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::OneHot { vocab: 5 },
        vec![indices],
        ty(vec![DimInfo::Lit(4), DimInfo::Lit(5)], Prim::F32),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::zero_pad(
            Prim::F32,
            vec![
                (RtDim::Lit(0), RtDim::Lit(0)),
                (RtDim::Lit(0), RtDim::Lit(0)),
            ],
        ),
        vec![f],
        f32_23(),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::Shrink {
            bounds: vec![
                (RtDim::Lit(0), RtDim::Lit(2)),
                (RtDim::Lit(0), RtDim::Lit(3)),
            ],
        },
        vec![f],
        f32_23(),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::Stride {
            strides: vec![RtDim::Lit(1), RtDim::Lit(1)],
        },
        vec![f],
        f32_23(),
    ));

    // Shape query and memory.
    nodes.push(add(
        &mut dag,
        RiscOp::ExtentWitness {
            site: chelis_ir::dag::ExtentWitnessSite::Caller,
            parameter: "f".into(),
            axis: RtAxis::Lit(0),
            requirements: vec![chelis_types::scalar_from_i64("load", Prim::Int64, 2).unwrap()],
            claims: Vec::new(),
        },
        vec![f],
        scalar(Prim::Int64),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::Shape { axis: 0 },
        vec![f],
        scalar(Prim::Int64),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        scalar(Prim::F32),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::synth_const_tensor(Prim::F32, vec![0.0; 6]),
        vec![],
        f32_23(),
    ));

    let operand = load(&mut dag, "unit", vec![named("unit"), DimInfo::Lit(3)]);
    let witness = add(
        &mut dag,
        RiscOp::ExtentWitness {
            site: chelis_ir::dag::ExtentWitnessSite::Caller,
            parameter: "unit".into(),
            axis: RtAxis::Lit(0),
            requirements: vec![chelis_types::scalar_from_i64("load", Prim::Int64, 1).unwrap()],
            claims: Vec::new(),
        },
        vec![operand],
        scalar(Prim::Int64),
    );
    let required = add(
        &mut dag,
        RiscOp::Const {
            value: chelis_types::scalar_from_i64("reshape", Prim::Int64, 1).unwrap(),
        },
        vec![],
        scalar(Prim::Int64),
    );
    nodes.push(add(
        &mut dag,
        RiscOp::CheckedReshapeExtent {
            claims: vec!["unit".into()],
            axis: RtAxis::Lit(0),
        },
        vec![witness, required],
        scalar(Prim::Int64),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::CheckedUnitAxis {
            axis: RtAxis::Lit(0),
        },
        vec![operand, witness],
        ty(vec![DimInfo::Lit(1), DimInfo::Lit(3)], Prim::F32),
    ));

    // Backend specialization and sparse.
    nodes.push(add(
        &mut dag,
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Concrete(2)],
            m: DimExpr::Concrete(4),
            n: DimExpr::Concrete(5),
            k: DimExpr::Concrete(3),
            accumulator: Prim::F32,
        },
        vec![lhs, rhs],
        ty(
            vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(5)],
            Prim::F32,
        ),
    ));
    nodes.push(add(
        &mut dag,
        RiscOp::Gather { axis: 1 },
        vec![values, indices],
        ty(
            vec![DimInfo::Lit(2), DimInfo::Lit(4), DimInfo::Lit(7)],
            Prim::F32,
        ),
    ));
    for op in [RiscOp::ScatterAdd { axis: 1 }, RiscOp::Scatter { axis: 1 }] {
        nodes.push(add(
            &mut dag,
            op,
            vec![values, indices, updates],
            ty(
                vec![DimInfo::Lit(2), DimInfo::Lit(5), DimInfo::Lit(7)],
                Prim::F32,
            ),
        ));
    }
    nodes.push(add(
        &mut dag,
        RiscOp::ScatterElements { axis: 1 },
        vec![f, cell_indices, g],
        f32_23(),
    ));

    let mut covered = vec![0usize; RISC_OP_VARIANTS];
    for id in &nodes {
        let node = dag.get(*id).expect("node");
        let sources = output_axis_sources(&dag, *id);
        assert_eq!(
            sources.len(),
            node.output_type.dims.len(),
            "node {} op {:?} produced {} sources for rank {}",
            id.0,
            node.op,
            sources.len(),
            node.output_type.dims.len()
        );
        covered[variant_index(&node.op)] += 1;
    }

    let uncovered: Vec<usize> = covered
        .iter()
        .enumerate()
        .filter(|(_, count)| **count == 0)
        .map(|(index, _)| index)
        .collect();
    assert!(
        uncovered.is_empty(),
        "variant index(es) {uncovered:?} have no constructed node; \
         every RiscOp arm must be exercised, not merely present"
    );
    assert_eq!(
        nodes.len(),
        RISC_OP_VARIANTS,
        "one node per variant, so a duplicate row cannot mask a missing one"
    );

    // Every source the whole vocabulary produced refers to a real edge.
    assert!(check_axis_sources(&dag, Stage::Lowering).is_ok());
}

/// chelis#1480 / `spec/05` section 2.4.1: a `ToEnd` end is well formed only
/// when the start paired with it is `Lit(0)`. A `ToEnd` end over any other
/// start is a malformed bound that every stage validating a bound rejects
/// rather than resolving to a slice.
#[test]
fn to_end_shrink_end_requires_a_literal_zero_start() {
    let full_axis = |start: RtDim| {
        let mut dag = Dag::new();
        let operand = load(&mut dag, "x", vec![named("n")]);
        let sliced = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(start, RtDim::ToEnd)],
            },
            vec![operand],
            ty(vec![named("m")], Prim::F32),
            None,
        );
        dag.add_root(sliced);
        dag
    };

    // The one well-formed spelling: the identity slice of a symbolic
    // bystander axis.
    assert!(
        chelis_ir::verify::verify(&full_axis(RtDim::Lit(0))).is_empty(),
        "the full-axis sentinel is the legal ToEnd spelling"
    );

    // A nonzero literal start is malformed. Before chelis#1480 nothing
    // rejected it: `verify` constrained only the START carrier, and
    // `bind_symbolic_dims` resolved `(Lit(1), ToEnd)` into `(Lit(1),
    // Lit(1 + size))`, a slice the spec says does not exist.
    let errors = chelis_ir::verify::verify(&full_axis(RtDim::Lit(1)));
    assert!(
        errors
            .iter()
            .any(|error| error.contains("pairs the ToEnd sentinel with a start that is not Lit(0)")),
        "a nonzero start must be rejected: {errors:?}"
    );

    // A runtime start is malformed for the same reason: the sentinel means
    // the whole axis, and only a literal zero says so.
    let mut runtime_start = Dag::new();
    let operand = load(&mut runtime_start, "x", vec![named("n")]);
    let offset = runtime_start.add_node(
        RiscOp::Shape { axis: 0 },
        vec![operand],
        scalar(Prim::Int64),
        None,
    );
    let sliced = runtime_start.add_node(
        RiscOp::Shrink {
            bounds: vec![(RtDim::Node(1), RtDim::ToEnd)],
        },
        vec![operand, offset],
        ty(vec![named("m")], Prim::F32),
        None,
    );
    runtime_start.add_root(sliced);
    let errors = chelis_ir::verify::verify(&runtime_start);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("pairs the ToEnd sentinel with a start that is not Lit(0)")),
        "a runtime start must be rejected: {errors:?}"
    );

    // The existing control, unchanged: a `ToEnd` START was already rejected,
    // which is what proved the verifier looks at the carrier at all.
    let mut start_sentinel = Dag::new();
    let operand = load(&mut start_sentinel, "x", vec![named("n")]);
    let sliced = start_sentinel.add_node(
        RiscOp::Shrink {
            bounds: vec![(RtDim::ToEnd, RtDim::Lit(2))],
        },
        vec![operand],
        ty(vec![named("m")], Prim::F32),
        None,
    );
    start_sentinel.add_root(sliced);
    assert!(
        chelis_ir::verify::verify(&start_sentinel)
            .iter()
            .any(|error| error.contains("only valid as an end")),
        "a ToEnd start is rejected"
    );
}

/// The resolution stage rejects the same malformed bound rather than
/// resolving it. `bind_symbolic_dims` is the stage eval reaches for every
/// `ToEnd` bound, so this is what keeps the eval lane from computing a
/// slice the spec says is not one.
#[test]
fn binding_a_to_end_bound_rejects_a_start_that_is_not_literal_zero() {
    let bound = |start: RtDim| {
        let mut dag = Dag::new();
        let operand = load(&mut dag, "x", vec![named("n")]);
        let sliced = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(start, RtDim::ToEnd)],
            },
            vec![operand],
            ty(vec![named("n")], Prim::F32),
            None,
        );
        dag.add_root(sliced);
        dag
    };
    let mut bindings = chelis_unord::UnordMap::new();
    bindings.insert("n".to_string(), 4usize);

    let resolved = chelis_ir::dag::bind_symbolic_dims(&bound(RtDim::Lit(0)), &bindings)
        .expect("the identity slice binds");
    assert!(matches!(
        &resolved.get(NodeId(1)).expect("shrink").op,
        RiscOp::Shrink { bounds }
            if bounds == &vec![(RtDim::Lit(0), RtDim::Lit(4))]
    ));

    let error = chelis_ir::dag::bind_symbolic_dims(&bound(RtDim::Lit(1)), &bindings)
        .expect_err("a nonzero start is a malformed bound, not a slice");
    assert!(error.contains("requires a literal zero start"), "{error}");
}

// ===========================================================================
// C4.4's declaration half: `resolve_axis_extent` and `dim_extent_origins`.
//
// `output_axis_sources` answers one hop. A declaration consumer needs the
// terminal answer, because the emitted C allocates by NAME and every name it
// renders needs one place that assigns it. These rows pin what that
// resolution answers for the shapes chelis#665 and chelis#1556 report.
// ===========================================================================

/// chelis#665 with the two spellings the lowerer actually produces: the
/// `Stride`'s own axis is `m`, and the kept axis of the rank-raising `Expand`
/// carries the lowerer's fresh `_anon_dim_2_1`. The name-keyed walk cannot
/// declare the second, because no `Load` carries that string. The resolution
/// answers in one hop, and it answers with the STRIDE, not with `x`.
///
/// EVIDENTIARY STATUS: regression test for the derivation. There was no
/// resolution to answer before this change, and the consumer it feeds ICEd.
#[test]
fn a_kept_axis_under_a_second_spelling_resolves_to_the_operation_that_computes_it() {
    let mut dag = Dag::new();
    let source = load(&mut dag, "x", vec![named("n")]);
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![source],
        ty(vec![named("m")], Prim::F32),
        None,
    );
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(3),
        },
        vec![strided],
        ty(vec![DimInfo::Lit(3), named("_anon_dim_2_1")], Prim::F32),
        None,
    );
    dag.add_root(expanded);

    assert_eq!(
        resolve_axis_extent(&dag, expanded, 1),
        Some(ExtentOrigin::OpComputed {
            op: strided,
            axis: 0
        }),
        "the kept axis's extent is produced by the stride, whatever it is spelled",
    );
    assert_eq!(
        resolve_axis_extent(&dag, strided, 0),
        Some(ExtentOrigin::OpComputed {
            op: strided,
            axis: 0
        }),
    );
    // The operand's own name still resolves to the input it is declared by,
    // which is what the prologue declares it from.
    assert_eq!(
        resolve_axis_extent(&dag, source, 0),
        Some(ExtentOrigin::ExternalAxis {
            load: source,
            axis: 0
        }),
    );
    assert_eq!(
        dim_extent_origins(&dag),
        vec![
            (
                "n".to_string(),
                ExtentOrigin::ExternalAxis {
                    load: source,
                    axis: 0
                }
            ),
            (
                "m".to_string(),
                ExtentOrigin::OpComputed {
                    op: strided,
                    axis: 0
                }
            ),
            (
                "_anon_dim_2_1".to_string(),
                ExtentOrigin::OpComputed {
                    op: strided,
                    axis: 0
                }
            ),
        ],
        "every rendered name has one origin, in node-id (= emission) order",
    );
    assert!(unresolved_dim_names(&dag).is_empty());
}

/// chelis#1556's shape: a shape-preserving operation over an operand whose
/// axis is a literal resolves in two hops to that literal. The consumer then
/// declares `int64_t d43 = 3;` instead of raising a missing-binding error
/// against a kernel that has no `Load` at all.
///
/// EVIDENTIARY STATUS: regression test for the derivation.
#[test]
fn a_shape_preserving_axis_over_a_literal_operand_resolves_to_the_literal() {
    let mut dag = Dag::new();
    // The real program's `scalar_to_tensor(1.0f32)`; a rank-0 operand is all
    // the insertion needs, and its own kind is not what this row measures.
    let seed = dag.add_node(
        RiscOp::Load { name: "s".into() },
        vec![],
        scalar(Prim::F32),
        None,
    );
    let filled = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(3),
        },
        vec![seed],
        ty(vec![DimInfo::Lit(3)], Prim::F32),
        None,
    );
    let noised = dag.add_node(
        RiscOp::UniformLike {
            low: 0.0,
            high: 1.0,
            seed: 42,
        },
        vec![filled],
        ty(vec![named("d43")], Prim::F32),
        None,
    );
    dag.add_root(noised);

    assert_eq!(
        resolve_axis_extent(&dag, noised, 0),
        Some(ExtentOrigin::Literal(3)),
        "two hops: the uniform preserves its operand's shape, which is a literal",
    );
    assert_eq!(
        dim_extent_origins(&dag),
        vec![("d43".to_string(), ExtentOrigin::Literal(3))],
    );
    assert!(unresolved_dim_names(&dag).is_empty());
}

/// The negative twin: a name whose axis has NO source resolves to no origin
/// and is reported, so a consumer can turn it into a typed receipt instead of
/// panicking or guessing. This is chelis#1482's shape, which
/// `check_axis_sources` already refuses first; the report exists so a
/// consumer that reaches a name the refusal did not cover still fails closed.
///
/// EVIDENTIARY STATUS: regression test for the derivation.
#[test]
fn a_named_axis_with_no_source_resolves_to_no_origin_and_is_reported() {
    let mut dag = Dag::new();
    // A `Const` fill whose only positive-rank axis is ANONYMOUS has no
    // literal, no operand, no class and no shape dependency to size it.
    let orphan = dag.add_node(
        RiscOp::Const {
            value: chelis_types::scalar_from_f64("test", Prim::F32, 0.0).expect("zero fill"),
        },
        vec![],
        ty(vec![named("")], Prim::F32),
        None,
    );
    let renamed = dag.add_node(
        RiscOp::Neg,
        vec![orphan],
        ty(vec![named("_anon_dim_1_0")], Prim::F32),
        None,
    );
    dag.add_root(renamed);

    assert_eq!(resolve_axis_extent(&dag, renamed, 0), None);
    assert_eq!(dim_extent_origins(&dag), vec![]);
    assert_eq!(
        unresolved_dim_names(&dag),
        vec!["_anon_dim_1_0".to_string()],
    );
}
