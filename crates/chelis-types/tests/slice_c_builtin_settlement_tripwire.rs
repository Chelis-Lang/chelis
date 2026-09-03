//! Slice C c7: every builtin declares what it does with a deferred positional
//! `expand` result, and the declaration is proved rather than trusted.
//!
//! Two obligations, kept separate on purpose.
//!
//! **Coverage** is total and runs in both directions. Every builtin declaring
//! a disposition other than `NoTensorOperand` belongs to exactly one family
//! here, and every family member is a builtin that declares that family's
//! disposition. A new builtin that declares a disposition without a family
//! fails, which is the case `spec/design/runtime_extents.md` C3 names.
//!
//! **Behaviour** is proved once per family, by executing that family's
//! representative against a real one-def program whose tensor operand is
//! `expand(x, 0, 3i64)`. A representative proves its own row and stands for
//! its family; it does not prove every member. Members no probe can reach are
//! named in `UNDISCRIMINATED` with the reason, rather than absorbed into a
//! family boundary drawn to make the coverage claim come out clean.

use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::{BUILTINS, TensorSettlement, check_typed_program};
use std::collections::BTreeSet;

/// One inference family: a disposition, the builtin whose behaviour is
/// executed for it, and the members that ride on that execution.
struct Family {
    name: &'static str,
    settlement: TensorSettlement,
    representative: &'static str,
    /// Element dtype of the producer, so bool and integer families get an
    /// operand their signatures admit.
    dtype: &'static str,
    /// The call, with `e` bound to the pending `expand` result.
    call: &'static str,
    /// Extra parameters the call needs, appended to the def's parameter list.
    extra_params: &'static str,
    /// The call's own result type when the operand takes its insertion form.
    /// `None` where no spelling discriminates, which forces the family into
    /// `UNDISCRIMINATED`.
    insertion_result: Option<&'static str>,
    members: &'static [&'static str],
}

const FAMILIES: &[Family] = &[
    Family {
        name: "elementwise binary",
        settlement: TensorSettlement::Constrains,
        representative: "add",
        dtype: "f32",
        call: "add(e, e)",
        extra_params: "",
        insertion_result: Some("tensor[3, 2, f32]"),
        members: &[
            "add",
            "mul",
            "sub",
            "div",
            "floor_div",
            "trunc_div",
            "max_elem",
            "min_elem",
        ],
    },
    Family {
        name: "elementwise unary",
        settlement: TensorSettlement::Constrains,
        representative: "neg",
        dtype: "f32",
        call: "neg(e)",
        extra_params: "",
        insertion_result: Some("tensor[3, 2, f32]"),
        members: &[
            "neg",
            "recip",
            "exp",
            "log",
            "sin",
            "sqrt",
            "cos",
            "tan",
            "atan",
            "abs",
            "floor",
            "ceil",
            "round",
            "relu",
            "sigmoid",
            "tanh",
            "silu",
            "gelu",
            "normalize",
        ],
    },
    Family {
        name: "comparison",
        settlement: TensorSettlement::Constrains,
        representative: "eq",
        dtype: "f32",
        call: "eq(e, e)",
        extra_params: "",
        insertion_result: Some("tensor[3, 2, bool]"),
        members: &["cmplt", "eq", "neq", "lt", "gt", "lte", "gte"],
    },
    Family {
        name: "integer binary",
        settlement: TensorSettlement::Constrains,
        representative: "bitand",
        dtype: "int32",
        call: "bitand(e, e)",
        extra_params: "",
        insertion_result: Some("tensor[3, 2, int32]"),
        members: &["mod", "bitand", "bitor", "bitxor", "shl", "shr"],
    },
    Family {
        name: "logical",
        settlement: TensorSettlement::Constrains,
        representative: "and",
        dtype: "bool",
        call: "and(e, e)",
        extra_params: "",
        insertion_result: Some("tensor[3, 2, bool]"),
        members: &["and", "or", "not"],
    },
    Family {
        name: "shape preserving",
        settlement: TensorSettlement::Constrains,
        representative: "clamp",
        dtype: "f32",
        call: "clamp(e, 0.0f32, 1.0f32)",
        extra_params: "",
        insertion_result: Some("tensor[3, 2, f32]"),
        members: &[
            "where",
            "clamp",
            "uniform_like",
            "softmax",
            "layer_norm",
            "cumsum",
            "sort",
            "stride",
            "debug",
        ],
    },
    Family {
        name: "reshape",
        settlement: TensorSettlement::Constrains,
        representative: "reshape",
        dtype: "f32",
        call: "reshape(e, [6i64])",
        extra_params: "",
        insertion_result: Some("tensor[6, f32]"),
        members: &["reshape"],
    },
    // Split from the scatter family because measurement disagreed with family
    // membership: its siblings refuse an unresolved operand and it does not.
    Family {
        name: "scatter elements",
        settlement: TensorSettlement::Constrains,
        representative: "scatter_elements",
        dtype: "f32",
        call: "scatter_elements(e, i, e, 0)",
        extra_params: ", i: tensor[3, 2, int32]",
        insertion_result: Some("tensor[3, 2, f32]"),
        members: &["scatter_elements"],
    },
    Family {
        name: "matmul",
        settlement: TensorSettlement::Constrains,
        representative: "matmul",
        dtype: "f32",
        call: "matmul(e, b)",
        extra_params: ", b: tensor[2, 7, f32]",
        insertion_result: Some("tensor[3, 7, f32]"),
        members: &["matmul"],
    },
    Family {
        name: "reduction",
        settlement: TensorSettlement::Propagates,
        representative: "sum",
        dtype: "f32",
        call: "sum(e, 0)",
        extra_params: "",
        insertion_result: None,
        members: &[
            "mean",
            "sum",
            "count",
            "max_reduce",
            "min_reduce",
            "prod_reduce",
            "argmax_reduce",
            "argmin_reduce",
        ],
    },
    Family {
        name: "windowed reduction",
        settlement: TensorSettlement::Propagates,
        representative: "reduce_window_sum",
        dtype: "f32",
        call: "reduce_window_sum(e, [2i64, 2i64], [1i64, 1i64])",
        extra_params: "",
        insertion_result: None,
        members: &[
            "reduce_window_max",
            "reduce_window_min",
            "reduce_window_sum",
            "reduce_window_mean",
        ],
    },
    Family {
        name: "rank changing movement",
        settlement: TensorSettlement::Propagates,
        representative: "permute",
        dtype: "f32",
        call: "permute(e, 1, 0)",
        extra_params: "",
        insertion_result: None,
        members: &["permute", "expand", "pad", "shrink", "conv2d"],
    },
    Family {
        name: "shape query",
        settlement: TensorSettlement::Propagates,
        representative: "rank",
        dtype: "f32",
        call: "rank(e)",
        extra_params: "",
        insertion_result: None,
        members: &["rank", "numel", "tensor_to_scalar"],
    },
    Family {
        name: "generic passthrough",
        settlement: TensorSettlement::Propagates,
        representative: "print",
        dtype: "f32",
        call: "print(e)",
        extra_params: "",
        insertion_result: None,
        members: &[
            "print",
            "to_string",
            "fail",
            "test_assert",
            "test_assert_eq",
            "test_assert_eq_tensor",
        ],
    },
    // No member of this family has a call spelling that discriminates, so the
    // whole family is covered-by-coverage-only. See UNDISCRIMINATED.
    Family {
        name: "container producing",
        settlement: TensorSettlement::Propagates,
        representative: "to_list",
        dtype: "f32",
        call: "to_list(e)",
        extra_params: "",
        insertion_result: None,
        members: &["split", "to_list", "einsum", "tensor_scan"],
    },
    Family {
        name: "shape neutral freeze",
        settlement: TensorSettlement::Freezes,
        representative: "shape",
        dtype: "f32",
        call: "shape(e, 0)",
        extra_params: "",
        insertion_result: None,
        members: &["shape", "test_assert_close_tensor"],
    },
    Family {
        name: "refuses an unresolved operand",
        settlement: TensorSettlement::RejectsUnresolved,
        representative: "gather",
        dtype: "f32",
        call: "gather(e, idx, 0)",
        extra_params: ", idx: tensor[1, int32]",
        insertion_result: None,
        members: &[
            "gather",
            "concat",
            "diagonal",
            "trace",
            "scatter",
            "scatter_replace",
            "round_to",
        ],
    },
];

/// Declarations no probe in this file reaches, with the reason each resists
/// one. These are declared from their signature family and are NOT proved by
/// a representative; the coverage tests still bind them.
const UNDISCRIMINATED: &[(&str, &str)] = &[
    (
        "split",
        "returns a list of tensors; no declared result discriminates",
    ),
    (
        "to_list",
        "returns a list; no declared result discriminates",
    ),
    (
        "einsum",
        "both candidate forms appear in the rendered program",
    ),
    (
        "tensor_scan",
        "no well-typed call could be constructed for its (T, int64) -> T callback",
    ),
    (
        "conv2d",
        "needs a rank-4 producer and four arguments, which the shared probe shape cannot express",
    ),
];

fn check(source: &str) -> Result<String, String> {
    let decls = parse_surf(source).map_err(|error| format!("parse: {error:?}"))?;
    match check_typed_program(&desugar_program(&decls)) {
        Ok(checked) => Ok(print_canonical(checked.annotated_exprs())),
        Err(report) => Err(format!(
            "[{:?}] {}",
            report.errors[0].kind, report.errors[0].message
        )),
    }
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn declared(name: &str) -> TensorSettlement {
    BUILTINS
        .iter()
        .find(|decl| decl.name == name)
        .unwrap_or_else(|| panic!("`{name}` is not a registered builtin"))
        .tensor_settlement
}

// ---------------------------------------------------------------------
// Coverage, both directions.
// ---------------------------------------------------------------------

#[test]
fn every_declared_consumer_belongs_to_exactly_one_family() {
    let declared_consumers = BUILTINS
        .iter()
        .filter(|decl| decl.tensor_settlement != TensorSettlement::NoTensorOperand)
        .map(|decl| decl.name)
        .collect::<BTreeSet<_>>();
    let mut covered = BTreeSet::new();
    for family in FAMILIES {
        for member in family.members {
            assert!(
                covered.insert(*member),
                "`{member}` appears in more than one family; a builtin has one \
                 settlement disposition and therefore one family"
            );
        }
    }
    assert_eq!(
        declared_consumers, covered,
        "every builtin declaring a settlement disposition needs a family, and \
         every family member must declare one. A new builtin that declares a \
         disposition without a spelling here fails on this line."
    );
}

#[test]
fn every_family_member_declares_its_family_disposition() {
    for family in FAMILIES {
        for member in family.members {
            assert_eq!(
                declared(member),
                family.settlement,
                "`{member}` sits in family `{}`, which is {:?}",
                family.name,
                family.settlement
            );
        }
        assert!(
            family.members.contains(&family.representative),
            "family `{}` must execute one of its own members",
            family.name
        );
    }
}

#[test]
fn undiscriminated_declarations_are_named_and_are_a_strict_subset() {
    for (name, reason) in UNDISCRIMINATED {
        assert!(
            !reason.trim().is_empty(),
            "`{name}` needs a reason no probe reaches it"
        );
        assert_ne!(
            declared(name),
            TensorSettlement::NoTensorOperand,
            "`{name}` is listed as undiscriminated, so it must declare a \
             disposition; a builtin with no tensor operand needs no probe"
        );
    }
    let named = UNDISCRIMINATED
        .iter()
        .map(|(name, _)| *name)
        .collect::<BTreeSet<_>>();
    let representatives = FAMILIES
        .iter()
        .map(|family| family.representative)
        .collect::<BTreeSet<_>>();
    let both = named.intersection(&representatives).collect::<Vec<_>>();
    assert!(
        both.len() <= 1,
        "a family whose representative is undiscriminated proves nothing, and \
         only the container-producing family is allowed to be in that state: {both:?}"
    );
}

// ---------------------------------------------------------------------
// Behaviour, once per family.
// ---------------------------------------------------------------------

/// Does the call leave the operand selectable afterwards?
fn survives_the_call(family: &Family) -> Result<String, String> {
    check(&format!(
        "def sink(x: tensor[3, 2, {dtype}]) -> int32 = 0\n\
         def f(a: tensor[2, {dtype}]{extra}) -> int32 = {{\n  \
         e = expand(a, 0, 3i64)\n  u = {call}\n  sink(e)\n}}\n",
        dtype = family.dtype,
        extra = family.extra_params,
        call = family.call,
    ))
}

#[test]
fn each_family_representative_behaves_as_its_family_declares() {
    for family in FAMILIES {
        let survived = survives_the_call(family);
        match family.settlement {
            TensorSettlement::Constrains => {
                let result = family
                    .insertion_result
                    .expect("a Constrains family states its insertion result");
                let rendered = check(&format!(
                    "def f(a: tensor[2, {dtype}]{extra}) -> {result} = {{\n  \
                     e = expand(a, 0, 3i64)\n  {call}\n}}\n",
                    dtype = family.dtype,
                    extra = family.extra_params,
                    call = family.call,
                ))
                .unwrap_or_else(|error| {
                    panic!("{}: expected acceptance, got {error}", family.name)
                });
                let insertion = collapse(&format!(
                    "(t-tensor {{}} (d-lit {{}} 3) (d-lit {{}} 2) (t-prim {{}} {}))",
                    family.dtype
                ));
                assert!(
                    collapse(&rendered).contains(&insertion),
                    "{}: a declared result must reach the operand and select \
                     its insertion form:\n{rendered}",
                    family.name
                );
                assert!(
                    !rendered.contains("(t-var {} t"),
                    "{}: a consumer that fixes its operand publishes no \
                     unresolved variable:\n{rendered}",
                    family.name
                );
            }
            TensorSettlement::Propagates => {
                survived.unwrap_or_else(|error| {
                    panic!(
                        "{}: propagation leaves the choice open, so a later \
                         consumer must still select the insertion form, got \
                         {error}",
                        family.name
                    )
                });
            }
            TensorSettlement::Freezes => {
                let error = survived.expect_err(&format!(
                    "{}: freezing fixes the operand at the replacement form, \
                     so the later insertion-form consumer must be rejected",
                    family.name
                ));
                assert!(
                    error.contains("rank mismatch") || error.contains("DimensionMismatch"),
                    "{}: the rejection must name the shape disagreement, got {error}",
                    family.name
                );
            }
            TensorSettlement::RejectsUnresolved => {
                let error = survived.expect_err(&format!(
                    "{}: this family refuses an unresolved operand outright",
                    family.name
                ));
                assert!(
                    error.contains('?'),
                    "{}: the rejection must name the unresolved operand, got {error}",
                    family.name
                );
            }
            TensorSettlement::NoTensorOperand => {
                panic!("{}: a family cannot be NoTensorOperand", family.name)
            }
        }
    }
}
