//! Slice C c7: every builtin declares what it does with a deferred positional
//! `expand` result, and the declaration is proved rather than trusted.
//!
//! Every value in the registry was established by execution rather than by
//! reading the builtin's inference route. What this file proves about those
//! values is narrower, and the limit is structural rather than an oversight.
//!
//! A family whose declared result binds nothing accepts mutually incompatible
//! declarations from one body, so no declared result can move the operand and
//! no probe of this shape can separate `Constrains` from `Propagates` there.
//!
//! Those families are named in `UNDISCRIMINATED` with that reason, alongside
//! the declarations no probe could reach at all. What makes them unprovable is
//! chelis#1512 rather than anything the `tensor_settlement` field describes.
//!
//! Two obligations, kept separate on purpose.
//!
//! **Coverage** is total and runs in both directions. Every builtin declaring
//! a disposition other than `NoTensorOperand` belongs to exactly one family
//! here, and every family member is a builtin that declares that family's
//! disposition. A new builtin that declares a disposition without a family
//! fails, which is the case `spec/design/runtime_extents.md` C3 names.
//!
//! **Behaviour** is proved once per family by executing that family's
//! representative, after a precondition establishes that the probe can say
//! anything at all. A representative proves its own row and stands for its
//! family; it does not prove every member.

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
    /// `None` for the `Freezes` and `RejectsUnresolved` arms, which probe the
    /// call directly rather than through a declared result.
    declared_result: Option<&'static str>,
    /// A well-formed but incorrect result type. The precondition requires this
    /// to be REJECTED: a family that accepts it accepts anything, so its
    /// declared result reaches nothing and the probe measures the freeze
    /// default rather than the call.
    wrong_result: Option<&'static str>,
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
        declared_result: Some("tensor[3, 2, f32]"),
        wrong_result: Some("tensor[9, 9, f32]"),
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
        declared_result: Some("tensor[3, 2, f32]"),
        wrong_result: Some("tensor[9, 9, f32]"),
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
        declared_result: Some("tensor[3, 2, bool]"),
        wrong_result: Some("tensor[9, 9, bool]"),
        members: &["cmplt", "eq", "neq", "lt", "gt", "lte", "gte"],
    },
    Family {
        name: "integer binary",
        settlement: TensorSettlement::Constrains,
        representative: "bitand",
        dtype: "int32",
        call: "bitand(e, e)",
        extra_params: "",
        declared_result: Some("tensor[3, 2, int32]"),
        wrong_result: Some("tensor[9, 9, int32]"),
        members: &["mod", "bitand", "bitor", "bitxor", "shl", "shr"],
    },
    Family {
        name: "logical",
        settlement: TensorSettlement::Constrains,
        representative: "and",
        dtype: "bool",
        call: "and(e, e)",
        extra_params: "",
        declared_result: Some("tensor[3, 2, bool]"),
        wrong_result: Some("tensor[9, 9, bool]"),
        members: &["and", "or", "not"],
    },
    Family {
        name: "shape preserving",
        settlement: TensorSettlement::Constrains,
        // `clamp` binds nothing, so the family is represented by a member
        // whose declared result does bind. Measured, not chosen for tidiness.
        representative: "softmax",
        dtype: "f32",
        call: "softmax(e, 0)",
        extra_params: "",
        declared_result: Some("tensor[3, 2, f32]"),
        wrong_result: Some("tensor[9, 9, f32]"),
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
        declared_result: Some("tensor[6, f32]"),
        wrong_result: Some("tensor[9, f32]"),
        members: &["reshape"],
    },
    Family {
        name: "scatter elements",
        settlement: TensorSettlement::Constrains,
        representative: "scatter_elements",
        dtype: "f32",
        call: "scatter_elements(e, i, e, 0)",
        extra_params: ", i: tensor[3, 2, int32]",
        declared_result: Some("tensor[3, 2, f32]"),
        wrong_result: Some("tensor[9, 9, f32]"),
        members: &["scatter_elements"],
    },
    Family {
        name: "matmul",
        settlement: TensorSettlement::Constrains,
        representative: "matmul",
        dtype: "f32",
        call: "matmul(e, b)",
        extra_params: ", b: tensor[2, 7, f32]",
        declared_result: Some("tensor[3, 7, f32]"),
        wrong_result: Some("tensor[9, 9, f32]"),
        members: &["matmul"],
    },
    Family {
        name: "reduction",
        settlement: TensorSettlement::Propagates,
        representative: "sum",
        dtype: "f32",
        call: "sum(e, 0)",
        extra_params: "",
        declared_result: Some("tensor[2, f32]"),
        wrong_result: Some("tensor[9, 9, f32]"),
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
        settlement: TensorSettlement::Constrains,
        representative: "reduce_window_sum",
        dtype: "f32",
        call: "reduce_window_sum(e, [2i64, 2i64], [1i64, 1i64])",
        extra_params: "",
        // Same chelis#1512 artifact as the movement family above. With a
        // resolved tensor[3, 2, f32] operand this declaration is REJECTED and
        // tensor[2, 1, f32] is accepted, for all four members. After the repair
        // this family's declared result becomes tensor[2, 1, f32].
        declared_result: Some("tensor[3, 2, f32]"),
        wrong_result: Some("tensor[9, 9, f32]"),
        members: &[
            "reduce_window_max",
            "reduce_window_min",
            "reduce_window_sum",
            "reduce_window_mean",
        ],
    },
    Family {
        name: "shape carrying movement",
        settlement: TensorSettlement::Constrains,
        representative: "permute",
        dtype: "f32",
        call: "permute(e, 1, 0)",
        extra_params: "",
        // Accepted only because chelis#1512 skips `permute`'s own transpose
        // rule while the operand is pending, unifying the declared result with
        // the operand's type instead. The resolved control proves it: with a
        // resolved tensor[3, 2, f32] operand this declaration is REJECTED and
        // tensor[2, 3, f32] is accepted. When chelis#1512 is repaired this
        // family's declared result becomes tensor[2, 3, f32]; a red here after
        // that repair is the tripwire working, not a test to restore.
        // pad and shrink share the defect: their resolved-correct results are
        // tensor[5, 2, f32] and tensor[1, 1, f32].
        declared_result: Some("tensor[3, 2, f32]"),
        wrong_result: Some("tensor[9, 9, f32]"),
        members: &["permute", "pad", "shrink"],
    },
    // `expand` and `conv2d` measure as `Propagates` and neither can be proved
    // here, so they are their own family rather than riding on a
    // `Constrains` representative that does not describe them.
    Family {
        name: "unprovable movement",
        settlement: TensorSettlement::Propagates,
        representative: "expand",
        dtype: "f32",
        call: "expand(e, 0, 2i64)",
        extra_params: "",
        declared_result: Some("tensor[3, 2, f32]"),
        wrong_result: Some("tensor[9, 9, f32]"),
        members: &["expand", "conv2d"],
    },
    Family {
        name: "shape query",
        settlement: TensorSettlement::Propagates,
        representative: "rank",
        dtype: "f32",
        call: "rank(e)",
        extra_params: "",
        declared_result: Some("int32"),
        wrong_result: Some("string"),
        members: &["rank", "numel", "tensor_to_scalar"],
    },
    Family {
        name: "generic passthrough",
        settlement: TensorSettlement::Propagates,
        representative: "print",
        dtype: "f32",
        call: "print(e)",
        extra_params: "",
        declared_result: Some("unit"),
        wrong_result: Some("string"),
        members: &[
            "print",
            "to_string",
            "fail",
            "test_assert",
            "test_assert_eq",
            "test_assert_eq_tensor",
        ],
    },
    Family {
        name: "container producing",
        settlement: TensorSettlement::Propagates,
        representative: "to_list",
        dtype: "f32",
        call: "to_list(e)",
        extra_params: "",
        declared_result: Some("List[f32]"),
        wrong_result: Some("string"),
        members: &["split", "to_list", "einsum", "tensor_scan"],
    },
    Family {
        name: "io passthrough",
        settlement: TensorSettlement::Propagates,
        representative: "file_exists",
        dtype: "f32",
        call: "file_exists(e)",
        extra_params: "",
        declared_result: Some("bool"),
        wrong_result: Some("string"),
        members: &[
            "file_exists",
            "list_dir",
            "mmap_file",
            "mmap_len",
            "mmap_read",
            "process_run",
            "read_bytes",
            "read_file",
            "read_lines",
            "write_file",
        ],
    },
    Family {
        name: "positive test skipped",
        settlement: TensorSettlement::Propagates,
        representative: "len",
        dtype: "f32",
        call: "len(e)",
        extra_params: "",
        declared_result: Some("int64"),
        wrong_result: Some("string"),
        members: &[
            "append",
            "chunk",
            "dict_contains",
            "dict_entries",
            "dict_get",
            "dict_insert",
            "dict_keys",
            "dict_merge",
            "dict_of",
            "dict_remove",
            "dict_values",
            "drop",
            "enumerate",
            "flatten",
            "index",
            "len",
            "pad_sequences",
            "pad_sequences_to",
            "range",
            "scalar_to_tensor",
            "string_concat",
            "string_len",
            "string_slice",
            "string_trim",
            "take",
            "to_float",
            "to_int",
            "to_tensor",
            "zip",
        ],
    },
    Family {
        name: "shape neutral freeze",
        settlement: TensorSettlement::Freezes,
        representative: "shape",
        dtype: "f32",
        call: "shape(e, 0)",
        extra_params: "",
        // Present so the mutation matrix can declare this family `Constrains`
        // or `Propagates` and exercise those arms. The `Freezes` arm itself
        // does not read them.
        declared_result: Some("int64"),
        wrong_result: Some("string"),
        members: &["shape", "test_assert_close_tensor"],
    },
    Family {
        name: "refuses an unresolved operand",
        settlement: TensorSettlement::RejectsUnresolved,
        representative: "gather",
        dtype: "f32",
        call: "gather(e, idx, 0)",
        extra_params: ", idx: tensor[1, int32]",
        // Present so the mutation matrix can declare this family a settlement
        // value and exercise those arms. The `RejectsUnresolved` arm does not
        // read them, and the precondition skips this family for that reason.
        declared_result: Some("tensor[3, 2, f32]"),
        wrong_result: Some("tensor[9, 9, f32]"),
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
    Family {
        name: "csv ingest",
        settlement: TensorSettlement::RejectsUnresolved,
        representative: "to_csv",
        dtype: "f32",
        call: "to_csv(e)",
        extra_params: "",
        // As above: for the matrix, not for this family's own arm.
        declared_result: Some("string"),
        wrong_result: Some("int64"),
        members: &[
            "csv_cols",
            "csv_f64",
            "csv_f64s",
            "csv_int",
            "csv_ints",
            "csv_nrows",
            "csv_str",
            "csv_strs",
            "parse_csv",
            "to_csv",
        ],
    },
];

/// Families whose representative these probes cannot discriminate, with the
/// reason each resists one.
///
/// **This list is per family, not per row, and that is the whole rule.** A
/// representative stands for its family: a family whose representative can be
/// discriminated proves every member to the same degree, and one whose
/// representative cannot proves none of them. Mixing the two granularities is
/// how a count of proved rows goes wrong in both directions at once.
///
/// Three kinds, and the difference decides who can fix them. A representative
/// with no discriminating spelling will not yield to a better assertion; one
/// nobody could spell is simply unfinished; one whose declared result binds
/// nothing is chelis#1512 and no probe of this shape reaches it.
const UNDISCRIMINATED_FAMILIES: &[(&str, &str)] = &[
    (
        "container producing",
        "its representative `to_list` returns a container, so no declared result \
         discriminates; `split` and `einsum` share that and `tensor_scan` could \
         not be spelled at all",
    ),
    (
        "reduction",
        "its representative `sum` binds nothing: one body accepts -> f32, \
         -> tensor[1, f32], -> tensor[2, f32] and -> tensor[3, f32] alike, \
         chelis#1512",
    ),
    (
        "unprovable movement",
        "its representative `expand` binds nothing, and `conv2d` needs a rank-4 \
         producer and four arguments the shared probe shape cannot express",
    ),
    (
        "scatter elements",
        "its representative binds nothing and is the family's only member, so \
         nothing else can stand for it, chelis#1512",
    ),
];

/// Rows that individually accept any declared result, though their family's
/// representative binds and therefore carries the proof.
///
/// These are counted as proved, because the family is what the mechanism
/// proves. Recorded anyway: a reader who sees `clamp` among the proved rows
/// deserves to know that `clamp` on its own binds nothing, and that what proves
/// it is `softmax` standing for the family. Each is chelis#1512.
const BINDS_NOTHING_BUT_COVERED: &[&str] = &["where", "clamp", "cumsum", "sort", "drop"];

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

/// Does a diagnostic name an unresolved inference variable?
///
/// The number is allocation-order dependent and is deliberately not pinned: a
/// rebase that shifts allocation must not turn a test red. This is the property
/// that separates a route refusing the operand from one that considered its
/// candidate forms and admitted neither.
fn names_an_unresolved_variable(message: &str) -> bool {
    message
        .match_indices('?')
        .any(|(at, _)| message[at + 1..].starts_with(|c: char| c.is_ascii_digit()))
}

fn declared(name: &str) -> TensorSettlement {
    BUILTINS
        .iter()
        .find(|decl| decl.name == name)
        .unwrap_or_else(|| panic!("`{name}` is not a registered builtin"))
        .tensor_settlement
}

/// The one program shape both settlement arms use: a declared result on the
/// call and no other consumer, so the only thing that can move the operand off
/// its §4.7.2 freeze default is the call itself.
fn declared_result_program(family: &Family, result: &str) -> String {
    format!(
        "def f(a: tensor[2, {dtype}]{extra}) -> {result} = {{\n  \
         e = expand(a, 0, 3i64)\n  {call}\n}}\n",
        dtype = family.dtype,
        extra = family.extra_params,
        call = family.call,
    )
}

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

/// The `Freezes` arm's second producer, with the consumer's form as a
/// parameter so both candidate forms can be read.
///
/// `expand(b, 0, 6i64)` on `tensor[1, f32]` has candidate forms
/// `tensor[6, f32]` and `tensor[6, 1, f32]`, and **both hold six elements**, so
/// a rule that eliminates by element count cannot select between them. The
/// element type is fixed at `f32` rather than taken from the family, because
/// the only family that reaches this producer is `f32`; a freeze family of
/// another dtype would need its own.
fn under_producer_b(family: &Family, consumer: &str) -> Result<String, String> {
    check(&format!(
        "def sink_b(x: {consumer}) -> int32 = 0\n\
         def f(b: tensor[1, f32]{extra}) -> int32 = {{\n  \
         e = expand(b, 0, 6i64)\n  u = {call}\n  sink_b(e)\n}}\n",
        extra = family.extra_params,
        call = family.call,
    ))
}

fn insertion_form(dtype: &str) -> String {
    collapse(&format!(
        "(t-tensor {{}} (d-lit {{}} 3) (d-lit {{}} 2) (t-prim {{}} {dtype}))"
    ))
}

fn replacement_form(dtype: &str) -> String {
    collapse(&format!(
        "(t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} {dtype}))"
    ))
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
fn undiscriminated_families_are_named_with_a_reason() {
    let family_names = FAMILIES
        .iter()
        .map(|family| family.name)
        .collect::<BTreeSet<_>>();
    for (name, reason) in UNDISCRIMINATED_FAMILIES {
        assert!(
            !reason.trim().is_empty(),
            "family `{name}` needs a reason no probe discriminates it"
        );
        assert!(
            family_names.contains(name),
            "`{name}` is named as an undiscriminated family but is not a family"
        );
    }
    // The row-level note is about rows that ARE proved, by their family, so
    // every entry must sit in a family that is not itself undiscriminated.
    let unprovable = UNDISCRIMINATED_FAMILIES
        .iter()
        .map(|(name, _)| *name)
        .collect::<BTreeSet<_>>();
    for row in BINDS_NOTHING_BUT_COVERED {
        let family = FAMILIES
            .iter()
            .find(|family| family.members.contains(row))
            .unwrap_or_else(|| panic!("`{row}` belongs to no family"));
        assert!(
            !unprovable.contains(family.name),
            "`{row}` is noted as covered by its family, but family `{}` is \
             itself undiscriminated, so nothing proves it",
            family.name
        );
        assert_ne!(
            declared(row),
            TensorSettlement::NoTensorOperand,
            "`{row}` needs no probe if it admits no tensor"
        );
    }
}

/// The proved / covered-not-proved split, computed from the tables rather than
/// asserted, so the number in the pull request body cannot drift from the code.
#[test]
fn the_proved_and_unproved_counts_partition_every_covered_row() {
    let unprovable = UNDISCRIMINATED_FAMILIES
        .iter()
        .map(|(name, _)| *name)
        .collect::<BTreeSet<_>>();
    let mut proved = 0usize;
    let mut not_proved = 0usize;
    for family in FAMILIES {
        if unprovable.contains(family.name) {
            not_proved += family.members.len();
        } else {
            proved += family.members.len();
        }
    }
    let covered = BUILTINS
        .iter()
        .filter(|decl| decl.tensor_settlement != TensorSettlement::NoTensorOperand)
        .count();
    assert_eq!(
        proved + not_proved,
        covered,
        "every covered row is either proved by its family or not"
    );
    assert_eq!((proved, not_proved), (128, 15), "the split the body states");
    assert_eq!(BUILTINS.len() - covered, 9, "rows outside the coverage set");
}

// ---------------------------------------------------------------------
// The precondition: can the probe say anything at all?
// ---------------------------------------------------------------------

/// A settlement family's declared result must actually bind before its stamp
/// is evidence about the call.
///
/// A family that accepts a wrong result type accepts anything, so the operand's
/// stamp is the §4.7.2 freeze default rather than a consequence of the call,
/// and reading it would measure nothing. Such a family cannot be proved by a
/// probe whose only channel is the declared result, and it must say so in
/// `UNDISCRIMINATED` rather than assert through the gap.
#[test]
fn every_settlement_family_declared_result_binds() {
    let named = UNDISCRIMINATED_FAMILIES
        .iter()
        .map(|(name, _)| *name)
        .collect::<BTreeSet<_>>();
    let mut unprovable = Vec::new();
    for family in FAMILIES {
        // Only the two arms that read a declared result have a precondition;
        // `Freezes` and `RejectsUnresolved` probe the call directly and carry
        // declared results solely so the mutation matrix can exercise the other
        // arms on them.
        if !matches!(
            family.settlement,
            TensorSettlement::Constrains | TensorSettlement::Propagates
        ) {
            continue;
        }
        let (Some(declared_result), Some(wrong_result)) =
            (family.declared_result, family.wrong_result)
        else {
            continue;
        };
        let correct = check(&declared_result_program(family, declared_result));
        if let Err(error) = &correct {
            // A rejection naming an unresolved variable is not a misspelled
            // probe. It is the call refusing the operand before considering its
            // candidate forms, which is `RejectsUnresolved` and cannot be either
            // settlement value. Saying that is the discrimination; saying
            // "misspelled" would describe the fixture instead.
            assert!(
                !names_an_unresolved_variable(error),
                "family `{}` refuses an unresolved operand, so it cannot be \
                 {:?}: {error}",
                family.name,
                family.settlement
            );
        }
        assert!(
            correct.is_ok(),
            "family `{}`: its own declared result `{declared_result}` is \
             rejected, so the probe is misspelled rather than the family being \
             unprovable: {}",
            family.name,
            correct.clone().unwrap_err()
        );
        if check(&declared_result_program(family, wrong_result)).is_ok() {
            unprovable.push(family.name);
        }
    }
    let unnamed = unprovable
        .iter()
        .filter(|name| !named.contains(*name))
        .collect::<Vec<_>>();
    assert!(
        unnamed.is_empty(),
        "these families accept a wrong declared result, so their declared \
         result binds nothing and no probe of this shape can prove them. Each \
         must be named in UNDISCRIMINATED with that reason: {unnamed:?}"
    );
}

// ---------------------------------------------------------------------
// Behaviour, once per family.
// ---------------------------------------------------------------------

/// What this proves, and what it does not.
///
/// The `Constrains` arm proves the call transmits *an* equation and that the
/// equation lands on the insertion form. It does not prove the call would
/// transmit a *different* equation to the replacement form, because each
/// family carries one declared result. §4.7.2's own sentence allows a consumer
/// that "supplies no complete shape equation" to eliminate candidates by its
/// typing rule alone, so a family could eliminate by rank and still be
/// `Constrains`; this arm does not separate that from a full shape equation,
/// and it does not need to.
///
/// The `Propagates` arm is the mirror and asserts the operand stayed at its
/// freeze default, and additionally that the operand is still selectable after
/// the call, which a freezing call destroys. The two are mutually exclusive on
/// the same program, which is what makes the mutation in either direction go
/// red.
///
/// **Why the `Freezes` arm reads two producers and both consumers.** With one
/// producer it asserts only that a later insertion-form consumer is rejected,
/// and that follows from the operand resting on the replacement form however it
/// got there. A `Constrains` call whose own typing rule eliminates the
/// insertion candidate lands on the same form and is byte-identical:
/// `reshape(e, [3i64])`, where §4.7.3's element-count rule admits only the
/// replacement form, rejects with "tensor rank mismatch: 2 dims vs 1 dims",
/// exactly as `shape(e, 0)` does.
///
/// Producer B's candidate forms hold equal element counts, which is necessary
/// and not sufficient. Its rejection column does not separate the two either:
/// measured, `reshape(e, [3i64])` is rejected under producer B as well, with
/// "reshape target has 3 elements but input tensor has 6". What separates them
/// is the **acceptance**. A freeze leaves the operand on `tensor[6, f32]`, so a
/// consumer of that form is accepted while the insertion-form consumer is
/// rejected; a call whose own rule refused the operand rejects both. The arm
/// therefore requires the rejection under both producers and the acceptance
/// under producer B.
///
/// **What neither producer can separate, and the repair that creates it.** A
/// call that eliminates a candidate when its own rule can and freezes cleanly
/// when it cannot passes both producers. `to_list` is the registry's one
/// rank-upper-bound builtin (`to_list expects a rank-1 tensor, got rank 2
/// tensor`), and on a deferred operand that rule is skipped, which is
/// chelis#1512. So `to_list` accepts the insertion-form consumer today and
/// fails the rejection column rather than passing it. Repair chelis#1512 and
/// `to_list` begins eliminating by rank on a deferred operand, at which point
/// its two columns are identical to a freeze under both producers and nothing
/// here goes red. No producer built from a positional `expand` separates them,
/// because the insertion form always carries exactly one more dimension than
/// the replacement form. The sentinel
/// `to_list_accepts_the_insertion_consumer_until_chelis_1512_is_repaired`
/// turns that silent day into a red one.
///
/// Search scope, as run: 94 builtins, the operand in first argument position,
/// one of six argument fillers, the first spelling that was not an arity error
/// at both rank 1 and rank 2. Not covered: spellings with the tensor in a later
/// argument position, builtins needing a rank-4 producer or a callback, and
/// fillers beyond the six. A rank upper bound reachable only through one of
/// those was not found.
///
/// **Producer A is contingent, in the shape of the chelis#1512 labels above.**
/// `spec/05` §2.4.1, landing in chelis#1523, makes the same-rank `expand` form
/// well formed only over a unit operand extent. Producer A expands
/// `tensor[2, f32]`, whose operand extent is 2, so once Slice B's b2.5 enforces
/// that precondition, freezing producer A to the same-rank form is a type error
/// and this fixture reds. That red reads as the precondition being enforced,
/// not as the arm breaking and not as a test to restore. Producer B expands
/// `tensor[1, f32]`, a unit extent, and is unaffected.
///
/// Under enforcement the rejection itself becomes the freeze discriminator,
/// which is this arm's successor rather than a repair to it: a frozen same-rank
/// `expand` over a non-unit operand extent rejects, while a propagating call
/// leaves the operand open.
///
#[test]
fn each_family_representative_behaves_as_its_family_declares() {
    let named = UNDISCRIMINATED_FAMILIES
        .iter()
        .map(|(name, _)| *name)
        .collect::<BTreeSet<_>>();
    for family in FAMILIES {
        match family.settlement {
            TensorSettlement::Constrains | TensorSettlement::Propagates => {
                let (Some(declared_result), Some(wrong_result)) =
                    (family.declared_result, family.wrong_result)
                else {
                    panic!("family `{}` needs a declared result", family.name)
                };
                // A family whose declared result binds nothing is proved by
                // nothing here; the precondition test owns that verdict.
                if check(&declared_result_program(family, wrong_result)).is_ok() {
                    assert!(
                        named.contains(&family.name),
                        "family `{}` binds nothing and is not named in \
                         UNDISCRIMINATED_FAMILIES",
                        family.name
                    );
                    continue;
                }
                let rendered = check(&declared_result_program(family, declared_result))
                    .unwrap_or_else(|error| {
                        panic!("{}: expected acceptance, got {error}", family.name)
                    });
                let rendered = collapse(&rendered);
                let insertion = rendered.contains(&insertion_form(family.dtype));
                let replacement = rendered.contains(&replacement_form(family.dtype));
                if family.settlement == TensorSettlement::Constrains {
                    assert!(
                        insertion && !replacement,
                        "{}: a constraining call must carry the declared result \
                         onto the operand and select the insertion form; got \
                         insertion={insertion} replacement={replacement}",
                        family.name
                    );
                } else {
                    assert!(
                        replacement && !insertion,
                        "{}: a propagating call adds no evidence, so the operand \
                         must stay at its freeze default; got \
                         insertion={insertion} replacement={replacement}",
                        family.name
                    );
                    // The stamp alone cannot separate propagation from freezing:
                    // the freeze default IS the replacement form, so both read
                    // identically. Propagation additionally leaves the operand
                    // selectable, which is the property a freezing call destroys.
                    survives_the_call(family).unwrap_or_else(|error| {
                        panic!(
                            "{}: a propagating call leaves the operand selectable, \
                             so a later insertion-form consumer must still be \
                             accepted; a freezing call is what makes this fail: \
                             {error}",
                            family.name
                        )
                    });
                }
            }
            TensorSettlement::Freezes => {
                // The rejection column, under both producers. It is necessary
                // and on its own not sufficient: a call whose own typing rule
                // eliminates the insertion candidate is rejected here too.
                for (producer, probe) in [
                    ("producer A", survives_the_call(family)),
                    ("producer B", under_producer_b(family, "tensor[6, 1, f32]")),
                ] {
                    let error = probe.err().unwrap_or_else(|| {
                        panic!(
                            "{}: under {producer}, freezing fixes the operand at \
                             the replacement form, so the later insertion-form \
                             consumer must be rejected. A call accepted here \
                             selected by its own typing rule rather than \
                             freezing.",
                            family.name
                        )
                    });
                    // A shape disagreement alone is too coarse: a route that
                    // refuses an unresolved operand could in principle do so
                    // with a `DimensionMismatch` and pass.
                    let names_a_shape_disagreement =
                        error.contains("rank mismatch") || error.contains("DimensionMismatch");
                    assert!(
                        names_a_shape_disagreement && !names_an_unresolved_variable(&error),
                        "{}: under {producer}, the rejection must name a shape \
                         disagreement and must not name an unresolved variable, \
                         which is what separates freezing the operand from \
                         refusing it; got {error}",
                        family.name
                    );
                }
                // The acceptance column, which is where the two differ. A
                // freeze leaves the operand on producer B's replacement form,
                // so a consumer of that form is accepted. A call that refused
                // the operand rejects this consumer as well.
                under_producer_b(family, "tensor[6, f32]").unwrap_or_else(|error| {
                    panic!(
                        "{}: under producer B, freezing leaves the operand on \
                         the replacement form, so a consumer of that form must \
                         be accepted. A call rejected here refused the operand \
                         rather than freezing it: {error}",
                        family.name
                    )
                });
            }
            TensorSettlement::RejectsUnresolved => {
                let error = survives_the_call(family).expect_err(&format!(
                    "{}: this family refuses an unresolved operand outright",
                    family.name
                ));
                assert!(
                    names_an_unresolved_variable(&error),
                    "{}: the rejection must name an unresolved variable, which \
                     is what distinguishes refusing the operand from admitting \
                     neither of its candidate forms; got {error}",
                    family.name
                );
            }
            TensorSettlement::NoTensorOperand => {
                panic!("{}: a family cannot be NoTensorOperand", family.name)
            }
        }
    }
}

/// A labelled sentinel for the `Freezes` arm's documented pass-through, in the
/// shape of the chelis#1512 artifact labels above.
///
/// `to_list` accepts the insertion-form consumer on a deferred operand because
/// its rank rule is skipped while the operand is pending. Repairing
/// chelis#1512 makes that rule apply, `to_list` starts eliminating the
/// insertion candidate by rank, and its two columns become indistinguishable
/// from a freeze under both producers.
///
/// A red here after that repair means the `Freezes` arm has acquired the
/// pass-through its doc comment describes. It does not mean `to_list` broke and
/// this is not a test to restore: what the red asks for is a discriminator the
/// arm does not have, not a change to this assertion.
#[test]
fn to_list_accepts_the_insertion_consumer_until_chelis_1512_is_repaired() {
    let family = FAMILIES
        .iter()
        .find(|family| family.representative == "to_list")
        .expect("`to_list` is a family representative");
    for (producer, probe) in [
        ("producer A", survives_the_call(family)),
        ("producer B", under_producer_b(family, "tensor[6, 1, f32]")),
    ] {
        if let Err(error) = probe {
            panic!(
                "under {producer}, `to_list` on a deferred operand still \
                 accepts the insertion-form consumer. A rejection here means \
                 its rank rule now applies to a pending operand, so \
                 chelis#1512 has been repaired and the `Freezes` arm has \
                 acquired the pass-through its doc comment describes: {error}"
            )
        }
    }
    // The control: the rule already rejects a RESOLVED rank-2 operand, so a red
    // above is that rule reaching a pending operand rather than an inference.
    let resolved = check("def f(a: tensor[3, 2, f32]) -> int32 = {\n  u = to_list(a)\n  0\n}\n");
    assert!(
        matches!(&resolved, Err(error) if error.contains("rank-1 tensor")),
        "control: `to_list` must already reject a resolved rank-2 operand with \
         its own rank rule, so a red above is that rule reaching a pending \
         operand rather than an inference; got {resolved:?}"
    );
}
