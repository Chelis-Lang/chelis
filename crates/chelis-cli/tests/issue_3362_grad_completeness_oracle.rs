//! Chelis-Lang/chelis#3362: the spec-derived `grad` completeness oracle.
//!
//! spec/05 §5 requires every numeric callable to state an adjoint, a zero
//! cotangent, or a structural `grad` rejection, and forbids an adjoint from a
//! backend fallback. A fourth outcome, "application of `<op>` has no numeric
//! IR lowering", is the defect class this oracle closes.
//!
//! The inventory is read live from the builtin identity registry
//! (`spec/registry/builtin_semantic_identities.md`): every surface `Numeric`
//! identity must appear in [`CONTRACTS`] exactly once, and no entry may name an
//! identity the registry lacks, so a new atom row fails here until it is
//! classified. CamelCase identities are IR-internal adjoint nodes with no
//! surface spelling. A row is either a float adjoint with a minimal
//! application, or carries a phrase that must occur verbatim in its governing
//! atom, so the classification is anchored in the numbered spec rather than
//! asserted here. Every float-adjoint application must lower `grad` in eval
//! and in C emission at every float dtype, except the explicitly named
//! residual gaps below, which must keep failing exactly as recorded so a fix
//! removes them.
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

enum Contract {
    /// The atom states a float adjoint; the application takes one parameter
    /// `x` of the given type (`P` is the float dtype).
    Adjoint(&'static str, &'static str),
    /// The atom's verbatim statement that no float adjoint exists, in a
    /// sentence that names the identity, or in an atom all of whose surface
    /// identities lack a float adjoint (so a family sentence is unambiguous).
    NoFloatAdjoint(&'static str),
    /// As [`NoFloatAdjoint`], in a mixed atom whose statement designates the
    /// identity by this family noun rather than by name. The oracle cannot
    /// verify that a noun covers an identity, so only the identities in
    /// [`FAMILY_DESIGNATED`] may use it.
    NoFloatFamily(&'static str, &'static str),
}
use Contract::{Adjoint, NoFloatAdjoint, NoFloatFamily};

const CONTRACTS: &[(&str, Contract)] = &[
    (
        "abs",
        Adjoint("tensor[2, 2, P]", "sum(sum(abs(x), 1i32), 0i32)"),
    ),
    (
        "add",
        Adjoint("tensor[2, 2, P]", "sum(sum(add(x, x), 1i32), 0i32)"),
    ),
    ("and", NoFloatAdjoint("`grad` rejects it")),
    ("argmax_reduce", NoFloatAdjoint("`grad` rejects it")),
    ("argmin_reduce", NoFloatAdjoint("non-differentiability")),
    (
        "atan",
        Adjoint("tensor[2, 2, P]", "sum(sum(atan(x), 1i32), 0i32)"),
    ),
    (
        "bitand",
        NoFloatAdjoint("structurally reject differentiation"),
    ),
    (
        "bitor",
        NoFloatAdjoint("structurally reject differentiation"),
    ),
    (
        "bitxor",
        NoFloatAdjoint("structurally reject differentiation"),
    ),
    (
        "cast",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(cast(cast(x, f64), P), 1i32), 0i32)",
        ),
    ),
    ("cast_saturate", NoFloatAdjoint("`grad` rejects it")),
    ("cast_trunc", NoFloatAdjoint("non-differentiable")),
    ("cast_wrap", NoFloatAdjoint("`grad` rejects it")),
    (
        "ceil",
        NoFloatAdjoint("float floor/ceil/round structurally reject"),
    ),
    (
        "clamp",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(clamp(x, scalar_to_tensor(-0.5P), scalar_to_tensor(0.5P)), 1i32), 0i32)",
        ),
    ),
    ("cmplt", NoFloatAdjoint("zero cotangent")),
    (
        "conv",
        Adjoint(
            "tensor[1, 1, 3, P]",
            "sum(sum(sum(conv(x, to_tensor([[[0.5P, -1.0P]]]), [1i64], [(0i64, 0i64)]), 2i32), 1i32), 0i32)",
        ),
    ),
    (
        "cos",
        Adjoint("tensor[2, 2, P]", "sum(sum(cos(x), 1i32), 0i32)"),
    ),
    (
        "count",
        NoFloatAdjoint("AdRejectionReason::IntegerReductionOutput"),
    ),
    (
        "cumsum",
        Adjoint("tensor[2, 2, P]", "sum(sum(cumsum(x, 1i32), 1i32), 0i32)"),
    ),
    (
        "diagonal",
        Adjoint("tensor[2, 2, P]", "sum(diagonal(x, 0i32, 1i32), 0i32)"),
    ),
    (
        "div",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(div(x, add(mul(x, x), to_tensor([[1.0P, 1.0P], [1.0P, 1.0P]]))), 1i32), 0i32)",
        ),
    ),
    (
        "dropout",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(dropout(key_from_seed(7i64), x, 0.5P), 1i32), 0i32)",
        ),
    ),
    (
        "einsum",
        Adjoint("tensor[2, 2, P]", "einsum(\"ij,ij->\", x, x)"),
    ),
    ("eq", NoFloatAdjoint("zero cotangent")),
    (
        "erf",
        Adjoint("tensor[2, 2, P]", "sum(sum(erf(x), 1i32), 0i32)"),
    ),
    (
        "erfc",
        Adjoint("tensor[2, 2, P]", "sum(sum(erfc(x), 1i32), 0i32)"),
    ),
    (
        "exp",
        Adjoint("tensor[2, 2, P]", "sum(sum(exp(x), 1i32), 0i32)"),
    ),
    (
        "expand",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(expand(shrink(x, [[0i64, 2i64], [0i64, 1i64]]), 1i32, 3i64), 1i32), 0i32)",
        ),
    ),
    (
        "floor",
        NoFloatAdjoint("float floor/ceil/round structurally reject"),
    ),
    (
        "floor_div",
        NoFloatFamily(
            "structurally reject differentiation",
            "floor and truncation operations",
        ),
    ),
    ("fold_in", NoFloatAdjoint("carries no cotangent")),
    (
        "gather",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(gather(x, to_tensor([1i32, 0i32, 1i32]), 0i32), 1i32), 0i32)",
        ),
    ),
    (
        "gelu",
        Adjoint("tensor[2, 2, P]", "sum(sum(gelu(x), 1i32), 0i32)"),
    ),
    (
        "gelu_tanh",
        Adjoint("tensor[2, 2, P]", "sum(sum(gelu_tanh(x), 1i32), 0i32)"),
    ),
    ("gt", NoFloatAdjoint("zero cotangent")),
    ("gte", NoFloatAdjoint("zero cotangent")),
    (
        "guarded_fail",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(if lt(tensor_to_scalar(sum(sum(x, 1i32), 0i32)), 100.0P) then x else fail(\"too big\"), 1i32), 0i32)",
        ),
    ),
    (
        "insert",
        Adjoint(
            "tensor[2, P]",
            "sum(sum(insert(x, 1i32, 3i64), 1i32), 0i32)",
        ),
    ),
    (
        "key_from_seed",
        NoFloatAdjoint("the seed receives no cotangent"),
    ),
    (
        "layer_norm",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(layer_norm(x, to_tensor([1.5P, 0.5P]), to_tensor([0.1P, -0.2P]), 0.00001P), 1i32), 0i32)",
        ),
    ),
    (
        "log",
        Adjoint("tensor[2, 2, P]", "sum(sum(log(mul(x, x)), 1i32), 0i32)"),
    ),
    ("lt", NoFloatAdjoint("zero cotangent")),
    ("lte", NoFloatAdjoint("zero cotangent")),
    (
        "matmul",
        Adjoint("tensor[2, 2, P]", "sum(sum(matmul(x, x), 1i32), 0i32)"),
    ),
    (
        "max_elem",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(max_elem(x, to_tensor([[0.0P, 0.0P], [0.0P, 0.0P]])), 1i32), 0i32)",
        ),
    ),
    (
        "max_reduce",
        Adjoint("tensor[2, 2, P]", "sum(max_reduce(x, 1i32), 0i32)"),
    ),
    (
        "mean",
        Adjoint("tensor[2, 2, P]", "sum(mean(x, 1i32), 0i32)"),
    ),
    (
        "min_elem",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(min_elem(x, to_tensor([[0.0P, 0.0P], [0.0P, 0.0P]])), 1i32), 0i32)",
        ),
    ),
    (
        "min_reduce",
        Adjoint("tensor[2, 2, P]", "sum(min_reduce(x, 1i32), 0i32)"),
    ),
    ("mod", NoFloatAdjoint("structurally reject differentiation")),
    (
        "mul",
        Adjoint("tensor[2, 2, P]", "sum(sum(mul(x, x), 1i32), 0i32)"),
    ),
    (
        "neg",
        Adjoint("tensor[2, 2, P]", "sum(sum(neg(x), 1i32), 0i32)"),
    ),
    ("neq", NoFloatAdjoint("zero cotangent")),
    ("not", NoFloatAdjoint("non-differentiable")),
    (
        "numel",
        NoFloatFamily(
            "Shape observations have zero cotangent",
            "Shape observations",
        ),
    ),
    (
        "or",
        NoFloatAdjoint("differentiation contract of [05-OP-26]"),
    ),
    (
        "pad",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(pad(x, [[1i64, 0i64], [0i64, 2i64]], 0.0P), 1i32), 0i32)",
        ),
    ),
    (
        "permute",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(mul(permute(x, 1i32, 0i32), x), 1i32), 0i32)",
        ),
    ),
    (
        "prod_reduce",
        Adjoint("tensor[2, 2, P]", "sum(prod_reduce(x, 1i32), 0i32)"),
    ),
    (
        "rank",
        NoFloatFamily(
            "Shape observations have zero cotangent",
            "Shape observations",
        ),
    ),
    (
        "recip",
        Adjoint("tensor[2, 2, P]", "sum(sum(recip(x), 1i32), 0i32)"),
    ),
    (
        "reduce_window_max",
        Adjoint(
            "tensor[4, P]",
            "sum(reduce_window_max(x, [2i64], [2i64]), 0i32)",
        ),
    ),
    (
        "reduce_window_mean",
        Adjoint(
            "tensor[4, P]",
            "sum(reduce_window_mean(x, [2i64], [2i64]), 0i32)",
        ),
    ),
    (
        "reduce_window_min",
        Adjoint(
            "tensor[4, P]",
            "sum(reduce_window_min(x, [2i64], [2i64]), 0i32)",
        ),
    ),
    (
        "reduce_window_sum",
        Adjoint(
            "tensor[4, P]",
            "sum(reduce_window_sum(x, [2i64], [2i64]), 0i32)",
        ),
    ),
    (
        "relu",
        Adjoint("tensor[2, 2, P]", "sum(sum(relu(x), 1i32), 0i32)"),
    ),
    (
        "reshape",
        Adjoint("tensor[2, 2, P]", "sum(reshape(x, [4i64]), 0i32)"),
    ),
    (
        "round",
        NoFloatAdjoint("float floor/ceil/round structurally reject"),
    ),
    ("round_to", NoFloatAdjoint("structurally rejected")),
    (
        "scalar_to_tensor",
        Adjoint(
            "tensor[2, 2, P]",
            "scalar_to_tensor(tensor_to_scalar(sum(sum(x, 1i32), 0i32)))",
        ),
    ),
    (
        "scatter",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(scatter(neg(x), to_tensor([0i32]), shrink(x, [[1i64, 2i64], [0i64, 2i64]]), 0i32, \"add\"), 1i32), 0i32)",
        ),
    ),
    (
        "scatter_elements",
        NoFloatAdjoint("structurally reject differentiation"),
    ),
    (
        "scatter_replace",
        NoFloatAdjoint("structurally reject differentiation"),
    ),
    ("shape", NoFloatAdjoint("zero-cotangent adjoint")),
    ("shl", NoFloatAdjoint("structurally reject differentiation")),
    ("shr", NoFloatAdjoint("structurally reject differentiation")),
    (
        "shrink",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(shrink(x, [[0i64, 1i64], [0i64, 2i64]]), 1i32), 0i32)",
        ),
    ),
    (
        "sigmoid",
        Adjoint("tensor[2, 2, P]", "sum(sum(sigmoid(x), 1i32), 0i32)"),
    ),
    (
        "silu",
        Adjoint("tensor[2, 2, P]", "sum(sum(silu(x), 1i32), 0i32)"),
    ),
    (
        "sin",
        Adjoint("tensor[2, 2, P]", "sum(sum(sin(x), 1i32), 0i32)"),
    ),
    (
        "softmax",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(mul(softmax(x, 1i32), x), 1i32), 0i32)",
        ),
    ),
    (
        "sort",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(mul(sort(x, 1i32).0, x), 1i32), 0i32)",
        ),
    ),
    ("split_key", NoFloatAdjoint("neither carries a cotangent")),
    ("split_keys", NoFloatAdjoint("No row carries a cotangent")),
    (
        "sqrt",
        Adjoint("tensor[2, 2, P]", "sum(sum(sqrt(mul(x, x)), 1i32), 0i32)"),
    ),
    (
        "standard_normal_cdf",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(standard_normal_cdf(x), 1i32), 0i32)",
        ),
    ),
    (
        "stride",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(stride(x, 2i64, 1i64), 1i32), 0i32)",
        ),
    ),
    (
        "sub",
        Adjoint("tensor[2, 2, P]", "sum(sum(sub(x, mul(x, x)), 1i32), 0i32)"),
    ),
    ("sum", Adjoint("tensor[2, 2, P]", "sum(sum(x, 1i32), 0i32)")),
    (
        "tan",
        Adjoint("tensor[2, 2, P]", "sum(sum(tan(x), 1i32), 0i32)"),
    ),
    (
        "tanh",
        Adjoint("tensor[2, 2, P]", "sum(sum(tanh(x), 1i32), 0i32)"),
    ),
    (
        "tensor_to_scalar",
        Adjoint(
            "tensor[2, 2, P]",
            "scalar_to_tensor(mul(tensor_to_scalar(sum(sum(x, 1i32), 0i32)), 2.0P))",
        ),
    ),
    ("trace", Adjoint("tensor[2, 2, P]", "trace(x, 0i32, 1i32)")),
    (
        "trunc_div",
        NoFloatFamily(
            "structurally reject differentiation",
            "floor and truncation operations",
        ),
    ),
    (
        "uniform_like",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(uniform_like(key_from_seed(3i64), x, tensor_to_scalar(sum(sum(x, 1i32), 0i32)), 4.0P), 1i32), 0i32)",
        ),
    ),
    (
        "where",
        Adjoint(
            "tensor[2, 2, P]",
            "sum(sum(where(gt(x, to_tensor([[0.0P, 0.0P], [0.0P, 0.0P]])), x, neg(x)), 1i32), 0i32)",
        ),
    ),
];

/// Float adjoints whose `grad` still reaches the fallback, each with the
/// chelis#3362 sub-issue that owns its lowering. Each must keep failing with
/// the fallback diagnostic; once its lowering lands this oracle fails until
/// the entry is removed.
const FALLBACK_GAPS: &[(&str, u32)] = &[("clamp", 3374), ("scatter", 3376), ("sort", 3375)];

/// C emission rejections owned by another class: the C reduce kernels for
/// these identities emit only f32 (chelis#174 T5-d). Each must keep failing at
/// the non-f32 widths with exactly this diagnostic.
const C_WIDTH_GAPS: &[(&str, u32, &str)] = &[
    ("prod_reduce", 174, "unimplemented chelis#729"),
    ("reduce_window_max", 174, "on f32 tensors only"),
    ("reduce_window_mean", 174, "on f32 tensors only"),
    ("reduce_window_min", 174, "on f32 tensors only"),
    ("reduce_window_sum", 174, "on f32 tensors only"),
];

/// The reviewed identities whose atom states their missing float adjoint
/// only through a family noun: [05-OP-64]'s "floor and truncation
/// operations" and [05-OP-50]'s "Shape observations".
const FAMILY_DESIGNATED: [&str; 4] = ["floor_div", "numel", "rank", "trunc_div"];

const FALLBACK: &str = "has no numeric IR lowering";
const FLOATS: [&str; 4] = ["f16", "bf16", "f32", "f64"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// `(identity, atom)` for every `Numeric` row of the registry.
fn registry_rows() -> Vec<(String, String)> {
    let text =
        std::fs::read_to_string(repo_root().join("spec/registry/builtin_semantic_identities.md"))
            .expect("registry");
    let rows: Vec<(String, String)> = text
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("| `Numeric:")?;
            let (name, rest) = rest.split_once(':')?;
            let atom = rest.split('[').nth(1)?.split(']').next()?;
            Some((name.to_string(), atom.to_string()))
        })
        .collect();
    assert!(
        rows.len() > 90,
        "the registry parse found only {} Numeric rows",
        rows.len()
    );
    rows
}

/// The text of `> **[atom]**` through the end of its quoted block, with the
/// quote markers removed and whitespace collapsed.
fn atom_text(spec: &str, atom: &str) -> String {
    let head = format!("> **[{atom}]**");
    let mut lines = spec.lines().skip_while(|line| !line.starts_with(&head));
    let mut block = Vec::new();
    for line in lines.by_ref() {
        let Some(body) = line.strip_prefix('>') else {
            break;
        };
        block.push(body.trim());
    }
    assert!(!block.is_empty(), "spec/05 has no `{head}` atom");
    block
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Sentences of collapsed atom text, split after `.`, `;` or `:` followed by
/// whitespace.
fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        if matches!(bytes[i], b'.' | b';' | b':') && bytes[i + 1] == b' ' {
            out.push(text[start..=i].trim());
            start = i + 1;
        }
    }
    out.push(text[start..].trim());
    out.retain(|sentence| !sentence.is_empty());
    out
}

/// Whether every `_`-separated token of `identity` is a word of `text`,
/// ignoring connective tokens such as `to` when the identity has others.
fn names_identity(text: &str, identity: &str) -> bool {
    let words: Vec<String> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .map(str::to_ascii_lowercase)
        .collect();
    let tokens: Vec<&str> = identity.split('_').collect();
    let significant: Vec<&str> = tokens
        .iter()
        .copied()
        .filter(|token| token.len() > 2)
        .collect();
    let tokens = if significant.is_empty() {
        tokens
    } else {
        significant
    };
    tokens
        .iter()
        .all(|token| words.iter().any(|word| word == token))
}

/// Clauses of the atom that state a differentiation rule without a rejection
/// or zero marker: the clauses of its `Adjoint:` paragraph, and any clause
/// mentioning an adjoint or a cotangent. In an atom governing several identities only the
/// clauses naming `identity` count; an atom's sole identity owns them all.
fn float_adjoint_clauses(text: &str, identity: Option<&str>) -> Vec<String> {
    const MARKERS: [&str; 15] = [
        "neither carries",
        "no row carries",
        "reject",
        "non-differentiab",
        "forward-only",
        "no cotangent",
        "zero cotangent",
        "zero-cotangent",
        "carries no",
        "carry no",
        "carries none",
        "receives no",
        "receives none",
        "receives zero",
        "no adjoint",
    ];
    let adjoint_paragraph = text
        .split_once("Adjoint:")
        .map(|(_, rest)| rest.split(" Accumulator:").next().unwrap_or(rest))
        .unwrap_or("");
    let mut clauses: Vec<&str> = adjoint_paragraph.split([',', ';', '.']).collect();
    clauses.extend(text.split([',', ';', '.']).filter(|clause| {
        let lower = clause.to_ascii_lowercase();
        lower.contains("adjoint") || lower.contains("cotangent")
    }));
    clauses
        .into_iter()
        .filter(|clause| clause.chars().any(char::is_alphanumeric))
        .filter(|clause| identity.is_none_or(|identity| names_identity(clause, identity)))
        .filter(|clause| {
            let lower = clause.to_ascii_lowercase();
            !MARKERS.iter().any(|marker| lower.contains(marker))
        })
        .map(|clause| clause.trim().to_string())
        .collect()
}

fn input_literal(ty: &str) -> &'static str {
    match ty {
        "tensor[2, 2, P]" => "[[0.3P, -0.7P], [1.1P, 0.4P]]",
        "tensor[2, P]" => "[0.3P, -0.7P]",
        "tensor[4, P]" => "[0.3P, -0.7P, 1.1P, 0.4P]",
        "tensor[1, 1, 3, P]" => "[[[0.3P, -0.7P, 1.1P]]]",
        other => panic!("no oracle input for `{other}`"),
    }
}

fn program(names: &[&str], dtype: &str) -> String {
    let mut defs = String::from("module Demo.Main\n");
    let mut roots = String::new();
    for &name in names {
        let Some((_, Adjoint(ty, body))) = CONTRACTS.iter().find(|(n, _)| *n == name) else {
            panic!("`{name}` is not a float-adjoint row");
        };
        defs.push_str(&format!("def f_{name}(x: {ty}) -> tensor[P] = {body}\n"));
        roots.push_str(&format!(
            "g_{name} = grad(f_{name})(to_tensor({}))\n",
            input_literal(ty)
        ));
    }
    (defs + &roots).replace('P', dtype)
}

/// Run `chelis eval` (or, with `emit_c`, `chelis build --emit-c`) over the
/// applications; returns stderr on failure.
fn lower(names: &[&str], dtype: &str, emit_c: bool) -> Result<String, String> {
    let (_dir, reef, app) = common::make_app("issue-3362-oracle");
    common::write_file(&app.join("src/main.ch"), &program(names, dtype));
    let out_dir = app.join("c-out");
    let mut command = Command::cargo_bin("chelis").unwrap();
    command
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef)
        .current_dir(&app);
    if emit_c {
        command.args([
            "build",
            "src/main.ch",
            "--emit-c",
            "--output",
            out_dir.to_str().unwrap(),
        ]);
    } else {
        command.args(["eval", "--file", "src/main.ch"]);
    }
    let output = command.output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if output.status.success() {
        Ok(String::from_utf8(output.stdout).unwrap())
    } else {
        Err(stderr)
    }
}

/// Lower the batch; on failure, name every failing member individually.
fn assert_batch_lowers(names: &[&str], dtype: &str, emit_c: bool) {
    let lane = if emit_c { "C emission" } else { "eval" };
    match lower(names, dtype, emit_c) {
        Ok(stdout) => {
            if !emit_c {
                for name in names {
                    assert!(
                        stdout.contains(&format!("g_{name} = ")),
                        "{lane} {dtype}: no gradient for `{name}`"
                    );
                }
            }
        }
        Err(_) => {
            let failures: Vec<String> = names
                .iter()
                .filter_map(|name| {
                    lower(&[name], dtype, emit_c)
                        .err()
                        .map(|stderr| format!("`{name}`: {}", stderr.trim()))
                })
                .collect();
            panic!(
                "{lane} {dtype}: grad does not lower for:\n{}",
                failures.join("\n")
            );
        }
    }
}

#[test]
fn every_numeric_identity_is_classified_against_its_atom() {
    let spec =
        std::fs::read_to_string(repo_root().join("spec/05-risc-primitives.md")).expect("spec/05");
    let registry = registry_rows();
    let surface: BTreeSet<&str> = registry
        .iter()
        .map(|(name, _)| name.as_str())
        .filter(|name| name.starts_with(|c: char| c.is_ascii_lowercase()))
        .collect();
    let classified: BTreeSet<&str> = CONTRACTS.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        classified.len(),
        CONTRACTS.len(),
        "a CONTRACTS row is duplicated"
    );
    let unclassified: Vec<_> = surface.difference(&classified).collect();
    let stale: Vec<_> = classified.difference(&surface).collect();
    assert!(
        unclassified.is_empty() && stale.is_empty(),
        "registry Numeric identities without a contract row: {unclassified:?}; rows without a registry identity: {stale:?}"
    );
    for (name, atom) in &registry {
        let (phrase, family) = match CONTRACTS.iter().find(|(n, _)| n == name) {
            Some((_, NoFloatAdjoint(phrase))) => (*phrase, None),
            Some((_, NoFloatFamily(phrase, family))) => {
                assert!(
                    FAMILY_DESIGNATED.contains(&name.as_str()),
                    "`{name}` uses a family designation outside the reviewed FAMILY_DESIGNATED set"
                );
                (*phrase, Some(*family))
            }
            _ => continue,
        };
        let text = atom_text(&spec, atom);
        let statements: Vec<&str> = sentences(&text)
            .into_iter()
            .filter(|sentence| sentence.contains(phrase))
            .collect();
        assert!(
            !statements.is_empty(),
            "`{name}` is classified without a float adjoint, but its atom [{atom}] does not state `{phrase}`"
        );
        let homogeneous = registry
            .iter()
            .filter(|(other, other_atom)| other_atom == atom && surface.contains(other.as_str()))
            .all(|(other, _)| {
                !matches!(
                    CONTRACTS.iter().find(|(n, _)| n == other),
                    Some((_, Adjoint(..)))
                )
            });
        let designated = statements.iter().any(|sentence| match family {
            None => homogeneous || names_identity(sentence, name),
            Some(family) => sentence.contains(family),
        });
        assert!(
            designated,
            "`{name}`: no sentence of [{atom}] stating `{phrase}` designates it{}",
            family.map_or(String::new(), |family| format!(" as `{family}`"))
        );
        let sole = registry
            .iter()
            .filter(|(other, other_atom)| other_atom == atom && surface.contains(other.as_str()))
            .count()
            == 1;
        let adjoint_clauses = float_adjoint_clauses(&text, (!sole).then_some(name.as_str()));
        assert!(
            adjoint_clauses.is_empty(),
            "`{name}` is classified without a float adjoint, but [{atom}] states one: {adjoint_clauses:?}"
        );
    }
    for gap in FALLBACK_GAPS
        .iter()
        .map(|(name, _)| name)
        .chain(C_WIDTH_GAPS.iter().map(|(name, _, _)| name))
    {
        assert!(
            matches!(
                CONTRACTS.iter().find(|(n, _)| n == gap),
                Some((_, Adjoint(..)))
            ),
            "gap `{gap}` is not a float-adjoint row"
        );
    }
}

fn adjoint_rows() -> Vec<&'static str> {
    CONTRACTS
        .iter()
        .filter(|(name, contract)| {
            matches!(contract, Adjoint(..)) && !FALLBACK_GAPS.iter().any(|(gap, _)| gap == name)
        })
        .map(|(name, _)| *name)
        .collect()
}

#[test]
fn grad_of_every_float_adjoint_lowers_in_eval_at_every_float_width() {
    let names = adjoint_rows();
    for dtype in FLOATS {
        assert_batch_lowers(&names, dtype, false);
    }
}

#[test]
fn grad_of_every_float_adjoint_lowers_in_c_emission_at_every_float_width() {
    for dtype in FLOATS {
        let names: Vec<&str> = adjoint_rows()
            .into_iter()
            .filter(|name| dtype == "f32" || !C_WIDTH_GAPS.iter().any(|(gap, _, _)| gap == name))
            .collect();
        assert_batch_lowers(&names, dtype, true);
    }
}

#[test]
fn recorded_gaps_still_fail_exactly_as_recorded() {
    for (name, issue) in FALLBACK_GAPS {
        for emit_c in [false, true] {
            let stderr = lower(&[name], "f32", emit_c).err().unwrap_or_else(|| {
                panic!("`{name}` now lowers; remove it from FALLBACK_GAPS (chelis#{issue})")
            });
            assert!(
                stderr.contains(FALLBACK),
                "`{name}` fails differently: {stderr}"
            );
        }
    }
    for (name, issue, diagnostic) in C_WIDTH_GAPS {
        let stderr = lower(&[name], "f64", true).err().unwrap_or_else(|| {
            panic!("`{name}` now emits C at f64; remove it from C_WIDTH_GAPS (chelis#{issue})")
        });
        assert!(
            stderr.contains(diagnostic) && !stderr.contains(FALLBACK),
            "`{name}` fails differently at f64: {stderr}"
        );
    }
}
