//! chelis#3114: a bracket literal is a `List` unless a tensor is declared for
//! it, and every spelling survives a Deep round trip.
//!
//! `spec/02-surf-syntax.md` §P10b and `spec/04-type-system.md` §5.6: a bracket
//! literal is a tensor literal only as the argument of `to_tensor(...)`, or as
//! the bare right-hand side of a binding (inline annotation or `sig`) or body
//! of a function (inline result or `sig`) whose declared type is a tensor.
//! Neither element spelling nor any other position selects the kind. The
//! `spec/02-surf-syntax.md` §0.1 law `desugar(resugar(deep)) = deep`, modulo
//! the derived metadata `normalize_deep_for_surface_roundtrip` removes, must
//! hold for every combination, including the rejected ones.
//!
//! The matrix crosses element spellings, positions, declared types and the
//! two tensor-literal spellings. Macro expansion runs before resugaring, as
//! `chelis deep` and `chelis surf` do, so a macro contributes its expansion.

use chelis_deep::Expr;
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_program;
use chelis_surf::parser::parse_str;
use chelis_surf::resugar::{normalize_deep_for_surface_roundtrip, resugar_program};
use chelis_types::infer::infer_program;

const PRELUDE: &str = "a = 1.5f64\nmacro half() = 1.5\nmacro minus_half() = neg(1.5)\n";

/// Element spellings of a two-element literal. The f64 declarations below
/// differ from the f32 literal default, so a lost or gained adoption shows.
const FLAT_FORMS: &[(&str, &str)] = &[
    ("literal", "[1.5, 2.5]"),
    ("negated literal", "[-1.5, 2.5]"),
    ("neg call", "[neg(1.5), 2.5]"),
    ("ascribed literal", "[(1.5 : f32), 2.5]"),
    ("suffixed literal", "[1.5f64, 2.5]"),
    ("macro literal", "[half(), 2.5]"),
    ("macro neg call", "[minus_half(), 2.5]"),
    ("cast", "[cast(1.5, f64), 2.5]"),
    ("variable", "[a, 2.5]"),
];

const NESTED_FORM: (&str, &str) = ("nested", "[[1.5, 2.5], [-3.5, neg(4.5)]]");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Declared {
    Nothing,
    List,
    Tensor,
    TensorAlias,
    ListAlias,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Spelling {
    Bare,
    ToTensor,
    /// `[...] |> to_tensor`, the pipe spelling of the same call.
    PipeToTensor,
}

struct Position {
    name: &'static str,
    /// Whether a bare bracket literal here is converted by its declaration.
    declares_kind: bool,
    /// How many times the literal appears.
    occurrences: usize,
    /// Renders the program from the literal and an optional declared type.
    render: fn(&str, Option<&str>) -> Option<String>,
}

fn annotation(ty: Option<&str>) -> String {
    ty.map(|ty| format!(": {ty}")).unwrap_or_default()
}

fn result_annotation(ty: Option<&str>) -> String {
    ty.map(|ty| format!(" -> {ty}")).unwrap_or_default()
}

const POSITIONS: &[Position] = &[
    Position {
        name: "top-level binding",
        declares_kind: true,
        occurrences: 1,
        render: |literal, ty| Some(format!("values{} = {literal}\n", annotation(ty))),
    },
    Position {
        name: "top-level binding with sig",
        declares_kind: true,
        occurrences: 1,
        render: |literal, ty| ty.map(|ty| format!("sig values: {ty}\nvalues = {literal}\n")),
    },
    Position {
        name: "block binding",
        declares_kind: true,
        occurrences: 1,
        render: |literal, ty| {
            Some(format!(
                "def f() = {{\n  values{} = {literal}\n  values\n}}\n",
                annotation(ty)
            ))
        },
    },
    Position {
        name: "function body",
        declares_kind: true,
        occurrences: 1,
        render: |literal, ty| Some(format!("def f(){} = {literal}\n", result_annotation(ty))),
    },
    Position {
        name: "function body with sig",
        declares_kind: true,
        occurrences: 1,
        render: |literal, ty| ty.map(|ty| format!("sig f: i32 -> {ty}\ndef f(n) = {literal}\n")),
    },
    Position {
        name: "lambda body",
        declares_kind: false,
        occurrences: 1,
        render: |literal, ty| {
            Some(format!(
                "def f(){} = {{\n  g = fn (n: i32) -> {literal}\n  g(1)\n}}\n",
                result_annotation(ty)
            ))
        },
    },
    Position {
        name: "if branches",
        declares_kind: false,
        occurrences: 2,
        render: |literal, ty| {
            Some(format!(
                "def f(c: bool){} = if c then {literal} else {literal}\n",
                result_annotation(ty)
            ))
        },
    },
    Position {
        name: "tuple component",
        declares_kind: false,
        occurrences: 1,
        render: |literal, ty| {
            Some(match ty {
                Some(ty) => format!("pair: ({ty}, i32) = ({literal}, 1)\n"),
                None => format!("pair = ({literal}, 1)\n"),
            })
        },
    },
    Position {
        name: "call argument",
        declares_kind: false,
        occurrences: 1,
        render: |literal, ty| {
            Some(format!(
                "def take(x{}) -> i32 = 1\nvalues = take({literal})\n",
                annotation(ty)
            ))
        },
    },
    Position {
        name: "block result",
        declares_kind: false,
        occurrences: 1,
        render: |literal, ty| {
            Some(format!(
                "def f(){} = {{\n  n = 1\n  {literal}\n}}\n",
                result_annotation(ty)
            ))
        },
    },
    Position {
        name: "cast operand",
        declares_kind: false,
        occurrences: 1,
        render: |literal, ty| Some(format!("values{} = cast({literal}, f64)\n", annotation(ty))),
    },
    Position {
        name: "pipe cast stage",
        declares_kind: false,
        occurrences: 1,
        render: |literal, ty| {
            Some(format!(
                "values{} = {literal} |> cast(f64)\n",
                annotation(ty)
            ))
        },
    },
    Position {
        name: "pipe callee stage",
        declares_kind: false,
        occurrences: 1,
        render: |literal, ty| {
            Some(format!(
                "def take(x{}) -> i32 = 1\nvalues = {literal} |> take\n",
                annotation(ty)
            ))
        },
    },
    Position {
        name: "pipe call stage",
        declares_kind: false,
        occurrences: 1,
        render: |literal, ty| {
            Some(format!(
                "def take(x{}, n: i32) -> i32 = n\nvalues = {literal} |> take(1)\n",
                annotation(ty)
            ))
        },
    },
];

fn deep(source: &str) -> Vec<Expr> {
    let decls = parse_str(source).unwrap_or_else(|error| panic!("parse: {error}\n{source}"));
    let desugared =
        desugar_program(&decls).unwrap_or_else(|error| panic!("desugar: {error}\n{source}"));
    chelis_macros::expand_program(&desugared, &chelis_macros::ExpansionOptions::default())
        .unwrap_or_else(|error| panic!("expand: {error}\n{source}"))
        .into_exprs()
}

fn normalized(exprs: &[Expr]) -> String {
    print_canonical(
        &normalize_deep_for_surface_roundtrip(exprs).expect("valid round-trip metadata"),
    )
}

fn tensor_conversions(normalized: &str) -> usize {
    normalized.matches("(var {} to_tensor)").count()
}

enum Expected {
    /// The literal is converted this many times.
    Conversions(usize),
    /// A bare literal under a tensor alias stays a `List` and the checker
    /// rejects it as one.
    RejectedAsList,
}

/// Checks one program and returns a failure description, if any.
fn check(source: &str, expected: Expected) -> Option<String> {
    let original = deep(source);
    let original_text = normalized(&original);
    let actual = tensor_conversions(&original_text);
    match expected {
        Expected::Conversions(expected) if actual != expected => {
            return Some(format!(
                "kind: expected {expected} tensor conversion(s), found {actual}\n{source}{original_text}"
            ));
        }
        Expected::Conversions(_) => {}
        Expected::RejectedAsList => {
            let errors = infer_program(&original).errors;
            let rejected_as_list = errors
                .iter()
                .any(|error| error.message.contains("List") && error.message.contains("tensor["));
            if actual != 0 || !rejected_as_list {
                return Some(format!(
                    "a bare literal under a tensor alias must stay a List that the checker \
                     rejects: {actual} conversion(s), errors {errors:?}\n{source}"
                ));
            }
        }
    }
    let resugared = match resugar_program(&original) {
        Ok(declarations) => format_program(&declarations),
        Err(error) => return Some(format!("resugar failed: {error}\n{source}")),
    };
    let roundtrip_text = normalized(&deep(&resugared));
    (roundtrip_text != original_text).then(|| {
        format!(
            "round trip changed the Deep\n--- authored\n{source}--- resugared\n{resugared}--- authored Deep\n{original_text}\n--- resugared Deep\n{roundtrip_text}"
        )
    })
}

fn declared_type(declared: Declared, nested: bool) -> Option<&'static str> {
    match (declared, nested) {
        (Declared::Nothing, _) => None,
        (Declared::List, false) => Some("List[f64]"),
        (Declared::List, true) => Some("List[List[f64]]"),
        (Declared::Tensor, false) => Some("tensor[2, f64]"),
        (Declared::Tensor, true) => Some("tensor[2, 2, f64]"),
        (Declared::TensorAlias, _) => Some("V"),
        (Declared::ListAlias, _) => Some("L"),
    }
}

#[test]
fn every_bracket_literal_spelling_keeps_its_kind_and_round_trips() {
    let mut failures = Vec::new();
    let mut cases = 0usize;
    let forms = FLAT_FORMS
        .iter()
        .map(|form| (*form, false))
        .chain(std::iter::once((NESTED_FORM, true)));
    for ((form, literal), nested) in forms {
        let alias = if nested {
            "type V = tensor[2, 2, f64]\ntype L = List[List[f64]]\n"
        } else {
            "type V = tensor[2, f64]\ntype L = List[f64]\n"
        };
        for position in POSITIONS {
            for declared in [
                Declared::Nothing,
                Declared::List,
                Declared::Tensor,
                Declared::TensorAlias,
                Declared::ListAlias,
            ] {
                for spelling in [Spelling::Bare, Spelling::ToTensor, Spelling::PipeToTensor] {
                    let written = match spelling {
                        Spelling::Bare => literal.to_string(),
                        Spelling::ToTensor => format!("to_tensor({literal})"),
                        Spelling::PipeToTensor => format!("{literal} |> to_tensor"),
                    };
                    let Some(body) = (position.render)(&written, declared_type(declared, nested))
                    else {
                        continue;
                    };
                    let source = format!("{PRELUDE}{alias}{body}");
                    let tensor = spelling != Spelling::Bare
                        || (position.declares_kind && declared == Declared::Tensor);
                    // The desugarer reads a declaration's spelling, so a bare
                    // literal under a tensor alias stays a List, which the
                    // checker rejects loudly (chelis#3114). Resolving only
                    // local aliases would make the kind depend on where the
                    // alias is declared; the typed-phase decision is
                    // chelis#3122. A change that accepts these rows must
                    // settle that decision, not pass unnoticed.
                    let expected = if position.declares_kind
                        && declared == Declared::TensorAlias
                        && spelling == Spelling::Bare
                    {
                        Expected::RejectedAsList
                    } else {
                        Expected::Conversions(if tensor { position.occurrences } else { 0 })
                    };
                    cases += 1;
                    if let Some(failure) = check(&source, expected) {
                        failures.push(format!(
                            "[{form} / {} / {declared:?} / {spelling:?}] {failure}",
                            position.name
                        ));
                    }
                }
            }
        }
    }
    assert!(cases > 2000, "the matrix shrank to {cases} cases");
    assert!(
        failures.is_empty(),
        "{} of {cases} cases failed:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// A Deep `Cons` chain where a tensor is declared has no bracket spelling: a
/// bare bracket there is a tensor literal. It must print as explicit
/// constructor calls rather than turn into a tensor on re-desugaring.
#[test]
fn a_list_value_under_a_tensor_declaration_resugars_as_constructor_calls() {
    for (deep_source, expected) in [
        (
            "(defsig {} values (t-tensor {} (d-lit {} 2) (t-prim {} f64)))\n\
             (def {} values (app {} (var {} Cons) (lit {type: (t-prim {} f64)} 1.5)\n\
               (app {} (var {} Cons) (lit {type: (t-prim {} f64)} 2.5) (var {} Nil))))",
            "values: tensor[2, f64] = Cons(1.5f64, Cons(2.5f64, Nil))\n",
        ),
        (
            "(defsig {} f (t-fn {} (t-prim {} i32) (t-tensor {} (d-lit {} 1) (t-prim {} f32))))\n\
             (def {} f (fn {} (params {} (n {type: (t-var {} _)}))\n\
               (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 1.5) (var {} Nil))))",
            "def f(n: i32) -> tensor[1, f32] = Cons(1.5, Nil)\n",
        ),
    ] {
        let original = chelis_deep::parser::parse_str(deep_source).expect("Deep parses");
        let resugared = format_program(&resugar_program(&original).expect("Deep resugars"));
        assert_eq!(resugared, expected);
        assert_eq!(normalized(&deep(&resugared)), normalized(&original));
    }
}

/// The adopted signed minimum is spelled as a negated literal inside a bare
/// tensor literal, where the desugarer folds the sign into the literal.
#[test]
fn an_adopted_signed_minimum_round_trips_inside_a_tensor_literal() {
    for (source, expected) in [
        (
            "values: tensor[2, i8] = [-128, 127]\n",
            "values: tensor[2, i8] = [-128, 127]\n",
        ),
        (
            "values: tensor[2, i16] = [-32768, 1]\n",
            "values: tensor[2, i16] = [-32768, 1]\n",
        ),
        (
            "values = cast(to_tensor([-128, 1]), i8)\n",
            "values = cast(to_tensor([-128, 1]), i8)\n",
        ),
    ] {
        let original = deep(source);
        let resugared = format_program(&resugar_program(&original).expect("Deep resugars"));
        assert_eq!(resugared, expected);
        assert_eq!(normalized(&deep(&resugared)), normalized(&original));
    }
}

/// The literal metadata of every `lit` node, spans removed, in source order.
fn literal_types(exprs: &[Expr]) -> Vec<String> {
    let text = normalized(exprs);
    let mut out = Vec::new();
    let mut rest = text.as_str();
    while let Some(start) = rest.find("(lit {") {
        let tail = &rest[start..];
        // Metadata maps nest parentheses (type syntax); take through the
        // literal's closing parenthesis by balancing.
        let mut depth = 0usize;
        let mut close = tail.len();
        for (index, ch) in tail.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = index + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push(tail[..close].to_string());
        rest = &tail[close..];
    }
    out
}

/// spec/02-surf-syntax.md: `x |> f(y)` is the call `f(x, y)` and `x |> cast(p)`
/// is `cast(x, p)`. Every pipe spelling of an adopting position therefore
/// types its literals exactly as the nested call does (§5.6), for a tensor
/// literal and for a scalar, at a primitive and at a dtype binder.
#[test]
fn pipe_spellings_adopt_exactly_like_the_nested_call() {
    let groups: &[&[&str]] = &[
        &[
            "values = cast(to_tensor([1.1, -2.2]), f64)\n",
            "values = to_tensor([1.1, -2.2]) |> cast(f64)\n",
            "values = [1.1, -2.2] |> to_tensor |> cast(f64)\n",
            "values = cast([1.1, -2.2] |> to_tensor, f64)\n",
        ],
        &[
            "values = cast(to_tensor([1, -2]), i8)\n",
            "values = [1, -2] |> to_tensor |> cast(i8)\n",
        ],
        &[
            "def take(x: tensor[2, f64]) -> tensor[2, f64] = x\nvalues = take(to_tensor([1.1, -2.2]))\n",
            "def take(x: tensor[2, f64]) -> tensor[2, f64] = x\nvalues = to_tensor([1.1, -2.2]) |> take\n",
            "def take(x: tensor[2, f64]) -> tensor[2, f64] = x\nvalues = [1.1, -2.2] |> to_tensor |> take\n",
            "def take(x: tensor[2, f64]) -> tensor[2, f64] = x\nvalues = take([1.1, -2.2] |> to_tensor)\n",
        ],
        &[
            "def take(x: tensor[2, f64], n: i32) -> tensor[2, f64] = x\nvalues = take(to_tensor([1.1, -2.2]), 1)\n",
            "def take(x: tensor[2, f64], n: i32) -> tensor[2, f64] = x\nvalues = to_tensor([1.1, -2.2]) |> take(1)\n",
            "def take(x: tensor[2, f64], n: i32) -> tensor[2, f64] = x\nvalues = [1.1, -2.2] |> to_tensor |> take(1)\n",
        ],
        &[
            "values: tensor[2, f64] = to_tensor([1.1, -2.2])\n",
            "values: tensor[2, f64] = [1.1, -2.2] |> to_tensor\n",
        ],
        &[
            "def make() -> tensor[2, f64] = to_tensor([1.1, -2.2])\n",
            "def make() -> tensor[2, f64] = [1.1, -2.2] |> to_tensor\n",
        ],
        &[
            "def g[p: Float](w: p) -> tensor[2, p] = cast(to_tensor([1.1, -2.2]), p)\n",
            "def g[p: Float](w: p) -> tensor[2, p] = to_tensor([1.1, -2.2]) |> cast(p)\n",
            "def g[p: Float](w: p) -> tensor[2, p] = [1.1, -2.2] |> to_tensor |> cast(p)\n",
        ],
        &["values = cast(1.1, f64)\n", "values = 1.1 |> cast(f64)\n"],
        &["values = cast(-1.1, f64)\n", "values = -1.1 |> cast(f64)\n"],
        &[
            "values = cast(3000000000, i64)\n",
            "values = 3000000000 |> cast(i64)\n",
        ],
        &[
            "def g[p: Float](w: p) -> p = cast(1.1, p)\n",
            "def g[p: Float](w: p) -> p = 1.1 |> cast(p)\n",
        ],
    ];
    let mut failures = Vec::new();
    for group in groups {
        let expected = literal_types(&deep(group[0]));
        for source in &group[1..] {
            let actual = literal_types(&deep(source));
            if actual != expected {
                failures.push(format!(
                    "{source}literals {actual:?}\nbut the call spelling {}gives {expected:?}",
                    group[0]
                ));
            }
            if let Some(failure) = check(
                source,
                Expected::Conversions(source.matches("to_tensor").count()),
            ) {
                failures.push(failure);
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// Position 2 reads a callee's declared signature only when the call names the
/// top-level function. A local binding of the name, under any binder kind, is
/// another callable, so its argument adopts nothing (spec/04 §5.6).
#[test]
fn a_lexically_rebound_callee_adopts_nothing() {
    const DECLARED: &str = "def f(x: tensor[2, f64]) -> tensor[2, f64] = x\n";
    let shadowed = [
        "def g() = {\n  f = fn (y) -> cast(y, f64)\n  f(to_tensor([1.1, 2.2]))\n}\n",
        "def g(f: tensor[2, f32] -> tensor[2, f32]) -> tensor[2, f32] = f(to_tensor([1.1, 2.2]))\n",
        "def g() = {\n  h = fn (f: tensor[2, f32] -> tensor[2, f32]) -> f(to_tensor([1.1, 2.2]))\n  h\n}\n",
        "def g(o: Option[tensor[2, f32] -> tensor[2, f32]]) = match o with {\n  | Some(f) => f(to_tensor([1.1, 2.2]))\n  | None => to_tensor([0.0, 0.0])\n}\n",
        "def g() = {\n  (f, k) = (fn (y: tensor[2, f32]) -> y, 1)\n  f(to_tensor([1.1, 2.2]))\n}\n",
        "def g(f: tensor[2, f32] -> tensor[2, f32]) -> tensor[2, f32] = to_tensor([1.1, 2.2]) |> f\n",
        "def g(f: tensor[2, f32] -> tensor[2, f32]) -> tensor[2, f32] = [1.1, 2.2] |> to_tensor |> f\n",
    ];
    for body in shadowed {
        let source = format!("{DECLARED}{body}");
        let types = literal_types(&deep(&source));
        assert!(
            types.iter().any(|ty| ty.contains("1.1"))
                && !types.iter().any(|ty| ty.contains("f64)} 1.1")),
            "a rebound `f` must not adopt the top-level signature:\n{source}{types:?}"
        );
        if let Some(failure) = check(
            &source,
            Expected::Conversions(source.matches("to_tensor").count()),
        ) {
            panic!("{failure}");
        }
        let resugared = format_program(&resugar_program(&deep(&source)).expect("Deep resugars"));
        assert!(
            !resugared.contains("1.1f32"),
            "the resugarer must not force suffixes under a rebound callee:\n{resugared}"
        );
    }
    // Control: the top-level function itself adopts.
    let types = literal_types(&deep(&format!("{DECLARED}r = f(to_tensor([1.1, 2.2]))\n")));
    assert!(types.iter().any(|ty| ty.contains("f64)} 1.1")), "{types:?}");
}

/// §5.6 position 4 admits a dtype-bounded binder target for a tensor literal
/// too. Its elements are typed at the binder in Deep, which the checker
/// refuses until it adopts them at each instantiation, rather than rounding
/// them through the f32 default.
#[test]
fn a_tensor_literal_cast_to_a_dtype_binder_is_typed_at_the_binder_and_refused() {
    for body in [
        "cast(to_tensor([1.1, 2.2]), p)",
        "to_tensor([1.1, 2.2]) |> cast(p)",
        "[1.1, 2.2] |> to_tensor |> cast(p)",
    ] {
        let source = format!("def g[p: Float](w: p) -> tensor[2, p] = {body}\nr = g(0.0f64)\n");
        let exprs = deep(&source);
        let types = literal_types(&exprs);
        assert!(
            types
                .iter()
                .filter(|ty| ty.contains("(t-var {} p)"))
                .count()
                == 2,
            "{source}{types:?}"
        );
        if let Some(failure) = check(&source, Expected::Conversions(1)) {
            panic!("{failure}");
        }
        let errors = infer_program(&exprs).errors;
        assert!(
            errors
                .iter()
                .any(|error| error.message.contains("to_tensor")
                    && error.message.contains("`p`")
                    && error.message.contains("cast(1.1, p)")),
            "{source}{errors:?}"
        );
    }
    // Negative parity: suffixed elements do not adopt, and a scalar binder
    // cast adopts as before.
    for source in [
        "def g[p: Float](w: p) -> tensor[2, p] = cast(to_tensor([1.1f64, 2.2f64]), p)\nr = g(0.0f64)\n",
        "def g[p: Float](w: p) -> p = cast(1.1, p)\nr = g(0.0f64)\n",
        "def g[p: Float](w: p) -> tensor[2, p] = to_tensor([cast(1.1, p), cast(2.2, p)])\nr = g(0.0f64)\n",
    ] {
        let errors = infer_program(&deep(source)).errors;
        assert!(errors.is_empty(), "{source}{errors:?}");
    }
}
