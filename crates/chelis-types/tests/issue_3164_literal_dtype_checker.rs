//! chelis#3164 (A1): a numeric literal's dtype is written at its site, through
//! the checker, Deep ingress, and the Surf printer.
//!
//! `spec/04-type-system.md` §5.3, §5.6 and §8.6, `spec/05-risc-primitives.md`
//! [05-OP-57], and `spec/03-deep-syntax.md` §6.3.1 and §6.4:
//!
//! - a declaration binds its literal initializer, and every checking rule
//!   (ascriptions, aliases, declared Lists, range, finiteness) still applies;
//! - `to_tensor(xs, p)` requires the leaf dtype of `xs` to be `p` and never
//!   converts;
//! - Deep has no dtype-stating constructs: an untyped literal element of a
//!   `to_tensor` argument is rejected, and a `to_tensor` binder is rejected as
//!   `ReservedName`;
//! - a literal written as a macro argument or in a macro body is governed by
//!   the constructs where it is written;
//! - `chelis surf` prints a literal bare only where re-reading it binds the
//!   same dtype, so `desugar(resugar(deep))` keeps every literal's dtype and
//!   every `cast` node. The generated matrix checks that law together with
//!   the dtype §5.6 assigns to each literal, so a desugarer and a printer that
//!   agree on a wrong rule still fail.

use chelis_deep::Expr;
use chelis_deep::parser::parse_and_stamp_file;
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_program;
use chelis_surf::parser::parse_str;
use chelis_surf::resugar::{normalize_deep_for_surface_roundtrip, resugar_program};
use chelis_types::check_typed_program;

/// Parse, desugar and expand macros, as `chelis deep` does.
fn front_end(source: &str) -> Result<Vec<Expr>, String> {
    let decls = parse_str(source).unwrap_or_else(|error| panic!("parse: {error}\n{source}"));
    let desugared = desugar_program(&decls).map_err(|error| error.to_string())?;
    chelis_macros::expand_program(&desugared, &chelis_macros::ExpansionOptions::default())
        .map(|expanded| expanded.into_exprs())
        .map_err(|error| error.to_string())
}

fn deep(source: &str) -> Vec<Expr> {
    front_end(source).unwrap_or_else(|error| panic!("front end: {error}\n{source}"))
}

/// Every error the program produces at Surf ingress or in the checker, with
/// each checker error's kind, or `None` when it checks.
fn rejection(source: &str) -> Option<String> {
    let exprs = match front_end(source) {
        Ok(exprs) => exprs,
        Err(error) => return Some(error),
    };
    check_typed_program(&exprs).err().map(|result| {
        result
            .errors
            .iter()
            .map(|error| format!("{:?}: {}", error.kind, error.message))
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn accepted(source: &str) {
    if let Some(errors) = rejection(source) {
        panic!("expected `{source}` to check, got:\n{errors}");
    }
}

fn rejected(source: &str, needles: &[&str]) {
    let errors = rejection(source).unwrap_or_else(|| panic!("expected a rejection of:\n{source}"));
    for needle in needles {
        assert!(
            errors.contains(needle),
            "expected `{needle}` in the rejection of:\n{source}--- errors\n{errors}"
        );
    }
}

/// The checked type of top-level `name`, printed as Deep.
fn root_type(source: &str, name: &str) -> String {
    let checked = check_typed_program(&deep(source))
        .unwrap_or_else(|result| panic!("{source}: {:?}", result.errors));
    let ty = checked
        .type_env()
        .get(name)
        .unwrap_or_else(|| panic!("no type for `{name}` in {source}"));
    print_canonical(std::slice::from_ref(ty))
}

fn deep_file(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source).unwrap_or_else(|error| panic!("Deep: {error:?}\n{source}"))
}

fn deep_rejection(source: &str) -> Option<String> {
    check_typed_program(&deep_file(source)).err().map(|result| {
        result
            .errors
            .iter()
            .map(|error| format!("{:?}: {}", error.kind, error.message))
            .collect::<Vec<_>>()
            .join("\n")
    })
}

fn normalized(exprs: &[Expr]) -> String {
    print_canonical(
        &normalize_deep_for_surface_roundtrip(exprs).expect("valid round-trip metadata"),
    )
}

/// `desugar(resugar(deep))` against `normalize(deep)`; a failure description.
fn round_trip(original: &[Expr], label: &str) -> Option<String> {
    let original_text = normalized(original);
    let resugared = match resugar_program(original) {
        Ok(declarations) => format_program(&declarations),
        Err(error) => return Some(format!("resugar failed: {error}\n{label}")),
    };
    let reread = match front_end(&resugared) {
        Ok(exprs) => exprs,
        Err(error) => {
            return Some(format!(
                "the printed Surf does not desugar: {error}\n{label}--- printed\n{resugared}"
            ));
        }
    };
    let reread_text = normalized(&reread);
    (reread_text != original_text).then(|| {
        format!(
            "round trip changed the Deep\n{label}--- printed\n{resugared}--- original Deep\n{original_text}\n--- re-read Deep\n{reread_text}"
        )
    })
}

/// The `type` of every `lit` node, in order: a primitive name, or a binder
/// name for a `t-var`.
fn literal_dtypes(text: &str) -> Vec<String> {
    let mut dtypes = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("(lit {") {
        let meta = &rest[start + "(lit ".len()..];
        let mut depth = 0usize;
        let mut end = 0usize;
        for (index, ch) in meta.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = index;
                        break;
                    }
                }
                _ => {}
            }
        }
        let meta = &meta[..end];
        let dtype = ["type: (t-prim {} ", "type: (t-var {} "]
            .iter()
            .find_map(|prefix| {
                meta.find(prefix).map(|at| {
                    let name = &meta[at + prefix.len()..];
                    name[..name.find(')').expect("closed type")].to_string()
                })
            })
            .unwrap_or_else(|| "untyped".to_string());
        dtypes.push(dtype);
        rest = &rest[start + "(lit {".len()..];
    }
    dtypes
}

// === Declarations ===

/// §5.6 Declaration with §5.3's checking rules intact.
#[test]
fn a_declaration_binds_its_literal_and_every_check_still_applies() {
    for source in [
        "x: f64 = 1.1\n",
        "sig x: f64\nx = 1.1\n",
        "def f() -> i8 = -128\n",
        "x: i64 = 3000000000\n",
        "x: i64 = -9223372036854775808\n",
        "def f() -> f64 = {\n  y: f64 = 1.1\n  y\n}\n",
        "sig g: f64 -> f64\ndef g(a) = 1.1\n",
        "x: f64 = 5\n",
        "x: bf16 = 1.5\n",
        "x: f32 = 1.1\n",
    ] {
        accepted(source);
    }
    assert!(root_type("x: f64 = 1.1\n", "x").contains("f64"));
    assert!(root_type("x = cast(1, bool)\n", "x").contains("bool"));

    for (source, needles) in [
        ("x: f64 = neg(1.1)\n", &["whole initializer"][..]),
        (
            "x: f64 = if true then 1.1 else 2.2\n",
            &["whole initializer"][..],
        ),
        ("x: i8 = 300\n", &["range"][..]),
        ("x: i8 = -129\n", &["range"][..]),
        ("x: f16 = 70000.0\n", &["f16"][..]),
        ("x: f64 = 1.1f32\n", &[][..]),
        ("type P = f64\nx: P = 1.1\n", &[][..]),
        ("x = (1.1 : f64)\n", &["1.1f64"][..]),
        ("out = {\n  y = (1.1 : f64)\n  y\n}\n", &["1.1f64"][..]),
        ("shape: List[i64] = [2, 3]\n", &[][..]),
        (
            "def main() -> i64 = add(2147483648, 0i64)\n",
            &["cast(2147483648, i64)"][..],
        ),
    ] {
        rejected(source, needles);
    }
}

// === Dtype argument ===

/// [05-OP-57]: a written `T` states the leaf dtype and never converts.
#[test]
fn a_dtype_argument_is_checked_against_the_leaf_dtype_and_never_converts() {
    for (source, element) in [
        ("x = to_tensor([1.5, 2.5], f64)\n", "f64"),
        ("xs = [1.5f64]\nx = to_tensor(xs, f64)\n", "f64"),
        ("x = to_tensor([], f64)\n", "f64"),
        ("x = to_tensor([true, false], bool)\n", "bool"),
        ("x = to_tensor([[1, 2], [3, 4]], i64)\n", "i64"),
        ("x = {\n  f64 = 2\n  to_tensor([1.5], f64)\n}\n", "f64"),
        ("x = to_tensor([1.5f64, 2.5], f64)\n", "f64"),
        (
            "def k[p: Float](v: p) -> tensor[1, p] = to_tensor([cast(1.5, p)], p)\nx = k(1.0f64)\n",
            "f64",
        ),
        (
            "def k[p: Float](v: p) -> tensor[1, p] = to_tensor([v], p)\nx = k(1.0f64)\n",
            "f64",
        ),
    ] {
        let ty = root_type(source, "x");
        assert!(
            ty.contains("t-tensor") && ty.contains(&format!("(t-prim {{}} {element})")),
            "{source}: {ty}"
        );
    }
    for (source, needles) in [
        (
            "xs = [1.5, 2.5]\nx = to_tensor(xs, f64)\n",
            &["never converts"][..],
        ),
        ("x = to_tensor([1.0f64], f32)\n", &["f64", "f32"][..]),
        (
            "x = to_tensor([1, 0], bool)\n",
            &["cannot bind at bool"][..],
        ),
        (
            "macro half() = 1.5\nx = to_tensor([half()], f64)\n",
            &["never converts"][..],
        ),
        ("x = to_tensor([1.5], f64, f64)\n", &[][..]),
        (
            "def k[p: Float](v: f32) -> tensor[1, p] = to_tensor([v], p)\n",
            &["never converts"][..],
        ),
    ] {
        rejected(source, needles);
    }
}

// === Macros ===

/// §5.6: the rule is judged on the source text in which the literal is
/// written. A literal passed as a macro argument, or written in a macro body,
/// is not bound by a cast on the other side of the expansion.
#[test]
fn a_macro_argument_or_body_literal_is_governed_where_it_is_written() {
    let direct = normalized(&deep("r = cast(1.1, f64)\n"));
    assert!(!direct.contains("(cast "), "{direct}");
    assert_eq!(literal_dtypes(&direct), ["f64"], "{direct}");

    for source in [
        "macro tf64(x) = cast(x, f64)\nr = tf64(1.1)\n",
        "macro half() = 1.1\nr = cast(half(), f64)\n",
        "macro mh() = neg(1.1)\nr = cast(mh(), f64)\n",
    ] {
        let text = normalized(&deep(source));
        assert!(text.contains("(cast "), "{source}{text}");
        assert!(
            literal_dtypes(&text).iter().all(|dtype| dtype == "f32"),
            "{source}{text}"
        );
    }
}

// === Deep ingress ===

/// spec/03 §6.4: Deep has no dtype-stating constructs. A Deep literal's dtype
/// is its `type`; an untyped literal element of a `to_tensor` argument is
/// rejected, with or without a dtype child. The three-child form is checked.
#[test]
fn deep_literals_under_to_tensor_are_explicit_and_the_dtype_child_is_checked() {
    const TYPED_F64: &str = "(app {} (var {} Cons) (lit {type: (t-prim {} f64)} 1.5) (var {} Nil))";
    const UNTYPED: &str = "(app {} (var {} Cons) (lit {} 1.5) (var {} Nil))";
    for source in [
        format!("(def {{}} a (app {{}} (var {{}} to_tensor) {UNTYPED}))"),
        format!("(def {{}} a (app {{}} (var {{}} to_tensor) {UNTYPED} (t-prim {{}} f32)))"),
    ] {
        let errors = deep_rejection(&source).unwrap_or_else(|| panic!("accepted: {source}"));
        assert!(errors.contains("no `type` metadata"), "{source}: {errors}");
    }
    let mismatch =
        format!("(def {{}} a (app {{}} (var {{}} to_tensor) {TYPED_F64} (t-prim {{}} f32)))");
    let errors = deep_rejection(&mismatch).unwrap_or_else(|| panic!("accepted: {mismatch}"));
    assert!(errors.contains("never converts"), "{errors}");

    // Mirrors: the typed one- and three-child forms, and an untyped literal
    // outside `to_tensor`, which keeps its default.
    for source in [
        format!("(def {{}} a (app {{}} (var {{}} to_tensor) {TYPED_F64}))"),
        format!("(def {{}} a (app {{}} (var {{}} to_tensor) {TYPED_F64} (t-prim {{}} f64)))"),
        "(def {} a (lit {} 1.5))".to_string(),
    ] {
        assert_eq!(deep_rejection(&source), None, "{source}");
    }
}

/// §8.6: a Deep binder of `to_tensor`, in any scope, is `ReservedName`.
#[test]
fn every_deep_binder_of_to_tensor_is_reserved() {
    for source in [
        "(defsig {} f (t-fn {} (t-prim {} i32) (t-prim {} i32)))\n\
         (def {} f (fn {} (params {} (to_tensor {type: (t-var {} _)})) (var {} to_tensor)))",
        "(def {} g (fn {} (params {} (to_tensor {type: (t-prim {} i32)})) (var {} to_tensor)))",
        "(def {} r (let {} (bind {} to_tensor (lit {type: (t-prim {} i32)} 1)) (var {} to_tensor)))",
        "(def {} r (let {} (bind {destructure: true} to_tensor (lit {type: (t-prim {} i32)} 1)) (var {} to_tensor)))",
        "(def {} r (match {} (app {} (var {} Some) (lit {type: (t-prim {} i32)} 1))\n\
           (arm {} (pat-ctor {} Some (pat-var {} to_tensor)) () (var {} to_tensor))\n\
           (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} i32)} 0))))",
        "(def {} to_tensor (lit {type: (t-prim {} i32)} 1))",
        "(defsig {} to_tensor (t-fn {} (t-prim {} i32) (t-prim {} i32)))\n\
         (def {} to_tensor (fn {} (params {} (x {type: (t-var {} _)})) (var {} x)))",
        "(import {surf_path: \"Std.Sort\"} std.sort (to_tensor))\n\
         (def {} r (lit {type: (t-prim {} i32)} 1))",
    ] {
        let errors = deep_rejection(source).unwrap_or_else(|| panic!("accepted: {source}"));
        assert!(errors.contains("ReservedName"), "{source}: {errors}");
        assert!(
            errors.contains("`to_tensor` is reserved"),
            "{source}: {errors}"
        );
    }
    let call = "(def {} r (app {} (var {} to_tensor) (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 1.5) (var {} Nil))))";
    assert_eq!(deep_rejection(call), None, "a call is not a binder");
}

// === Printer ===

#[derive(Clone, Copy)]
enum Site {
    Plain,
    Declared,
    SigDeclared,
    DefResult,
    SigDefResult,
    BlockDeclared,
    NegCall,
    Cast,
    NamedCast,
    BinderCast,
    DtypeArgument,
    NoDtypeArgument,
    TensorLiteral,
    ListElement,
    CallArgument,
    PipeCast,
    PipeDtypeArgument,
    PipeNoDtypeArgument,
    MacroCast,
}

impl Site {
    const ALL: [Site; 19] = [
        Site::Plain,
        Site::Declared,
        Site::SigDeclared,
        Site::DefResult,
        Site::SigDefResult,
        Site::BlockDeclared,
        Site::NegCall,
        Site::Cast,
        Site::NamedCast,
        Site::BinderCast,
        Site::DtypeArgument,
        Site::NoDtypeArgument,
        Site::TensorLiteral,
        Site::ListElement,
        Site::CallArgument,
        Site::PipeCast,
        Site::PipeDtypeArgument,
        Site::PipeNoDtypeArgument,
        Site::MacroCast,
    ];

    fn uses_dtype(self) -> bool {
        !matches!(
            self,
            Site::Plain
                | Site::BinderCast
                | Site::NoDtypeArgument
                | Site::ListElement
                | Site::PipeNoDtypeArgument
        )
    }

    /// The program, or `None` where the spelling is not well formed: a
    /// negated literal cannot head an ungrouped pipe (spec/02 [02-PIPE-2]).
    fn render(self, literal: &str, dtype: &str) -> Option<String> {
        Some(match self {
            Site::Plain => format!("x = {literal}\n"),
            Site::Declared => format!("x: {dtype} = {literal}\n"),
            Site::SigDeclared => format!("sig x: {dtype}\nx = {literal}\n"),
            Site::DefResult => format!("def f() -> {dtype} = {literal}\n"),
            Site::SigDefResult => format!("sig g: {dtype} -> {dtype}\ndef g(a) = {literal}\n"),
            Site::BlockDeclared => {
                format!("def f() -> {dtype} = {{\n  y: {dtype} = {literal}\n  y\n}}\n")
            }
            Site::NegCall => format!("x: {dtype} = neg({literal})\n"),
            Site::Cast => format!("x = cast({literal}, {dtype})\n"),
            Site::NamedCast => format!("x = cast_saturate({literal}, {dtype})\n"),
            Site::BinderCast => {
                format!("def h[p: Numeric](v: p) -> p = add(v, cast({literal}, p))\n")
            }
            Site::DtypeArgument => format!("x = to_tensor([{literal}, {literal}], {dtype})\n"),
            Site::NoDtypeArgument => format!("x = to_tensor([{literal}, {literal}])\n"),
            Site::TensorLiteral => format!("xs: tensor[2, {dtype}] = [{literal}, {literal}]\n"),
            Site::ListElement => format!("xs = [{literal}, {literal}]\n"),
            Site::CallArgument => format!("def id(v: {dtype}) -> {dtype} = v\nx = id({literal})\n"),
            Site::PipeCast if literal.starts_with('-') => return None,
            Site::PipeCast => format!("x = {literal} |> cast({dtype})\n"),
            Site::PipeDtypeArgument => {
                format!("x = [{literal}, {literal}] |> to_tensor({dtype})\n")
            }
            Site::PipeNoDtypeArgument => format!("x = [{literal}, {literal}] |> to_tensor\n"),
            // The macro body's literal is not the cast's operand in the
            // source, so the expanded cast converts a literal whose default
            // has no `explicit` marker: the printer must keep its suffix.
            Site::MacroCast => format!("macro m() = {literal}\nx = cast(m(), {dtype})\n"),
        })
    }
}

enum Expected {
    /// Desugars: every literal has this dtype, and there are this many
    /// literals and `cast` nodes.
    Literals {
        dtype: String,
        count: usize,
        casts: usize,
    },
    /// Rejected at Surf ingress with this message fragment.
    Rejected(&'static str),
}

/// The §5.6 outcome for `literal` at `site` with stated dtype `dtype`.
fn expected(site: Site, literal: &str, dtype: &str) -> Expected {
    const SUFFIXES: [&str; 8] = ["bf16", "f16", "f32", "f64", "i8", "i16", "i32", "i64"];
    let suffix = SUFFIXES
        .iter()
        .find(|suffix| literal.ends_with(*suffix))
        .copied();
    let float = literal.contains('.');
    let default = if float { "f32" } else { "i32" };
    let own = suffix.unwrap_or(default).to_string();
    let float_dtype = dtype.starts_with('f') || dtype == "bf16";
    let numeric_dtype = dtype != "bool";
    let admits = numeric_dtype && (!float || float_dtype);
    let stated = |count: usize| match (suffix, admits) {
        (Some(_), _) => Expected::Literals {
            dtype: own.clone(),
            count,
            casts: 0,
        },
        (None, true) => Expected::Literals {
            dtype: dtype.to_string(),
            count,
            casts: 0,
        },
        (None, false) => Expected::Rejected("cannot bind"),
    };
    let own_literals = |count: usize, casts: usize| Expected::Literals {
        dtype: own.clone(),
        count,
        casts,
    };
    match site {
        Site::Plain | Site::NegCall | Site::CallArgument => own_literals(1, 0),
        Site::ListElement => own_literals(2, 0),
        Site::NamedCast | Site::MacroCast => own_literals(1, 1),
        Site::Declared
        | Site::SigDeclared
        | Site::DefResult
        | Site::SigDefResult
        | Site::BlockDeclared => stated(1),
        Site::DtypeArgument | Site::TensorLiteral | Site::PipeDtypeArgument => stated(2),
        Site::Cast | Site::PipeCast => match (suffix, admits) {
            (None, true) => Expected::Literals {
                dtype: dtype.to_string(),
                count: 1,
                casts: 0,
            },
            _ => own_literals(1, 1),
        },
        Site::BinderCast => Expected::Literals {
            dtype: if suffix.is_some() {
                own.clone()
            } else {
                "p".to_string()
            },
            count: 1,
            casts: 1,
        },
        Site::NoDtypeArgument | Site::PipeNoDtypeArgument => match suffix {
            Some(_) => own_literals(2, 0),
            None => Expected::Rejected("states no dtype"),
        },
    }
}

/// Sites × literal spellings × dtypes: each literal gets the §5.6 dtype, and
/// `chelis surf` output re-reads to the same Deep. A pipe site expects exactly
/// what its normalized call does (spec/02 [02-PIPE-1]). A printer that strips a
/// cast operand's suffix, or trusts the unsuffixed marker, fails here.
#[test]
fn every_site_gives_its_literal_the_stated_dtype_and_survives_resugaring() {
    const LITERALS: [&str; 12] = [
        "1.5", "-1.5", "0.1", "7", "-7", "1.5f64", "1.5f32", "-1.5f64", "7i64", "7i8", "7f64",
        "2.5bf16",
    ];
    const DTYPES: [&str; 8] = ["f32", "f64", "bf16", "f16", "i8", "i32", "i64", "bool"];
    let mut failures = Vec::new();
    let mut cases = 0usize;
    for site in Site::ALL {
        for literal in LITERALS {
            for dtype in DTYPES {
                if !site.uses_dtype() && dtype != DTYPES[0] {
                    continue;
                }
                let Some(source) = site.render(literal, dtype) else {
                    continue;
                };
                cases += 1;
                let outcome = front_end(&source);
                match (expected(site, literal, dtype), outcome) {
                    (Expected::Rejected(needle), Err(error)) => {
                        if !error.contains(needle) {
                            failures.push(format!("expected `{needle}`, got `{error}`\n{source}"));
                        }
                    }
                    (Expected::Rejected(needle), Ok(exprs)) => failures.push(format!(
                        "expected a rejection naming `{needle}`, got\n{}\n{source}",
                        normalized(&exprs)
                    )),
                    (Expected::Literals { .. }, Err(error)) => {
                        failures.push(format!("unexpected rejection `{error}`\n{source}"));
                    }
                    (
                        Expected::Literals {
                            dtype: want,
                            count,
                            casts,
                        },
                        Ok(exprs),
                    ) => {
                        let text = normalized(&exprs);
                        let dtypes = literal_dtypes(&text);
                        let found_casts = text.matches("(cast ").count();
                        if dtypes.len() != count
                            || dtypes.iter().any(|found| *found != want)
                            || found_casts != casts
                        {
                            failures.push(format!(
                                "expected {count} literal(s) at {want} and {casts} cast(s), \
                                 found {dtypes:?} and {found_casts}\n{source}{text}"
                            ));
                        } else if let Some(failure) = round_trip(&exprs, &source) {
                            failures.push(failure);
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 1300, "the matrix shrank to {cases} cases");
    assert!(
        failures.is_empty(),
        "{} of {cases} cases failed:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// chelis#3165, spec/03 §6.3.1's F1 rows, and hand-written Deep the printer
/// must not trust: each keeps its literal dtypes and `cast` nodes.
#[test]
fn literals_the_printer_must_suffix_survive_resugaring() {
    let mut failures = Vec::new();
    for source in [
        // chelis#3165
        "macro mh() = neg(1.1)\nr = cast(mh(), f64)\n",
        "macro half() = 1.1\ndef k[p: Float](x: p) -> p = cast(half(), p)\nr = k(1.0f64)\n",
        "macro half() = 1.1\nr = cast(half(), f64)\n",
        "r = cast((1.1 : f32), f64)\n",
        // A default-suffixed cast operand converts, and must keep converting.
        "r = cast(1.1f32, f64)\n",
        "r = cast(1.1f32, f32)\n",
        "r = cast(-128, i8)\n",
        "r: i8 = -128\n",
        "r = to_tensor([1.1, 2.2], f64)\n",
        "r = to_tensor([1.1f32, 2.2f32], f32)\n",
    ] {
        if let Some(failure) = round_trip(&deep(source), source) {
            failures.push(failure);
        }
    }
    for source in [
        "(def {} b (lit {surf_literal_style: \"unsuffixed\", type: (t-prim {} f64)} 1.1))",
        "(def {} a (cast {} (lit {type: (t-prim {} f32)} 1.1) (t-prim {} f64)))",
        "(def {} c (cast {} (app {} (var {} neg) (lit {surf_literal_style: \"unsuffixed\", type: (t-prim {} f32)} 1.1)) (t-prim {} f64)))",
        "(def {} d (cast {} (lit {type: (t-prim {} f32)} 1.1) (t-prim {} f32)))",
        "(def {} e (lit {type: (t-prim {} i8)} -128))",
        "(def {} t (app {} (var {} to_tensor) (app {} (var {} Cons) (lit {type: (t-prim {} f64)} 1.5) (var {} Nil)) (t-prim {} f64)))",
    ] {
        if let Some(failure) = round_trip(&deep_file(source), source) {
            failures.push(failure);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
