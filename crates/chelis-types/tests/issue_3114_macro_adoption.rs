//! chelis#3114: a macro's literals adopt where expansion places them.
//!
//! `spec/02-surf-syntax.md` §P5b resolves a macro body's free references in
//! the caller's scope, and §P10b decides adoption on the expanded program. A
//! macro body's free callee, its `to_tensor`, its casts and its pipe stages
//! therefore resolve at the use site, and a macro argument adopts in the
//! position the body gives it. Each case below writes the same program twice,
//! once through a macro and once as the text the expansion produces, and
//! requires identical literal types and the `spec/02-surf-syntax.md` §0.1
//! round-trip law, both for the expanded program (`chelis deep`, then
//! `chelis surf`) and for the unexpanded one, whose macros survive resugaring.

use chelis_deep::{Atom, DeepTag, Expr};
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_program;
use chelis_surf::parser::parse_str;
use chelis_surf::resugar::{normalize_deep_for_surface_roundtrip, resugar_program};

const PRELUDE: &str = "def f(x: tensor[2, f64]) -> tensor[2, f64] = x\n\
                       def f2(n: i32, x: tensor[2, f64]) -> tensor[2, f64] = x\n\
                       def f2b(x: tensor[2, f64], n: i32) -> tensor[2, f64] = x\n\
                       def tt[p: Float](l: List[p]) -> tensor[2, p] = to_tensor(l)\n";

/// Zero-parameter macro bodies, with the name each one reads that a use site
/// can rebind. The first ten are round 2's R2-1 reproduction bodies.
const BODIES: &[(&str, &str)] = &[
    ("f", "f(to_tensor([1.1, 2.2]))"),
    ("f", "to_tensor([1.1, 2.2]) |> f"),
    ("f", "[1.1, 2.2] |> to_tensor |> f"),
    ("f2", "f2(1, to_tensor([1.1, 2.2]))"),
    ("f2", "1 |> f2(to_tensor([1.1, 2.2]))"),
    ("to_tensor", "cast(to_tensor([1.1, 2.2]), f64)"),
    ("to_tensor", "to_tensor([1.1, 2.2]) |> cast(f64)"),
    ("to_tensor", "[1.1, 2.2] |> to_tensor |> cast(f64)"),
    ("to_tensor", "cast([1.1, 2.2] |> to_tensor, f64)"),
    ("to_tensor", "f(to_tensor([1.1, 2.2]))"),
    ("f", "cast(1.1, f64) |> fn (y) -> f(to_tensor([y, 2.2]))"),
    ("to_tensor", "cast(-1.1, f64)"),
    ("to_tensor", "-1.1 |> cast(f64)"),
];

/// A local value for each rebindable name.
fn rebinding(name: &str) -> &'static str {
    match name {
        "f" => "fn (x) -> x",
        "f2" => "fn (n, x) -> x",
        "f2b" => "fn (x, n) -> x",
        "to_tensor" => "tt",
        other => unreachable!("no rebinding for {other}"),
    }
}

/// The ways a use site can place an expression `e`, rebinding `name` or not.
fn sites(name: &str) -> Vec<(&'static str, Box<dyn Fn(&str) -> String>)> {
    let rebound = rebinding(name);
    let block_name = name.to_string();
    let match_name = name.to_string();
    let lambda_name = name.to_string();
    vec![
        ("top level", Box::new(|e: &str| e.to_string())),
        (
            "block rebinding",
            Box::new(move |e: &str| format!("{{\n  {block_name} = {rebound}\n  {e}\n}}")),
        ),
        (
            "match rebinding",
            Box::new(move |e: &str| {
                format!(
                    "match Some({rebound}) with {{\n  | Some({match_name}) => {e}\n  | None => {e}\n}}"
                )
            }),
        ),
        (
            "parameter rebinding",
            Box::new(move |e: &str| format!("(fn ({lambda_name}) -> {e})({rebound})")),
        ),
    ]
}

fn desugared(source: &str) -> Vec<Expr> {
    let decls = parse_str(source).unwrap_or_else(|error| panic!("parse: {error}\n{source}"));
    desugar_program(&decls).unwrap_or_else(|error| panic!("desugar: {error}\n{source}"))
}

fn expanded(exprs: &[Expr], source: &str) -> Vec<Expr> {
    chelis_macros::expand_program(exprs, &chelis_macros::ExpansionOptions::default())
        .unwrap_or_else(|error| panic!("expand: {error}\n{source}"))
        .into_exprs()
}

fn deep(source: &str) -> Vec<Expr> {
    expanded(&desugared(source), source)
}

fn normalized(exprs: &[Expr]) -> Vec<Expr> {
    normalize_deep_for_surface_roundtrip(exprs).expect("valid round-trip metadata")
}

/// The metadata of every `lit` node under the top-level `def name`, spans
/// and the macro source record removed, in source order.
fn literal_types(exprs: &[Expr], name: &str) -> Vec<String> {
    let def = normalized(exprs)
        .into_iter()
        .find(|expr| {
            matches!(expr, Expr::Node(node, _) if node.tag() == DeepTag::Def
                && matches!(node.children_slice().first(),
                    Some(Expr::Atom(Atom::Name(found), _)) if found == name))
        })
        .unwrap_or_else(|| panic!("no def `{name}`"));
    let text = print_canonical(std::slice::from_ref(&def));
    let mut out = Vec::new();
    let mut rest = text.as_str();
    while let Some(start) = rest.find("(lit {") {
        let tail = &rest[start..];
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

/// `resugar`, then re-desugar and expand: the expanded program round trips.
fn expanded_round_trip(source: &str) -> Option<String> {
    let original = deep(source);
    let resugared = match resugar_program(&original) {
        Ok(declarations) => format_program(&declarations),
        Err(error) => return Some(format!("resugar failed: {error}\n{source}")),
    };
    let before = print_canonical(&normalized(&original));
    let after = print_canonical(&normalized(&deep(&resugared)));
    (before != after).then(|| {
        format!(
            "expanded round trip changed the Deep\n--- authored\n{source}--- resugared\n{resugared}--- authored Deep\n{before}\n--- resugared Deep\n{after}"
        )
    })
}

/// The unexpanded program, macros included, resugars to Surf that expands to
/// the same program.
fn unexpanded_round_trip(source: &str) -> Option<String> {
    let original = desugared(source);
    let resugared = match resugar_program(&original) {
        Ok(declarations) => format_program(&declarations),
        Err(error) => return Some(format!("resugar failed: {error}\n{source}")),
    };
    let before = print_canonical(&normalized(&expanded(&original, source)));
    let after = print_canonical(&normalized(&deep(&resugared)));
    (before != after).then(|| {
        format!(
            "unexpanded round trip changed the expanded Deep\n--- authored\n{source}--- resugared\n{resugared}--- authored Deep\n{before}\n--- resugared Deep\n{after}"
        )
    })
}

/// Compares the macro root with the direct root and checks both laws.
fn check_case(source: &str, failures: &mut Vec<String>) {
    let program = deep(source);
    let via_macro = literal_types(&program, "via_macro");
    let direct = literal_types(&program, "direct");
    if via_macro != direct {
        failures.push(format!(
            "macro form types its literals differently from the direct text\n{source}--- via macro\n{via_macro:#?}\n--- direct\n{direct:#?}"
        ));
    }
    failures.extend(expanded_round_trip(source));
    failures.extend(unexpanded_round_trip(source));
}

#[test]
fn a_macro_body_adopts_in_the_scope_of_each_use_site() {
    let mut failures = Vec::new();
    let mut cases = 0usize;
    for (name, body) in BODIES {
        for (site, place) in sites(name) {
            let source = format!(
                "{PRELUDE}macro m() = {body}\nvia_macro = {}\ndirect = {}\n",
                place("m()"),
                place(body)
            );
            cases += 1;
            let before = failures.len();
            check_case(&source, &mut failures);
            if failures.len() > before {
                failures.push(format!("(case: {site}, body `{body}`)"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failure(s) across {cases} cases:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// A macro argument adopts in the position its parameter occupies in the
/// body, in the call spelling and the pipe spelling of the invocation, and a
/// parameter used as a callee resolves to the argument at the use site.
#[test]
fn a_macro_argument_adopts_where_the_body_places_it() {
    let macros = "macro c(x) = cast(x, f64)\n\
                  macro p(x) = x |> cast(f64)\n\
                  macro k(x) = f(x)\n\
                  macro first(x, n) = f2b(x, n)\n\
                  macro cast_n(x, n) = cast(x, f64)\n\
                  macro apply(h) = h(to_tensor([1.1, 2.2]))\n";
    // (invocation, the text its expansion produces, a name the site rebinds)
    let cases: &[(&str, &str, &str)] = &[
        ("c(1.1)", "cast(1.1, f64)", "f"),
        ("c(-1.1)", "cast(-1.1, f64)", "f"),
        ("c(1)", "cast(1, f64)", "f"),
        ("c(to_tensor([1.1, 2.2]))", "cast(to_tensor([1.1, 2.2]), f64)", "to_tensor"),
        ("c([1.1, 2.2] |> to_tensor)", "cast([1.1, 2.2] |> to_tensor, f64)", "to_tensor"),
        ("p(1.1)", "1.1 |> cast(f64)", "f"),
        ("p(to_tensor([1.1, 2.2]))", "to_tensor([1.1, 2.2]) |> cast(f64)", "to_tensor"),
        ("k(to_tensor([1.1, 2.2]))", "f(to_tensor([1.1, 2.2]))", "f"),
        ("k(to_tensor([1.1, 2.2]))", "f(to_tensor([1.1, 2.2]))", "to_tensor"),
        ("to_tensor([1.1, 2.2]) |> first(0)", "to_tensor([1.1, 2.2]) |> f2b(0)", "f2b"),
        ("1.1 |> cast_n(0)", "1.1 |> cast(f64)", "f"),
        ("apply(f)", "f(to_tensor([1.1, 2.2]))", "f"),
        ("apply(f)", "f(to_tensor([1.1, 2.2]))", "to_tensor"),
    ];
    let mut failures = Vec::new();
    for (invocation, expansion, name) in cases {
        for (site, place) in sites(name) {
            let source = format!(
                "{PRELUDE}{macros}via_macro = {}\ndirect = {}\n",
                place(invocation),
                place(expansion)
            );
            let before = failures.len();
            check_case(&source, &mut failures);
            if failures.len() > before {
                failures.push(format!("(case: {site}, `{invocation}`)"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failure(s):\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// An unexpanded macro body or argument has no dtype until expansion, so its
/// authored spelling decides: `1.1` adopts at each expansion and `1.1f32`
/// never does. Both survive resugaring of the unexpanded program.
#[test]
fn unexpanded_macro_literals_keep_their_authored_spelling() {
    let source = "macro bare() = cast(1.1, f64)\n\
                  macro suffixed() = cast(1.1f32, f64)\n\
                  macro c(x) = cast(x, f64)\n\
                  a = bare()\n\
                  b = suffixed()\n\
                  c1 = c(1.1)\n\
                  c2 = c(1.1f32)\n\
                  c3 = c(neg(1.1f32))\n";
    let program = deep(source);
    let f64_literal = "(lit {type: (t-prim {} f64)} 1.1)";
    let f32_literal = "(lit {type: (t-prim {} f32)} 1.1)";
    for (name, expected) in [
        ("a", f64_literal),
        ("b", f32_literal),
        ("c1", f64_literal),
        ("c2", f32_literal),
        ("c3", f32_literal),
    ] {
        assert_eq!(
            literal_types(&program, name),
            vec![expected.to_string()],
            "{name} in\n{source}"
        );
    }
    assert_eq!(unexpanded_round_trip(source), None);
    assert_eq!(expanded_round_trip(source), None);
}

/// The printer decides a literal's suffix from the dtype an unsuffixed
/// literal derives where it is printed, never from how it was authored: an
/// `f64` literal at a default position keeps its suffix whatever its style
/// marker says.
#[test]
fn a_literal_prints_its_suffix_wherever_its_position_derives_another_dtype() {
    for deep_text in [
        "(def {} a (lit {surf_literal_style: \"unsuffixed\", type: (t-prim {} f64)} 1.1))",
        "(def {} a (lit {type: (t-prim {} f64)} 1.1))",
    ] {
        let exprs = chelis_deep::parse_and_stamp(deep_text).expect("Deep parses");
        let surf = format_program(&resugar_program(&exprs).expect("Deep resugars"));
        assert_eq!(surf, "a = 1.1f64\n", "{deep_text}");
    }
}
