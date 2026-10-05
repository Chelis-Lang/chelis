//! chelis#3164 (A1): a numeric literal's dtype is written at its site.
//!
//! `spec/04-type-system.md` §5.3 and §5.6, `spec/02-surf-syntax.md` §P10b,
//! and `spec/03-deep-syntax.md` §6.4: a literal takes its dtype from its
//! suffix, or from a dtype-stating construct that directly contains it (a
//! declaration, a `cast`, or the dtype argument of `to_tensor`), or else from
//! its default; an unsuffixed literal element of a `to_tensor` call without a
//! dtype argument has no default. These tests pin the Deep the desugarer
//! produces for each construct and the Surf-ingress rejections it owns:
//! the missing dtype, a literal whose kind cannot bind, a dtype argument
//! that is not a dtype, the chelis#3148 binder fence, and the reserved
//! `to_tensor` binder (`spec/04-type-system.md` §8.6).
//!
//! Every accepting row has a rejecting or non-adopting mirror in the same
//! test, so a mechanism that over-applies fails as surely as one that is
//! missing.

use chelis_deep::printer::print_expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;

/// The desugared program, one Deep form per line, whitespace flattened and
/// `span` metadata removed.
fn deep(source: &str) -> String {
    let decls = parse_str(source).unwrap_or_else(|error| panic!("parse: {error}\n{source}"));
    let exprs =
        desugar_program(&decls).unwrap_or_else(|error| panic!("desugar: {error}\n{source}"));
    strip_spans(
        &exprs
            .iter()
            .map(|expr| {
                print_expr(expr)
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// The desugaring error a Surf program must produce.
fn desugar_error(source: &str) -> String {
    let decls = parse_str(source).unwrap_or_else(|error| panic!("parse: {error}\n{source}"));
    match desugar_program(&decls) {
        Ok(exprs) => panic!(
            "expected a desugaring error, got:\n{}\n--- source\n{source}",
            exprs.iter().map(print_expr).collect::<Vec<_>>().join("\n")
        ),
        Err(error) => error.to_string(),
    }
}

/// Removes every `span` metadata entry, so programs written with different
/// spellings compare by their Deep alone.
fn strip_spans(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("span: \"") {
        out.push_str(&rest[..start]);
        let after = &rest[start + "span: \"".len()..];
        let end = after.find('"').expect("closed span string");
        rest = &after[end + 1..];
        if let Some(stripped) = rest.strip_prefix(", ") {
            rest = stripped;
        }
    }
    out.push_str(rest);
    out.replace(", }", "}")
}

fn assert_contains(haystack: &str, needle: &str, context: &str) {
    assert!(
        haystack.contains(needle),
        "expected `{needle}` in:\n{haystack}\n--- {context}"
    );
}

fn assert_lacks(haystack: &str, needle: &str, context: &str) {
    assert!(
        !haystack.contains(needle),
        "unexpected `{needle}` in:\n{haystack}\n--- {context}"
    );
}

// === Cast ===

/// §5.6 Cast and spec/03 §6.4: `cast(lit, p)` at a primitive `p` desugars to
/// exactly the Deep of the suffixed literal, with no `cast` node. A signed
/// operand folds into one signed literal at `p`. Everything else keeps its
/// `cast` node: a binder target (the cast converts at integer members), an
/// explicit `neg(...)` operand, a named cast, a float literal under an
/// integer target, and a suffixed operand.
#[test]
fn a_literal_cast_to_a_primitive_is_the_suffixed_literal_and_nothing_else_collapses() {
    for (cast, suffixed) in [
        ("cast(1.1, f64)", "1.1f64"),
        ("cast(1.1, f32)", "1.1f32"),
        ("cast(1.5, bf16)", "1.5bf16"),
        ("cast(1.5, f16)", "1.5f16"),
        ("cast(5, f64)", "5f64"),
        ("cast(3000000000, i64)", "3000000000i64"),
        ("cast(7, i8)", "7i8"),
        ("cast(7, i32)", "7i32"),
        // A pipe normalizes to the call before literal dtype selection.
        ("1.1 |> cast(f64)", "1.1f64"),
        ("7 |> cast(i64)", "7i64"),
    ] {
        let collapsed = strip_spans(&deep(&format!("x = {cast}\n")));
        let literal = strip_spans(&deep(&format!("x = {suffixed}\n")));
        assert_eq!(collapsed, literal, "{cast} must desugar as {suffixed}");
    }
    for (cast, folded) in [
        ("cast(-1.1, f64)", "type: (t-prim {} f64)} -1.1)"),
        ("cast(-128, i8)", "type: (t-prim {} i8)} -128)"),
        (
            "cast(-3000000000, i64)",
            "type: (t-prim {} i64)} -3000000000)",
        ),
        ("cast(-2, f64)", "type: (t-prim {} f64)} -2)"),
    ] {
        let text = deep(&format!("x = {cast}\n"));
        assert_contains(&text, folded, cast);
        assert_lacks(&text, "(cast ", cast);
        assert_lacks(&text, "(var {} neg)", cast);
    }

    // The mirrors keep their cast node and their operand's own dtype.
    for (cast, operand) in [
        ("cast(neg(1.1), f64)", "type: (t-prim {} f32)} 1.1)"),
        ("cast_trunc(1.9, i32)", "type: (t-prim {} f32)} 1.9)"),
        ("cast_wrap(300, i8)", "type: (t-prim {} i32)} 300)"),
        ("cast(2.5, i32)", "type: (t-prim {} f32)} 2.5)"),
        ("cast(1.1f32, f64)", "type: (t-prim {} f32)} 1.1)"),
        ("cast(1.1f32, f32)", "type: (t-prim {} f32)} 1.1)"),
        ("cast(1.1f64, f64)", "type: (t-prim {} f64)} 1.1)"),
        ("cast(1, bool)", "type: (t-prim {} i32)} 1)"),
        ("cast(1.0, bool)", "type: (t-prim {} f32)} 1.0)"),
        ("1.1f32 |> cast(f64)", "type: (t-prim {} f32)} 1.1)"),
        ("2.5 |> cast(i32)", "type: (t-prim {} f32)} 2.5)"),
    ] {
        let text = deep(&format!("x = {cast}\n"));
        assert_contains(&text, "(cast ", cast);
        assert_contains(&text, operand, cast);
    }
    let binder = deep("def h[p: Numeric](x: p) -> p = add(x, cast(2.5, p))\n");
    assert_contains(
        &binder,
        "surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 2.5) (t-var {} p))",
        "a binder cast keeps its node and the unsuffixed marker",
    );
    assert_contains(&binder, "(cast ", "binder cast");
}

// === Declaration ===

/// §5.6 Declaration: the declared type of a binding (inline annotation, else
/// standalone `sig`) or of a function result (inline, else `sig`) states the
/// dtype of an initializer that is a literal or its negation, top level and
/// in a block. An integer literal at a float type is the cross-family form.
#[test]
fn a_declaration_states_the_dtype_of_its_literal_initializer() {
    for (source, literal) in [
        ("x: f64 = 1.1\n", "type: (t-prim {} f64)} 1.1)"),
        ("sig x: f64\nx = 1.1\n", "type: (t-prim {} f64)} 1.1)"),
        ("def f() -> f64 = 1.1\n", "type: (t-prim {} f64)} 1.1)"),
        (
            "sig g: f64 -> f64\ndef g(a) = 1.1\n",
            "type: (t-prim {} f64)} 1.1)",
        ),
        (
            "def f() -> f64 = {\n  y: f64 = 1.1\n  y\n}\n",
            "type: (t-prim {} f64)} 1.1)",
        ),
        ("x: i8 = -128\n", "type: (t-prim {} i8)} -128)"),
        ("def f() -> i8 = -128\n", "type: (t-prim {} i8)} -128)"),
        ("x: f64 = -1.1\n", "type: (t-prim {} f64)} -1.1)"),
        (
            "x: i64 = 3000000000\n",
            "type: (t-prim {} i64)} 3000000000)",
        ),
        ("x: f16 = 1.5\n", "type: (t-prim {} f16)} 1.5)"),
        ("x: f64 = 5\n", "type: (t-prim {} f64)} 5)"),
    ] {
        let text = deep(source);
        assert_contains(&text, literal, source);
        assert_lacks(&text, "(var {} neg)", source);
    }
    assert_contains(
        &deep("x: f64 = 5\n"),
        "literal_source: integer",
        "an integer literal at a declared float dtype",
    );

    // Not directly contained, an alias, a declared List, or an ascription:
    // the literal keeps its default.
    for source in [
        "x: f64 = neg(1.1)\n",
        "x: f64 = if true then 1.1 else 2.2\n",
        "x: f64 = {\n  y = 0\n  1.1\n}\n",
        "x: f64 = add(1.1, 0.0f64)\n",
        "type P = f64\nx: P = 1.1\n",
        "x = (1.1 : f64)\n",
    ] {
        let text = deep(source);
        assert_contains(&text, "type: (t-prim {} f32)} 1.1)", source);
        assert_lacks(&text, "type: (t-prim {} f64)} 1.1)", source);
    }
    let shape = deep("shape: List[i64] = [2, 3]\n");
    assert_contains(&shape, "type: (t-prim {} i32)} 2)", "a declared List");
    assert_lacks(&shape, "type: (t-prim {} i64)} 2)", "a declared List");
}

/// §5.6 Binding: a float literal cannot bind at an integer type, nor any
/// numeric literal at `bool`; in a declaration that is a type error naming the
/// literal and the declared dtype. A suffixed literal binds at its suffix and
/// is left for the checker.
#[test]
fn a_declaration_rejects_a_literal_whose_kind_cannot_bind() {
    for (source, needle) in [
        ("x: i32 = 1.5\n", "cannot bind at i32"),
        ("def f() -> i64 = 2.0\n", "cannot bind at i64"),
        ("b: bool = 1\n", "cannot bind at bool"),
        ("s: string = 1\n", "cannot bind at string"),
        ("xs: tensor[2, i32] = [1.5, 2]\n", "cannot bind at i32"),
    ] {
        let error = desugar_error(source);
        assert_contains(&error, needle, source);
        assert_contains(&error, "§5.6", source);
    }
    assert_contains(
        &deep("x: i32 = 1.5f64\n"),
        "type: (t-prim {} f64)} 1.5)",
        "a suffixed initializer keeps its suffix",
    );
}

// === Dtype argument ===

/// §5.6 Dtype argument and spec/03 §6.4: `to_tensor(xs, p)` keeps `p` as a
/// third `app` child and gives every literal element of a bracket-literal `xs`
/// the dtype `p`, recursively and through a folded negation. Elements that
/// are not literal elements keep their own dtype.
#[test]
fn a_dtype_argument_types_every_literal_element_and_stays_in_the_deep() {
    let flat = deep("x = to_tensor([1.1, 2.2], f64)\n");
    assert_contains(&flat, "(t-prim {} f64))", "dtype child");
    assert_contains(&flat, "type: (t-prim {} f64)} 1.1)", "first element");
    assert_contains(&flat, "type: (t-prim {} f64)} 2.2)", "second element");
    assert_lacks(&flat, "(t-prim {} f32)", "no default element");
    assert_lacks(&flat, "(var {} f64)", "the dtype is not a value");
    assert_eq!(
        strip_spans(&deep("x = [1.1, 2.2] |> to_tensor(f64)\n")),
        strip_spans(&flat),
        "a pipe into to_tensor(f64) is the call"
    );

    let nested = deep("x = to_tensor([[1, 2], [3, -4]], i64)\n");
    for literal in ["} 1)", "} 2)", "} 3)", "} -4)"] {
        assert_contains(
            &nested,
            &format!("type: (t-prim {{}} i64){literal}"),
            "nested elements",
        );
    }
    assert_contains(
        &deep("x = to_tensor([1, 2], f64)\n"),
        "literal_source: integer",
        "integer elements at a float dtype argument",
    );
    assert_contains(
        &deep("x = to_tensor([], f64)\n"),
        "(var {} Nil) (t-prim {} f64))",
        "an empty List with a dtype argument",
    );
    assert_contains(
        &deep("x = to_tensor([true, false], bool)\n"),
        "(t-prim {} bool))",
        "a bool dtype argument",
    );
    assert_contains(
        &deep("x = to_tensor([(1.5), 2.5], f64)\n"),
        "type: (t-prim {} f64)} 1.5)",
        "grouping parentheses are transparent",
    );

    // Not literal elements: a variable, a call's argument, a `neg(...)` call.
    let mixed = deep(
        "a = 1.5f64\ndef id(v: f32) -> f32 = v\nx = to_tensor([a, 2.5], f64)\ny = to_tensor([cast(id(1.5), f64), 2.5], f64)\nz = to_tensor([neg(3.5), 2.5], f64)\n",
    );
    assert_contains(&mixed, "(var {} a)", "a variable element");
    assert_contains(
        &mixed,
        "type: (t-prim {} f32)} 1.5)",
        "a call argument keeps its default",
    );
    assert_contains(
        &mixed,
        "type: (t-prim {} f32)} 3.5)",
        "a neg operand keeps its default",
    );
    assert_contains(
        &mixed,
        "type: (t-prim {} f64)} 2.5)",
        "literal elements beside them",
    );
}

/// §P9: the dtype argument position always names a dtype, never a value, even
/// where a value of the same name is in scope; an identifier that names no
/// dtype there is rejected.
#[test]
fn the_dtype_argument_position_names_a_dtype_and_never_a_value() {
    let shadowed = deep("r = {\n  f64 = 2\n  to_tensor([1.5], f64)\n}\n");
    assert_contains(
        &shadowed,
        "(t-prim {} f64))",
        "a value named f64 is ignored",
    );
    assert_contains(&shadowed, "type: (t-prim {} f64)} 1.5)", "the element");
    for source in [
        "v = 2\nr = to_tensor([1.5], v)\n",
        "r = to_tensor([1.5], 2)\n",
        "r = to_tensor([1.5], Float)\n",
    ] {
        let error = desugar_error(source);
        assert_contains(&error, "must be a dtype", source);
    }
}

/// §5.6 Binding, fenced under chelis#3148: a dtype-bounded binder stated by a
/// declaration or a dtype argument, rather than by a cast, is rejected loudly
/// with the working `cast(lit, p)` spelling. A binder dtype argument over
/// elements that are casts states no literal's dtype and is not fenced.
#[test]
fn a_binder_stated_outside_a_cast_is_fenced_under_3148() {
    for source in [
        "def k[p: Float](x: p) -> tensor[1, p] = to_tensor([1.5], p)\n",
        "def k[p: Float](x: p) -> p = 1.5\n",
        "def k[p: Float](x: p) -> tensor[2, p] = [1.5, 2.5]\n",
    ] {
        let error = desugar_error(source);
        assert_contains(&error, "chelis#3148", source);
        assert_contains(&error, "cast(", source);
    }
    let cast = deep("def k[p: Float](x: p) -> tensor[1, p] = to_tensor([cast(1.5, p)], p)\n");
    assert_contains(&cast, "(t-var {} p))", "the binder dtype child");
    assert_contains(
        &cast,
        "type: (t-var {} p)} 1.5) (t-var {} p))",
        "the binder cast element",
    );
}

// === Missing dtype ===

/// §5.6 "Tensor elements state their dtype": an unsuffixed literal element of
/// a `to_tensor` call without a dtype argument is a type error whose
/// diagnostic names the value-preserving dtype argument; under a cast it also
/// names the target. Suffixed elements and a `List` bound first are accepted.
#[test]
fn an_unsuffixed_to_tensor_element_without_a_dtype_argument_is_rejected() {
    for (source, fix) in [
        ("x = to_tensor([1.1, 2.2])\n", "to_tensor([1.1, 2.2], f32)"),
        ("x = to_tensor([1, 2])\n", "to_tensor([1, 2], i32)"),
        ("x = to_tensor([[1.5], [2.5]])\n", "f32"),
        ("x = to_tensor([-1.5, 2.5f32])\n", "f32"),
        ("a = 1.5f32\nx = to_tensor([a, 2.5])\n", "f32"),
        ("xs: tensor[2, f32] = to_tensor([1.1, 2.2])\n", "f32"),
        ("def f() -> tensor[2, f32] = to_tensor([1.5, 2.5])\n", "f32"),
        ("macro tt() = to_tensor([1.5, 2.5])\n", "f32"),
        ("x = [1.1, 2.2] |> to_tensor\n", "f32"),
    ] {
        let error = desugar_error(source);
        assert_contains(&error, "states no dtype", source);
        assert_contains(&error, "§5.6", source);
        assert_contains(&error, fix, source);
    }
    let under_cast = desugar_error("x = cast(to_tensor([1.1, 2.2]), f64)\n");
    assert_contains(
        &under_cast,
        "to_tensor([1.1, 2.2], f64)",
        "builds at the target",
    );
    assert_contains(
        &under_cast,
        "cast(to_tensor([1.1, 2.2], f32), f64)",
        "states the f32 rounding",
    );

    // Mirrors: every element suffixed, or the List bound before the call.
    let suffixed = deep("x = to_tensor([1.1f32, 2.2f32])\n");
    assert_contains(
        &suffixed,
        "type: (t-prim {} f32)} 1.1)",
        "suffixed elements",
    );
    let piped = deep("x = [1.1f32, 2.2f32] |> to_tensor\n");
    assert_contains(
        &piped,
        "type: (t-prim {} f32)} 1.1)",
        "suffixed piped elements",
    );
    let bound = deep("xs = [1.1, 2.2]\nx = to_tensor(xs)\n");
    assert_contains(&bound, "type: (t-prim {} f32)} 1.1)", "a List bound first");
    let declared = deep("xs: tensor[2, f64] = [1.1, 2.2]\n");
    assert_contains(&declared, "type: (t-prim {} f64)} 1.1)", "a tensor literal");
}

// === Not a site ===

/// §5.6 "directly": a callee's parameter type states nothing, and a literal
/// inside any other expression keeps its own suffix or default.
#[test]
fn a_callee_signature_states_no_dtype() {
    let text = deep("def take(v: f64) -> f64 = v\nx = take(1.5)\ny: f64 = cast(1.5, f64)\n");
    assert_contains(&text, "type: (t-prim {} f32)} 1.5)", "a call argument");
    assert_contains(&text, "type: (t-prim {} f64)} 1.5)", "the cast beside it");
    assert_eq!(
        text.matches("(cast ").count(),
        0,
        "the cast collapsed and the call added none:\n{text}"
    );
}

// === Reserved name ===

/// §8.6 and §P10b: `to_tensor` is reserved; no binder in any scope may bind
/// it, so a `to_tensor` call always names the intrinsic. chelis#3152's and
/// chelis#3167's reproductions are among the rows. A call and a value use
/// are not binders.
#[test]
fn every_binder_of_to_tensor_is_rejected_at_surf_ingress() {
    for source in [
        "def f(to_tensor: i32) -> i32 = to_tensor\n",
        "g = fn (to_tensor: i32) -> to_tensor\n",
        "r = {\n  to_tensor = 1\n  to_tensor\n}\n",
        "r = {\n  (to_tensor, y) = (1, 2)\n  y\n}\n",
        "r = match Some(1) with { | Some(to_tensor) => to_tensor | None => 0 }\n",
        "macro m(to_tensor) = to_tensor\nr = m(1)\n",
        "macro to_tensor(x) = x\n",
        "def to_tensor(x: i32) -> i32 = x\n",
        "sig to_tensor: i32 -> i32\ndef to_tensor(x) = x\n",
        "to_tensor = 1\n",
        "import Std.Sort (to_tensor)\nr = 1\n",
        "def sample() = {\n  to_tensor = fn (xs: List[i32]) -> xs\n  xs: tensor[2, i32] = [1, 2]\n  xs\n}\n",
        // chelis#3152
        "def mk(l: List[f32]) -> tensor[2, f32] = to_tensor(l)\n\
         def g2(to_tensor: List[f32] -> tensor[2, f32]) -> tensor[2, f32] = to_tensor([1.1f32, 2.2f32])\n\
         r2 = g2(mk)\n",
        // chelis#3167
        "def tt[p: Float](l: List[p]) -> tensor[2, p] = to_tensor(l)\n\
         macro m() = {\n  xs: tensor[2, f64] = [1.1, 2.2]\n  xs\n}\n\
         via_macro = {\n  to_tensor = tt\n  m()\n}\n",
    ] {
        let error = desugar_error(source);
        assert_contains(&error, "`to_tensor` is reserved", source);
        assert_contains(&error, "§8.6", source);
    }
    let uses = deep("xs = [1.5f32]\nf = to_tensor\nr = to_tensor(xs)\n");
    assert_contains(&uses, "(var {} to_tensor)", "uses of the intrinsic");
}
