//! chelis#2883: a compiled tensor function never reads a runtime extent name
//! before it declares it.
//!
//! The C emitter declares an extent name at the site its extent origin
//! resolves to. In the issue's program the name of `masked`'s `n` resolves to
//! the `insert` inside the later `filled(t, 0i64)`, while the `and` that
//! builds the mask reads the name in its guard first, so `chelis build`
//! succeeded and wrote C that does not compile. The emitter now refuses a
//! name the finished function reads before its declaration with the same
//! typed rejection it gives a name it never declares. Building the program
//! is the remaining work of chelis#2883.
//!
//! The order comes from marks the emitter puts on a name where it renders
//! or declares it, not from identifier spellings in the C text, so a
//! dimension spelled like a helper's parameter (`index`, `value`, `low`), a
//! local (`word`, `element`), a member (`dtype`) or a checked-cast helper's
//! variable (`sign`, `shift`) is never taken for an early read. Nor is a
//! span ID the C carries in a comment, whatever it spells, and a span ID
//! carrying one of the mark characters is refused where Deep is parsed.
//!
//! The rejected programs must fail at code generation with that rejection.
//! The neighbouring programs that already compiled must still compile, run
//! with every ledger allocation finalized, and print what `chelis eval`
//! prints.

mod ownership_support;

use chelis_compiler_api::compiler::{CompilerError, compile, desugar, eval};
use chelis_compiler_api::schema::{
    CompileRequest, CompileTarget, DesugarRequest, EvalRequest, SourceKind,
};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const PRELUDE: &str = "module Demo.Main\n\
type Col[n] =\n  | Col { xs: tensor[n, i64] }\n\
def filled[n](like: &tensor[n, i64], value: i64) -> tensor[n, i64] = value |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))\n\
def in_range[n](t: &tensor[n, i64]) -> tensor[n, bool] = and(gte(t, filled(t, -4371587i64)), lte(t, filled(t, 2932896i64)))\n\
def col_xs[n](c: Col[n]) -> tensor[n, i64] = c.xs\n";

const MASKED: &str = "def masked[n](t: tensor[n, i64]) -> (Col[n], tensor[n, bool]) = {\n  mask = in_range(t)\n  (Col { xs: where(mask, t, filled(t, 0i64)) }, mask)\n}\n";

const VALUES: &str = "to_tensor([5i64, 2932897i64, -9223372036854775807i64])";

struct Case {
    name: String,
    body: String,
}

fn source(case: &Case) -> String {
    format!("{PRELUDE}{}", case.body)
}

/// Each refused program with the half of the declaration invariant it
/// breaks.
fn refused() -> Vec<(Case, &'static str)> {
    vec![
        // The issue's program: the only site producing the extent comes
        // after the guard that reads it.
        (
            Case {
                name: "projection_of_where_built_record".into(),
                body: format!("{MASKED}r_a = col_xs(masked({VALUES}).0)\n"),
            },
            "is rendered before it is declared",
        ),
        // Without the third `filled` no site produces the extent at all.
        (
            Case {
                name: "projection_without_a_producing_site".into(),
                body: format!(
                    "def masked[n](t: tensor[n, i64]) -> (Col[n], tensor[n, bool]) = {{\n  mask = in_range(t)\n  (Col {{ xs: where(mask, t, t) }}, mask)\n}}\nr_a = col_xs(masked({VALUES}).0)\n"
                ),
            },
            "is rendered but never declared",
        ),
    ]
}

/// Dimension names spelled like identifiers the emitter writes before or
/// around the function's own declarations: the uniform-draw helpers'
/// parameters and locals, the tensor views' members, the entry loop's
/// variables, and a common binder.
const DIMENSION_NAMES: &[&str] = &[
    "index", "value", "low", "high", "word", "element", "dtype", "data", "row", "i", "n",
];

/// Dimension names spelled like the f16 and bf16 checked-cast helpers'
/// variables, which the emitter writes only when a program casts to them.
const CAST_DIMENSION_NAMES: &[&str] = &["sign", "shift"];

fn compiled() -> Vec<Case> {
    let mut cases = vec![
        Case {
            name: "mask_projection".into(),
            body: format!("{MASKED}r_a = masked({VALUES}).1\n"),
        },
        Case {
            name: "destructured_record".into(),
            body: format!("{MASKED}r_a = {{\n  (c, m) = masked({VALUES})\n  col_xs(c)\n}}\n"),
        },
        Case {
            name: "unwrapped_where".into(),
            body: format!(
                "r_a = {{\n  t = {VALUES}\n  mask = in_range(t)\n  where(mask, t, filled(t, 0i64))\n}}\n"
            ),
        },
    ];
    cases.extend(DIMENSION_NAMES.iter().map(|name| Case {
        name: format!("dimension_named_{name}"),
        body: format!(
            "def double_it[{name}](t: tensor[{name}, f32]) -> tensor[{name}, f32] = add(t, t)\nr_a = double_it(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
        ),
    }));
    cases.extend(CAST_DIMENSION_NAMES.iter().map(|name| Case {
        name: format!("f16_cast_dimension_named_{name}"),
        body: format!(
            "def halve[{name}](t: tensor[{name}, f32]) -> tensor[{name}, f16] = cast(t, f16)\nr_a = halve(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
        ),
    }));
    cases
}

/// The evaluator's rendering of every root, in the compiled driver's format.
fn evaluated(source: &str) -> String {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|error| panic!("evaluator rejected the case: {error:?}"));
    result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}\n",
                root.name.as_deref().expect("named root"),
                root.display.as_deref().expect("rendered root")
            )
        })
        .collect()
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn declaration_rejection(error: &CompilerError, how: &str) -> Result<(), String> {
    let messages: Vec<&str> = error
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect();
    let refused = messages.iter().any(|message| {
        message.contains("unsupported: extent `")
            && message.contains(how)
            && message.contains("(codegen:c)")
            && message.contains("unimplemented chelis#1277:")
    });
    if refused {
        Ok(())
    } else {
        Err(format!("not the extent declaration rejection: {error:?}"))
    }
}

fn check_refused(case: &Case, how: &str) -> Result<(), String> {
    let source = source(case);
    catch_unwind(AssertUnwindSafe(|| {
        // The evaluator runs the program, so the refusal is the C lane's.
        evaluated(&source);
        match compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            target: CompileTarget::C,
            entry_name: None,
        }) {
            Ok(_) => Err("compiled C that reads an extent before declaring it".to_string()),
            Err(error) => declaration_rejection(&error, how),
        }
    }))
    .map_err(panic_message)
    .and_then(|outcome| outcome)
    .map_err(|message| {
        let head: String = message.chars().take(600).collect();
        format!("{}:\n{source}\n  -> {head}", case.name)
    })
}

fn check_compiled(case: &Case) -> Result<(), String> {
    let source = source(case);
    catch_unwind(AssertUnwindSafe(|| {
        let expected = evaluated(&source);
        let generated = ownership_support::emit(&source, &case.name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
    }))
    .map_err(|payload| {
        let head: String = panic_message(payload).chars().take(600).collect();
        format!("{}:\n{source}\n  -> {head}", case.name)
    })
}

// REGRESSION TEST. On `1a772bea6` the issue's program built and its C failed
// to compile with an undeclared identifier.
#[test]
fn an_extent_read_before_its_declaration_is_refused() {
    let failures: Vec<String> = refused()
        .iter()
        .filter_map(|(case, how)| check_refused(case, how).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn neighbouring_programs_still_compile_as_eval_runs_them() {
    let failures: Vec<String> = compiled()
        .iter()
        .filter_map(|case| check_compiled(case).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// The program's Deep text, as `desugar` writes it with the ledger harness's
/// host functions, with every span ID replaced by `span`.
fn deep_with_spans(surf: &str, span: &str) -> String {
    let deep = desugar(DesugarRequest {
        source: format!("{surf}{}", ownership_support::HOST_ANCHOR),
    })
    .unwrap_or_else(|error| panic!("desugar: {error:?}"))
    .deep_text;
    let key = "span: \"";
    let mut out = String::new();
    let mut rest = deep.as_str();
    while let Some(start) = rest.find(key) {
        let value = start + key.len();
        let end = value + rest[value..].find('"').expect("a closed span ID");
        out.push_str(&rest[..value]);
        out.push_str(span);
        rest = &rest[end..];
    }
    assert!(out.contains(span), "no span ID to replace in:\n{deep}");
    out.push_str(rest);
    out
}

fn compile_c(source_kind: SourceKind, source: &str) -> Result<(), CompilerError> {
    compile(CompileRequest {
        source_kind,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .map(|_| ())
}

fn messages(error: &CompilerError) -> String {
    error
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

const DOUBLED: &str = "def double_it[n](t: tensor[n, f32]) -> tensor[n, f32] = add(t, t)\nr_a = double_it(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";

/// Span IDs are opaque producer strings, kept verbatim in the C's
/// `// span:` comments (spec/03 §1.1.1), and private-use characters are
/// legal in one. Bracketing with them spells an unclosed bracket, a
/// bracketed name that is no extent, and a bracketed extent name; none is an
/// extent read.
const PRIVATE_USE_SPANS: &[&str] = &["n_\u{E000}001", "\u{E000}zz\u{E002}", "\u{E000}n\u{E002}"];

#[test]
fn a_span_id_never_reads_or_declares_an_extent() {
    let surf = format!("{PRELUDE}{DOUBLED}");
    let expected = evaluated(&surf);
    let failures: Vec<String> = PRIVATE_USE_SPANS
        .iter()
        .filter_map(|span| {
            catch_unwind(AssertUnwindSafe(|| {
                let generated =
                    ownership_support::emit_deep(&deep_with_spans(&surf, span), "private_use_span");
                assert!(
                    generated.contains(&format!("// span: {span}\n")),
                    "{generated}"
                );
                let (summary, stdout) = ownership_support::run_program(&generated);
                ownership_support::balanced(&summary);
                assert_eq!(stdout, expected, "compiled output differs from eval");
            }))
            .err()
            .map(|payload| format!("{span:?}: {}", panic_message(payload)))
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// The issue's program, with every span ID spelling a declaration of the
/// extent it reads early, is still refused.
#[test]
fn a_span_id_does_not_hide_a_read_before_its_declaration() {
    let issue = format!("{PRELUDE}{MASKED}r_a = col_xs(masked({VALUES}).0)\n");
    let refused = compile_c(SourceKind::Deep, &deep_with_spans(&issue, "s"))
        .expect_err("the issue's program");
    let message = messages(&refused);
    let name = message
        .split("extent `")
        .nth(1)
        .and_then(|rest| rest.split('`').next())
        .unwrap_or_else(|| panic!("no extent named in {message}"));
    let deep = deep_with_spans(&issue, &format!("\u{E001}{name}\u{E002}"));
    let error = compile_c(SourceKind::Deep, &deep).expect_err("a span ID declares nothing");
    declaration_rejection(&error, "is rendered before it is declared").unwrap();
}

// The negative twin: the emitter marks extents with C0 control characters,
// which a span ID may not contain, so a span ID carrying one is refused where
// the Deep source is parsed and never reaches the emitter.
#[test]
fn a_span_id_with_a_mark_character_is_refused_at_parse() {
    let surf = format!("{PRELUDE}{DOUBLED}");
    for span in ["n_\u{1}001", "\u{1}zz\u{3}", "\u{1}n\u{3}", "\u{2}n\u{3}"] {
        let error = compile_c(SourceKind::Deep, &deep_with_spans(&surf, span))
            .expect_err("a control character in a span ID");
        let message = messages(&error);
        assert!(
            message.contains("metadata `span` requires a string without ASCII control characters (spec/03 §1.1.1)")
                && message.contains("forbidden character"),
            "{span:?}: {message}"
        );
    }
}
