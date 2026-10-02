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
//! The rejected programs must fail at code generation with that rejection.
//! The neighbouring programs that already compiled must still compile, run
//! with every ledger allocation finalized, and print what `chelis eval`
//! prints.

mod ownership_support;

use chelis_compiler_api::compiler::{CompilerError, compile, eval};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, EvalRequest, SourceKind};
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
    name: &'static str,
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
                name: "projection_of_where_built_record",
                body: format!("{MASKED}r_a = col_xs(masked({VALUES}).0)\n"),
            },
            "is rendered before it is declared",
        ),
        // Without the third `filled` no site produces the extent at all.
        (
            Case {
                name: "projection_without_a_producing_site",
                body: format!(
                    "def masked[n](t: tensor[n, i64]) -> (Col[n], tensor[n, bool]) = {{\n  mask = in_range(t)\n  (Col {{ xs: where(mask, t, t) }}, mask)\n}}\nr_a = col_xs(masked({VALUES}).0)\n"
                ),
            },
            "is rendered but never declared",
        ),
    ]
}

fn compiled() -> Vec<Case> {
    vec![
        Case {
            name: "mask_projection",
            body: format!("{MASKED}r_a = masked({VALUES}).1\n"),
        },
        Case {
            name: "destructured_record",
            body: format!("{MASKED}r_a = {{\n  (c, m) = masked({VALUES})\n  col_xs(c)\n}}\n"),
        },
        Case {
            name: "unwrapped_where",
            body: format!(
                "r_a = {{\n  t = {VALUES}\n  mask = in_range(t)\n  where(mask, t, filled(t, 0i64))\n}}\n"
            ),
        },
    ]
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
        let generated = ownership_support::emit(&source, case.name);
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
