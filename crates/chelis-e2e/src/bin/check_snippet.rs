use std::env;
use std::io::{self, Read};

use chelis_deep::validate::validate;
use chelis_surf::parser::parse_str as parse_surf;
use serde::Serialize;

/// The snippet checker's report document (chelis#886).
///
/// This binary assembled its JSON with `format!` in four places -- two error
/// loops, a warning loop spelling the kind with `{:?}`, and four copies of the
/// document body -- which is the defect chelis#886 names, surviving in a
/// second producer after #1384 converted the CLI's.
///
/// Field order here IS the wire order, and NOTHING PINS THAT: no test in the
/// repo executes this binary's output, so a field reorder changes the wire
/// silently. The sibling `bench_phase1e` in this crate is output-tested via
/// `Command::cargo_bin`; this one is not.
///
/// The document is JSON-VALUE-identical to the hand-written version, not
/// byte-identical. Two spellings changed:
///
///   * `fitness` gained its integral fraction (`1` -> `1.0`). The old success
///     path used Rust `Display` while the parse-error path pushed a hardcoded
///     `0.0`, so one field was spelled two ways depending on which branch
///     produced it. It is now uniform.
///   * Control-character escaping. `json_escape` escaped every
///     `char::is_control()` codepoint as `\uXXXX`; serde escapes only
///     `0x00-0x1F`, with named short forms. Exactly 35 codepoints differ,
///     enumerated rather than sampled: `U+0008` -> `\b`, `U+000C` -> `\f`,
///     and all 33 of `U+007F`-`U+009F` emitted raw.
///
///     An earlier revision listed four of those 35, having checked the cases
///     it thought of rather than the domain. The set is small and closed, so
///     it is stated in full.
///
/// Both remain valid JSON that `json.loads` reads to equal values, and
/// `skill_eval.py`'s verdicts are unchanged across the divergent inputs.
///
/// `SnippetReport.fitness` is a bare `f64` on a serde root. No §C6 enumerator
/// covers `chelis-e2e` -- `capacity_census_typed.py` filters to
/// `chelis_compiler_api::schema` and `chelis_python` -- so no gate is broken,
/// and the same number was already on this wire. Recorded because #1384
/// reasoned explicitly about not creating a numeric wire root outside the
/// census, and this makes the opposite move; extending the census is a
/// different change.
#[derive(Serialize)]
struct SnippetReport {
    lang: &'static str,
    parse_error: Option<String>,
    fitness: f64,
    warnings: Vec<SnippetDiagnostic>,
    errors: Vec<SnippetDiagnostic>,
}

/// One reported item. Deliberately NOT `schema::Diagnostic`: this document has
/// its own narrower shape (`kind` and `message` only), and widening it here
/// would be a wire migration rather than the producer repair chelis#886 asks
/// for. The kind is always a governed spelling, never a Rust identifier.
#[derive(Serialize)]
struct SnippetDiagnostic {
    kind: &'static str,
    message: String,
}

impl SnippetReport {
    fn parse_failed(lang: &'static str, error: String) -> Self {
        Self {
            lang,
            parse_error: Some(error),
            fitness: 0.0,
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn emit(&self) {
        println!(
            "{}",
            serde_json::to_string(self).expect("the snippet report is plain data and cannot fail")
        );
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(flag) = args.next() else {
        eprintln!("usage: check_snippet --lang surf|deep");
        std::process::exit(2);
    };
    if flag != "--lang" {
        eprintln!("usage: check_snippet --lang surf|deep");
        std::process::exit(2);
    }
    let Some(lang) = args.next() else {
        eprintln!("usage: check_snippet --lang surf|deep");
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("usage: check_snippet --lang surf|deep");
        std::process::exit(2);
    }

    let mut source = String::new();
    io::stdin()
        .read_to_string(&mut source)
        .expect("failed to read stdin");

    match lang.as_str() {
        "surf" => check_surf(&source),
        "deep" => check_deep(&source),
        _ => {
            eprintln!("unknown lang: {lang}");
            std::process::exit(2);
        }
    }
}

fn analyze_fitness(
    prepared: chelis_compiler_api::pipeline::PreparedProgram,
) -> chelis_types::FitnessReport {
    let outcome = chelis_compiler_api::pipeline::run_prepared(
        prepared,
        chelis_compiler_api::pipeline::PipelineGoal::TypeAnalysis,
    )
    .expect("prepared type analysis cannot fail before the type stage");
    let chelis_compiler_api::pipeline::PipelineOutcome::TypeAnalysis(analysis) = outcome else {
        unreachable!("the type-analysis goal returns only a type-analysis outcome")
    };
    match analysis {
        chelis_types::TypeAnalysisOutcome::Rejected { fitness }
        | chelis_types::TypeAnalysisOutcome::Accepted { fitness, .. } => fitness,
    }
}

fn check_surf(source: &str) {
    match parse_surf(source) {
        Ok(decls) => {
            let prepared = chelis_compiler_api::pipeline::prepare_surf_decls(&decls, None)
                .expect("macro expansion should succeed for snippet checking");
            let report = analyze_fitness(prepared);
            SnippetReport {
                lang: "surf",
                parse_error: None,
                fitness: report.score,
                warnings: Vec::new(),
                errors: report
                    .errors
                    .iter()
                    .map(|error| SnippetDiagnostic {
                        kind: error.kind.diagnostic_name(),
                        message: error.message.clone(),
                    })
                    .collect(),
            }
            .emit();
        }
        Err(error) => SnippetReport::parse_failed("surf", error.to_string()).emit(),
    }
}

fn check_deep(source: &str) {
    // chelis#1088: the snippet checker feeds `prepare_deep`, so it must use
    // the same stamped `.dp` ingress the compiler does. A weaker parse here
    // would report a fitness score for a tree the compiler never accepts.
    match chelis_deep::parse_and_stamp_file(source) {
        Ok(exprs) => {
            let warnings = validate(&exprs);
            let prepared = chelis_compiler_api::pipeline::prepare_deep(exprs, None);
            let report = analyze_fitness(prepared);
            SnippetReport {
                lang: "deep",
                parse_error: None,
                fitness: report.score,
                warnings: warnings
                    .iter()
                    .map(|warning| SnippetDiagnostic {
                        kind: warning.kind.wire_name(),
                        message: warning.message.clone(),
                    })
                    .collect(),
                errors: report
                    .errors
                    .iter()
                    .map(|error| SnippetDiagnostic {
                        kind: error.kind.diagnostic_name(),
                        message: error.message.clone(),
                    })
                    .collect(),
            }
            .emit();
        }
        Err(error) => SnippetReport::parse_failed("deep", error.to_string()).emit(),
    }
}
