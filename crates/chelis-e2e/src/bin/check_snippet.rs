use std::env;
use std::io::{self, Read};

use chelis_deep::validate::validate;
use chelis_surf::parser::parse_str as parse_surf;

fn json_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 8);
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
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
    let parse = parse_surf(source);
    let mut json = String::new();
    match parse {
        Ok(decls) => {
            let prepared = chelis_compiler_api::pipeline::prepare_surf_decls(&decls, None)
                .expect("macro expansion should succeed for snippet checking");
            let report = analyze_fitness(prepared);
            json.push('{');
            json.push_str("\"lang\":\"surf\",");
            json.push_str("\"parse_error\":null,");
            json.push_str(&format!("\"fitness\":{},", report.score));
            json.push_str("\"warnings\":[],");
            json.push_str("\"errors\":[");
            for (index, error) in report.errors.iter().enumerate() {
                if index > 0 {
                    json.push(',');
                }
                json.push_str(&format!(
                    "{{\"kind\":\"{}\",\"message\":\"{}\"}}",
                    json_escape(&format!("{:?}", error.kind)),
                    json_escape(&error.message)
                ));
            }
            json.push_str("]}");
        }
        Err(error) => {
            json.push('{');
            json.push_str("\"lang\":\"surf\",");
            json.push_str(&format!(
                "\"parse_error\":\"{}\",",
                json_escape(&error.to_string())
            ));
            json.push_str("\"fitness\":0.0,");
            json.push_str("\"warnings\":[],");
            json.push_str("\"errors\":[]}");
        }
    }
    println!("{json}");
}

fn check_deep(source: &str) {
    let strict = chelis_deep::parser::parse_str_strict(source);
    let mut json = String::new();
    match strict {
        Ok(exprs) => {
            let warnings = validate(&exprs);
            let prepared = chelis_compiler_api::pipeline::prepare_deep(exprs, None);
            let report = analyze_fitness(prepared);
            json.push('{');
            json.push_str("\"lang\":\"deep\",");
            json.push_str("\"parse_error\":null,");
            json.push_str(&format!("\"fitness\":{},", report.score));
            json.push_str("\"warnings\":[");
            for (index, warning) in warnings.iter().enumerate() {
                if index > 0 {
                    json.push(',');
                }
                json.push_str(&format!(
                    "{{\"kind\":\"{}\",\"message\":\"{}\"}}",
                    json_escape(&format!("{:?}", warning.kind)),
                    json_escape(&warning.message)
                ));
            }
            json.push_str("],");
            json.push_str("\"errors\":[");
            for (index, error) in report.errors.iter().enumerate() {
                if index > 0 {
                    json.push(',');
                }
                json.push_str(&format!(
                    "{{\"kind\":\"{}\",\"message\":\"{}\"}}",
                    json_escape(&format!("{:?}", error.kind)),
                    json_escape(&error.message)
                ));
            }
            json.push_str("]}");
        }
        Err(error) => {
            json.push('{');
            json.push_str("\"lang\":\"deep\",");
            json.push_str(&format!(
                "\"parse_error\":\"{}\",",
                json_escape(&error.to_string())
            ));
            json.push_str("\"fitness\":0.0,");
            json.push_str("\"warnings\":[],");
            json.push_str("\"errors\":[]}");
        }
    }
    println!("{json}");
}
