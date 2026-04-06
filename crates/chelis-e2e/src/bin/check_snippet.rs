use std::env;
use std::io::{self, Read};

use chelis_deep::validate::validate;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::fitness::check_program;

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

fn check_surf(source: &str) {
    let parse = parse_surf(source);
    let mut json = String::new();
    match parse {
        Ok(decls) => {
            let deep = desugar_program(&decls);
            let report = check_program(&deep);
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
            let report = check_program(&exprs);
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
