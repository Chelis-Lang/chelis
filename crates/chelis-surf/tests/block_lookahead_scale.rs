//! BlockBinding lookahead must preserve the parser cursor and scale to generated modules.
use chelis_surf::parser::parse_str;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

#[test]
fn generated_scalar_block_has_bounded_parse_time() {
    let mut source = String::from("def large(x: f64) -> f64 = {\n");
    for index in 0..20_000 {
        writeln!(source, "v{index} = (x + {index}.0f64)").unwrap();
    }
    source.push_str("v19999\n}\n");
    let started = Instant::now();
    let parsed = parse_str(&source).unwrap();
    assert_eq!(parsed.len(), 1);
    // A deliberately generous ceiling. The former full token-vector clone at
    // every binding took tens of seconds; a shared immutable stream takes ms.
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "parse took {:?}",
        started.elapsed()
    );
}

#[test]
fn lookahead_preserves_tuple_typed_binding_and_tail_boundaries() {
    for source in [
        "def f(x: f64) = {\n(a, (b, _)) = (x, (x, x))\nc: f64 = a\n(c, b)\n}",
        "def f(x: f64) = {\na = x\n(a, x)\n}",
    ] {
        assert!(parse_str(source).is_ok());
    }
    for source in [
        "def f(x: f64) = {\n(a, b) = (x, x)\n}\n",
        "def f(x: f64) = {\na: f64 = x\n(a, x)\nx\n}\n",
    ] {
        assert!(parse_str(source).is_err());
    }
}
