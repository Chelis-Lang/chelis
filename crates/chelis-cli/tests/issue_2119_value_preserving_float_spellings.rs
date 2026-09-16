// chelis#2119: the parser rejected value-preserving float spellings, so
// `chelis fmt --inplace` -- the tool whose job is canonicalization -- failed
// with the same parse error instead of applying the fix the message printed.
//
// The reported cost was concrete: transcribing a reference implementation hits
// this on the first constant (`0.319381530` is the A&S 7.1.26 erf coefficient;
// `0.99999999999980993`, `86.50532032941677` and `771.32342877765313` are the
// Lanczos g=7 gamma coefficients), one literal was reported per invocation, and
// an agent resorted to wrapping `fmt` in a `sed` loop to get past it.
//
// These tests drive the real CLI rather than the library so the reported
// workflow -- write the file, run `fmt --inplace`, then `check` -- is what is
// pinned.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

/// Every constant in the issue's table, with the spelling the compiler used to
/// demand. Bit-identity is asserted below rather than assumed.
const TRANSCRIBED: &str = concat!(
    "erf_a1 = 0.319381530f64\n",
    "lanczos_g0 = 0.99999999999980993f64\n",
    "lanczos_g2 = 86.50532032941677f64\n",
    "lanczos_g3 = 771.32342877765313f64\n",
    "padded = 1.10f64\n",
    "inv_sqrt_two_pi = 0.398942280401432677939946059934f64\n",
);

const CANONICAL: &str = concat!(
    "erf_a1 = 0.31938153f64\n",
    "lanczos_g0 = 0.9999999999998099f64\n",
    "lanczos_g2 = 86.50532032941678f64\n",
    "lanczos_g3 = 771.3234287776531f64\n",
    "padded = 1.1f64\n",
    "inv_sqrt_two_pi = 0.3989422804014327f64\n",
);

fn fmt_inplace(path: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .output()
        .expect("run chelis fmt --inplace")
}

/// The headline defect: one `fmt --inplace` invocation must canonicalize the
/// whole file. Before the fix this exited non-zero with
/// "literal `0.319381530f64` is not an accepted spelling", leaving the file
/// untouched and reporting only the first of six literals.
#[test]
fn fmt_inplace_canonicalizes_every_transcribed_constant_in_one_pass() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("constants.ch");
    fs::write(&path, TRANSCRIBED).expect("write fixture");

    let output = fmt_inplace(&path);
    assert!(
        output.status.success(),
        "fmt --inplace failed: {}",
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        fs::read_to_string(&path).expect("read formatted fixture"),
        CANONICAL,
    );

    // Idempotent on its own output.
    let second = fmt_inplace(&path);
    assert!(second.status.success(), "second fmt pass failed");
    assert_eq!(
        fs::read_to_string(&path).expect("read fixture"),
        CANONICAL,
        "fmt --inplace is not a fixed point on its own output",
    );
}

/// The rewrite the formatter performs is value-preserving, which is the whole
/// justification for accepting the input. `86.50532032941677` being printed
/// back as `...78` is the sharpest case: the final digit differs from the one
/// in the book and the doubles are still bit-identical.
#[test]
fn the_canonical_rewrite_is_bit_identical() {
    for (transcribed, canonical) in [
        ("0.319381530", "0.31938153"),
        ("0.99999999999980993", "0.9999999999998099"),
        ("86.50532032941677", "86.50532032941678"),
        ("771.32342877765313", "771.3234287776531"),
        ("1.10", "1.1"),
        ("0.398942280401432677939946059934", "0.3989422804014327"),
    ] {
        let authored: f64 = transcribed.parse().expect("authored spelling decodes");
        let printed: f64 = canonical.parse().expect("canonical spelling decodes");
        assert_eq!(
            authored.to_bits(),
            printed.to_bits(),
            "{transcribed} and {canonical} are not the same double",
        );
    }
}

/// `chelis fmt --check` must still report the file as needing formatting, so
/// the repository form is unchanged: the parser widened, the canonical output
/// did not.
#[test]
fn fmt_check_still_reports_a_transcribed_constant_as_unformatted() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("constants.ch");
    fs::write(&path, TRANSCRIBED).expect("write fixture");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--check", path.to_str().unwrap()])
        .assert()
        .failure();

    fs::write(&path, CANONICAL).expect("write canonical fixture");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--check", path.to_str().unwrap()])
        .assert()
        .success();
}

/// Negative parity: a spelling that is not value-preserving is still a parse
/// error, and `fmt --inplace` still refuses it rather than guessing. A leading
/// zero on an integer body reads as C octal to a human, so it is refused
/// instead of silently normalized.
#[test]
fn fmt_inplace_still_refuses_a_non_value_preserving_spelling() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("octal_looking.ch");
    let source = "value = 007\n";
    fs::write(&path, source).expect("write fixture");

    let output = fmt_inplace(&path);
    assert!(
        !output.status.success(),
        "a redundant leading zero must not be silently normalized",
    );
    assert_eq!(
        fs::read_to_string(&path).expect("read fixture"),
        source,
        "a failed fmt --inplace must leave the input untouched",
    );
}

/// spec/02-surf-syntax.md §P10a: an integer body under a float suffix is
/// [04-LIT-1]'s exact `literal_source: integer` form, finalized once at the
/// declared width. It parses, and the formatter preserves it rather than
/// substituting a decimal decode.
#[test]
fn a_float_suffixed_integer_body_survives_formatting_unchanged() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("integer_bodied.ch");
    let source = "scale = 8000000f64\ncount = 42f32\n";
    fs::write(&path, source).expect("write fixture");

    let output = fmt_inplace(&path);
    assert!(
        output.status.success(),
        "fmt --inplace failed: {}",
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(
        fs::read_to_string(&path).expect("read fixture"),
        source,
        "the formatter must preserve the integer body",
    );
}
