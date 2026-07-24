//! The chelis#732 Phase 0 round-trip harness: the mechanical detector for
//! `spec/design/faithful_observation.md` §C2.1.
//!
//! ## The invariant under test
//!
//! For every dtype and every storable value, the text a lane emits at any
//! exit (`print`, `to_list`, diagnostics, wire rendering) must parse back
//! to exactly the stored bits AT THE DTYPE'S OWN WIDTH. Phase 0 asserted
//! the invariant at the VALUE level only; chelis#732 Phase 1 froze the
//! number grammar (§C1.3, ratified as spec/05 §8.1) and adopted it at
//! every EVAL exit - the value-level assertions here survived that §B2.1
//! migration unchanged, which is the migration's proof that only rendering
//! moved. The C lane still renders through its pre-contract paths until
//! Phase 2.
//!
//! Two assertion tiers, chosen so that VALUE bugs (chelis#684, #714, #717 -
//! all [#729] territory per §I1) never redden a cell here:
//!
//! * **tier 1 - intra-lane exit agreement** (§C2.2): every exit of one lane
//!   decodes to the same bits. Used alone where the lanes' STORED value is
//!   known to diverge from the constructed one (eval's f64-backed int64
//!   tensors above 2^53, chelis#684).
//! * **tier 2 - absolute faithfulness**: the decoded bits equal the
//!   constructed value's bits. Used where construction is
//!   storage-strategy-independent (integers within 2^53; float values
//!   exactly representable at the dtype, so chelis#717's missing rounding
//!   has nothing to round; inf/NaN, which survive every width).
//!
//! ## The value tables (FROZEN, append-only from Phase 0's exit)
//!
//! Per `faithful_observation.md` Phase 0, the per-dtype tables below are
//! frozen at this file's landing: rows may be APPENDED (with a PR that says
//! why) but never edited or removed. Labels are stable row ids.
//!
//! ## Known-red cells (`#[ignore]`, §B2.3: red-to-green only by un-ignoring)
//!
//! | cell | issue |
//! |---|---|
//! | C print of f16/bf16 tensors (reads 2-byte buffers as f32) | chelis#716 |
//! | C `to_list` of f16/bf16 tensors (runtime abort) | chelis#716 |
//! | C print of int64 tensors above 2^53 (renders through double) | chelis#723 |
//! | bool tensor `print` (1.0/0.0) vs `to_list` (true/false), C lane | chelis#726 observation half (eval half went green at #732 Phase 1) |
//! | C print format selection (`%.1f` collapses tiny values, `%.16g` starves 17-digit f64) | chelis#748 (§B2.5 discovery) |
//! | C nested-in-list tensor renderer (int64 via f64, 10-element silent truncation) | chelis#749 (§B2.5 discovery) |
//! | eval int64 SCALAR ROOT above 2^53 (rank-0 f64 realization collapses the value before the renderer) | chelis#684 ([#729] value layer; PR #792 red-team F1) |
//!
//! Everything else is green by contract; a new red here is a new
//! faithful-observation bug (file it, per §B2.5).
//!
//! Excluded by ownership (§I1): f16/bf16 SCALARS in the compiled lane never
//! reach the print helper honestly (chelis#714 stores them in int64_t - an
//! ingress/value bug), so their C cells are absent rather than red. The
//! wire's capacity limits (JSON cannot carry NaN/inf as numbers; int64
//! above 2^53 in `Vec<f64>` tensor data) are [#729]/[#686] storage decisions
//! recorded at the schema (§C2.4), so the wire rows here cover the
//! in-capacity set only.
//!
//! Run: `cargo nextest run -p chelis-cli --test observation_roundtrip_harness`
//! Red cells: append `-- --ignored`.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

// ---------------------------------------------------------------------------
// Lane drivers (self-contained on purpose: the sibling matrix files' drivers
// return only the first stdout line, and [#729] Phase 0 edits those shared
// shapes on its own branch; these capture the FULL stdout because the harness
// reads several exits from one run)
// ---------------------------------------------------------------------------

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `chelis eval` a full program; full stdout or stderr.
fn eval_stdout(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Build `program` to C, link, run; full stdout or stage error.
fn c_stdout(program: &str, name: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    if !built.status.success() {
        return Err(String::from_utf8_lossy(&built.stderr).into_owned());
    }
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    if !run.status.success() {
        return Err(format!(
            "binary exited {}: {}",
            run.status,
            String::from_utf8_lossy(&run.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&run.stdout).into_owned())
}

/// The exit program. `print` returns unit in BOTH lanes (verified by
/// execution: print roots render as `name = ()`), so the labeled-root
/// renderer is exercised by value roots (`troot`/`lroot`) while the print
/// calls exercise the transcript renderer. `troot = mk()` is a DEF-CALL
/// value root: the compiled lane used to silently DROP it while eval
/// rendered it (chelis#750), so this harness inlined the body as a
/// workaround. chelis#750 is fixed (the host lane now re-attaches a
/// def-call-valued display root to `emit_main`), so `troot` calls `mk()`
/// directly and the render-count assertions below pin BOTH lanes to
/// exactly two tensor / two list renders — this harness is the regression
/// lock for chelis#750.
fn exits_program(ret: &str, body: &str) -> String {
    format!(
        "module M.Main\n\
         def mk() -> {ret} = {body}\n\
         shown = print(mk())\n\
         listed = print(to_list(mk()))\n\
         troot = mk()\n\
         lroot = to_list(mk())\n"
    )
}

/// Lines rendering the tensor itself: the `print` transcript line plus the
/// `troot = ...` labeled root. Exactly two per lane.
fn tensor_lines(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter(|l| l.contains("tensor(shape="))
        .collect()
}

/// Lines rendering the to_list value: the transcript `[...]` line plus the
/// `lroot = [...]` labeled root. Exactly two per lane.
fn list_lines(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with('[') || l.starts_with("lroot = ["))
        .collect()
}

/// Element texts of a `data=[...]` payload.
fn tensor_elems(line: &str) -> Vec<String> {
    let start = line.find("data=[").expect("data marker") + "data=[".len();
    let end = start + line[start..].find(']').expect("closing bracket");
    bracket_elems(&line[start..end])
}

/// Element texts of a `[...]` list payload (with or without `name = `).
fn list_payload_elems(line: &str) -> Vec<String> {
    let start = line.find('[').expect("open bracket") + 1;
    let end = line.rfind(']').expect("closing bracket");
    bracket_elems(&line[start..end])
}

fn bracket_elems(payload: &str) -> Vec<String> {
    payload
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Bit decoding at the dtype's own width
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Width {
    F64,
    F32,
    F16,
    Bf16,
}

/// Bits of `v` at `w`, widened into u64. NaN normalizes to one canonical
/// pattern per width: exit text spells NaN without a payload, so the
/// round-trip contract for NaN is class-level, not payload-level.
fn value_bits_at(v: f64, w: Width) -> u64 {
    if v.is_nan() {
        return match w {
            Width::F64 => f64::NAN.to_bits(),
            Width::F32 => u64::from(f32::NAN.to_bits()),
            Width::F16 => u64::from(half::f16::NAN.to_bits()),
            Width::Bf16 => u64::from(half::bf16::NAN.to_bits()),
        };
    }
    match w {
        Width::F64 => v.to_bits(),
        // Narrowing here re-rounds from the f64 image. Every table value is
        // exactly representable at its width (or is inf/NaN), so no
        // double-rounding case exists for these rows; a future appended row
        // must keep that property or extend this decoder.
        Width::F32 => u64::from((v as f32).to_bits()),
        Width::F16 => u64::from(half::f16::from_f64(v).to_bits()),
        Width::Bf16 => u64::from(half::bf16::from_f64(v).to_bits()),
    }
}

/// Parse exit text into bits at `w`. Accepts the grammars live today
/// (Rust Debug/Display shapes, C printf `inf`/`nan` spellings); Rust's f64
/// parser covers all of them.
fn text_bits_at(text: &str, w: Width) -> Result<u64, String> {
    let v: f64 = text
        .parse()
        .map_err(|e| format!("`{text}` is not a float: {e}"))?;
    Ok(value_bits_at(v, w))
}

/// Lenient integer decode: today's exits render integer elements either as
/// integers (`750`, exact) or float-shaped (`750.0`, the pre-migration
/// tensor form). The float shape is accepted and decoded through f64, which
/// is precisely what makes chelis#723's above-2^53 lie DETECTABLE: the
/// float-shaped text decodes to a different i64 than the exact to_list text.
fn text_int_lenient(text: &str) -> Result<i64, String> {
    if let Ok(v) = text.parse::<i64>() {
        return Ok(v);
    }
    let f: f64 = text
        .parse()
        .map_err(|e| format!("`{text}` is neither i64 nor float: {e}"))?;
    if f.fract() != 0.0 || !f.is_finite() {
        return Err(format!("`{text}` is not an integral value"));
    }
    Ok(f as i64)
}

// ---------------------------------------------------------------------------
// The frozen value tables. `elem` is the Chelis element expression; `value`
// is the constructed value's exact f64 image. `c_print_safe` marks rows the
// emitted C print helper's CURRENT `%.1f`/`%.16g` selection can round-trip;
// unsafe rows are asserted (red) in the Phase B-filed discovery test below
// and stay out of the green C print assertions until Phase 2 fixes them.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FRow {
    label: &'static str,
    elem: &'static str,
    value: f64,
    c_print_safe: bool,
}

/// f64 boundary rows (§C2.1's set: max/min, first-gap neighbors, subnormal,
/// -0.0, plus the audit's 17-digit and e-notation values).
const F64_ROWS: &[FRow] = &[
    FRow {
        label: "f64-max",
        elem: "cast(1.7976931348623157e308, f64)",
        value: f64::MAX,
        // %.16g emits 16 significant digits; the parse-back lands past the
        // overflow midpoint and reads as inf (discovery row).
        c_print_safe: false,
    },
    FRow {
        label: "f64-min-subnormal",
        elem: "cast(5e-324, f64)",
        value: 5e-324,
        c_print_safe: false, // |x| < 1e-9 hits the %.1f arm and prints 0.0
    },
    FRow {
        label: "f64-min-normal",
        elem: "cast(2.2250738585072014e-308, f64)",
        value: 2.2250738585072014e-308,
        c_print_safe: false, // same %.1f collapse
    },
    FRow {
        label: "f64-neg-zero",
        elem: "cast(-0.0, f64)",
        value: -0.0,
        c_print_safe: true,
    },
    FRow {
        label: "f64-tenth",
        elem: "cast(0.1, f64)",
        value: 0.1,
        c_print_safe: true,
    },
    FRow {
        label: "f64-17-digit",
        elem: "cast(0.30000000000000004, f64)",
        value: 0.30000000000000004,
        c_print_safe: false, // %.16g starves the 17th digit; reads back 0.3
    },
    FRow {
        label: "f64-2p53",
        elem: "cast(9007199254740992.0, f64)",
        value: 9007199254740992.0,
        c_print_safe: true,
    },
    FRow {
        label: "f64-2p53-plus-2",
        elem: "cast(9007199254740994.0, f64)",
        value: 9007199254740994.0,
        c_print_safe: true,
    },
    FRow {
        label: "f64-audit-e19",
        elem: "cast(9.999999980506448e19, f64)",
        value: 9.999999980506448e19,
        c_print_safe: true,
    },
    FRow {
        label: "f64-neg",
        elem: "cast(-1.5, f64)",
        value: -1.5,
        c_print_safe: true,
    },
];

/// f32 rows. Literals are the default float width, so they bind directly.
const F32_ROWS: &[FRow] = &[
    FRow {
        label: "f32-max",
        elem: "3.4028234663852886e38",
        value: 3.4028234663852886e38,
        c_print_safe: true, // the f64 image needs only 16 digits
    },
    FRow {
        label: "f32-min-subnormal",
        elem: "1e-45",
        value: 1.401298464324817e-45,
        c_print_safe: false, // %.1f collapse
    },
    FRow {
        label: "f32-min-normal",
        elem: "1.1754943508222875e-38",
        value: 1.1754943508222875e-38,
        c_print_safe: false, // %.1f collapse
    },
    FRow {
        label: "f32-neg-zero",
        elem: "-0.0",
        value: -0.0,
        c_print_safe: true,
    },
    FRow {
        label: "f32-tenth",
        elem: "0.1",
        value: 0.10000000149011612,
        c_print_safe: true,
    },
    FRow {
        label: "f32-2p24",
        elem: "16777216.0",
        value: 16777216.0,
        c_print_safe: true,
    },
    FRow {
        label: "f32-2p24-plus-2",
        elem: "16777218.0",
        value: 16777218.0,
        c_print_safe: true,
    },
    FRow {
        label: "f32-2049",
        elem: "2049.0",
        value: 2049.0,
        c_print_safe: true,
    },
    FRow {
        label: "f32-neg",
        elem: "-1.5",
        value: -1.5,
        c_print_safe: true,
    },
];

/// f16 rows: every value is EXACTLY representable in f16 (and in the f32
/// literals used to construct it), so eval's missing tensor-lane rounding
/// (chelis#717, a [#729] value bug) has nothing to round and cannot redden
/// these cells. C-lane print/to_list for this table live in the chelis#716
/// ignored tests only.
const F16_ROWS: &[FRow] = &[
    FRow {
        label: "f16-max",
        elem: "65504.0",
        value: 65504.0,
        c_print_safe: true,
    },
    FRow {
        label: "f16-min-normal",
        elem: "0.00006103515625",
        value: 0.00006103515625,
        c_print_safe: true,
    },
    FRow {
        label: "f16-min-subnormal",
        elem: "0.000000059604644775390625",
        value: 5.960464477539063e-8,
        c_print_safe: true,
    },
    FRow {
        label: "f16-neg-zero",
        elem: "-0.0",
        value: -0.0,
        c_print_safe: true,
    },
    FRow {
        label: "f16-frac",
        elem: "0.75",
        value: 0.75,
        c_print_safe: true,
    },
    FRow {
        label: "f16-2048",
        elem: "2048.0",
        value: 2048.0,
        c_print_safe: true,
    },
    FRow {
        label: "f16-2050",
        elem: "2050.0",
        value: 2050.0,
        c_print_safe: true,
    },
    FRow {
        label: "f16-neg",
        elem: "-1.5",
        value: -1.5,
        c_print_safe: true,
    },
];

/// bf16 rows, same exact-representability rule as f16.
const BF16_ROWS: &[FRow] = &[
    FRow {
        label: "bf16-max",
        elem: "3.3895313892515355e38",
        value: 3.3895313892515355e38,
        c_print_safe: true,
    },
    FRow {
        label: "bf16-min-normal",
        elem: "1.1754943508222875e-38",
        value: 1.1754943508222875e-38,
        c_print_safe: true,
    },
    FRow {
        label: "bf16-min-subnormal",
        elem: "9.183549615799121e-41",
        value: 9.183549615799121e-41,
        c_print_safe: true,
    },
    FRow {
        label: "bf16-neg-zero",
        elem: "-0.0",
        value: -0.0,
        c_print_safe: true,
    },
    FRow {
        label: "bf16-frac",
        elem: "0.75",
        value: 0.75,
        c_print_safe: true,
    },
    FRow {
        label: "bf16-256",
        elem: "256.0",
        value: 256.0,
        c_print_safe: true,
    },
    FRow {
        label: "bf16-258",
        elem: "258.0",
        value: 258.0,
        c_print_safe: true,
    },
    FRow {
        label: "bf16-neg",
        elem: "-1.5",
        value: -1.5,
        c_print_safe: true,
    },
];

/// The specials program per float dtype: [inf, NaN, -inf] from IEEE division
/// (no trapping float path in either lane; inf and NaN survive every
/// narrowing width, so chelis#717 cannot distort them either).
fn float_specials_body(dt: &str) -> (String, String) {
    let ret = format!("tensor[3, {dt}]");
    let body = match dt {
        "f32" => "div(to_tensor([1.0, 0.0, -1.0]), to_tensor([0.0, 0.0, 0.0]))".to_string(),
        "f64" => "div(to_tensor([cast(1.0, f64), cast(0.0, f64), cast(-1.0, f64)]), \
                  to_tensor([cast(0.0, f64), cast(0.0, f64), cast(0.0, f64)]))"
            .to_string(),
        narrow => {
            format!("cast(div(to_tensor([1.0, 0.0, -1.0]), to_tensor([0.0, 0.0, 0.0])), {narrow})")
        }
    };
    (ret, body)
}

const SPECIALS: &[(&str, f64)] = &[
    ("inf", f64::INFINITY),
    ("nan", f64::NAN),
    ("-inf", f64::NEG_INFINITY),
];

/// The specials table as `FRow`s (the `elem` column is unused: the values
/// come from `float_specials_body`'s division, not per-element literals).
fn special_rows() -> Vec<FRow> {
    SPECIALS
        .iter()
        .map(|&(label, value)| FRow {
            label,
            elem: "",
            value,
            c_print_safe: true,
        })
        .collect()
}

struct IRow {
    label: &'static str,
    elem: &'static str,
    value: i64,
}

/// Integer rows within +/- 2^53: exact in BOTH storage strategies (i64 and
/// eval's f64-backed tensors), so tier 2 holds in both lanes. MIN endpoints
/// are written as -(MAX) because the grammar's negative literal is neg
/// applied to a positive literal, and i8/-128-style magnitudes are an
/// ingress question ([#729]) this harness does not take a position on.
const INT_ROWS: &[(&str, &[IRow])] = &[
    (
        "int8",
        &[
            IRow {
                label: "i8-max",
                elem: "cast(127, int8)",
                value: 127,
            },
            IRow {
                label: "i8-neg-max",
                elem: "cast(-127, int8)",
                value: -127,
            },
            IRow {
                label: "i8-zero",
                elem: "cast(0, int8)",
                value: 0,
            },
        ],
    ),
    (
        "int16",
        &[
            IRow {
                label: "i16-max",
                elem: "cast(32767, int16)",
                value: 32767,
            },
            IRow {
                label: "i16-neg-max",
                elem: "cast(-32767, int16)",
                value: -32767,
            },
        ],
    ),
    (
        "int32",
        &[
            IRow {
                label: "i32-max",
                elem: "cast(2147483647, int32)",
                value: 2147483647,
            },
            IRow {
                label: "i32-neg-max",
                elem: "cast(-2147483647, int32)",
                value: -2147483647,
            },
        ],
    ),
    (
        "int64",
        &[
            IRow {
                label: "i64-2p53",
                elem: "cast(9007199254740992, int64)",
                value: 9007199254740992,
            },
            IRow {
                label: "i64-neg-2p53",
                elem: "cast(-9007199254740992, int64)",
                value: -9007199254740992,
            },
            IRow {
                label: "i64-small",
                elem: "cast(750, int64)",
                value: 750,
            },
        ],
    ),
];

// ---------------------------------------------------------------------------
// Shared assertions
// ---------------------------------------------------------------------------

/// Tier 2 over a float dtype: all four exit renders decode to the table's
/// bits at `w`. `print_filter` limits which rows the two print lines are
/// held to and `list_filter` the two to_list lines (eval passes `all_rows`
/// for both; the C tests carve out the C-lane exclusions and the
/// print-format discovery cells).
fn assert_float_exits(
    stdout: &str,
    w: Width,
    rows: &[FRow],
    print_filter: fn(&FRow) -> bool,
    list_filter: fn(&FRow) -> bool,
    ctx: &str,
) {
    // Full four-exit programs (`exits_program`) render the print transcript
    // AND the `troot` value root: exactly two tensor renders.
    assert_float_print_exits(stdout, w, rows, print_filter, 2, ctx);
    let llines = list_lines(stdout);
    // chelis#750: exactly two — the `print(to_list(mk()))` transcript line
    // and the `lroot = [...]` labeled root. Both lanes render both now that
    // the def-call root drop is fixed; a count other than two is a
    // regression (a dropped or duplicated root render).
    assert_eq!(
        llines.len(),
        2,
        "[{ctx}] expected the transcript and root to_list renders, got:\n{stdout}"
    );
    for line in llines {
        let elems = list_payload_elems(line);
        assert_eq!(elems.len(), rows.len(), "[{ctx}] element count: {line}");
        for (row, text) in rows.iter().zip(&elems) {
            if !list_filter(row) {
                continue;
            }
            let got = text_bits_at(text, w)
                .unwrap_or_else(|e| panic!("[{ctx}/{}] to_list exit: {e}", row.label));
            assert_eq!(
                got,
                value_bits_at(row.value, w),
                "[{ctx}/{}] to_list exit text `{text}` does not round-trip to the stored bits",
                row.label
            );
        }
    }
}

/// The print half alone, for programs whose to_list exit cannot run yet
/// (the chelis#716 f16/bf16 print cells: to_list of the same tensor aborts,
/// and each red cell must fail on ITS OWN exit).
///
/// `expected_tensor_renders` pins the exact tensor-render count: a full
/// `exits_program` passes 2 (the `print(mk())` transcript plus the
/// `troot = tensor(...)` value root — chelis#750 now emits both in both
/// lanes), while a print-only program (no value root, e.g. the chelis#716
/// f16/bf16 cells) passes 1. An off-by-one here is a dropped or duplicated
/// root render.
fn assert_float_print_exits(
    stdout: &str,
    w: Width,
    rows: &[FRow],
    print_filter: fn(&FRow) -> bool,
    expected_tensor_renders: usize,
    ctx: &str,
) {
    let tlines = tensor_lines(stdout);
    assert_eq!(
        tlines.len(),
        expected_tensor_renders,
        "[{ctx}] expected {expected_tensor_renders} tensor render(s), got:\n{stdout}"
    );
    for line in tlines {
        let elems = tensor_elems(line);
        assert_eq!(elems.len(), rows.len(), "[{ctx}] element count: {line}");
        for (row, text) in rows.iter().zip(&elems) {
            if !print_filter(row) {
                continue;
            }
            let got = text_bits_at(text, w)
                .unwrap_or_else(|e| panic!("[{ctx}/{}] print exit: {e}", row.label));
            assert_eq!(
                got,
                value_bits_at(row.value, w),
                "[{ctx}/{}] print exit text `{text}` does not round-trip to the stored bits",
                row.label
            );
        }
    }
}

/// Tier 2 over an integer dtype (both exits, both renders).
fn assert_int_exits(stdout: &str, rows: &[IRow], ctx: &str) {
    // chelis#750: exactly two per exit — the `print(...)` transcript line
    // and the labeled root (`troot`/`lroot`). The def-call root drop is
    // fixed, so both lanes render both; a count other than two is a
    // regression (a dropped or duplicated root render).
    assert_eq!(
        tensor_lines(stdout).len(),
        2,
        "[{ctx}] expected the transcript and root tensor renders, got:\n{stdout}"
    );
    assert_eq!(
        list_lines(stdout).len(),
        2,
        "[{ctx}] expected the transcript and root to_list renders, got:\n{stdout}"
    );
    for line in tensor_lines(stdout) {
        let elems = tensor_elems(line);
        assert_eq!(elems.len(), rows.len(), "[{ctx}] element count: {line}");
        for (row, text) in rows.iter().zip(&elems) {
            let got = text_int_lenient(text)
                .unwrap_or_else(|e| panic!("[{ctx}/{}] print exit: {e}", row.label));
            assert_eq!(
                got, row.value,
                "[{ctx}/{}] print exit text `{text}`",
                row.label
            );
        }
    }
    for line in list_lines(stdout) {
        let elems = list_payload_elems(line);
        assert_eq!(elems.len(), rows.len(), "[{ctx}] element count: {line}");
        for (row, text) in rows.iter().zip(&elems) {
            let got = text_int_lenient(text)
                .unwrap_or_else(|e| panic!("[{ctx}/{}] to_list exit: {e}", row.label));
            assert_eq!(
                got, row.value,
                "[{ctx}/{}] to_list exit text `{text}`",
                row.label
            );
        }
    }
}

fn all_rows(_r: &FRow) -> bool {
    true
}

/// Rows excluded from the compiled-lane PROGRAMS (not just assertions), by
/// table label - all for one ingress defect, verified by execution:
/// `host_emit.rs`'s Float arm renders constants into the C source with Rust
/// `{}` Display, which has no e-notation, so any integral-valued float
/// constant >= 2^64 becomes a raw C integer literal clang REJECTS
/// ("integer literal is too large"): f64::MAX, 9.999999980506448e19, and
/// f32::MAX produce programs that do not compile. The same Display path
/// spells -0.0 as `-0`, an integer literal whose double conversion drops
/// the sign; probed 2026-07-17: the C scalar and cast-element (f64 tensor)
/// routes LOSE the sign, while the direct f32 literal tensor route
/// preserves it, so only the f64 row stays excluded. All faces are one
/// INGRESS value defect, chelis#751 ([#729] territory per §I1, probe
/// outcomes recorded on [#729]), so the affected cells are unconstructible
/// in the C lane today rather than red rendering cells; labels leave this
/// list when chelis#751 lands.
const C_LANE_EXCLUDED: &[&str] = &["f64-neg-zero", "f64-max", "f64-audit-e19", "f32-max"];

fn c_lane_rows(r: &FRow) -> bool {
    !C_LANE_EXCLUDED.contains(&r.label)
}

/// Rows excluded from the EVAL to_list assertions: the eval tensor lane
/// pins its runtime precision tag at F32 even for checker-typed f64
/// tensors (chelis#717's per-op-chaos family, verified here by execution:
/// print shows the stored f64 while to_list narrows every element through
/// the F32 tag - f64::MAX reads back as `inf`, 0.1 as its f32 image). A
/// VALUE/metadata bug, [#729]'s per §I1, so these are exclusions with a
/// probe comment on [#729], not red rendering cells. Only f32-exact f64
/// values survive the tag; the print exit is asserted on EVERY row.
///
/// `f64-2p53` joined the list at chelis#732 Phase 1: 2^53 IS f32-exact, so
/// the narrowed VALUE survives the tag, but the Phase 1 own-width scalar
/// renderer now prints the to_list element AT ITS (narrowed) F32 width -
/// and the shortest f32 string for 2^53 does not parse back to the same
/// f64. The row's earlier green was rendering-accidental: the pre-contract
/// f64-width renderer masked the chelis#717 narrowing for exactly this
/// value class. Faithful rendering makes the value bug visible instead of
/// laundered (faithful_observation.md, non-goals) - the cell returns when
/// [#729] repairs the to_list value path.
const EVAL_F64_LIST_EXCLUDED: &[&str] = &[
    "f64-max",
    "f64-min-subnormal",
    "f64-min-normal",
    "f64-tenth",
    "f64-17-digit",
    "f64-2p53",
    "f64-2p53-plus-2",
    "f64-audit-e19",
];

fn eval_f64_list_rows(r: &FRow) -> bool {
    !EVAL_F64_LIST_EXCLUDED.contains(&r.label)
}

/// The compiled-lane subset of a table: the rows whose constants the C
/// backend can render into compilable source (see `C_LANE_EXCLUDED`).
fn c_rows(rows: &[FRow]) -> Vec<FRow> {
    rows.iter().filter(|r| c_lane_rows(r)).cloned().collect()
}

fn c_print_safe_rows(r: &FRow) -> bool {
    c_lane_rows(r) && r.c_print_safe
}

fn c_print_red_rows(r: &FRow) -> bool {
    c_lane_rows(r) && !r.c_print_safe
}

fn float_table_program(dt: &str, rows: &[FRow], via_cast: bool) -> String {
    let elems: Vec<&str> = rows.iter().map(|r| r.elem).collect();
    let n = rows.len();
    let body = if via_cast {
        format!("cast(to_tensor([{}]), {dt})", elems.join(", "))
    } else {
        format!("to_tensor([{}])", elems.join(", "))
    };
    exits_program(&format!("tensor[{n}, {dt}]"), &body)
}

fn int_table_program(dt: &str, rows: &[IRow]) -> String {
    let elems: Vec<&str> = rows.iter().map(|r| r.elem).collect();
    let n = rows.len();
    exits_program(
        &format!("tensor[{n}, {dt}]"),
        &format!("to_tensor([{}])", elems.join(", ")),
    )
}

// ===========================================================================
// GREEN - eval lane, tensor + to_list exits (transcript and root renders)
// ===========================================================================

#[test]
fn eval_f64_tensor_exits_round_trip() {
    let out = eval_stdout(&float_table_program("f64", F64_ROWS, false)).expect("eval");
    assert_float_exits(
        &out,
        Width::F64,
        F64_ROWS,
        all_rows,
        eval_f64_list_rows,
        "eval/f64",
    );

    let (ret, body) = float_specials_body("f64");
    let out = eval_stdout(&exits_program(&ret, &body)).expect("eval specials");
    assert_float_exits(
        &out,
        Width::F64,
        &special_rows(),
        all_rows,
        all_rows,
        "eval/f64-specials",
    );
}

#[test]
fn eval_f32_tensor_exits_round_trip() {
    let out = eval_stdout(&float_table_program("f32", F32_ROWS, false)).expect("eval");
    assert_float_exits(&out, Width::F32, F32_ROWS, all_rows, all_rows, "eval/f32");

    let (ret, body) = float_specials_body("f32");
    let out = eval_stdout(&exits_program(&ret, &body)).expect("eval specials");
    assert_float_exits(
        &out,
        Width::F32,
        &special_rows(),
        all_rows,
        all_rows,
        "eval/f32-specials",
    );
}

#[test]
fn eval_f16_bf16_tensor_exits_round_trip() {
    for (dt, w, rows) in [
        ("f16", Width::F16, F16_ROWS),
        ("bf16", Width::Bf16, BF16_ROWS),
    ] {
        let out = eval_stdout(&float_table_program(dt, rows, true)).expect("eval");
        assert_float_exits(&out, w, rows, all_rows, all_rows, &format!("eval/{dt}"));

        let (ret, body) = float_specials_body(dt);
        let out = eval_stdout(&exits_program(&ret, &body)).expect("eval specials");
        assert_float_exits(
            &out,
            w,
            &special_rows(),
            all_rows,
            all_rows,
            &format!("eval/{dt}-specials"),
        );
    }
}

#[test]
fn eval_int_tensor_exits_round_trip() {
    for (dt, rows) in INT_ROWS {
        let out = eval_stdout(&int_table_program(dt, rows)).expect("eval");
        assert_int_exits(&out, rows, &format!("eval/{dt}"));
    }
}

/// int64 ABOVE 2^53: tier 1 only. Pre-chelis#729 the f64-backed storage
/// collapsed 2^53+1 before ANY exit rendered it, so this row was green by
/// uniform wrongness (every exit agreed on the collapsed value). Since
/// chelis#729 Phase 1 the host-lane exits (print/to_list and their
/// labeled roots) are EXACT, and the one remaining collapse point is the
/// tensor-lane labeled root, whose binding lowers through the f64 DAG
/// literal payload (chelis#856) - so the intra-lane agreement this row
/// asserts is genuinely violated, three exits right and one wrong, until
/// chelis#856 lands. The C lane's tier-2 version of this row is the
/// chelis#723 ignored test below.
#[test]
#[ignore = "chelis#856: the tensor-lane labeled root renders the DAG-literal-collapsed value \
            (9007199254740992) while print/to_list and the host-lane root are exact after \
            chelis#729 Phase 1; the intra-lane agreement returns when the DAG literal payload \
            carries exact integers. Run with \
            `cargo test -p chelis-cli --test observation_roundtrip_harness -- --ignored`."]
fn eval_int64_above_2p53_exits_agree_within_lane() {
    let program = exits_program(
        "tensor[1, int64]",
        "to_tensor([cast(9007199254740993, int64)])",
    );
    let out = eval_stdout(&program).expect("eval");
    let mut decoded: Vec<i64> = Vec::new();
    for line in tensor_lines(&out) {
        decoded.push(text_int_lenient(&tensor_elems(line)[0]).expect("print exit"));
    }
    for line in list_lines(&out) {
        decoded.push(text_int_lenient(&list_payload_elems(line)[0]).expect("to_list exit"));
    }
    assert_eq!(decoded.len(), 4, "four exit renders expected:\n{out}");
    assert!(
        decoded.windows(2).all(|w| w[0] == w[1]),
        "eval exits disagree on one stored int64 tensor: {decoded:?}\n{out}"
    );
}

// ===========================================================================
// GREEN - eval lane, scalar exits (transcript and root renders)
// ===========================================================================

/// Scalar exit rows: (expression, dtype width or None for integer, exact
/// f64 image / i64 value). Scalars take a different path from tensors in
/// BOTH lanes (render_value scalar arms in eval; emit_print_value in C),
/// so they get their own rows.
#[test]
fn eval_scalar_exits_round_trip() {
    let float_rows: &[(&str, Width, f64)] = &[
        (
            "cast(0.30000000000000004, f64)",
            Width::F64,
            0.30000000000000004,
        ),
        ("cast(1.7976931348623157e308, f64)", Width::F64, f64::MAX),
        ("cast(5e-324, f64)", Width::F64, 5e-324),
        ("cast(-0.0, f64)", Width::F64, -0.0),
        ("cast(0.1, f64)", Width::F64, 0.1),
        ("0.1", Width::F32, 0.10000000149011612),
        ("3.4028234663852886e38", Width::F32, 3.4028234663852886e38),
        ("-0.0", Width::F32, -0.0),
        ("cast(0.75, f16)", Width::F16, 0.75),
        ("cast(2048.0, f16)", Width::F16, 2048.0),
        ("cast(0.75, bf16)", Width::Bf16, 0.75),
        (
            "div(cast(1.0, f64), cast(0.0, f64))",
            Width::F64,
            f64::INFINITY,
        ),
        ("div(cast(0.0, f64), cast(0.0, f64))", Width::F64, f64::NAN),
    ];
    for (expr, w, value) in float_rows {
        let out = eval_stdout(&format!("module M.Main\nshown = print({expr})\n")).expect("eval");
        for line in scalar_render_lines(&out) {
            let got = text_bits_at(&line, *w)
                .unwrap_or_else(|e| panic!("[eval scalar {expr}] {e}\n{out}"));
            assert_eq!(
                got,
                value_bits_at(*value, *w),
                "[eval scalar {expr}] text `{line}` does not round-trip"
            );
        }
    }
    let int_rows: &[(&str, i64)] = &[
        ("cast(9223372036854775807, int64)", i64::MAX),
        ("cast(-9223372036854775807, int64)", -i64::MAX),
        ("cast(9007199254740993, int64)", 9007199254740993),
        ("cast(2147483647, int32)", 2147483647),
    ];
    for (expr, value) in int_rows {
        let out = eval_stdout(&format!("module M.Main\nshown = print({expr})\n")).expect("eval");
        for line in scalar_render_lines(&out) {
            let got =
                text_int_lenient(&line).unwrap_or_else(|e| panic!("[eval scalar {expr}] {e}"));
            assert_eq!(got, *value, "[eval scalar {expr}] text `{line}`");
        }
    }
    let out = eval_stdout("module M.Main\nshown = print(true)\n").expect("eval");
    for line in scalar_render_lines(&out) {
        assert_eq!(line, "true", "bool scalar exit");
    }
}

/// The scalar print-transcript render of a single `shown = print(expr)`
/// program. The print root itself renders as `()` (print returns unit) and
/// is dropped. Scalar VALUE roots - deliberately undriven at Phase 0 while
/// the canonical scalar-root rendering was undecided (chelis#775) - are
/// driven since chelis#732 Phase 1 by `eval_scalar_value_roots_render_bare`
/// below: [05-OBS-4] makes the bare scalar canonical at every exit, so the
/// eval-internal rank-0 realization no longer leaks into root renders.
fn scalar_render_lines(stdout: &str) -> Vec<String> {
    let lines: Vec<String> = stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.ends_with("()"))
        .map(str::to_string)
        .collect();
    assert_eq!(
        lines.len(),
        1,
        "expected exactly the scalar print transcript, got:\n{stdout}"
    );
    lines
}

/// RED (chelis#684, [#729] value layer; surfaced by PR #792's red team,
/// F1): an int64 SCALAR ROOT above 2^53 loses exactness at the labeled
/// root while print and to_string of the same def render it exactly -
/// the interpreter's rank-0 f64 realization collapses the value BEFORE
/// the renderer sees it, so this is a stored-value defect upstream of the
/// [05-OBS] rendering contract, not a formatter bug. The exception is
/// annexed in spec/05 §8's [05-OBS-1]/[05-OBS-4] status text; the cell
/// goes green (by un-ignoring, §B2.3) when [#729] repairs scalar-root
/// storage. Rendering must NOT paper over it: the root faithfully shows
/// the collapsed stored value.
#[test]
#[ignore = "chelis#684 ([#729] value layer): the rank-0 f64 realization collapses int64 \
            scalar roots above 2^53 before the renderer sees them; print/to_string are \
            exact, the labeled root is not. Un-ignore when [#729] repairs scalar-root \
            storage. Run with \
            `cargo test -p chelis-cli --test observation_roundtrip_harness -- --ignored`."]
fn eval_int64_scalar_root_above_2p53_renders_exact() {
    let program = "module M.Main\n\
         def run() -> int64 = cast(9007199254740993, int64)\n\
         shown = print(run())\n\
         sroot = run()\n";
    let out = eval_stdout(program).expect("eval");
    // Green half (control): the print transcript is exact at own width.
    let transcript = out
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.ends_with("()") && !l.starts_with("sroot = "))
        .unwrap_or_else(|| panic!("no print transcript in:\n{out}"));
    assert_eq!(transcript, "9007199254740993", "print exit must stay exact");
    // Red half: the labeled root must carry the same exact value. Today it
    // renders the f64-collapsed 9007199254740992 (chelis#684).
    let root = out
        .lines()
        .find(|l| l.starts_with("sroot = "))
        .unwrap_or_else(|| panic!("no sroot line in:\n{out}"));
    assert_eq!(
        root, "sroot = 9007199254740993",
        "the labeled root must carry the exact stored int64; [05-OBS-1] \
         intra-lane exit agreement is broken by the rank-0 realization"
    );
}

/// [05-OBS-4] (the chelis#775 decision, eval half): a scalar-typed
/// top-level root renders as the BARE scalar - byte-identical to the print
/// transcript of the same value (intra-lane exit agreement, §C2.1/2) - and
/// the rank-0 `tensor(shape=[], data=[..])` wrapper appears at no exit.
/// Values are chosen exactly representable at their dtype so the [#684]
/// storage collapse (a value bug, not a rendering one - its above-2^53
/// int64 face is the ignored red cell directly above) cannot blur the
/// row; the C lane's conformance for the same repro is locked by
/// `c_def_call_scalar_root_emitted_issue_750` below.
#[test]
fn eval_scalar_value_roots_render_bare() {
    let rows: &[(&str, &str, &str)] = &[
        ("f64", "cast(0.1, f64)", "0.1"),
        ("f32", "0.5", "0.5"),
        ("int64", "cast(750, int64)", "750"),
        ("int32", "7", "7"),
        ("bool", "and(true, true)", "true"),
    ];
    for (ret, expr, expected) in rows {
        let program = format!(
            "module M.Main\n\
             def run() -> {ret} = {expr}\n\
             shown = print(run())\n\
             root = run()\n"
        );
        let out = eval_stdout(&program).expect("eval");
        assert!(
            !out.contains("tensor(shape="),
            "[eval scalar root {expr}] the rank-0 tensor wrapper is not an \
             exit form ([05-OBS-4]), got:\n{out}"
        );
        let root_line = out
            .lines()
            .find(|l| l.starts_with("root = "))
            .unwrap_or_else(|| panic!("[eval scalar root {expr}] no root line in:\n{out}"));
        assert_eq!(
            root_line,
            format!("root = {expected}"),
            "[eval scalar root {expr}] bare scalar root render"
        );
        // Intra-lane exit agreement: the print transcript of the same
        // value is byte-identical to the root's payload.
        let transcript = out
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty() && !l.ends_with("()") && !l.starts_with("root = "))
            .unwrap_or_else(|| panic!("[eval scalar root {expr}] no transcript in:\n{out}"));
        assert_eq!(
            transcript, *expected,
            "[eval scalar root {expr}] print transcript and root payload \
             must agree within the lane"
        );
    }
}

// ===========================================================================
// GREEN - chelis#732 Phase 1: the frozen eval grammar (§C1.3 / spec/05 §8.1)
//
// Exact-string locks, deliberately scoped to surfaces the grammar freeze
// makes permanent: integer/bool tensor elements, f64 digits, own-width
// scalars, and the truncation form. Narrow-float TENSOR element digits are
// NOT locked here: eval renders those at the stored f64 width until [#729]
// repairs the chelis#717 precision metadata, and locking the interim
// digits would turn that value-layer fix into a formatting break.
// ===========================================================================

#[test]
fn eval_exit_grammar_locks() {
    let cases: &[(&str, &str)] = &[
        // Integers print as integers in tensor data ([05-OBS-2]).
        (
            "print(to_tensor([cast(127, int8), cast(-127, int8), cast(0, int8)]))",
            "tensor(shape=[3], data=[127, -127, 0])",
        ),
        (
            "print(to_tensor([cast(9007199254740992, int64)]))",
            "tensor(shape=[1], data=[9007199254740992])",
        ),
        // bool tensor data prints true/false (chelis#726's eval half).
        (
            "print(to_tensor([true, false]))",
            "tensor(shape=[2], data=[true, false])",
        ),
        // f64 scalars: shortest round-trip digits, Debug grammar.
        (
            "print(cast(0.30000000000000004, f64))",
            "0.30000000000000004",
        ),
        (
            "print(cast(9.999999980506448e19, f64))",
            "9.999999980506448e19",
        ),
        ("print(cast(-0.0, f64))", "-0.0"),
        ("print(div(cast(1.0, f64), cast(0.0, f64)))", "inf"),
        ("print(div(cast(-1.0, f64), cast(0.0, f64)))", "-inf"),
        ("print(div(cast(0.0, f64), cast(0.0, f64)))", "NaN"),
        // f32 scalar at OWN width: the f64-image digits are gone.
        ("print(0.1)", "0.1"),
        // f16 scalar: integral decimal keeps one fractional digit.
        ("print(cast(2048.0, f16))", "2048.0"),
    ];
    for (expr, expected) in cases {
        let out = eval_stdout(&format!("module M.Main\nshown = {expr}\n")).expect("eval");
        let line = out
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty() && !l.ends_with("()"))
            .unwrap_or_else(|| panic!("[grammar {expr}] no render in:\n{out}"));
        assert_eq!(line, *expected, "[grammar {expr}]");
    }
}

/// [05-OBS-5]: every eval exit truncates tensor element rendering after 32
/// elements with the `, ...` marker (the compiled lane's existing form).
/// Locks BOTH eval renders: the transcript (unlimited before chelis#732
/// Phase 1 - the one place the §B2.1 migration changed how MUCH is
/// printed) and the labeled root (whose old marker was `+ ...`). The
/// elements are integers 1..=33 held at the bit level by construction, so
/// the truncation change provably altered rendering only (§B2.2).
#[test]
fn eval_tensor_renders_truncate_at_32_with_marker() {
    let elems: Vec<String> = (1..=33).map(|i| format!("{i}.0")).collect();
    let program = format!(
        "module M.Main\n\
         def mk() -> tensor[33, f32] = to_tensor([{}])\n\
         shown = print(mk())\n\
         troot = mk()\n",
        elems.join(", ")
    );
    let out = eval_stdout(&program).expect("eval");
    let visible: Vec<String> = (1..=32).map(|i| format!("{i}.0")).collect();
    let expected = format!("tensor(shape=[33], data=[{}, ...])", visible.join(", "));
    let tlines = tensor_lines(&out);
    assert_eq!(
        tlines.len(),
        2,
        "expected the transcript and root renders:\n{out}"
    );
    assert_eq!(tlines[0], expected, "transcript truncation form");
    assert_eq!(
        tlines[1],
        format!("troot = {expected}"),
        "labeled-root truncation form"
    );
    // Bit-level companion: the full 33 elements stay reachable through
    // to_list (full-element fidelity is to_list's job, [05-OBS-5]).
    let list_program = format!(
        "module M.Main\n\
         def mk() -> tensor[33, f32] = to_tensor([{}])\n\
         lroot = to_list(mk())\n",
        elems.join(", ")
    );
    let out = eval_stdout(&list_program).expect("eval");
    let lline = list_lines(&out)
        .first()
        .copied()
        .unwrap_or_else(|| panic!("no to_list render in:\n{out}"))
        .to_string();
    let listed = list_payload_elems(&lline);
    assert_eq!(
        listed.len(),
        33,
        "to_list must carry every element: {lline}"
    );
    for (i, text) in listed.iter().enumerate() {
        let got =
            text_bits_at(text, Width::F32).unwrap_or_else(|e| panic!("to_list element {i}: {e}"));
        assert_eq!(
            got,
            value_bits_at((i + 1) as f64, Width::F32),
            "to_list element {i} drifted: {text}"
        );
    }
}

// ===========================================================================
// GREEN - compiled C lane
// ===========================================================================

#[test]
fn c_f64_tensor_exits_round_trip() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    // to_list is asserted on EVERY row (the runtime's f64 list render is
    // Display, which is shortest-round-trip); print only on the rows the
    // current %.1f/%.16g selection can carry - the rest are the discovery
    // test's red cells.
    let rows = c_rows(F64_ROWS);
    let out = c_stdout(&float_table_program("f64", &rows, false), "obs_f64").expect("C lane");
    assert_float_exits(
        &out,
        Width::F64,
        &rows,
        c_print_safe_rows,
        all_rows,
        "c/f64",
    );

    let (ret, body) = float_specials_body("f64");
    let out = c_stdout(&exits_program(&ret, &body), "obs_f64_sp").expect("C specials");
    assert_float_exits(
        &out,
        Width::F64,
        &special_rows(),
        all_rows,
        all_rows,
        "c/f64-specials",
    );
}

#[test]
fn c_f32_tensor_exits_round_trip() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let rows = c_rows(F32_ROWS);
    let out = c_stdout(&float_table_program("f32", &rows, false), "obs_f32").expect("C lane");
    assert_float_exits(
        &out,
        Width::F32,
        &rows,
        c_print_safe_rows,
        all_rows,
        "c/f32",
    );

    let (ret, body) = float_specials_body("f32");
    let out = c_stdout(&exits_program(&ret, &body), "obs_f32_sp").expect("C specials");
    assert_float_exits(
        &out,
        Width::F32,
        &special_rows(),
        all_rows,
        all_rows,
        "c/f32-specials",
    );
}

#[test]
fn c_int_tensor_exits_round_trip() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    for (dt, rows) in INT_ROWS {
        let out = c_stdout(&int_table_program(dt, rows), &format!("obs_{dt}")).expect("C lane");
        assert_int_exits(&out, rows, &format!("c/{dt}"));
    }
}

/// The C lane's int64 to_list exit is EXACT above 2^53 (the audit's proof
/// instrument for chelis#723; mirrors the sum-based lock in
/// reduction_and_bitwise_matrix.rs without moving it). Print of the same
/// tensor is the chelis#723 ignored test below.
#[test]
fn c_int64_to_list_is_exact_above_2p53() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "module M.Main\n\
         def mk() -> tensor[2, int64] = to_tensor([cast(9007199254740993, int64), \
         cast(9223372036854775807, int64)])\n\
         shown = print(to_list(mk()))\n\
         lroot = to_list(mk())\n";
    let out = c_stdout(program, "obs_i64_list").expect("C lane");
    for line in list_lines(&out) {
        let elems = list_payload_elems(line);
        assert_eq!(
            text_int_lenient(&elems[0]).expect("elem 0"),
            9007199254740993,
            "to_list must carry 2^53+1 exactly: {line}"
        );
        assert_eq!(
            text_int_lenient(&elems[1]).expect("elem 1"),
            i64::MAX,
            "to_list must carry i64::MAX exactly: {line}"
        );
    }
}

#[test]
fn c_scalar_exits_round_trip() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    // f16/bf16 scalar rows are deliberately ABSENT: the compiled lane stores
    // them in int64_t (chelis#714, ingress/value - [#729]'s side of §I1), so
    // there is no faithfully-stored value for this harness to read yet.
    let float_rows: &[(&str, &str, Width, f64)] = &[
        ("f64", "cast(0.1, f64)", Width::F64, 0.1),
        (
            "f64",
            "cast(9007199254740992.0, f64)",
            Width::F64,
            9007199254740992.0,
        ),
        ("f64", "cast(5e-324, f64)", Width::F64, 5e-324),
        (
            "f64",
            "div(cast(1.0, f64), cast(0.0, f64))",
            Width::F64,
            f64::INFINITY,
        ),
        (
            "f64",
            "div(cast(0.0, f64), cast(0.0, f64))",
            Width::F64,
            f64::NAN,
        ),
        ("f32", "0.1", Width::F32, 0.10000000149011612),
        // f32::MAX is absent: the emitted constant is an integer literal
        // clang rejects (the C_LANE_EXCLUDED ingress defect, [#729]).
        ("f32", "1e-45", Width::F32, 1.401298464324817e-45),
    ];
    for (i, (ret, expr, w, value)) in float_rows.iter().enumerate() {
        let program = format!("module M.Main\ndef run() -> {ret} = {expr}\nshown = print(run())\n");
        let out = c_stdout(&program, &format!("obs_sc_f{i}")).expect("C lane");
        for line in scalar_render_lines(&out) {
            let got = text_bits_at(&line, *w).unwrap_or_else(|e| panic!("[c scalar {expr}] {e}"));
            assert_eq!(
                got,
                value_bits_at(*value, *w),
                "[c scalar {expr}] text `{line}` does not round-trip"
            );
        }
    }
    let int_rows: &[(&str, i64)] = &[
        ("cast(9223372036854775807, int64)", i64::MAX),
        ("cast(9007199254740993, int64)", 9007199254740993),
    ];
    for (i, (expr, value)) in int_rows.iter().enumerate() {
        let program = format!("module M.Main\ndef run() -> int64 = {expr}\nshown = print(run())\n");
        let out = c_stdout(&program, &format!("obs_sc_i{i}")).expect("C lane");
        for line in scalar_render_lines(&out) {
            assert_eq!(
                text_int_lenient(&line).expect("int scalar"),
                *value,
                "[c scalar {expr}] text `{line}`"
            );
        }
    }
    let out = c_stdout(
        "module M.Main\ndef run() -> bool = and(true, true)\nshown = print(run())\n",
        "obs_sc_b",
    )
    .expect("C lane");
    for line in scalar_render_lines(&out) {
        assert_eq!(line, "true", "bool scalar exit");
    }
}

// ===========================================================================
// GREEN - chelis#750: def-call-valued top-level display roots
//
// The host lane emits labeled roots ONLY from `host.globals`, and a
// DAG-lowered non-`fn` value binding (`troot = mk()`) used to be dropped
// from `host.globals` while eval still rendered it. These pin the fix: the
// compiled lane now re-attaches such a root to `emit_main`, byte-identical
// to a direct-construction root of the same value.
// ===========================================================================

/// chelis#750 (tensor): the issue's exact int8 repro. The def-call value
/// root `troot = mk()` was silently dropped by the compiled lane; it is
/// now emitted with the exact stored values (127 / -127 / 0). Each lane's
/// exact line is pinned: eval renders integers as integers per [05-OBS-2]
/// (chelis#732 Phase 1's §B2.1 migration moved this expectation), while
/// the compiled lane still prints the pre-contract `127.0` form until its
/// Phase 2 migration - so this lock asserts VALUE-level cross-lane
/// agreement plus line-count parity, and the byte-identical assertion
/// returns with the Phase 2 generated printer (§C2.3).
#[test]
fn c_def_call_tensor_root_matches_eval_issue_750() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "module M.Main\n\
         def mk() -> tensor[3, int8] = \
         to_tensor([cast(127, int8), cast(-127, int8), cast(0, int8)])\n\
         shown = print(mk())\n\
         troot = mk()\n";
    let eval_out = eval_stdout(program).expect("eval");
    let c_out = c_stdout(program, "issue750_tensor").expect("C lane");

    let expected_c = "troot = tensor(shape=[3], data=[127.0, -127.0, 0.0])";
    assert!(
        c_out.lines().any(|l| l == expected_c),
        "chelis#750: the compiled lane must emit the def-call tensor root \
         `{expected_c}`, got:\n{c_out}"
    );
    let expected_eval = "troot = tensor(shape=[3], data=[127, -127, 0])";
    assert!(
        eval_out.lines().any(|l| l == expected_eval),
        "chelis#750/[05-OBS-2]: eval must emit the def-call tensor root \
         `{expected_eval}`, got:\n{eval_out}"
    );
    // Cross-lane agreement at the value level (byte equality is Phase 2's
    // exit): both troot renders decode to the same stored integers.
    let eval_troot = eval_out
        .lines()
        .find(|l| l.starts_with("troot = "))
        .expect("eval troot line");
    let c_troot = c_out
        .lines()
        .find(|l| l.starts_with("troot = "))
        .expect("c troot line");
    let decode = |line: &str| -> Vec<i64> {
        tensor_elems(line)
            .iter()
            .map(|text| text_int_lenient(text).expect("troot element"))
            .collect()
    };
    assert_eq!(
        decode(eval_troot),
        decode(c_troot),
        "chelis#750: the troot values diverge between lanes:\n eval: {eval_troot}\n c:    {c_troot}"
    );
    assert_eq!(
        eval_out.lines().count(),
        c_out.lines().count(),
        "chelis#750: line-count parity broken:\n--- eval ---\n{eval_out}\n--- c ---\n{c_out}"
    );
}

/// chelis#750 (scalar) + chelis#775 (eval half): the issues' exact f64
/// repro. The def-call value root `root = run()` was silently dropped by
/// the compiled lane (chelis#750, fixed by PR #774); eval then rendered its
/// rank-0 realization (`root = tensor(shape=[], data=[0.1])`) while the C
/// lane printed the bare scalar - the chelis#775 shape divergence. The
/// [05-OBS-4] decision (chelis#732 Phase 1) makes the bare scalar
/// canonical: a scalar-typed root renders exactly as `print` of the same
/// value would, and the rank-0 wrapper is not an exit form. This is the
/// cross-lane exact lock chelis#775 asked for: both lanes emit
/// `root = 0.1`, byte-identical, with full line-count parity.
#[test]
fn c_def_call_scalar_root_emitted_issue_750() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "module M.Main\n\
         def run() -> f64 = cast(0.1, f64)\n\
         shown = print(run())\n\
         root = run()\n";
    let eval_out = eval_stdout(program).expect("eval");
    let c_out = c_stdout(program, "issue750_scalar").expect("C lane");

    // No longer dropped: the compiled lane emits the scalar root, matching a
    // direct-construction scalar root of the same value.
    assert!(
        c_out.lines().any(|l| l == "root = 0.1"),
        "chelis#750: the compiled lane must emit the def-call scalar root \
         `root = 0.1`, got:\n{c_out}"
    );
    // [05-OBS-4]: eval renders the same scalar root bare - the rank-0
    // realization must not leak into the observation channel (chelis#775).
    assert!(
        eval_out.lines().any(|l| l == "root = 0.1"),
        "chelis#775/[05-OBS-4]: eval must render the scalar root bare as \
         `root = 0.1`, got:\n{eval_out}"
    );
    // The root render is byte-identical across lanes.
    let eval_root = eval_out
        .lines()
        .find(|l| l.starts_with("root = "))
        .expect("eval root line");
    let c_root = c_out
        .lines()
        .find(|l| l.starts_with("root = "))
        .expect("c root line");
    assert_eq!(
        eval_root, c_root,
        "chelis#775: the scalar root render diverges between lanes:\n eval: {eval_root}\n c:    {c_root}"
    );
    // The lanes still agree on line count (no silent drop).
    assert_eq!(
        eval_out.lines().count(),
        c_out.lines().count(),
        "chelis#750: line-count parity broken:\n--- eval ---\n{eval_out}\n--- c ---\n{c_out}"
    );
}

/// chelis#750 (negative): a `fn`-typed top-level binding is NOT a display
/// root — the CLI display-name pass maps `HostType::Fn` to `None`, so
/// neither lane emits a labeled root for it, even though the #750 rescue now
/// keeps non-`fn` value roots. The value roots beside it are still emitted,
/// pinning the rescue to VALUE roots only (a spurious `myfn = <ptr>` root
/// would be the regression).
#[test]
fn fn_typed_top_level_binding_emits_no_root_issue_750() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "module M.Main\n\
         def add1(x: f64) -> f64 = add(x, cast(1.0, f64))\n\
         shown = print(add1(cast(2.0, f64)))\n\
         troot = to_tensor([cast(1, int8), cast(2, int8)])\n\
         myfn = add1\n";
    let eval_out = eval_stdout(program).expect("eval");
    let c_out = c_stdout(program, "issue750_fnroot").expect("C lane");

    for (lane, out) in [("eval", &eval_out), ("c", &c_out)] {
        assert!(
            !out.lines().any(|l| l.starts_with("myfn =")),
            "[{lane}] chelis#750: a fn-typed binding must not emit a labeled root, got:\n{out}"
        );
        assert!(
            out.lines().any(|l| l.starts_with("troot = tensor(")),
            "[{lane}] chelis#750: the value root beside the fn binding must still \
             be emitted, got:\n{out}"
        );
    }
}

/// chelis#750 (predicate rescue): a def-call root ALONGSIDE a
/// direct-construction root, with NO `print`. The direct-construction root
/// (`rb = to_tensor([...])`) forces a host `main` on its own (it is never
/// DAG-lowered), so `program_emits_host_main` is true and the def-call root
/// (`ra = mk()`) must be rescued too. Before chelis#750 the compiled lane
/// emitted ONLY the direct root and silently dropped the def-call root; both
/// must now render, byte-identical to eval, with full line-count parity.
/// This is the mixed-root face of the fix (the pure def-call-only,
/// no-`print` program stays on the kernel lane and is out of scope).
#[test]
fn c_defcall_root_rescued_beside_direct_root_issue_750() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "module M.Main\n\
         def mk() -> tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])\n\
         ra = mk()\n\
         rb = to_tensor([9.0, 8.0])\n";
    let eval_out = eval_stdout(program).expect("eval");
    let c_out = c_stdout(program, "issue750_mixed").expect("C lane");

    for root in [
        "ra = tensor(shape=[3], data=[1.0, 2.0, 3.0])",
        "rb = tensor(shape=[2], data=[9.0, 8.0])",
    ] {
        assert!(
            c_out.lines().any(|l| l == root),
            "chelis#750: the compiled lane must emit `{root}`, got:\n{c_out}"
        );
        assert!(
            eval_out.lines().any(|l| l == root),
            "chelis#750: eval must emit `{root}`, got:\n{eval_out}"
        );
    }
    assert_eq!(
        eval_out.lines().count(),
        c_out.lines().count(),
        "chelis#750: line-count parity broken:\n--- eval ---\n{eval_out}\n--- c ---\n{c_out}"
    );
}

// ===========================================================================
// GREEN - wire rendering (§C2.4, in-capacity set)
// ===========================================================================

/// The wire exit: `ExecutionValue`/`TensorValue` serialize numeric payloads
/// through serde_json. Finite f64 and full-range i64 must round-trip
/// bit-exactly. Out-of-capacity cells (NaN/inf have no JSON number form;
/// int64 above 2^53 cannot ride `TensorValue`'s `Vec<f64>`) are [#729]/[#686]
/// storage decisions recorded at the schema (§C2.4) - deliberately no
/// assertion pins them here.
#[test]
fn wire_execution_value_rendering_round_trips() {
    use chelis_compiler_api::schema::{ExecutionValue, TensorValue};

    let finite: Vec<f64> = F64_ROWS.iter().map(|r| r.value).collect();
    for &v in &finite {
        let json = serde_json::to_string(&ExecutionValue::Float64 { value: v }).expect("serialize");
        let back: ExecutionValue = serde_json::from_str(&json).expect("parse");
        match back {
            ExecutionValue::Float64 { value } => assert_eq!(
                value.to_bits(),
                v.to_bits(),
                "wire f64 {v:?} did not round-trip through `{json}`"
            ),
            other => panic!("wire round-trip changed the variant: {other:?}"),
        }
    }

    let tensor = ExecutionValue::Tensor {
        value: TensorValue {
            shape: vec![finite.len()],
            data: chelis_compiler_api::schema::TensorElements::from_f64_vec(finite.clone()),
        },
    };
    let json = serde_json::to_string(&tensor).expect("serialize");
    let back: ExecutionValue = serde_json::from_str(&json).expect("parse");
    match back {
        ExecutionValue::Tensor { value } => {
            for (a, b) in value.data.to_f64_lossy_vec().iter().zip(&finite) {
                assert_eq!(a.to_bits(), b.to_bits(), "wire tensor element drifted");
            }
        }
        other => panic!("wire round-trip changed the variant: {other:?}"),
    }

    for v in [i64::MAX, -i64::MAX, 9007199254740993_i64, 0] {
        let json = serde_json::to_string(&ExecutionValue::Int64 { value: v }).expect("serialize");
        let back: ExecutionValue = serde_json::from_str(&json).expect("parse");
        match back {
            ExecutionValue::Int64 { value } => {
                assert_eq!(value, v, "wire i64 {v} did not round-trip through `{json}`")
            }
            other => panic!("wire round-trip changed the variant: {other:?}"),
        }
    }
}

// ===========================================================================
// GREEN - diagnostics embedding values (§C1.6's exit, eval lane)
// ===========================================================================

/// The one eval diagnostic that embeds a numeric ELEMENT payload today: the
/// decode chokepoint's invariant-violation message renders the offending
/// value through render_value (invariant.rs:570). The embedded digits must
/// round-trip to the stored bits at the field's width. Violating values are
/// f32-exact so ingress narrowing questions ([#729]) cannot blur the row.
#[test]
fn diagnostics_invariant_violation_embeds_faithful_value() {
    use chelis_compiler_api::schema::ExecutionValue;
    use chelis_compiler_api::{DecodeError, try_decode_adt_value};

    const SRC: &str = r#"
module Obs.Prob

@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability = | Probability { value: f32 }

def make(x: f32) -> Probability = Probability { value: x }
"#;
    let decls = chelis_surf::parser::parse_str(SRC).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);

    for violating in [2.5_f64, -0.5_f64] {
        let payload = ExecutionValue::Adt {
            ctor: "Probability".to_string(),
            fields: vec![ExecutionValue::Float64 { value: violating }],
        };
        let err = try_decode_adt_value(&exprs, &payload)
            .expect_err("out-of-band probability must be rejected");
        let msg = match &err {
            DecodeError::Invariant(msg) => msg.clone(),
            other => panic!("expected an invariant violation, got {other:?}"),
        };
        let start = msg
            .find("Probability(")
            .unwrap_or_else(|| panic!("diagnostic must embed the value: {msg}"))
            + "Probability(".len();
        let end = start
            + msg[start..]
                .find(')')
                .unwrap_or_else(|| panic!("unclosed value in: {msg}"));
        let text = &msg[start..end];
        let got = text_bits_at(text, Width::F32)
            .unwrap_or_else(|e| panic!("diagnostic payload `{text}`: {e}"));
        assert_eq!(
            got,
            value_bits_at(violating, Width::F32),
            "the diagnostic's embedded value `{text}` does not round-trip \
             to the rejected bits (§C1.6): {msg}"
        );
    }
}

// ===========================================================================
// RED - the known-unfaithful cells (§B2.3: go green only by un-ignoring)
// ===========================================================================

/// §C1.4/§C1.5: one bool tensor must read identically from print and
/// to_list. The EVAL half of chelis#726's observation split went green at
/// chelis#732 Phase 1 (un-ignored per §B2.3): every eval exit - print
/// transcript, labeled root, both to_list renders - says `true`/`false`.
#[test]
fn eval_bool_tensor_print_matches_to_list_exit() {
    let program = exits_program("tensor[2, bool]", "to_tensor([true, false])");
    let expected = ["true", "false"];

    let out = eval_stdout(&program).expect("eval");
    let tlines = tensor_lines(&out);
    assert_eq!(tlines.len(), 2, "transcript and root renders:\n{out}");
    for line in tlines {
        assert_eq!(
            tensor_elems(line),
            expected,
            "eval bool tensor print must render true/false: {line}"
        );
    }
    let llines = list_lines(&out);
    assert_eq!(
        llines.len(),
        2,
        "transcript and root to_list renders:\n{out}"
    );
    for line in llines {
        assert_eq!(list_payload_elems(line), expected, "eval to_list: {line}");
    }
}

/// The C half of the same split stays red: the emitted print helper still
/// renders bool tensor elements as `1.0`/`0.0` while to_list says
/// `true`/`false` (chelis#726's observation half, fixed by chelis#732
/// Phase 2's generated print helper).
#[test]
#[ignore = "chelis#726 (observation half, C lane; fixed by chelis#732 Phase 2): the emitted \
            bool tensor print says 1.0/0.0 while to_list says true/false. Run with \
            `cargo test -p chelis-cli --test observation_roundtrip_harness -- --ignored`."]
fn c_bool_tensor_print_matches_to_list_exit() {
    let program = exits_program("tensor[2, bool]", "to_tensor([true, false])");
    let expected = ["true", "false"];

    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let out = c_stdout(&program, "obs_bool_split").expect("C lane");
    for line in tensor_lines(&out) {
        assert_eq!(
            tensor_elems(line),
            expected,
            "C bool tensor print must render true/false: {line}"
        );
    }
    for line in list_lines(&out) {
        assert_eq!(list_payload_elems(line), expected, "C to_list: {line}");
    }
}

/// chelis#716: the emitted print helper's `default:` arm reads f16/bf16
/// buffers as f32. Value-level assertion (not exact strings) so this test
/// survives the §B2.1 grammar migration and goes green at Phase 2 untouched.
#[test]
#[ignore = "chelis#716: the emitted C print helper reads f16/bf16 tensor buffers as f32 and \
            prints garbage over CORRECT kernel results. Run with \
            `cargo test -p chelis-cli --test observation_roundtrip_harness -- --ignored`."]
fn c_f16_bf16_tensor_print_is_dtype_faithful() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    for (dt, w, rows) in [
        ("f16", Width::F16, F16_ROWS),
        ("bf16", Width::Bf16, BF16_ROWS),
    ] {
        // Print-only program: the sibling to_list exit aborts today
        // (chelis#716's other half, its own test below), and each red cell
        // must fail on ITS OWN exit.
        let elems: Vec<&str> = rows.iter().map(|r| r.elem).collect();
        let program = format!(
            "module M.Main\n\
             def mk() -> tensor[{n}, {dt}] = cast(to_tensor([{e}]), {dt})\n\
             shown = print(mk())\n",
            n = rows.len(),
            e = elems.join(", ")
        );
        let out = c_stdout(&program, &format!("obs_{dt}_print"))
            .unwrap_or_else(|e| panic!("chelis#716: the {dt} program must run: {e}"));
        // Print-only program (no value root): exactly one tensor render, the
        // `print(mk())` transcript. This cell fails on the VALUE assertion
        // below (chelis#716 reads f16/bf16 as f32), not on the render count.
        assert_float_print_exits(&out, w, rows, all_rows, 1, &format!("c/{dt}"));
    }
}

/// chelis#716: to_list of an f16/bf16 tensor aborts at runtime in the
/// compiled lane (`to_list expects numeric or bool tensor input`).
#[test]
#[ignore = "chelis#716: to_list of f16/bf16 tensors aborts at runtime in the compiled lane. \
            Run with \
            `cargo test -p chelis-cli --test observation_roundtrip_harness -- --ignored`."]
fn c_f16_bf16_to_list_completes_per_dtype() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    for (dt, w, rows) in [
        ("f16", Width::F16, F16_ROWS),
        ("bf16", Width::Bf16, BF16_ROWS),
    ] {
        let elems: Vec<&str> = rows.iter().map(|r| r.elem).collect();
        let program = format!(
            "module M.Main\n\
             def mk() -> tensor[{n}, {dt}] = cast(to_tensor([{e}]), {dt})\n\
             out = print(to_list(mk()))\n",
            n = rows.len(),
            e = elems.join(", ")
        );
        let out = c_stdout(&program, &format!("obs_{dt}_list"))
            .unwrap_or_else(|e| panic!("chelis#716: {dt} to_list must not abort: {e}"));
        for line in list_lines(&out) {
            let elems = list_payload_elems(line);
            assert_eq!(elems.len(), rows.len(), "[c/{dt}] element count: {line}");
            for (row, text) in rows.iter().zip(&elems) {
                let got =
                    text_bits_at(text, w).unwrap_or_else(|e| panic!("[c/{dt}/{}] {e}", row.label));
                assert_eq!(got, value_bits_at(row.value, w), "[c/{dt}/{}]", row.label);
            }
        }
    }
}

/// chelis#723: the emitted print helper renders int64 through double, so
/// print of a stored 2^53+1 reads back as 2^53 while to_list (green control
/// above) carries it exactly.
#[test]
#[ignore = "chelis#723: the emitted C print helper renders int64 tensor elements through \
            double; 9007199254740993 prints as a value that decodes to 9007199254740992. \
            Run with \
            `cargo test -p chelis-cli --test observation_roundtrip_harness -- --ignored`."]
fn c_int64_tensor_print_round_trips_above_2p53() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = exits_program(
        "tensor[2, int64]",
        "to_tensor([cast(9007199254740993, int64), cast(9223372036854775807, int64)])",
    );
    let out = c_stdout(&program, "obs_i64_print").expect("C lane");
    let expected = [9007199254740993_i64, i64::MAX];
    for line in tensor_lines(&out) {
        let elems = tensor_elems(line);
        for (want, text) in expected.iter().zip(&elems) {
            let got = text_int_lenient(text).unwrap_or_else(|e| panic!("print exit: {e}"));
            assert_eq!(
                got, *want,
                "the printed int64 tensor must carry the exact stored value; got `{text}`"
            );
        }
    }
}

/// DISCOVERY chelis#748 (§B2.5, found while building this harness): the
/// emitted print paths' format SELECTION loses whole value classes
/// independently of the dtype-funnel bugs:
///
/// * the near-integer branch `fabs(v - round(v)) < 1e-9 -> %.1f` collapses
///   EVERY |x| < 1e-9 to `0.0` - f32/f64 subnormals, min-normals, tiny
///   gradients (host_emit.rs:516; runtime lib.rs:3772 has the same test);
/// * `%.16g` emits 16 significant digits, one short of f64's worst case -
///   0.30000000000000004 reads back as 0.3, and f64::MAX reads back as INF
///   (host_emit.rs:518 tensor path, :4275/:4366 scalar paths).
///
/// Both are Phase 2 casualties (chelis_format_shortest replaces the split).
#[test]
#[ignore = "chelis#748 (B2.5 discovery): \
            the emitted %.1f arm collapses |x|<1e-9 to 0.0 and %.16g starves 17-digit f64 \
            values. Run with \
            `cargo test -p chelis-cli --test observation_roundtrip_harness -- --ignored`."]
fn c_print_format_selection_preserves_small_and_17_digit_values() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    // The compilable f64/f32 table rows excluded from the green print
    // assertions (the C_LANE_EXCLUDED cells never reach a renderer at all).
    let rows = c_rows(F64_ROWS);
    let out = c_stdout(&float_table_program("f64", &rows, false), "obs_f64_red").expect("C lane");
    assert_float_exits(
        &out,
        Width::F64,
        &rows,
        c_print_red_rows,
        all_rows,
        "c/f64-red",
    );

    let rows = c_rows(F32_ROWS);
    let out = c_stdout(&float_table_program("f32", &rows, false), "obs_f32_red").expect("C lane");
    assert_float_exits(
        &out,
        Width::F32,
        &rows,
        c_print_red_rows,
        all_rows,
        "c/f32-red",
    );

    // The scalar %.16g path starves the same 17-digit f64 values. (f64::MAX
    // would starve too - %.16g parses back as inf - but its C cell is
    // unconstructible today: the emitted constant is an integer literal
    // clang rejects, the C_LANE_EXCLUDED ingress defect.)
    for (i, (expr, value)) in [("cast(0.30000000000000004, f64)", 0.30000000000000004)]
        .iter()
        .enumerate()
    {
        let program = format!("module M.Main\ndef run() -> f64 = {expr}\nshown = print(run())\n");
        let out = c_stdout(&program, &format!("obs_sc_red{i}")).expect("C lane");
        for line in scalar_render_lines(&out) {
            let got =
                text_bits_at(&line, Width::F64).unwrap_or_else(|e| panic!("[c scalar {expr}] {e}"));
            assert_eq!(
                got,
                value_bits_at(*value, Width::F64),
                "[c scalar {expr}] text `{line}` does not round-trip"
            );
        }
    }
}

/// DISCOVERY chelis#749 (§B2.5): tensors rendered INSIDE a
/// list go through the runtime's `tensor_to_string` (chelis-runtime
/// lib.rs:3730), a SECOND hand-written dtype funnel: int64 read `as f64`
/// (chelis#723's shape at a different site), f16/bf16 through the `_ =>`
/// f32 fallback (chelis#716's shape), and a 10-element truncation with NO
/// marker. This row pins the int64 case at the value level.
#[test]
#[ignore = "chelis#749 (B2.5 discovery): \
            the runtime nested-value tensor renderer reads int64 elements as f64 and \
            silently truncates at 10 elements. Run with \
            `cargo test -p chelis-cli --test observation_roundtrip_harness -- --ignored`."]
fn c_nested_tensor_in_list_renders_int64_faithfully() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def mk() -> tensor[1, int64] = to_tensor([cast(9007199254740993, int64)])\n\
         out = print([mk()])\n";
    let out = c_stdout(program, "obs_i64_nested").expect("C lane");
    let line = out
        .lines()
        .find(|l| l.contains("tensor(shape="))
        .unwrap_or_else(|| panic!("no nested tensor render in:\n{out}"));
    let got = text_int_lenient(&tensor_elems(line)[0]).expect("nested elem");
    assert_eq!(
        got, 9007199254740993,
        "the nested tensor render must carry the exact stored int64: {line}"
    );
}
