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
//! None. The chelis#865 boxed-f32 cell became an ordinary regression row when
//! chelis#729 Phase 3 routed non-f64 boxed scalars through the tagged rank-0
//! tensor carrier. The chelis#1110 suffixed-literal cell followed when host
//! lowering began finalizing each literal at its checker-stamped width before
//! any enclosing cast.
//!
//! The six C-side cells went green at chelis#732 Phase 2 (un-ignored per
//! §B2.3, each on its original assertion): #716 print + to_list, #723,
//! #726's C half, #748, #749. The compiled lane now renders through the
//! generated print helper and `chelis_format_shortest`, and §C2.3
//! cross-lane byte equality is locked below for agreeing bits
//! (`cross_lane_stdout_is_byte_identical_where_bits_agree`).
//!
//! No value cell is skipped silently. The phase's authoritative
//! oracle (`.venv/bin/python scripts/faithful_observation_phase2_oracle.py`)
//! holds the empty known-red ledger and requires this file's `#[ignore]`
//! inventory to EQUAL it (an undeclared ignore is a silently
//! narrowed corpus; a stale ledger row is a false statement about
//! coverage), re-runs each cell, and FAILS if one is red for an
//! undeclared reason OR has gone green. The green case is the [#729]
//! handoff: when its value/capacity repair lands, the oracle goes red
//! until the cell is un-ignored on its original assertion and the ledger
//! row deleted in that change set. Phase 3 also retired the final chelis#751
//! corpus exclusions: exact-bit C literal emission lets every frozen float
//! row execute in the compiled lane. The §C2.3 cross-lane byte-identity
//! corpus may grow but never shrink.
//!
//! Everything else is green by contract; a new red here is a new
//! faithful-observation bug (file it, per §B2.5).
//!
//! The wire's capacity limits (JSON cannot carry NaN/inf as numbers; int64
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
// is the constructed value's exact f64 image. `c_print_safe` marks the rows
// the PRE-Phase-2 `%.1f`/`%.16g` selection could round-trip; since
// chelis#732 Phase 2 every compilable row prints faithfully, and the split
// only partitions which test asserts the row (the historical green set
// below, the once-red set in the un-ignored chelis#748 test) - together
// they cover every row.
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
        elem: "5.960464477539063e-8",
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

/// chelis#751 regression: Phase 3 emits every floating literal from its exact
/// bit pattern, so the compiled lane has no constant-ingress exclusions.
/// Keeping this helper as an identity makes every existing call site execute
/// the complete frozen table and prevents a future source-format shortcut
/// from silently reintroducing a filtered subset.
fn c_rows(rows: &[FRow]) -> Vec<FRow> {
    rows.to_vec()
}

fn c_print_safe_rows(r: &FRow) -> bool {
    r.c_print_safe
}

fn c_print_red_rows(r: &FRow) -> bool {
    !r.c_print_safe
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
    assert_float_exits(&out, Width::F64, F64_ROWS, all_rows, all_rows, "eval/f64");

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
/// collapsed 2^53+1 before ANY exit rendered it (green by uniform
/// wrongness); chelis#729 Phase 1 made the host-lane exits exact with
/// the tensor-lane labeled root still collapsing through the f64 DAG
/// literal payload (green-to-ignored against chelis#856); the chelis#729
/// rework sealed the DAG literal payload, so ALL FOUR exits are exact
/// and the row is green again for the right reason. The C lane's
/// tier-2 version of this row is the chelis#723 ignored test below.
#[test]
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

/// GREEN regression (chelis#684, [#729] value layer; surfaced by PR
/// #792's red team, F1): an int64 SCALAR ROOT above 2^53 stays exact at
/// the labeled root, agreeing with print and to_string of the same def.
/// The interpreter's rank-0 f64 realization used to collapse the value
/// BEFORE the renderer saw it, which made this a stored-value defect
/// upstream of the [05-OBS] rendering contract rather than a formatter
/// bug. chelis#729's per-dtype storage carries the exact i64 through the
/// scalar-root path, so the original red assertion stays as the
/// [05-OBS-1] regression lock (un-ignored per §B2.3, chelis#1078).
#[test]
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
    // The labeled root must carry the same exact value. Before chelis#684
    // it rendered the f64-collapsed 9007199254740992.
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

/// GREEN regression (chelis#864; PR #863 red-team F1): eval's transcript
/// and labeled-root exits agree for a cast-constructed f64 tensor. The
/// post-Phase-3 diagnosis established that the root tag was already F64;
/// the static `to_tensor` DAG shortcut had retained lexical f64 decimals
/// in its F32 source node, so widening produced different stored values
/// from the host path. Finalizing that shortcut's f32 ingress before the
/// cast repairs the value, and this original red assertion stays as the
/// [05-OBS-1] regression lock.
#[test]
fn eval_f64_cast_tensor_root_renders_stored_width() {
    let cases = [
        (
            "f32 tensor widened as a tensor",
            "cast(to_tensor([0.1, 0.3]), f64)",
        ),
        (
            "f32-suffixed leaves widened as scalars",
            "to_tensor([cast(0.1f32, f64), cast(0.3f32, f64)])",
        ),
        (
            "explicit inner f32 casts widened as scalars",
            "to_tensor([cast(cast(0.1, f32), f64), cast(cast(0.3, f32), f64)])",
        ),
    ];
    for (case, expression) in cases {
        let program = format!(
            "module M.Main\ndef mk() -> tensor[2, f64] = {expression}\n\
             shown = print(mk())\ntroot = mk()\n"
        );
        let out = eval_stdout(&program).expect("eval");
        let tlines = tensor_lines(&out);
        assert_eq!(
            tlines.len(),
            2,
            "{case}: transcript and root renders:\n{out}"
        );
        // Green half (control): the transcript renders the STORED bits (the
        // f64 images of the f32-constructed elements) at the stored width.
        assert_eq!(
            tlines[0], "tensor(shape=[2], data=[0.10000000149011612, 0.30000001192092896])",
            "{case}: the print transcript must keep rendering the stored bits"
        );
        // The labeled root must agree with the transcript ([05-OBS-1]
        // intra-lane exit agreement). Before chelis#864 the static DAG
        // shortcut could widen unfinalized lexical decimals instead of the
        // host path's stored f32 values.
        assert_eq!(
            tlines[1],
            format!("troot = {}", tlines[0]),
            "{case}: the labeled root must render the same stored bits as print"
        );
    }
}

/// chelis#865 regression: boxed f32 values retain their width through the
/// tagged rank-0 tensor carrier, so list rendering uses the [05-OBS-2]
/// shortest-at-own-width form rather than f64-image digits.
#[test]
fn c_boxed_f32_renders_at_own_width() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\nout = print(to_list(to_tensor([0.1, 0.3])))\n";
    let out = c_stdout(program, "obs_boxed_f32").expect("C lane");
    let lline = list_lines(&out)
        .first()
        .copied()
        .unwrap_or_else(|| panic!("no to_list render in:\n{out}"))
        .to_string();
    assert_eq!(
        list_payload_elems(&lline),
        ["0.1", "0.3"],
        "boxed f32 elements must render shortest at their own width ([05-OBS-2]): {lline}"
    );
}

/// chelis#1110 regression: a float literal is finalized at its checked width
/// before an enclosing cast widens it for the compiled lane. The narrow rows
/// lock all three affected dtypes; the f64 row is their negative-parity guard
/// against manufacturing an f32 rounding step for an explicitly f64 leaf.
#[test]
fn c_suffixed_f32_literal_widens_from_its_stored_width() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let cases = [
        (
            "f16",
            "0.1f16",
            "0.3f16",
            "tensor(shape=[2], data=[0.0999755859375, 0.300048828125])",
        ),
        (
            "bf16",
            "0.1bf16",
            "0.3bf16",
            "tensor(shape=[2], data=[0.10009765625, 0.30078125])",
        ),
        (
            "f32",
            "0.1f32",
            "0.3f32",
            "tensor(shape=[2], data=[0.10000000149011612, 0.30000001192092896])",
        ),
        (
            "f64",
            "0.1f64",
            "0.3f64",
            "tensor(shape=[2], data=[0.1, 0.3])",
        ),
    ];
    for (dtype, lhs, rhs, expected) in cases {
        let program = format!(
            "module M.Main\n\
             def mk() -> tensor[2, f64] = to_tensor([cast({lhs}, f64), cast({rhs}, f64)])\n\
             shown = print(mk())\n"
        );
        let out = c_stdout(&program, &format!("obs_suffixed_{dtype}"))
            .unwrap_or_else(|error| panic!("{dtype} C lane: {error}"));
        let tline = tensor_lines(&out)
            .first()
            .copied()
            .unwrap_or_else(|| panic!("{dtype}: no tensor render in:\n{out}"))
            .to_string();
        let eval_out =
            eval_stdout(&program).unwrap_or_else(|error| panic!("{dtype} eval: {error}"));
        let eval_tline = tensor_lines(&eval_out)
            .first()
            .copied()
            .unwrap_or_else(|| panic!("{dtype}: no eval tensor render in:\n{eval_out}"))
            .to_string();
        assert_eq!(
            eval_tline, expected,
            "{dtype}: control lane must finalize the suffixed leaf at its declared width"
        );
        assert_eq!(
            tline, eval_tline,
            "{dtype}: the compiled lane must widen the suffixed literal's stored value, \
             agreeing with eval ([05-OBS-1] cross-lane exit agreement): {tline}"
        );
    }
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

/// GREEN regression (chelis#862, closed by [05-OBS-6] under chelis#912): a
/// program whose ONLY display root is a unit-valued PRINT root renders that
/// root WITH its `name = ` label.
///
/// Eval used to drop the prefix for exactly this shape - a lone bare `()`
/// where the compiled lane printed `out = ()`. Adding any second root made
/// eval label the unit root normally, so the divergence class was precisely
/// "programs whose only display root is print-valued", which is why the
/// multi-root rows elsewhere in this file never caught it. [05-OBS-6] removed
/// the bare-when-single form in both lanes; this cell is the regression lock.
///
/// Two deliberate choices, both load-bearing:
///
/// * The assertion is on the ROOT RENDER LINE, never the print transcript.
///   `c_suffixed_f32_literal_widens_from_its_stored_width` above builds the
///   same single-print-root shape but asserts on `tensor_lines().first()`, so
///   an assertion copied from there would pass vacuously for a second,
///   independent reason.
/// * No `c_toolchain_available()` guard, because none is needed: the
///   divergence was eval-internal (the C lane was already correct), so this
///   cell must not inherit the silent skip that
///   `cross_lane_stdout_is_byte_identical_where_bits_agree` opens with.
///   Joining the single-print-root shape to that cross-lane corpus is a
///   separate follow-up, not this regression.
#[test]
fn eval_unit_valued_sole_print_root_keeps_its_name_issue_862() {
    // (label, program, expected sole root line). Each program's only root is
    // the unit-valued print binding; the binding name varies so the label is
    // proven to come from the binding, not from a constant.
    let rows: &[(&str, String, &str)] = &[
        (
            "tensor-reduction-root",
            "module M.Main\n\
             def f(x: tensor[4, f32]) -> tensor[f32] = sum(x, 0)\n\
             out = print(f(to_tensor([1.5, 4.5, 2.5, 0.5])))\n"
                .to_string(),
            "out = ()",
        ),
        (
            "f64-scalar-root",
            "module M.Main\n\
             def run() -> f64 = cast(0.1, f64)\n\
             shown = print(run())\n"
                .to_string(),
            "shown = ()",
        ),
        (
            "int32-scalar-root-distinct-name",
            "module M.Main\n\
             def run() -> int32 = 7\n\
             whatever = print(run())\n"
                .to_string(),
            "whatever = ()",
        ),
    ];

    for (label, program, expected_root) in rows {
        let out = eval_stdout(program).unwrap_or_else(|error| panic!("[{label}] eval: {error}"));
        let unit_lines: Vec<&str> = out
            .lines()
            .map(str::trim)
            .filter(|line| line.ends_with("()"))
            .collect();
        assert_eq!(
            unit_lines,
            vec![*expected_root],
            "[{label}] [05-OBS-6]: the sole unit-valued print root is labelled, \
             and the bare-when-single form is gone. Full output:\n{out}"
        );
        assert!(
            !out.lines().any(|line| line.trim() == "()"),
            "[{label}] [05-OBS-6]: no exit may render a root as a bare `()`. \
             Full output:\n{out}"
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
    let float_rows: &[(&str, &str, Width, f64)] = &[
        ("f16", "cast(0.1, f16)", Width::F16, 0.0999755859375),
        ("bf16", "cast(0.1, bf16)", Width::Bf16, 0.10009765625),
        ("f64", "cast(0.1, f64)", Width::F64, 0.1),
        ("f64", "cast(-0.0, f64)", Width::F64, -0.0),
        (
            "f64",
            "cast(1.7976931348623157e308, f64)",
            Width::F64,
            f64::MAX,
        ),
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
        (
            "f32",
            "3.4028234663852886e38",
            Width::F32,
            3.4028234663852886e38,
        ),
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
/// now emitted with the exact stored values (127 / -127 / 0). Since
/// chelis#732 Phase 2 BOTH lanes render integers as integers ([05-OBS-2]),
/// so the lock is the full §C2.3 byte-identical assertion (the Phase 1
/// interim compared values only, while the compiled lane still printed
/// `127.0`).
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

    let expected = "troot = tensor(shape=[3], data=[127, -127, 0])";
    for (lane, out) in [("eval", &eval_out), ("c", &c_out)] {
        assert!(
            out.lines().any(|l| l == expected),
            "chelis#750/[05-OBS-2] [{lane}]: the def-call tensor root must \
             render `{expected}`, got:\n{out}"
        );
    }
    // §C2.3: identical stored bits, byte-identical stdout.
    assert_eq!(
        eval_out, c_out,
        "chelis#750/§C2.3: the lanes' stdout must be byte-identical:\n--- eval ---\n{eval_out}\n--- c ---\n{c_out}"
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
            fields: vec![ExecutionValue::Float32 {
                value: violating as f32,
            }],
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

/// The C half of the same split, green since chelis#732 Phase 2
/// (un-ignored per §B2.3): the generated print helper renders bool tensor
/// elements `true`/`false`, agreeing with to_list (chelis#726 closed).
#[test]
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

/// chelis#716's print half, green since chelis#732 Phase 2 (un-ignored per
/// §B2.3, assertions untouched): the generated helper decodes f16/bf16
/// storage through the WS-1 conversion helpers and formats at the value's
/// own width. (The pre-fix helper read the 2-byte buffers as f32; the
/// chelis#730 interim then aborted loudly instead.)
#[test]
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

/// chelis#716's to_list half, green since chelis#732 Phase 2 (un-ignored
/// per §B2.3): `chelis_list_from_tensor` gained F16/BF16 arms reading the
/// 2-byte storage exactly (the pre-fix runtime aborted; after PR #799 the
/// build itself rejected `list[f16]` until the boxed-element ABI carve-out
/// in `host_abi.rs` named that state). The list box carries each element's
/// exact f64 image, so the value-level round-trip at the half width holds;
/// BYTE equality with eval's own-width to_list digits for non-dyadic
/// values waits on the chelis#729 width repair (the section 8.1 boundary).
#[test]
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

/// chelis#723, green since chelis#732 Phase 2 (un-ignored per §B2.3): the
/// generated helper prints int64 elements through `long long` printf, all
/// digits exact, never through double.
#[test]
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
/// Both went green at chelis#732 Phase 2 (un-ignored per §B2.3):
/// `chelis_format_shortest` replaced the near-integer/`%g` split at every
/// compiled print exit, so the once-collapsed rows now round-trip.
#[test]
fn c_print_format_selection_preserves_small_and_17_digit_values() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    // Every frozen f64/f32 row now reaches the compiled renderer. The
    // predicate selects the historical formatting-red subset only.
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

    // The scalar %.16g path starved the same 17-digit f64 values. Both an
    // ordinary 17-digit value and f64::MAX now execute end to end.
    for (i, (expr, value)) in [
        ("cast(0.30000000000000004, f64)", 0.30000000000000004),
        ("cast(1.7976931348623157e308, f64)", f64::MAX),
    ]
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
/// marker. This row pins the int64 case at the value level. Green since
/// chelis#732 Phase 2 (un-ignored per §B2.3): `tensor_to_string` renders
/// per dtype through the shared runtime formatter, and its truncation is
/// the [05-OBS-5] 32-with-marker rule
/// (`c_nested_tensor_truncates_at_32_with_marker` below locks that half).
#[test]
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

/// The int32 face of the same nested-in-list exit (census row C4). The
/// runtime's element decoder read NATIVE int32 storage through the f32
/// view, so this exit rendered `5i32` as `7.006492321624085e-45`,
/// `i32::MAX` (`0x7FFFFFFF`, a NaN pattern) as `NaN`, and `i32::MIN`
/// (`0x80000000`, `-0.0`) as a plausible-looking `0` - while `to_list`
/// and the generated print helper, which both decode natively, returned
/// the right integers from the SAME tensor. That is §C2.2 (intra-lane
/// exit agreement) and §C2.3 (cross-lane byte identity, since eval was
/// correct) failing together.
///
/// The cell hid because the harness's int32 rows are driven by
/// `c_int_tensor_exits_round_trip`, which exercises the GENERATED print
/// helper; `tensor_to_string` is a different exit, reached only through
/// the nested-value renderer (int64-only coverage until now) and through
/// the now-removed zero-emitter `chelis_print_f32`. A Phase 0 census gap
/// against its own deliverable, not a misread oracle - and the shape the
/// known-red ledger cannot catch, because the ledger polices DECLARED
/// skips, never exits nobody censused.
///
/// `i32::MIN` is the row that matters most: NaN screams, `0` does not.
#[test]
fn c_nested_int32_tensor_in_list_decodes_natively() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def mk() -> tensor[3, int32] = to_tensor([cast(5, int32), \
         cast(2147483647, int32), cast(-2147483648, int32)])\n\
         nested = print([mk()])\n\
         listed = print(to_list(mk()))\n";
    let out = c_stdout(program, "obs_i32_nested").expect("C lane");
    let nested = out
        .lines()
        .find(|l| l.contains("tensor(shape="))
        .unwrap_or_else(|| panic!("no nested tensor render in:\n{out}"));
    assert_eq!(
        tensor_elems(nested),
        ["5", "2147483647", "-2147483648"],
        "the nested int32 render must decode native two's-complement \
         storage ([05-OBS-2]); an f32 view yields 7.006492321624085e-45 / \
         NaN / 0: {nested}"
    );
    // §C2.2: the sibling exit on the same stored bits must agree. Both
    // renders are list lines, so select the `to_list` one by the absence
    // of the nested tensor wrapper.
    let listed = list_lines(&out)
        .into_iter()
        .find(|l| !l.contains("tensor(shape="))
        .unwrap_or_else(|| panic!("no to_list render in:\n{out}"))
        .to_string();
    assert_eq!(
        list_payload_elems(&listed),
        ["5", "2147483647", "-2147483648"],
        "to_list must agree with the nested render: {listed}"
    );
    // §C2.3: eval holds identical bits and must emit identical bytes.
    let eval_out = eval_stdout(
        "module M.Main\n\
         def mk() -> tensor[3, int32] = to_tensor([cast(5, int32), \
         cast(2147483647, int32), cast(-2147483648, int32)])\n\
         nested = print([mk()])\n\
         listed = print(to_list(mk()))\n",
    )
    .expect("eval");
    assert_eq!(
        eval_out, out,
        "§C2.3: the nested int32 exit must be byte-identical across lanes:\n\
         --- eval ---\n{eval_out}\n--- c ---\n{out}"
    );
}

/// chelis#749's truncation half ([05-OBS-5]): a tensor nested inside a
/// list truncates at 32 elements WITH the `, ...` marker (the pre-fix
/// runtime renderer cut at 10 with no marker - a 33-element tensor was
/// indistinguishable from a 10-element one).
#[test]
fn c_nested_tensor_truncates_at_32_with_marker() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let elems: Vec<String> = (1..=33).map(|i| format!("{i}.0")).collect();
    let program = format!(
        "module M.Main\n\
         def mk() -> tensor[33, f32] = to_tensor([{}])\n\
         out = print([mk()])\n",
        elems.join(", ")
    );
    let out = c_stdout(&program, "obs_nested_trunc").expect("C lane");
    let visible: Vec<String> = (1..=32).map(|i| format!("{i}.0")).collect();
    let expected = format!("[tensor(shape=[33], data=[{}, ...])]", visible.join(", "));
    assert!(
        out.lines().any(|l| l.trim() == expected),
        "the nested render must truncate at 32 with the marker, got:\n{out}"
    );
}

/// The compiled-lane scalar `to_string` trio (PR #863 round-1 F1;
/// census row C7, appended per §B2.5): for every stringifiable scalar
/// dtype, `print(to_string(x))` and `print(x)` must agree byte-for-byte
/// within the compiled lane (§C2.2 intra-lane exit agreement) AND the
/// full stdout must be byte-identical to eval's (§C2.3). The f32 row is
/// the fixed funnel: the pre-fix emission promoted f32 through
/// `chelis_string_from_f64`, so `to_string(cast(0.1, f32))` printed the
/// f64-image digits while `print` of the same stored value printed
/// `0.1`. The other rows lock the widths that were already correct.
/// (f16/bf16 scalars have no C-host ABI cell and cannot reach
/// `to_string`; their absence is chelis#714's, not this row's.)
#[test]
fn c_scalar_to_string_matches_print_exit_across_dtypes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let rows: &[(&str, &str, &str)] = &[
        ("f32", "cast(0.1, f32)", "0.1"),
        ("f32-dyadic", "cast(0.75, f32)", "0.75"),
        (
            "f64",
            "cast(0.30000000000000004, f64)",
            "0.30000000000000004",
        ),
        ("f64-integral", "cast(6.0, f64)", "6.0"),
        ("int64", "cast(9007199254740993, int64)", "9007199254740993"),
        ("int32", "cast(2147483647, int32)", "2147483647"),
        ("bool", "and(true, true)", "true"),
    ];
    for (label, expr, expected) in rows {
        let program = format!(
            "module M.Main\n\
             def run() -> string = to_string({expr})\n\
             shown = print(run())\n\
             also = print({expr})\n"
        );
        let name = format!("obs_tostr_{}", label.replace('-', "_"));
        let c_out = c_stdout(&program, &name).unwrap_or_else(|e| panic!("[{label}] C lane: {e}"));
        let eval_out = eval_stdout(&program).unwrap_or_else(|e| panic!("[{label}] eval: {e}"));
        let c_lines: Vec<&str> = c_out.lines().collect();
        assert!(
            c_lines.len() >= 2 && c_lines[0] == *expected && c_lines[1] == *expected,
            "[{label}] the to_string and print exits must both render `{expected}` \
             within the compiled lane, got:\n{c_out}"
        );
        assert_eq!(
            eval_out, c_out,
            "[{label}] §C2.3: the scalar to_string trio must be byte-identical \
             across lanes:\n--- eval ---\n{eval_out}\n--- c ---\n{c_out}"
        );
    }
}

// ===========================================================================
// GREEN - chelis#732 Phase 2: §C2.3 cross-lane byte equality (frozen at
// this phase's exit for identical stored bits)
// ===========================================================================

/// Eval and compiled C emit BYTE-IDENTICAL stdout for identical stored
/// bits (§C2.3, [05-OBS-2]'s cross-lane grammar identity). The corpus is
/// deliberately restricted to cells where the lanes hold identical bits
/// AND render at the same width today:
///
/// * integer and bool tensors (exact in both storage strategies within
///   2^53);
/// * f64 SCALAR exits (both lanes store and render f64);
/// * float tensors whose elements are dyadic short decimals (exactly
///   representable at every width in play, so eval's deliberate
///   stored-f64-width tensor rendering - the spec/05 section 8.1 width
///   note - produces the same digits as the C lane's own-width
///   rendering).
///
/// NOT here, by the section 8.1 boundary (chelis#729's subject matter,
/// deliberately visible until its width repair): non-dyadic narrow-float
/// tensor cells, where eval renders the stored f64 image's digits while
/// the C lane renders own-width digits (e.g. an f32 tensor holding 0.1:
/// eval `0.10000000149011612`, C `0.1` - same bits, different declared
/// widths); and eval to_list of f64 tensors, whose values narrow through
/// the chelis#717 F32 tag before rendering (different stored bits). Rows
/// for those cells belong to the chelis#729 exit criteria, not this lock.
#[test]
fn cross_lane_stdout_is_byte_identical_where_bits_agree() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let programs: &[(&str, String)] = &[
        ("int8-exits", int_table_program("int8", INT_ROWS[0].1)),
        ("int64-exits", int_table_program("int64", INT_ROWS[3].1)),
        (
            "bool-exits",
            exits_program("tensor[2, bool]", "to_tensor([true, false])"),
        ),
        (
            "f32-dyadic-exits",
            exits_program("tensor[4, f32]", "to_tensor([1.5, -0.25, 2048.0, 0.75])"),
        ),
        (
            "f16-dyadic-print",
            "module M.Main\n\
             def mk() -> tensor[3, f16] = cast(to_tensor([0.75, 2048.0, -1.5]), f16)\n\
             shown = print(mk())\n\
             listed = print(to_list(mk()))\n"
                .to_string(),
        ),
        (
            "f64-scalar-17-digit",
            "module M.Main\n\
             def run() -> f64 = cast(0.30000000000000004, f64)\n\
             shown = print(run())\n\
             root = run()\n"
                .to_string(),
        ),
        (
            "f64-scalar-specials",
            "module M.Main\n\
             def run() -> f64 = div(cast(1.0, f64), cast(0.0, f64))\n\
             shown = print(run())\n\
             root = run()\n"
                .to_string(),
        ),
        // The program carries a VALUE root beside the print. That shape was
        // originally chosen to avoid the chelis#862 unit-root naming
        // divergence (eval rendered a LONE unit root bare while the compiled
        // lane named it); [05-OBS-6] closed that, and
        // `eval_unit_valued_sole_print_root_keeps_its_name_issue_862` above
        // locks it. The multi-root shape is kept here because moving this
        // cell to the single-print-root form would widen the §C2.3 corpus,
        // which is a separate change - the corpus may grow but never shrink.
        (
            "rank0-reduction-root",
            "module M.Main\n\
             def f(x: tensor[4, f32]) -> tensor[f32] = sum(x, 0)\n\
             shown = print(f(to_tensor([1.5, 4.5, 2.5, 0.5])))\n\
             root = f(to_tensor([1.5, 4.5, 2.5, 0.5]))\n"
                .to_string(),
        ),
    ];
    for (label, program) in programs {
        let eval_out = eval_stdout(program).unwrap_or_else(|e| panic!("[{label}] eval: {e}"));
        let c_out = c_stdout(program, &format!("obs_xlane_{label}").replace('-', "_"))
            .unwrap_or_else(|e| panic!("[{label}] C lane: {e}"));
        assert_eq!(
            eval_out, c_out,
            "[{label}] §C2.3: identical stored bits must render byte-identically:\n\
             --- eval ---\n{eval_out}\n--- c ---\n{c_out}"
        );
    }
}
