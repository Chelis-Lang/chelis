//! Host-lane CSV I/O core (chelis#903).
//!
//! Pure helpers behind the `parse_csv` / `to_csv` / `csv_*` builtins
//! dispatched from `eval.rs`. A **Csv document** is not a new value kind:
//! it rides the prelude `Json` ADT (chelis#890, `runtime/json.rs`) as a
//! fixed-shape document
//!
//! ```text
//! JDict {
//!   "columns": JList [JStr col, ...],          -- header, in file order
//!   "rows":    JList [JDict {col: cell, ...}]  -- one JDict per data row
//! }
//! ```
//!
//! so every `json_*` accessor from #890 works on it (`json_list(c, "rows")`,
//! per-row `json_str(row, "col")`, `to_json(c)` for debugging), while the
//! `csv_*` accessors below are the ergonomic column-oriented surface. The
//! separate `columns` list (not just the row-dict keys) is what preserves
//! header order for zero-row files.
//!
//! Design contract (see `docs/CHELIS_SURFACE.md` §3.9):
//!
//! * **Parsing is RFC-4180-ish and loud.** First row = header; quoted
//!   fields with doubled embedded quotes; commas/quotes/newlines are
//!   literal inside quotes; LF or CRLF row separators (mixed tolerated); a
//!   leading UTF-8 BOM is stripped. Everything malformed — unclosed quote,
//!   content after a closing quote, a bare `"` in an unquoted field, bare
//!   CR, ragged rows, interior blank rows, duplicate header names — fails
//!   naming the 1-based row (record) number, plus the 1-based column
//!   (field) number where one applies (quote and separator errors; ragged
//!   and blank rows are whole-row errors). Blank rows are tolerated only
//!   at end of input.
//! * **Cells are strings at parse time.** No silent numeric coercion (a
//!   column of zero-padded IDs must not type-flap mid-column). The numeric
//!   accessors convert at access time under the strict JSON number grammar
//!   (literally `parse_json`'s number scanner), tolerating surrounding
//!   ASCII spaces/tabs (matching Python `float()`), and fail loudly on
//!   anything else — naming the column, the row, and the offending text.
//!   Empty cells in a numeric accessor are errors: **no silent NaN**.
//! * **Accessors read a cell's own type only — no cross-type coercion.**
//!   `csv_str` reads `JStr` verbatim; `csv_f64` reads `JStr` (strict
//!   parse) or a finite `JNum` (hand-assembled documents; non-finite is a
//!   loud error). `JBool`/`JNull` are to_csv's write-side cell encodings
//!   and fail loudly in every read accessor, each error naming the actual
//!   cell type.
//! * **Document-shape validation is eager and uniform; cell validation is
//!   lazy.** Every accessor (including `csv_nrows`/`csv_cols`) validates
//!   the same shape up front — `columns` a JList of unique strings, every
//!   `rows` element a JDict — so a hand-mangled document fails identically
//!   everywhere; individual cell types are checked at read.
//! * **Row indices are 0-based data rows** (`csv_f64(c, 0, "col")` is the
//!   first row *after* the header); parse errors count 1-based physical
//!   records (header = row 1). Error messages state which space they use.
//! * **`to_csv` serializes the exact document shape `parse_csv` returns,
//!   and the round-trip is values-as-text.** Cells may be `JStr`
//!   (verbatim), `JNum` (shortest-round-trip f64 via the shared
//!   `format_f64_json` — the same fidelity contract as `to_json`, never
//!   the print channel: chelis#748/#723/#734), `JBool` (`true`/`false`),
//!   or `JNull` (empty cell). `parse_csv(to_csv(c))` reproduces the table
//!   with every cell normalized to the `JStr` of its serialized field
//!   text — an all-`JStr` document round-trips identically, and a `JNum`
//!   cell re-reads bit-exactly through `csv_f64` — but cell TYPES other
//!   than `JStr` do not survive the trip (CSV is untyped text). Container
//!   cells, non-finite numbers, unexpected top-level keys, rows missing a
//!   declared column, and rows carrying an undeclared or duplicate key
//!   all fail loudly — no silent data loss.

use super::json::{jdict, jlist, jstr};
use super::{RuntimeValue, truncate_rendered, truncated_debug};

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// One physical CSV record. `raw_empty` distinguishes a genuinely blank
/// line (zero characters before the row separator) from a single-column
/// row holding a quoted empty cell (`""`) — both decode to one empty
/// field, but only the former is a blank line.
struct CsvRecord {
    fields: Vec<String>,
    raw_empty: bool,
}

/// Parse CSV text into the fixed-shape Csv document described in the
/// module docs. First row = header. All failures are loud, with 1-based
/// row/column numbers.
pub(super) fn parse_csv_text(text: &str) -> Result<RuntimeValue, String> {
    // Strip a leading UTF-8 BOM (common in Windows-exported CSVs); left in
    // place it would silently glue onto the first header name and every
    // by-name lookup of that column would fail mysteriously.
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);

    let mut records = parse_records(text)?;
    // Trailing blank rows are a line-terminator artifact, not data; a
    // single-column empty cell must be written as `""` to survive here.
    while matches!(records.last(), Some(record) if record.raw_empty) {
        records.pop();
    }
    let Some((header, data)) = records.split_first() else {
        return Err("parse_csv: input is empty (expected a header row)".to_string());
    };
    if header.raw_empty {
        return Err("parse_csv: row 1 is blank (expected a header row)".to_string());
    }
    for (index, name) in header.fields.iter().enumerate() {
        if let Some(first) = header.fields[..index]
            .iter()
            .position(|other| other == name)
        {
            return Err(format!(
                "parse_csv: duplicate header column `{name}` (columns {} and {}); \
                 columns must be addressable by unique names",
                first + 1,
                index + 1
            ));
        }
    }

    let mut rows: Vec<RuntimeValue> = Vec::with_capacity(data.len());
    for (index, record) in data.iter().enumerate() {
        let row_number = index + 2; // 1-based records; header is row 1.
        if record.raw_empty {
            return Err(format!(
                "parse_csv: row {row_number} is blank (blank rows are only tolerated at \
                 end of input; to mean an empty cell, quote it as \"\")"
            ));
        }
        if record.fields.len() != header.fields.len() {
            return Err(format!(
                "parse_csv: row {row_number} has {} fields, expected {} (the header \
                 row names {} columns)",
                record.fields.len(),
                header.fields.len(),
                header.fields.len()
            ));
        }
        rows.push(jdict(
            header
                .fields
                .iter()
                .cloned()
                .zip(record.fields.iter().cloned().map(jstr))
                .collect(),
        ));
    }

    Ok(jdict(vec![
        (
            "columns".to_string(),
            jlist(header.fields.iter().cloned().map(jstr).collect()),
        ),
        ("rows".to_string(), jlist(rows)),
    ]))
}

fn parse_records(text: &str) -> Result<Vec<CsvRecord>, String> {
    let mut records = Vec::new();
    let mut chars = text.chars().peekable();
    while chars.peek().is_some() {
        let row = records.len() + 1;
        records.push(parse_record(&mut chars, row)?);
    }
    Ok(records)
}

fn parse_record(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    row: usize,
) -> Result<CsvRecord, String> {
    let mut fields: Vec<String> = Vec::new();
    let mut raw_empty = true;
    loop {
        let column = fields.len() + 1;
        let (field, was_quoted) = parse_field(chars, row, column)?;
        if was_quoted || !field.is_empty() {
            raw_empty = false;
        }
        fields.push(field);
        match chars.peek().copied() {
            Some(',') => {
                chars.next();
                raw_empty = false;
            }
            Some('\n') => {
                chars.next();
                break;
            }
            Some('\r') => {
                chars.next();
                if chars.peek() == Some(&'\n') {
                    chars.next();
                    break;
                }
                return Err(format!(
                    "parse_csv: row {row}, column {column}: bare carriage return (CR \
                     without LF); only LF or CRLF row separators are supported"
                ));
            }
            None => break,
            // `parse_field` only stops at a comma, CR, LF, or end of input
            // (a closing quote followed by anything else is rejected there).
            Some(other) => {
                return Err(format!(
                    "parse_csv: row {row}, column {column}: unexpected character `{other}` \
                     after field"
                ));
            }
        }
    }
    Ok(CsvRecord { fields, raw_empty })
}

/// Parse one field, returning `(content, was_quoted)`. Stops at (without
/// consuming) the field's terminator: comma, CR, LF, or end of input.
fn parse_field(
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    row: usize,
    column: usize,
) -> Result<(String, bool), String> {
    let mut out = String::new();
    if chars.peek() == Some(&'"') {
        chars.next();
        loop {
            match chars.next() {
                None => {
                    return Err(format!(
                        "parse_csv: row {row}, column {column}: unclosed quoted field \
                         (missing closing `\"` before end of input)"
                    ));
                }
                Some('"') => {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        out.push('"');
                        continue;
                    }
                    return match chars.peek().copied() {
                        Some(',') | Some('\n') | Some('\r') | None => Ok((out, true)),
                        Some(other) => Err(format!(
                            "parse_csv: row {row}, column {column}: unexpected character \
                             `{other}` after closing quote (a quoted field must be \
                             followed by a comma or end of row)"
                        )),
                    };
                }
                // Commas, CRs, and LFs are literal inside quotes.
                Some(ch) => out.push(ch),
            }
        }
    }
    loop {
        match chars.peek().copied() {
            Some(',') | Some('\n') | Some('\r') | None => return Ok((out, false)),
            Some('"') => {
                return Err(format!(
                    "parse_csv: row {row}, column {column}: bare `\"` inside an unquoted \
                     field (quote the whole field and double embedded quotes)"
                ));
            }
            Some(ch) => {
                chars.next();
                out.push(ch);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Document access
// ---------------------------------------------------------------------------

/// A borrowed view of a Csv document's `columns` and `rows`.
struct CsvDoc<'v> {
    columns: Vec<&'v str>,
    rows: &'v [RuntimeValue],
}

/// The prelude Json ADT's constructor vocabulary, for diagnostics that
/// must distinguish "a Json value of the wrong kind" from "not a Json
/// value at all" (chelis#903 review).
const JSON_CTORS: [&str; 6] = ["JNull", "JBool", "JNum", "JStr", "JList", "JDict"];

fn as_jlist(value: &RuntimeValue) -> Option<&[RuntimeValue]> {
    match value {
        RuntimeValue::Adt { ctor, fields, .. } if ctor == "JList" => match fields.as_slice() {
            [RuntimeValue::List(items)] => Some(items),
            _ => None,
        },
        _ => None,
    }
}

fn as_jstr(value: &RuntimeValue) -> Option<&str> {
    match value {
        RuntimeValue::Adt { ctor, fields, .. } if ctor == "JStr" => match fields.as_slice() {
            [RuntimeValue::String(s)] => Some(s),
            _ => None,
        },
        _ => None,
    }
}

fn as_jdict_entries(value: &RuntimeValue) -> Option<&[(RuntimeValue, RuntimeValue)]> {
    match value {
        RuntimeValue::Adt { ctor, fields, .. } if ctor == "JDict" => match fields.as_slice() {
            [RuntimeValue::Dict(entries)] => Some(entries),
            _ => None,
        },
        _ => None,
    }
}

/// Validate the fixed document shape and borrow its parts. Shape
/// validation is eager and uniform across the whole accessor family
/// (chelis#903 review): `columns` must be a JList of UNIQUE strings and
/// every `rows` element a JDict, so a hand-mangled document fails
/// identically at `csv_nrows` and `csv_f64s` alike; cell-level types stay
/// lazily checked at read. Every failure names the builtin and says what
/// a Csv document is, so a plain `parse_json` value fails loudly instead
/// of producing garbage.
fn csv_doc<'v>(builtin: &str, value: &'v RuntimeValue) -> Result<CsvDoc<'v>, String> {
    let shape_err = |detail: &str| {
        format!(
            "{builtin}: expected a Csv document (the `{{\"columns\": .., \"rows\": ..}}` \
             JDict that parse_csv returns); {detail}"
        )
    };
    let Some(entries) = as_jdict_entries(value) else {
        return Err(match value {
            RuntimeValue::Adt { ctor, .. } if ctor == "JDict" => {
                shape_err("this JDict value has a malformed payload")
            }
            RuntimeValue::Adt { ctor, .. } if JSON_CTORS.contains(&ctor.as_str()) => {
                shape_err(&format!("got a Json `{ctor}` value, not a JDict"))
            }
            RuntimeValue::Adt { ctor, .. } => {
                shape_err(&format!("got a non-Json value (constructor `{ctor}`)"))
            }
            other => shape_err(&format!("got a non-Json value: {}", truncated_debug(other))),
        });
    };
    let lookup = |key: &str| {
        entries.iter().find_map(|(k, item)| match k {
            RuntimeValue::String(s) if s == key => Some(item),
            _ => None,
        })
    };
    let columns_value =
        lookup("columns").ok_or_else(|| shape_err("this JDict has no `columns` key"))?;
    let rows_value = lookup("rows").ok_or_else(|| shape_err("this JDict has no `rows` key"))?;
    let columns = as_jlist(columns_value)
        .ok_or_else(|| shape_err("`columns` must be a JList of strings"))?
        .iter()
        .map(|item| as_jstr(item).ok_or_else(|| shape_err("`columns` must contain only strings")))
        .collect::<Result<Vec<&str>, String>>()?;
    for (index, name) in columns.iter().enumerate() {
        if let Some(first) = columns[..index].iter().position(|other| other == name) {
            return Err(format!(
                "{builtin}: duplicate column `{name}` in `columns` (columns {} and {}); \
                 columns must be addressable by unique names",
                first + 1,
                index + 1
            ));
        }
    }
    let rows = as_jlist(rows_value).ok_or_else(|| shape_err("`rows` must be a JList"))?;
    for (row_idx, row) in rows.iter().enumerate() {
        if as_jdict_entries(row).is_none() {
            return Err(format!(
                "{builtin}: data row {row_idx} (0-based) is not a JDict of \
                 column -> cell; this is not a Csv document from parse_csv"
            ));
        }
    }
    Ok(CsvDoc { columns, rows })
}

/// `... available columns: `a`, `b`, ...` — the same courtesy the JSON
/// accessors extend for missing dict keys (chelis#890).
fn available_columns(columns: &[&str]) -> String {
    const MAX_LISTED: usize = 12;
    let mut names: Vec<String> = columns
        .iter()
        .take(MAX_LISTED)
        .map(|name| format!("`{name}`"))
        .collect();
    let elided = columns.len().saturating_sub(MAX_LISTED);
    if elided > 0 {
        names.push(format!("... {elided} more"));
    }
    names.join(", ")
}

/// Resolve a column name to its position in `columns`, or fail listing
/// the available columns. The returned index feeds [`cell`]'s O(1)
/// positional probe.
fn require_column(builtin: &str, doc: &CsvDoc<'_>, column: &str) -> Result<usize, String> {
    doc.columns
        .iter()
        .position(|name| *name == column)
        .ok_or_else(|| {
            format!(
                "{builtin}: column `{column}` not found; available columns: {}",
                available_columns(&doc.columns)
            )
        })
}

/// Shared context prefix for cell-level diagnostics. Accessor errors
/// speak 0-based data-row indices (the `csv_f64`/`csv_str` argument
/// space); parse errors speak 1-based physical records — the qualifier
/// keeps the two spaces distinguishable (chelis#903 review).
fn cell_context(builtin: &str, column: &str, row_idx: usize) -> String {
    format!("{builtin}: column `{column}`, data row {row_idx} (0-based)")
}

fn row_entries<'v>(
    builtin: &str,
    row: &'v RuntimeValue,
    row_idx: usize,
) -> Result<&'v [(RuntimeValue, RuntimeValue)], String> {
    as_jdict_entries(row).ok_or_else(|| {
        format!(
            "{builtin}: data row {row_idx} (0-based) is not a JDict of column -> cell; \
             this is not a Csv document from parse_csv"
        )
    })
}

/// Fetch one cell. `col_idx` is the column's position in `columns`:
/// `parse_csv` builds every row positionally aligned with the header, so
/// the O(1) positional probe hits and a column extraction is linear in
/// the row count; hand-assembled rows (arbitrary key order) fall back to
/// a by-name scan (chelis#903 review).
fn cell<'v>(
    builtin: &str,
    row: &'v RuntimeValue,
    row_idx: usize,
    column: &str,
    col_idx: usize,
) -> Result<&'v RuntimeValue, String> {
    let entries = row_entries(builtin, row, row_idx)?;
    if let Some((key, item)) = entries.get(col_idx)
        && matches!(key, RuntimeValue::String(k) if k == column)
    {
        return Ok(item);
    }
    entries
        .iter()
        .find_map(|(key, item)| match key {
            RuntimeValue::String(s) if s == column => Some(item),
            _ => None,
        })
        .ok_or_else(|| {
            format!(
                "{}: row has no cell for this column (malformed Csv document)",
                cell_context(builtin, column, row_idx)
            )
        })
}

/// One-phrase description of a cell for diagnostics. Foreign constructors
/// are labeled as non-Json rather than mislabeled as Json values
/// (chelis#903 review).
fn cell_kind(node: &RuntimeValue) -> String {
    match node {
        RuntimeValue::Adt { ctor, .. } => match ctor.as_str() {
            "JNull" => "null".to_string(),
            "JBool" => "a bool".to_string(),
            "JNum" => "a number".to_string(),
            "JStr" => "a string".to_string(),
            "JList" => "a list".to_string(),
            "JDict" => "a dict".to_string(),
            other => format!("not a Json value (constructor `{other}`)"),
        },
        other => format!("not a Json value ({})", truncated_debug(other)),
    }
}

/// Read a string cell: `JStr` only, verbatim. Every other cell type is a
/// loud per-type error — no cross-type coercion: `JNum` points at
/// csv_f64, and `JBool`/`JNull` are to_csv's write-side cell encodings,
/// not readable text (chelis#903 review).
fn cell_str<'v>(
    builtin: &str,
    row: &'v RuntimeValue,
    row_idx: usize,
    column: &str,
    col_idx: usize,
) -> Result<&'v str, String> {
    let node = cell(builtin, row, row_idx, column, col_idx)?;
    if let Some(text) = as_jstr(node) {
        return Ok(text);
    }
    let ctx = cell_context(builtin, column, row_idx);
    Err(match node.as_adt() {
        Some(("JNum", _)) => {
            format!("{ctx}: cell is a number, not a string; read it with csv_f64/csv_f64s")
        }
        Some(("JBool", _)) => {
            format!("{ctx}: cell is a bool, not a string (no silent true/false-to-text coercion)")
        }
        Some(("JNull", _)) => {
            format!("{ctx}: cell is null, not a string (null is to_csv's empty-cell encoding)")
        }
        _ => format!("{ctx}: cell is {}, not a string", cell_kind(node)),
    })
}

/// Strict JSON number grammar — literally the shared `parse_json` scanner
/// ([`super::json::json_number_token_len`]) required to consume the whole
/// cell, so "the same grammar as JSON" holds by shared code. Notably
/// rejects: empty, `inf`/`NaN`, hex, leading `+`, leading zeros (`007`),
/// bare `.5`/`1.`, and separators (`1_000`, `1,000`).
fn is_strict_json_number(text: &str) -> bool {
    matches!(
        super::json::json_number_token_len(text.as_bytes()),
        Ok(len) if len == text.len()
    )
}

/// Convert one cell's text to f64: strict JSON number grammar after
/// trimming surrounding ASCII spaces/tabs (the one tolerance, matching
/// Python `float()`; the digits themselves stay strict). Empty cells and
/// non-numbers are loud errors naming the column, the 0-based data row,
/// and the offending text — never a silent NaN.
fn parse_cell_number(
    builtin: &str,
    column: &str,
    row_idx: usize,
    raw: &str,
) -> Result<f64, String> {
    let ctx = || cell_context(builtin, column, row_idx);
    let trimmed = raw.trim_matches(|c| c == ' ' || c == '\t');
    if trimmed.is_empty() {
        return Err(format!(
            "{}: cell is empty (no silent NaN/defaults; read optional columns with \
             csv_str/csv_strs)",
            ctx()
        ));
    }
    if !is_strict_json_number(trimmed) {
        return Err(format!(
            "{}: cell `{raw}` is not a number (strict JSON number syntax; surrounding \
             spaces/tabs tolerated)",
            ctx()
        ));
    }
    let value: f64 = trimmed
        .parse()
        .map_err(|err| format!("{}: internal error parsing `{trimmed}`: {err}", ctx()))?;
    if !value.is_finite() {
        return Err(format!("{}: number `{raw}` overflows f64", ctx()));
    }
    Ok(value)
}

/// Read a numeric cell: a `JStr` cell converts under
/// [`parse_cell_number`]; a finite `JNum` cell (hand-assembled documents)
/// is read directly — a non-finite `JNum` fails loudly, matching the
/// to_csv write-side contract (no silent NaN, chelis#903 review).
/// `JBool`/`JNull` are loud per-type errors, consistent with [`cell_str`].
fn cell_f64(
    builtin: &str,
    row: &RuntimeValue,
    row_idx: usize,
    column: &str,
    col_idx: usize,
) -> Result<f64, String> {
    let node = cell(builtin, row, row_idx, column, col_idx)?;
    let ctx = || cell_context(builtin, column, row_idx);
    match node.as_adt() {
        Some(("JNum", [RuntimeValue::Scalar(payload)])) if payload.dtype().is_float() => {
            let value = payload.bits().as_f64();
            if !value.is_finite() {
                return Err(format!(
                    "{}: cell is the non-finite number `{value}` (CSV cannot represent \
                     non-finite numbers; no silent NaN/defaults)",
                    ctx()
                ));
            }
            Ok(value)
        }
        Some(("JNum", fields)) => Err(format!(
            "{}: malformed JNum cell {}",
            ctx(),
            truncate_rendered(format!("{fields:?}"))
        )),
        Some(("JStr", [RuntimeValue::String(text)])) => {
            parse_cell_number(builtin, column, row_idx, text)
        }
        Some(("JStr", fields)) => Err(format!(
            "{}: malformed JStr cell {}",
            ctx(),
            truncate_rendered(format!("{fields:?}"))
        )),
        Some(("JBool", _)) => Err(format!(
            "{}: cell is a bool, not a number (no silent true/false-to-number coercion)",
            ctx()
        )),
        Some(("JNull", _)) => Err(format!(
            "{}: cell is null (no silent NaN/defaults; null is to_csv's empty-cell \
             encoding)",
            ctx()
        )),
        _ => Err(format!(
            "{}: cell is {}, not a number",
            ctx(),
            cell_kind(node)
        )),
    }
}

fn require_row_index<'v>(
    builtin: &str,
    doc: &CsvDoc<'v>,
    row_idx: i64,
) -> Result<(usize, &'v RuntimeValue), String> {
    let count = doc.rows.len();
    let index = usize::try_from(row_idx).ok().filter(|i| *i < count);
    match index {
        Some(i) => Ok((i, &doc.rows[i])),
        None => Err(format!(
            "{builtin}: row index {row_idx} out of range ({count} data rows; row \
             indices are 0-based)"
        )),
    }
}

// ---------------------------------------------------------------------------
// Accessors (the eval.rs surface)
// ---------------------------------------------------------------------------

pub(super) fn csv_cols_of(value: &RuntimeValue) -> Result<Vec<String>, String> {
    let doc = csv_doc("csv_cols", value)?;
    Ok(doc.columns.iter().map(|name| name.to_string()).collect())
}

pub(super) fn csv_nrows_of(value: &RuntimeValue) -> Result<i64, String> {
    let doc = csv_doc("csv_nrows", value)?;
    i64::try_from(doc.rows.len())
        .map_err(|_| format!("csv_nrows: row count {} overflows int64", doc.rows.len()))
}

pub(super) fn csv_strs_at(value: &RuntimeValue, column: &str) -> Result<Vec<String>, String> {
    let doc = csv_doc("csv_strs", value)?;
    let col_idx = require_column("csv_strs", &doc, column)?;
    doc.rows
        .iter()
        .enumerate()
        .map(|(row_idx, row)| {
            cell_str("csv_strs", row, row_idx, column, col_idx).map(|text| text.to_string())
        })
        .collect()
}

pub(super) fn csv_f64s_at(value: &RuntimeValue, column: &str) -> Result<Vec<f64>, String> {
    let doc = csv_doc("csv_f64s", value)?;
    let col_idx = require_column("csv_f64s", &doc, column)?;
    doc.rows
        .iter()
        .enumerate()
        .map(|(row_idx, row)| cell_f64("csv_f64s", row, row_idx, column, col_idx))
        .collect()
}

pub(super) fn csv_str_at(
    value: &RuntimeValue,
    row_idx: i64,
    column: &str,
) -> Result<String, String> {
    let doc = csv_doc("csv_str", value)?;
    let col_idx = require_column("csv_str", &doc, column)?;
    let (index, row) = require_row_index("csv_str", &doc, row_idx)?;
    cell_str("csv_str", row, index, column, col_idx).map(|text| text.to_string())
}

pub(super) fn csv_f64_at(value: &RuntimeValue, row_idx: i64, column: &str) -> Result<f64, String> {
    let doc = csv_doc("csv_f64", value)?;
    let col_idx = require_column("csv_f64", &doc, column)?;
    let (index, row) = require_row_index("csv_f64", &doc, row_idx)?;
    cell_f64("csv_f64", row, index, column, col_idx)
}

// ---------------------------------------------------------------------------
// Serialization (to_csv)
// ---------------------------------------------------------------------------

/// Append one field with minimal RFC 4180 quoting: quoted only when the
/// text contains a comma, quote, CR, or LF (embedded quotes doubled), or
/// when `force_quote` demands it — the empty sole field of a
/// single-column row (unquoted it would read back as a blank line), and a
/// first header field starting with U+FEFF (unquoted it would sit at the
/// start of the output text and parse_csv's BOM strip would silently
/// rename the column on re-read).
fn write_csv_field(out: &mut String, text: &str, force_quote: bool) {
    let needs_quote = force_quote
        || text.contains(',')
        || text.contains('"')
        || text.contains('\n')
        || text.contains('\r');
    if !needs_quote {
        out.push_str(text);
        return;
    }
    out.push('"');
    for ch in text.chars() {
        if ch == '"' {
            out.push('"');
        }
        out.push(ch);
    }
    out.push('"');
}

/// Render one cell of a document to CSV field text. `JStr` is verbatim;
/// `JNum` uses the shared shortest-round-trip formatter
/// ([`super::json::format_f64_json`] — the same fidelity contract as
/// `to_json`, never the print channel: chelis#748/#723/#734); `JBool` is
/// `true`/`false`; `JNull` is the empty cell. Containers and non-finite
/// numbers fail loudly.
fn cell_to_field_text(column: &str, row_idx: usize, node: &RuntimeValue) -> Result<String, String> {
    let ctx = || cell_context("to_csv", column, row_idx);
    let Some((ctor, fields)) = node.as_adt() else {
        return Err(format!(
            "{}: cell is not a Json value ({})",
            ctx(),
            truncated_debug(node)
        ));
    };
    match (ctor, fields) {
        ("JStr", [RuntimeValue::String(s)]) => Ok(s.clone()),
        ("JNum", [RuntimeValue::Scalar(payload)]) if payload.dtype().is_float() => {
            let value = payload.bits().as_f64();
            if !value.is_finite() {
                return Err(format!(
                    "{}: cannot represent non-finite number `{value}` in CSV",
                    ctx()
                ));
            }
            Ok(super::json::format_f64_json(value).expect("finite f64 always formats"))
        }
        ("JBool", [RuntimeValue::Bool(b)]) => Ok(if *b { "true" } else { "false" }.to_string()),
        ("JNull", []) => Ok(String::new()),
        ("JList" | "JDict", _) => Err(format!(
            "{}: cell is a {}; CSV cells must be scalars (string, number, bool, or null)",
            ctx(),
            if ctor == "JList" { "list" } else { "dict" }
        )),
        _ => Err(format!(
            "{}: malformed Json cell (constructor `{ctor}` with fields {})",
            ctx(),
            truncate_rendered(format!("{fields:?}"))
        )),
    }
}

/// Serialize a Csv document (the exact shape `parse_csv` returns) to CSV
/// text: header row from `columns`, one row per `rows` entry, minimal
/// quoting, LF row separators, trailing final newline. Deterministic and
/// byte-stable. Loud errors, no silent data loss anywhere: unexpected
/// top-level keys (e.g. a `json_set`-added subtree), rows missing a
/// declared column, rows carrying an undeclared or duplicate key, and
/// non-scalar cells all fail by name (chelis#903 review).
pub(super) fn csv_to_text(value: &RuntimeValue) -> Result<String, String> {
    let doc = csv_doc("to_csv", value)?;
    // Only `columns` and `rows` serialize; silently ignoring any other
    // top-level entry (a json_set-added `meta` subtree, say) would be
    // exactly the silent data loss this surface exists to kill. Read
    // accessors tolerate extra keys — reads drop nothing — but the write
    // path must not.
    for (key, _) in as_jdict_entries(value).expect("csv_doc validated the JDict") {
        match key {
            RuntimeValue::String(key) if key == "columns" || key == "rows" => {}
            RuntimeValue::String(key) => {
                return Err(format!(
                    "to_csv: document has unexpected top-level key `{key}` (only \
                     `columns` and `rows` serialize; refusing to silently drop data)"
                ));
            }
            other => {
                return Err(format!(
                    "to_csv: document has a non-string top-level key {}",
                    truncated_debug(other)
                ));
            }
        }
    }
    if doc.columns.is_empty() {
        return Err("to_csv: `columns` is empty (a CSV needs at least one column)".to_string());
    }
    let single_column = doc.columns.len() == 1;
    let mut out = String::new();
    for (index, name) in doc.columns.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let force_quote =
            (single_column && name.is_empty()) || (index == 0 && name.starts_with('\u{feff}'));
        write_csv_field(&mut out, name, force_quote);
    }
    out.push('\n');
    for (row_idx, row) in doc.rows.iter().enumerate() {
        let entries = row_entries("to_csv", row, row_idx)?;
        // Refuse to silently drop or arbitrarily pick data: every key in
        // the row must be a declared column, at most once (a declared
        // column missing from the row fails in the emit loop below).
        let mut seen: Vec<&str> = Vec::with_capacity(entries.len());
        for (key, _) in entries {
            let RuntimeValue::String(key) = key else {
                return Err(format!(
                    "to_csv: data row {row_idx} (0-based) has a non-string key {}",
                    truncated_debug(key)
                ));
            };
            if !doc.columns.contains(&key.as_str()) {
                return Err(format!(
                    "to_csv: data row {row_idx} (0-based) has key `{key}` that is not \
                     in `columns`; refusing to silently drop data"
                ));
            }
            if seen.contains(&key.as_str()) {
                return Err(format!(
                    "to_csv: data row {row_idx} (0-based) has duplicate key `{key}`; \
                     refusing to silently pick one of its values"
                ));
            }
            seen.push(key);
        }
        for (index, column) in doc.columns.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            let node = cell("to_csv", row, row_idx, column, index)?;
            let text = cell_to_field_text(column, row_idx, node)?;
            write_csv_field(&mut out, &text, single_column && text.is_empty());
        }
        out.push('\n');
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::json::{jnum, json_value_to_text};

    fn doc(text: &str) -> RuntimeValue {
        parse_csv_text(text).expect("parse_csv should succeed")
    }

    fn err(text: &str) -> String {
        parse_csv_text(text).expect_err("parse_csv should fail")
    }

    // -- parser: happy paths ------------------------------------------------

    #[test]
    fn parse_basic_header_and_rows() {
        let c = doc("id,qty,px\nalpha,2,101.5\nbeta,3,99.25\n");
        assert_eq!(csv_cols_of(&c).unwrap(), vec!["id", "qty", "px"]);
        assert_eq!(csv_nrows_of(&c).unwrap(), 2);
        assert_eq!(csv_strs_at(&c, "id").unwrap(), vec!["alpha", "beta"]);
        assert_eq!(csv_f64s_at(&c, "px").unwrap(), vec![101.5, 99.25]);
        assert_eq!(csv_f64_at(&c, 1, "qty").unwrap(), 3.0);
        assert_eq!(csv_str_at(&c, 0, "id").unwrap(), "alpha");
    }

    #[test]
    fn parse_without_trailing_newline() {
        let c = doc("a,b\n1,2");
        assert_eq!(csv_nrows_of(&c).unwrap(), 1);
        assert_eq!(csv_f64_at(&c, 0, "b").unwrap(), 2.0);
    }

    #[test]
    fn parse_quoted_fields_embedded_comma_quote_newline() {
        let c = doc("name,note\n\"alpha, senior\",\"says \"\"hi\"\"\"\n\"line1\nline2\",plain\n");
        assert_eq!(
            csv_strs_at(&c, "name").unwrap(),
            vec!["alpha, senior", "line1\nline2"]
        );
        assert_eq!(
            csv_strs_at(&c, "note").unwrap(),
            vec!["says \"hi\"", "plain"]
        );
    }

    #[test]
    fn parse_crlf_and_mixed_line_endings() {
        let c = doc("a,b\r\n1,2\n3,4\r\n");
        assert_eq!(csv_nrows_of(&c).unwrap(), 2);
        assert_eq!(csv_f64s_at(&c, "a").unwrap(), vec![1.0, 3.0]);
        // CRLF inside quotes is literal content, not a row separator.
        let quoted = doc("a\n\"x\r\ny\"\n");
        assert_eq!(csv_str_at(&quoted, 0, "a").unwrap(), "x\r\ny");
    }

    #[test]
    fn parse_strips_utf8_bom() {
        let c = doc("\u{feff}date,px\n2020-01-02,1.5\n");
        assert_eq!(csv_cols_of(&c).unwrap(), vec!["date", "px"]);
        assert_eq!(csv_f64_at(&c, 0, "px").unwrap(), 1.5);
    }

    #[test]
    fn parse_header_only_preserves_columns_for_zero_rows() {
        let c = doc("id,qty,px\n");
        assert_eq!(csv_cols_of(&c).unwrap(), vec!["id", "qty", "px"]);
        assert_eq!(csv_nrows_of(&c).unwrap(), 0);
        assert_eq!(csv_f64s_at(&c, "px").unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn parse_tolerates_trailing_blank_lines() {
        let c = doc("a,b\n1,2\n\n\r\n");
        assert_eq!(csv_nrows_of(&c).unwrap(), 1);
    }

    #[test]
    fn parse_allows_empty_header_name_pandas_index_column() {
        // pandas `to_csv` with a written index emits an unnamed first
        // column; the empty name becomes the key "".
        let c = doc(",id,px\n0,alpha,1.5\n");
        assert_eq!(csv_cols_of(&c).unwrap(), vec!["", "id", "px"]);
        assert_eq!(csv_str_at(&c, 0, "").unwrap(), "0");
        assert_eq!(csv_f64_at(&c, 0, "px").unwrap(), 1.5);
    }

    #[test]
    fn parse_single_column_quoted_empty_cell_is_a_row() {
        let c = doc("x\n\"\"\n1\n");
        assert_eq!(csv_nrows_of(&c).unwrap(), 2);
        assert_eq!(csv_str_at(&c, 0, "x").unwrap(), "");
        assert_eq!(csv_str_at(&c, 1, "x").unwrap(), "1");
    }

    #[test]
    fn parse_unicode_cells_survive() {
        let c = doc("ccy,label\n€,naïve ✓\n");
        assert_eq!(csv_str_at(&c, 0, "ccy").unwrap(), "€");
        assert_eq!(csv_str_at(&c, 0, "label").unwrap(), "naïve ✓");
    }

    // -- parser: malformed taxonomy ----------------------------------------

    #[test]
    fn parse_empty_input_fails() {
        let e = err("");
        assert!(e.contains("input is empty"), "got `{e}`");
        let e = err("\n\n");
        assert!(e.contains("input is empty"), "got `{e}`");
    }

    #[test]
    fn parse_unclosed_quote_names_row_and_column() {
        let e = err("a,b\n1,\"oops\n");
        assert!(e.contains("row 2, column 2"), "got `{e}`");
        assert!(e.contains("unclosed quoted field"), "got `{e}`");
    }

    #[test]
    fn parse_content_after_closing_quote_fails() {
        let e = err("a,b\n\"x\"y,2\n");
        assert!(e.contains("row 2, column 1"), "got `{e}`");
        assert!(e.contains("after closing quote"), "got `{e}`");
    }

    #[test]
    fn parse_bare_quote_in_unquoted_field_fails() {
        let e = err("a,b\n1,mid\"dle\n");
        assert!(e.contains("row 2, column 2"), "got `{e}`");
        assert!(
            e.contains("bare `\"` inside an unquoted field"),
            "got `{e}`"
        );
    }

    #[test]
    fn parse_bare_carriage_return_fails() {
        // The CR follows field 2 (`b`), so the error carries both the row
        // and the column ordinal (chelis#903 review).
        let e = err("a,b\r1,2\r");
        assert!(e.contains("row 1, column 2"), "got `{e}`");
        assert!(e.contains("bare carriage return"), "got `{e}`");
    }

    #[test]
    fn parse_ragged_row_names_row_and_counts() {
        let e = err("a,b,c\n1,2,3\n4,5\n");
        assert!(e.contains("row 3 has 2 fields, expected 3"), "got `{e}`");
    }

    #[test]
    fn parse_duplicate_header_fails_with_positions() {
        let e = err("px,id,px\n1,a,2\n");
        assert!(e.contains("duplicate header column `px`"), "got `{e}`");
        assert!(e.contains("columns 1 and 3"), "got `{e}`");
    }

    #[test]
    fn parse_interior_blank_line_fails() {
        let e = err("a,b\n1,2\n\n3,4\n");
        assert!(e.contains("row 3 is blank"), "got `{e}`");
    }

    #[test]
    fn parse_blank_header_row_fails() {
        let e = err("\na,b\n1,2\n");
        assert!(e.contains("row 1 is blank"), "got `{e}`");
    }

    // -- numeric accessor strictness ---------------------------------------

    #[test]
    fn csv_f64s_strict_grammar_and_exactness() {
        let c = doc("v\n0.1\n-2.5e-3\n 1.5\t\n42\n");
        // Whitespace-trimmed, otherwise strict; values are exact f64.
        assert_eq!(csv_f64s_at(&c, "v").unwrap(), vec![0.1, -2.5e-3, 1.5, 42.0]);
    }

    #[test]
    fn csv_f64_rejects_non_numbers_naming_column_row_text() {
        for bad in [
            "N/A", "007", "1,000", "inf", "NaN", "0x10", "1.", ".5", "+1", "1_000",
        ] {
            let quoted = if bad.contains(',') {
                format!("\"{bad}\"")
            } else {
                bad.to_string()
            };
            let c = doc(&format!("px\n{quoted}\n"));
            let e = csv_f64_at(&c, 0, "px").expect_err(bad);
            assert!(e.contains("csv_f64"), "{bad}: got `{e}`");
            assert!(e.contains("column `px`"), "{bad}: got `{e}`");
            assert!(e.contains("data row 0"), "{bad}: got `{e}`");
            assert!(e.contains(&format!("cell `{bad}`")), "{bad}: got `{e}`");
            assert!(e.contains("is not a number"), "{bad}: got `{e}`");
        }
    }

    #[test]
    fn csv_f64_empty_cell_is_a_loud_error() {
        let c = doc("px,id\n,alpha\n");
        let e = csv_f64_at(&c, 0, "px").expect_err("empty cell");
        assert!(e.contains("column `px`"), "got `{e}`");
        assert!(e.contains("cell is empty"), "got `{e}`");
        assert!(e.contains("no silent NaN"), "got `{e}`");
    }

    #[test]
    fn csv_f64_overflowing_number_fails() {
        let c = doc("px\n1e999\n");
        let e = csv_f64_at(&c, 0, "px").expect_err("overflow");
        assert!(e.contains("overflows f64"), "got `{e}`");
    }

    // -- accessor error paths ----------------------------------------------

    #[test]
    fn missing_column_lists_available_columns() {
        let c = doc("date,mid,qty\n2020-01-02,1.5,3\n");
        for (name, e) in [
            ("csv_f64s", csv_f64s_at(&c, "px").expect_err("missing")),
            ("csv_strs", csv_strs_at(&c, "px").expect_err("missing")),
            ("csv_f64", csv_f64_at(&c, 0, "px").expect_err("missing")),
            ("csv_str", csv_str_at(&c, 0, "px").expect_err("missing")),
        ] {
            assert!(e.contains(name), "{name}: got `{e}`");
            assert!(e.contains("column `px` not found"), "{name}: got `{e}`");
            assert!(
                e.contains("available columns: `date`, `mid`, `qty`"),
                "{name}: got `{e}`"
            );
        }
    }

    #[test]
    fn row_index_out_of_range_is_loud() {
        let c = doc("a\n1\n2\n");
        for idx in [2_i64, -1, 100] {
            let e = csv_f64_at(&c, idx, "a").expect_err("out of range");
            assert!(
                e.contains(&format!("row index {idx} out of range")),
                "got `{e}`"
            );
            assert!(e.contains("2 data rows"), "got `{e}`");
        }
    }

    #[test]
    fn accessors_reject_non_csv_json_values() {
        // A JSON document without the Csv shape...
        let json = crate::runtime::json::parse_json_text("{\"a\": [1.5]}").unwrap();
        let e = csv_nrows_of(&json).expect_err("not a Csv doc");
        assert!(e.contains("csv_nrows"), "got `{e}`");
        assert!(e.contains("expected a Csv document"), "got `{e}`");
        assert!(e.contains("no `columns` key"), "got `{e}`");
        // ... a Json scalar ...
        let e = csv_cols_of(&jnum(1.5)).expect_err("scalar");
        assert!(e.contains("Json `JNum` value"), "got `{e}`");
        // ... and a non-Json runtime value.
        let e = csv_f64s_at(&RuntimeValue::String("raw".to_string()), "a").expect_err("string");
        assert!(e.contains("non-Json value"), "got `{e}`");
    }

    #[test]
    fn csv_f64_reads_jnum_cells_in_assembled_documents() {
        let c = jdict(vec![
            ("columns".to_string(), jlist(vec![jstr("pv".to_string())])),
            (
                "rows".to_string(),
                jlist(vec![jdict(vec![("pv".to_string(), jnum(2.5))])]),
            ),
        ]);
        assert_eq!(csv_f64_at(&c, 0, "pv").unwrap(), 2.5);
        let e = csv_str_at(&c, 0, "pv").expect_err("number cell as string");
        assert!(e.contains("cell is a number"), "got `{e}`");
        assert!(e.contains("csv_f64"), "suggests the right accessor: `{e}`");
    }

    // -- to_csv -------------------------------------------------------------

    #[test]
    fn to_csv_emits_minimal_quoting_and_trailing_newline() {
        let c = doc("name,note\n\"alpha, senior\",plain\nbeta,\"says \"\"hi\"\"\"\n");
        assert_eq!(
            csv_to_text(&c).unwrap(),
            "name,note\n\"alpha, senior\",plain\nbeta,\"says \"\"hi\"\"\"\n"
        );
    }

    #[test]
    fn to_csv_parse_csv_round_trips_documents() {
        for text in [
            "id,qty,px\nalpha,2,101.5\nbeta,3,99.25\n",
            "name,note\n\"alpha, senior\",\"says \"\"hi\"\"\"\n\"l1\nl2\",x\n",
            "x\n\"\"\n1\n",
            "id,qty,px\n",
        ] {
            let original = doc(text);
            let reparsed = doc(&csv_to_text(&original).unwrap());
            assert_eq!(
                json_value_to_text(&reparsed).unwrap(),
                json_value_to_text(&original).unwrap(),
                "round-trip failed for {text:?}"
            );
        }
    }

    #[test]
    fn to_csv_numbers_are_shortest_round_trip_exact() {
        let values = [0.1_f64, 1e-9, 0.123_456_789_012_345_68, 28125.01875, -0.0];
        let rows = values
            .iter()
            .map(|v| jdict(vec![("pv".to_string(), jnum(*v))]))
            .collect();
        let c = jdict(vec![
            ("columns".to_string(), jlist(vec![jstr("pv".to_string())])),
            ("rows".to_string(), jlist(rows)),
        ]);
        let text = csv_to_text(&c).unwrap();
        let read_back = csv_f64s_at(&doc(&text), "pv").unwrap();
        for (expected, got) in values.iter().zip(&read_back) {
            assert_eq!(
                expected.to_bits(),
                got.to_bits(),
                "f64 must survive to_csv -> parse_csv bit-for-bit"
            );
        }
    }

    #[test]
    fn to_csv_bool_and_null_cells() {
        let c = jdict(vec![
            (
                "columns".to_string(),
                jlist(vec![jstr("flag".to_string()), jstr("note".to_string())]),
            ),
            (
                "rows".to_string(),
                jlist(vec![jdict(vec![
                    (
                        "flag".to_string(),
                        RuntimeValue::Adt {
                            ctor: "JBool".to_string(),
                            fields: vec![RuntimeValue::Bool(true)],
                            field_names: None,
                        },
                    ),
                    (
                        "note".to_string(),
                        RuntimeValue::Adt {
                            ctor: "JNull".to_string(),
                            fields: vec![],
                            field_names: None,
                        },
                    ),
                ])]),
            ),
        ]);
        assert_eq!(csv_to_text(&c).unwrap(), "flag,note\ntrue,\n");
    }

    #[test]
    fn to_csv_non_finite_number_fails() {
        let c = jdict(vec![
            ("columns".to_string(), jlist(vec![jstr("pv".to_string())])),
            (
                "rows".to_string(),
                jlist(vec![jdict(vec![("pv".to_string(), jnum(f64::INFINITY))])]),
            ),
        ]);
        let e = csv_to_text(&c).expect_err("non-finite");
        assert!(e.contains("non-finite"), "got `{e}`");
        assert!(e.contains("column `pv`"), "got `{e}`");
    }

    #[test]
    fn to_csv_refuses_missing_and_undeclared_row_keys() {
        let missing = jdict(vec![
            (
                "columns".to_string(),
                jlist(vec![jstr("a".to_string()), jstr("b".to_string())]),
            ),
            (
                "rows".to_string(),
                jlist(vec![jdict(vec![("a".to_string(), jstr("1".to_string()))])]),
            ),
        ]);
        let e = csv_to_text(&missing).expect_err("missing column value");
        assert!(e.contains("column `b`"), "got `{e}`");
        assert!(e.contains("no cell for this column"), "got `{e}`");

        let extra = jdict(vec![
            ("columns".to_string(), jlist(vec![jstr("a".to_string())])),
            (
                "rows".to_string(),
                jlist(vec![jdict(vec![
                    ("a".to_string(), jstr("1".to_string())),
                    ("stray".to_string(), jstr("2".to_string())),
                ])]),
            ),
        ]);
        let e = csv_to_text(&extra).expect_err("undeclared key");
        assert!(e.contains("key `stray`"), "got `{e}`");
        assert!(e.contains("refusing to silently drop data"), "got `{e}`");
    }

    #[test]
    fn to_csv_container_cell_fails() {
        let c = jdict(vec![
            ("columns".to_string(), jlist(vec![jstr("a".to_string())])),
            (
                "rows".to_string(),
                jlist(vec![jdict(vec![("a".to_string(), jlist(vec![]))])]),
            ),
        ]);
        let e = csv_to_text(&c).expect_err("container cell");
        assert!(e.contains("cell is a list"), "got `{e}`");
        assert!(e.contains("must be scalars"), "got `{e}`");
    }

    #[test]
    fn to_csv_empty_or_duplicate_columns_fail() {
        let empty = jdict(vec![
            ("columns".to_string(), jlist(vec![])),
            ("rows".to_string(), jlist(vec![])),
        ]);
        let e = csv_to_text(&empty).expect_err("empty columns");
        assert!(e.contains("`columns` is empty"), "got `{e}`");

        let dup = jdict(vec![
            (
                "columns".to_string(),
                jlist(vec![jstr("a".to_string()), jstr("a".to_string())]),
            ),
            ("rows".to_string(), jlist(vec![])),
        ]);
        let e = csv_to_text(&dup).expect_err("duplicate columns");
        assert!(e.contains("duplicate column `a`"), "got `{e}`");
        assert!(e.contains("columns 1 and 2"), "got `{e}`");
    }

    #[test]
    fn to_csv_single_column_empty_cell_quotes_and_round_trips() {
        let c = doc("x\n\"\"\n1\n");
        let text = csv_to_text(&c).unwrap();
        assert_eq!(text, "x\n\"\"\n1\n");
        assert_eq!(csv_nrows_of(&doc(&text)).unwrap(), 2);
    }

    #[test]
    fn parse_csv_document_composes_with_json_accessors() {
        // The whole point of riding the Json ADT: #890's accessors work on
        // the document without any bridging.
        let c = doc("id,px\nalpha,1.5\n");
        let rows = crate::runtime::json::json_list_at(&c, "rows").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            crate::runtime::json::json_str_at(&c, "rows.0.id").unwrap(),
            "alpha"
        );
        assert_eq!(
            json_value_to_text(&c).unwrap(),
            r#"{"columns":["id","px"],"rows":[{"id":"alpha","px":"1.5"}]}"#
        );
    }

    // -- chelis#903 review probes -------------------------------------------

    fn jbool_cell(value: bool) -> RuntimeValue {
        RuntimeValue::Adt {
            ctor: "JBool".to_string(),
            fields: vec![RuntimeValue::Bool(value)],
            field_names: None,
        }
    }

    fn jnull_cell() -> RuntimeValue {
        RuntimeValue::Adt {
            ctor: "JNull".to_string(),
            fields: vec![],
            field_names: None,
        }
    }

    /// One-column document with one cell per row, for cell-type probes.
    fn one_col_doc(column: &str, cells: Vec<RuntimeValue>) -> RuntimeValue {
        jdict(vec![
            ("columns".to_string(), jlist(vec![jstr(column.to_string())])),
            (
                "rows".to_string(),
                jlist(
                    cells
                        .into_iter()
                        .map(|cell| jdict(vec![(column.to_string(), cell)]))
                        .collect(),
                ),
            ),
        ])
    }

    /// Review finding 1: a first header field starting with U+FEFF must be
    /// quoted, or parse_csv's BOM strip silently renames the column on
    /// re-read (and a `["\u{feff}a", "a"]` header would collapse into a
    /// duplicate-header parse error).
    #[test]
    fn to_csv_bom_leading_first_column_round_trips() {
        let c = jdict(vec![
            (
                "columns".to_string(),
                jlist(vec![
                    jstr("\u{feff}date".to_string()),
                    jstr("px".to_string()),
                ]),
            ),
            (
                "rows".to_string(),
                jlist(vec![jdict(vec![
                    ("\u{feff}date".to_string(), jstr("d1".to_string())),
                    ("px".to_string(), jstr("1.5".to_string())),
                ])]),
            ),
        ]);
        let text = csv_to_text(&c).unwrap();
        assert!(
            text.starts_with("\"\u{feff}date\""),
            "BOM-leading first header field must be quoted, got {text:?}"
        );
        assert_eq!(
            csv_cols_of(&doc(&text)).unwrap(),
            vec!["\u{feff}date", "px"]
        );

        // BOM-vs-plain sibling names must survive as two distinct columns.
        let tricky = jdict(vec![
            (
                "columns".to_string(),
                jlist(vec![jstr("\u{feff}a".to_string()), jstr("a".to_string())]),
            ),
            ("rows".to_string(), jlist(vec![])),
        ]);
        let text = csv_to_text(&tricky).unwrap();
        assert_eq!(csv_cols_of(&doc(&text)).unwrap(), vec!["\u{feff}a", "a"]);

        // A double-BOM input: parse strips exactly one; the surviving
        // BOM-leading column then round-trips through to_csv.
        let c2 = doc("\u{feff}\u{feff}a\n1\n");
        assert_eq!(csv_cols_of(&c2).unwrap(), vec!["\u{feff}a"]);
        let round = doc(&csv_to_text(&c2).unwrap());
        assert_eq!(csv_cols_of(&round).unwrap(), vec!["\u{feff}a"]);
    }

    /// Review finding 2: a non-finite `JNum` cell must not leak through the
    /// numeric accessors as a silent NaN.
    #[test]
    fn csv_f64_non_finite_jnum_cell_is_loud() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let c = one_col_doc("pv", vec![jnum(bad)]);
            let e = csv_f64_at(&c, 0, "pv").expect_err("non-finite must fail");
            assert!(e.contains("non-finite"), "{bad}: got `{e}`");
            assert!(e.contains("column `pv`"), "{bad}: got `{e}`");
            let e = csv_f64s_at(&c, "pv").expect_err("non-finite must fail");
            assert!(e.contains("non-finite"), "{bad}: got `{e}`");
        }
    }

    /// Review finding 3: to_csv must not silently drop a top-level subtree
    /// added via json_set; read accessors still tolerate it (reads drop
    /// nothing).
    #[test]
    fn to_csv_unexpected_top_level_key_is_loud() {
        let c = doc("id,px\nalpha,1.5\n");
        let augmented = crate::runtime::json::json_set_at(
            &c,
            "meta.note",
            &jstr("generic annotation".to_string()),
        )
        .expect("json_set on a Csv document");
        let e = csv_to_text(&augmented).expect_err("extra top-level key must fail");
        assert!(e.contains("unexpected top-level key `meta`"), "got `{e}`");
        assert!(e.contains("refusing to silently drop data"), "got `{e}`");
        // Reads keep working on the augmented document.
        assert_eq!(csv_nrows_of(&augmented).unwrap(), 1);
        assert_eq!(csv_f64_at(&augmented, 0, "px").unwrap(), 1.5);
    }

    /// Review finding 4 (decision lock): the to_csv -> parse_csv round trip
    /// is values-as-text — every cell comes back as the `JStr` of its
    /// serialized field text; non-`JStr` cell TYPES do not survive. A
    /// `JBool(true)` cell and a `jstr("true")` cell serialize identically
    /// by design.
    #[test]
    fn round_trip_normalizes_cell_types_to_jstr() {
        let c = jdict(vec![
            (
                "columns".to_string(),
                jlist(vec![
                    jstr("num".to_string()),
                    jstr("flag".to_string()),
                    jstr("note".to_string()),
                ]),
            ),
            (
                "rows".to_string(),
                jlist(vec![jdict(vec![
                    ("num".to_string(), jnum(2.5)),
                    ("flag".to_string(), jbool_cell(true)),
                    ("note".to_string(), jnull_cell()),
                ])]),
            ),
        ]);
        let reparsed = doc(&csv_to_text(&c).unwrap());
        assert_eq!(
            json_value_to_text(&reparsed).unwrap(),
            r#"{"columns":["num","flag","note"],"rows":[{"num":"2.5","flag":"true","note":""}]}"#,
            "cells normalize to JStr of their serialized text"
        );
        // The JNum path stays bit-exact through the numeric accessor.
        assert_eq!(csv_f64_at(&reparsed, 0, "num").unwrap(), 2.5);
        // JBool(true) and jstr("true") are indistinguishable after the
        // trip — the documented normalization.
        let as_text = one_col_doc("flag", vec![jstr("true".to_string())]);
        let as_bool = one_col_doc("flag", vec![jbool_cell(true)]);
        assert_eq!(
            csv_to_text(&as_text).unwrap(),
            csv_to_text(&as_bool).unwrap()
        );
    }

    /// Review finding 5: JBool/JNull cells fail the read accessors with
    /// per-type messages (not a generic "not a string" mislabel), and the
    /// JNull numeric error explains the empty-cell encoding.
    #[test]
    fn csv_accessors_reject_bool_and_null_cells() {
        let bools = one_col_doc("flag", vec![jbool_cell(true)]);
        let e = csv_f64_at(&bools, 0, "flag").expect_err("bool as number");
        assert!(e.contains("cell is a bool, not a number"), "got `{e}`");
        let e = csv_str_at(&bools, 0, "flag").expect_err("bool as string");
        assert!(e.contains("cell is a bool, not a string"), "got `{e}`");

        let nulls = one_col_doc("note", vec![jnull_cell()]);
        let e = csv_f64_at(&nulls, 0, "note").expect_err("null as number");
        assert!(e.contains("cell is null"), "got `{e}`");
        assert!(e.contains("no silent NaN"), "got `{e}`");
        assert!(e.contains("empty-cell encoding"), "got `{e}`");
        let e = csv_str_at(&nulls, 0, "note").expect_err("null as string");
        assert!(e.contains("cell is null, not a string"), "got `{e}`");
    }

    /// Review finding 6: duplicate names in `columns` are rejected by the
    /// whole accessor family (eager csv_doc validation), not just by
    /// parse_csv/to_csv.
    #[test]
    fn hand_built_duplicate_columns_rejected_by_all_accessors() {
        let dup = jdict(vec![
            (
                "columns".to_string(),
                jlist(vec![jstr("a".to_string()), jstr("a".to_string())]),
            ),
            ("rows".to_string(), jlist(vec![])),
        ]);
        for e in [
            csv_nrows_of(&dup).expect_err("nrows"),
            csv_cols_of(&dup).expect_err("cols"),
            csv_f64s_at(&dup, "a").expect_err("f64s"),
            csv_strs_at(&dup, "a").expect_err("strs"),
        ] {
            assert!(e.contains("duplicate column `a` in `columns`"), "got `{e}`");
            assert!(e.contains("columns 1 and 2"), "got `{e}`");
        }
    }

    /// Review finding 7: a foreign ADT constructor is labeled as non-Json,
    /// not mislabeled "a Json `Some` value".
    #[test]
    fn foreign_adt_is_not_labeled_json() {
        let foreign = RuntimeValue::Adt {
            ctor: "Some".to_string(),
            fields: vec![RuntimeValue::float64(1.0)],
            field_names: None,
        };
        let e = csv_nrows_of(&foreign).expect_err("foreign ADT");
        assert!(
            e.contains("non-Json value (constructor `Some`)"),
            "got `{e}`"
        );
        assert!(!e.contains("Json `Some`"), "got `{e}`");
    }

    /// Review finding 8: shape errors truncate huge payloads instead of
    /// interpolating the entire value.
    #[test]
    fn shape_error_truncates_huge_values() {
        let huge = RuntimeValue::List(vec![RuntimeValue::float64(1.0); 20_000]);
        let e = csv_nrows_of(&huge).expect_err("non-Json value");
        assert!(
            e.len() < 600,
            "error must stay readable, got {} bytes",
            e.len()
        );
        assert!(e.contains("elided"), "got `{e}`");
    }

    /// Review finding 11: csv_nrows/csv_cols validate `rows` elements
    /// eagerly like every other accessor — garbage rows do not pass
    /// silently anywhere.
    #[test]
    fn garbage_rows_rejected_eagerly_by_nrows_and_cols() {
        let garbage = jdict(vec![
            ("columns".to_string(), jlist(vec![jstr("a".to_string())])),
            ("rows".to_string(), jlist(vec![jnum(1.0)])),
        ]);
        for e in [
            csv_nrows_of(&garbage).expect_err("nrows"),
            csv_cols_of(&garbage).expect_err("cols"),
            csv_strs_at(&garbage, "a").expect_err("strs"),
        ] {
            assert!(
                e.contains("data row 0 (0-based) is not a JDict"),
                "got `{e}`"
            );
        }
    }

    /// Review finding 12: a row carrying the same declared key twice is a
    /// loud to_csv error (unreachable from surf, latent for Rust callers).
    #[test]
    fn to_csv_duplicate_row_key_is_loud() {
        let row = RuntimeValue::Adt {
            ctor: "JDict".to_string(),
            fields: vec![RuntimeValue::Dict(vec![
                (RuntimeValue::String("a".to_string()), jstr("1".to_string())),
                (RuntimeValue::String("a".to_string()), jstr("2".to_string())),
            ])],
            field_names: None,
        };
        let c = jdict(vec![
            ("columns".to_string(), jlist(vec![jstr("a".to_string())])),
            ("rows".to_string(), jlist(vec![row])),
        ]);
        let e = csv_to_text(&c).expect_err("duplicate row key");
        assert!(e.contains("duplicate key `a`"), "got `{e}`");
    }

    /// Review finding 15: the positional fast path never changes results —
    /// hand-assembled rows with reordered keys read correctly through the
    /// by-name fallback.
    #[test]
    fn reordered_row_keys_read_via_fallback() {
        let c = jdict(vec![
            (
                "columns".to_string(),
                jlist(vec![jstr("a".to_string()), jstr("b".to_string())]),
            ),
            (
                "rows".to_string(),
                jlist(vec![jdict(vec![
                    ("b".to_string(), jstr("2".to_string())),
                    ("a".to_string(), jstr("1".to_string())),
                ])]),
            ),
        ]);
        assert_eq!(csv_str_at(&c, 0, "a").unwrap(), "1");
        assert_eq!(csv_str_at(&c, 0, "b").unwrap(), "2");
        assert_eq!(csv_strs_at(&c, "b").unwrap(), vec!["2"]);
        assert_eq!(csv_to_text(&c).unwrap(), "a,b\n1,2\n");
    }
}
