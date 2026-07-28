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
//!   naming 1-based row (record) and column (field) numbers. Blank rows
//!   are tolerated only at end of input.
//! * **Cells are strings at parse time.** No silent numeric coercion (a
//!   column of zero-padded IDs must not type-flap mid-column). The numeric
//!   accessors convert at access time under the strict JSON number grammar
//!   (the same grammar `parse_json` enforces), tolerating surrounding
//!   ASCII spaces/tabs (matching Python `float()`), and fail loudly on
//!   anything else — naming the column, the row, and the offending text.
//!   Empty cells in a numeric accessor are errors: **no silent NaN**.
//! * **Row indices are 0-based data rows** (`csv_f64(c, 0, "col")` is the
//!   first row *after* the header); parse errors count 1-based physical
//!   records (header = row 1). Error messages state which space they use.
//! * **`to_csv` is the exact inverse shape.** It serializes the same
//!   document form `parse_csv` returns, so `parse_csv(to_csv(c))`
//!   round-trips. Cells may be `JStr` (verbatim), `JNum` (shortest-
//!   round-trip f64 — the same formatter contract as `to_json`, never the
//!   print channel: chelis#748/#723/#734), `JBool`, or `JNull` (empty
//!   cell); container cells, non-finite numbers, rows missing a declared
//!   column, and rows carrying an undeclared key all fail loudly — no
//!   silent data loss.

use super::RuntimeValue;
use super::json::{jdict, jlist, jstr};

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
                    "parse_csv: row {row}: bare carriage return (CR without LF); only \
                     LF or CRLF row separators are supported"
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

/// Validate the fixed document shape and borrow its parts. Every failure
/// names the builtin and says what a Csv document is, so a plain
/// `parse_json` value or a hand-mangled document fails loudly instead of
/// producing garbage.
fn csv_doc<'v>(builtin: &str, value: &'v RuntimeValue) -> Result<CsvDoc<'v>, String> {
    let shape_err = |detail: &str| {
        format!(
            "{builtin}: expected a Csv document (the `{{\"columns\": .., \"rows\": ..}}` \
             JDict that parse_csv returns); {detail}"
        )
    };
    let Some(entries) = as_jdict_entries(value) else {
        return match value {
            RuntimeValue::Adt { ctor, .. } => Err(shape_err(&format!(
                "got a Json `{ctor}` value, not a JDict"
            ))),
            other => Err(shape_err(&format!("got a non-Json value: {other:?}"))),
        };
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
    let rows = as_jlist(rows_value).ok_or_else(|| shape_err("`rows` must be a JList"))?;
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

fn require_column(builtin: &str, doc: &CsvDoc<'_>, column: &str) -> Result<(), String> {
    if doc.columns.contains(&column) {
        return Ok(());
    }
    Err(format!(
        "{builtin}: column `{column}` not found; available columns: {}",
        available_columns(&doc.columns)
    ))
}

/// Fetch one cell. `row_idx` is a 0-based data-row index (the header is
/// not a data row). Documents from `parse_csv` hold every cell as `JStr`;
/// hand-assembled output documents may hold other Json scalars — callers
/// decide what to accept.
fn cell<'v>(
    builtin: &str,
    row: &'v RuntimeValue,
    row_idx: usize,
    column: &str,
) -> Result<&'v RuntimeValue, String> {
    let entries = as_jdict_entries(row).ok_or_else(|| {
        format!(
            "{builtin}: data row {row_idx} is not a JDict; this is not a Csv \
             document from parse_csv"
        )
    })?;
    entries
        .iter()
        .find_map(|(key, item)| match key {
            RuntimeValue::String(s) if s == column => Some(item),
            _ => None,
        })
        .ok_or_else(|| {
            format!(
                "{builtin}: data row {row_idx} has no cell for column `{column}` \
                 (malformed Csv document)"
            )
        })
}

fn cell_str<'v>(
    builtin: &str,
    row: &'v RuntimeValue,
    row_idx: usize,
    column: &str,
) -> Result<&'v str, String> {
    let node = cell(builtin, row, row_idx, column)?;
    as_jstr(node).ok_or_else(|| {
        let kind = match node {
            RuntimeValue::Adt { ctor, .. } if ctor == "JNum" => {
                return format!(
                    "{builtin}: column `{column}`, data row {row_idx}: cell is a number, \
                     not a string; read it with csv_f64/csv_f64s"
                );
            }
            RuntimeValue::Adt { ctor, .. } => format!("a Json `{ctor}` value"),
            other => format!("{other:?}"),
        };
        format!("{builtin}: column `{column}`, data row {row_idx}: cell is {kind}, not a string")
    })
}

/// Strict JSON number grammar
/// (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`) — the same grammar
/// `parse_json` enforces. Notably rejects: empty, `inf`/`NaN`, hex,
/// leading `+`, leading zeros (`007`), bare `.5`/`1.`, and separators
/// (`1_000`, `1,000`).
fn is_strict_json_number(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0;
    if bytes.get(i) == Some(&b'-') {
        i += 1;
    }
    match bytes.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => {
            while matches!(bytes.get(i), Some(b'0'..=b'9')) {
                i += 1;
            }
        }
        _ => return false,
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        if !matches!(bytes.get(i), Some(b'0'..=b'9')) {
            return false;
        }
        while matches!(bytes.get(i), Some(b'0'..=b'9')) {
            i += 1;
        }
    }
    if matches!(bytes.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        if !matches!(bytes.get(i), Some(b'0'..=b'9')) {
            return false;
        }
        while matches!(bytes.get(i), Some(b'0'..=b'9')) {
            i += 1;
        }
    }
    i == bytes.len()
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
    let trimmed = raw.trim_matches(|c| c == ' ' || c == '\t');
    if trimmed.is_empty() {
        return Err(format!(
            "{builtin}: column `{column}`, data row {row_idx} (0-based): cell is empty \
             (no silent NaN/defaults; read optional columns with csv_str/csv_strs)"
        ));
    }
    if !is_strict_json_number(trimmed) {
        return Err(format!(
            "{builtin}: column `{column}`, data row {row_idx} (0-based): cell `{raw}` is \
             not a number (strict JSON number syntax; surrounding spaces/tabs tolerated)"
        ));
    }
    let value: f64 = trimmed.parse().map_err(|err| {
        format!(
            "{builtin}: column `{column}`, data row {row_idx} (0-based): internal error \
             parsing `{trimmed}`: {err}"
        )
    })?;
    if !value.is_finite() {
        return Err(format!(
            "{builtin}: column `{column}`, data row {row_idx} (0-based): number `{raw}` \
             overflows f64"
        ));
    }
    Ok(value)
}

/// Read a numeric cell: a `JStr` cell converts under [`parse_cell_number`];
/// a `JNum` cell (hand-assembled output documents) is read directly.
fn cell_f64(
    builtin: &str,
    row: &RuntimeValue,
    row_idx: usize,
    column: &str,
) -> Result<f64, String> {
    let node = cell(builtin, row, row_idx, column)?;
    if let RuntimeValue::Adt { ctor, fields, .. } = node
        && ctor == "JNum"
    {
        if let [RuntimeValue::Scalar(payload)] = fields.as_slice()
            && payload.dtype().is_float()
        {
            return Ok(payload.bits().as_f64());
        }
        return Err(format!(
            "{builtin}: column `{column}`, data row {row_idx}: malformed JNum cell \
             {fields:?}"
        ));
    }
    let text = cell_str(builtin, row, row_idx, column)?;
    parse_cell_number(builtin, column, row_idx, text)
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
    require_column("csv_strs", &doc, column)?;
    doc.rows
        .iter()
        .enumerate()
        .map(|(row_idx, row)| {
            cell_str("csv_strs", row, row_idx, column).map(|text| text.to_string())
        })
        .collect()
}

pub(super) fn csv_f64s_at(value: &RuntimeValue, column: &str) -> Result<Vec<f64>, String> {
    let doc = csv_doc("csv_f64s", value)?;
    require_column("csv_f64s", &doc, column)?;
    doc.rows
        .iter()
        .enumerate()
        .map(|(row_idx, row)| cell_f64("csv_f64s", row, row_idx, column))
        .collect()
}

pub(super) fn csv_str_at(
    value: &RuntimeValue,
    row_idx: i64,
    column: &str,
) -> Result<String, String> {
    let doc = csv_doc("csv_str", value)?;
    require_column("csv_str", &doc, column)?;
    let (index, row) = require_row_index("csv_str", &doc, row_idx)?;
    cell_str("csv_str", row, index, column).map(|text| text.to_string())
}

pub(super) fn csv_f64_at(value: &RuntimeValue, row_idx: i64, column: &str) -> Result<f64, String> {
    let doc = csv_doc("csv_f64", value)?;
    require_column("csv_f64", &doc, column)?;
    let (index, row) = require_row_index("csv_f64", &doc, row_idx)?;
    cell_f64("csv_f64", row, index, column)
}

// ---------------------------------------------------------------------------
// Serialization (to_csv)
// ---------------------------------------------------------------------------

/// Append one field with minimal RFC 4180 quoting: quoted only when the
/// text contains a comma, quote, CR, or LF (embedded quotes doubled), or
/// when `force_quote` demands it (the empty sole field of a single-column
/// row, which unquoted would read back as a blank line).
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

/// Render one cell of a hand- or parse-assembled document to field text.
fn cell_to_field_text(column: &str, row_idx: usize, node: &RuntimeValue) -> Result<String, String> {
    let RuntimeValue::Adt { ctor, fields, .. } = node else {
        return Err(format!(
            "to_csv: column `{column}`, data row {row_idx}: cell is not a Json value \
             ({node:?})"
        ));
    };
    match (ctor.as_str(), fields.as_slice()) {
        ("JStr", [RuntimeValue::String(s)]) => Ok(s.clone()),
        ("JNum", [RuntimeValue::Scalar(payload)]) if payload.dtype().is_float() => {
            let value = payload.bits().as_f64();
            if !value.is_finite() {
                return Err(format!(
                    "to_csv: column `{column}`, data row {row_idx}: cannot represent \
                     non-finite number `{value}` in CSV"
                ));
            }
            // Shortest-round-trip f64 — the same fidelity contract as
            // `to_json` (never the print channel: chelis#748/#723/#734).
            Ok(format!("{value:?}"))
        }
        ("JBool", [RuntimeValue::Bool(b)]) => Ok(if *b { "true" } else { "false" }.to_string()),
        ("JNull", []) => Ok(String::new()),
        ("JList" | "JDict", _) => Err(format!(
            "to_csv: column `{column}`, data row {row_idx}: cell is a {}; CSV cells \
             must be scalars (string, number, bool, or null)",
            if ctor == "JList" { "list" } else { "dict" }
        )),
        _ => Err(format!(
            "to_csv: column `{column}`, data row {row_idx}: malformed Json cell \
             (constructor `{ctor}` with fields {fields:?})"
        )),
    }
}

/// Serialize a Csv document (the exact shape `parse_csv` returns) to CSV
/// text: header row from `columns`, one row per `rows` entry, minimal
/// quoting, LF row separators, trailing final newline. Deterministic and
/// byte-stable. Loud errors, no silent data loss: rows missing a declared
/// column or carrying an undeclared key fail by name.
pub(super) fn csv_to_text(value: &RuntimeValue) -> Result<String, String> {
    let doc = csv_doc("to_csv", value)?;
    if doc.columns.is_empty() {
        return Err("to_csv: `columns` is empty (a CSV needs at least one column)".to_string());
    }
    for (index, name) in doc.columns.iter().enumerate() {
        if let Some(first) = doc.columns[..index].iter().position(|other| other == name) {
            return Err(format!(
                "to_csv: duplicate column `{name}` (columns {} and {}); the result \
                 could not be re-read by parse_csv",
                first + 1,
                index + 1
            ));
        }
    }
    let single_column = doc.columns.len() == 1;
    let mut out = String::new();
    for (index, name) in doc.columns.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        write_csv_field(&mut out, name, single_column && name.is_empty());
    }
    out.push('\n');
    for (row_idx, row) in doc.rows.iter().enumerate() {
        let entries = as_jdict_entries(row).ok_or_else(|| {
            format!("to_csv: data row {row_idx} is not a JDict of column -> cell")
        })?;
        // Refuse to silently drop data: every key in the row must be a
        // declared column (the reverse — a declared column missing from
        // the row — fails in the emit loop below).
        for (key, _) in entries {
            let RuntimeValue::String(key) = key else {
                return Err(format!(
                    "to_csv: data row {row_idx} has a non-string key {key:?}"
                ));
            };
            if !doc.columns.iter().any(|name| name == key) {
                return Err(format!(
                    "to_csv: data row {row_idx} has key `{key}` that is not in \
                     `columns`; refusing to silently drop data"
                ));
            }
        }
        for (index, column) in doc.columns.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            let node = cell("to_csv", row, row_idx, column)?;
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
        let e = err("a,b\r1,2\r");
        assert!(e.contains("row 1"), "got `{e}`");
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
        assert!(e.contains("no cell for column `b`"), "got `{e}`");

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
}
