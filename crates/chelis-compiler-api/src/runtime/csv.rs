//! Host-lane CSV I/O over the canonical text-table carrier.
//!
//! A table is exactly `List[Dict[string,string]]`. CSV parsing never infers a
//! numeric cell type; numeric meaning enters only through an explicit
//! `csv_int*` or `csv_f64*` accessor under [05-OP-2..3].

use std::collections::HashSet;

use super::RuntimeValue;
use super::host_ops::describe_value;
use super::numeric_text::json_number_token_len;

struct CsvRecord {
    fields: Vec<String>,
    raw_empty: bool,
}

/// Parse a header-first CSV document into `List[Dict[string,string]]`.
pub(super) fn parse_csv_text(text: &str) -> Result<RuntimeValue, String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut records = parse_records(text)?;
    while matches!(records.last(), Some(record) if record.raw_empty) {
        records.pop();
    }
    let Some((header, data)) = records.split_first() else {
        return Err("parse_csv: input is empty (expected a header row)".to_string());
    };
    if header.raw_empty {
        return Err("parse_csv: row 1 is blank (expected a header row)".to_string());
    }
    assert_unique_columns("parse_csv", &header.fields)?;

    let mut rows = Vec::with_capacity(data.len());
    for (index, record) in data.iter().enumerate() {
        let row_number = index + 2;
        if record.raw_empty {
            return Err(format!(
                "parse_csv: row {row_number} is blank (blank rows are only tolerated at end of input)"
            ));
        }
        if record.fields.len() != header.fields.len() {
            return Err(format!(
                "parse_csv: row {row_number} has {} fields, expected {}",
                record.fields.len(),
                header.fields.len()
            ));
        }
        rows.push(RuntimeValue::Dict(
            header
                .fields
                .iter()
                .cloned()
                .zip(record.fields.iter().cloned())
                .map(|(key, value)| (RuntimeValue::String(key), RuntimeValue::String(value)))
                .collect(),
        ));
    }
    Ok(RuntimeValue::List(rows))
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
    let mut fields = Vec::new();
    let mut raw_empty = true;
    loop {
        let column = fields.len() + 1;
        let (field, quoted) = parse_field(chars, row, column)?;
        raw_empty &= !quoted && field.is_empty() && fields.is_empty();
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
                if chars.next() != Some('\n') {
                    return Err(format!(
                        "parse_csv: row {row}, column {column}: bare carriage return"
                    ));
                }
                break;
            }
            None => break,
            Some(other) => {
                return Err(format!(
                    "parse_csv: row {row}, column {column}: unexpected character `{other}` after field"
                ));
            }
        }
    }
    Ok(CsvRecord { fields, raw_empty })
}

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
                        "parse_csv: row {row}, column {column}: unclosed quoted field"
                    ));
                }
                Some('"') if chars.peek() == Some(&'"') => {
                    chars.next();
                    out.push('"');
                }
                Some('"') => {
                    return match chars.peek().copied() {
                        Some(',') | Some('\n') | Some('\r') | None => Ok((out, true)),
                        Some(other) => Err(format!(
                            "parse_csv: row {row}, column {column}: unexpected character `{other}` after closing quote"
                        )),
                    };
                }
                Some(ch) => out.push(ch),
            }
        }
    }
    loop {
        match chars.peek().copied() {
            Some(',') | Some('\n') | Some('\r') | None => return Ok((out, false)),
            Some('"') => {
                return Err(format!(
                    "parse_csv: row {row}, column {column}: bare quote inside unquoted field"
                ));
            }
            Some(ch) => {
                chars.next();
                out.push(ch);
            }
        }
    }
}

struct CsvTable<'a> {
    columns: Vec<&'a str>,
    rows: Vec<&'a [(RuntimeValue, RuntimeValue)]>,
}

fn csv_table<'a>(builtin: &str, value: &'a RuntimeValue) -> Result<CsvTable<'a>, String> {
    let RuntimeValue::List(items) = value else {
        return Err(format!(
            "{builtin}: expected List[Dict[string,string]], got {}",
            describe_value(value)
        ));
    };
    let mut rows = Vec::with_capacity(items.len());
    let mut columns = Vec::new();
    for (row_index, row) in items.iter().enumerate() {
        let RuntimeValue::Dict(entries) = row else {
            return Err(format!(
                "{builtin}: data row {row_index} is not Dict[string,string]"
            ));
        };
        let mut names = Vec::with_capacity(entries.len());
        for (key, cell) in entries {
            let RuntimeValue::String(key) = key else {
                return Err(format!(
                    "{builtin}: data row {row_index} has a non-string key"
                ));
            };
            if !matches!(cell, RuntimeValue::String(_)) {
                return Err(format!(
                    "{builtin}: column `{key}`, data row {row_index}: expected a string cell, got {}",
                    describe_value(cell)
                ));
            }
            names.push(key.as_str());
        }
        assert_unique_columns(builtin, &names)?;
        if row_index == 0 {
            columns = names;
        } else {
            let expected: HashSet<_> = columns.iter().copied().collect();
            let actual: HashSet<_> = names.iter().copied().collect();
            if actual != expected {
                return Err(format!(
                    "{builtin}: data row {row_index} does not have the same columns as row 0"
                ));
            }
        }
        rows.push(entries.as_slice());
    }
    Ok(CsvTable { columns, rows })
}

fn assert_unique_columns<S: AsRef<str>>(builtin: &str, columns: &[S]) -> Result<(), String> {
    let mut seen = HashSet::new();
    for (index, column) in columns.iter().enumerate() {
        let column = column.as_ref();
        if !seen.insert(column) {
            return Err(format!(
                "{builtin}: duplicate column `{column}` at position {}",
                index + 1
            ));
        }
    }
    Ok(())
}

fn require_column<'a>(builtin: &str, table: &CsvTable<'a>, column: &str) -> Result<(), String> {
    if table.columns.contains(&column) {
        Ok(())
    } else {
        Err(format!(
            "{builtin}: column `{column}` not found; available columns: {}",
            table.columns.join(", ")
        ))
    }
}

fn cell<'a>(
    builtin: &str,
    table: &CsvTable<'a>,
    row_index: usize,
    column: &str,
) -> Result<&'a str, String> {
    require_column(builtin, table, column)?;
    let row = table.rows.get(row_index).ok_or_else(|| {
        format!(
            "{builtin}: data row {row_index} out of range for {} rows",
            table.rows.len()
        )
    })?;
    row.iter()
        .find_map(|(key, value)| match (key, value) {
            (RuntimeValue::String(key), RuntimeValue::String(value)) if key == column => {
                Some(value.as_str())
            }
            _ => None,
        })
        .ok_or_else(|| format!("{builtin}: data row {row_index} has no column `{column}`"))
}

fn checked_row_index(builtin: &str, index: i64) -> Result<usize, String> {
    usize::try_from(index)
        .map_err(|_| format!("{builtin}: row index must be nonnegative, got {index}"))
}

fn parse_f64_cell(builtin: &str, text: &str) -> Result<f64, String> {
    let trimmed = text.trim_matches([' ', '\t']);
    let len = json_number_token_len(trimmed.as_bytes())
        .map_err(|(_, reason)| format!("{builtin}: cell `{text}` is not a finite f64: {reason}"))?;
    if len != trimmed.len() {
        return Err(format!("{builtin}: cell `{text}` is not a finite f64"));
    }
    let value = trimmed
        .parse::<f64>()
        .map_err(|_| format!("{builtin}: cell `{text}` is not a finite f64"))?;
    if !value.is_finite() {
        return Err(format!("{builtin}: cell `{text}` overflows f64"));
    }
    Ok(value)
}

fn parse_int_cell(builtin: &str, text: &str) -> Result<i64, String> {
    let trimmed = text.trim_matches([' ', '\t']);
    let len = json_number_token_len(trimmed.as_bytes()).map_err(|(_, reason)| {
        format!("{builtin}: cell `{text}` is not an exact int64: {reason}")
    })?;
    if len != trimmed.len() {
        return Err(format!("{builtin}: cell `{text}` is not an exact int64"));
    }
    if trimmed.contains(['.', 'e', 'E']) {
        return Err(format!(
            "{builtin}: cell `{text}` uses float syntax; use the corresponding csv_f64 accessor"
        ));
    }
    trimmed
        .parse::<i64>()
        .map_err(|_| format!("{builtin}: cell `{text}` overflows int64"))
}

pub(super) fn csv_cols_of(value: &RuntimeValue) -> Result<Vec<String>, String> {
    Ok(csv_table("csv_cols", value)?
        .columns
        .into_iter()
        .map(str::to_string)
        .collect())
}

pub(super) fn csv_nrows_of(value: &RuntimeValue) -> Result<i64, String> {
    i64::try_from(csv_table("csv_nrows", value)?.rows.len())
        .map_err(|_| "csv_nrows: row count overflows int64".to_string())
}

pub(super) fn csv_strs_at(value: &RuntimeValue, column: &str) -> Result<Vec<String>, String> {
    let table = csv_table("csv_strs", value)?;
    require_column("csv_strs", &table, column)?;
    (0..table.rows.len())
        .map(|row| cell("csv_strs", &table, row, column).map(str::to_string))
        .collect()
}

pub(super) fn csv_f64s_at(value: &RuntimeValue, column: &str) -> Result<Vec<f64>, String> {
    let table = csv_table("csv_f64s", value)?;
    require_column("csv_f64s", &table, column)?;
    (0..table.rows.len())
        .map(|row| {
            cell("csv_f64s", &table, row, column).and_then(|v| parse_f64_cell("csv_f64s", v))
        })
        .collect()
}

pub(super) fn csv_ints_at(value: &RuntimeValue, column: &str) -> Result<Vec<i64>, String> {
    let table = csv_table("csv_ints", value)?;
    require_column("csv_ints", &table, column)?;
    (0..table.rows.len())
        .map(|row| {
            cell("csv_ints", &table, row, column).and_then(|v| parse_int_cell("csv_ints", v))
        })
        .collect()
}

pub(super) fn csv_str_at(
    value: &RuntimeValue,
    row_index: i64,
    column: &str,
) -> Result<String, String> {
    let table = csv_table("csv_str", value)?;
    cell(
        "csv_str",
        &table,
        checked_row_index("csv_str", row_index)?,
        column,
    )
    .map(str::to_string)
}

pub(super) fn csv_f64_at(
    value: &RuntimeValue,
    row_index: i64,
    column: &str,
) -> Result<f64, String> {
    let table = csv_table("csv_f64", value)?;
    let text = cell(
        "csv_f64",
        &table,
        checked_row_index("csv_f64", row_index)?,
        column,
    )?;
    parse_f64_cell("csv_f64", text)
}

pub(super) fn csv_int_at(
    value: &RuntimeValue,
    row_index: i64,
    column: &str,
) -> Result<i64, String> {
    let table = csv_table("csv_int", value)?;
    let text = cell(
        "csv_int",
        &table,
        checked_row_index("csv_int", row_index)?,
        column,
    )?;
    parse_int_cell("csv_int", text)
}

fn write_field(out: &mut String, field: &str) {
    if field.contains([',', '"', '\r', '\n']) {
        out.push('"');
        for ch in field.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
    } else {
        out.push_str(field);
    }
}

pub(super) fn csv_to_text(value: &RuntimeValue) -> Result<String, String> {
    let table = csv_table("to_csv", value)?;
    if table.rows.is_empty() {
        return Ok(String::new());
    }
    let mut out = String::new();
    for (index, column) in table.columns.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        write_field(&mut out, column);
    }
    out.push('\n');
    for (row_index, _) in table.rows.iter().enumerate() {
        for (column_index, column) in table.columns.iter().enumerate() {
            if column_index > 0 {
                out.push(',');
            }
            write_field(&mut out, cell("to_csv", &table, row_index, column)?);
        }
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_returns_the_text_table_carrier() {
        let value = parse_csv_text("id,px\n9007199254740993,1.5\n").unwrap();
        assert!(matches!(value, RuntimeValue::List(_)));
        assert_eq!(csv_cols_of(&value).unwrap(), ["id", "px"]);
        assert_eq!(csv_nrows_of(&value).unwrap(), 1);
        assert_eq!(csv_int_at(&value, 0, "id").unwrap(), 9_007_199_254_740_993);
        assert_eq!(csv_f64_at(&value, 0, "px").unwrap(), 1.5);
    }

    #[test]
    fn numeric_access_is_explicit_and_grammar_checked() {
        let value = parse_csv_text("id\n1.0\n").unwrap();
        let error = csv_int_at(&value, 0, "id").unwrap_err();
        assert!(error.contains("csv_f64"), "{error}");
        assert_eq!(csv_f64_at(&value, 0, "id").unwrap(), 1.0);
    }

    #[test]
    fn parse_and_serialize_preserve_quoted_text() {
        let value = parse_csv_text("name,note\nA,\"x,y\"\n").unwrap();
        assert_eq!(csv_str_at(&value, 0, "note").unwrap(), "x,y");
        assert_eq!(csv_to_text(&value).unwrap(), "name,note\nA,\"x,y\"\n");
    }

    #[test]
    fn duplicate_and_ragged_rows_fail_loudly() {
        assert!(
            parse_csv_text("a,a\n1,2\n")
                .unwrap_err()
                .contains("duplicate")
        );
        assert!(
            parse_csv_text("a,b\n1\n")
                .unwrap_err()
                .contains("expected 2")
        );
    }

    #[test]
    fn wrong_runtime_shape_is_not_reinterpreted() {
        let error = csv_nrows_of(&RuntimeValue::Bool(true)).unwrap_err();
        assert!(error.contains("List[Dict[string,string]]"), "{error}");
    }
}
