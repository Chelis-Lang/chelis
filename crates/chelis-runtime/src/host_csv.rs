//! CSV over the canonical `List[Dict[string,string]]` text table
//! ([05-OP-61], [05-OP-2..3], [05-OP-5]), shared by every execution lane.
//!
//! A table is a list of rows, each an ordered list of `(column, cell)` text
//! pairs. Parsing never infers a numeric cell type; numeric meaning enters
//! only through an explicit integer or float accessor. The evaluator and the
//! `chelis_csv_*` C exports adapt their own carriers to [`CsvTable`] and call
//! these definitions, so parsing, the accessors, serialization, and every
//! failure message have one definition.

use std::collections::BTreeSet;

/// One parsed data row: its `(column, cell)` pairs in header order.
pub type CsvRow = Vec<(String, String)>;

struct CsvRecord {
    fields: Vec<String>,
    raw_empty: bool,
}

/// Parse a header-first CSV document into its data rows.
pub fn parse_csv_text(text: &str) -> Result<Vec<CsvRow>, String> {
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
        rows.push(
            header
                .fields
                .iter()
                .cloned()
                .zip(record.fields.iter().cloned())
                .collect(),
        );
    }
    Ok(rows)
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

/// A validated text table borrowed from a lane's own carrier: every row has
/// the same set of unique columns, named in row 0's order.
pub struct CsvTable<'a> {
    columns: Vec<&'a str>,
    rows: Vec<Vec<(&'a str, &'a str)>>,
}

impl<'a> CsvTable<'a> {
    /// Validates `rows` for `builtin`: unique columns per row, and the same
    /// columns in every row.
    pub fn new(builtin: &str, rows: Vec<Vec<(&'a str, &'a str)>>) -> Result<Self, String> {
        let mut columns = Vec::new();
        for (row_index, row) in rows.iter().enumerate() {
            let names: Vec<&str> = row.iter().map(|(key, _)| *key).collect();
            assert_unique_columns(builtin, &names)?;
            if row_index == 0 {
                columns = names;
            } else {
                let expected: BTreeSet<_> = columns.iter().copied().collect();
                let actual: BTreeSet<_> = names.iter().copied().collect();
                if actual != expected {
                    return Err(format!(
                        "{builtin}: data row {row_index} does not have the same columns as row 0"
                    ));
                }
            }
        }
        Ok(Self { columns, rows })
    }

    fn require_column(&self, builtin: &str, column: &str) -> Result<(), String> {
        if self.columns.contains(&column) {
            Ok(())
        } else {
            Err(format!(
                "{builtin}: column `{column}` not found; available columns: {}",
                self.columns.join(", ")
            ))
        }
    }

    fn cell(&self, builtin: &str, row_index: usize, column: &str) -> Result<&'a str, String> {
        self.require_column(builtin, column)?;
        let row = self.rows.get(row_index).ok_or_else(|| {
            format!(
                "{builtin}: data row {row_index} out of range for {} rows",
                self.rows.len()
            )
        })?;
        row.iter()
            .find_map(|(key, value)| (*key == column).then_some(*value))
            .ok_or_else(|| format!("{builtin}: data row {row_index} has no column `{column}`"))
    }

    fn column_cells(&self, builtin: &str, column: &str) -> Result<Vec<&'a str>, String> {
        self.require_column(builtin, column)?;
        (0..self.rows.len())
            .map(|row| self.cell(builtin, row, column))
            .collect()
    }
}

fn assert_unique_columns<S: AsRef<str>>(builtin: &str, columns: &[S]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
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

fn checked_row_index(builtin: &str, index: i64) -> Result<usize, String> {
    usize::try_from(index)
        .map_err(|_| format!("{builtin}: row index must be nonnegative, got {index}"))
}

/// Scan one JSON number token
/// (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`) at the start of
/// `bytes`. `Ok(len)` is the token length; `Err((offset, reason))` pinpoints
/// the first offending byte.
pub fn json_number_token_len(bytes: &[u8]) -> Result<usize, (usize, &'static str)> {
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
        _ => return Err((i, "invalid number (expected a digit)")),
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        if !matches!(bytes.get(i), Some(b'0'..=b'9')) {
            return Err((i, "invalid number (expected a digit after `.`)"));
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
            return Err((i, "invalid number (expected a digit in exponent)"));
        }
        while matches!(bytes.get(i), Some(b'0'..=b'9')) {
            i += 1;
        }
    }
    Ok(i)
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
    let len = json_number_token_len(trimmed.as_bytes())
        .map_err(|(_, reason)| format!("{builtin}: cell `{text}` is not an exact i64: {reason}"))?;
    if len != trimmed.len() {
        return Err(format!("{builtin}: cell `{text}` is not an exact i64"));
    }
    if trimmed.contains(['.', 'e', 'E']) {
        return Err(format!(
            "{builtin}: cell `{text}` uses float syntax; use the corresponding csv_f64 accessor"
        ));
    }
    trimmed
        .parse::<i64>()
        .map_err(|_| format!("{builtin}: cell `{text}` overflows i64"))
}

pub fn csv_cols(rows: Vec<Vec<(&str, &str)>>) -> Result<Vec<String>, String> {
    Ok(CsvTable::new("csv_cols", rows)?
        .columns
        .into_iter()
        .map(str::to_string)
        .collect())
}

pub fn csv_nrows(rows: Vec<Vec<(&str, &str)>>) -> Result<i64, String> {
    i64::try_from(CsvTable::new("csv_nrows", rows)?.rows.len())
        .map_err(|_| "csv_nrows: row count overflows i64".to_string())
}

pub fn csv_strs(rows: Vec<Vec<(&str, &str)>>, column: &str) -> Result<Vec<String>, String> {
    let table = CsvTable::new("csv_strs", rows)?;
    Ok(table
        .column_cells("csv_strs", column)?
        .into_iter()
        .map(str::to_string)
        .collect())
}

pub fn csv_f64s(rows: Vec<Vec<(&str, &str)>>, column: &str) -> Result<Vec<f64>, String> {
    let table = CsvTable::new("csv_f64s", rows)?;
    table
        .column_cells("csv_f64s", column)?
        .into_iter()
        .map(|cell| parse_f64_cell("csv_f64s", cell))
        .collect()
}

pub fn csv_ints(rows: Vec<Vec<(&str, &str)>>, column: &str) -> Result<Vec<i64>, String> {
    let table = CsvTable::new("csv_ints", rows)?;
    table
        .column_cells("csv_ints", column)?
        .into_iter()
        .map(|cell| parse_int_cell("csv_ints", cell))
        .collect()
}

pub fn csv_str(
    rows: Vec<Vec<(&str, &str)>>,
    row_index: i64,
    column: &str,
) -> Result<String, String> {
    let table = CsvTable::new("csv_str", rows)?;
    table
        .cell("csv_str", checked_row_index("csv_str", row_index)?, column)
        .map(str::to_string)
}

pub fn csv_f64(rows: Vec<Vec<(&str, &str)>>, row_index: i64, column: &str) -> Result<f64, String> {
    let table = CsvTable::new("csv_f64", rows)?;
    let text = table.cell("csv_f64", checked_row_index("csv_f64", row_index)?, column)?;
    parse_f64_cell("csv_f64", text)
}

pub fn csv_int(rows: Vec<Vec<(&str, &str)>>, row_index: i64, column: &str) -> Result<i64, String> {
    let table = CsvTable::new("csv_int", rows)?;
    let text = table.cell("csv_int", checked_row_index("csv_int", row_index)?, column)?;
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

pub fn to_csv(rows: Vec<Vec<(&str, &str)>>) -> Result<String, String> {
    let table = CsvTable::new("to_csv", rows)?;
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
    for row_index in 0..table.rows.len() {
        for (column_index, column) in table.columns.iter().enumerate() {
            if column_index > 0 {
                out.push(',');
            }
            write_field(&mut out, table.cell("to_csv", row_index, column)?);
        }
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn borrowed(rows: &[CsvRow]) -> Vec<Vec<(&str, &str)>> {
        rows.iter()
            .map(|row| row.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect())
            .collect()
    }

    #[test]
    fn number_scanner_requires_the_complete_json_grammar() {
        assert_eq!(json_number_token_len(b"-12.5e+2"), Ok(8));
        assert!(json_number_token_len(b".5").is_err());
        assert!(json_number_token_len(b"01").is_ok());
    }

    #[test]
    fn parse_returns_text_rows_and_exact_accessors() {
        let rows = parse_csv_text("id,px\n9007199254740993,1.5\n").unwrap();
        assert_eq!(csv_cols(borrowed(&rows)).unwrap(), ["id", "px"]);
        assert_eq!(csv_nrows(borrowed(&rows)).unwrap(), 1);
        assert_eq!(
            csv_int(borrowed(&rows), 0, "id").unwrap(),
            9_007_199_254_740_993
        );
        assert_eq!(csv_f64(borrowed(&rows), 0, "px").unwrap(), 1.5);
    }

    #[test]
    fn numeric_access_is_explicit_and_grammar_checked() {
        let rows = parse_csv_text("id\n1.0\n").unwrap();
        let error = csv_int(borrowed(&rows), 0, "id").unwrap_err();
        assert!(error.contains("csv_f64"), "{error}");
        assert_eq!(csv_f64(borrowed(&rows), 0, "id").unwrap(), 1.0);
    }

    #[test]
    fn parse_and_serialize_preserve_quoted_text() {
        let rows = parse_csv_text("name,note\nA,\"x,y\"\n").unwrap();
        assert_eq!(csv_str(borrowed(&rows), 0, "note").unwrap(), "x,y");
        assert_eq!(to_csv(borrowed(&rows)).unwrap(), "name,note\nA,\"x,y\"\n");
    }

    #[test]
    fn duplicate_and_ragged_rows_fail_loudly() {
        assert!(parse_csv_text("a,a\n1,2\n")
            .unwrap_err()
            .contains("duplicate"));
        assert!(parse_csv_text("a,b\n1\n")
            .unwrap_err()
            .contains("expected 2"));
        let error = CsvTable::new("csv_nrows", vec![vec![("a", "1")], vec![("b", "2")]])
            .err()
            .unwrap();
        assert!(error.contains("same columns"), "{error}");
    }
}
