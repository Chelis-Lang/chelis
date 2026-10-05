//! Host-lane CSV I/O over the canonical text-table carrier.
//!
//! A table is exactly `List[Dict[string,string]]`. Parsing, the accessors,
//! serialization, and their failure text are `chelis-runtime`'s `host_csv`
//! definitions, which compiled C calls too (chelis#1297); this module only
//! adapts the evaluator's values to and from them.

use chelis_runtime::host_csv;

use super::RuntimeValue;
use super::host_ops::describe_value;

/// Parse a header-first CSV document into `List[Dict[string,string]]`.
pub(super) fn parse_csv_text(text: &str) -> Result<RuntimeValue, String> {
    let rows = host_csv::parse_csv_text(text)?;
    Ok(RuntimeValue::List(
        rows.into_iter()
            .map(|row| {
                RuntimeValue::Dict(
                    row.into_iter()
                        .map(|(key, value)| {
                            (RuntimeValue::String(key), RuntimeValue::String(value))
                        })
                        .collect(),
                )
            })
            .collect(),
    ))
}

/// The evaluator's table as the runtime's borrowed rows. A value that is not
/// `List[Dict[string,string]]` is refused, never reinterpreted.
fn table_rows<'a>(
    builtin: &str,
    value: &'a RuntimeValue,
) -> Result<Vec<Vec<(&'a str, &'a str)>>, String> {
    let RuntimeValue::List(items) = value else {
        return Err(format!(
            "{builtin}: expected List[Dict[string,string]], got {}",
            describe_value(value)
        ));
    };
    let mut rows = Vec::with_capacity(items.len());
    for (row_index, row) in items.iter().enumerate() {
        let RuntimeValue::Dict(entries) = row else {
            return Err(format!(
                "{builtin}: data row {row_index} is not Dict[string,string]"
            ));
        };
        let mut pairs = Vec::with_capacity(entries.len());
        for (key, cell) in entries {
            let RuntimeValue::String(key) = key else {
                return Err(format!(
                    "{builtin}: data row {row_index} has a non-string key"
                ));
            };
            let RuntimeValue::String(cell) = cell else {
                return Err(format!(
                    "{builtin}: column `{key}`, data row {row_index}: expected a string cell, got {}",
                    describe_value(cell)
                ));
            };
            pairs.push((key.as_str(), cell.as_str()));
        }
        rows.push(pairs);
    }
    Ok(rows)
}

pub(super) fn csv_cols_of(value: &RuntimeValue) -> Result<Vec<String>, String> {
    host_csv::csv_cols(table_rows("csv_cols", value)?)
}

pub(super) fn csv_nrows_of(value: &RuntimeValue) -> Result<i64, String> {
    host_csv::csv_nrows(table_rows("csv_nrows", value)?)
}

pub(super) fn csv_strs_at(value: &RuntimeValue, column: &str) -> Result<Vec<String>, String> {
    host_csv::csv_strs(table_rows("csv_strs", value)?, column)
}

pub(super) fn csv_f64s_at(value: &RuntimeValue, column: &str) -> Result<Vec<f64>, String> {
    host_csv::csv_f64s(table_rows("csv_f64s", value)?, column)
}

pub(super) fn csv_ints_at(value: &RuntimeValue, column: &str) -> Result<Vec<i64>, String> {
    host_csv::csv_ints(table_rows("csv_ints", value)?, column)
}

pub(super) fn csv_str_at(
    value: &RuntimeValue,
    row_index: i64,
    column: &str,
) -> Result<String, String> {
    host_csv::csv_str(table_rows("csv_str", value)?, row_index, column)
}

pub(super) fn csv_f64_at(
    value: &RuntimeValue,
    row_index: i64,
    column: &str,
) -> Result<f64, String> {
    host_csv::csv_f64(table_rows("csv_f64", value)?, row_index, column)
}

pub(super) fn csv_int_at(
    value: &RuntimeValue,
    row_index: i64,
    column: &str,
) -> Result<i64, String> {
    host_csv::csv_int(table_rows("csv_int", value)?, row_index, column)
}

pub(super) fn csv_to_text(value: &RuntimeValue) -> Result<String, String> {
    host_csv::to_csv(table_rows("to_csv", value)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_returns_the_text_table_carrier() {
        let value = parse_csv_text("id,px\n9007199254740993,1.5\n").unwrap();
        assert!(matches!(value, RuntimeValue::List(_)));
        assert_eq!(csv_cols_of(&value).unwrap(), ["id", "px"]);
        assert_eq!(csv_int_at(&value, 0, "id").unwrap(), 9_007_199_254_740_993);
    }

    #[test]
    fn wrong_runtime_shape_is_not_reinterpreted() {
        let error = csv_nrows_of(&RuntimeValue::Bool(true)).unwrap_err();
        assert!(error.contains("List[Dict[string,string]]"), "{error}");
    }
}
