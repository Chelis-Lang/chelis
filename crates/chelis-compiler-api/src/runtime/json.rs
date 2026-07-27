//! Host-lane JSON I/O core (chelis#890).
//!
//! Pure helpers behind the `parse_json` / `to_json` / `json_*` / `j*` /
//! `round_to` builtins dispatched from `eval.rs`. Json values are ordinary
//! [`RuntimeValue::Adt`] values over the prelude `Json` ADT
//! (`crates/chelis-types/src/builtins.rs::register_prelude_adts`):
//!
//! ```text
//! Json = JNull | JBool bool | JNum f64 | JStr string
//!      | JList List[Json] | JDict Dict[string, Json]
//! ```
//!
//! Design contract (see `docs/CHELIS_SURFACE.md` §3.8):
//!
//! * **Numbers are f64.** JSON numbers parse through Rust's `f64::from_str`
//!   (correctly rounded); integers beyond 2^53 lose precision, exactly as
//!   in every f64-backed JSON reader.
//! * **Serialization is shortest-round-trip.** `to_json` formats each f64
//!   with Rust's shortest-representation formatter (`{:?}`), so
//!   `parse_json(to_json(v))` reproduces every finite f64 bit-for-bit.
//!   This deliberately does NOT reuse the print/`to_string` channel, whose
//!   C-side format selection destroys value classes (chelis#748, #723,
//!   #734). Non-finite numbers fail loudly (JSON has no NaN/Infinity).
//! * **Key order is insertion order** — document order for `parse_json`,
//!   construction order for `jdict`/`json_set` (new keys append). Duplicate
//!   keys keep the first occurrence's position with the last occurrence's
//!   value (Python `json` semantics). Serialization is therefore
//!   deterministic and byte-stable across runs.
//! * **Paths are dot-separated.** A segment on a `JDict` is a key; on a
//!   `JList` it must be an all-digits zero-based index. Missing keys,
//!   out-of-range indices, and type mismatches all `fail` loudly — there
//!   are no silent defaults.
//! * **`round_to` is decimal rounding with ties-to-even** (banker's
//!   rounding), applied to the exact binary value of the input — matching
//!   Python's built-in `round(x, places)`. See [`round_to_impl`].

use super::RuntimeValue;

/// Nesting depth cap for the recursive-descent parser and serializer.
/// Deeply nested inputs fail loudly instead of overflowing the stack.
const MAX_DEPTH: usize = 512;

// ---------------------------------------------------------------------------
// Json ADT value constructors / inspectors
// ---------------------------------------------------------------------------

fn jnull() -> RuntimeValue {
    RuntimeValue::Adt {
        ctor: "JNull".to_string(),
        fields: Vec::new(),
        field_names: None,
    }
}

fn jbool(value: bool) -> RuntimeValue {
    RuntimeValue::Adt {
        ctor: "JBool".to_string(),
        fields: vec![RuntimeValue::Bool(value)],
        field_names: None,
    }
}

pub(super) fn jnum(value: f64) -> RuntimeValue {
    RuntimeValue::Adt {
        ctor: "JNum".to_string(),
        fields: vec![RuntimeValue::float64(value)],
        field_names: None,
    }
}

pub(super) fn jstr(value: String) -> RuntimeValue {
    RuntimeValue::Adt {
        ctor: "JStr".to_string(),
        fields: vec![RuntimeValue::String(value)],
        field_names: None,
    }
}

pub(super) fn jlist(items: Vec<RuntimeValue>) -> RuntimeValue {
    RuntimeValue::Adt {
        ctor: "JList".to_string(),
        fields: vec![RuntimeValue::List(items)],
        field_names: None,
    }
}

pub(super) fn jdict(entries: Vec<(String, RuntimeValue)>) -> RuntimeValue {
    RuntimeValue::Adt {
        ctor: "JDict".to_string(),
        fields: vec![RuntimeValue::Dict(
            entries
                .into_iter()
                .map(|(key, value)| (RuntimeValue::String(key), value))
                .collect(),
        )],
        field_names: None,
    }
}

/// One-word description of a Json node for diagnostics ("number",
/// "string", ...), or an error if the value is not a Json ADT value.
fn json_kind(value: &RuntimeValue) -> Result<&'static str, String> {
    match value {
        RuntimeValue::Adt { ctor, .. } => match ctor.as_str() {
            "JNull" => Ok("null"),
            "JBool" => Ok("bool"),
            "JNum" => Ok("number"),
            "JStr" => Ok("string"),
            "JList" => Ok("list"),
            "JDict" => Ok("dict"),
            other => Err(format!(
                "expected a Json value (JNull/JBool/JNum/JStr/JList/JDict), got constructor `{other}`"
            )),
        },
        other => Err(format!("expected a Json value, got {other:?}")),
    }
}

/// Validate that a runtime value is a well-formed Json ADT value (correct
/// constructor names and field shapes, recursively). Constructors are
/// checker-enforced in typed programs; this guards the dynamically-typed
/// eval paths so a malformed value fails at the builtin boundary instead
/// of deep inside serialization.
pub(super) fn ensure_json_value(value: &RuntimeValue) -> Result<(), String> {
    ensure_json_value_depth(value, 0)
}

fn ensure_json_value_depth(value: &RuntimeValue, depth: usize) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err(format!("Json value exceeds maximum depth {MAX_DEPTH}"));
    }
    let RuntimeValue::Adt { ctor, fields, .. } = value else {
        return Err(format!("expected a Json value, got {value:?}"));
    };
    match (ctor.as_str(), fields.as_slice()) {
        ("JNull", []) => Ok(()),
        ("JBool", [RuntimeValue::Bool(_)]) => Ok(()),
        ("JNum", [RuntimeValue::Scalar(payload)]) if payload.dtype().is_float() => Ok(()),
        ("JStr", [RuntimeValue::String(_)]) => Ok(()),
        ("JList", [RuntimeValue::List(items)]) => {
            for item in items {
                ensure_json_value_depth(item, depth + 1)?;
            }
            Ok(())
        }
        ("JDict", [RuntimeValue::Dict(entries)]) => {
            for (key, item) in entries {
                let RuntimeValue::String(_) = key else {
                    return Err(format!("JDict keys must be strings, got {key:?}"));
                };
                ensure_json_value_depth(item, depth + 1)?;
            }
            Ok(())
        }
        ("JNull" | "JBool" | "JNum" | "JStr" | "JList" | "JDict", _) => Err(format!(
            "malformed Json value: constructor `{ctor}` has unexpected fields {fields:?}"
        )),
        (other, _) => Err(format!(
            "expected a Json value (JNull/JBool/JNum/JStr/JList/JDict), got constructor `{other}`"
        )),
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

struct Parser<'a> {
    bytes: &'a [u8],
    text: &'a str,
    pos: usize,
}

/// Parse a JSON document into a Json ADT value. Strict RFC 8259: one
/// top-level value, no trailing content, no trailing commas, no comments,
/// no NaN/Infinity tokens. Loud errors carry the byte offset.
pub(super) fn parse_json_text(text: &str) -> Result<RuntimeValue, String> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        text,
        pos: 0,
    };
    parser.skip_ws();
    if parser.pos >= parser.bytes.len() {
        return Err("parse_json: empty input (expected a JSON value)".to_string());
    }
    let value = parser.parse_value(0)?;
    parser.skip_ws();
    if parser.pos < parser.bytes.len() {
        return Err(format!(
            "parse_json: trailing content at byte {} (after the top-level value)",
            parser.pos
        ));
    }
    Ok(value)
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while let Some(b) = self.bytes.get(self.pos) {
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn err(&self, message: &str) -> String {
        format!("parse_json: {message} at byte {}", self.pos)
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), String> {
        if self.peek() == Some(expected) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected `{}`", expected as char)))
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<RuntimeValue, String> {
        if depth > MAX_DEPTH {
            return Err(self.err(&format!("nesting exceeds maximum depth {MAX_DEPTH}")));
        }
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(depth),
            Some(b'[') => self.parse_array(depth),
            Some(b'"') => Ok(jstr(self.parse_string()?)),
            Some(b't') => self.parse_keyword("true", jbool(true)),
            Some(b'f') => self.parse_keyword("false", jbool(false)),
            Some(b'n') => self.parse_keyword("null", jnull()),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(other) => Err(self.err(&format!(
                "unexpected character `{}` (expected a JSON value)",
                other as char
            ))),
            None => Err(self.err("unexpected end of input (expected a JSON value)")),
        }
    }

    fn parse_keyword(
        &mut self,
        keyword: &str,
        value: RuntimeValue,
    ) -> Result<RuntimeValue, String> {
        if self.text[self.pos..].starts_with(keyword) {
            self.pos += keyword.len();
            Ok(value)
        } else {
            Err(self.err(&format!("invalid token (expected `{keyword}`)")))
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<RuntimeValue, String> {
        self.expect_byte(b'{')?;
        let mut entries: Vec<(RuntimeValue, RuntimeValue)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(RuntimeValue::Adt {
                ctor: "JDict".to_string(),
                fields: vec![RuntimeValue::Dict(entries)],
                field_names: None,
            });
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected a string object key"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect_byte(b':')?;
            let value = self.parse_value(depth + 1)?;
            // Duplicate keys: keep the first occurrence's position with the
            // last occurrence's value (matches `upsert_dict_entry` and
            // Python's `json.loads`).
            if let Some(slot) = entries
                .iter_mut()
                .find(|(existing, _)| matches!(existing, RuntimeValue::String(k) if *k == key))
            {
                slot.1 = value;
            } else {
                entries.push((RuntimeValue::String(key), value));
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(RuntimeValue::Adt {
                        ctor: "JDict".to_string(),
                        fields: vec![RuntimeValue::Dict(entries)],
                        field_names: None,
                    });
                }
                _ => return Err(self.err("expected `,` or `}` in object")),
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<RuntimeValue, String> {
        self.expect_byte(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(jlist(items));
        }
        loop {
            let value = self.parse_value(depth + 1)?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(jlist(items));
                }
                _ => return Err(self.err("expected `,` or `]` in array")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect_byte(b'"')?;
        let mut out = String::new();
        loop {
            let Some(b) = self.peek() else {
                return Err(self.err("unterminated string"));
            };
            match b {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let Some(esc) = self.peek() else {
                        return Err(self.err("unterminated escape sequence"));
                    };
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.parse_hex4()?;
                            let ch = if (0xD800..=0xDBFF).contains(&unit) {
                                // High surrogate: require a following \uXXXX
                                // low surrogate and combine.
                                if self.peek() == Some(b'\\')
                                    && self.bytes.get(self.pos + 1) == Some(&b'u')
                                {
                                    self.pos += 2;
                                    let low = self.parse_hex4()?;
                                    if !(0xDC00..=0xDFFF).contains(&low) {
                                        return Err(
                                            self.err("invalid low surrogate in \\u escape pair")
                                        );
                                    }
                                    let combined =
                                        0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
                                    char::from_u32(combined)
                                        .ok_or_else(|| self.err("invalid surrogate pair"))?
                                } else {
                                    return Err(self.err("lone high surrogate in \\u escape"));
                                }
                            } else if (0xDC00..=0xDFFF).contains(&unit) {
                                return Err(self.err("lone low surrogate in \\u escape"));
                            } else {
                                char::from_u32(unit)
                                    .ok_or_else(|| self.err("invalid \\u escape"))?
                            };
                            out.push(ch);
                        }
                        other => {
                            return Err(self.err(&format!("invalid escape `\\{}`", other as char)));
                        }
                    }
                }
                0x00..=0x1F => {
                    return Err(self.err("unescaped control character in string"));
                }
                _ => {
                    // Consume one full UTF-8 character (input is a &str, so
                    // boundaries are valid by construction).
                    let ch = self.text[self.pos..]
                        .chars()
                        .next()
                        .ok_or_else(|| self.err("invalid UTF-8 in string"))?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, String> {
        let end = self.pos + 4;
        if end > self.bytes.len() {
            return Err(self.err("truncated \\u escape (need 4 hex digits)"));
        }
        let hex = &self.text[self.pos..end];
        let value = u32::from_str_radix(hex, 16)
            .map_err(|_| self.err("invalid \\u escape (need 4 hex digits)"))?;
        self.pos = end;
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<RuntimeValue, String> {
        let start = self.pos;
        // Validate against the JSON number grammar
        // (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`), then hand the
        // validated token to Rust's correctly-rounded `f64::from_str`.
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.err("invalid number (expected a digit)")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("invalid number (expected a digit after `.`)"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("invalid number (expected a digit in exponent)"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let token = &self.text[start..self.pos];
        let value: f64 = token
            .parse()
            .map_err(|_| self.err(&format!("invalid number `{token}`")))?;
        // Overflowing literals (e.g. `1e999`) parse to +/-inf, which JSON
        // cannot round-trip; reject rather than store a non-finite number.
        if !value.is_finite() {
            return Err(format!(
                "parse_json: number `{token}` overflows f64 at byte {start}"
            ));
        }
        Ok(jnum(value))
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Shortest-round-trip formatting for a finite f64, as a JSON number token.
///
/// Rust's `{:?}` float formatter emits the shortest decimal string that
/// parses back to exactly the same f64 (`1.0`, `0.1`, `2.675`, `1e-9`,
/// `1.7976931348623157e308`), switching to exponent form only when it is
/// shorter. Every finite output is a valid JSON number. This is the
/// fidelity contract the print channel breaks (chelis#748: the
/// near-integer arm's near-zero collapse and the fixed-16-significant-
/// digit starvation) — do not route number
/// formatting back through `render_value`/`to_string`.
fn format_f64_json(value: f64) -> Result<String, String> {
    if !value.is_finite() {
        return Err(format!(
            "to_json: JSON cannot represent non-finite number `{value}`"
        ));
    }
    Ok(format!("{value:?}"))
}

fn escape_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            // Non-ASCII characters are emitted as raw UTF-8 (no \u
            // escaping); the output string is UTF-8 by construction.
            ch => out.push(ch),
        }
    }
    out.push('"');
}

/// Serialize a Json ADT value to compact JSON text (no whitespace),
/// insertion-order keys, shortest-round-trip numbers. Deterministic:
/// equal values always produce identical bytes.
pub(super) fn json_value_to_text(value: &RuntimeValue) -> Result<String, String> {
    let mut out = String::new();
    write_json_value(&mut out, value, 0)?;
    Ok(out)
}

fn write_json_value(out: &mut String, value: &RuntimeValue, depth: usize) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err(format!("to_json: value exceeds maximum depth {MAX_DEPTH}"));
    }
    let RuntimeValue::Adt { ctor, fields, .. } = value else {
        return Err(format!("to_json: expected a Json value, got {value:?}"));
    };
    match (ctor.as_str(), fields.as_slice()) {
        ("JNull", []) => {
            out.push_str("null");
            Ok(())
        }
        ("JBool", [RuntimeValue::Bool(b)]) => {
            out.push_str(if *b { "true" } else { "false" });
            Ok(())
        }
        ("JNum", [RuntimeValue::Scalar(payload)]) if payload.dtype().is_float() => {
            out.push_str(&format_f64_json(payload.bits().as_f64())?);
            Ok(())
        }
        ("JStr", [RuntimeValue::String(s)]) => {
            escape_json_string(out, s);
            Ok(())
        }
        ("JList", [RuntimeValue::List(items)]) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_json_value(out, item, depth + 1)?;
            }
            out.push(']');
            Ok(())
        }
        ("JDict", [RuntimeValue::Dict(entries)]) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                let RuntimeValue::String(key) = key else {
                    return Err(format!("to_json: JDict keys must be strings, got {key:?}"));
                };
                escape_json_string(out, key);
                out.push(':');
                write_json_value(out, item, depth + 1)?;
            }
            out.push('}');
            Ok(())
        }
        _ => Err(format!(
            "to_json: malformed Json value (constructor `{ctor}` with fields {fields:?})"
        )),
    }
}

// ---------------------------------------------------------------------------
// Path navigation
// ---------------------------------------------------------------------------

fn split_path<'p>(builtin: &str, path: &'p str) -> Result<Vec<&'p str>, String> {
    if path.is_empty() {
        return Err(format!("{builtin}: path must be non-empty"));
    }
    let segments: Vec<&str> = path.split('.').collect();
    if segments.iter().any(|segment| segment.is_empty()) {
        return Err(format!(
            "{builtin}: path `{path}` has an empty segment (paths are dot-separated keys/indices)"
        ));
    }
    Ok(segments)
}

fn dict_available_keys(entries: &[(RuntimeValue, RuntimeValue)]) -> String {
    const MAX_LISTED: usize = 12;
    let mut keys: Vec<String> = entries
        .iter()
        .filter_map(|(key, _)| match key {
            RuntimeValue::String(k) => Some(format!("`{k}`")),
            _ => None,
        })
        .collect();
    let elided = keys.len().saturating_sub(MAX_LISTED);
    keys.truncate(MAX_LISTED);
    if elided > 0 {
        keys.push(format!("... {elided} more"));
    }
    keys.join(", ")
}

/// Resolve a dot-separated path against a Json value. Every failure names
/// the builtin, the full path, and the failing segment.
fn get_path<'v>(
    builtin: &str,
    value: &'v RuntimeValue,
    path: &str,
) -> Result<&'v RuntimeValue, String> {
    let segments = split_path(builtin, path)?;
    let mut current = value;
    for segment in segments {
        let RuntimeValue::Adt { ctor, fields, .. } = current else {
            return Err(format!(
                "{builtin}: path `{path}`: expected a Json value, got {current:?}"
            ));
        };
        match (ctor.as_str(), fields.as_slice()) {
            ("JDict", [RuntimeValue::Dict(entries)]) => {
                let found = entries.iter().find_map(|(key, item)| match key {
                    RuntimeValue::String(k) if k == segment => Some(item),
                    _ => None,
                });
                match found {
                    Some(item) => current = item,
                    None => {
                        return Err(format!(
                            "{builtin}: path `{path}`: key `{segment}` not found; available keys: {}",
                            dict_available_keys(entries)
                        ));
                    }
                }
            }
            ("JList", [RuntimeValue::List(items)]) => {
                let index: usize = segment.parse().map_err(|_| {
                    format!(
                        "{builtin}: path `{path}`: segment `{segment}` is not a valid \
                         list index (the node here is a list of {} elements)",
                        items.len()
                    )
                })?;
                current = items.get(index).ok_or_else(|| {
                    format!(
                        "{builtin}: path `{path}`: index {index} out of range for \
                         list of {} elements",
                        items.len()
                    )
                })?;
            }
            (other_ctor, _) => {
                let kind = match other_ctor {
                    "JNull" => "null",
                    "JBool" => "a bool",
                    "JNum" => "a number",
                    "JStr" => "a string",
                    _ => "a non-container value",
                };
                return Err(format!(
                    "{builtin}: path `{path}`: cannot descend into segment `{segment}`: \
                     the node here is {kind}, not an object or list"
                ));
            }
        }
    }
    Ok(current)
}

pub(super) fn json_f64_at(value: &RuntimeValue, path: &str) -> Result<f64, String> {
    let node = get_path("json_f64", value, path)?;
    match node {
        RuntimeValue::Adt { ctor, fields, .. } if ctor == "JNum" => match fields.as_slice() {
            [RuntimeValue::Scalar(payload)] if payload.dtype().is_float() => {
                Ok(payload.bits().as_f64())
            }
            _ => Err(format!(
                "json_f64: path `{path}`: malformed JNum fields {fields:?}"
            )),
        },
        other => Err(format!(
            "json_f64: path `{path}`: expected a number, got {}",
            json_kind(other)?
        )),
    }
}

pub(super) fn json_str_at(value: &RuntimeValue, path: &str) -> Result<String, String> {
    let node = get_path("json_str", value, path)?;
    match node {
        RuntimeValue::Adt { ctor, fields, .. } if ctor == "JStr" => match fields.as_slice() {
            [RuntimeValue::String(s)] => Ok(s.clone()),
            _ => Err(format!(
                "json_str: path `{path}`: malformed JStr fields {fields:?}"
            )),
        },
        other => Err(format!(
            "json_str: path `{path}`: expected a string, got {}",
            json_kind(other)?
        )),
    }
}

pub(super) fn json_list_at(value: &RuntimeValue, path: &str) -> Result<Vec<RuntimeValue>, String> {
    let node = get_path("json_list", value, path)?;
    match node {
        RuntimeValue::Adt { ctor, fields, .. } if ctor == "JList" => match fields.as_slice() {
            [RuntimeValue::List(items)] => Ok(items.clone()),
            _ => Err(format!(
                "json_list: path `{path}`: malformed JList fields {fields:?}"
            )),
        },
        other => Err(format!(
            "json_list: path `{path}`: expected a list, got {}",
            json_kind(other)?
        )),
    }
}

pub(super) fn json_f64s_at(value: &RuntimeValue, path: &str) -> Result<Vec<f64>, String> {
    let node = get_path("json_f64s", value, path)?;
    let items = match node {
        RuntimeValue::Adt { ctor, fields, .. } if ctor == "JList" => match fields.as_slice() {
            [RuntimeValue::List(items)] => items,
            _ => {
                return Err(format!(
                    "json_f64s: path `{path}`: malformed JList fields {fields:?}"
                ));
            }
        },
        other => {
            return Err(format!(
                "json_f64s: path `{path}`: expected a list of numbers, got {}",
                json_kind(other)?
            ));
        }
    };
    let mut out = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item {
            RuntimeValue::Adt { ctor, fields, .. } if ctor == "JNum" => match fields.as_slice() {
                [RuntimeValue::Scalar(payload)] if payload.dtype().is_float() => {
                    out.push(payload.bits().as_f64());
                }
                _ => {
                    return Err(format!(
                        "json_f64s: path `{path}`: malformed JNum fields at index {index}"
                    ));
                }
            },
            other => {
                return Err(format!(
                    "json_f64s: path `{path}`: element {index} is {}, not a number",
                    json_kind(other)?
                ));
            }
        }
    }
    Ok(out)
}

/// Set `new_value` at a dot-separated path, returning the updated Json
/// value (inputs are immutable). Output-assembly semantics:
///
/// * a missing key on an intermediate `JDict` auto-creates a nested
///   `JDict` (like `mkdir -p`) — this is a constructor, not a silent
///   default: the write destination is being built;
/// * the final key on a `JDict` upserts (new keys append, preserving
///   insertion order);
/// * an all-digits segment on a `JList` replaces an EXISTING element
///   (out-of-range indices fail; `json_set` never grows a list);
/// * descending into a non-container (number/string/bool/null) fails.
pub(super) fn json_set_at(
    value: &RuntimeValue,
    path: &str,
    new_value: &RuntimeValue,
) -> Result<RuntimeValue, String> {
    let segments = split_path("json_set", path)?;
    set_path_rec(value, path, &segments, new_value)
}

fn set_path_rec(
    current: &RuntimeValue,
    full_path: &str,
    segments: &[&str],
    new_value: &RuntimeValue,
) -> Result<RuntimeValue, String> {
    let segment = segments[0];
    let rest = &segments[1..];
    let RuntimeValue::Adt { ctor, fields, .. } = current else {
        return Err(format!(
            "json_set: path `{full_path}`: expected a Json value, got {current:?}"
        ));
    };
    match (ctor.as_str(), fields.as_slice()) {
        ("JDict", [RuntimeValue::Dict(entries)]) => {
            let mut entries = entries.clone();
            let existing = entries.iter_mut().find_map(|(key, item)| match key {
                RuntimeValue::String(k) if k == segment => Some(item),
                _ => None,
            });
            match (existing, rest.is_empty()) {
                (Some(slot), true) => {
                    *slot = new_value.clone();
                }
                (Some(slot), false) => {
                    *slot = set_path_rec(slot, full_path, rest, new_value)?;
                }
                (None, true) => {
                    entries.push((RuntimeValue::String(segment.to_string()), new_value.clone()));
                }
                (None, false) => {
                    // Auto-create the missing intermediate object, then
                    // recurse into the empty dict to build the rest.
                    let built = set_path_rec(&jdict(Vec::new()), full_path, rest, new_value)?;
                    entries.push((RuntimeValue::String(segment.to_string()), built));
                }
            }
            Ok(RuntimeValue::Adt {
                ctor: "JDict".to_string(),
                fields: vec![RuntimeValue::Dict(entries)],
                field_names: None,
            })
        }
        ("JList", [RuntimeValue::List(items)]) => {
            let index: usize = segment.parse().map_err(|_| {
                format!(
                    "json_set: path `{full_path}`: segment `{segment}` is not a valid \
                     list index (the node here is a list of {} elements)",
                    items.len()
                )
            })?;
            if index >= items.len() {
                return Err(format!(
                    "json_set: path `{full_path}`: index {index} out of range for list \
                     of {} elements (json_set replaces existing elements only)",
                    items.len()
                ));
            }
            let mut items = items.clone();
            items[index] = if rest.is_empty() {
                new_value.clone()
            } else {
                set_path_rec(&items[index], full_path, rest, new_value)?
            };
            Ok(jlist(items))
        }
        (other_ctor, _) => Err(format!(
            "json_set: path `{full_path}`: cannot descend into segment `{segment}`: \
             the node here is `{other_ctor}`, not an object or list"
        )),
    }
}

// ---------------------------------------------------------------------------
// round_to
// ---------------------------------------------------------------------------

/// Decimal rounding with **ties-to-even** (banker's rounding), applied to
/// the exact binary value of `x` — the same semantics as Python's
/// `round(x, places)` and IEEE 754 `roundTiesToEven` at a decimal digit
/// boundary. Implemented via Rust's correctly-rounded fixed-precision
/// float formatter plus a correctly-rounded parse, so there is no
/// scale-multiply double-rounding: `round_to(2.675, 2) == 2.67` (2.675 is
/// actually 2.67499999999999982...), `round_to(0.125, 2) == 0.12` and
/// `round_to(0.375, 2) == 0.38` (exact ties go to the even digit).
///
/// * `places` must be in `0..=100` (negative places are not supported —
///   fail loudly rather than guess at tens-rounding semantics);
/// * non-finite inputs pass through unchanged (matches Python).
pub(super) fn round_to_impl(x: f64, places: i64) -> Result<f64, String> {
    if !(0..=100).contains(&places) {
        return Err(format!(
            "round_to: places must be in 0..=100, got {places} \
             (negative decimal places are not supported)"
        ));
    }
    if !x.is_finite() {
        return Ok(x);
    }
    let formatted = format!("{x:.prec$}", prec = places as usize);
    formatted
        .parse::<f64>()
        .map_err(|err| format!("round_to: internal error re-parsing `{formatted}`: {err}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> RuntimeValue {
        parse_json_text(text).expect("parse should succeed")
    }

    fn to_text(value: &RuntimeValue) -> String {
        json_value_to_text(value).expect("serialize should succeed")
    }

    // -- parsing ----------------------------------------------------------

    #[test]
    fn parses_scalars() {
        assert_eq!(to_text(&parse("null")), "null");
        assert_eq!(to_text(&parse("true")), "true");
        assert_eq!(to_text(&parse("false")), "false");
        assert_eq!(to_text(&parse("0")), "0.0");
        assert_eq!(to_text(&parse("-2.5e3")), "-2500.0");
        assert_eq!(to_text(&parse("\"hi\"")), "\"hi\"");
    }

    #[test]
    fn parses_nested_structures() {
        let value = parse(r#"{"a": {"b": [1, 2.5, {"c": "x"}]}, "d": null}"#);
        assert_eq!(
            to_text(&value),
            r#"{"a":{"b":[1.0,2.5,{"c":"x"}]},"d":null}"#
        );
        assert_eq!(json_f64_at(&value, "a.b.1").unwrap(), 2.5);
        assert_eq!(json_str_at(&value, "a.b.2.c").unwrap(), "x");
    }

    #[test]
    fn parses_string_escapes() {
        let value = parse(r#""a\"b\\c\/d\b\f\n\r\t\u00e9\ud83d\ude00""#);
        match &value {
            RuntimeValue::Adt { fields, .. } => match &fields[0] {
                RuntimeValue::String(s) => {
                    assert_eq!(s, "a\"b\\c/d\u{8}\u{c}\n\r\t\u{e9}\u{1F600}");
                }
                other => panic!("expected string, got {other:?}"),
            },
            other => panic!("expected JStr, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_keys_last_wins_first_position() {
        let value = parse(r#"{"a": 1, "b": 2, "a": 3}"#);
        assert_eq!(to_text(&value), r#"{"a":3.0,"b":2.0}"#);
    }

    #[test]
    fn malformed_inputs_fail_loudly() {
        for (input, fragment) in [
            ("", "empty input"),
            ("{", "expected a string object key"),
            ("[1, 2,]", "expected a JSON value"),
            ("{\"a\": 1,}", "expected a string object key"),
            ("[1, 2] junk", "trailing content"),
            ("NaN", "unexpected character"),
            ("Infinity", "unexpected character"),
            ("nul", "expected `null`"),
            ("\"unterminated", "unterminated string"),
            ("\"\\ud800\"", "lone high surrogate"),
            ("\"\\udc00 alone\"", "lone low surrogate"),
            ("\"\\x\"", "invalid escape"),
            ("\"ctrl \u{0001}\"", "unescaped control character"),
            ("01", "trailing content"),
            ("1.", "expected a digit after `.`"),
            ("1e", "expected a digit in exponent"),
            ("-", "expected a digit"),
            ("1e999", "overflows f64"),
        ] {
            let err = parse_json_text(input).expect_err(input);
            assert!(
                err.contains(fragment),
                "input {input:?}: error `{err}` should contain `{fragment}`"
            );
        }
    }

    #[test]
    fn depth_cap_fails_loudly() {
        let deep = "[".repeat(MAX_DEPTH + 2) + &"]".repeat(MAX_DEPTH + 2);
        let err = parse_json_text(&deep).expect_err("deep nesting");
        assert!(err.contains("maximum depth"), "got `{err}`");
    }

    // -- serialization / f64 fidelity -------------------------------------

    #[test]
    fn f64_round_trip_is_bit_exact() {
        // The #748 failure classes are explicit here: 1e-9 (the
        // near-integer print arm's near-zero collapse victim) and
        // 17-significant-digit values (starved by a fixed-16-digit
        // format) must round-trip exactly.
        for x in [
            0.0,
            -0.0,
            1.0,
            0.1,
            1.0 / 3.0,
            1e-9,
            -1e-9,
            0.30000000000000004,
            2.675,
            9.869604401089358,
            1.7976931348623157e308,
            5e-324,
            -5e-324,
            123456789.12345679,
            1e16,
            1e-7,
        ] {
            let text = format_f64_json(x).expect("finite");
            let parsed: f64 = text.parse().expect("round-trip parse");
            assert_eq!(
                parsed.to_bits(),
                x.to_bits(),
                "value {x:?} formatted as `{text}` did not round-trip"
            );
        }
    }

    #[test]
    fn number_formatting_is_shortest_form() {
        assert_eq!(to_text(&jnum(1.0)), "1.0");
        assert_eq!(to_text(&jnum(0.1)), "0.1");
        assert_eq!(to_text(&jnum(-0.0)), "-0.0");
        assert_eq!(to_text(&jnum(1e-9)), "1e-9");
        assert_eq!(to_text(&jnum(0.30000000000000004)), "0.30000000000000004");
    }

    #[test]
    fn non_finite_numbers_fail_serialization() {
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = json_value_to_text(&jnum(x)).expect_err("non-finite");
            assert!(err.contains("non-finite"), "got `{err}`");
        }
    }

    #[test]
    fn string_escaping_round_trips() {
        let original = "quote\" backslash\\ newline\n tab\t ctrl\u{0001} é 😀";
        let text = to_text(&jstr(original.to_string()));
        assert_eq!(
            text,
            "\"quote\\\" backslash\\\\ newline\\n tab\\t ctrl\\u0001 é 😀\""
        );
        let reparsed = parse(&text);
        match &reparsed {
            RuntimeValue::Adt { fields, .. } => match &fields[0] {
                RuntimeValue::String(s) => assert_eq!(s, original),
                other => panic!("expected string, got {other:?}"),
            },
            other => panic!("expected JStr, got {other:?}"),
        }
    }

    #[test]
    fn key_order_is_insertion_order_and_deterministic() {
        // Parse preserves document order (not sorted).
        let value = parse(r#"{"zeta": 1, "alpha": 2, "mid": 3}"#);
        let first = to_text(&value);
        assert_eq!(first, r#"{"zeta":1.0,"alpha":2.0,"mid":3.0}"#);
        // Serialize -> parse -> serialize is byte-stable.
        assert_eq!(to_text(&parse(&first)), first);
        // jdict/json_set append new keys in construction order.
        let built = jdict(vec![("b".to_string(), jnum(1.0))]);
        let built = json_set_at(&built, "a", &jnum(2.0)).unwrap();
        assert_eq!(to_text(&built), r#"{"b":1.0,"a":2.0}"#);
    }

    // -- accessors ---------------------------------------------------------

    #[test]
    fn accessor_error_paths_are_loud_and_named() {
        let value = parse(r#"{"rates": {"usd": 1.25}, "tags": ["a", "b"], "n": 3}"#);
        let missing = json_f64_at(&value, "rates.eur").expect_err("missing key");
        assert!(missing.contains("json_f64"), "got `{missing}`");
        assert!(missing.contains("key `eur` not found"), "got `{missing}`");
        assert!(
            missing.contains("`usd`"),
            "available keys listed: `{missing}`"
        );

        let mismatch = json_f64_at(&value, "tags.0").expect_err("wrong type");
        assert!(
            mismatch.contains("expected a number, got string"),
            "got `{mismatch}`"
        );

        let through_leaf = json_f64_at(&value, "n.deeper").expect_err("descend into number");
        assert!(
            through_leaf.contains("cannot descend"),
            "got `{through_leaf}`"
        );

        let bad_index = json_str_at(&value, "tags.7").expect_err("index out of range");
        assert!(
            bad_index.contains("index 7 out of range"),
            "got `{bad_index}`"
        );

        let non_numeric_index = json_str_at(&value, "tags.first").expect_err("non-numeric index");
        assert!(
            non_numeric_index.contains("not a valid list index"),
            "got `{non_numeric_index}`"
        );

        let empty = json_f64_at(&value, "").expect_err("empty path");
        assert!(empty.contains("non-empty"), "got `{empty}`");

        let empty_segment = json_f64_at(&value, "rates..usd").expect_err("empty segment");
        assert!(
            empty_segment.contains("empty segment"),
            "got `{empty_segment}`"
        );
    }

    #[test]
    fn digit_key_on_dict_is_a_key_not_an_index() {
        let value = parse(r#"{"0": 42}"#);
        assert_eq!(json_f64_at(&value, "0").unwrap(), 42.0);
    }

    #[test]
    fn json_f64s_extracts_and_rejects() {
        let value = parse(r#"{"xs": [1.5, 2.5, 3.5], "mixed": [1, "two"]}"#);
        assert_eq!(json_f64s_at(&value, "xs").unwrap(), vec![1.5, 2.5, 3.5]);
        let err = json_f64s_at(&value, "mixed").expect_err("mixed list");
        assert!(
            err.contains("element 1 is string, not a number"),
            "got `{err}`"
        );
    }

    #[test]
    fn json_list_returns_json_elements() {
        let value = parse(r#"[{"px": 1.0}, {"px": 2.0}]"#);
        let items = json_list_at(&value, "0").err();
        // Root itself is a list; path "0" descends INTO it, returning the
        // first object — which is a dict, not a list, hence the error here.
        assert!(items.is_some());
        // Fetch elements through a wrapper object instead.
        let wrapped = parse(r#"{"rows": [{"px": 1.0}, {"px": 2.0}]}"#);
        let rows = json_list_at(&wrapped, "rows").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(json_f64_at(&rows[1], "px").unwrap(), 2.0);
    }

    // -- json_set ----------------------------------------------------------

    #[test]
    fn json_set_builds_nested_output() {
        let out = jdict(Vec::new());
        let out = json_set_at(&out, "forwards.eur_usd.3m", &jnum(1.0925)).unwrap();
        let out = json_set_at(&out, "forwards.eur_usd.6m", &jnum(1.0951)).unwrap();
        let out = json_set_at(&out, "meta.count", &jnum(2.0)).unwrap();
        assert_eq!(
            to_text(&out),
            r#"{"forwards":{"eur_usd":{"3m":1.0925,"6m":1.0951}},"meta":{"count":2.0}}"#
        );
        // Overwrite keeps position.
        let out = json_set_at(&out, "forwards.eur_usd.3m", &jnum(9.9)).unwrap();
        assert_eq!(
            to_text(&out),
            r#"{"forwards":{"eur_usd":{"3m":9.9,"6m":1.0951}},"meta":{"count":2.0}}"#
        );
    }

    #[test]
    fn json_set_replaces_existing_list_elements_only() {
        let value = parse(r#"{"xs": [1, 2, 3]}"#);
        let updated = json_set_at(&value, "xs.1", &jnum(9.0)).unwrap();
        assert_eq!(to_text(&updated), r#"{"xs":[1.0,9.0,3.0]}"#);
        let oob = json_set_at(&value, "xs.3", &jnum(9.0)).expect_err("no growth");
        assert!(oob.contains("out of range"), "got `{oob}`");
    }

    #[test]
    fn json_set_rejects_descending_into_leaves() {
        let value = parse(r#"{"n": 3}"#);
        let err = json_set_at(&value, "n.deeper", &jnum(1.0)).expect_err("leaf");
        assert!(err.contains("cannot descend"), "got `{err}`");
    }

    // -- round_to ----------------------------------------------------------

    #[test]
    fn round_to_ties_go_to_even() {
        // Exact binary ties — the digit boundary value is representable
        // exactly, so the tie-break rule is observable.
        assert_eq!(round_to_impl(0.5, 0).unwrap(), 0.0);
        assert_eq!(round_to_impl(1.5, 0).unwrap(), 2.0);
        assert_eq!(round_to_impl(2.5, 0).unwrap(), 2.0);
        assert_eq!(round_to_impl(-2.5, 0).unwrap(), -2.0);
        assert_eq!(round_to_impl(0.125, 2).unwrap(), 0.12);
        assert_eq!(round_to_impl(0.375, 2).unwrap(), 0.38);
        assert_eq!(round_to_impl(-0.125, 2).unwrap(), -0.12);
    }

    #[test]
    fn round_to_rounds_the_exact_binary_value() {
        // 2.675 is stored as 2.67499999999999982...; correct decimal
        // rounding yields 2.67 (matching Python's round(2.675, 2)), NOT
        // the 2.68 a naive scale-multiply would produce.
        assert_eq!(round_to_impl(2.675, 2).unwrap(), 2.67);
        // 0.135 is stored slightly ABOVE the tie (0.13500000000000001);
        // it rounds up, tie-break not involved. Same for 1.0055
        // (1.00550000000000006). Both match Python's round.
        assert_eq!(round_to_impl(0.135, 2).unwrap(), 0.14);
        assert_eq!(round_to_impl(1.0055, 3).unwrap(), 1.006);
    }

    #[test]
    fn round_to_general_cases() {
        assert_eq!(round_to_impl(8.76543215, 4).unwrap(), 8.7654);
        assert_eq!(round_to_impl(-8.76543215, 4).unwrap(), -8.7654);
        assert_eq!(round_to_impl(1234.5678, 0).unwrap(), 1235.0);
        assert_eq!(round_to_impl(1e-9, 2).unwrap(), 0.0);
        // places beyond the value's precision: identity.
        assert_eq!(round_to_impl(0.1, 20).unwrap(), 0.1);
    }

    #[test]
    fn round_to_non_finite_passes_through() {
        assert!(round_to_impl(f64::NAN, 2).unwrap().is_nan());
        assert_eq!(round_to_impl(f64::INFINITY, 2).unwrap(), f64::INFINITY);
        assert_eq!(
            round_to_impl(f64::NEG_INFINITY, 2).unwrap(),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn round_to_rejects_out_of_range_places() {
        for places in [-1, -2, 101, i64::MIN, i64::MAX] {
            let err = round_to_impl(1.5, places).expect_err("bad places");
            assert!(err.contains("0..=100"), "got `{err}`");
        }
    }

    // -- ensure_json_value -------------------------------------------------

    #[test]
    fn ensure_json_value_accepts_constructed_and_rejects_foreign() {
        let good = jdict(vec![
            ("a".to_string(), jnum(1.0)),
            ("b".to_string(), jlist(vec![jbool(true), jnull()])),
        ]);
        ensure_json_value(&good).expect("well-formed");
        let foreign = RuntimeValue::Adt {
            ctor: "Some".to_string(),
            fields: vec![RuntimeValue::Bool(true)],
            field_names: None,
        };
        let err = ensure_json_value(&foreign).expect_err("foreign ctor");
        assert!(err.contains("expected a Json value"), "got `{err}`");
        let malformed = RuntimeValue::Adt {
            ctor: "JNum".to_string(),
            fields: vec![RuntimeValue::Bool(true)],
            field_names: None,
        };
        let err = ensure_json_value(&malformed).expect_err("malformed JNum");
        assert!(err.contains("malformed Json value"), "got `{err}`");
    }
}
