//! Host-lane JSON I/O core (chelis#890).
//!
//! Pure helpers behind the `parse_json` / `to_json` / `json_*` / `j*` /
//! `round_to` builtins dispatched from `eval.rs`. Json values are ordinary
//! [`RuntimeValue::Adt`] values over the prelude `Json` ADT
//! (`crates/chelis-types/src/builtins.rs::register_prelude_adts`):
//!
//! ```text
//! Json = JNull | JBool bool | JInt int64 | JNum f64 | JStr string
//!      | JList List[Json] | JDict Dict[string, Json]
//! ```
//!
//! Design contract (see `docs/CHELIS_SURFACE.md` §3.8):
//!
//! * **Integers stay exact; fractions are f64.** JSON has a single number
//!   type, so int-vs-float is decided at *parse time*: a token carrying a
//!   `.`, `e` or `E` becomes `JNum` (through Rust's correctly-rounded
//!   `f64::from_str`), anything else becomes `JInt` with exact int64.
//!   This is the rule `Std.Io.Json` already uses
//!   (`packages/chelis-std/src/io/json.ch:211-219`) and the rule Python's
//!   `json` uses. `json_f64` widens `JInt` transparently, so callers that
//!   only want a number are unaffected; `json_int` reads the exact value.
//!   An integer literal too wide for int64 falls back to `JNum`, which is
//!   the one remaining lossy case (chelis#729).
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

use super::host_ops::{OrderedStringDictBuilder, dict_lookup};
use super::{RuntimeValue, truncate_rendered, truncated_debug};

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

pub(super) fn jint(value: i64) -> RuntimeValue {
    RuntimeValue::Adt {
        ctor: "JInt".to_string(),
        fields: vec![RuntimeValue::int64(value)],
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

// Lib-side callers moved to `OrderedStringDictBuilder` (chelis#891 review
// finding 11); this constructor remains for the unit tests and for the
// stacked CSV branch (chelis#903), which builds documents through it.
#[cfg_attr(not(test), allow(dead_code))]
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
            "JInt" => Ok("number"),
            "JNum" => Ok("number"),
            "JStr" => Ok("string"),
            "JList" => Ok("list"),
            "JDict" => Ok("dict"),
            other => Err(format!(
                "expected a Json value (JNull/JBool/JInt/JNum/JStr/JList/JDict), got constructor `{other}`"
            )),
        },
        other => Err(format!(
            "expected a Json value, got {}",
            truncated_debug(other)
        )),
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
        return Err(format!(
            "expected a Json value, got {}",
            truncated_debug(value)
        ));
    };
    match (ctor.as_str(), fields.as_slice()) {
        ("JNull", []) => Ok(()),
        ("JBool", [RuntimeValue::Bool(_)]) => Ok(()),
        ("JInt", [RuntimeValue::Scalar(payload)]) if payload.dtype().is_integer() => Ok(()),
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
                    return Err(format!(
                        "JDict keys must be strings, got {}",
                        truncated_debug(key)
                    ));
                };
                ensure_json_value_depth(item, depth + 1)?;
            }
            Ok(())
        }
        ("JNull" | "JBool" | "JInt" | "JNum" | "JStr" | "JList" | "JDict", _) => Err(format!(
            "malformed Json value: constructor `{ctor}` has unexpected fields {}",
            truncate_rendered(format!("{fields:?}"))
        )),
        (other, _) => Err(format!(
            "expected a Json value (JNull/JBool/JInt/JNum/JStr/JList/JDict), got constructor `{other}`"
        )),
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Scan one JSON number token
/// (`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`) at the start of
/// `bytes`. `Ok(len)` is the token length; `Err((offset, reason))`
/// pinpoints the first offending byte. Shared by [`Parser::parse_number`]
/// and the CSV numeric accessors (chelis#903), so "CSV cells parse under
/// the same number grammar as JSON" holds by shared code rather than
/// parallel maintenance.
pub(super) fn json_number_token_len(bytes: &[u8]) -> Result<usize, (usize, &'static str)> {
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

struct Parser<'a> {
    bytes: &'a [u8],
    text: &'a str,
    pos: usize,
}

/// Parse a JSON document into a Json ADT value. Strict RFC 8259: one
/// top-level value, no trailing content, no trailing commas, no comments,
/// no NaN/Infinity tokens. Loud errors carry the byte offset.
pub(super) fn parse_json_text(text: &str) -> Result<RuntimeValue, String> {
    // RFC 8259 §8.1 permits ignoring a leading BOM; strip exactly one so a
    // Windows-exported file parses instead of dying on mojibake ("`ï`"),
    // consistent with `parse_csv`'s BOM handling (chelis#891 review
    // finding 12).
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
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
            Some(_) => {
                // Decode the actual character for the diagnostic — casting a
                // single UTF-8 byte to `char` renders mojibake for anything
                // non-ASCII (chelis#891 review finding 12).
                let ch = self.text[self.pos..].chars().next().unwrap_or('\u{fffd}');
                Err(self.err(&format!(
                    "unexpected character `{ch}` (expected a JSON value)"
                )))
            }
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
        let mut entries = OrderedStringDictBuilder::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(RuntimeValue::Adt {
                ctor: "JDict".to_string(),
                fields: vec![RuntimeValue::Dict(entries.into_entries())],
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
            // last occurrence's value (the `upsert_dict_entry` semantics,
            // matching Python's `json.loads`) — via the shared hash-assisted
            // builder so a pathological 100k-key object parses in O(n)
            // (chelis#891 review findings 11 and 15).
            entries.upsert(key, value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(RuntimeValue::Adt {
                        ctor: "JDict".to_string(),
                        fields: vec![RuntimeValue::Dict(entries.into_entries())],
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
                        _ => {
                            // Decode the full character (see the finding-12
                            // note above): a multibyte char after `\\` must
                            // render as itself, not as its lead byte.
                            let ch = self.text[self.pos..].chars().next().unwrap_or('\u{fffd}');
                            return Err(self.err(&format!("invalid escape `\\{ch}`")));
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
        // Byte-wise decoding (chelis#891 review findings 1 and 9): RFC 8259
        // requires exactly 4 hex DIGITS, so a sign (`+`/`-`), whitespace, or
        // a multibyte character inside the 4-byte window is a loud parse
        // error — and slicing `self.text` at an arbitrary `pos + 4` could
        // split a multibyte character and panic on the char boundary.
        let mut value: u32 = 0;
        for &byte in &self.bytes[self.pos..end] {
            let digit = match byte {
                b'0'..=b'9' => u32::from(byte - b'0'),
                b'a'..=b'f' => u32::from(byte - b'a' + 10),
                b'A'..=b'F' => u32::from(byte - b'A' + 10),
                _ => return Err(self.err("invalid \\u escape (need 4 hex digits)")),
            };
            value = value * 16 + digit;
        }
        self.pos = end;
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<RuntimeValue, String> {
        let start = self.pos;
        // Validate against the JSON number grammar via the shared scanner
        // ([`json_number_token_len`], also the CSV cell grammar), then hand
        // the validated token to Rust's correctly-rounded `f64::from_str`.
        match json_number_token_len(&self.bytes[start..]) {
            Ok(len) => self.pos = start + len,
            Err((offset, reason)) => {
                self.pos = start + offset;
                return Err(self.err(reason));
            }
        }
        let token = &self.text[start..self.pos];
        // chelis#729: JSON has a single number type, so int-vs-float is a
        // parse-time decision. A token carrying a fraction or an exponent
        // is a float; anything else is an exact integer. This is the rule
        // `Std.Io.Json` already uses
        // (`packages/chelis-std/src/io/json.ch:211-219`) and the rule
        // Python's `json` uses, so the two lanes agree.
        //
        // Before this, every number became `JNum f64` and
        // `9007199254740993` silently returned `9007199254740992.0`.
        let is_float_token = token.contains(['.', 'e', 'E']);
        // A non-float token that does not fit i64 falls through to the f64
        // path below. That keeps input which parses today parsing, rather
        // than turning a silent narrowing into a new rejection; the
        // narrowing survives for that one case and is documented in
        // `docs/CHELIS_SURFACE.md` §3.8.
        if !is_float_token && let Ok(value) = token.parse::<i64>() {
            return Ok(jint(value));
        }
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
/// formatting back through `render_value`/`to_string`. Shared with
/// `to_csv` (chelis#903) so both serializers carry the same fidelity
/// contract by shared code.
pub(super) fn format_f64_json(value: f64) -> Result<String, String> {
    if !value.is_finite() {
        return Err(format!(
            "to_json: JSON cannot represent non-finite number `{value}`"
        ));
    }
    // Routed through the section C4 generated formatter rather than a
    // hand-rolled `format!("{value:?}")` (chelis#891 review): the bytes are
    // identical today -- verified on every boundary value in the review --
    // but a second numeric-formatting implementation makes the one-change-
    // set migration protocol unhonorable when the formatter next moves.
    Ok(chelis_types::format_element(
        chelis_types::types::Prim::F64,
        chelis_types::ElementRef::F64(value),
    ))
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
        return Err(format!(
            "to_json: expected a Json value, got {}",
            truncated_debug(value)
        ));
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
        ("JInt", [RuntimeValue::Scalar(payload)]) if payload.dtype().is_integer() => {
            // Exact: no decimal point, no f64 round-trip. This is the
            // whole point of the variant (chelis#729).
            out.push_str(&payload.bits().as_i64().to_string());
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
                    return Err(format!(
                        "to_json: JDict keys must be strings, got {}",
                        truncated_debug(key)
                    ));
                };
                escape_json_string(out, key);
                out.push(':');
                write_json_value(out, item, depth + 1)?;
            }
            out.push('}');
            Ok(())
        }
        _ => Err(format!(
            "to_json: malformed Json value (constructor `{ctor}` with fields {})",
            truncate_rendered(format!("{fields:?}"))
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

/// Parse a path segment as a list index under the strict "all-digits,
/// zero-based" contract (chelis#891 review finding 10): ASCII digits
/// only — no sign, no whitespace — and no leading zeros (`0` itself is
/// fine), so `+1` and `007` are contract errors instead of silently
/// resolving to elements 1 and 7.
fn parse_index_segment(segment: &str) -> Option<usize> {
    let bytes = segment.as_bytes();
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    if bytes.len() > 1 && bytes[0] == b'0' {
        return None;
    }
    segment.parse::<usize>().ok()
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
                "{builtin}: path `{path}`: expected a Json value, got {}",
                truncated_debug(current)
            ));
        };
        match (ctor.as_str(), fields.as_slice()) {
            ("JDict", [RuntimeValue::Dict(entries)]) => {
                // Shared lookup helper (chelis#891 review finding 15).
                let found = dict_lookup(entries, &RuntimeValue::String(segment.to_string()));
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
                let index: usize = parse_index_segment(segment).ok_or_else(|| {
                    format!(
                        "{builtin}: path `{path}`: segment `{segment}` is not a valid \
                         list index (all digits, zero-based, no leading zeros; the \
                         node here is a list of {} elements)",
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
                    "JInt" => "a number",
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
        // chelis#729: `JInt` widens transparently, mirroring
        // `Std.Io.Json`'s `json_float` (`io/json.ch:62`). Every caller
        // written against the f64-only ADT keeps working unchanged; the
        // widening is lossy only above 2^53, which is what `json_int` is
        // for.
        RuntimeValue::Adt { ctor, fields, .. } if ctor == "JInt" => match fields.as_slice() {
            [RuntimeValue::Scalar(payload)] if payload.dtype().is_integer() => {
                Ok(payload.bits().as_i64() as f64)
            }
            _ => Err(format!(
                "json_f64: path `{path}`: malformed JInt fields {fields:?}"
            )),
        },
        other => Err(format!(
            "json_f64: path `{path}`: expected a number, got {}",
            json_kind(other)?
        )),
    }
}

/// Exact int64 read. The counterpart to `json_f64` for the case the f64
/// channel cannot represent: a JSON integer beyond 2^53.
///
/// Deliberately does *not* accept `JNum`. Narrowing a float to an integer
/// is a silent-substitution shape (`loud_unsupported.md` §C1.1), so a
/// caller who reaches a fractional value here gets a diagnostic naming
/// `json_f64` rather than a truncated answer.
pub(super) fn json_int_at(value: &RuntimeValue, path: &str) -> Result<i64, String> {
    let node = get_path("json_int", value, path)?;
    match node {
        RuntimeValue::Adt { ctor, fields, .. } if ctor == "JInt" => match fields.as_slice() {
            [RuntimeValue::Scalar(payload)] if payload.dtype().is_integer() => {
                Ok(payload.bits().as_i64())
            }
            _ => Err(format!(
                "json_int: path `{path}`: malformed JInt fields {fields:?}"
            )),
        },
        RuntimeValue::Adt { ctor, .. } if ctor == "JNum" => Err(format!(
            "json_int: path `{path}`: value is a float, not an exact integer; \
             use `json_f64` to read it as f64"
        )),
        other => Err(format!(
            "json_int: path `{path}`: expected a number, got {}",
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
            // chelis#729: `JInt` widens here for the same reason it does
            // in `json_f64` -- a numeric list must not start rejecting
            // integer elements it used to accept.
            RuntimeValue::Adt { ctor, fields, .. } if ctor == "JInt" => match fields.as_slice() {
                [RuntimeValue::Scalar(payload)] if payload.dtype().is_integer() => {
                    out.push(payload.bits().as_i64() as f64);
                }
                _ => {
                    return Err(format!(
                        "json_f64s: path `{path}`: malformed JInt fields at index {index}"
                    ));
                }
            },
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
    // Cap the recursion up front (chelis#891 review finding 5): the
    // parser, serializer, and `ensure_json_value` all cap at MAX_DEPTH,
    // but `set_path_rec` recursed once per segment, so a
    // tens-of-thousands-segment path overflowed the stack.
    if segments.len() > MAX_DEPTH {
        return Err(format!(
            "json_set: path `{path}` has {} segments, exceeding the maximum depth \
             {MAX_DEPTH}",
            segments.len()
        ));
    }
    let result = set_path_iterative(value, path, &segments, new_value)?;
    // And refuse to BUILD a value deeper than every consumer (to_json,
    // parse_json, ensure_json_value) will accept: auto-created
    // intermediates plus a deep replacement value can push the result past
    // the cap even when each input is individually within it.
    ensure_json_value(&result).map_err(|err| format!("json_set: result: {err}"))?;
    Ok(result)
}

/// One level of the iterative `json_set` descend/rebuild. The container
/// contents are cloned on the way down (as the recursive form also did —
/// inputs are immutable); the rebuild folds the updated child back in on
/// the way up.
enum SetPathLevel<'p> {
    Dict {
        entries: Vec<(RuntimeValue, RuntimeValue)>,
        key: &'p str,
        existing: Option<usize>,
    },
    List {
        items: Vec<RuntimeValue>,
        index: usize,
    },
}

/// Iterative core of [`json_set_at`] (chelis#891 review finding 5): the
/// former per-segment recursion overflowed the stack on pathological
/// paths well before any depth check could fire; descending and
/// rebuilding through an explicit level stack keeps the machine stack
/// flat regardless of path length.
fn set_path_iterative(
    root: &RuntimeValue,
    full_path: &str,
    segments: &[&str],
    new_value: &RuntimeValue,
) -> Result<RuntimeValue, String> {
    let mut levels: Vec<SetPathLevel<'_>> = Vec::with_capacity(segments.len());
    // `current` walks the ORIGINAL tree; once a missing intermediate key
    // sends us into auto-create mode, every remaining level is a fresh
    // empty dict (the `mkdir -p` output-assembly semantics).
    let mut current: Option<&RuntimeValue> = Some(root);
    for (position, segment) in segments.iter().enumerate() {
        let last = position + 1 == segments.len();
        let Some(node) = current else {
            levels.push(SetPathLevel::Dict {
                entries: Vec::new(),
                key: segment,
                existing: None,
            });
            continue;
        };
        let RuntimeValue::Adt { ctor, fields, .. } = node else {
            return Err(format!(
                "json_set: path `{full_path}`: expected a Json value, got {}",
                truncated_debug(node)
            ));
        };
        match (ctor.as_str(), fields.as_slice()) {
            ("JDict", [RuntimeValue::Dict(entries)]) => {
                let existing = entries
                    .iter()
                    .position(|(key, _)| matches!(key, RuntimeValue::String(k) if k == *segment));
                current = match existing {
                    Some(index) if !last => Some(&entries[index].1),
                    // Auto-create the missing intermediate object; the
                    // remaining levels build fresh dicts.
                    None if !last => None,
                    _ => Some(node), // unused: the rebuild writes the leaf
                };
                levels.push(SetPathLevel::Dict {
                    entries: entries.clone(),
                    key: segment,
                    existing,
                });
            }
            ("JList", [RuntimeValue::List(items)]) => {
                let index: usize = parse_index_segment(segment).ok_or_else(|| {
                    format!(
                        "json_set: path `{full_path}`: segment `{segment}` is not a valid \
                         list index (all digits, zero-based, no leading zeros; the node \
                         here is a list of {} elements)",
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
                current = Some(&items[index]);
                levels.push(SetPathLevel::List {
                    items: items.clone(),
                    index,
                });
            }
            (other_ctor, _) => {
                return Err(format!(
                    "json_set: path `{full_path}`: cannot descend into segment `{segment}`: \
                     the node here is `{other_ctor}`, not an object or list"
                ));
            }
        }
    }
    // Rebuild bottom-up.
    let mut built = new_value.clone();
    while let Some(level) = levels.pop() {
        built = match level {
            SetPathLevel::Dict {
                mut entries,
                key,
                existing,
            } => {
                match existing {
                    Some(index) => entries[index].1 = built,
                    None => entries.push((RuntimeValue::String(key.to_string()), built)),
                }
                RuntimeValue::Adt {
                    ctor: "JDict".to_string(),
                    fields: vec![RuntimeValue::Dict(entries)],
                    field_names: None,
                }
            }
            SetPathLevel::List { mut items, index } => {
                items[index] = built;
                jlist(items)
            }
        };
    }
    Ok(built)
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
        // chelis#729: an integer token now round-trips as an integer.
        // This asserted "0.0" before, i.e. `to_json(parse_json(x))` was not
        // idempotent for any document containing an integer.
        assert_eq!(to_text(&parse("0")), "0");
        assert_eq!(to_text(&parse("-2.5e3")), "-2500.0");
        assert_eq!(to_text(&parse("\"hi\"")), "\"hi\"");
    }

    #[test]
    fn parses_nested_structures() {
        let value = parse(r#"{"a": {"b": [1, 2.5, {"c": "x"}]}, "d": null}"#);
        assert_eq!(to_text(&value), r#"{"a":{"b":[1,2.5,{"c":"x"}]},"d":null}"#);
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
        assert_eq!(to_text(&value), r#"{"a":3,"b":2}"#);
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
        assert_eq!(first, r#"{"zeta":1,"alpha":2,"mid":3}"#);
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
        assert_eq!(to_text(&updated), r#"{"xs":[1,9.0,3]}"#);
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

    // -- chelis#891 review probes -------------------------------------------

    /// Review findings 1 + 9: the \u escape window is decoded byte-wise —
    /// a multibyte character inside the 4-byte window is a loud parse
    /// error (previously a char-boundary PANIC), and a sign or embedded
    /// whitespace is rejected (previously `from_str_radix` accepted `+`).
    #[test]
    fn parse_hex4_is_boundary_safe_and_strict() {
        let err = parse_json_text("\"\\u00\u{e9}x\"").expect_err("multibyte in window");
        assert!(err.contains("invalid \\u escape"), "got `{err}`");
        for input in ["\"\\u+0FF\"", "\"\\u-123\"", "\"\\u 123\"", "\"\\u12 3\""] {
            let err = parse_json_text(input).expect_err(input);
            assert!(err.contains("invalid \\u escape"), "{input}: got `{err}`");
        }
        // Genuine 4-hex-digit escapes still decode.
        match parse_json_text("\"\\u00e9\"").expect("valid escape") {
            RuntimeValue::Adt { fields, .. } => match fields.as_slice() {
                [RuntimeValue::String(s)] => assert_eq!(s, "\u{e9}"),
                other => panic!("expected one string field, got {other:?}"),
            },
            other => panic!("expected JStr, got {other:?}"),
        }
    }

    /// Review finding 12: a leading BOM is skipped (RFC 8259 §8.1,
    /// consistent with `parse_csv`), and unexpected characters render as
    /// themselves, not as their UTF-8 lead byte's mojibake.
    #[test]
    fn parse_json_strips_leading_bom_and_renders_real_chars() {
        let value = parse_json_text("\u{feff}{\"a\": 1.5}").expect("BOM-prefixed JSON");
        assert_eq!(json_f64_at(&value, "a").unwrap(), 1.5);
        // Exactly one BOM is stripped; a second is a real unexpected
        // character and renders as itself.
        let err = parse_json_text("\u{feff}\u{feff}1").expect_err("double BOM");
        assert!(err.contains('\u{feff}'), "got `{err}`");
        let err = parse_json_text("\u{e9}").expect_err("stray char");
        assert!(err.contains("unexpected character `\u{e9}`"), "got `{err}`");
    }

    /// Review finding 10: list-index path segments are strictly
    /// all-digits — `+1` and `007` are contract errors, not silent
    /// element accesses.
    #[test]
    fn path_index_segments_are_strictly_all_digits() {
        let doc = parse_json_text("[10.5, 20.5, 30.5]").unwrap();
        assert_eq!(json_f64_at(&doc, "0").unwrap(), 10.5);
        for bad in ["+1", "-1", "007", "1 ", " 1", "0x1"] {
            let err = json_f64_at(&doc, bad).expect_err(bad);
            assert!(err.contains("not a valid list index"), "{bad}: got `{err}`");
            assert!(err.contains("all digits"), "{bad}: got `{err}`");
        }
        let err = json_set_at(&doc, "007", &jnum(1.5)).expect_err("leading zeros");
        assert!(err.contains("no leading zeros"), "got `{err}`");
    }

    /// Review finding 5: `json_set` caps both the path depth and the
    /// depth of the value it builds (previously a
    /// tens-of-thousands-segment path overflowed the stack, and
    /// composing within-cap inputs could build a value `to_json` then
    /// refuses).
    #[test]
    fn json_set_depth_caps_are_loud_not_crashes() {
        let deep_path = vec!["a"; 100_000].join(".");
        let err = json_set_at(&jdict(Vec::new()), &deep_path, &jnum(1.5)).expect_err("deep path");
        assert!(err.contains("exceeding the maximum depth"), "got `{err}`");

        // Composing within-cap inputs cannot BUILD an over-cap value: a
        // 450-segment path (within the segment cap) attaching a 100-deep
        // value pushes the result to depth ~550, which the result check
        // rejects (previously to_json would refuse a value json_set
        // happily built).
        let mut deep = jnum(1.5);
        for _ in 0..100 {
            deep = jlist(vec![deep]);
        }
        let path_450 = vec!["a"; 450].join(".");
        let err = json_set_at(&jdict(Vec::new()), &path_450, &deep).expect_err("over-cap result");
        assert!(err.contains("maximum depth"), "got `{err}`");
        // The same value attaches fine at a shallow path.
        json_set_at(&jdict(Vec::new()), "a.b.c", &deep).expect("within cap");
    }

    /// Review finding 11: duplicate-key upsert stays first-position /
    /// last-value across the hash-index threshold (large objects switch
    /// from the linear scan to the key -> slot index).
    #[test]
    fn large_object_duplicate_keys_upsert_across_index_threshold() {
        let mut body = String::from("{");
        for i in 0..40 {
            body.push_str(&format!("\"k{i}\": {}.5, ", i));
        }
        // Re-set the very first key after the index has kicked in.
        body.push_str("\"k0\": 99.5}");
        let value = parse_json_text(&body).expect("large object parses");
        assert_eq!(json_f64_at(&value, "k0").unwrap(), 99.5, "last value wins");
        let text = json_value_to_text(&value).unwrap();
        assert!(
            text.starts_with("{\"k0\":99.5,\"k1\":1.5"),
            "first position is kept: {}",
            &text[..40.min(text.len())]
        );
        // 40 distinct keys stay distinct.
        assert_eq!(json_f64_at(&value, "k39").unwrap(), 39.5);
    }

    // -----------------------------------------------------------------
    // chelis#729: exact integers (the `JInt` variant)
    // -----------------------------------------------------------------

    /// The defect this variant exists for. Before it, every JSON number
    /// became `JNum f64` and 2^53+1 came back as `9007199254740992.0`
    /// through both the round-trip and `json_f64`, silently.
    #[test]
    fn integers_beyond_2_53_survive_exactly() {
        let value = parse_json_text("{\"big\": 9007199254740993}").expect("parses");
        assert_eq!(json_int_at(&value, "big").unwrap(), 9007199254740993);
        assert_eq!(
            json_value_to_text(&value).unwrap(),
            "{\"big\":9007199254740993}",
            "serialization must not route the integer through f64"
        );
    }

    /// The parse-time int/float split, matching `Std.Io.Json`
    /// (`packages/chelis-std/src/io/json.ch:211-219`) and Python's `json`:
    /// a `.`, `e` or `E` makes it a float, otherwise it is an exact int.
    #[test]
    fn number_tokens_split_on_fraction_or_exponent() {
        for (text, want_int) in [
            ("0", true),
            ("-42", true),
            ("9007199254740993", true),
            ("1.5", false),
            ("1e3", false),
            ("1E3", false),
            ("-0.0", false),
            ("2.0", false),
        ] {
            let value = parse_json_text(&format!("{{\"v\": {text}}}")).expect("parses");
            let is_int = json_int_at(&value, "v").is_ok();
            assert_eq!(is_int, want_int, "`{text}` classified wrong");
        }
    }

    /// `json_f64` widens `JInt` transparently, mirroring `Std.Io.Json`'s
    /// `json_float` (`io/json.ch:62`). Every caller written against the
    /// f64-only ADT keeps working.
    #[test]
    fn json_f64_widens_an_exact_integer() {
        let value = parse_json_text("{\"n\": 42}").expect("parses");
        assert_eq!(json_f64_at(&value, "n").unwrap(), 42.0);
        // And the widening is the documented lossy case above 2^53.
        let big = parse_json_text("{\"n\": 9007199254740993}").expect("parses");
        assert_eq!(json_f64_at(&big, "n").unwrap(), 9007199254740992.0);
    }

    /// `json_int` refuses a float rather than truncating it — narrowing
    /// here would be the silent-substitution shape §C1.1 forbids.
    #[test]
    fn json_int_refuses_a_float_instead_of_truncating() {
        let value = parse_json_text("{\"n\": 1.5}").expect("parses");
        let err = json_int_at(&value, "n").expect_err("must not truncate");
        assert!(
            err.contains("not an exact integer") && err.contains("json_f64"),
            "diagnostic must name the remedy; got: {err}"
        );
    }

    /// An integer literal too wide for int64 falls back to `JNum` rather
    /// than becoming a new parse rejection: input that parsed before this
    /// change must still parse.
    #[test]
    fn integers_too_wide_for_int64_fall_back_to_f64() {
        let text = "{\"n\": 99999999999999999999999}";
        let value = parse_json_text(text).expect("must still parse, not reject");
        assert!(
            json_int_at(&value, "n").is_err(),
            "too-wide literal is not an exact int"
        );
        assert!(json_f64_at(&value, "n").unwrap() > 9.9e22);
    }

    /// Round-trip stability across both numeric variants in one document.
    #[test]
    fn mixed_numeric_document_round_trips() {
        let text = "{\"i\":-7,\"f\":1.5,\"big\":9007199254740993,\"e\":1e3}";
        let value = parse_json_text(text).expect("parses");
        let out = json_value_to_text(&value).unwrap();
        let again = parse_json_text(&out).expect("re-parses");
        assert_eq!(
            json_value_to_text(&again).unwrap(),
            out,
            "round-trip must be stable"
        );
        assert_eq!(json_int_at(&value, "i").unwrap(), -7);
        assert_eq!(json_int_at(&value, "big").unwrap(), 9007199254740993);
        assert_eq!(json_f64_at(&value, "f").unwrap(), 1.5);
    }

    /// `ensure_json_value` accepts a hand-built `JInt` and still rejects a
    /// malformed one.
    #[test]
    fn ensure_json_value_accepts_jint_and_rejects_malformed() {
        assert!(ensure_json_value(&jint(5)).is_ok());
        let malformed = RuntimeValue::Adt {
            ctor: "JInt".to_string(),
            fields: vec![RuntimeValue::String("nope".to_string())],
            field_names: None,
        };
        assert!(ensure_json_value(&malformed).is_err());
    }
}
