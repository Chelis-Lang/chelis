//! String and `str` methods and conversions of the production crates (Aeneas.Std String,
//! StringIter, Parse): `String::as_bytes` (chelis-types `context`), `&str == String` (chelis-deep
//! `authoring`), `From<&String> for String` (chelis-ir `host`), `From<&str> for Box<str>`
//! (chelis-types `unsupported`), `str::get(a..b)` (chelis-deep `extension_data`), `str::rsplit`
//! (chelis-deep `authoring::bare_name`) and `i64::from_str_radix` (chelis-deep `lexer`).

use std::num::IntErrorKind;

/// As chelis-types `context`: the UTF-8 bytes of a canonical string.
pub fn bytes_of(canonical: &String) -> Vec<u8> {
    canonical.as_bytes().to_vec()
}

/// As chelis-deep `authoring`: a `&str` compared with a `String`.
pub fn is_target(name: &str, target: &String) -> (bool, bool) {
    (name == *target, name != *target)
}

/// `From<&String> for String`, through `Into`.
pub fn owned_name(name: &String) -> String {
    let owned: String = name.into();
    owned
}

fn context_of(context: impl Into<Box<str>>) -> Box<str> {
    context.into()
}

/// As chelis-types `unsupported`: a boxed context from a `&str`.
pub fn boxed_context(context: &str) -> (usize, Box<str>) {
    let boxed = context_of(context);
    (boxed.len(), boxed)
}

/// As chelis-deep `extension_data`: the source text of a token span, when it is one.
pub fn spelling(source: Option<&str>, offset: usize, end: usize) -> Option<String> {
    source.and_then(|s| s.get(offset..end)).map(|text| text.into())
}

/// As chelis-deep `authoring::bare_name`: the last dotted segment.
pub fn bare_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// Every segment from the end.
pub fn segments_from_end(name: &str) -> Vec<&str> {
    name.rsplit('.').collect()
}

fn kind_code(kind: &IntErrorKind) -> u8 {
    match kind {
        IntErrorKind::Empty => 1,
        IntErrorKind::InvalidDigit => 2,
        IntErrorKind::PosOverflow => 3,
        IntErrorKind::NegOverflow => 4,
        _ => 5,
    }
}

/// As chelis-deep `lexer`: a hexadecimal or binary literal, or the kind of its error.
pub fn parse_radix(text: &str, radix: u32) -> Result<i64, u8> {
    i64::from_str_radix(text, radix).map_err(|e| kind_code(e.kind()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings() {
        assert_eq!(bytes_of(&"aé".to_string()), vec![97, 195, 169]);
        assert_eq!(bytes_of(&String::new()), Vec::<u8>::new());
        assert_eq!(is_target("map", &"map".to_string()), (true, false));
        assert_eq!(is_target("map", &"mapé".to_string()), (false, true));
        assert_eq!(is_target("", &String::new()), (true, false));
        assert_eq!(owned_name(&"x.y".to_string()), "x.y");
        let (len, boxed) = boxed_context("héllo");
        assert_eq!((len, &*boxed), (6, "héllo"));
    }

    #[test]
    fn slices() {
        let source = Some("héllo");
        assert_eq!(spelling(source, 1, 3), Some("é".to_string()));
        assert_eq!(spelling(source, 1, 2), None);
        assert_eq!(spelling(source, 3, 2), None);
        assert_eq!(spelling(source, 0, 6), Some("héllo".to_string()));
        assert_eq!(spelling(source, 0, 7), None);
        assert_eq!(spelling(source, 6, 6), Some(String::new()));
        assert_eq!(spelling(None, 0, 1), None);
    }

    #[test]
    fn rsplits() {
        assert_eq!(bare_name("core.ops.Add"), "Add");
        assert_eq!(bare_name("Add"), "Add");
        assert_eq!(bare_name("a."), "");
        assert_eq!(segments_from_end("a.b..c"), vec!["c", "", "b", "a"]);
        assert_eq!(segments_from_end(""), vec![""]);
    }

    #[test]
    fn radixes() {
        assert_eq!(parse_radix("ff", 16), Ok(255));
        assert_eq!(parse_radix("Ff", 16), Ok(255));
        assert_eq!(parse_radix("+7f", 16), Ok(127));
        assert_eq!(parse_radix("-ff", 16), Ok(-255));
        assert_eq!(parse_radix("7fffffffffffffff", 16), Ok(i64::MAX));
        assert_eq!(parse_radix("8000000000000000", 16), Err(3));
        assert_eq!(parse_radix("-8000000000000000", 16), Ok(i64::MIN));
        assert_eq!(parse_radix("-8000000000000001", 16), Err(4));
        assert_eq!(parse_radix("1g", 16), Err(2));
        assert_eq!(parse_radix("fffffffffffffffffg", 16), Err(3));
        assert_eq!(parse_radix("7fffffffffffffffg", 16), Err(2));
        assert_eq!(parse_radix("101", 2), Ok(5));
        assert_eq!(parse_radix("102", 2), Err(2));
        assert_eq!(parse_radix("zz", 36), Ok(1295));
        assert_eq!(parse_radix("", 16), Err(1));
        assert_eq!(parse_radix("-", 16), Err(2));
        assert_eq!(parse_radix("+", 2), Err(2));
        assert_eq!(parse_radix("0x10", 16), Err(2));
    }

    #[test]
    #[should_panic]
    fn radix_out_of_range() {
        let _ = parse_radix("1", 37);
    }
}
