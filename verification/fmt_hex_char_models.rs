//! Formatting impls of the production crates (Aeneas.Std Scalar/Fmt, Core/Fmt, FmtWithSlice):
//! `LowerHex` of `u32` (chelis-types `agreement`: `{eval_bits:#010x}`) and `u64` (chelis-types),
//! `UpperHex` of `u32` and of `&T` (chelis-deep `metadata`/`parser`: `U+{code_point:04X}`),
//! `Debug` and `Display` of `char` (chelis-deep, chelis-ir `load_store_name`: `{character:?}`), and
//! `Formatter::debug_tuple_field3_finish` (the derived `Debug` of a three-field tuple variant,
//! chelis-deep `ast`).

/// As chelis-types `agreement`: the alternate form pads after the `0x` prefix.
pub fn bits32(bits: u32) -> String {
    format!("{bits:#010x}|{bits:x}|{bits:08x}|{bits:#x}|{bits:>6x}|{bits:<6x}|")
}

/// `{:x}` of `u64`, with and without the prefix.
pub fn bits64(bits: u64) -> String {
    format!("{bits:#018x}|{bits:x}|{bits:#x}")
}

/// As chelis-deep `metadata`: a code point in upper-case hexadecimal.
pub fn code_point(code: u32) -> String {
    format!("U+{code:04X}|{code:#X}|{code:#06X}")
}

/// `UpperHex for &T` forwards to `T`.
pub fn code_point_ref(code: &u32) -> String {
    format!("U+{code:04X}")
}

/// As chelis-ir `load_store_name`: a character in `Debug`, then in `Display` (padded and
/// truncated).
pub fn character(c: char) -> String {
    format!("{c:?}|{c}|{c:>3}|{c:<3}|{c:.0}|{c:5?}")
}

/// The derived `Debug` of a three-field tuple variant (as chelis-deep `ast::ExprCarrier`).
#[derive(Debug)]
pub enum Carrier {
    Triple(u32, bool, char),
    Single(u32),
}

pub fn carrier(value: u32, flag: bool, c: char, pretty: bool) -> String {
    let triple = Carrier::Triple(value, flag, c);
    let single = Carrier::Single(value);
    if pretty { format!("{triple:#?}") } else { format!("{triple:?} {single:?}") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex() {
        assert_eq!(bits32(0x3f80), "0x00003f80|3f80|00003f80|0x3f80|  3f80|3f80  |");
        assert_eq!(bits32(0), "0x00000000|0|00000000|0x0|     0|0     |");
        assert_eq!(bits32(u32::MAX), "0xffffffff|ffffffff|ffffffff|0xffffffff|ffffffff|ffffffff|");
        assert_eq!(bits64(0x7ff8_0000_0000_0000), "0x7ff8000000000000|7ff8000000000000|0x7ff8000000000000");
        assert_eq!(bits64(10), "0x000000000000000a|a|0xa");
        assert_eq!(code_point(0x1f), "U+001F|0x1F|0x001F");
        assert_eq!(code_point(0x10ffff), "U+10FFFF|0x10FFFF|0x10FFFF");
        assert_eq!(code_point_ref(&0xab), "U+00AB");
    }

    #[test]
    fn chars() {
        assert_eq!(character('a'), "'a'|a|  a|a  ||'a'");
        assert_eq!(character('\''), "'\\''|'|  '|'  ||'\\''");
        assert_eq!(character('"'), "'\"'|\"|  \"|\"  ||'\"'");
        assert_eq!(character('\n'), "'\\n'|\n|  \n|\n  ||'\\n'");
        assert_eq!(character('é'), "'é'|é|  é|é  ||'é'");
        assert_eq!(character('\u{301}'), "'\\u{301}'|\u{301}|  \u{301}|\u{301}  ||'\\u{301}'");
        assert_eq!(character('\u{7f}'), "'\\u{7f}'|\u{7f}|  \u{7f}|\u{7f}  ||'\\u{7f}'");
        assert_eq!(character('\\'), "'\\\\'|\\|  \\|\\  ||'\\\\'");
    }

    #[test]
    fn tuples() {
        assert_eq!(carrier(7, true, 'x', false), "Triple(7, true, 'x') Single(7)");
        assert_eq!(carrier(7, false, '\t', true), "Triple(\n    7,\n    false,\n    '\\t',\n)");
    }
}
