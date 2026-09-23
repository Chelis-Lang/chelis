use std::fmt;
#[cfg(test)]
use std::fmt::Write;

// These strings are copied from the production AD path, rather than invented
// stand-ins for the compiler's encoded format_args templates.
pub fn production_failure(why: &str) -> String {
    format!("failed to construct backward DAG ({why})")
}

pub fn production_node(op: &str, node: usize) -> String {
    format!(
        "grad: no reverse-mode adjoint is defined for `{}` (node {})",
        op, node
    )
}

// This production template encodes to 175 bytes, including the extended literal
// length and placeholder opcodes. Its generated Array.make must check that
// length without recursively simplifying every byte in the literal.
pub fn production_long_literal(op: &str) -> String {
    format!(
        "finalize_scalar: f8e4m3 is not in the active dtype set (spec/04-type-system.md section 1.1.1); the checker rejects it, so no numeric value can reach finalize at it (op {op})"
    )
}

pub fn escaped(text: &str) -> String {
    format!("{text:?}")
}

pub fn flags(value: i32, text: &str, width: usize, precision: usize) -> String {
    format!("{value:+08}|{text:>width$.precision$}|{value:width$}")
}

#[derive(Debug)]
pub struct Frame {
    pub name: String,
    pub values: Vec<Option<(i32, bool)>>,
}

pub fn nested(pretty: bool) -> String {
    let frame = Frame {
        name: "a\nb".to_string(),
        values: vec![Some((-3, true)), None],
    };
    if pretty {
        format!("{frame:#?}")
    } else {
        format!("{frame:?}")
    }
}

#[derive(Debug)]
pub enum RecursiveValue<T> {
    Leaf(T),
    Child(Box<RecursiveValue<T>>),
    Pair(Box<RecursiveValue<T>>, Box<RecursiveValue<T>>),
}

impl<T: fmt::Display> fmt::Display for RecursiveValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leaf(n) => write!(f, "{n}"),
            Self::Child(value) => {
                f.write_str("[")?;
                fmt::Display::fmt(value, f)?;
                f.write_str("]")
            }
            Self::Pair(lhs, rhs) => write!(f, "({lhs} + {rhs})"),
        }
    }
}

pub fn recursive_formatting(display: bool) -> String {
    let value: RecursiveValue<usize> = RecursiveValue::Pair(
        Box::new(RecursiveValue::Child(Box::new(RecursiveValue::Leaf(2)))),
        Box::new(RecursiveValue::Pair(
            Box::new(RecursiveValue::Leaf(3)),
            Box::new(RecursiveValue::Leaf(5)),
        )),
    );
    if display {
        format!("{value}")
    } else {
        format!("{value:?}")
    }
}

pub fn debug_float32(bits: u32) -> String {
    format!("{:?}", f32::from_bits(bits))
}

pub fn debug_float64(bits: u64) -> String {
    format!("{:?}", f64::from_bits(bits))
}

pub fn display_float64(bits: u64) -> String {
    f64::from_bits(bits).to_string()
}

pub fn precision_float64(bits: u64, precision: usize) -> String {
    format!("{:.*}", precision, f64::from_bits(bits))
}

#[cfg(test)]
pub struct Fails;
#[cfg(test)]
impl fmt::Debug for Fails {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("partial")?;
        Err(fmt::Error)
    }
}

#[cfg(test)]
pub fn callback_failure() -> (bool, String) {
    let mut output = String::new();
    let failed = write!(&mut output, "prefix {:?} suffix", Fails).is_err();
    (failed, output)
}

#[cfg(test)]
pub struct BoundedWriter {
    pub output: String,
    pub limit: usize,
}
#[cfg(test)]
impl Write for BoundedWriter {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        if self.output.len() + text.len() > self.limit {
            return Err(fmt::Error);
        }
        self.output.push_str(text);
        Ok(())
    }
}

#[cfg(test)]
pub fn writer_failure() -> (bool, String) {
    let mut writer = BoundedWriter {
        output: String::new(),
        limit: 7,
    };
    let value = 42;
    let failed = write!(&mut writer, "prefix {} suffix", value).is_err();
    (failed, writer.output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_templates_and_unicode_are_observable() {
        assert_eq!(
            production_failure("bad shape"),
            "failed to construct backward DAG (bad shape)"
        );
        assert_eq!(
            production_node("Argmax", 17),
            "grad: no reverse-mode adjoint is defined for `Argmax` (node 17)"
        );
        assert_eq!(
            escaped("a\n\t\0\"\\é\u{301}"),
            "\"a\\n\\t\\0\\\"\\\\é\\u{301}\""
        );
        assert_eq!(
            nested(false),
            "Frame { name: \"a\\nb\", values: [Some((-3, true)), None] }"
        );
        assert_eq!(flags(-3, "é猫x", 5, 2), "-0000003|   é猫|   -3");
    }

    #[test]
    fn long_production_template_preserves_exact_output() {
        assert_eq!(
            production_long_literal("Add"),
            "finalize_scalar: f8e4m3 is not in the active dtype set (spec/04-type-system.md section 1.1.1); the checker rejects it, so no numeric value can reach finalize at it (op Add)"
        );
    }

    #[test]
    fn recursive_boxed_dictionaries_preserve_nested_text() {
        assert_eq!(
            recursive_formatting(false),
            "Pair(Child(Leaf(2)), Pair(Leaf(3), Leaf(5)))"
        );
        assert_eq!(recursive_formatting(true), "([2] + (3 + 5))");
    }

    #[test]
    fn real_callbacks_and_writers_stop_at_the_first_error() {
        assert_eq!(callback_failure(), (true, "prefix partial".to_string()));
        assert_eq!(writer_failure(), (true, "prefix ".to_string()));
        assert!(std::panic::catch_unwind(|| format!("{:?}", Fails)).is_err());
    }

    #[test]
    fn float_rounding_notation_and_special_values() {
        assert_eq!(debug_float32(0x3dcccccd), "0.1");
        assert_eq!(debug_float64(0x8000000000000000), "-0.0");
        assert_eq!(display_float64(0x8000000000000000), "-0");
        assert_eq!(debug_float64(1), "5e-324");
        assert_eq!(debug_float64(0x7fefffffffffffff), "1.7976931348623157e308");
        assert_eq!(precision_float64(2.5f64.to_bits(), 0), "2");
        assert_eq!(precision_float64(3.5f64.to_bits(), 0), "4");
        assert_eq!(format!("{:+}", f64::NAN), "NaN");
        assert_eq!(debug_float64(f64::NEG_INFINITY.to_bits()), "-inf");
    }
}
