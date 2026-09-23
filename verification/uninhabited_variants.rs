pub enum Empty {}

pub struct Wrapper<T>(pub T);

pub enum Packet<T> {
    Present(u32),
    Absent(Wrapper<Wrapper<T>>),
}

// MIR projects Present directly: the other variant has no possible value.
pub fn read_present(packet: Packet<Empty>) -> u32 {
    match packet {
        Packet::Present(value) => value,
        Packet::Absent(Wrapper(Wrapper(impossible))) => match impossible {},
    }
}

// Result's FromResidual implementation projects Err from Result<Infallible, E>.
pub fn propagate_result(value: Result<u32, u16>) -> Result<u32, u16> {
    Ok(value? + 1)
}

#[inline(never)]
fn missing_value() -> ! {
    panic!("missing value")
}

pub fn value_or_panic(value: Option<u32>) -> u32 {
    match value {
        Some(value) => value,
        None => missing_value(),
    }
}

pub fn unwrap_value(value: Option<u32>) -> u32 {
    value.unwrap()
}

pub fn sum_present(values: &[Option<u32>]) -> u32 {
    let mut total = 0;
    for value in values {
        match value {
            Some(value) => total += *value,
            None => missing_value(),
        }
    }
    total
}

// Lean elaborates method signatures in the enum's namespace.
pub enum PrimitiveNames {
    Bool,
    Char,
    Std,
    Unit,
    F32,
    String,
}

impl PrimitiveNames {
    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Bool)
    }

    pub fn marker(&self) -> char {
        if self.is_bool() {
            'b'
        } else {
            'c'
        }
    }

    pub fn number(&self) -> u32 {
        if self.is_bool() {
            17
        } else {
            23
        }
    }

    pub fn discard(self) {}

    pub fn fraction(&self) -> f32 {
        -1.5
    }

    pub fn text(&self, value: String) -> String {
        value
    }
}

pub mod graph {
    pub enum Node<T> {
        Value(T),
        Missing,
    }

    pub fn wrap<T>(value: T) -> Node<T> {
        Node::Value(value)
    }

    pub fn count_down(graph: u32) -> u32 {
        if graph == 0 {
            0
        } else {
            1 + count_down(graph - 1)
        }
    }
}

// The binder must not capture either the type or function namespace in Lean.
pub fn namespace_shadow<T>(graph: graph::Node<T>, replacement: T) -> graph::Node<T> {
    match graph {
        graph::Node::Value(value) => graph::wrap(value),
        graph::Node::Missing => graph::wrap(replacement),
    }
}

pub struct KeywordFields {
    pub full: Option<u32>,
}

pub fn update_full(value: KeywordFields, full: u32) -> (Option<u32>, KeywordFields) {
    (value.full, KeywordFields { full: Some(full) })
}

#[cfg(test)]
mod tests {
    #[test]
    fn primitive_names_do_not_capture_method_result_types() {
        assert!(super::PrimitiveNames::Bool.is_bool());
        assert!(!super::PrimitiveNames::Char.is_bool());
        assert_eq!(super::PrimitiveNames::Bool.marker(), 'b');
        assert_eq!(super::PrimitiveNames::Char.marker(), 'c');
        assert_eq!(super::PrimitiveNames::Std.number(), 23);
        assert_eq!(super::PrimitiveNames::F32.fraction(), -1.5);
        assert_eq!(
            super::PrimitiveNames::String.text("model namespace".to_owned()),
            "model namespace"
        );
    }

    #[test]
    fn a_module_named_binder_preserves_generic_values_and_constructors() {
        assert_eq!(super::graph::count_down(4), 4);
        match super::namespace_shadow(super::graph::Node::Value(31u32), 47) {
            super::graph::Node::Value(value) => assert_eq!(value, 31),
            super::graph::Node::Missing => panic!("lost the existing value"),
        }
        match super::namespace_shadow(super::graph::Node::Missing, 47u32) {
            super::graph::Node::Value(value) => assert_eq!(value, 47),
            super::graph::Node::Missing => panic!("lost the replacement value"),
        }
    }

    #[test]
    fn keyword_fields_preserve_old_and_replacement_values() {
        let (old, updated) = super::update_full(super::KeywordFields { full: Some(11) }, 29);
        assert_eq!(old, Some(11));
        assert_eq!(updated.full, Some(29));
    }

    #[test]
    fn reads_the_only_inhabited_nested_generic_variant() {
        assert_eq!(super::read_present(super::Packet::Present(37)), 37);
    }

    #[test]
    fn residual_conversion_preserves_success_and_error_paths() {
        assert_eq!(super::propagate_result(Ok(41)), Ok(42));
        assert_eq!(super::propagate_result(Err(17)), Err(17));
    }

    #[test]
    fn returns_values_without_entering_the_never_branch() {
        assert_eq!(super::value_or_panic(Some(17)), 17);
        assert_eq!(super::unwrap_value(Some(23)), 23);
        assert_eq!(super::sum_present(&[Some(2), Some(3)]), 5);
    }

    #[test]
    fn never_returning_calls_propagate_panics() {
        assert!(std::panic::catch_unwind(|| super::value_or_panic(None)).is_err());
        assert!(std::panic::catch_unwind(|| super::unwrap_value(None)).is_err());
        assert!(
            std::panic::catch_unwind(|| super::sum_present(&[Some(2), None, Some(3)])).is_err()
        );
    }
}
