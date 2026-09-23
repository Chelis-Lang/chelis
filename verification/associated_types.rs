// The explicit method bound must remain distinct from the trait's implicit Self bound.
// Substituting the latter with Value<T>'s impl must not turn Item = U into an
// unrepresented equality between the impl parameter T and the method parameter U.
pub trait Produce {
    type Item;

    fn produce(self) -> Self::Item;

    fn retype<U>(self) -> U
    where
        Self: Sized + Produce<Item = U>,
    {
        self.produce()
    }

    fn copied<'a, U: Copy + 'a>(self) -> U
    where
        Self: Sized + Produce<Item = &'a U>,
    {
        *self.produce()
    }
}

pub struct Value<T>(pub T);

impl<T> Produce for Value<T> {
    type Item = T;

    fn produce(self) -> T {
        self.0
    }
}

pub struct Offset(pub u32);

impl Produce for Offset {
    type Item = u32;

    fn produce(self) -> u32 {
        self.0 + 7
    }
}

// A known impl must not bypass the explicitly supplied, equality-constrained bound.
pub fn explicit_bound<T>(value: Value<T>) -> T
where
    Value<T>: Produce<Item = T>,
{
    value.produce()
}

pub fn value_default(value: u32) -> u32 {
    Value(value).retype()
}

pub fn known_local_bound(value: u32) -> u32 {
    explicit_bound(Value(value))
}

pub fn custom_dispatch(value: u32) -> u32 {
    Offset(value).retype()
}

pub fn borrowed_default(value: &u32) -> u32 {
    Value(value).copied()
}

// Dictionary demand must preserve virtual defaults and overrides, while direct
// calls still select overrides whose unused dictionary fields can be omitted.
pub trait Adjust {
    fn base(self) -> u32;

    fn adjusted(self) -> u32
    where
        Self: Sized,
    {
        self.base() + 1
    }

    fn static_only(self) -> u32
    where
        Self: Sized,
    {
        self.base() + 2
    }
}

impl Adjust for Value<u32> {
    fn base(self) -> u32 {
        self.0
    }
}

impl Adjust for Offset {
    fn base(self) -> u32 {
        self.0
    }

    fn adjusted(self) -> u32 {
        self.0 + 10
    }

    fn static_only(self) -> u32 {
        self.0 + 20
    }
}

pub fn through_adjust<A: Adjust>(value: A) -> u32 {
    value.adjusted()
}

pub fn virtual_default(value: u32) -> u32 {
    through_adjust(Value(value))
}

pub fn virtual_override(value: u32) -> u32 {
    through_adjust(Offset(value))
}

pub fn static_default(value: u32) -> u32 {
    Value(value).static_only()
}

pub fn static_override(value: u32) -> u32 {
    Offset(value).static_only()
}

pub trait Make {
    fn make() -> Self;
}

impl Make for Value<u32> {
    fn make() -> Self {
        Value(31)
    }
}

impl Make for Offset {
    fn make() -> Self {
        Offset(17)
    }
}

pub fn invoke_item<T, F: FnOnce() -> T>(factory: F) -> T {
    factory()
}

// The function item has no method generics: its Self type and dictionary come
// from the trait reference, rather than from the function item's generic args.
pub fn make_from_item<T: Make>() -> T {
    invoke_item(T::make)
}

pub fn trait_item_value() -> u32 {
    let value: Value<u32> = make_from_item();
    value.0
}

pub fn trait_item_offset() -> u32 {
    let value: Offset = make_from_item();
    value.0
}

pub fn plain_factory() -> u32 {
    29
}

pub fn regular_item() -> u32 {
    invoke_item(plain_factory)
}

pub fn invoke_pair<F: FnOnce(u32, u32) -> u32>(f: F, left: u32, right: u32) -> u32 {
    f(left, right)
}

pub fn pair_digits(left: u32, right: u32) -> u32 {
    left * 10 + right
}

pub fn pair_from_item(left: u32, right: u32) -> u32 {
    invoke_pair(pair_digits, left, right)
}

fn borrowed_increment(value: &u32) -> u32 {
    *value + 1
}

// collect substitutes the higher-ranked function item beneath trait binders.
// Its own call lifetime must not escape or become captured by those binders.
pub fn collected_items(values: &[u32]) -> Vec<u32> {
    values.iter().map(borrowed_increment).collect()
}

pub fn tuple_constructor_item(value: u32) -> u32 {
    match Some(value).map(Value) {
        Some(value) => value.0,
        None => 0,
    }
}

pub enum ConstructedPair {
    Pair(u32, u32),
}

pub fn variant_constructor_item(left: u32, right: u32) -> u32 {
    let make = ConstructedPair::Pair;
    let ConstructedPair::Pair(left, right) = make(left, right);
    left * 10 + right
}

// Both Option<T>::is_some and its Option<&mut T> specialization refer to the
// same backend builtin. Both instantiations must leave the payload available.
pub fn option_builtin_aliases(value: u32) -> (bool, bool, u32) {
    let owned = Some(value);
    let mut payload = value;
    let borrowed = Some(&mut payload);
    let borrowed_present = borrowed.is_some();
    (owned.is_some(), borrowed_present, payload)
}

struct InequalityOnly(u32);

impl PartialEq for InequalityOnly {
    fn eq(&self, _other: &Self) -> bool {
        panic!("vector inequality must use element inequality");
    }

    fn ne(&self, other: &Self) -> bool {
        self.0 != other.0
    }
}

pub fn vector_inequality_dispatch() -> (bool, bool, bool) {
    let values = vec![InequalityOnly(7)];
    (
        values != vec![InequalityOnly(7)],
        values != vec![InequalityOnly(9)],
        values != Vec::<InequalityOnly>::new(),
    )
}

pub fn boxed_inequality_dispatch() -> (bool, bool) {
    (
        Box::new(InequalityOnly(7)) != Box::new(InequalityOnly(7)),
        Box::new(InequalityOnly(7)) != Box::new(InequalityOnly(9)),
    )
}

pub fn slice_array_comparison_dispatch() -> (bool, bool, bool, bool) {
    let values = [InequalityOnly(7)];
    let empty: [InequalityOnly; 0] = [];
    (
        values.as_slice() != [InequalityOnly(7)],
        values.as_slice() != [InequalityOnly(9)],
        <[InequalityOnly] as PartialEq<[InequalityOnly; 1]>>::eq(
            empty.as_slice(),
            &[InequalityOnly(7)],
        ),
        empty.as_slice() != [InequalityOnly(7)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrained_default_methods_preserve_values_and_dispatch() {
        assert_eq!(value_default(41), 41);
        assert_eq!(known_local_bound(43), 43);
        assert_eq!(custom_dispatch(5), 12);
        assert_eq!(borrowed_default(&47), 47);
    }

    #[test]
    fn dictionary_demand_preserves_virtual_and_static_dispatch() {
        assert_eq!(virtual_default(5), 6);
        assert_eq!(virtual_override(5), 15);
        assert_eq!(static_default(5), 7);
        assert_eq!(static_override(5), 25);
    }

    #[test]
    fn trait_function_items_preserve_self_type_and_dictionary_dispatch() {
        assert_eq!(trait_item_value(), 31);
        assert_eq!(trait_item_offset(), 17);
    }

    #[test]
    fn higher_ranked_function_item_survives_iterator_collection() {
        assert_eq!(collected_items(&[1, 3, 5]), vec![2, 4, 6]);
    }

    #[test]
    fn function_items_preserve_zero_and_multiple_argument_calling_conventions() {
        assert_eq!(regular_item(), 29);
        assert_eq!(pair_from_item(4, 7), 47);
    }

    #[test]
    fn constructor_items_preserve_payloads_and_argument_order() {
        assert_eq!(tuple_constructor_item(53), 53);
        assert_eq!(variant_constructor_item(4, 7), 47);
    }

    #[test]
    fn builtin_aliases_preserve_option_observations() {
        assert_eq!(option_builtin_aliases(59), (true, true, 59));
    }

    #[test]
    fn vector_inequality_uses_element_inequality() {
        assert_eq!(vector_inequality_dispatch(), (false, true, true));
    }

    #[test]
    fn boxed_inequality_preserves_the_selected_override() {
        assert_eq!(boxed_inequality_dispatch(), (false, true));
    }

    #[test]
    fn slice_array_comparisons_preserve_dispatch_and_check_lengths_first() {
        assert_eq!(
            slice_array_comparison_dispatch(),
            (false, true, false, true),
        );
    }
}
