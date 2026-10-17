//! chelis#2205: the consuming list mutators.
//!
//! `chelis_list_append_owned` and `chelis_list_concat_owned` take ownership of
//! their first operand. A uniquely owned list is extended in place and comes
//! back as the same allocation; a shared one is left untouched, a fresh list
//! is returned exactly as the cloning entry points build it, and the consumed
//! input loses the one owner the caller gave up. The static half of the
//! contract (no un-retained reference survives) is the ownership verifier's;
//! this file locks the runtime half.
use chelis_runtime::{
    chelis_list_append_owned, chelis_list_concat_owned, chelis_list_empty, chelis_list_index,
    chelis_list_len, chelis_list_push_moved, chelis_list_release, chelis_list_retain,
    chelis_list_take, chelis_list_take_owned, chelis_scalar_from_bits, chelis_value,
    chelis_value_box_scalar, chelis_value_release, chelis_value_unbox_scalar, CHELIS_DTYPE_I64,
};

unsafe fn int_value(value: i64) -> chelis_value {
    chelis_value_box_scalar(chelis_scalar_from_bits(
        CHELIS_DTYPE_I64,
        u64::from_ne_bytes(value.to_ne_bytes()),
    ))
}

unsafe fn int_at(list: *const chelis_runtime::chelis_list, index: i64) -> i64 {
    let value = chelis_list_index(list, index);
    let bits = chelis_value_unbox_scalar(value).bits;
    chelis_value_release(value);
    i64::from_ne_bytes(bits.to_ne_bytes())
}

#[test]
fn unique_append_owned_mutates_in_place_and_returns_the_same_list() {
    unsafe {
        let list = chelis_list_empty();
        chelis_list_push_moved(list, int_value(1));
        let one = int_value(2);
        let result = chelis_list_append_owned(list, one);
        chelis_value_release(one);
        assert!(
            std::ptr::eq(result, list),
            "a uniquely owned list is extended in place"
        );
        assert_eq!(chelis_list_len(result), 2);
        assert_eq!(int_at(result, 1), 2);
        chelis_list_release(result);
    }
}

#[test]
fn shared_append_owned_clones_and_leaves_the_shared_view_untouched() {
    unsafe {
        let list = chelis_list_empty();
        chelis_list_push_moved(list, int_value(1));
        // A second owner: the observable source list of the alias controls.
        chelis_list_retain(list);
        let three = int_value(3);
        let result = chelis_list_append_owned(list, three);
        chelis_value_release(three);
        assert!(
            !std::ptr::eq(result, list),
            "a shared list is never mutated in place"
        );
        assert_eq!(chelis_list_len(list), 1, "the shared view keeps its length");
        assert_eq!(chelis_list_len(result), 2);
        assert_eq!(int_at(result, 1), 3);
        // The consumed owner was released: exactly one strong owner is left,
        // so this release must free the source without a ledger complaint.
        chelis_list_release(list);
        chelis_list_release(result);
    }
}

#[test]
fn unique_concat_owned_extends_in_place_and_shared_concat_owned_clones() {
    unsafe {
        let rhs = chelis_list_empty();
        chelis_list_push_moved(rhs, int_value(7));
        let unique = chelis_list_empty();
        chelis_list_push_moved(unique, int_value(5));
        let grown = chelis_list_concat_owned(unique, rhs);
        assert!(std::ptr::eq(grown, unique));
        assert_eq!(chelis_list_len(grown), 2);
        assert_eq!(int_at(grown, 1), 7);
        chelis_list_release(grown);

        let shared = chelis_list_empty();
        chelis_list_push_moved(shared, int_value(5));
        chelis_list_retain(shared);
        let fresh = chelis_list_concat_owned(shared, rhs);
        assert!(!std::ptr::eq(fresh, shared));
        assert_eq!(chelis_list_len(shared), 1);
        assert_eq!(chelis_list_len(fresh), 2);
        chelis_list_release(shared);
        chelis_list_release(fresh);
        chelis_list_release(rhs);
    }
}

/// RT-2225 round 1 P0: an rhs that aliases the consumed lhs is a retained
/// second owner (`concat(xs, kept)` with `kept` an alias of `xs`), so it must
/// take the cloning path rather than the alias guard's abort.
#[test]
fn concat_owned_with_an_aliasing_rhs_clones_instead_of_aborting() {
    unsafe {
        let list = chelis_list_empty();
        chelis_list_push_moved(list, int_value(1));
        chelis_list_push_moved(list, int_value(2));
        chelis_list_retain(list);
        let joined = chelis_list_concat_owned(list, list);
        assert!(
            !std::ptr::eq(joined, list),
            "an aliasing rhs never extends in place"
        );
        assert_eq!(chelis_list_len(joined), 4);
        assert_eq!(chelis_list_len(list), 2, "the shared view is untouched");
        assert_eq!(int_at(joined, 3), 2);
        chelis_list_release(list);
        chelis_list_release(joined);
    }
}

#[test]
fn null_inputs_behave_like_the_cloning_entry_points() {
    unsafe {
        let two = int_value(2);
        let from_null = chelis_list_append_owned(std::ptr::null_mut(), two);
        chelis_value_release(two);
        assert_eq!(chelis_list_len(from_null), 1);
        chelis_list_release(from_null);
        let rhs = chelis_list_empty();
        chelis_list_push_moved(rhs, int_value(9));
        let joined = chelis_list_concat_owned(std::ptr::null_mut(), rhs);
        assert_eq!(chelis_list_len(joined), 1);
        chelis_list_release(joined);
        chelis_list_release(rhs);
    }
}

unsafe fn ints(values: &[i64]) -> *mut chelis_runtime::chelis_list {
    let list = chelis_list_empty();
    for value in values {
        chelis_list_push_moved(list, int_value(*value));
    }
    list
}

unsafe fn contents(list: *const chelis_runtime::chelis_list) -> Vec<i64> {
    (0..chelis_list_len(list))
        .map(|index| int_at(list, index))
        .collect()
}

/// chelis#3724: a uniquely owned list is truncated in place and comes back as
/// the same allocation, holding the prefix.
#[test]
fn unique_take_owned_truncates_in_place_and_returns_the_same_list() {
    unsafe {
        let list = ints(&[1, 2, 3, 4]);
        let result = chelis_list_take_owned(list, 2);
        assert!(
            std::ptr::eq(result, list),
            "a uniquely owned list is truncated in place"
        );
        assert_eq!(contents(result), vec![1, 2]);
        chelis_list_release(result);
    }
}

/// A shared list is never truncated: the consuming take clones the prefix,
/// leaves the shared view whole, and releases the owner the caller gave up.
#[test]
fn shared_take_owned_clones_and_leaves_the_shared_view_untouched() {
    unsafe {
        let list = ints(&[1, 2, 3]);
        chelis_list_retain(list);
        let result = chelis_list_take_owned(list, 1);
        assert!(
            !std::ptr::eq(result, list),
            "a shared list is never truncated in place"
        );
        assert_eq!(
            contents(list),
            vec![1, 2, 3],
            "the shared view keeps its length"
        );
        assert_eq!(contents(result), vec![1]);
        chelis_list_release(list);
        chelis_list_release(result);
    }
}

/// The consuming take agrees with the cloning take on every count, including
/// zero and counts at and above the length, where `list_take` truncates at
/// the end ([05-OP-32]).
#[test]
fn take_owned_agrees_with_the_cloning_take_on_every_count() {
    unsafe {
        for count in 0..=6 {
            let source = ints(&[10, 20, 30, 40]);
            let expected = chelis_list_take(source, count);
            let owned = chelis_list_take_owned(source, count);
            assert_eq!(contents(owned), contents(expected), "count {count}");
            chelis_list_release(expected);
            chelis_list_release(owned);
        }
    }
}

/// A null list takes the cloning entry point's path.
#[test]
fn take_owned_of_a_null_list_behaves_like_the_cloning_entry_point() {
    unsafe {
        let result = chelis_list_take_owned(std::ptr::null_mut(), 3);
        assert_eq!(chelis_list_len(result), 0);
        chelis_list_release(result);
    }
}
