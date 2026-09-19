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
    chelis_list_len, chelis_list_push, chelis_list_release, chelis_list_retain,
    chelis_scalar_from_bits, chelis_value, chelis_value_box_scalar, chelis_value_release,
    chelis_value_unbox_scalar, CHELIS_DTYPE_I64,
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
        chelis_list_push(list, int_value(1));
        let one = int_value(2);
        let result = chelis_list_append_owned(list, one);
        chelis_value_release(one);
        assert!(std::ptr::eq(result, list), "a uniquely owned list is extended in place");
        assert_eq!(chelis_list_len(result), 2);
        assert_eq!(int_at(result, 1), 2);
        chelis_list_release(result);
    }
}

#[test]
fn shared_append_owned_clones_and_leaves_the_shared_view_untouched() {
    unsafe {
        let list = chelis_list_empty();
        chelis_list_push(list, int_value(1));
        // A second owner: the observable source list of the alias controls.
        chelis_list_retain(list);
        let three = int_value(3);
        let result = chelis_list_append_owned(list, three);
        chelis_value_release(three);
        assert!(!std::ptr::eq(result, list), "a shared list is never mutated in place");
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
        chelis_list_push(rhs, int_value(7));
        let unique = chelis_list_empty();
        chelis_list_push(unique, int_value(5));
        let grown = chelis_list_concat_owned(unique, rhs);
        assert!(std::ptr::eq(grown, unique));
        assert_eq!(chelis_list_len(grown), 2);
        assert_eq!(int_at(grown, 1), 7);
        chelis_list_release(grown);

        let shared = chelis_list_empty();
        chelis_list_push(shared, int_value(5));
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

#[test]
fn null_inputs_behave_like_the_cloning_entry_points() {
    unsafe {
        let two = int_value(2);
        let from_null = chelis_list_append_owned(std::ptr::null_mut(), two);
        chelis_value_release(two);
        assert_eq!(chelis_list_len(from_null), 1);
        chelis_list_release(from_null);
        let rhs = chelis_list_empty();
        chelis_list_push(rhs, int_value(9));
        let joined = chelis_list_concat_owned(std::ptr::null_mut(), rhs);
        assert_eq!(chelis_list_len(joined), 1);
        chelis_list_release(joined);
        chelis_list_release(rhs);
    }
}
