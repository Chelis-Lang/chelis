//! chelis#2205: the consuming dictionary mutators.
//!
//! `chelis_dict_insert_owned`, `chelis_dict_merge_owned` and
//! `chelis_dict_remove_owned` take ownership of their first operand. A
//! uniquely owned dictionary is rewritten in place and comes back as the same
//! allocation; a shared one is left untouched, a fresh dictionary is returned
//! exactly as the cloning entry points build it, and the consumed input loses
//! the one owner the caller gave up. The static half of the contract (no
//! un-retained reference survives) is the ownership verifier's; this file
//! locks the runtime half, exactly as `list_owned_mutators.rs` does for the
//! list rows.
use chelis_runtime::{
    chelis_dict, chelis_dict_contains, chelis_dict_get, chelis_dict_insert,
    chelis_dict_insert_owned, chelis_dict_len, chelis_dict_merge, chelis_dict_merge_owned,
    chelis_dict_release, chelis_dict_remove, chelis_dict_remove_owned, chelis_dict_retain,
    chelis_list_empty, chelis_list_len, chelis_list_push, chelis_option_is_some,
    chelis_option_release, chelis_option_unwrap, chelis_scalar_from_bits, chelis_string_from_utf8,
    chelis_value, chelis_value_release, chelis_value_take_list, chelis_value_take_string,
    chelis_value_unbox_scalar, CHELIS_DTYPE_I64,
};

unsafe fn int_value(value: i64) -> chelis_value {
    chelis_runtime::chelis_value_box_scalar(chelis_scalar_from_bits(
        CHELIS_DTYPE_I64,
        u64::from_ne_bytes(value.to_ne_bytes()),
    ))
}

/// The i64 stored at `key`, or `None` when the key is absent.
unsafe fn int_at(dict: *const chelis_dict, key: i64) -> Option<i64> {
    let option = chelis_dict_get(dict, int_value(key));
    let found = if chelis_option_is_some(option) {
        let value = chelis_option_unwrap(option);
        let bits = chelis_value_unbox_scalar(value).bits;
        chelis_value_release(value);
        Some(i64::from_ne_bytes(bits.to_ne_bytes()))
    } else {
        None
    };
    chelis_option_release(option);
    found
}

unsafe fn text_value(text: &str) -> chelis_value {
    chelis_value_take_string(chelis_string_from_utf8(text.as_ptr(), text.len() as i64))
}

/// A fresh single-entry dictionary. There is no `chelis_dict_empty`; the
/// cloning insert over a null dictionary is how every other caller builds
/// one.
unsafe fn dict_of(key: i64, value: i64) -> *mut chelis_dict {
    let boxed_key = int_value(key);
    let boxed_value = int_value(value);
    let dict = chelis_dict_insert(std::ptr::null(), boxed_key, boxed_value);
    chelis_value_release(boxed_key);
    chelis_value_release(boxed_value);
    dict
}

#[test]
fn unique_insert_owned_mutates_in_place_and_returns_the_same_dict() {
    unsafe {
        let dict = dict_of(1, 10);
        let key = int_value(2);
        let value = int_value(20);
        let result = chelis_dict_insert_owned(dict, key, value);
        chelis_value_release(key);
        chelis_value_release(value);
        assert!(
            std::ptr::eq(result, dict),
            "a uniquely owned dictionary is extended in place"
        );
        assert_eq!(chelis_dict_len(result), 2);
        assert_eq!(int_at(result, 1), Some(10));
        assert_eq!(int_at(result, 2), Some(20));
        chelis_dict_release(result);
    }
}

#[test]
fn unique_insert_owned_over_an_existing_key_replaces_in_place() {
    unsafe {
        let dict = dict_of(1, 10);
        let key = int_value(1);
        let value = int_value(99);
        let result = chelis_dict_insert_owned(dict, key, value);
        chelis_value_release(key);
        chelis_value_release(value);
        assert!(std::ptr::eq(result, dict));
        assert_eq!(
            chelis_dict_len(result),
            1,
            "replacing a key does not grow the dictionary"
        );
        assert_eq!(int_at(result, 1), Some(99));
        chelis_dict_release(result);
    }
}

#[test]
fn shared_insert_owned_clones_and_leaves_the_shared_view_untouched() {
    unsafe {
        let dict = dict_of(1, 10);
        // A second owner: the observable source dictionary of the alias
        // controls.
        chelis_dict_retain(dict);
        let key = int_value(2);
        let value = int_value(20);
        let result = chelis_dict_insert_owned(dict, key, value);
        chelis_value_release(key);
        chelis_value_release(value);
        assert!(
            !std::ptr::eq(result, dict),
            "a shared dictionary is never mutated in place"
        );
        assert_eq!(
            chelis_dict_len(dict),
            1,
            "the shared view keeps its entries"
        );
        assert_eq!(chelis_dict_len(result), 2);
        assert_eq!(int_at(result, 2), Some(20));
        // The consumed owner was released: exactly one strong owner is left,
        // so this release must free the source without a ledger complaint.
        chelis_dict_release(dict);
        chelis_dict_release(result);
    }
}

/// The in-place insert clones the incoming value before releasing the one it
/// replaces.
///
/// Evidentiary status: DISPOSITION LOCK on the entry point's ABI contract,
/// not a regression the compiled lane reaches. The verified C lane always
/// hands the value operand a counted reference, so the two can never be the
/// same heap value held exactly once there. A C caller can still construct
/// that shape, which is what this builds: the dictionary is the only strong
/// owner of `inner`, and the value operand is an uncounted alias of it.
///
/// Proven failing first against the variant that mirrors
/// `chelis_dict_insert`'s own order, releasing the replaced value and then
/// cloning the incoming one. That frees `inner` at count zero and the clone
/// reads the freed handle, which the ledger reports as
/// `chelis_value_clone: heap kind mismatch (expected List, got Dict)`. The
/// hoisted clone in `chelis_dict_insert_owned` is what makes the order
/// unreachable rather than merely correct as written.
#[test]
fn insert_owned_clones_the_incoming_value_before_releasing_the_replaced_one() {
    unsafe {
        let inner = chelis_list_empty();
        chelis_list_push(inner, int_value(42));
        let key = int_value(7);
        // The cloning insert retains `inner`, so the dictionary and this
        // frame each hold one count.
        let boxed = chelis_value_take_list(inner);
        let dict = chelis_dict_insert(std::ptr::null(), key, boxed);
        chelis_value_release(boxed);
        // Only the dictionary owns `inner` now. Re-derive the tagged value
        // without taking a count: the uncounted alias.
        let alias = chelis_value_take_list(inner);
        let result = chelis_dict_insert_owned(dict, key, alias);
        chelis_value_release(key);
        assert!(std::ptr::eq(result, dict));
        assert_eq!(chelis_dict_len(result), 1);
        let option = chelis_dict_get(result, key);
        assert!(chelis_option_is_some(option));
        let stored = chelis_option_unwrap(option);
        assert_eq!(
            chelis_list_len(chelis_runtime::chelis_list_borrow_value(stored)),
            1,
            "the replaced value survived the replacement"
        );
        chelis_value_release(stored);
        chelis_option_release(option);
        chelis_dict_release(result);
    }
}

#[test]
fn unique_merge_owned_extends_in_place_and_shared_merge_owned_clones() {
    unsafe {
        let rhs = dict_of(2, 20);
        let unique = dict_of(1, 10);
        let grown = chelis_dict_merge_owned(unique, rhs);
        assert!(std::ptr::eq(grown, unique));
        assert_eq!(chelis_dict_len(grown), 2);
        assert_eq!(int_at(grown, 2), Some(20));
        chelis_dict_release(grown);

        let shared = dict_of(1, 10);
        chelis_dict_retain(shared);
        let fresh = chelis_dict_merge_owned(shared, rhs);
        assert!(!std::ptr::eq(fresh, shared));
        assert_eq!(chelis_dict_len(shared), 1);
        assert_eq!(chelis_dict_len(fresh), 2);
        chelis_dict_release(shared);
        chelis_dict_release(fresh);
        chelis_dict_release(rhs);
    }
}

#[test]
fn merge_owned_lets_the_right_hand_side_win_a_shared_key_in_place() {
    unsafe {
        let rhs = dict_of(1, 99);
        let unique = dict_of(1, 10);
        let merged = chelis_dict_merge_owned(unique, rhs);
        assert!(std::ptr::eq(merged, unique));
        assert_eq!(chelis_dict_len(merged), 1);
        assert_eq!(
            int_at(merged, 1),
            Some(99),
            "the in-place arm keeps `chelis_dict_merge`'s last-write-wins rule"
        );
        chelis_dict_release(merged);
        chelis_dict_release(rhs);
    }
}

/// The dictionary counterpart of RT-2225 round 1's P0: an rhs that aliases
/// the consumed lhs is a retained second owner, so it must take the cloning
/// path rather than an alias guard's abort.
#[test]
fn merge_owned_with_an_aliasing_rhs_clones_instead_of_aborting() {
    unsafe {
        let dict = dict_of(1, 10);
        chelis_dict_retain(dict);
        let merged = chelis_dict_merge_owned(dict, dict);
        assert!(
            !std::ptr::eq(merged, dict),
            "an aliasing rhs never merges in place"
        );
        assert_eq!(chelis_dict_len(merged), 1);
        assert_eq!(int_at(merged, 1), Some(10));
        assert_eq!(chelis_dict_len(dict), 1, "the shared view is untouched");
        chelis_dict_release(dict);
        chelis_dict_release(merged);
    }
}

#[test]
fn unique_remove_owned_drops_the_entry_in_place_and_shared_remove_owned_clones() {
    unsafe {
        let dict = dict_of(1, 10);
        let extra = int_value(2);
        let extra_value = int_value(20);
        let dict = chelis_dict_insert_owned(dict, extra, extra_value);
        chelis_value_release(extra);
        chelis_value_release(extra_value);
        let key = int_value(1);
        let shrunk = chelis_dict_remove_owned(dict, key);
        assert!(std::ptr::eq(shrunk, dict));
        assert_eq!(chelis_dict_len(shrunk), 1);
        assert!(!chelis_dict_contains(shrunk, key));
        assert_eq!(int_at(shrunk, 2), Some(20));

        chelis_dict_retain(shrunk);
        let present = int_value(2);
        let fresh = chelis_dict_remove_owned(shrunk, present);
        chelis_value_release(present);
        assert!(!std::ptr::eq(fresh, shrunk));
        assert_eq!(chelis_dict_len(shrunk), 1, "the shared view is untouched");
        assert_eq!(chelis_dict_len(fresh), 0);
        chelis_value_release(key);
        chelis_dict_release(shrunk);
        chelis_dict_release(fresh);
    }
}

#[test]
fn remove_owned_of_an_absent_key_returns_the_same_unique_dict() {
    unsafe {
        let dict = dict_of(1, 10);
        let absent = int_value(9);
        let result = chelis_dict_remove_owned(dict, absent);
        chelis_value_release(absent);
        assert!(std::ptr::eq(result, dict));
        assert_eq!(chelis_dict_len(result), 1);
        assert_eq!(int_at(result, 1), Some(10));
        chelis_dict_release(result);
    }
}

#[test]
fn null_inputs_behave_like_the_cloning_entry_points() {
    unsafe {
        let key = int_value(1);
        let value = int_value(10);
        let from_null = chelis_dict_insert_owned(std::ptr::null_mut(), key, value);
        assert_eq!(chelis_dict_len(from_null), 1);
        assert_eq!(int_at(from_null, 1), Some(10));
        chelis_dict_release(from_null);

        let rhs = dict_of(2, 20);
        let merged = chelis_dict_merge_owned(std::ptr::null_mut(), rhs);
        assert_eq!(chelis_dict_len(merged), 1);
        chelis_dict_release(merged);

        let removed = chelis_dict_remove_owned(std::ptr::null_mut(), key);
        assert_eq!(chelis_dict_len(removed), 0);
        chelis_dict_release(removed);

        chelis_value_release(key);
        chelis_value_release(value);
        chelis_dict_release(rhs);
    }
}

/// The owned entries agree with the cloning ones on every observable value,
/// which is the property the compiled lane's parity rows rest on.
#[test]
fn owned_entries_agree_with_the_cloning_entries() {
    unsafe {
        let key = int_value(3);
        let value = int_value(30);

        let cloning_source = dict_of(1, 10);
        let cloned = chelis_dict_insert(cloning_source, key, value);
        let owned_source = dict_of(1, 10);
        let owned = chelis_dict_insert_owned(owned_source, key, value);
        assert_eq!(chelis_dict_len(cloned), chelis_dict_len(owned));
        assert_eq!(int_at(cloned, 1), int_at(owned, 1));
        assert_eq!(int_at(cloned, 3), int_at(owned, 3));
        chelis_dict_release(cloning_source);
        chelis_dict_release(cloned);
        chelis_dict_release(owned);

        let rhs = dict_of(1, 77);
        let cloning_lhs = dict_of(1, 10);
        let cloned = chelis_dict_merge(cloning_lhs, rhs);
        let owned_lhs = dict_of(1, 10);
        let owned = chelis_dict_merge_owned(owned_lhs, rhs);
        assert_eq!(int_at(cloned, 1), int_at(owned, 1));
        chelis_dict_release(cloning_lhs);
        chelis_dict_release(cloned);
        chelis_dict_release(owned);

        let cloning_removed = dict_of(1, 10);
        let cloned = chelis_dict_remove(cloning_removed, int_value(1));
        let owned_removed = dict_of(1, 10);
        let owned = chelis_dict_remove_owned(owned_removed, int_value(1));
        assert_eq!(chelis_dict_len(cloned), chelis_dict_len(owned));
        chelis_dict_release(cloning_removed);
        chelis_dict_release(cloned);
        chelis_dict_release(owned);

        chelis_value_release(key);
        chelis_value_release(value);
        chelis_dict_release(rhs);
    }
}

/// String keys and heap values, so the consuming entries actually execute
/// their release paths on refcounted handles.
///
/// Every other fixture in this file uses `i64` keys and values, and
/// releasing a scalar is a no-op, so those rows never run one line of the
/// release work `chelis_dict_remove_owned` and `chelis_dict_insert_owned`
/// owe. This one does: the removed key is a `chelis_string` and the removed
/// value is a `chelis_list`, both with real strong counts.
///
/// What this covers and what it does not, stated because the distinction
/// decided the shape of the test. Releasing one time too many, releasing the
/// wrong handle, or reading a handle after releasing it all abort here,
/// because `require_live_kind` rejects a handle at count zero and the
/// dictionary's own entries are read after the removal. Releasing one time
/// too few does not: a leaked strong count is invisible in-process without
/// the allocation ledger, which is a compile-time feature this test binary
/// does not enable. chelis#2242 tracks the omission half, which belongs in
/// chelis#1286's ledger oracle rather than here, whose child universe is
/// frozen and is not mine to edit.
#[test]
fn string_keyed_entries_survive_the_consuming_entries_release_paths() {
    unsafe {
        let key = text_value("alpha");
        let other = text_value("beta");
        let inner = chelis_list_empty();
        chelis_list_push(inner, int_value(5));
        let payload = chelis_value_take_list(inner);

        let dict = chelis_dict_insert(std::ptr::null(), key, payload);
        chelis_value_release(payload);
        let dict = chelis_dict_insert_owned(dict, other, text_value("kept"));
        assert_eq!(chelis_dict_len(dict), 2);

        // Replacing a string-keyed entry releases the list it displaces and
        // retains the string that replaces it.
        let replaced = chelis_dict_insert_owned(dict, key, text_value("replacement"));
        assert!(std::ptr::eq(replaced, dict));
        assert_eq!(chelis_dict_len(replaced), 2);
        assert!(chelis_dict_contains(replaced, key));
        assert!(chelis_dict_contains(replaced, other));

        // Removing it releases the string key and the string value. Reading
        // the survivor afterwards is what turns an over-release into an
        // abort rather than a silent pass.
        let shrunk = chelis_dict_remove_owned(replaced, key);
        assert!(std::ptr::eq(shrunk, replaced));
        assert_eq!(chelis_dict_len(shrunk), 1);
        assert!(!chelis_dict_contains(shrunk, key));
        assert!(chelis_dict_contains(shrunk, other));
        let survivor = chelis_dict_get(shrunk, other);
        assert!(chelis_option_is_some(survivor));
        let held = chelis_option_unwrap(survivor);
        chelis_value_release(held);
        chelis_option_release(survivor);

        // A merge that displaces the survivor releases it and keeps the
        // right-hand side's string, again read back afterwards.
        let rhs_key = text_value("beta");
        let rhs = chelis_dict_insert(std::ptr::null(), rhs_key, text_value("merged"));
        let merged = chelis_dict_merge_owned(shrunk, rhs);
        assert!(std::ptr::eq(merged, shrunk));
        assert_eq!(chelis_dict_len(merged), 1);
        let after = chelis_dict_get(merged, other);
        assert!(chelis_option_is_some(after));
        let value = chelis_option_unwrap(after);
        chelis_value_release(value);
        chelis_option_release(after);

        chelis_value_release(key);
        chelis_value_release(other);
        chelis_value_release(rhs_key);
        chelis_dict_release(rhs);
        chelis_dict_release(merged);
    }
}
