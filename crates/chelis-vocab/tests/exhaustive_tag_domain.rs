//! Exhaustive check of the runtime dtype tag decoder over the entire `i32`
//! domain.
//!
//! This is the complete oracle for round-trip, injectivity, and exhaustive
//! rejection: every representable tag value is visited, so the properties hold
//! by enumeration rather than by sampling. `closed_vocabulary.rs` carries the
//! bounded companion that runs in the local profile.
//!
//! Measured 2026-07-27 on aarch64-apple-darwin: 10.4s release, 128.5s debug.
//! It therefore runs in a dedicated per-PR release step and is excluded from
//! the local `default` nextest profile, not from CI. Keep it single-threaded:
//! a threaded form saturates every core and would need a nextest test group
//! with `threads-required = 'num-cpus'` to schedule safely.
//!
//! Iteration is over `u32` cast to `i32` so the domain is covered exactly once
//! without end-of-range overflow handling.

use chelis_vocab::{RuntimeDType, RuntimeDTypeDecodeError};
use std::hint::black_box;

#[test]
fn every_i32_either_round_trips_or_is_rejected_with_its_own_value() {
    let mut accepted = 0u64;

    for u in 0..=u32::MAX {
        // Reinterpret, not convert: the wrap is the point, since it is what
        // walks the whole signed domain exactly once.
        let id = u.cast_signed();
        // black_box is load-bearing: without it the whole loop is elided and
        // the test proves nothing.
        match RuntimeDType::decode_id(black_box(id)) {
            Ok(dtype) => {
                assert_eq!(dtype.id(), id, "round-trip failed at {id}");
                accepted += 1;
            }
            Err(RuntimeDTypeDecodeError::InvalidId { id: reported }) => {
                assert_eq!(
                    reported, id,
                    "rejection dropped the offending value at {id}"
                );
            }
        }
    }

    assert_eq!(
        accepted,
        RuntimeDType::ALL.len() as u64,
        "exactly the closed vocabulary may be accepted across the whole domain"
    );
}
