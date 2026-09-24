//! chelis#2413: the [05-RNG-2] key derivations and [05-OP-69..72] against an
//! independent transcription.
//!
//! Two independent sources pin the kernels. The hexadecimal constants are the
//! worked values of `key_ref.py`, the design pass's Python transcription of
//! `spec/design/randomness_explicit_keys.md` (never of Rust or Lean). The
//! `reference` module below is a second transcription, of the spec/05 text,
//! compared with the kernels over a sweep of keys and indices. Changing a
//! rotation (29, 41, 17) or a derive index (0, 1, 2) in the kernels fails
//! both.

use chelis_types::dtype_semantics::{
    KeyHalf, PreparedDropout, PreparedUniformLike, RandomKey, StorageView, TensorStorage,
    fold_in_storage, key_from_seed_storage, split_key_storage, split_keys_storage,
};
use chelis_types::types::Prim;
use chelis_types::{ScalarValue, finalize_tensor, scalar_from_f64, scalar_from_i64};

/// spec/05 [05-RNG-1] and [05-RNG-2], transcribed from the chapter text.
mod reference {
    pub fn splitmix64(x: u64) -> u64 {
        let x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    // Spelled out from the definition rather than through the kernels'
    // `rotate_left`, so this transcription shares no helper with them.
    #[allow(clippy::manual_rotate)]
    pub fn rotl64(x: u64, r: u32) -> u64 {
        (x << r) | (x >> (64 - r))
    }

    pub fn derive(k: u64, j: u64) -> u64 {
        splitmix64(k ^ rotl64(splitmix64(j), 29))
    }

    pub fn word(k: u64, i: u64) -> u64 {
        splitmix64(k ^ rotl64(splitmix64(i), 41))
    }

    pub fn unit(k: u64, i: u64) -> f64 {
        (word(k, i) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn fold_in(k: u64, n: i64) -> u64 {
        derive(derive(k, 2), n as u64)
    }

    pub fn of_draw_key(seed: i64, ordinal: u64) -> u64 {
        (seed as u64) ^ rotl64(splitmix64(ordinal), 17)
    }
}

fn i64_scalar(value: i64) -> ScalarValue {
    scalar_from_i64("test", Prim::Int64, value).unwrap()
}

fn key(seed: i64) -> RandomKey {
    RandomKey::from_seed(i64_scalar(seed)).unwrap()
}

/// The exact f64 units of the draw keyed by `key`: `uniform_like(0, 1)` at
/// f64 stores `fma(1, u, 0) = u`.
fn units(key: RandomKey, len: usize) -> Vec<f64> {
    let zero = scalar_from_f64("test", Prim::F64, 0.0).unwrap();
    let one = scalar_from_f64("test", Prim::F64, 1.0).unwrap();
    let storage = PreparedUniformLike::new(Prim::F64, len, zero, one)
        .unwrap()
        .apply(key)
        .unwrap();
    let StorageView::F64(values) = storage.view() else {
        panic!("f64 draw");
    };
    values.to_vec()
}

fn bits(storage: &TensorStorage) -> Vec<u64> {
    storage
        .keys()
        .expect("key storage")
        .iter()
        .map(|key| key.bits())
        .collect()
}

#[test]
fn key_ref_worked_values_are_the_kernels_values() {
    let k7 = key(7);
    assert_eq!(k7.bits(), 0x0000_0000_0000_0007);
    assert_eq!(key(-1).bits(), 0xffff_ffff_ffff_ffff);
    let (left, right) = k7.split();
    assert_eq!(left.bits(), 0xaa38_9617_2f9a_3213);
    assert_eq!(right.bits(), 0x8fd0_6b2e_7bad_8630);
    assert_eq!(
        k7.fold_in(i64_scalar(3)).unwrap().bits(),
        0x53c6_f7e8_3810_b049
    );
    assert_eq!(
        k7.fold_in(i64_scalar(-1)).unwrap().bits(),
        0x45c8_0b55_7fb9_4ddb
    );
    assert_eq!(
        k7.split_n(3)
            .iter()
            .map(|key| key.bits())
            .collect::<Vec<_>>(),
        [
            0x25ea_33e6_1c10_576f,
            0x7071_24fb_ecd5_f054,
            0x8239_3615_3a56_5205
        ]
    );
    assert_eq!(k7.derive(2).bits(), 0x4ed9_4e35_099b_b63d);
    assert_eq!(units(k7, 1), [0.12654528231070938]);
    assert_eq!(RandomKey::from_counter(7, 0).bits(), 0x5072_f63b_9b5f_c446);
    assert_eq!(RandomKey::from_counter(7, 1).bits(), 0x5bd9_1204_b983_2213);
}

#[test]
fn the_chain_of_the_lane_oracle_matches_key_ref() {
    // key_ref_ext.py: k = key(-3); (L, R) = split(k); F = fold_in(L, -5);
    // S = split_n(F, 3); G = fold_in(R, 9).
    let (left, right) = key(-3).split();
    assert_eq!(key(-3).bits(), 0xffff_ffff_ffff_fffd);
    assert_eq!(left.bits(), 0x7b39_5716_aaf0_df98);
    assert_eq!(right.bits(), 0x3da2_4528_f4bd_ba88);
    let folded = left.fold_in(i64_scalar(-5)).unwrap();
    assert_eq!(folded.bits(), 0x0e13_3f2d_fb75_babc);
    assert_eq!(
        folded
            .split_n(3)
            .iter()
            .map(|key| key.bits())
            .collect::<Vec<_>>(),
        [
            0xfa91_e16d_2c37_662f,
            0xa561_587d_4583_1567,
            0xa67b_6073_3765_a1bc
        ]
    );
    assert_eq!(
        right.fold_in(i64_scalar(9)).unwrap().bits(),
        0x2334_cf03_8b09_85b4
    );
}

#[test]
fn kernels_agree_with_the_transcription_over_a_sweep() {
    let seeds = [0, 1, 7, -1, -3, i64::MAX, i64::MIN, 0x0123_4567_89ab_cdef];
    let ns = [0, 1, 2, 3, -1, -5, i64::MAX, i64::MIN];
    for seed in seeds {
        let root = key(seed);
        assert_eq!(root.bits(), seed as u64, "key_from_seed({seed})");
        let (left, right) = root.split();
        assert_eq!(left.bits(), reference::derive(seed as u64, 0));
        assert_eq!(right.bits(), reference::derive(seed as u64, 1));
        for n in ns {
            assert_eq!(
                root.fold_in(i64_scalar(n)).unwrap().bits(),
                reference::fold_in(seed as u64, n),
                "fold_in(key({seed}), {n})"
            );
        }
        for (j, row) in root.split_n(5).iter().enumerate() {
            assert_eq!(row.bits(), reference::fold_in(seed as u64, j as i64));
        }
        for j in [0, 1, 2, 3, u64::MAX] {
            assert_eq!(root.derive(j).bits(), reference::derive(seed as u64, j));
        }
        let drawn = units(left, 9);
        for (i, unit) in drawn.iter().enumerate() {
            assert_eq!(
                unit.to_bits(),
                reference::unit(left.bits(), i as u64).to_bits()
            );
        }
        for ordinal in [0, 1, 2, 1_000_000] {
            assert_eq!(
                RandomKey::from_counter(seed as u64, ordinal).bits(),
                reference::of_draw_key(seed, ordinal)
            );
        }
    }
}

#[test]
fn derivations_are_finalized_mixes_not_symmetric_xors() {
    // The chelis#2408 class: a chained fold must not commute.
    let root = key(7);
    let ab = root
        .fold_in(i64_scalar(3))
        .unwrap()
        .fold_in(i64_scalar(5))
        .unwrap();
    let ba = root
        .fold_in(i64_scalar(5))
        .unwrap()
        .fold_in(i64_scalar(3))
        .unwrap();
    assert_ne!(ab, ba);
    let (left, right) = root.split();
    assert_ne!(left, right);
    assert_ne!(key(7).split().0, key(8).split().0);
}

#[test]
fn every_counter_word_is_the_word_of_its_bridge_key() {
    // [05-RNG-2]: the draw of seed bits s and ordinal c is the draw keyed by
    // s XOR rotl64(splitmix64(c), 17). Transcribed from [05-RNG-1] directly.
    let rng1_unit = |seed: i64, c: u64, i: u64| {
        let w = reference::splitmix64(
            (seed as u64)
                ^ reference::rotl64(reference::splitmix64(c), 17)
                ^ reference::rotl64(reference::splitmix64(i), 41),
        );
        (w >> 11) as f64 / (1u64 << 53) as f64
    };
    for seed in [0, 7, -1, i64::MAX, i64::MIN] {
        for c in 0..5 {
            let drawn = units(RandomKey::from_counter(seed as u64, c), 5);
            for (i, unit) in drawn.iter().enumerate() {
                assert_eq!(unit.to_bits(), rng1_unit(seed, c, i as u64).to_bits());
            }
        }
    }
}

#[test]
fn storage_kernels_apply_the_scalar_derivations_element_by_element() {
    let seeds = finalize_tensor(
        "test",
        Prim::Int64,
        chelis_types::RawTensor::Int(vec![7, -3, 0]),
    )
    .unwrap();
    let keys = key_from_seed_storage(&seeds).unwrap();
    assert_eq!(keys.prim(), Prim::Key);
    assert_eq!(bits(&keys), [7, (-3i64) as u64, 0]);
    let left = split_key_storage(&keys, KeyHalf::Left).unwrap();
    let right = split_key_storage(&keys, KeyHalf::Right).unwrap();
    for (index, seed) in [7i64, -3, 0].into_iter().enumerate() {
        assert_eq!(bits(&left)[index], reference::derive(seed as u64, 0));
        assert_eq!(bits(&right)[index], reference::derive(seed as u64, 1));
    }
    let ns = finalize_tensor(
        "test",
        Prim::Int64,
        chelis_types::RawTensor::Int(vec![-5, 9, i64::MIN]),
    )
    .unwrap();
    let folded = fold_in_storage(&keys, &ns).unwrap();
    for (index, (seed, n)) in [(7i64, -5i64), (-3, 9), (0, i64::MIN)]
        .into_iter()
        .enumerate()
    {
        assert_eq!(bits(&folded)[index], reference::fold_in(seed as u64, n));
    }
    // The new axis is last: element (i, j) is row j of key i.
    let rows = split_keys_storage(&keys, 2).unwrap();
    let expected: Vec<u64> = [7i64, -3, 0]
        .into_iter()
        .flat_map(|seed| (0..2).map(move |j| reference::fold_in(seed as u64, j)))
        .collect();
    assert_eq!(bits(&rows), expected);
    assert!(split_keys_storage(&keys, 0).unwrap().is_empty());
}

#[test]
fn key_kernels_reject_non_key_and_non_i64_operands() {
    let f32_scalar = scalar_from_f64("test", Prim::F32, 1.0).unwrap();
    let i32_scalar = scalar_from_i64("test", Prim::Int32, 1).unwrap();
    assert!(RandomKey::from_seed(i32_scalar).is_err());
    assert!(RandomKey::from_seed(f32_scalar).is_err());
    assert!(key(1).fold_in(i32_scalar).is_err());
    let ints = finalize_tensor("test", Prim::Int64, chelis_types::RawTensor::Int(vec![1])).unwrap();
    assert!(split_key_storage(&ints, KeyHalf::Left).is_err());
    assert!(split_keys_storage(&ints, 2).is_err());
    assert!(fold_in_storage(&ints, &ints).is_err());
    let keys = key_from_seed_storage(&ints).unwrap();
    let two = finalize_tensor(
        "test",
        Prim::Int64,
        chelis_types::RawTensor::Int(vec![1, 2]),
    )
    .unwrap();
    assert!(fold_in_storage(&keys, &two).is_err(), "no broadcasting");
    assert!(key_from_seed_storage(&keys).is_err());
}

#[test]
fn a_row_batched_draw_is_the_stack_of_its_rows_scalar_draws() {
    let rows = key(-3)
        .split()
        .0
        .fold_in(i64_scalar(-5))
        .unwrap()
        .split_n(3);
    let zero = scalar_from_f64("test", Prim::F32, 0.0).unwrap();
    let one = scalar_from_f64("test", Prim::F32, 1.0).unwrap();
    let batched = PreparedUniformLike::new(Prim::F32, 12, zero, one)
        .unwrap()
        .apply_rows(&rows)
        .unwrap();
    let stacked: Vec<f64> = rows
        .iter()
        .flat_map(|row| {
            PreparedUniformLike::new(Prim::F32, 4, zero, one)
                .unwrap()
                .apply(*row)
                .unwrap()
                .to_f64_lossy_vec()
        })
        .collect();
    assert_eq!(batched.to_f64_lossy_vec(), stacked);
    // key_ref_ext.py's f32 words for the first row, S0.
    let StorageView::F32(values) = batched.view() else {
        panic!("f32 draw");
    };
    assert_eq!(
        values[..4].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [0x3d77_27f8, 0x3ed5_3b14, 0x3d8a_16fd, 0x3ef3_5934]
    );

    let input = finalize_tensor(
        "test",
        Prim::F32,
        chelis_types::RawTensor::Float((1..=12).map(|x| f64::from(x % 4 + 1)).collect()),
    )
    .unwrap();
    let rate = scalar_from_f64("test", Prim::F32, 0.5).unwrap();
    let dropped = PreparedDropout::new(&input, rate)
        .unwrap()
        .apply_rows(&rows)
        .unwrap();
    // key_ref_ext.py: dropout_half over [1,2,3,4] is S0 [0,0,0,0], S1
    // [0,4,6,0] and S2 [2,4,6,8]; this input's rows are [2,3,4,1] each.
    let expect_row = |kept: [bool; 4]| {
        [2.0, 3.0, 4.0, 1.0]
            .into_iter()
            .zip(kept)
            .map(|(x, keep)| if keep { 2.0 * x } else { 0.0 })
            .collect::<Vec<f64>>()
    };
    let mut expected = expect_row([false, false, false, false]);
    expected.extend(expect_row([false, true, true, false]));
    expected.extend(expect_row([true, true, true, true]));
    assert_eq!(dropped.to_f64_lossy_vec(), expected);
    assert!(
        PreparedUniformLike::new(Prim::F32, 10, zero, one)
            .unwrap()
            .apply_rows(&rows)
            .is_err(),
        "rows must split the elements evenly"
    );
}

#[test]
fn key_storage_has_no_literal_carrier_or_numeric_reading() {
    let keys = key_from_seed_storage(
        &finalize_tensor("test", Prim::Int64, chelis_types::RawTensor::Int(vec![1])).unwrap(),
    )
    .unwrap();
    assert!(serde_json::to_string(&keys).is_err());
    assert!(serde_json::to_string(&ScalarValue::from_key(key(1))).is_err());
    assert_eq!(keys.to_i64_exact_vec(), None);
    assert!(std::panic::catch_unwind(|| keys.to_f64_lossy_vec()).is_err());
    assert_eq!(ScalarValue::from_key(key(1)).as_i64_exact(), None);
    assert_eq!(ScalarValue::from_key(key(1)).as_key(), Some(key(1)));
    assert_eq!(i64_scalar(1).as_key(), None);
}
