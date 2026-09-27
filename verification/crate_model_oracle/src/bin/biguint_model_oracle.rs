//! Native oracle for BigUintModelChecks.lean: compare stdout verbatim. Exercises
//! the num-bigint 0.4.6 `BigUint` operations modeled in `Aeneas/Std/BigUint.lean`,
//! including the zero-divisor panics and the bytes `Hash` writes.

use num_bigint::BigUint;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// A hasher that records every byte written and counts the `write` calls.
#[derive(Default)]
struct Log {
    bytes: Vec<u8>,
    writes: u32,
}

impl Hasher for Log {
    fn finish(&self) -> u64 {
        0
    }

    fn write(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
        self.writes += 1;
    }
}

fn show(r: std::thread::Result<BigUint>) -> String {
    match r {
        Ok(v) => v.to_string(),
        Err(_) => "panic".to_string(),
    }
}

fn ord(o: Ordering) -> &'static str {
    match o {
        Ordering::Less => "lt",
        Ordering::Equal => "eq",
        Ordering::Greater => "gt",
    }
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let p64 = BigUint::from(u64::MAX) + BigUint::from(1u8);
    let values: Vec<BigUint> = vec![
        BigUint::from(0u8),
        BigUint::from(1u8),
        BigUint::from(2u16),
        BigUint::from(7u32),
        BigUint::from(12usize),
        BigUint::from(u32::MAX),
        BigUint::from(u32::MAX as u64 + 1),
        BigUint::from(u64::MAX),
        p64.clone(),
        BigUint::from(u128::MAX),
        <BigUint as std::ops::Mul<BigUint>>::mul(p64.clone(), BigUint::from(3u8)),
        <BigUint as std::ops::Mul<BigUint>>::mul(BigUint::from(u128::MAX), BigUint::from(u128::MAX)),
    ];
    for (i, v) in values.iter().enumerate() {
        println!("value-{i}={v}");
        let clone = v.clone();
        println!("clone-{i}={clone}");
        match u64::try_from(v.clone()) {
            Ok(n) => println!("try-u64-{i}=ok {n}"),
            Err(e) => println!("try-u64-{i}=err {}", e.into_original()),
        }
        match usize::try_from(v.clone()) {
            Ok(n) => println!("try-usize-{i}=ok {n}"),
            Err(e) => println!("try-usize-{i}=err {}", e.into_original()),
        }
    }
    for (i, a) in values.iter().enumerate() {
        for (j, b) in values.iter().enumerate() {
            println!("mul-{i}-{j}={}", <BigUint as std::ops::Mul<BigUint>>::mul(a.clone(), b.clone()));
            println!("mulref-{i}-{j}={}", a * b);
            println!("div-{i}-{j}={}", show(catch_unwind(AssertUnwindSafe(|| a / b))));
            println!("rem-{i}-{j}={}", show(catch_unwind(AssertUnwindSafe(|| a % b))));
            println!("divval-{i}-{j}={}", show(catch_unwind(AssertUnwindSafe(|| a.clone() / b.clone()))));
            println!("remval-{i}-{j}={}", show(catch_unwind(AssertUnwindSafe(|| a.clone() % b.clone()))));
            println!("eq-{i}-{j}={}", a == b);
            println!("ne-{i}-{j}={}", a != b);
            println!("cmp-{i}-{j}={}", ord(a.cmp(b)));
            println!("partial_cmp-{i}-{j}={}", a.partial_cmp(b).map_or("none", ord));
        }
    }
    for (i, n) in [0u8, 1, 255].into_iter().enumerate() {
        println!("from-u8-{i}={}", BigUint::from(n));
    }
    for (i, n) in [0u64, 1, u64::MAX].into_iter().enumerate() {
        println!("from-u64-{i}={}", BigUint::from(n));
    }
    for (i, n) in [0usize, 1, usize::MAX].into_iter().enumerate() {
        println!("from-usize-{i}={}", BigUint::from(n));
    }
    for (i, n) in [0u128, u128::MAX].into_iter().enumerate() {
        println!("from-u128-{i}={}", BigUint::from(n));
    }
    // `Hash`: the digit vector's `Hash` (`[u64]`: length prefix, then one write of the digits).
    for (i, v) in values.iter().enumerate() {
        let mut log = Log::default();
        v.hash(&mut log);
        println!("hash-{i}={} {:?}", log.writes, log.bytes);
    }
}
