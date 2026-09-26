//! Native oracle for HalfModelChecks.lean: compare stdout verbatim. Calls the
//! real `half` 2.7.1 API (hardware-dispatched where the crate dispatches).
//! Large sample classes are folded into FNV hashes per line.

use crate_model_oracle::{cmp_operands, fnv, mantissas32, mantissas64_full, mantissas64_small, FNV_START};
use half::{bf16, f16};
use std::cmp::Ordering;

fn ord_code(o: Option<Ordering>) -> u64 {
    match o {
        None => 0,
        Some(Ordering::Less) => 1,
        Some(Ordering::Equal) => 2,
        Some(Ordering::Greater) => 3,
    }
}

macro_rules! cmp_rows {
    ($label:expr, $ty:ty, $inf:expr, $one:expr) => {{
        let operands = cmp_operands($inf, $one);
        for (i, &a) in operands.iter().enumerate() {
            let x = <$ty>::from_bits(a as u16);
            let mut h = FNV_START;
            for &b in &operands {
                let y = <$ty>::from_bits(b as u16);
                for v in [(x == y) as u64, (x != y) as u64, ord_code(x.partial_cmp(&y)),
                    (x < y) as u64, (x <= y) as u64, (x > y) as u64, (x >= y) as u64] {
                    h = fnv(h, v);
                }
            }
            println!("{}-cmp-{i}={h}", $label);
        }
    }};
}

fn main() {
    for (name, x) in [("ZERO", f16::ZERO), ("NEG_ZERO", f16::NEG_ZERO), ("ONE", f16::ONE),
        ("INFINITY", f16::INFINITY), ("NAN", f16::NAN), ("MAX", f16::MAX)] {
        println!("f16-{name}={}", x.to_bits());
    }
    for (name, x) in [("ZERO", bf16::ZERO), ("NEG_ZERO", bf16::NEG_ZERO), ("ONE", bf16::ONE),
        ("INFINITY", bf16::INFINITY), ("NAN", bf16::NAN), ("MAX", bf16::MAX)] {
        println!("bf16-{name}={}", x.to_bits());
    }
    let explicit32: [u32; 14] = [0x7F80_0001, 0xFF80_0001, 0x7FC0_0000, 0x7FBF_FFFF, 0x3300_0000,
        0x3300_0001, 0x477F_F000, 0x477F_EFFF, 0x3880_0000, 0x387F_C000, 0x3F80_1000, 0x3F80_3000,
        0x3F80_8000, 0x3F81_8000];
    for (i, b) in explicit32.into_iter().enumerate() {
        println!("f16-from-f32-{i}={}", f16::from_f32(f32::from_bits(b)).to_bits());
        println!("bf16-from-f32-{i}={}", bf16::from_f32(f32::from_bits(b)).to_bits());
    }
    let explicit64: [u64; 8] = [0x7FF0_0000_0000_0001, 0xFFF0_0000_0000_0001, 0x7FF8_0000_0000_0000,
        0x3FF0_0000_8000_0000, 0x3FF0_1000_0000_0001, 0x3FF0_1000_0000_0000, 0x3FF0_3000_0000_0000,
        0x3800_0000_0000_0000];
    for (i, b) in explicit64.into_iter().enumerate() {
        println!("bf16-from-f64-{i}={}", bf16::from_f64(f64::from_bits(b)).to_bits());
    }
    // f16::from_f64 (FCVT on aarch64): ties, sticky bits below the high word,
    // subnormal and overflow boundaries, NaN payloads.
    let explicit64_f16: [u64; 12] = [0x3FF0_0200_0000_0000, 0x3FF0_0200_0000_0001,
        0x3FF0_0600_0000_0000, 0x3E70_0000_0000_0000, 0x3E70_0000_0000_0001, 0x3E60_0000_0000_0000,
        0x40EF_FE00_0000_0000, 0x40EF_FDFF_FFFF_FFFF, 0x7FF0_0000_0000_0001, 0xFFF4_0000_0000_0001,
        0x7FF8_0000_0000_0000, 0xFFF0_0000_0000_0000];
    for (i, b) in explicit64_f16.into_iter().enumerate() {
        println!("f16-from-f64-{i}={}", f16::from_f64(f64::from_bits(b)).to_bits());
    }
    let man32 = mantissas32();
    for sign in 0..2u64 {
        for exp in 0..256u64 {
            let (mut h, mut hb) = (FNV_START, FNV_START);
            for &m in &man32 {
                let f = f32::from_bits((sign << 31 | exp << 23 | m) as u32);
                h = fnv(h, f16::from_f32(f).to_bits() as u64);
                hb = fnv(hb, bf16::from_f32(f).to_bits() as u64);
            }
            println!("f32-to-f16-{sign}-{exp}={h}");
            println!("f32-to-bf16-{sign}-{exp}={hb}");
        }
    }
    let (full, small) = (mantissas64_full(), mantissas64_small());
    for sign in 0..2u64 {
        for exp in 0..2048u64 {
            let near = exp == 0 || exp == 2047 || (1023 - 140..=1023 + 130).contains(&exp);
            let mut h = FNV_START;
            for &m in if near { &full } else { &small } {
                h = fnv(h, bf16::from_f64(f64::from_bits(sign << 63 | exp << 52 | m)).to_bits() as u64);
            }
            println!("f64-to-bf16-{sign}-{exp}={h}");
        }
    }
    for sign in 0..2u64 {
        for exp in 0..2048u64 {
            let near = exp == 0 || exp == 2047 || (1023 - 30..=1023 + 20).contains(&exp);
            let mut h = FNV_START;
            for &m in if near { &full } else { &small } {
                h = fnv(h, f16::from_f64(f64::from_bits(sign << 63 | exp << 52 | m)).to_bits() as u64);
            }
            println!("f64-to-f16-{sign}-{exp}={h}");
        }
    }
    for hi in 0..256u64 {
        let mut hs = [FNV_START; 6];
        for lo in 0..256u64 {
            let bits = (hi << 8 | lo) as u16;
            let (x, y) = (f16::from_bits(bits), bf16::from_bits(bits));
            let values = [x.to_f32().to_bits() as u64, f64::from(x).to_bits(), x.is_nan() as u64,
                y.to_f32().to_bits() as u64, f64::from(y).to_bits(), y.is_nan() as u64];
            for (h, v) in hs.iter_mut().zip(values) {
                *h = fnv(*h, v);
            }
        }
        for (label, h) in ["f16-to-f32", "f16-to-f64", "f16-is-nan", "bf16-to-f32", "bf16-to-f64",
            "bf16-is-nan"].into_iter().zip(hs) {
            println!("{label}-{hi}={h}");
        }
    }
    cmp_rows!("f16", f16, 0x7C00, 0x3C00);
    cmp_rows!("bf16", bf16, 0x7F80, 0x3F80);
}
