//! Shared pieces of the external-crate model oracles.
//!
//! `sw` transliterates half 2.7.1's software conversion algorithms
//! (`src/binary16/arch.rs` `*_fallback`, `src/bfloat/convert.rs`) exactly as
//! the Lean models in `Aeneas/Std/Half.lean` do. The sample enumerations and
//! the FNV hash are mirrored verbatim by `HalfModelChecks.lean`.

pub mod sw {
    pub const fn leading_zeros_u16(x: u16) -> u32 {
        x.leading_zeros()
    }

    pub const fn f32_to_f16(x: u32) -> u16 {
        let sign = x & 0x8000_0000u32;
        let exp = x & 0x7F80_0000u32;
        let man = x & 0x007F_FFFFu32;
        if exp == 0x7F80_0000u32 {
            let nan_bit = if man == 0 { 0 } else { 0x0200u32 };
            return ((sign >> 16) | 0x7C00u32 | nan_bit | (man >> 13)) as u16;
        }
        let half_sign = sign >> 16;
        let unbiased_exp = ((exp >> 23) as i32) - 127;
        let half_exp = unbiased_exp + 15;
        if half_exp >= 0x1F {
            return (half_sign | 0x7C00u32) as u16;
        }
        if half_exp <= 0 {
            if 14 - half_exp > 24 {
                return half_sign as u16;
            }
            let man = man | 0x0080_0000u32;
            let mut half_man = man >> (14 - half_exp);
            let round_bit = 1 << (13 - half_exp);
            if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
                half_man += 1;
            }
            return (half_sign | half_man) as u16;
        }
        let half_exp = (half_exp as u32) << 10;
        let half_man = man >> 13;
        let round_bit = 0x0000_1000u32;
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            ((half_sign | half_exp | half_man) + 1) as u16
        } else {
            (half_sign | half_exp | half_man) as u16
        }
    }

    pub const fn f16_to_f32(i: u16) -> u32 {
        if i & 0x7FFFu16 == 0 {
            return (i as u32) << 16;
        }
        let half_sign = (i & 0x8000u16) as u32;
        let half_exp = (i & 0x7C00u16) as u32;
        let half_man = (i & 0x03FFu16) as u32;
        if half_exp == 0x7C00u32 {
            if half_man == 0 {
                return (half_sign << 16) | 0x7F80_0000u32;
            } else {
                return (half_sign << 16) | 0x7FC0_0000u32 | (half_man << 13);
            }
        }
        let sign = half_sign << 16;
        let unbiased_exp = ((half_exp as i32) >> 10) - 15;
        if half_exp == 0 {
            let e = leading_zeros_u16(half_man as u16) - 6;
            let exp = (127 - 15 - e) << 23;
            let man = (half_man << (14 + e)) & 0x7F_FF_FFu32;
            return sign | exp | man;
        }
        let exp = ((unbiased_exp + 127) as u32) << 23;
        let man = (half_man & 0x03FFu32) << 13;
        sign | exp | man
    }

    pub const fn f16_to_f64(i: u16) -> u64 {
        if i & 0x7FFFu16 == 0 {
            return (i as u64) << 48;
        }
        let half_sign = (i & 0x8000u16) as u64;
        let half_exp = (i & 0x7C00u16) as u64;
        let half_man = (i & 0x03FFu16) as u64;
        if half_exp == 0x7C00u64 {
            if half_man == 0 {
                return (half_sign << 48) | 0x7FF0_0000_0000_0000u64;
            } else {
                return (half_sign << 48) | 0x7FF8_0000_0000_0000u64 | (half_man << 42);
            }
        }
        let sign = half_sign << 48;
        let unbiased_exp = ((half_exp as i64) >> 10) - 15;
        if half_exp == 0 {
            let e = leading_zeros_u16(half_man as u16) - 6;
            let exp = ((1023 - 15 - e) as u64) << 52;
            let man = (half_man << (43 + e)) & 0xF_FFFF_FFFF_FFFFu64;
            return sign | exp | man;
        }
        let exp = ((unbiased_exp + 1023) as u64) << 52;
        let man = (half_man & 0x03FFu64) << 42;
        sign | exp | man
    }

    pub const fn f32_to_bf16(x: u32) -> u16 {
        if x & 0x7FFF_FFFFu32 > 0x7F80_0000u32 {
            return ((x >> 16) | 0x0040u32) as u16;
        }
        let round_bit = 0x0000_8000u32;
        if (x & round_bit) != 0 && (x & (3 * round_bit - 1)) != 0 {
            (x >> 16) as u16 + 1
        } else {
            (x >> 16) as u16
        }
    }

    pub const fn f64_to_bf16(val: u64) -> u16 {
        let x = (val >> 32) as u32;
        let sign = x & 0x8000_0000u32;
        let exp = x & 0x7FF0_0000u32;
        let man = x & 0x000F_FFFFu32;
        if exp == 0x7FF0_0000u32 {
            let nan_bit = if man == 0 && (val as u32 == 0) { 0 } else { 0x0040u32 };
            return ((sign >> 16) | 0x7F80u32 | nan_bit | (man >> 13)) as u16;
        }
        let half_sign = sign >> 16;
        let unbiased_exp = ((exp >> 20) as i64) - 1023;
        let half_exp = unbiased_exp + 127;
        if half_exp >= 0xFF {
            return (half_sign | 0x7F80u32) as u16;
        }
        if half_exp <= 0 {
            if 7 - half_exp > 21 {
                return half_sign as u16;
            }
            let man = man | 0x0010_0000u32;
            let mut half_man = man >> (14 - half_exp);
            let round_bit = 1 << (13 - half_exp);
            if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
                half_man += 1;
            }
            return (half_sign | half_man) as u16;
        }
        let half_exp = (half_exp as u32) << 7;
        let half_man = man >> 13;
        let round_bit = 0x0000_1000u32;
        if (man & round_bit) != 0 && (man & (3 * round_bit - 1)) != 0 {
            ((half_sign | half_exp | half_man) + 1) as u16
        } else {
            (half_sign | half_exp | half_man) as u16
        }
    }

    pub const fn bf16_to_f32(i: u16) -> u32 {
        if i & 0x7FFFu16 > 0x7F80u16 {
            (i as u32 | 0x0040u32) << 16
        } else {
            (i as u32) << 16
        }
    }

    pub const fn bf16_to_f64(i: u16) -> u64 {
        if i & 0x7FFFu16 == 0 {
            return (i as u64) << 48;
        }
        let half_sign = (i & 0x8000u16) as u64;
        let half_exp = (i & 0x7F80u16) as u64;
        let half_man = (i & 0x007Fu16) as u64;
        if half_exp == 0x7F80u64 {
            if half_man == 0 {
                return (half_sign << 48) | 0x7FF0_0000_0000_0000u64;
            } else {
                return (half_sign << 48) | 0x7FF8_0000_0000_0000u64 | (half_man << 45);
            }
        }
        let sign = half_sign << 48;
        let unbiased_exp = ((half_exp as i64) >> 7) - 127;
        if half_exp == 0 {
            let e = leading_zeros_u16(half_man as u16) - 9;
            let exp = ((1023 - 127 - e) as u64) << 52;
            let man = (half_man << (46 + e)) & 0xF_FFFF_FFFF_FFFFu64;
            return sign | exp | man;
        }
        let exp = ((unbiased_exp + 1023) as u64) << 52;
        let man = (half_man & 0x007Fu64) << 45;
        sign | exp | man
    }
}

/// FNV-1a over whole values, mirrored by `HalfModelChecks.fnv`.
pub fn fnv(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(0x0000_0100_0000_01B3)
}

pub const FNV_START: u64 = 0xCBF2_9CE4_8422_2325;

/// Deterministic LCG values, mirrored by `HalfModelChecks.lcg`.
pub fn lcg(seed: u64, count: usize) -> Vec<u64> {
    let mut s = seed;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        s = (s * 1_103_515_245 + 12_345) % (1 << 32);
        out.push(s);
    }
    out
}

/// Mantissas sampled for every f32 exponent: bit boundaries around each
/// rounding position plus pseudo-random fill.
pub fn mantissas32() -> Vec<u64> {
    let mask = (1u64 << 23) - 1;
    let mut out = vec![0, mask];
    for p in 0..23u64 {
        let b = 1u64 << p;
        out.extend([b, b | 1, b - 1, (3 * b) & mask, mask ^ b]);
    }
    out.extend(lcg(0x1234_5678, 32).into_iter().map(|s| s & mask));
    out
}

/// Mantissas sampled for f64 exponents near the bfloat16 range.
pub fn mantissas64_full() -> Vec<u64> {
    let mask = (1u64 << 52) - 1;
    let mut out = vec![0, mask];
    for p in 0..52u64 {
        let b = 1u64 << p;
        out.extend([b, b | 1, b - 1]);
    }
    out.extend(lcg(0x0BAD_5EED, 16).into_iter().map(|s| (s << 20 | s) & mask));
    out
}

/// Mantissas sampled for the remaining f64 exponents.
pub fn mantissas64_small() -> Vec<u64> {
    vec![0, 1, 0xFFFF_FFFF, 1 << 32, 1 << 51, (1 << 52) - 1, 1 << 44, (1 << 44) | 1]
}

/// Comparison operands for a 16-bit format with infinity `inf` and one `one`.
pub fn cmp_operands(inf: u64, one: u64) -> Vec<u64> {
    let quiet = inf | ((inf & !(inf - 1)) >> 1);
    let mut out = vec![
        0, 0x8000, 1, 0x8001, inf - 1, 0x8000 | (inf - 1), inf, 0x8000 | inf, inf + 1,
        0x8000 | (inf + 1), quiet, 0x8000 | quiet, 0x7FFF, 0xFFFF, one, one + 1, 0x8000 | one,
        0x8000 | (one + 1), one - 1, 0x0400, 0x0080,
    ];
    out.extend(lcg(0x00C0_FFEE, 19).into_iter().map(|s| s & 0xFFFF));
    out
}
