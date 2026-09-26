//! Exhaustive check that the `half` 2.7.1 conversions actually executed on this
//! host (on aarch64 with `fp16`: FCVT) equal the crate's software algorithms and
//! their transliteration in `crate_model_oracle::sw` (= the Lean models), and
//! that the crate's comparison impls equal IEEE comparison of the exactly
//! widened values. Exits nonzero on the first class of disagreement.

use crate_model_oracle::sw;
use half::{bf16, f16};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

const THREADS: u64 = 16;

struct Check {
    name: &'static str,
    mismatches: AtomicU64,
    first: AtomicU64,
}

impl Check {
    const fn new(name: &'static str) -> Self {
        Check { name, mismatches: AtomicU64::new(0), first: AtomicU64::new(u64::MAX) }
    }
    fn fail(&self, input: u64) {
        self.mismatches.fetch_add(1, Ordering::Relaxed);
        self.first.fetch_min(input, Ordering::Relaxed);
    }
    fn report(&self, domain: u64) -> bool {
        let n = self.mismatches.load(Ordering::Relaxed);
        if n == 0 {
            println!("{}: {domain} inputs agree", self.name);
            true
        } else {
            println!(
                "{}: {n} of {domain} inputs DISAGREE (first input {:#x})",
                self.name,
                self.first.load(Ordering::Relaxed)
            );
            false
        }
    }
}

/// Run `body` on every u32, split over threads.
fn all_u32(body: impl Fn(u32) + Sync) {
    let chunk = (1u64 << 32) / THREADS;
    std::thread::scope(|scope| {
        for t in 0..THREADS {
            let body = &body;
            scope.spawn(move || {
                for x in t * chunk..(t + 1) * chunk {
                    body(x as u32);
                }
            });
        }
    });
}

fn main() {
    println!(
        "target_arch={} target_feature(fp16)={}",
        std::env::consts::ARCH,
        cfg!(target_feature = "fp16")
    );
    #[cfg(target_arch = "aarch64")]
    println!("runtime fp16 detected={}", std::arch::is_aarch64_feature_detected!("fp16"));

    let from_f32_hw_sw = Check::new("f16::from_f32 (dispatched) == f16::from_f32_const");
    let from_f32_sw_model = Check::new("f16::from_f32_const == transliteration");
    let bf_from_f32 = Check::new("bf16::from_f32 == transliteration");
    let bf_from_f64 = Check::new("bf16::from_f64 == transliteration (all high words x 4 low words)");
    all_u32(|x| {
        let f = f32::from_bits(x);
        let hw = f16::from_f32(black_box(f)).to_bits();
        let crate_sw = f16::from_f32_const(black_box(f)).to_bits();
        if hw != crate_sw {
            from_f32_hw_sw.fail(x as u64);
        }
        if crate_sw != sw::f32_to_f16(x) {
            from_f32_sw_model.fail(x as u64);
        }
        if bf16::from_f32(black_box(f)).to_bits() != sw::f32_to_bf16(x) {
            bf_from_f32.fail(x as u64);
        }
        for low in [0u64, 1, 0x8000_0000, 0xFFFF_FFFF] {
            let bits = ((x as u64) << 32) | low;
            if bf16::from_f64(black_box(f64::from_bits(bits))).to_bits() != sw::f64_to_bf16(bits) {
                bf_from_f64.fail(bits);
            }
        }
    });

    let to_f32_hw_sw = Check::new("f16::to_f32 (dispatched) == f16::to_f32_const");
    let to_f32_sw_model = Check::new("f16::to_f32_const == transliteration");
    let to_f64_hw_sw = Check::new("f16::to_f64 (dispatched) == f16::to_f64_const");
    let to_f64_sw_model = Check::new("f16::to_f64_const == transliteration");
    let bf_to_f32 = Check::new("bf16::to_f32 == transliteration");
    let bf_to_f64 = Check::new("bf16::to_f64 == transliteration");
    for i in 0..=u16::MAX {
        let h = f16::from_bits(black_box(i));
        let crate32 = h.to_f32_const().to_bits();
        if h.to_f32().to_bits() != crate32 {
            to_f32_hw_sw.fail(i as u64);
        }
        if crate32 != sw::f16_to_f32(i) {
            to_f32_sw_model.fail(i as u64);
        }
        let crate64 = h.to_f64_const().to_bits();
        if h.to_f64().to_bits() != crate64 {
            to_f64_hw_sw.fail(i as u64);
        }
        if crate64 != sw::f16_to_f64(i) {
            to_f64_sw_model.fail(i as u64);
        }
        if f64::from(h).to_bits() != crate64 {
            to_f64_hw_sw.fail(i as u64);
        }
        let b = bf16::from_bits(black_box(i));
        if b.to_f32().to_bits() != sw::bf16_to_f32(i) {
            bf_to_f32.fail(i as u64);
        }
        if b.to_f64().to_bits() != sw::bf16_to_f64(i) || f64::from(b).to_bits() != sw::bf16_to_f64(i) {
            bf_to_f64.fail(i as u64);
        }
    }

    // All ordered pairs: the crate impls against IEEE comparison of the exactly
    // widened f32 values (the semantics the Lean lemmas prove the models equal).
    let f16_cmp = Check::new("f16 PartialEq/PartialOrd == IEEE on to_f32 (all pairs)");
    let bf16_cmp = Check::new("bf16 PartialEq/PartialOrd == IEEE on to_f32 (all pairs)");
    all_u32(|pair| {
        let (a, b) = ((pair >> 16) as u16, pair as u16);
        let (x, y) = (f16::from_bits(a), f16::from_bits(b));
        let (fx, fy) = (sw::f16_to_f32(a), sw::f16_to_f32(b));
        let (fx, fy) = (f32::from_bits(fx), f32::from_bits(fy));
        if (x == y) != (fx == fy) || (x != y) != (fx != fy) || x.partial_cmp(&y) != fx.partial_cmp(&fy)
            || (x < y) != (fx < fy) || (x <= y) != (fx <= fy) || (x > y) != (fx > fy)
            || (x >= y) != (fx >= fy) || x.is_nan() != fx.is_nan()
        {
            f16_cmp.fail(pair as u64);
        }
        let (x, y) = (bf16::from_bits(a), bf16::from_bits(b));
        let (fx, fy) = (f32::from_bits(sw::bf16_to_f32(a)), f32::from_bits(sw::bf16_to_f32(b)));
        if (x == y) != (fx == fy) || (x != y) != (fx != fy) || x.partial_cmp(&y) != fx.partial_cmp(&fy)
            || (x < y) != (fx < fy) || (x <= y) != (fx <= fy) || (x > y) != (fx > fy)
            || (x >= y) != (fx >= fy) || x.is_nan() != fx.is_nan()
        {
            bf16_cmp.fail(pair as u64);
        }
    });

    let mut ok = true;
    for check in [&from_f32_hw_sw, &from_f32_sw_model, &bf_from_f32] {
        ok &= check.report(1 << 32);
    }
    ok &= bf_from_f64.report(4 << 32);
    for check in [&to_f32_hw_sw, &to_f32_sw_model, &to_f64_hw_sw, &to_f64_sw_model, &bf_to_f32, &bf_to_f64] {
        ok &= check.report(1 << 16);
    }
    for check in [&f16_cmp, &bf16_cmp] {
        ok &= check.report(1 << 32);
    }
    if !ok {
        std::process::exit(1);
    }
    println!("half hardware/software/model agreement: all exhaustive checks pass");
}
