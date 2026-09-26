// Native-only oracle for FloatModelChecks.lean: compare stdout verbatim.
// NaN arithmetic payloads are not portable, so those results are canonicalized
// to the model's permitted policy; bit conversion/neg/abs are never canonicalized.
macro_rules! check_width {
    ($float:ty, $uint:ty, $width:expr, $frac:expr, $bias:expr, $max_exp:expr) => {{
        let width = $width;
        let frac = $frac;
        let sign: $uint = 1 << (width - 1);
        let one: $uint = $bias << frac;
        let inf: $uint = $max_exp << frac;
        let values = [0, sign, 1, sign + 1, (1 << frac) - 1, 1 << frac,
            one, one + 1, inf - 1, inf, sign + inf, inf + 1, sign + inf + 123,
            inf + (1 << (frac - 1)) + 37];
        for (i, b) in values.into_iter().enumerate() {
            let x = <$float>::from_bits(b);
            println!("bits{width}-{i}={}", x.to_bits());
            for (label, value) in [("is_nan", x.is_nan()), ("is_infinite", x.is_infinite()),
                ("is_finite", x.is_finite())] {
                println!("{label}{width}-{i}={value}");
            }
            for (label, value) in [("neg", -x), ("abs", x.abs())] {
                println!("{label}{width}-{i}={}", value.to_bits());
            }
            for (label, value) in [("trunc", x.trunc()), ("fract", x.fract()), ("floor", x.floor())] {
                if x.is_finite() { println!("{label}{width}-{i}={}", value.to_bits()); }
                else { println!("{label}nan{width}-{i}={}", value.is_nan()); }
            }
            for (j, c) in [0, sign, one, inf, inf + 1].into_iter().enumerate() {
                let y = <$float>::from_bits(c);
                for (label, value) in [("eq", x == y), ("lt", x < y), ("le", x <= y)] {
                    println!("{label}{width}-{i}-{j}={value}");
                }
            }
        }
        let pairs = [(one, one - (frac + 1) * (1 << frac)),
            (one + 1, one - (frac + 1) * (1 << frac)), (1, one - (1 << frac)),
            (3, one - (1 << frac)), ((1 << frac) - 1, 1), (inf - 1, inf - 1),
            (one, 3 * one / 2), (0, sign), (sign, sign), (inf, sign + inf),
            (inf, 0), (one, 0), (0, 0)];
        for (i, (b, c)) in pairs.into_iter().enumerate() {
            let x = <$float>::from_bits(b);
            let y = <$float>::from_bits(c);
            for (label, value) in [("add", x + y), ("sub", x - y),
                ("mul", x * y), ("div", x / y), ("rem", x % y)] {
                println!("{label}{width}-{i}-nan={}", value.is_nan());
                let bits = if value.is_nan() { inf + (1 << (frac - 1)) } else { value.to_bits() };
                println!("{label}{width}-{i}={bits}");
            }
        }
        for (i, n) in [0i64, 1, -1, (1 << (frac + 1)) + 1,
            (1 << (frac + 1)) + 3, i64::MIN, i64::MAX].into_iter().enumerate() {
            println!("fromi64-{width}-{i}={}", (n as $float).to_bits());
        }
        for (i, n) in [0u64, u64::MAX, (1 << 63) + 1].into_iter().enumerate() {
            println!("fromu64-{width}-{i}={}", (n as $float).to_bits());
        }
    }};
}

fn main() {
    check_width!(f32, u32, 32, 23, 127, 255);
    check_width!(f64, u64, 64, 52, 1023, 2047);
    println!("narrow-0={}", (f64::from_bits(0) as f32).to_bits());
    println!("narrow-1={}", (f64::from_bits(9223372036854775808) as f32).to_bits());
    println!("narrow-2={}", (f64::from_bits(4607182419068452864) as f32).to_bits());
    println!("narrow-3={}", (f64::from_bits(4607182419068452865) as f32).to_bits());
    println!("narrow-4={}", (f64::from_bits(4607182419605323776) as f32).to_bits());
    println!("narrow-5={}", (f64::from_bits(3931642474694443008) as f32).to_bits());
    println!("narrow-6={}", (f64::from_bits(3931642474694443009) as f32).to_bits());
    println!("narrow-7={}", (f64::from_bits(3938397874135498752) as f32).to_bits());
    println!("narrow-8={}", (f64::from_bits(4039728865214464000) as f32).to_bits());
    println!("narrow-9={}", (f64::from_bits(4039728865214464001) as f32).to_bits());
    println!("narrow-10={}", (f64::from_bits(5183643170566569984) as f32).to_bits());
    println!("narrow-11={}", (f64::from_bits(5183643170835005440) as f32).to_bits());
    println!("narrow-12={}", (f64::from_bits(5183643170835005439) as f32).to_bits());
    println!("narrow-13={}", (f64::from_bits(9218868437227405312) as f32).to_bits());
    println!("narrow-14={}", (f64::from_bits(18442240474082181120) as f32).to_bits());
    println!("widen-0={}", (f32::from_bits(0) as f64).to_bits());
    println!("widen-1={}", (f32::from_bits(2147483648) as f64).to_bits());
    println!("widen-2={}", (f32::from_bits(1) as f64).to_bits());
    println!("widen-3={}", (f32::from_bits(8388607) as f64).to_bits());
    println!("widen-4={}", (f32::from_bits(8388608) as f64).to_bits());
    println!("widen-5={}", (f32::from_bits(1065353217) as f64).to_bits());
    println!("widen-6={}", (f32::from_bits(2139095039) as f64).to_bits());
    println!("widen-7={}", (f32::from_bits(2139095040) as f64).to_bits());
    println!("widen-8={}", (f32::from_bits(4286578688) as f64).to_bits());
    for (i, b) in [0u32, 2147483648, 1, 1065353217, 2139095040, 4286578688].into_iter().enumerate() {
        println!("widen-from-{i}={}", f64::from(f32::from_bits(b)).to_bits());
    }
    let cast_inputs: [u64; 14] = [0x0, 0x8000000000000000, 0x3ff8000000000000, 0xbff8000000000000, 0x43dfffffffffffff, 0x43e0000000000000, 0xc3e0000000000000, 0xc3e0000000000001, 0x43efffffffffffff, 0x43f0000000000000, 0x47e0000000000000, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff0000000000001];
    for (i, b) in cast_inputs.into_iter().enumerate() {
        println!("cast-i8-{i}={}", f64::from_bits(b) as i8);
        println!("cast-i16-{i}={}", f64::from_bits(b) as i16);
        println!("cast-i32-{i}={}", f64::from_bits(b) as i32);
        println!("cast-i64-{i}={}", f64::from_bits(b) as i64);
        println!("cast-i128-{i}={}", f64::from_bits(b) as i128);
        println!("cast-u8-{i}={}", f64::from_bits(b) as u8);
        println!("cast-u16-{i}={}", f64::from_bits(b) as u16);
        println!("cast-u32-{i}={}", f64::from_bits(b) as u32);
        println!("cast-u64-{i}={}", f64::from_bits(b) as u64);
        println!("cast-u128-{i}={}", f64::from_bits(b) as u128);
    }
    println!("powi-0={}", f64::from_bits(4611686018427387904).powi(63).to_bits());
    println!("powi-1={}", f64::from_bits(4611686018427387904).powi(-63).to_bits());
    println!("powi-2={}", f64::from_bits(13835058055282163712).powi(63).to_bits());
    println!("powi-3={}", f64::from_bits(0).powi(0).to_bits());
    println!("powi-4={}", f64::from_bits(9223372036854775808).powi(3).to_bits());
    println!("powi-5={}", f64::from_bits(9223372036854775808).powi(-3).to_bits());
    println!("powi-6={}", f64::from_bits(9218868437227405312).powi(-2).to_bits());
    println!("powi-7={}", f64::from_bits(9218868437227405313).powi(0).to_bits());
    println!("powi-8={}", 2.0f64.powi(i32::MAX).to_bits());
    println!("powi-9={}", 2.0f64.powi(i32::MIN).to_bits());
    println!("powi-10={}", 1.0f64.powi(i32::MAX).to_bits());
}
