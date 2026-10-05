//! Compare erased proof code with plain identical code and the former linear admission.

use std::{hint::black_box, time::Instant};
#[inline(never)]
fn linear(axes: &[usize], rank: usize) -> bool {
    if axes.len() != rank {
        return false;
    }
    let mut seen = vec![false; rank];
    for &a in axes {
        if a >= rank || seen[a] {
            return false;
        }
        seen[a] = true;
    }
    true
}
#[inline(never)]
fn plain_quadratic(axes: &[usize], rank: usize) -> bool {
    if axes.len() != rank {
        return false;
    }
    let mut i = 0;
    while i < rank {
        if axes[i] >= rank {
            return false;
        }
        let mut j = 0;
        while j < i {
            if axes[j] == axes[i] {
                return false;
            }
            j += 1;
        }
        i += 1;
    }
    true
}
#[cfg(axis_verified)]
fn candidate(axes: &[usize], rank: usize) -> bool {
    chelis_axis_core::is_permutation(axes, rank)
}
#[cfg(not(axis_verified))]
fn candidate(axes: &[usize], rank: usize) -> bool {
    plain_quadratic(axes, rank)
}
fn nanos(f: fn(&[usize], usize) -> bool, a: &[usize], n: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..n {
        black_box(f(black_box(a), black_box(a.len())));
    }
    start.elapsed().as_nanos() as f64 / n as f64
}
fn main() {
    for rank in 0usize..=5 {
        let base = rank + 1;
        for value in 0..base.pow(rank as u32) {
            let mut x = value;
            let a: Vec<usize> = (0..rank)
                .map(|_| {
                    let d = x % base;
                    x /= base;
                    d
                })
                .collect();
            assert_eq!(linear(&a, rank), candidate(&a, rank));
            assert_eq!(plain_quadratic(&a, rank), candidate(&a, rank));
        }
    }
    println!("rank,candidate_ns,plain_quadratic_ns,linear_ns");
    for rank in [4, 16, 64, 256, 1024] {
        let a: Vec<usize> = (0..rank).rev().collect();
        let n = (20_000_000usize / (rank * rank)).clamp(100, 200_000);
        let mut samples = Vec::new();
        for _ in 0..5 {
            samples.push((
                nanos(candidate, &a, n),
                nanos(plain_quadratic, &a, n),
                nanos(linear, &a, n),
            ));
        }
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (c, p, l) = samples[2];
        println!("{rank},{c:.2},{p:.2},{l:.2}");
    }
}
