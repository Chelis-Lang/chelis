//! The f16 and bf16 storage conversions against the integer reference
//! (`chelis_crmath::profile::storage_reference`, [04-NUM-2], [04-NUM-14]).
//!
//! Per pull request: the header's `chelis_f32_to_f16` and `chelis_f32_to_bf16`,
//! compiled with the product profile, on every f32 storage-conversion row of the
//! profile obligation table and on every f32 within two ulps of an f16 or bf16
//! rounding boundary; and the evaluator's `chelis_types::{f16,bf16}_from_f64_rne` on
//! every f64 row and at every f64 rounding boundary. The manual gate
//! (`docs/manual_gates.md`) checks every f32 input, through the header and through
//! the evaluator's f64 conversion of its exact widening.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use chelis_crmath::profile::{Output, rows, storage_midpoint_inputs, storage_reference};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "chelis-storage-narrowing-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The driver compiled with the product's floating-point profile.
fn driver(scratch: &Scratch) -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let program = scratch.0.join("storage_narrowing");
    let output = Command::new("cc")
        .args(["-std=c11", "-O2", "-ffp-contract=off", "-fno-fast-math", "-I"])
        .arg(manifest.join("include"))
        .arg(manifest.join("tests/fixtures/storage_narrowing.c"))
        .arg("-o")
        .arg(&program)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    program
}

/// The header's (f16, bf16) bits for each f32 bit pattern.
fn header_bits(program: &Path, inputs: &[u32]) -> Vec<(u16, u16)> {
    let mut child = Command::new(program)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let text: String = inputs.iter().map(|bits| format!("{bits:x}\n")).collect();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(text.as_bytes()));
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap().unwrap();
    assert!(output.status.success());
    let parsed: Vec<(u16, u16)> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| {
            let (f16, bf16) = line.split_once(' ').unwrap();
            (
                u16::from_str_radix(f16, 16).unwrap(),
                u16::from_str_radix(bf16, 16).unwrap(),
            )
        })
        .collect();
    assert_eq!(parsed.len(), inputs.len());
    parsed
}

fn evaluator_bits(bits: u64, output: Output) -> u16 {
    let value = f64::from_bits(bits);
    match output {
        Output::F16 => chelis_types::f16_from_f64_rne(value).to_bits(),
        Output::Bf16 => chelis_types::bf16_from_f64_rne(value).to_bits(),
        other => unreachable!("{other:?}"),
    }
}

fn mismatch(lane: &str, bits: u64, output: Output, got: u16, expected: u16) -> String {
    format!("{lane} {output:?} input {bits:x}: got {got:04x}, reference {expected:04x}")
}

/// Every f32 within two ulps of an f16 or bf16 rounding boundary, and the f32 rows.
fn f32_boundary_inputs() -> Vec<u32> {
    let mut inputs: Vec<u32> = rows()
        .iter()
        .filter(|row| {
            row.primitive.width == 32 && matches!(row.primitive.result, Output::F16 | Output::Bf16)
        })
        .map(|row| u32::try_from(row.operands[0]).unwrap())
        .collect();
    for output in [Output::F16, Output::Bf16] {
        for bits in storage_midpoint_inputs(output) {
            #[allow(clippy::cast_possible_truncation)]
            let middle = f64::from_bits(bits) as f32;
            if f64::from(middle).to_bits() != bits || !middle.is_finite() {
                continue;
            }
            let center = middle.to_bits();
            inputs.extend((0..=4).filter_map(|step| (center + 2).checked_sub(step)));
        }
    }
    inputs.sort_unstable();
    inputs.dedup();
    inputs
}

#[test]
fn header_narrowing_matches_the_reference_at_every_f32_boundary() {
    let scratch = Scratch::new();
    let program = driver(&scratch);
    let inputs = f32_boundary_inputs();
    assert!(inputs.len() > 100_000, "{} inputs", inputs.len());
    let mut bad = Vec::new();
    for (&bits, (f16, bf16)) in inputs.iter().zip(header_bits(&program, &inputs)) {
        for (output, got) in [(Output::F16, f16), (Output::Bf16, bf16)] {
            let expected = storage_reference(u64::from(bits), 32, output);
            if got != expected {
                bad.push(mismatch("header", u64::from(bits), output, got, expected));
            }
        }
    }
    assert!(bad.is_empty(), "{} mismatches, first: {:?}", bad.len(), &bad[..bad.len().min(8)]);
}

#[test]
fn evaluator_narrowing_matches_the_reference_at_every_f64_boundary() {
    let mut bad = Vec::new();
    for output in [Output::F16, Output::Bf16] {
        let row_inputs = rows()
            .iter()
            .filter(|row| row.primitive.width == 64 && row.primitive.result == output)
            .map(|row| row.operands[0]);
        for bits in storage_midpoint_inputs(output).into_iter().chain(row_inputs) {
            let expected = storage_reference(bits, 64, output);
            let got = evaluator_bits(bits, output);
            if got != expected {
                bad.push(mismatch("evaluator", bits, output, got, expected));
            }
        }
    }
    assert!(bad.is_empty(), "{} mismatches, first: {:?}", bad.len(), &bad[..bad.len().min(8)]);
}

/// Manual gate: every f32 bit pattern through the header, and its exact f64
/// widening through the evaluator. `CHELIS_NARROWING_THREADS` sets the worker count
/// (default: half the available CPUs).
#[test]
#[ignore = "manual gate: 2^32 inputs (docs/manual_gates.md)"]
fn every_f32_narrows_once_to_f16_and_bf16_storage() {
    const CHUNK: u64 = 1 << 24;
    let scratch = Scratch::new();
    let program = driver(&scratch);
    let threads = std::env::var("CHELIS_NARROWING_THREADS")
        .ok()
        .and_then(|text| text.parse::<u64>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, |count| (count.get() as u64 / 2).max(1))
        });
    let next = AtomicU64::new(0);
    let counts: Vec<(u64, Vec<String>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut count = 0_u64;
                    let mut examples = Vec::new();
                    loop {
                        let low = next.fetch_add(CHUNK, Ordering::Relaxed);
                        if low >= 1 << 32 {
                            break;
                        }
                        let output = Command::new(&program)
                            .args([format!("{low:x}"), format!("{:x}", low + CHUNK)])
                            .output()
                            .unwrap();
                        assert!(output.status.success());
                        let words: Vec<u16> = output
                            .stdout
                            .chunks_exact(2)
                            .map(|pair| u16::from_ne_bytes([pair[0], pair[1]]))
                            .collect();
                        assert_eq!(words.len() as u64, 2 * CHUNK);
                        for offset in 0..CHUNK {
                            let bits = low + offset;
                            let value = f32::from_bits(u32::try_from(bits).unwrap());
                            let wide = f64::from(value).to_bits();
                            for (index, output) in [Output::F16, Output::Bf16].into_iter().enumerate() {
                                let expected = storage_reference(bits, 32, output);
                                let header = words[(2 * offset) as usize + index];
                                let evaluator = evaluator_bits(wide, output);
                                for (lane, got) in [("header", header), ("evaluator", evaluator)] {
                                    if got != expected {
                                        count += 1;
                                        if examples.len() < 4 {
                                            examples.push(mismatch(lane, bits, output, got, expected));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    (count, examples)
                })
            })
            .collect();
        workers.into_iter().map(|worker| worker.join().unwrap()).collect()
    });
    let total: u64 = counts.iter().map(|(count, _)| count).sum();
    let examples: Vec<&String> = counts.iter().flat_map(|(_, examples)| examples).collect();
    println!("f32 storage narrowing: {total} mismatches over 2^32 inputs");
    assert_eq!(total, 0, "first mismatches: {examples:?}");
}
