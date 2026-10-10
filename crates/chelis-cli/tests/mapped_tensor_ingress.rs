//! CLI surface coverage for [05-OP-80] `mmap_tensor`, [05-OP-81] `mmap_text`
//! and `mmap_sha256`, and [05-OP-82] `Std.Io.Tensors`.
//!
//! Every accepted program runs through `chelis eval` and through a compiled C
//! executable, and the two lanes' outputs must be byte-equal as well as equal
//! to the expected text. Every failure has the same final lines in both
//! lanes. Fixture files are written by the test with absolute paths, so the
//! lanes' working directories cannot change what they read.

use assert_cmd::Command;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::{TempDir, tempdir};

fn chelis() -> Command {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

/// The exit status, stdout and stderr of one lane.
struct Run {
    success: bool,
    stdout: String,
    stderr: String,
}

fn literal(path: &Path) -> String {
    format!("{:?}", path.to_str().expect("UTF-8 fixture path"))
}

fn write_source(dir: &TempDir, name: &str, source: &str) -> std::path::PathBuf {
    let path = dir.path().join(format!("{name}.ch"));
    fs::write(&path, source).expect("write source");
    path
}

fn eval(dir: &TempDir, name: &str, source: &str) -> Run {
    let path = write_source(dir, name, source);
    let output = chelis()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval runs");
    Run {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn compiled(dir: &TempDir, name: &str, source: &str) -> Run {
    let path = write_source(dir, name, source);
    let out_dir = dir.path().join(format!("{name}-out"));
    let build = chelis()
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(&out_dir)
        .output()
        .expect("chelis build runs");
    assert!(
        build.status.success(),
        "build of `{name}` failed:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let output = StdCommand::new(out_dir.join(name))
        .output()
        .expect("the compiled executable runs");
    Run {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Both lanes succeed with exactly `expected` on stdout.
fn assert_lanes_print(dir: &TempDir, name: &str, source: &str, expected: &str) {
    for (lane, run) in [
        ("eval", eval(dir, name, source)),
        ("C", compiled(dir, name, source)),
    ] {
        assert!(run.success, "{lane} failed for `{name}`:\n{}", run.stderr);
        assert_eq!(run.stdout, expected, "{lane} stdout for `{name}`");
    }
}

/// The last `lines` nonempty lines of a failure's stderr, without the lane's
/// `error: ` boundary prefix.
fn tail(stderr: &str, lines: usize) -> Vec<String> {
    let nonempty: Vec<&str> = stderr
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    nonempty[nonempty.len().saturating_sub(lines)..]
        .iter()
        .map(|line| line.trim_start_matches("error: ").to_string())
        .collect()
}

/// Both lanes fail, and their stderr ends with exactly `expected`.
fn assert_lanes_fail(dir: &TempDir, name: &str, source: &str, expected: &[&str]) {
    for (lane, run) in [
        ("eval", eval(dir, name, source)),
        ("C", compiled(dir, name, source)),
    ] {
        assert!(
            !run.success,
            "{lane} must fail for `{name}`; stdout:\n{}",
            run.stdout
        );
        assert_eq!(
            tail(&run.stderr, expected.len()),
            expected,
            "{lane} failure for `{name}`; stderr:\n{}",
            run.stderr
        );
    }
}

/// A program whose `main` opens `path` and runs `body` with the handle `m`.
fn program(path: &Path, body: &str) -> String {
    format!(
        "def main() -> unit ! {{ IO }} = {{\n  m = mmap_file({})\n{body}\n}}\nrun = main()\n",
        literal(path)
    )
}

/// One `(dtype, byte offset, element count, expected rendering)` row.
type DtypeRow = (&'static str, usize, usize, &'static str);

/// One payload per data element dtype, each at an unaligned offset after a
/// three-byte prefix, with its expected rendering.
fn every_dtype_fixture() -> (Vec<u8>, Vec<DtypeRow>) {
    let mut bytes = b"pre".to_vec();
    let mut rows = Vec::new();
    let mut push = |dtype, count, payload: Vec<u8>, rendered| {
        rows.push((dtype, bytes.len(), count, rendered));
        bytes.extend(payload);
    };
    push(
        "f64",
        2,
        [0.1f64, -0.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        "tensor(shape=[2], data=[0.1, -0.0])",
    );
    push(
        "f32",
        3,
        [1.5f32, f32::NEG_INFINITY, -2.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        "tensor(shape=[3], data=[1.5, -inf, -2.0])",
    );
    push(
        "f16",
        2,
        [0x3c00u16, 0xc000]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        "tensor(shape=[2], data=[1.0, -2.0])",
    );
    push(
        "bf16",
        2,
        [0x3f80u16, 0x3f00]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        "tensor(shape=[2], data=[1.0, 0.5])",
    );
    push(
        "i64",
        2,
        [i64::MIN, 7].iter().flat_map(|x| x.to_le_bytes()).collect(),
        "tensor(shape=[2], data=[-9223372036854775808, 7])",
    );
    push(
        "i32",
        2,
        [-9i32, i32::MAX]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
        "tensor(shape=[2], data=[-9, 2147483647])",
    );
    push(
        "i16",
        2,
        [i16::MIN, 5].iter().flat_map(|x| x.to_le_bytes()).collect(),
        "tensor(shape=[2], data=[-32768, 5])",
    );
    push(
        "i8",
        2,
        vec![0x80, 0x7f],
        "tensor(shape=[2], data=[-128, 127])",
    );
    push(
        "bool",
        3,
        vec![1, 0, 1],
        "tensor(shape=[3], data=[true, false, true])",
    );
    (bytes, rows)
}

/// [05-OP-80]: every active data element dtype reads its little-endian
/// payload bit for bit, at any offset, identically in both lanes.
#[test]
fn every_data_dtype_reads_its_payload_in_eval_and_c() {
    let dir = tempdir().expect("tempdir");
    let data = dir.path().join("payload.bin");
    let (bytes, rows) = every_dtype_fixture();
    fs::write(&data, bytes).expect("write payload");
    let body = rows
        .iter()
        .map(|(dtype, offset, count, _)| {
            format!("  _ = print(mmap_tensor(m, {offset}i64, {count}i64, {dtype}))")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let expected = rows
        .iter()
        .map(|(_, _, _, rendered)| format!("{rendered}\n"))
        .collect::<String>()
        + "run = ()\n";
    assert_lanes_print(
        &dir,
        "every_dtype",
        &program(&data, &format!("{body}\n  ()")),
        &expected,
    );
}

/// A computed count is a fresh extent, and a declared result extent is
/// guarded against it at run time; the guard names `mmap_tensor`.
#[test]
fn a_declared_extent_guards_a_computed_count() {
    let dir = tempdir().expect("tempdir");
    let data = dir.path().join("payload.bin");
    fs::write(
        &data,
        [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .expect("write payload");
    let read = |declared: usize| {
        program(
            &data,
            &format!(
                "  n = add(mmap_len(m), 0i64) |> floor_div(4i64)\n  w: tensor[{declared}, f32] = mmap_tensor(m, 0i64, n, f32)\n  print(w)"
            ),
        )
    };
    assert_lanes_print(
        &dir,
        "agreeing_extent",
        &read(3),
        "tensor(shape=[3], data=[1.0, 2.0, 3.0])\nrun = ()\n",
    );
    for run in [
        eval(&dir, "disagreeing_extent", &read(4)),
        compiled(&dir, "disagreeing_extent", &read(4)),
    ] {
        assert!(
            !run.success,
            "a disagreeing extent must trap: {}",
            run.stdout
        );
        assert_eq!(
            tail(&run.stderr, 1),
            ["numeric trap: domain in mmap_tensor at i64"],
            "{}",
            run.stderr
        );
    }
}

/// [05-OP-80]: a range outside the mapping, a negative or overflowing count,
/// and a `bool` byte other than 0 or 1 trap identically in both lanes. The
/// overflowing count is computed, so the trap is the run-time one.
#[test]
fn invalid_tensor_reads_trap_in_both_lanes() {
    let dir = tempdir().expect("tempdir");
    let data = dir.path().join("payload.bin");
    fs::write(&data, [0u8, 1, 2, 3, 4, 5, 6, 7]).expect("write payload");
    let cases: [(&str, &str, &[&str]); 5] = [
        (
            "past_end",
            "mmap_tensor(m, 4i64, 2i64, i32)",
            &[
                "mmap_tensor offset 4, count 2, byte length 8, mapping length 8: the range ends past the mapping",
                "numeric trap: domain in mmap_tensor at i64",
            ],
        ),
        (
            "negative_offset",
            "mmap_tensor(m, -1i64, 1i64, i8)",
            &[
                "mmap_tensor offset -1, count 1, byte length 1, mapping length 8: the offset is negative",
                "numeric trap: domain in mmap_tensor at i64",
            ],
        ),
        (
            "negative_count",
            "mmap_tensor(m, 0i64, -1i64, i8)",
            &[
                "mmap_tensor offset 0, count -1, mapping length 8: the count is negative",
                "numeric trap: domain in mmap_tensor at i64",
            ],
        ),
        (
            "overflowing_count",
            "mmap_tensor(m, 0i64, mul(mmap_len(m), 576460752303423488i64), f32)",
            &[
                "mmap_tensor offset 0, count 4611686018427387904, mapping length 8: the byte length overflows i64",
                "numeric trap: overflow in mmap_tensor at i64",
            ],
        ),
        (
            "bool_byte",
            "mmap_tensor(m, 0i64, 4i64, bool)",
            &[
                "mmap_tensor bool element 2 has byte 2, expected 0 or 1",
                "numeric trap: domain in mmap_tensor at bool",
            ],
        ),
    ];
    for (name, call, expected) in cases {
        assert_lanes_fail(
            &dir,
            name,
            &program(&data, &format!("  print({call})")),
            expected,
        );
    }
}

/// [05-OP-60]: a byte read past the end of the mapping traps rather than
/// returning the bytes that exist.
#[test]
fn a_byte_read_past_the_mapping_traps() {
    let dir = tempdir().expect("tempdir");
    let data = dir.path().join("payload.bin");
    fs::write(&data, b"abc").expect("write payload");
    assert_lanes_print(
        &dir,
        "byte_read_inside",
        &program(&data, "  print(mmap_read(m, 1i64, 2i64))"),
        "[98, 99]\nrun = ()\n",
    );
    assert_lanes_fail(
        &dir,
        "byte_read_past",
        &program(&data, "  print(mmap_read(m, 1i64, 3i64))"),
        &[
            "mmap_read offset 1, byte length 3, mapping length 3: the range ends past the mapping",
            "numeric trap: domain in mmap_read at i64",
        ],
    );
    // [05-OP-35]: `Std.Io.read_head_bytes` still returns the first
    // `min(count, file_length)` bytes.
    assert_lanes_print(
        &dir,
        "head_bytes",
        &format!(
            "import Std.Io (read_head_bytes)\ndef main() -> unit ! {{ IO }} = print(read_head_bytes({}, 10i64))\nrun = main()\n",
            literal(&data)
        ),
        "[97, 98, 99]\nrun = ()\n",
    );
}

/// [05-OP-81]: exact UTF-8 text and the lowercase hexadecimal SHA-256 of a
/// range, and the failures of an invalid range or invalid UTF-8.
#[test]
fn text_and_digests_of_a_range() {
    let dir = tempdir().expect("tempdir");
    let data = dir.path().join("payload.bin");
    let mut bytes = "abc h\u{e9}llo".as_bytes().to_vec();
    bytes.push(0xff);
    fs::write(&data, &bytes).expect("write payload");
    let empty = format!("{:x}", Sha256::digest(b""));
    let abc = format!("{:x}", Sha256::digest(b"abc"));
    assert_lanes_print(
        &dir,
        "text_and_digests",
        &program(
            &data,
            "  _ = print(mmap_text(m, 4i64, 6i64))\n  _ = print(mmap_sha256(m, 0i64, 3i64))\n  print(mmap_sha256(m, 3i64, 0i64))",
        ),
        &format!("h\u{e9}llo\n{abc}\n{empty}\nrun = ()\n"),
    );
    assert_lanes_fail(
        &dir,
        "invalid_utf8",
        &program(&data, "  print(mmap_text(m, 4i64, 7i64))"),
        &["mmap_text: invalid UTF-8 at byte 10"],
    );
    assert_lanes_fail(
        &dir,
        "digest_past_end",
        &program(&data, "  print(mmap_sha256(m, 8i64, 4i64))"),
        &[
            "mmap_sha256 offset 8, byte length 4, mapping length 11: the range ends past the mapping",
            "numeric trap: domain in mmap_sha256 at i64",
        ],
    );
}

fn check(dir: &TempDir, name: &str, source: &str) -> Run {
    let path = write_source(dir, name, source);
    let output = chelis()
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check runs");
    Run {
        success: output.status.success(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// spec/04 §8.6 and spec/02 §P9: `mmap_tensor` is reserved, and its fourth
/// argument names a dtype, never a value.
#[test]
fn the_dtype_position_is_reserved_syntax() {
    let dir = tempdir().expect("tempdir");
    let accepted = check(
        &dir,
        "accepted",
        "def ok(m: MappedFile) -> tensor[*, f32] = mmap_tensor(m, 0i64, 1i64, f32)\n",
    );
    assert!(accepted.success, "{}{}", accepted.stdout, accepted.stderr);
    let bound = check(
        &dir,
        "bound",
        "def ok(mmap_tensor: i64) -> i64 = mmap_tensor\n",
    );
    assert!(!bound.success);
    assert!(
        format!("{}{}", bound.stdout, bound.stderr)
            .contains("`mmap_tensor` is reserved and cannot be bound"),
        "{}{}",
        bound.stdout,
        bound.stderr
    );
    let value = check(
        &dir,
        "value",
        "def bad(m: MappedFile, n: i64) -> tensor[*, f32] = mmap_tensor(m, 0i64, 1i64, n)\n",
    );
    assert!(!value.success);
    assert!(
        format!("{}{}", value.stdout, value.stderr)
            .contains("the fourth argument of `mmap_tensor` must be a dtype"),
        "{}{}",
        value.stdout,
        value.stderr
    );
}

/// The dtype argument survives formatting and the Deep round trip.
#[test]
fn the_dtype_argument_round_trips_through_deep() {
    let dir = tempdir().expect("tempdir");
    let source = "def ok(m: MappedFile) -> tensor[*, bf16] = mmap_tensor(m, 0i64, 1i64, bf16)\n";
    let path = write_source(&dir, "round", source);
    chelis()
        .args(["fmt", "--check", path.to_str().unwrap()])
        .assert()
        .success();
    let deep = chelis()
        .args(["deep", path.to_str().unwrap()])
        .output()
        .expect("chelis deep runs");
    assert!(
        deep.status.success(),
        "{}",
        String::from_utf8_lossy(&deep.stderr)
    );
    let deep_text = String::from_utf8(deep.stdout).expect("utf-8");
    assert!(deep_text.contains("(t-prim {} bf16)"), "{deep_text}");
    let deep_path = dir.path().join("round.dp");
    fs::write(&deep_path, &deep_text).expect("write deep");
    let surf = chelis()
        .args(["surf", deep_path.to_str().unwrap()])
        .output()
        .expect("chelis surf runs");
    assert!(
        surf.status.success(),
        "{}",
        String::from_utf8_lossy(&surf.stderr)
    );
    assert_eq!(String::from_utf8(surf.stdout).expect("utf-8"), source);
}

/// One tensor of a `.hnw` archive: id, dtype, shape and payload bytes.
type ArchiveTensor = (&'static str, &'static str, Vec<i64>, Vec<u8>);

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A hydronnx `.hnw` archive (format 1.0) with a canonical manifest:
/// sorted keys, no insignificant whitespace, 64-byte aligned payloads.
fn hnw(tensors: &[ArchiveTensor]) -> Vec<u8> {
    hnw_edited(tensors, |manifest| manifest)
}

/// [`hnw`] with `edit` applied to the manifest text before its digest is
/// recorded, so the archive is intact but its metadata says something else.
fn hnw_edited(tensors: &[ArchiveTensor], edit: impl Fn(String) -> String) -> Vec<u8> {
    let mut payload = Vec::new();
    let mut entries = Vec::new();
    for (id, dtype, shape, data) in tensors {
        while !payload.len().is_multiple_of(64) {
            payload.push(0);
        }
        let shape = shape
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            "{{\"byte_len\":{},\"consumer\":\"M.{id}\",\"dtype\":\"{dtype}\",\"encoding\":\"raw-le\",\"id\":\"{id}\",\"layout\":\"onnx-row-major\",\"offset\":{},\"onnx_name\":\"{id}\",\"sha256\":\"{}\",\"shape\":[{shape}]}}",
            data.len(),
            payload.len(),
            hex(&Sha256::digest(data))
        ));
        payload.extend_from_slice(data);
    }
    let zero = "0".repeat(64);
    let manifest = format!(
        "{{\"format\":{{\"major\":1,\"minor\":0,\"name\":\"hydronnx-weights\"}},\"module\":{{\"generated_source_sha256\":\"{zero}\",\"name\":\"M\"}},\"payload\":{{\"byte_len\":{},\"sha256\":\"{}\"}},\"producer\":{{\"chelis_min\":\"0.19.1\",\"hydronnx_version\":\"test\",\"onnx_opset\":17}},\"source\":{{\"graph_name\":\"g\",\"onnx_sha256\":\"{zero}\"}},\"tensors\":[{}]}}",
        payload.len(),
        hex(&Sha256::digest(&payload)),
        entries.join(",")
    );
    let manifest = edit(manifest);
    let mut archive = b"HNXWGT\x00\x01".to_vec();
    archive.extend_from_slice(&80u32.to_le_bytes());
    archive.extend_from_slice(&u32::try_from(manifest.len()).unwrap().to_le_bytes());
    archive.extend_from_slice(&Sha256::digest(manifest.as_bytes()));
    archive.extend_from_slice(&Sha256::digest(&payload));
    archive.extend_from_slice(manifest.as_bytes());
    while !archive.len().is_multiple_of(64) {
        archive.push(0);
    }
    archive.extend_from_slice(&payload);
    archive
}

fn le<T: Copy, const N: usize>(values: &[T], bytes: fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|value| bytes(*value)).collect()
}

fn model_archive() -> Vec<u8> {
    hnw(&[
        (
            "w",
            "f32",
            vec![2, 3],
            le(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.5], f32::to_le_bytes),
        ),
        ("b", "f64", vec![2], le(&[0.1f64, -0.25], f64::to_le_bytes)),
        (
            "ids",
            "i64",
            vec![2],
            le(&[-1i64, 1 << 40], i64::to_le_bytes),
        ),
        ("idx", "i32", vec![3], le(&[7i32, -8, 9], i32::to_le_bytes)),
        ("mask", "bool", vec![3], vec![1, 0, 1]),
    ])
}

const ARCHIVE_IMPORT: &str =
    "import Std.Io.Tensors (open_hnw, read_f32, read_f64, read_i64, read_i32, read_bool)\n";

fn archive_program(path: &Path, body: &str) -> String {
    format!(
        "{ARCHIVE_IMPORT}def main() -> unit ! {{ IO }} = {{\n  a = open_hnw({})\n{body}\n}}\nrun = main()\n",
        literal(path)
    )
}

/// [05-OP-82]: every hnw dtype reads back typed and shaped, identically in
/// both lanes.
#[test]
fn an_archive_reads_typed_tensors_in_eval_and_c() {
    let dir = tempdir().expect("tempdir");
    let data = dir.path().join("model.hnw");
    fs::write(&data, model_archive()).expect("write archive");
    let body = "  w: tensor[2, 3, f32] = reshape(read_f32(a, \"w\", [2i64, 3i64]), [2i64, 3i64])\n  _ = print(w)\n  _ = print(read_f64(a, \"b\", [2i64]))\n  _ = print(read_i64(a, \"ids\", [2i64]))\n  _ = print(read_i32(a, \"idx\", [3i64]))\n  print(read_bool(a, \"mask\", [3i64]))";
    assert_lanes_print(
        &dir,
        "archive",
        &archive_program(&data, body),
        "tensor(shape=[2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.5])\n\
         tensor(shape=[2], data=[0.1, -0.25])\n\
         tensor(shape=[2], data=[-1, 1099511627776])\n\
         tensor(shape=[3], data=[7, -8, 9])\n\
         tensor(shape=[3], data=[true, false, true])\n\
         run = ()\n",
    );
}

/// [05-OP-82]: a reader fails, before it reads an element, for a wrong
/// dtype, a wrong shape, an absent name, and a corrupted payload; opening
/// fails for a corrupted manifest, wrong magic, and a truncated file.
#[test]
fn archive_mismatches_fail_with_a_named_reason() {
    let dir = tempdir().expect("tempdir");
    let good = dir.path().join("model.hnw");
    let archive = model_archive();
    fs::write(&good, &archive).expect("write archive");
    let path_text = good.to_str().unwrap();
    let reader_cases = [
        (
            "wrong_dtype",
            "  print(read_f64(a, \"w\", [2i64, 3i64]))",
            "tensor w has dtype f32, read as f64",
        ),
        (
            "wrong_shape",
            "  print(read_f32(a, \"w\", [3i64, 2i64]))",
            "tensor w has shape [2, 3], read as [3, 2]",
        ),
        (
            "absent_name",
            "  print(read_f32(a, \"v\", [2i64, 3i64]))",
            "no tensor named v",
        ),
    ];
    for (name, body, detail) in reader_cases {
        let expected = format!("Std.Io.Tensors: {path_text}: {detail}");
        assert_lanes_fail(
            &dir,
            name,
            &archive_program(&good, body),
            &[expected.as_str()],
        );
    }
    let corrupt = |name: &str, bytes: Vec<u8>| {
        let path = dir.path().join(format!("{name}.hnw"));
        fs::write(&path, bytes).expect("write corrupted archive");
        path
    };
    let mut payload_flip = archive.clone();
    let last = payload_flip.len() - 1;
    payload_flip[last] ^= 1;
    let mut manifest_flip = archive.clone();
    manifest_flip[100] ^= 1;
    let mut magic = archive.clone();
    magic[0] = b'X';
    let opened = [
        (
            "payload_flip",
            corrupt("payload_flip", payload_flip),
            "  print(read_bool(a, \"mask\", [3i64]))",
            "tensor mask checksum mismatch",
        ),
        (
            "manifest_flip",
            corrupt("manifest_flip", manifest_flip),
            "  ()",
            "manifest checksum mismatch",
        ),
        (
            "bad_magic",
            corrupt("bad_magic", magic),
            "  ()",
            "bad magic",
        ),
        (
            "truncated",
            corrupt("truncated", archive[..40].to_vec()),
            "  ()",
            "truncated fixed header",
        ),
    ];
    for (name, path, body, detail) in opened {
        let expected = format!("Std.Io.Tensors: {}: {detail}", path.to_str().unwrap());
        assert_lanes_fail(
            &dir,
            name,
            &archive_program(&path, body),
            &[expected.as_str()],
        );
    }
}

/// [05-OP-82]: `open_hnw` refuses metadata it does not support or that cannot
/// describe a payload, and a reader refuses a byte length its shape
/// disagrees with, each with its named reason.
#[test]
fn archive_metadata_is_validated() {
    let dir = tempdir().expect("tempdir");
    let tensors: Vec<ArchiveTensor> =
        vec![("w", "f32", vec![2], le(&[1.0f32, 2.0], f32::to_le_bytes))];
    let cases: [(&str, &str, &str, &str); 7] = [
        (
            "\"layout\":\"onnx-row-major\"",
            "\"layout\":\"nchw\"",
            "  ()",
            "tensor w has unsupported layout nchw",
        ),
        (
            "\"encoding\":\"raw-le\"",
            "\"encoding\":\"zstd\"",
            "  ()",
            "tensor w has unsupported encoding zstd",
        ),
        (
            "\"name\":\"hydronnx-weights\"",
            "\"name\":\"other-weights\"",
            "  ()",
            "not a hydronnx-weights archive",
        ),
        (
            "\"major\":1",
            "\"major\":2",
            "  ()",
            "unsupported major version",
        ),
        (
            "\"shape\":[2]",
            "\"shape\":[-1,-2]",
            "  ()",
            "tensor w has a negative extent",
        ),
        (
            "\"offset\":0",
            "\"offset\":-64",
            "  ()",
            "tensor w has a negative offset",
        ),
        (
            "{\"byte_len\":8,\"consumer\"",
            "{\"byte_len\":4,\"consumer\"",
            "  print(read_f32(a, \"w\", [2i64]))",
            "tensor w has byte length 4, its shape needs 8",
        ),
    ];
    for (index, (from, to, body, detail)) in cases.into_iter().enumerate() {
        let path = dir.path().join(format!("edited{index}.hnw"));
        fs::write(
            &path,
            hnw_edited(&tensors, |manifest| {
                assert!(manifest.contains(from), "{from} not in {manifest}");
                manifest.replacen(from, to, 1)
            }),
        )
        .expect("write archive");
        let expected = format!("Std.Io.Tensors: {}: {detail}", path.to_str().unwrap());
        assert_lanes_fail(
            &dir,
            &format!("edited{index}"),
            &archive_program(&path, body),
            &[expected.as_str()],
        );
    }
}
