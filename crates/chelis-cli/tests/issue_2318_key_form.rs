//! chelis#2318 in key form, and the public-entry key carriers of spec/08
//! section 2 (chelis#2413 step 3, slice 4).
//!
//! #2318 showed an exported `def draw(c: f32) -> tensor[4, f32]` escaping an
//! unhandled `Random`: `chelis check` scored it 1.0, the tensor-DAG lane drew
//! against an inactive stream, and the C host lane aborted. With explicit
//! keys the keyless form is an arity error at `chelis check`, and the key
//! form builds in both lanes: the draw inside a tensor kernel (a foldable
//! template) and the draw in host code (a runtime template).
//!
//! A scalar key crosses an authored C export as the published
//! `chelis_key`, built by `chelis_key_from_seed`; a key tensor crosses as a
//! `chelis_tensor` of dtype key under [04-NUM-11]'s entry dtype check; keys
//! inside a tuple or a data type cross as rank-0 key tensors, as a tensor
//! does. The four-argument tensor ABI carries every input and output as a
//! `chelis_tensor`, so a scalar key there is a rank-0 key tensor.
//!
//! Every expected key word and draw is computed by
//! `briefs/switch-design-probes/key_ref.py` (the [05-RNG-2] transcription)
//! and `briefs/keys-b-slice2-probes/slice2_ref.py` ([05-OP-8]'s f32 fused
//! multiply-add), never by compiler code.

use std::path::Path;
use std::process::{Command as StdCommand, Output};

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{gcc_available, parse_tensor_data, write_file};

/// The draw inside a tensor kernel: the template folds, so the kernel reads
/// the key as its rank-0 key input.
const DAG_DRAW: &str = "def draw(k: key, c: f32) -> tensor[4, f32] = uniform_like(k, to_tensor([c, c, c, c]), 2.0f32, 5.0f32)\n";
/// The draw in host code: the template is a runtime value.
const HOST_DRAW: &str = "def bc(c: f32) -> tensor[4, f32] = to_tensor([c, c, c, c])\ndef draw(k: key, c: f32) -> tensor[4, f32] = uniform_like(k, bc(c), 2.0f32, 5.0f32)\n";
/// #2318's two programs, verbatim.
const KEYLESS_DAG: &str =
    "def draw(c: f32) -> tensor[4, f32] = uniform_like(to_tensor([c, c, c, c]), 2.0f32, 5.0f32)\n";
const KEYLESS_HOST: &str = "def bc(c: f32) -> tensor[4, f32] = to_tensor([c, c, c, c])\ndef draw(c: f32) -> tensor[4, f32] = uniform_like(bc(c), 2.0f32, 5.0f32)\n";

/// `uniform_like(key_from_seed(7), t, 2, 5)` over 4 f32 elements.
const U25_KEY7: [u32; 4] = [0x40184bf4, 0x40905eea, 0x4080a022, 0x40738594];
/// `uniform_like(key_from_seed(7), t, 0, 1)` over 4 f32 elements.
const U01_KEY7: [u32; 4] = [0x3e019516, 0x3f56526f, 0x3f2c55af, 0x3f1a0770];
/// `fold_in(key_from_seed(7), 3)`.
const FOLD_KEY7_3: u64 = 0x53c6f7e83810b049;
/// `split_keys(key_from_seed(7), 3)`.
const ROWS_KEY7: [u64; 3] = [0x25ea33e61c10576f, 0x707124fbecd5f054, 0x823936153a565205];
/// `split_key(key_from_seed(7))`.
const SPLIT_KEY7: [u64; 2] = [0xaa3896172f9a3213, 0x8fd06b2e7bad8630];

const DRAW_SYMBOL: &str = "chelis_fn_64726177";

fn chelis() -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

fn build(dir: &Path, stem: &str, source: &str) -> std::path::PathBuf {
    let path = dir.join(format!("{stem}.ch"));
    let out_dir = dir.join(format!("{stem}-out"));
    write_file(&path, source);
    chelis()
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(&out_dir)
        .assert()
        .success();
    out_dir
}

/// Compile the generated translation unit with `driver`, link the runtime,
/// and run the binary with `args`.
fn run_driver(out_dir: &Path, stem: &str, driver: &str, args: &[&str]) -> Output {
    write_file(&out_dir.join("driver.c"), driver);
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: common::generated_source_needs_blas(out_dir, &format!("{stem}.c")),
        },
    );
    let status = StdCommand::new(&toolchain.compiler)
        .current_dir(out_dir)
        .arg("-O2")
        .args(&toolchain.compile_flags)
        .args([format!("{stem}.c"), "driver.c".to_string()])
        .arg("libchelis_runtime.a")
        .args(&toolchain.link_flags)
        .args(["-o", "driver"])
        .status()
        .expect("host compiler should run");
    assert!(status.success(), "driver link failed for {stem}");
    StdCommand::new(out_dir.join("driver"))
        .args(args)
        .output()
        .expect("driver should run")
}

fn stdout_of(output: &Output) -> String {
    assert!(
        output.status.success(),
        "driver failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn hex_words(words: &[u64], width: usize) -> String {
    words
        .iter()
        .map(|word| format!("{word:0width$x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn f32_hex(words: &[u32]) -> String {
    hex_words(&words.iter().map(|w| u64::from(*w)).collect::<Vec<_>>(), 8)
}

/// A driver fragment that prints a key tensor's dtype, rank and words.
const KEY_TENSOR_PRINTER: &str = r#"
static void key_tensor(const char *label, const chelis_tensor *t) {
    chelis_read_view v = chelis_tensor_read_view(t);
    printf("%s dtype=%d rank=%d", label, (int)v.dtype, (int)chelis_tensor_rank(t));
    for (int64_t i = 0; i < v.count; i++) printf(" %016llx", (unsigned long long)((const uint64_t *)v.data)[i]);
    printf("\n");
}
"#;

#[test]
fn keyless_2318_programs_are_arity_errors_at_check() {
    let dir = tempdir().expect("tempdir");
    for (stem, source) in [("keyless_dag", KEYLESS_DAG), ("keyless_host", KEYLESS_HOST)] {
        let path = dir.path().join(format!("{stem}.ch"));
        write_file(&path, source);
        let output = chelis()
            .args(["check", path.to_str().unwrap()])
            .output()
            .expect("chelis check runs");
        let report: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("check prints its JSON report");
        assert!(
            report["score"].as_f64().expect("score") < 1.0,
            "{stem}: a keyless draw must not score 1.0: {report}"
        );
        let errors = report["errors"].as_array().expect("errors");
        assert!(
            errors.iter().any(|error| error["kind"] == "ArityMismatch"
                && error["message"]
                    .as_str()
                    .is_some_and(|message| message.contains(
                        "`uniform_like(t, low, high)` is the retired counter-stream spelling"
                    ))),
            "{stem}: expected uniform_like's retired-spelling error: {report}"
        );
        chelis()
            .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
            .arg(dir.path().join(format!("{stem}-out")))
            .assert()
            .failure();
    }
}

#[test]
fn key_form_2318_draws_match_eval_and_the_reference_in_both_lanes() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let driver = format!(
        "#include <stdio.h>\n#include \"chelis_runtime.h\"\n\
         chelis_tensor *{DRAW_SYMBOL}(chelis_key k, float c);\n\
         int main(void) {{\n\
             chelis_tensor *t = {DRAW_SYMBOL}(chelis_key_from_seed(7), 1.0f);\n\
             chelis_read_view v = chelis_tensor_read_view(t);\n\
             printf(\"dtype=%d\", (int)v.dtype);\n\
             for (int64_t i = 0; i < v.count; i++) printf(\" %08x\", ((const uint32_t *)v.data)[i]);\n\
             printf(\"\\n\");\n\
             return 0;\n\
         }}\n"
    );
    let dir = tempdir().expect("tempdir");
    for (stem, source, lane_marker) in [
        (
            "dag_draw",
            DAG_DRAW,
            "__tensor_0__private: input `k` at slot",
        ),
        ("host_draw", HOST_DRAW, "__uniform_dtype_"),
    ] {
        let out_dir = build(dir.path(), stem, source);
        let header = std::fs::read_to_string(out_dir.join(format!("{stem}.h"))).unwrap();
        assert!(
            header.contains(&format!(
                "chelis_tensor* {DRAW_SYMBOL}(chelis_key k, float c);"
            )),
            "{stem}: the export takes the published key carrier:\n{header}"
        );
        let emitted = std::fs::read_to_string(out_dir.join(format!("{stem}.c"))).unwrap();
        assert!(
            emitted.contains(lane_marker),
            "{stem}: the draw is not in the lane this case exercises"
        );
        let printed = stdout_of(&run_driver(&out_dir, stem, &driver, &[]));
        assert_eq!(
            printed.trim(),
            format!("dtype=0 {}", f32_hex(&U25_KEY7)),
            "{stem}: C draw against the reference"
        );

        let eval_path = dir.path().join(format!("{stem}_eval.ch"));
        write_file(
            &eval_path,
            &format!("{source}sampled = draw(key_from_seed(7i64), 1.0f32)\n"),
        );
        let output = chelis()
            .args(["eval", "--file", eval_path.to_str().unwrap()])
            .output()
            .expect("chelis eval runs");
        assert!(output.status.success(), "{stem}: eval failed");
        let eval_bits: Vec<u32> =
            parse_tensor_data(&String::from_utf8_lossy(&output.stdout), "sampled")
                .into_iter()
                .map(|value| (value as f32).to_bits())
                .collect();
        assert_eq!(eval_bits, U25_KEY7, "{stem}: eval against the reference");
    }
}

const HOST_KEYS: &str = "type Boxed = | Boxed { k: key }\n\
def mk(s: i64) -> key = key_from_seed(s)\n\
def child(k: key, n: i64) -> key = fold_in(k, n)\n\
def rows(k: key) -> tensor[3, key] = split_keys(k, 3i64)\n\
def ktid(ks: tensor[3, key]) -> tensor[3, key] = ks\n\
def pair(k: key) -> (key, key) = split_key(k)\n\
def boxed(k: key) -> Boxed = Boxed { k }\n";

#[test]
fn authored_exports_take_and_return_chelis_key_and_key_tensors() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let out_dir = build(dir.path(), "host_keys", HOST_KEYS);
    let header = std::fs::read_to_string(out_dir.join("host_keys.h")).unwrap();
    for declaration in [
        "chelis_key chelis_fn_6d6b(int64_t s);",
        "chelis_key chelis_fn_6368696c64(chelis_key k, int64_t n);",
        "chelis_tensor* chelis_fn_726f7773(chelis_key k);",
        "chelis_tensor* chelis_fn_6b746964(chelis_tensor* ks);",
        "chelis_tuple* chelis_fn_70616972(chelis_key k);",
        "chelis_adt* chelis_fn_626f786564(chelis_key k);",
    ] {
        assert!(
            header.contains(declaration),
            "missing `{declaration}`:\n{header}"
        );
    }
    let driver = format!(
        "#include <stdio.h>\n#include <string.h>\n#include \"chelis_runtime.h\"\n#include \"host_keys.h\"\n\
         {KEY_TENSOR_PRINTER}\n\
         int main(int argc, char **argv) {{\n\
             if (argc > 1) {{\n\
                 int64_t shape[1] = {{3}};\n\
                 chelis_fn_6b746964(chelis_alloc(1, shape, CHELIS_DTYPE_I64));\n\
                 printf(\"unreachable\\n\");\n\
                 return 0;\n\
             }}\n\
             printf(\"mk7 %016llx\\n\", (unsigned long long)chelis_fn_6d6b(7).bits);\n\
             printf(\"mkm1 %016llx\\n\", (unsigned long long)chelis_fn_6d6b(-1).bits);\n\
             printf(\"child %016llx\\n\", (unsigned long long)chelis_fn_6368696c64(chelis_key_from_seed(7), 3).bits);\n\
             chelis_tensor *r = chelis_fn_726f7773(chelis_key_from_seed(7));\n\
             key_tensor(\"rows\", r);\n\
             key_tensor(\"ktid\", chelis_fn_6b746964(r));\n\
             chelis_tuple *p = chelis_fn_70616972(chelis_key_from_seed(7));\n\
             for (int64_t i = 0; i < chelis_tuple_len(p); i++) {{\n\
                 chelis_value v = chelis_tuple_get(p, i);\n\
                 printf(\"pair tag=%d\", (int)v.tag);\n\
                 key_tensor(\"\", (const chelis_tensor *)v.payload.handle);\n\
             }}\n\
             chelis_value f = chelis_adt_get_field(chelis_fn_626f786564(chelis_key_from_seed(7)), 0);\n\
             printf(\"boxed tag=%d\", (int)f.tag);\n\
             key_tensor(\"\", (const chelis_tensor *)f.payload.handle);\n\
             return 0;\n\
         }}\n"
    );
    let printed = stdout_of(&run_driver(&out_dir, "host_keys", &driver, &[]));
    let expected = [
        "mk7 0000000000000007".to_string(),
        "mkm1 ffffffffffffffff".to_string(),
        format!("child {FOLD_KEY7_3:016x}"),
        format!("rows dtype=9 rank=1 {}", hex_words(&ROWS_KEY7, 16)),
        format!("ktid dtype=9 rank=1 {}", hex_words(&ROWS_KEY7, 16)),
        format!("pair tag=3 dtype=9 rank=0 {:016x}", SPLIT_KEY7[0]),
        format!("pair tag=3 dtype=9 rank=0 {:016x}", SPLIT_KEY7[1]),
        "boxed tag=3 dtype=9 rank=0 0000000000000007".to_string(),
    ];
    assert_eq!(printed.lines().collect::<Vec<_>>(), expected);

    // [04-NUM-11]: a key-tensor parameter handed an i64 tensor traps at the
    // entry, before its data is read.
    let wrong = run_driver(&out_dir, "host_keys", &driver, &["wrong"]);
    assert!(
        !wrong.status.success(),
        "a wrong-dtype key tensor must trap"
    );
    let stderr = String::from_utf8_lossy(&wrong.stderr);
    assert!(
        stderr.contains("input `ks` expected dtype key, got i64")
            && stderr.contains("numeric trap: domain in load at key"),
        "{stderr}"
    );
    assert!(!String::from_utf8_lossy(&wrong.stdout).contains("unreachable"));
}

#[test]
fn four_argument_entries_carry_a_scalar_key_as_a_rank0_key_tensor() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let draw_dir = build(
        dir.path(),
        "ut",
        "def ut(k: key, x: tensor[4, f32]) -> tensor[4, f32] = uniform_like(k, x, 0.0f32, 1.0f32)\n",
    );
    let header = std::fs::read_to_string(draw_dir.join("ut.h")).unwrap();
    assert!(
        header.contains(
            "void ut(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"
        ),
        "{header}"
    );
    let draw_driver = r#"
#include <stdio.h>
#include "chelis_runtime.h"
void ut(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(int argc, char **argv) {
    int64_t shape[1] = {4};
    chelis_tensor *x = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    chelis_tensor *k = chelis_alloc(0, NULL, argc > 1 ? CHELIS_DTYPE_F32 : CHELIS_DTYPE_KEY);
    if (argc == 1) {
        chelis_tensor_write *w = chelis_tensor_begin_write(k);
        ((uint64_t *)chelis_tensor_write_view(w).data)[0] = chelis_key_from_seed(7).bits;
        chelis_tensor_end_write(w);
    }
    chelis_tensor *in[2] = {k, x};
    chelis_tensor *out[1] = {NULL};
    ut(in, 2, out, 1);
    chelis_read_view v = chelis_tensor_read_view(out[0]);
    printf("dtype=%d", (int)v.dtype);
    for (int64_t i = 0; i < v.count; i++) printf(" %08x", ((const uint32_t *)v.data)[i]);
    printf("\n");
    return 0;
}
"#;
    let printed = stdout_of(&run_driver(&draw_dir, "ut", draw_driver, &[]));
    assert_eq!(printed.trim(), format!("dtype=0 {}", f32_hex(&U01_KEY7)));
    let wrong = run_driver(&draw_dir, "ut", draw_driver, &["wrong"]);
    assert!(
        !wrong.status.success(),
        "an f32 tensor for a key input must trap"
    );
    assert!(
        String::from_utf8_lossy(&wrong.stderr)
            .contains("ut: input `k` expected dtype key, got f32"),
        "{}",
        String::from_utf8_lossy(&wrong.stderr)
    );

    let seed_dir = build(
        dir.path(),
        "mk",
        "def mk(s: i64) -> key = key_from_seed(s)\n",
    );
    let seed_driver = r#"
#include <stdio.h>
#include "chelis_runtime.h"
void mk(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {
    chelis_tensor *in[1] = {chelis_scalar_tensor(chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(int64_t)-3))};
    chelis_tensor *out[1] = {NULL};
    mk(in, 1, out, 1);
    chelis_read_view v = chelis_tensor_read_view(out[0]);
    printf("dtype=%d rank=%d %016llx\n", (int)v.dtype, (int)chelis_tensor_rank(out[0]),
           (unsigned long long)((const uint64_t *)v.data)[0]);
    return (int)(((const uint64_t *)v.data)[0] != chelis_key_from_seed(-3).bits);
}
"#;
    let printed = stdout_of(&run_driver(&seed_dir, "mk", seed_driver, &[]));
    assert_eq!(printed.trim(), "dtype=9 rank=0 fffffffffffffffd");
}

/// A parameter is its declaration and its name (chelis#2413 B2): two
/// declarations that each name their key parameter `k` take two keys. (b) A
/// module exporting two such defs builds, and each export draws with the key
/// its caller passes. (c) A `main` calling two such helpers checks, and eval
/// and compiled C both draw each helper's own key.
///
/// Evidentiary status: DISPOSITION LOCK. Both shapes already ran at
/// ff8957386, where each export is its own host function and the helpers are
/// inlined into `main`; the whole-program graph they share is the
/// regression the `lower` and wire tests pin.
#[test]
fn two_declarations_key_parameters_of_one_name_are_two_keys() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let exported = build(
        dir.path(),
        "exported",
        "module Probe.Keys\nexport (draw, unit)\ndef draw(k: key, c: f32) -> tensor[4, f32] = uniform_like(k, to_tensor([c, c, c, c]), 2.0f32, 5.0f32)\ndef unit(k: key, c: f32) -> tensor[4, f32] = uniform_like(k, to_tensor([c, c, c, c]), 0.0f32, 1.0f32)\n",
    );
    let header = std::fs::read_to_string(exported.join("exported.h")).unwrap();
    for declaration in [
        format!("chelis_tensor* {DRAW_SYMBOL}(chelis_key k, float c);"),
        "chelis_tensor* chelis_fn_756e6974(chelis_key k, float c);".to_string(),
    ] {
        assert!(header.contains(&declaration), "{declaration}:\n{header}");
    }
    let driver = format!(
        "#include <stdio.h>\n#include \"chelis_runtime.h\"\n#include \"exported.h\"\n\
         static void print(const chelis_tensor *t) {{\n\
             chelis_read_view v = chelis_tensor_read_view(t);\n\
             for (int64_t i = 0; i < v.count; i++) printf(\" %08x\", ((const uint32_t *)v.data)[i]);\n\
             printf(\"\\n\");\n\
         }}\n\
         int main(void) {{\n\
             print({DRAW_SYMBOL}(chelis_key_from_seed(7), 1.0f));\n\
             print(chelis_fn_756e6974(chelis_key_from_seed(7), 1.0f));\n\
             return 0;\n\
         }}\n"
    );
    let printed = stdout_of(&run_driver(&exported, "exported", &driver, &[]));
    assert_eq!(
        printed.lines().map(str::trim).collect::<Vec<_>>(),
        [f32_hex(&U25_KEY7), f32_hex(&U01_KEY7)]
    );

    let source = "def h1(k: key, x: tensor[4, f32]) -> tensor[4, f32] = dropout(k, x, 0.5f32)\ndef h2(k: key, x: tensor[4, f32]) -> tensor[4, f32] = dropout(k, x, 0.25f32)\ndef main() -> tensor[4, f32] = {\n  (k1, k2) = split_key(key_from_seed(42i64))\n  x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n  add(h1(k1, copy(x)), h2(k2, x))\n}\n";
    let (k1, k2) = common::key_ref::split(common::key_ref::key_from_seed(42));
    let expected = common::key_ref::dropout_f32(k1, &[1.0; 4], 0.5)
        .into_iter()
        .zip(common::key_ref::dropout_f32(k2, &[1.0; 4], 0.25))
        .map(|(left, right)| (left + right).to_bits())
        .collect::<Vec<_>>();
    // A printed f32 reads back as the nearest f64; its f32 bits are exact.
    let bits = |stdout: &[u8]| {
        parse_tensor_data(&String::from_utf8_lossy(stdout), "main")
            .into_iter()
            .map(|value| (value as f32).to_bits())
            .collect::<Vec<_>>()
    };
    let path = dir.path().join("helpers.ch");
    write_file(&path, source);
    let checked = chelis()
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check runs");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["errors"], serde_json::json!([]), "{report}");
    let evaluated = chelis()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval runs");
    assert!(evaluated.status.success(), "eval failed");
    assert_eq!(bits(&evaluated.stdout), expected);
    let out_dir = build(dir.path(), "helpers", source);
    assert!(common::link_generated(&out_dir, "helpers.c", "helpers").success());
    let run = StdCommand::new(out_dir.join("helpers"))
        .output()
        .expect("compiled program runs");
    assert!(run.status.success());
    assert_eq!(bits(&run.stdout), expected);
}
