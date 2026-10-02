//! A local tensor ascription in an untaken `if` arm checks nothing, through
//! `chelis eval --file` (spec/10 section 3.2 with spec/04 section 4.7; #2413,
//! #2440): in a `grad` or `vmap` body spliced into the arm, over a value
//! produced before the arm, and in a `vmap`ped arm, where each row has its
//! own activation and the claim, about an extent every row shares, checks
//! when any row takes the arm. A guarded abort in a `grad` body under a
//! `vmap`ped arm fires only in the rows that take it. `eval --file` runs the
//! whole file; the kernel holding each `if` computes both arms. The
//! evaluator and C rows of the same shapes are in `chelis-compiler-api`'s
//! `local_ascription_activation`; the grad-body row's whole program also
//! builds and runs in C here. A shift's negative count in an untaken arm
//! checks nothing either, consumed or discarded, and traps when taken.
use assert_cmd::Command;
use std::path::Path;

const GRAD_BODY: &str = "def h(x: tensor[*, f32]) -> tensor[f32] = {
  y: tensor[3, f32] = pad(x, [[0i64, 0i64]], 0.0f32)
  sum(y, 0i32)
}
def f(x: tensor[*, f32]) -> tensor[*, f32] = {
  s = tensor_to_scalar(sum(copy(x), 0i32))
  if gt(s, 100.0f32) then grad(h)(x) else x
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";

const VMAP_BODY: &str = "def r(x: tensor[*, f32]) -> tensor[f32] = {
  y: tensor[3, f32] = pad(x, [[0i64, 0i64]], 0.0f32)
  sum(y, 0i32)
}
def f(xs: tensor[2, *, f32]) -> tensor[2, f32] = {
  s = tensor_to_scalar(sum(sum(copy(xs), 0i32), 0i32))
  if gt(s, 100.0f32) then vmap(r)(xs) else sum(xs, 1i32)
}
out = f(to_tensor([[1.0f32, 1.0f32, 1.0f32, 1.0f32], [1.0f32, 1.0f32, 1.0f32, 1.0f32]]))
";

const PRODUCED_BEFORE: &str = "def f(x: tensor[*, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(copy(x), 0i32))
  z = pad(&x, [[0i64, 0i64]], 0.0f32)
  if gt(s, 100.0f32) then {
    y: tensor[3, f32] = z
    sum(y, 0i32)
  } else sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";

const MAPPED_ARM: &str = "def r(x: tensor[4, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(copy(x), 0i32))
  if gt(s, 100.0f32) then {
    p = sub(shape(&x, 0i32), 4i64)
    y: tensor[3, f32] = pad(&x, [[0i64, p]], 0.0f32)
    sum(y, 0i32)
  } else sum(x, 0i32)
}
out = vmap(r)(to_tensor([[-3.0f32, -1.0f32, 1.0f32, 3.0f32], [1.0f32, 1.0f32, 1.0f32, 1.0f32]]))
";

const MAPPED_GRAD_BODY: &str = "def h(x: tensor[4, f32]) -> tensor[f32] = {
  p = sub(shape(&x, 0i32), 4i64)
  y: tensor[3, f32] = pad(&x, [[0i64, p]], 0.0f32)
  sum(y, 0i32)
}
def r(x: tensor[4, f32]) -> tensor[4, f32] = {
  s = tensor_to_scalar(sum(copy(x), 0i32))
  if gt(s, 100.0f32) then grad(h)(x) else x
}
out = vmap(r)(to_tensor([[-3.0f32, -1.0f32, 1.0f32, 3.0f32], [1.0f32, 1.0f32, 1.0f32, 1.0f32]]))
";

const MAPPED_ABORT: &str = "def h(x: tensor[f32]) -> tensor[f32] = {
  s = tensor_to_scalar(copy(x))
  if lt(s, 0.0f32) then fail(\"negative row\") else mul(&x, &x)
}
def r(x: tensor[f32]) -> tensor[f32] = {
  s = tensor_to_scalar(copy(x))
  if gt(s, 100.0f32) then grad(h)(x) else x
}
out = vmap(r)(to_tensor([-3.0f32, 1.0f32]))
";

/// `MAPPED_ABORT` with row 1 alone taking the arm: row 0 would abort, but
/// its arm is not taken; `grad(h)` at 1 is 2.
fn row_1(source: &str) -> String {
    source.replace("gt(s, 100.0f32)", "gt(s, -2.0f32)")
}

const PAD_CLAIM: &str = "numeric trap: domain in pad at i64";

/// Every shape, what it returns untaken, and the trap its arm raises taken.
const SHAPES: [(&str, &str, &str, &str); 6] = [
    (
        "grad body",
        GRAD_BODY,
        "out = tensor(shape=[4], data=[1.0, 1.0, 1.0, 1.0])",
        PAD_CLAIM,
    ),
    (
        "vmap body",
        VMAP_BODY,
        "out = tensor(shape=[2], data=[4.0, 4.0])",
        PAD_CLAIM,
    ),
    (
        "produced before the arm",
        PRODUCED_BEFORE,
        "out = 4.0",
        PAD_CLAIM,
    ),
    (
        "vmapped arm",
        MAPPED_ARM,
        "out = tensor(shape=[2], data=[0.0, 4.0])",
        PAD_CLAIM,
    ),
    (
        "grad body in a vmapped arm",
        MAPPED_GRAD_BODY,
        "out = tensor(shape=[2, 4], data=[-3.0, -1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0])",
        PAD_CLAIM,
    ),
    (
        "guarded abort in a grad body in a vmapped arm",
        MAPPED_ABORT,
        "out = tensor(shape=[2], data=[-3.0, 1.0])",
        "negative row",
    ),
];

/// The shape with its arm taken (in every row).
fn taken(source: &str) -> String {
    source.replace("gt(s, 100.0f32)", "gt(s, -5.0f32)")
}

fn chelis(directory: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(directory)
        .args(args)
        .output()
        .unwrap()
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Write `source`, check that it is canonical and lint-clean (so the style
/// gate cannot be what decides the row), and evaluate the whole file.
fn eval_file(directory: &Path, stem: &str, source: &str) -> std::process::Output {
    let path = format!("{stem}.ch");
    std::fs::write(directory.join(&path), source).unwrap();
    let formatted = chelis(directory, &["fmt", "--inplace", &path]);
    assert!(formatted.status.success(), "{}", text(&formatted));
    assert_eq!(
        std::fs::read_to_string(directory.join(&path)).unwrap(),
        source,
        "{stem} must already be canonical Surf"
    );
    let linted = chelis(directory, &["lint", "--check", &path]);
    assert!(linted.status.success(), "{}", text(&linted));
    chelis(directory, &["eval", "--file", &path])
}

/// Evidentiary status: REGRESSION TEST. At 224414e1f every row fails: the
/// ascription rows trap on the claim, except the produced-before row, which
/// fails with "local extent guard activation at node N is not available",
/// and the abort rows abort in row 0.
#[test]
fn an_untaken_arms_local_ascription_checks_nothing_in_eval_file() {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    let mut rows = SHAPES
        .iter()
        .map(|(shape, source, value, _)| (shape.to_string(), source.to_string(), *value))
        .collect::<Vec<_>>();
    rows.push((
        "guarded abort, row 1 alone takes the arm".into(),
        row_1(MAPPED_ABORT),
        "out = tensor(shape=[2], data=[-3.0, 2.0])",
    ));
    for (index, (shape, source, value)) in rows.into_iter().enumerate() {
        let output = eval_file(directory.path(), &format!("untaken_{index}"), &source);
        if !output.status.success() || text(&output).trim() != value {
            failures.push(format!("{shape}: {}", text(&output).trim()));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The positive control: taken, each arm's claim or abort traps, so the
/// untaken rows cannot pass by dropping the check.
///
/// Evidentiary status: DISPOSITION LOCK (green at 224414e1f), except the
/// produced-before row, a REGRESSION TEST: at 224414e1f it failed with the
/// unavailable-activation error rather than the claim.
#[test]
fn a_taken_arms_local_ascription_traps_in_eval_file() {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (shape, source, _, trap)) in SHAPES.into_iter().enumerate() {
        let output = eval_file(directory.path(), &format!("taken_{index}"), &taken(source));
        if output.status.success() || !text(&output).contains(trap) {
            failures.push(format!("{shape}: {}", text(&output).trim()));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The whole-program C lane: `chelis build --target c` and its published
/// executable.
/// Its stdout on success, or its build or run output on failure.
fn c_file(directory: &Path, stem: &str, source: &str) -> Result<String, String> {
    let path = directory.join(format!("{stem}.ch"));
    std::fs::write(&path, source).unwrap();
    let out_dir = directory.join(format!("{stem}-out"));
    let built = chelis(
        directory,
        &[
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ],
    );
    if !built.status.success() {
        return Err(format!("build: {}", text(&built)));
    }
    let run = std::process::Command::new(out_dir.join(stem))
        .output()
        .unwrap();
    if run.status.success() {
        Ok(String::from_utf8_lossy(&run.stdout).into_owned())
    } else {
        Err(text(&run))
    }
}

/// The grad-body row in the whole program's C: both arms of `f`'s join are
/// sized from `x`, which C names as one extent, so it builds, and returns
/// `x` untaken, as `eval --file` does; taken, it traps on the claim.
///
/// Evidentiary status: REGRESSION TEST for both rows: at 592dc55ce
/// `chelis build` refused each ("where at node N condition and branches
/// must have exactly matching shape"), the join condition sized from the
/// larger arm under a fresh identity; the untaken row built and returned `x`
/// at 096daea8c.
#[test]
fn the_grad_body_row_builds_and_agrees_with_eval_file_in_c() {
    let directory = tempfile::tempdir().unwrap();
    let (_, source, value, trap) = SHAPES[0];
    let mut failures = Vec::new();
    let eval = eval_file(directory.path(), "grad_body_eval", source);
    match c_file(directory.path(), "grad_body_c", source) {
        Ok(c) if c.trim() == value && text(&eval).trim() == value => {}
        c => failures.push(format!("untaken: eval {:?}, C {c:?}", text(&eval))),
    }
    match c_file(directory.path(), "grad_body_taken_c", &taken(source)) {
        Err(c) if c.contains(trap) => {}
        c => failures.push(format!("taken: C {c:?}")),
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// `shl` by the count `n = -1` in a runtime `if` arm, consumed.
const CONSUMED_SHIFT: &str = "def f(n: i32) -> i32 = if gt(n, 5i32) then shl(1i32, n) else n
out = f(-1i32)
";

/// The same shift, discarded.
const DISCARDED_SHIFT: &str = "def f(n: i32) -> i32 =
  if gt(n, 5i32) then {
    dead = shl(1i32, n)
    n
  } else n
out = f(-1i32)
";

/// Item 3: a shift is a scalar host builtin with no RISC node, so the host
/// interpreter runs it, only in the arm it takes; an untaken arm's negative
/// count checks nothing and the taken twin traps ([04-NUM-13]).
///
/// Evidentiary status: DISPOSITION LOCK (green at 224414e1f).
#[test]
fn a_shift_in_an_untaken_arm_checks_nothing_and_its_taken_twin_traps_in_eval_file() {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (shape, source)) in [("consumed", CONSUMED_SHIFT), ("discarded", DISCARDED_SHIFT)]
        .into_iter()
        .enumerate()
    {
        let output = eval_file(directory.path(), &format!("shift_untaken_{index}"), source);
        if !output.status.success() || text(&output).trim() != "out = -1" {
            failures.push(format!("untaken {shape}: {}", text(&output).trim()));
        }
        let taken = source.replace("gt(n, 5i32)", "lt(n, 5i32)");
        let output = eval_file(directory.path(), &format!("shift_taken_{index}"), &taken);
        if output.status.success()
            || !text(&output).contains("shift amount must be non-negative, got -1")
        {
            failures.push(format!("taken {shape}: {}", text(&output).trim()));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
