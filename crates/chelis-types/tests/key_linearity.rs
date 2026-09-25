//! chelis#2413 step 2: the checker's key affinity, [04-LIN-9] and spec/04
//! sections 1.1 and 8.4.1.
//!
//! Every negative is paired with a positive twin of the same shape that uses
//! its keys legitimately, usually by deriving fresh keys first. Each program
//! runs the type checker and then the linearity checker, the two phases
//! `chelis check` runs before effects are reported.
//!
//! The random draws [05-OP-8] and [05-OP-37] take their key first and
//! consume it, like the key operations.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_linearity, check_typed_program};

fn verdict(source: &str) -> Result<(), Vec<CheckError>> {
    let decls = parse_str(source).expect("fixture parses");
    let deep = desugar_program(&decls).expect("fixture desugars");
    let checked = check_typed_program(&deep).map_err(|result| result.errors)?;
    check_linearity(&checked).map(|_| ())
}

fn accepts(name: &str, source: &str) {
    if let Err(errors) = verdict(source) {
        panic!("{name}: expected acceptance, got {errors:?}");
    }
}

fn same_kind(lhs: &CheckErrorKind, rhs: &CheckErrorKind) -> bool {
    std::mem::discriminant(lhs) == std::mem::discriminant(rhs)
}

fn rejects(name: &str, source: &str, kind: CheckErrorKind) -> Vec<CheckError> {
    let errors = verdict(source).expect_err(&format!("{name}: expected a rejection"));
    assert!(
        errors.iter().any(|error| same_kind(&error.kind, &kind)),
        "{name}: expected {kind:?}, got {errors:?}"
    );
    errors
}

/// A `KeyReuse` diagnostic names the derivations, never `copy`.
fn rejects_reuse(name: &str, source: &str) {
    for error in rejects(name, source, CheckErrorKind::KeyReuse) {
        if same_kind(&error.kind, &CheckErrorKind::KeyReuse) {
            let suggestions = error.suggestions.join(" ");
            assert!(
                !suggestions.contains("copy("),
                "{name}: a key diagnostic must never suggest copy: {error:?}"
            );
        }
    }
}

#[test]
fn deriving_keys_is_accepted_and_each_key_is_used_once() {
    accepts(
        "split then use each half",
        "def two(k: key) -> (key, key) = {\n  (a, b) = split_key(k)\n  (a, b)\n}\n",
    );
    accepts(
        "a chain through every key operation",
        "def chain(seed: i64) -> tensor[4, key] = {\n  k = key_from_seed(seed)\n  (a, b) = \
         split_key(k)\n  c = fold_in(a, 3i64)\n  _ = b\n  split_keys(c, 4i64)\n}\n",
    );
    accepts(
        "an unused key is dropped",
        "def ignore(k: key) -> i64 = 0i64\n",
    );
}

#[test]
fn reusing_a_key_after_a_key_operation_is_rejected() {
    rejects_reuse(
        "split_key twice",
        "def bad(k: key) -> (key, key) = {\n  a = split_key(k)\n  b = split_key(k)\n  (a.0, \
         b.0)\n}\n",
    );
    rejects_reuse(
        "fold_in twice",
        "def bad(k: key) -> (key, key) = (fold_in(k, 1i64), fold_in(k, 2i64))\n",
    );
    rejects_reuse(
        "split_keys then fold_in",
        "def bad(k: key) -> (tensor[2, key], key) = (split_keys(k, 2i64), fold_in(k, 2i64))\n",
    );
    rejects_reuse(
        "a key tuple returned twice",
        "def bad(k: key, t: tensor[2, f32]) -> ((key, tensor[2, f32]), (key, tensor[2, f32])) \
         = {\n  p = (k, t)\n  (p, p)\n}\n",
    );
    rejects_reuse(
        "a sig-typed parameter used twice",
        "sig dup: key -> (key, key)\ndef dup(k) = (k, k)\n",
    );
}

#[test]
fn reusing_a_key_after_a_draw_is_rejected() {
    rejects_reuse(
        "dropout twice with one key",
        "def bad(k: key, x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = \
         (dropout(k, x, 0.5f32), dropout(k, x, 0.5f32))\n",
    );
    rejects_reuse(
        "uniform_like then split_key",
        "def bad(k: key, t: tensor[2, f32]) -> (tensor[2, f32], (key, key)) = {\n  u = \
         uniform_like(k, t, 0.0f32, 1.0f32)\n  (u, split_key(k))\n}\n",
    );
    rejects_reuse(
        "a draw after the key was split",
        "def bad(k: key, x: tensor[4, f32]) -> (tensor[4, f32], key) = {\n  (a, _) = \
         split_key(k)\n  (dropout(k, x, 0.5f32), a)\n}\n",
    );
    accepts(
        "each draw takes its own half",
        "def good(k: key, x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = {\n  (a, b) \
         = split_key(k)\n  (dropout(a, x, 0.5f32), uniform_like(b, x, 0.0f32, 1.0f32))\n}\n",
    );
    accepts(
        "one draw per selected arm",
        "def good(k: key, x: tensor[4, f32], c: bool) -> tensor[4, f32] = if c then dropout(k, x, \
         0.5f32) else uniform_like(k, x, 0.0f32, 1.0f32)\n",
    );
    // The tensor a draw reads is borrowed, so it stays live for the second
    // draw; only the key is consumed.
    accepts(
        "the data operand is borrowed",
        "def good(k: key, x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = {\n  (a, b) \
         = split_key(k)\n  (dropout(a, x, 0.5f32), dropout(b, x, 0.5f32))\n}\n",
    );
}

#[test]
fn a_key_is_never_copied_or_borrowed() {
    // A scalar `copy` or `&` is already a type error; the key tensor and the
    // tensor-carrying tuple reach linearity, which refuses them.
    verdict("def bad(k: key) -> (key, key) = (copy(k), k)\n").expect_err("copy of a key");
    verdict("def bad(k: key) -> key = {\n  _ = print(&k)\n  k\n}\n").expect_err("&k");
    rejects_reuse(
        "copy of a key tensor",
        "def bad(k: key) -> (tensor[2, key], tensor[2, key]) = {\n  ks = split_keys(k, \
         2i64)\n  (copy(ks), ks)\n}\n",
    );
    rejects_reuse(
        "&(k, t)",
        "def bad(k: key, t: tensor[2, f32]) -> (key, tensor[2, f32]) = {\n  p = (k, t)\n  _ = \
         print(&p)\n  p\n}\n",
    );
    rejects_reuse(
        "a builtin that borrows",
        "def bad(k: key) -> key = {\n  _ = print(k)\n  k\n}\n",
    );
    accepts(
        "the tensor half of the same tuple is still borrowable",
        "def good(k: key, t: tensor[2, f32]) -> (key, tensor[2, f32]) = {\n  _ = print(&t)\n  \
         (k, t)\n}\n",
    );
}

#[test]
fn binding_a_key_to_another_name_moves_it() {
    rejects_reuse(
        "y = k then both",
        "def bad(k: key) -> (key, key) = {\n  y = k\n  (fold_in(y, 1i64), fold_in(k, 1i64))\n}\n",
    );
    accepts(
        "y = k then only y",
        "def good(k: key) -> key = {\n  y = k\n  fold_in(y, 1i64)\n}\n",
    );
}

#[test]
fn a_tuple_projection_takes_each_key_component_once() {
    rejects_reuse(
        "p.0 twice",
        "def bad(k: key) -> (key, key) = {\n  p = split_key(k)\n  (p.0, p.0)\n}\n",
    );
    rejects_reuse(
        "the whole tuple after a projection",
        "def bad(k: key) -> ((key, key), key) = {\n  p = split_key(k)\n  a = p.0\n  (p, a)\n}\n",
    );
    accepts(
        "p.0 and p.1",
        "def good(k: key) -> (key, key) = {\n  p = split_key(k)\n  (p.0, p.1)\n}\n",
    );
}

#[test]
fn a_branch_consume_survives_the_join() {
    rejects_reuse(
        "if arm then after",
        "def bad(c: bool, k: key) -> key = {\n  x = if c then fold_in(k, 1i64) else \
         key_from_seed(0i64)\n  fold_in(k, 2i64)\n}\n",
    );
    accepts(
        "one use per arm",
        "def good(c: bool, k: key) -> key = if c then fold_in(k, 1i64) else fold_in(k, 2i64)\n",
    );
    rejects_reuse(
        "match arm then after",
        "def bad(o: Option[i64], k: key) -> key = {\n  x = match o with {\n    | Some(n) => \
         fold_in(k, n)\n    | None => key_from_seed(0i64)\n  }\n  fold_in(k, 7i64)\n}\n",
    );
    accepts(
        "one use per match arm",
        "def good(o: Option[i64], k: key) -> key = match o with {\n  | Some(n) => fold_in(k, \
         n)\n  | None => fold_in(k, 0i64)\n}\n",
    );
}

#[test]
fn a_closure_never_captures_a_key() {
    rejects_reuse(
        "a lambda capturing a key",
        "def bad(k: key) -> key = {\n  f = fn (n: i64) -> fold_in(k, n)\n  f(1i64)\n}\n",
    );
    rejects_reuse(
        "map over a captured key",
        "def bad(k: key, xs: List[i64]) -> List[key] = map(fn (n: i64) -> fold_in(k, n), xs)\n",
    );
    accepts(
        "the key passed as a parameter",
        "def good(k: key) -> key = {\n  f = fn (j: key, n: i64) -> fold_in(j, n)\n  f(k, \
         1i64)\n}\n",
    );
}

#[test]
fn a_signature_never_borrows_a_key() {
    rejects_reuse("&key", "def bad(k: &key) -> i64 = 0i64\n");
    rejects_reuse(
        "&(key, tensor)",
        "def bad(p: &(key, tensor[2, f32])) -> i64 = 0i64\n",
    );
    // A parameter that carries a key is never inferred read-only, so this
    // call consumes `ks` and the later use is a reuse, not a refused borrow.
    let errors = rejects(
        "an unused key tensor parameter",
        "def ignore(ks: tensor[3, key]) -> i64 = 0i64\ndef bad(k: key) -> (i64, tensor[3, key]) \
         = {\n  ks = split_keys(k, 3i64)\n  (ignore(ks), ks)\n}\n",
        CheckErrorKind::KeyReuse,
    );
    assert!(
        errors.iter().any(|error| error
            .message
            .contains("already consumed by call to `ignore`")),
        "the key parameter must consume at the call: {errors:?}"
    );
}

#[test]
fn transform_calls_consume_key_arguments() {
    let row = "def row(k: key, x: tensor[2, f32]) -> key = fold_in(k, 1i64)\n";
    rejects(
        "a scalar key to a vmapped key formal",
        &format!(
            "{row}def bad(k: key, xs: tensor[3, 2, f32]) -> tensor[3, key] = vmap(row)(k, xs)\n"
        ),
        CheckErrorKind::TypeMismatch,
    );
    accepts(
        "a key tensor to a vmapped key formal",
        &format!(
            "{row}def good(k: key, xs: tensor[3, 2, f32]) -> tensor[3, key] = \
             vmap(row)(split_keys(k, 3i64), xs)\n"
        ),
    );
    rejects_reuse(
        "vmap(f)(ks, xs) then ks",
        &format!(
            "{row}def bad(k: key, xs: tensor[3, 2, f32]) -> (tensor[3, key], tensor[3, key]) = \
             {{\n  ks = split_keys(k, 3i64)\n  (vmap(row)(ks, xs), vmap(row)(ks, xs))\n}}\n"
        ),
    );
    rejects_reuse(
        "a key captured by the vmapped function",
        "def bad(k: key, xs: tensor[3, 2, f32]) -> tensor[3, key] = vmap(fn (x: tensor[2, f32]) \
         -> fold_in(k, 1i64))(xs)\n",
    );
    let loss = "def loss(k: key, x: f32) -> f32 = mul(x, x)\n";
    rejects_reuse(
        "grad(f)(k, x) then k",
        &format!(
            "{loss}def bad(k: key, x: f32) -> (f32, key) = (grad(loss, wrt=x)(k, x), fold_in(k, \
             1i64))\n"
        ),
    );
    accepts(
        "grad(f)(k, x) once",
        &format!("{loss}def good(k: key, x: f32) -> f32 = grad(loss, wrt=x)(k, x)\n"),
    );
}

#[test]
fn a_key_container_is_never_read_in_place() {
    let pair = "  (a, b) = split_key(k)\n  ks = [a, b]\n";
    rejects_reuse(
        "index into a key list",
        &format!(
            "def bad(k: key) -> (key, key) = {{\n{pair}  (index(ks, 0i64), index(ks, 0i64))\n}}\n"
        ),
    );
    rejects_reuse(
        "len of a key list, then the list",
        &format!("def bad(k: key) -> List[key] = {{\n{pair}  _ = len(ks)\n  ks\n}}\n"),
    );
    accepts(
        "the key list moved whole",
        "def good(k: key) -> List[key] = {\n  (a, b) = split_key(k)\n  [a, b]\n}\n",
    );
    verdict("def bad(k: key) -> List[key] = to_list(split_keys(k, 2i64))\n")
        .expect_err("to_list of a key tensor");
}

#[test]
fn a_fold_carries_a_key_once_per_iteration() {
    rejects_reuse(
        "the accumulator used twice",
        "def bad(k: key, xs: List[i64]) -> key = fold(fn (acc: key, n: i64) -> {\n  _ = \
         fold_in(acc, n)\n  fold_in(acc, n)\n}, k, xs)\n",
    );
    accepts(
        "the accumulator used once",
        "def good(k: key, xs: List[i64]) -> key = fold(fn (acc: key, n: i64) -> fold_in(acc, n), \
         k, xs)\n",
    );
}

#[test]
fn a_field_access_consumes_the_whole_value() {
    let holder = "type Holder =\n  | Holder { k: key, x: tensor[2, f32] }\n";
    rejects_reuse(
        "h.k twice",
        &format!(
            "{holder}def bad(h: Holder) -> (key, key) = (fold_in(h.k, 1i64), fold_in(h.k, \
             2i64))\n"
        ),
    );
    accepts(
        "a destructuring match",
        &format!(
            "{holder}def good(h: Holder) -> (key, tensor[2, f32]) = match h with {{\n  | Holder \
             {{ k, x }} => (fold_in(k, 1i64), x)\n}}\n"
        ),
    );
}

#[test]
fn top_level_roots_observe_a_key_once() {
    rejects_reuse(
        "two roots over one key",
        "root = key_from_seed(1i64)\nfirst = fold_in(root, 1i64)\nsecond = fold_in(root, \
         2i64)\n",
    );
    accepts(
        "two roots over split halves",
        "root = key_from_seed(1i64)\npair = split_key(root)\nfirst = fold_in(pair.0, \
         1i64)\nsecond = fold_in(pair.1, 2i64)\n",
    );
}

#[test]
fn key_is_not_a_numeric_or_castable_dtype() {
    rejects(
        "cast to key",
        "def bad(x: i64) -> key = cast(x, key)\n",
        CheckErrorKind::UnsupportedTensorPrecision,
    );
    rejects(
        "cast from key",
        "def bad(k: key) -> i64 = cast(k, i64)\n",
        CheckErrorKind::UnsupportedTensorPrecision,
    );
    rejects(
        "cast from a key tensor",
        "def bad(k: key) -> tensor[2, i64] = cast(split_keys(k, 2i64), i64)\n",
        CheckErrorKind::UnsupportedTensorPrecision,
    );
    rejects(
        "add on keys",
        "def bad(k: key, j: key) -> key = add(k, j)\n",
        CheckErrorKind::TypeMismatch,
    );
    rejects(
        "eq on keys",
        "def bad(k: key, j: key) -> bool = eq(k, j)\n",
        CheckErrorKind::PrecisionMismatch,
    );
    rejects(
        "sum over a key tensor",
        "def bad(k: key) -> key = sum(split_keys(k, 3i64), 0i32)\n",
        CheckErrorKind::PrecisionMismatch,
    );
    rejects(
        "reshape of a key tensor",
        "def bad(k: key) -> tensor[2, 2, key] = reshape(split_keys(k, 4i64), [2i64, 2i64])\n",
        CheckErrorKind::PrecisionMismatch,
    );
    for family in ["Float", "Int", "Numeric"] {
        rejects(
            family,
            &format!("def ident[T: {family}](x: T) -> T = x\ndef bad(k: key) -> key = ident(k)\n"),
            CheckErrorKind::PrecisionMismatch,
        );
    }
    accepts(
        "an unbounded binder",
        "def ident[T](x: T) -> T = x\ndef good(k: key) -> key = ident(k)\n",
    );
    rejects(
        "integers are not keys",
        "def bad() -> tensor[2, key] = to_tensor([1i64, 2i64])\n",
        CheckErrorKind::TypeMismatch,
    );
    verdict("def bad() -> tensor[2, key] = {\n  x: tensor[2, key] = [1, 2]\n  x\n}\n")
        .expect_err("a contextual key literal");
}

/// chelis#2216 / PR #2517: a builtin's operand rule is skipped when the
/// operand is a rigid dtype binder, so a key tensor reaches `to_list`
/// through `listify`'s unbounded `p`. Un-ignore when #2517 merges.
#[test]
#[ignore = "chelis#2216: needs PR #2517's rigid-binder operand replay"]
fn a_key_tensor_cannot_reach_a_dtype_operation_through_a_rigid_binder() {
    verdict(
        "def listify[n, p](x: tensor[n, p]) -> List[p] = to_list(x)\ndef bad(k: key) -> \
         List[key] = listify(split_keys(k, 2i64))\n",
    )
    .expect_err("to_list of a key tensor through a binder");
}
