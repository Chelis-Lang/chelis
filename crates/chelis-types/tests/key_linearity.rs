//! chelis#2413 step 2: the checker's key affinity, [04-LIN-9], [04-LIN-10]
//! and spec/04 sections 1.1 and 8.4.1.
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
use chelis_types::{
    TypeEnv, build_type_env_from_library, check_ir_with_context, check_linearity,
    check_typed_program,
};

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

/// [04-LIN-10]: the type checker refuses a function's type parameter
/// instantiated at a key-carrying type, naming the generic and, when the
/// program spelled one, the parameter, and suggesting a concrete key
/// parameter, never `copy`.
fn rejects_key_instantiation(name: &str, source: &str, generic: &str, parameter: Option<&str>) {
    let errors = rejects(name, source, CheckErrorKind::KeyReuse);
    let error = errors
        .iter()
        .find(|error| same_kind(&error.kind, &CheckErrorKind::KeyReuse))
        .expect("rejects found the kind");
    assert!(
        error.message.contains(&format!("of generic `{generic}`"))
            && error.message.contains("[04-LIN-10]"),
        "{name}: the diagnostic must name generic `{generic}`: {error:?}"
    );
    let named_parameter = match parameter {
        Some(parameter) => format!("type parameter `{parameter}` of"),
        None => "an inferred type parameter of".to_string(),
    };
    assert!(
        error.message.contains(&named_parameter),
        "{name}: the diagnostic must name {named_parameter:?}: {error:?}"
    );
    let suggestions = error.suggestions.join(" ");
    assert!(
        suggestions.contains("concrete `key` or `tensor[n, key]` parameter")
            && !suggestions.contains("copy("),
        "{name}: the repair is a concrete key parameter, never copy: {error:?}"
    );
}

/// [04-LIN-10] for a generic that is a value binding, such as a generalized
/// `let` binding: the diagnostic names the binding, and its suggested repair
/// ascribes the binding a type or writes the value where it is used, never a
/// concrete key parameter (the binding has none) and never `copy`.
fn rejects_value_binding_instantiation(name: &str, source: &str, binding: &str) {
    let errors = rejects(name, source, CheckErrorKind::KeyReuse);
    let error = errors
        .iter()
        .find(|error| same_kind(&error.kind, &CheckErrorKind::KeyReuse))
        .expect("rejects found the kind");
    assert!(
        error.message.contains(&format!(
            "an inferred type parameter of generic `{binding}`"
        )) && error.message.contains("[04-LIN-10]"),
        "{name}: the diagnostic must name binding `{binding}`: {error:?}"
    );
    let suggestions = error.suggestions.join(" ");
    assert!(
        suggestions.contains(&format!("Ascribe `{binding}`"))
            && suggestions.contains(&format!("`{binding}: List[key] = Nil`"))
            && suggestions.contains("write its value where it is used")
            && !suggestions.contains("concrete `key` or `tensor[n, key]` parameter")
            && !suggestions.contains("copy("),
        "{name}: a value binding's repair is an ascription or the value in place: {error:?}"
    );
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

/// chelis#2544: a closure parameter the checker resolves to a key is a key
/// holder whether or not the program annotates it; linearity reads the
/// checked parameter type, so a second use is refused either way.
///
/// Evidentiary status: REGRESSION TEST. At `6f42b4e92` both unannotated
/// negatives were accepted: linearity declared an unannotated parameter
/// untyped.
#[test]
fn an_inferred_key_closure_parameter_is_used_once() {
    rejects_reuse(
        "map with an inferred key parameter used twice",
        "def bad(ks: List[key]) -> List[(key, key)] = map(fn (y) -> (y, y), ks)\n",
    );
    rejects_reuse(
        "an applied lambda with an inferred key parameter used twice",
        "def bad(k: key) -> (key, key) = (fn (y) -> (y, y))(k)\n",
    );
    rejects_reuse(
        "the annotated twin used twice",
        "def bad(k: key) -> (key, key) = (fn (y: key) -> (y, y))(k)\n",
    );
    accepts(
        "map with an inferred key parameter used once",
        "def good(ks: List[key]) -> List[(key, key)] = map(fn (y) -> split_key(y), ks)\n",
    );
    accepts(
        "an applied lambda with an inferred key parameter used once",
        "def good(k: key) -> (key, key) = (fn (y) -> split_key(y))(k)\n",
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

/// [04-LIN-6] and [04-LIN-9]: every top-level value binding is a root, and
/// observing a root consumes it, so a top-level key-carrying binding that
/// another binding also consumes is used twice. The `fold_in`, `dropout`,
/// `split_key` and projection forms agree; a key-carrying root that is only
/// observed checks.
///
/// Evidentiary status: REGRESSION TEST. At `b47fdd7d3` every single-use
/// negative checked, and so did "two roots over split halves", which this
/// test used to lock as accepted.
#[test]
fn top_level_roots_observe_a_key_once() {
    let x0 = "x0 = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])\n";
    rejects_reuse(
        "two roots over one key",
        "root = key_from_seed(1i64)\nfirst = fold_in(root, 1i64)\nsecond = fold_in(root, \
         2i64)\n",
    );
    for (name, source) in [
        (
            "a root that fold_in consumes",
            "root = key_from_seed(1i64)\nfirst = fold_in(root, 1i64)\n".to_string(),
        ),
        (
            "a root that dropout consumes",
            format!("root = key_from_seed(1i64)\n{x0}d = dropout(root, x0, 0.5f32)\n"),
        ),
        (
            "a root that split_key consumes",
            "root = key_from_seed(1i64)\npair = split_key(root)\n".to_string(),
        ),
        (
            "two roots over split halves",
            "pair = split_key(key_from_seed(1i64))\nfirst = fold_in(pair.0, 1i64)\nsecond = \
             fold_in(pair.1, 2i64)\n"
                .to_string(),
        ),
    ] {
        let errors = rejects(name, &source, CheckErrorKind::KeyReuse);
        assert!(
            errors
                .iter()
                .any(|error| error.message.contains("is a root")
                    && error.message.contains("[04-LIN-6]")),
            "{name}: expected the root-observation diagnostic: {errors:?}"
        );
    }
    for (name, source) in [
        (
            "an observed key root",
            "first = fold_in(key_from_seed(1i64), 1i64)\n".to_string(),
        ),
        (
            "an observed pair of keys",
            "pair = split_key(key_from_seed(1i64))\n".to_string(),
        ),
        (
            "an observed key tensor",
            "ks = split_keys(key_from_seed(1i64), 2i64)\n".to_string(),
        ),
        (
            "a draw keyed where it is bound",
            format!("{x0}d = dropout(key_from_seed(1i64), x0, 0.5f32)\n"),
        ),
    ] {
        accepts(name, &source);
    }
}

/// The `KeyReuse` diagnostics that reject a declaration's capture of the
/// top-level key `k0`.
fn top_level_key_captures(name: &str, source: &str) -> Vec<CheckError> {
    rejects(name, source, CheckErrorKind::KeyReuse)
        .into_iter()
        .filter(|error| {
            same_kind(&error.kind, &CheckErrorKind::KeyReuse)
                && error
                    .message
                    .contains("captures key-carrying variable `k0`")
                && error.message.contains("[04-LIN-9]")
        })
        .collect()
}

const TOP_LEVEL_KEY: &str =
    "k0 = key_from_seed(1i64)\nx0 = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])\n";

/// [04-LIN-9]: a named function declaration that refers to a top-level key
/// captures it, and every call would draw with it again, so the declaration
/// is refused whether or not anything calls it. A declaration that takes the
/// key as a parameter is the accepted twin.
///
/// Evidentiary status: DISPOSITION LOCK for chelis#2549, which checks a
/// declaration's body against the top-level scope rather than as a closure
/// created there. Each refusal must survive that change.
#[test]
fn a_function_declaration_never_captures_a_top_level_key() {
    let declaration = "def f(x: tensor[4, f32]) -> tensor[4, f32] = dropout(k0, x, 0.5f32)\n";
    for (call, suffix) in [("uncalled", ""), ("called", "out = f(x0)\n")] {
        let name = format!("a declaration reading `k0`, {call}");
        let captures =
            top_level_key_captures(&name, &format!("{TOP_LEVEL_KEY}{declaration}{suffix}"));
        assert_eq!(captures.len(), 1, "{name}: {captures:?}");
    }
    accepts(
        "the key passed to the declaration as a parameter",
        &format!(
            "{TOP_LEVEL_KEY}def f(k: key, x: tensor[4, f32]) -> tensor[4, f32] = dropout(k, x, \
             0.5f32)\nout = f(key_from_seed(2i64), x0)\n"
        ),
    );
}

/// Two declarations reading the same top-level key are each refused: the
/// first refusal does not consume `k0` on the second's behalf. One
/// declaration yields exactly one refusal (the test above), so two refusals
/// are one per declaration.
///
/// Evidentiary status: DISPOSITION LOCK for chelis#2549 (see above).
#[test]
fn every_function_declaration_reading_a_top_level_key_is_refused() {
    let name = "two declarations reading `k0`";
    let captures = top_level_key_captures(
        name,
        &format!(
            "{TOP_LEVEL_KEY}def f(x: tensor[4, f32]) -> tensor[4, f32] = dropout(k0, x, \
             0.5f32)\ndef g(x: tensor[4, f32]) -> tensor[4, f32] = uniform_like(k0, x, 0.0f32, \
             1.0f32)\n"
        ),
    );
    assert_eq!(captures.len(), 2, "{name}: {captures:?}");
}

/// [04-LIN-9] across top-level initializers: two draws from `k0` reuse it,
/// and a draw keyed where it is bound is its key's single use ([04-LIN-6]
/// makes `k0`'s own root observation its one use).
///
/// Evidentiary status: DISPOSITION LOCK for chelis#2549, which reorders how
/// top-level initializers and declarations are walked.
#[test]
fn top_level_initializers_draw_from_a_key_once() {
    rejects_reuse(
        "two top-level draws from `k0`",
        &format!(
            "{TOP_LEVEL_KEY}first = dropout(k0, x0, 0.5f32)\nsecond = dropout(k0, x0, 0.5f32)\n"
        ),
    );
    // [04-LIN-6]: `k0`'s root observation is its one use, so a draw keyed
    // where it is bound is the single-use twin.
    accepts(
        "one top-level draw keyed where it is bound",
        &format!("{TOP_LEVEL_KEY}first = dropout(key_from_seed(2i64), x0, 0.5f32)\n"),
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
    // [04-LIN-10]: an unbounded binder is no route either.
    rejects_key_instantiation(
        "an unbounded binder",
        "def ident[T](x: T) -> T = x\ndef bad(k: key) -> key = ident(k)\n",
        "ident",
        Some("T"),
    );
    rejects(
        "integers are not keys",
        "def bad() -> tensor[2, key] = to_tensor([1i64, 2i64])\n",
        CheckErrorKind::TypeMismatch,
    );
    verdict("def bad() -> tensor[2, key] = {\n  x: tensor[2, key] = [1, 2]\n  x\n}\n")
        .expect_err("a contextual key literal");
}

/// chelis#2216 / PR #2517: a builtin's operand rule is decided at every
/// instantiation of a rigid dtype binder, so a key tensor cannot reach
/// `to_list` through `listify`'s unbounded `p`.
#[test]
fn a_key_tensor_cannot_reach_a_dtype_operation_through_a_rigid_binder() {
    verdict(
        "def listify[n, p](x: tensor[n, p]) -> List[p] = to_list(x)\ndef bad(k: key) -> \
         List[key] = listify(split_keys(k, 2i64))\n",
    )
    .expect_err("to_list of a key tensor through a binder");
}

/// [04-LIN-10]: a function's type parameter, authored or an element-dtype
/// binder, is never instantiated at a key-carrying type (spec/04 section
/// 8.4.1): a key, a key tensor, or a tuple, list, option or data type that
/// holds one.
#[test]
fn a_type_parameter_is_never_instantiated_at_a_key() {
    let dup = "def dup[a](x: a) -> (a, a) = (x, x)\n";
    let id = "def id[a](x: a) -> a = x\n";
    let holder = "type Holder =\n  | Holder { k: key }\n";
    let cases: [(&str, String, &str, &str); 9] = [
        (
            "dup on a key",
            format!("{dup}def bad(k: key) -> (key, key) = dup(k)\n"),
            "dup",
            "a",
        ),
        (
            "an element-dtype binder on a key tensor",
            "def dup[n, p](x: tensor[n, p]) -> (tensor[n, p], tensor[n, p]) = (copy(x), x)\ndef \
             bad(k: key) -> (tensor[2, key], tensor[2, key]) = dup(split_keys(k, 2i64))\n"
                .to_string(),
            "dup",
            "p",
        ),
        (
            "id on a key",
            format!("{id}def bad(k: key) -> key = id(k)\n"),
            "id",
            "a",
        ),
        (
            "id on a key tensor",
            format!("{id}def bad(k: key) -> tensor[2, key] = id(split_keys(k, 2i64))\n"),
            "id",
            "a",
        ),
        (
            "id on an option of a key",
            format!("{id}def bad(k: key) -> Option[key] = id(Some(k))\n"),
            "id",
            "a",
        ),
        (
            "id on a tuple holding a key",
            format!("{id}def bad(k: key) -> (key, i64) = id((k, 1i64))\n"),
            "id",
            "a",
        ),
        (
            "id on a data type with a key field",
            format!("{holder}{id}def bad(k: key) -> Holder = id(Holder {{ k }})\n"),
            "id",
            "a",
        ),
        (
            "id on a data type holding a key-carrying data type",
            format!(
                "type Outer =\n  | Outer {{ h: Wrap[Holder] }}\ntype Wrap[t] =\n  | Wrap {{ v: t \
                 }}\n{holder}{id}def bad(o: Outer) -> Outer = id(o)\n"
            ),
            "id",
            "a",
        ),
        (
            "a list element binder",
            "def first[a](xs: List[a]) -> a = index(xs, 0i64)\ndef bad(ks: List[key]) -> key = \
             first(ks)\n"
                .to_string(),
            "first",
            "a",
        ),
    ];
    for (name, source, generic, parameter) in cases {
        rejects_key_instantiation(name, &source, generic, Some(parameter));
    }
}

/// [04-LIN-10] holds however the generic is reached: as another function's
/// argument, stored in a tuple or a data value, returned from a function, or
/// generalized by a `let`. These are the routes a rule that read generic
/// bodies missed (chelis#2541).
#[test]
fn a_generic_reached_indirectly_is_never_instantiated_at_a_key() {
    let dup = "def dup[a](x: a) -> (a, a) = (x, x)\n";
    let cases: [(&str, String, &str, Option<&str>); 9] = [
        (
            "map(dup, keys)",
            format!("{dup}def bad(ks: List[key]) -> List[(key, key)] = map(dup, ks)\n"),
            "dup",
            Some("a"),
        ),
        (
            "map(dup, [a, b]), chelis#2541's witness",
            format!(
                "{dup}def bad(k: key) -> List[(key, key)] = {{\n  (a, b) = split_key(k)\n  \
                 map(dup, [a, b])\n}}\n"
            ),
            "dup",
            Some("a"),
        ),
        (
            "a generic stored in a record and then applied",
            format!(
                "type Twice[a] =\n  | Twice {{ f: (a) -> (a, a) }}\n{dup}def bad(k: key) -> (key, \
                 key) = match Twice {{ f: dup }} with {{\n  | Twice {{ f }} => f(k)\n}}\n"
            ),
            "dup",
            Some("a"),
        ),
        (
            "a generic stored in a list and then applied",
            format!(
                "{dup}def bad(k: key) -> (key, key) = {{\n  fs = [dup]\n  (index(fs, 0i64))(k)\n}}\n"
            ),
            "dup",
            Some("a"),
        ),
        (
            "a generic stored in a tuple and then applied",
            format!("{dup}def bad(k: key) -> (key, key) = {{\n  t = (dup, 1i64)\n  (t.0)(k)\n}}\n"),
            "dup",
            Some("a"),
        ),
        (
            "a generic returned from a function and then applied",
            format!(
                "{dup}def get[a]() -> (a) -> (a, a) = dup\ndef bad(k: key) -> (key, key) = (get())(k)\n"
            ),
            "get",
            Some("a"),
        ),
        (
            "a wrapper over a generic",
            "def id[a](x: a) -> a = x\ndef wrap[b](x: b) -> b = id(x)\ndef bad(k: key) -> key = \
             wrap(k)\n"
                .to_string(),
            "wrap",
            Some("b"),
        ),
        (
            "a let-generalized closure",
            "def bad(k: key) -> key = {\n  g = fn (y) -> y\n  g(k)\n}\n".to_string(),
            "g",
            None,
        ),
        (
            "a duplicating let-generalized closure",
            "def bad(k: key) -> (key, key) = {\n  g = fn (y) -> (y, y)\n  g(k)\n}\n".to_string(),
            "g",
            None,
        ),
    ];
    for (name, source, generic, parameter) in cases {
        rejects_key_instantiation(name, &source, generic, parameter);
    }
    rejects_value_binding_instantiation(
        "a closure held in a let-generalized data value",
        "type Twice[a] =\n  | Twice { f: (a) -> (a, a) }\ndef bad(k: key) -> (key, key) = {\n  \
         d = Twice { f: fn (y) -> (y, y) }\n  match d with {\n    | Twice { f } => f(k)\n  \
         }\n}\n",
        "d",
    );
}

/// [04-LIN-10] covers every variable a `let` generalizes, not only a
/// closure's, because a tuple or data value can hold a closure over it. A
/// generic `let` value is therefore never instantiated at a key; the repair
/// writes the value in place or annotates it, and the diagnostic suggests
/// exactly that repair for a value binding, not a concrete key parameter.
///
/// Evidentiary status: the refusal and the accepted twins are DISPOSITION
/// LOCKS (they hold at `6f42b4e92`); the suggestion assertion is a
/// REGRESSION TEST (at `6f42b4e92` the refusal suggested a concrete key
/// parameter, which does not repair a `let` value).
#[test]
fn a_let_generalized_value_is_never_instantiated_at_a_key() {
    rejects_value_binding_instantiation(
        "a let-bound empty list",
        "def bad(k: key) -> List[key] = {\n  e = Nil\n  Cons(k, e)\n}\n",
        "e",
    );
    accepts(
        "the empty list written in place",
        "def good(k: key) -> List[key] = Cons(k, Nil)\n",
    );
    accepts(
        "an annotated empty list",
        "def good(k: key) -> List[key] = {\n  e: List[key] = Nil\n  Cons(k, e)\n}\n",
    );
}

/// [04-LIN-10]'s positive twins: keys pass through concrete `key` and
/// `tensor[n, key]` parameters, data-type parameters take keys, the builtin
/// operations take keys under [04-LIN-9], and a generic still serves every
/// other type, beside a key.
#[test]
fn keys_pass_through_concrete_parameters_and_data_types() {
    let dup = "def dup[a](x: a) -> (a, a) = (x, x)\n";
    let holder = "type Holder =\n  | Holder { k: key }\n";
    let cases: [(&str, String); 9] = [
        (
            "dup on a non-key type",
            format!("{dup}def good(x: f32) -> (f32, f32) = dup(x)\n"),
        ),
        (
            "a concrete key parameter",
            "def halves(k: key) -> (key, key) = split_key(k)\ndef good(k: key) -> (key, key) = \
             halves(k)\n"
                .to_string(),
        ),
        (
            "a concrete key tensor parameter",
            "def first_two(ks: tensor[2, key]) -> tensor[2, key] = ks\ndef good(k: key) -> \
             tensor[2, key] = first_two(split_keys(k, 2i64))\n"
                .to_string(),
        ),
        (
            "a Float generic beside a key parameter",
            "def scale[p: Float](x: tensor[3, p], k: key) -> (tensor[3, p], key) = (x, k)\ndef \
             good(x: tensor[3, f32], k: key) -> (tensor[3, f32], key) = scale(x, k)\n"
                .to_string(),
        ),
        (
            "an unbounded generic on numbers beside a key",
            "def pick[a](x: a, y: a) -> a = x\ndef good(k: key, x: f32, y: f32) -> (key, f32) = \
             (fold_in(k, 1i64), pick(x, y))\n"
                .to_string(),
        ),
        (
            "List[key] and Option[key] values",
            "def good(k: key) -> (List[key], Option[key]) = {\n  (a, b) = split_key(k)\n  \
             (Cons(a, Nil), Some(b))\n}\n"
                .to_string(),
        ),
        (
            "a data type with a key field to a concrete parameter",
            format!(
                "{holder}def open(h: Holder) -> key = match h with {{\n  | Holder {{ k }} => \
                 k\n}}\ndef good(k: key) -> key = open(Holder {{ k }})\n"
            ),
        ),
        (
            "a builtin over a list of keys",
            "def good(ks: List[key]) -> List[(key, key)] = map(split_key, ks)\n".to_string(),
        ),
        (
            "a let-generalized closure on numbers",
            "def good(x: f32) -> f32 = {\n  g = fn (y) -> y\n  g(x)\n}\n".to_string(),
        ),
    ];
    for (name, source) in cases {
        accepts(name, &source);
    }
}

fn surf_program(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("fixture parses");
    desugar_program(&decls).expect("fixture desugars")
}

fn context_verdict(context: &TypeEnv, source: &str) -> Result<(), Vec<CheckError>> {
    check_ir_with_context(context, &surf_program(source))
        .map(|_| ())
        .map_err(|result| result.errors)
}

/// [04-LIN-10] binds a library's generics too, before and after the checker
/// context round-trips through its serialized form: chelis-std's own
/// unbounded generics still serve numbers and refuse keys, and a library data
/// type with a key field is key-carrying in the code that imports it.
#[test]
fn a_library_generic_is_never_instantiated_at_a_key() {
    let mut library = Vec::new();
    for source in [
        include_str!("../../../packages/chelis-std/src/index.ch"),
        include_str!("../../../packages/chelis-std/src/scan.ch"),
        "def lib_id[a](x: a) -> a = x\ntype LibHolder =\n  | LibHolder { k: key }\n",
    ] {
        library.extend(surf_program(source));
    }
    let live = build_type_env_from_library(&library).expect("the library checks");
    let bytes = bincode::serialize(&live).expect("the context serializes");
    let restored: TypeEnv = bincode::deserialize(&bytes).expect("the context deserializes");
    for context in [&live, &restored] {
        for (name, source) in [
            (
                "std generics on numbers",
                "def good(xs: List[f32]) -> (f32, List[f32], List[f32]) = (list_index(xs, 0i64), \
                 take_list(xs, 1i64), skip_list(xs, 1i64))\n",
            ),
            (
                "a std generic over a step function",
                "def good(xs: List[i64]) -> List[i64] = scan_list(fn (acc, x) -> add(acc, x), \
                 0i64, xs)\n",
            ),
        ] {
            if let Err(errors) = context_verdict(context, source) {
                panic!("{name}: expected acceptance, got {errors:?}");
            }
        }
        for (name, source, generic, parameter) in [
            (
                "a std generic on a list of keys",
                "def bad(ks: List[key]) -> key = list_index(ks, 0i64)\n",
                "list_index",
                "item",
            ),
            (
                "another std generic on a list of keys",
                "def bad(ks: List[key]) -> List[key] = take_list(ks, 1i64)\n",
                "take_list",
                "item",
            ),
            (
                "a library generic on a library data type with a key field",
                "def bad(k: key) -> LibHolder = lib_id(LibHolder { k })\n",
                "lib_id",
                "a",
            ),
        ] {
            let errors = context_verdict(context, source)
                .expect_err(&format!("{name}: expected a rejection"));
            assert!(
                errors.iter().any(|error| {
                    same_kind(&error.kind, &CheckErrorKind::KeyReuse)
                        && error.message.contains(&format!(
                            "type parameter `{parameter}` of generic `{generic}`"
                        ))
                }),
                "{name}: expected [04-LIN-10] naming `{parameter}` of `{generic}`, got {errors:?}"
            );
        }
    }
}

/// Spec/04 section 1.1: a key has no cast in either direction, whatever the
/// target: a concrete dtype or a declaration's bounded dtype binder. The
/// numeric twins of every shape still check.
#[test]
fn a_key_has_no_cast_to_any_target() {
    for op in ["cast", "cast_trunc"] {
        for (name, source) in [
            (
                "a scalar key to a concrete target",
                format!("def bad(k: key) -> i32 = {op}(k, i32)\n"),
            ),
            (
                "a key tensor to a concrete target",
                format!("def bad(k: key) -> tensor[2, i32] = {op}(split_keys(k, 2i64), i32)\n"),
            ),
            (
                "a scalar key to a binder target",
                format!("def bad[p: Int](k: key) -> p = {op}(k, p)\n"),
            ),
            (
                "a key tensor to an Int binder target",
                format!("def bad[n, p: Int](x: tensor[n, key]) -> tensor[n, p] = {op}(x, p)\n"),
            ),
            (
                "a key tensor to a Float binder target",
                format!("def bad[n, p: Float](x: tensor[n, key]) -> tensor[n, p] = {op}(x, p)\n"),
            ),
            (
                "a derived key tensor to a binder target",
                format!("def bad[p: Int](k: key) -> tensor[2, p] = {op}(split_keys(k, 2i64), p)\n"),
            ),
            (
                "a scalar to key",
                format!("def bad(x: f32) -> key = {op}(x, key)\n"),
            ),
            (
                "a tensor to key",
                format!("def bad(x: tensor[2, f32]) -> tensor[2, key] = {op}(x, key)\n"),
            ),
            (
                "a binder scalar to key",
                format!("def bad[p: Float](x: p) -> key = {op}(x, key)\n"),
            ),
            (
                "a binder tensor to key",
                format!("def bad[n, p: Float](x: tensor[n, p]) -> tensor[n, key] = {op}(x, key)\n"),
            ),
        ] {
            rejects(
                &format!("{op}: {name}"),
                &source,
                CheckErrorKind::UnsupportedTensorPrecision,
            );
        }
        for (name, source) in [
            (
                "a scalar to a concrete target",
                format!("def good(x: f32) -> i32 = {op}(x, i32)\n"),
            ),
            (
                "a tensor to a concrete target",
                format!("def good(x: tensor[2, f32]) -> tensor[2, i32] = {op}(x, i32)\n"),
            ),
            (
                "a scalar to a binder target",
                format!("def good[p: Int](x: f32) -> p = {op}(x, p)\n"),
            ),
            (
                "a tensor to a binder target",
                format!("def good[n, p: Int](x: tensor[n, f32]) -> tensor[n, p] = {op}(x, p)\n"),
            ),
        ] {
            accepts(&format!("{op}: {name}"), &source);
        }
    }
}

/// [04-LIN-9]: a key inside a value is reached only by consuming that value,
/// so an as-pattern cannot bind a key-carrying whole together with a
/// key-carrying component, in any pattern form. A sub-pattern that binds no
/// key, one that binds only a non-key field, and an as-pattern over a value
/// with no key stay legal.
///
/// Evidentiary status: REGRESSION TEST. At `b47fdd7d3` every negative was
/// accepted: the whole and its components were separate owners.
#[test]
fn an_as_pattern_never_binds_a_key_twice() {
    let holder = "type Holder =\n  | Holder { k: key, n: i64 }\n";
    for (name, source) in [
        (
            "an option whole and its key",
            "def bad(o: Option[key]) -> (Option[key], key) = match o with {\n  | whole @ Some(j) \
             => (whole, j)\n  | None => (None, key_from_seed(0i64))\n}\n"
                .to_string(),
        ),
        (
            "a tuple whole and a key component",
            "def bad(k: key) -> ((key, key), key) = match split_key(k) with {\n  | whole @ (a, \
             b) => (whole, a)\n}\n"
                .to_string(),
        ),
        (
            "one key bound under two names",
            "def bad(k: key) -> (key, key) = match k with {\n  | a @ b => (a, b)\n}\n".to_string(),
        ),
        (
            "a record whole and its key field",
            format!(
                "{holder}def bad(h: Holder) -> (Holder, key) = match h with {{\n  | whole @ \
                 Holder {{ k, n }} => (whole, k)\n}}\n"
            ),
        ),
        (
            "a nested as-pattern",
            "def bad(o: Option[(key, i64)]) -> (Option[(key, i64)], key) = match o with {\n  | \
             Some(p @ (j, n)) => (Some(p), j)\n  | None => (None, key_from_seed(0i64))\n}\n"
                .to_string(),
        ),
    ] {
        let errors = rejects(name, &source, CheckErrorKind::KeyReuse);
        assert!(
            errors
                .iter()
                .any(|error| error.message.contains("as-pattern")
                    && error.message.contains("[04-LIN-9]")
                    && error
                        .suggestions
                        .iter()
                        .all(|suggestion| !suggestion.contains("copy("))),
            "{name}: expected the as-pattern diagnostic: {errors:?}"
        );
    }
    for (name, source) in [
        (
            "a sub-pattern that binds no key",
            "def good(o: Option[key]) -> Option[key] = match o with {\n  | whole @ Some(_) => \
             whole\n  | None => None\n}\n"
                .to_string(),
        ),
        (
            "a sub-pattern that binds only a non-key field",
            format!(
                "{holder}def good(h: Holder) -> (Holder, i64) = match h with {{\n  | whole @ \
                 Holder {{ k: _, n }} => (whole, n)\n}}\n"
            ),
        ),
        (
            "an as-pattern over a value with no key",
            "def good(o: Option[i64]) -> (Option[i64], i64) = match o with {\n  | whole @ \
             Some(j) => (whole, j)\n  | None => (None, 0i64)\n}\n"
                .to_string(),
        ),
    ] {
        accepts(name, &source);
    }
}

/// [04-LIN-9] on every control-flow path: an arm's guard runs before its
/// body and, when it fails, before every later arm, so a key the guard
/// consumed is consumed on entry to each of them, whether it is an outer
/// binding or a scrutinee component the arm's pattern bound. A later arm
/// that binds a disjoint component, and a guard whose keys no later arm
/// uses, stay legal.
///
/// Evidentiary status: REGRESSION TEST. At `b47fdd7d3` every negative was
/// accepted: each arm started from the match-entry scope.
#[test]
fn a_guard_consumes_its_keys_on_every_later_arm() {
    let pred = "def pred(k: key) -> bool = true\n";
    let low = "def low(k: key, x: tensor[4, f32]) -> bool = lt(index(to_list(uniform_like(k, x, \
               0.0f32, 1.0f32)), 0i64), 0.5f32)\n";
    let two = "type Two =\n  | A(key)\n  | B(key)\n";
    for (name, source) in [
        (
            "an outer key after the guard that consumed it",
            format!(
                "{pred}def bad(o: Option[i64], k: key) -> key = match o with {{\n  | Some(n) if \
                 pred(k) => key_from_seed(0i64)\n  | _ => k\n}}\n"
            ),
        ),
        (
            "a draw in a guard, then a draw in a later arm",
            format!(
                "{low}def bad(k: key, x: tensor[4, f32], n: i64) -> tensor[4, f32] = match n with \
                 {{\n  | 0 if low(k, copy(x)) => x\n  | _ => uniform_like(k, x, 0.0f32, \
                 1.0f32)\n}}\n"
            ),
        ),
        (
            "a component the guard consumed, bound again by a later arm",
            format!(
                "{pred}def bad(o: Option[key]) -> key = match o with {{\n  | Some(j) if pred(j) \
                 => key_from_seed(0i64)\n  | Some(j2) => j2\n  | None => \
                 key_from_seed(1i64)\n}}\n"
            ),
        ),
        (
            "the whole scrutinee bound by a later arm",
            format!(
                "{pred}def bad(o: Option[key]) -> Option[key] = match o with {{\n  | Some(j) if \
                 pred(j) => None\n  | other => other\n}}\n"
            ),
        ),
        (
            "a key component the guard projected, taken again by a later arm",
            format!(
                "{pred}def bad(o: Option[(key, key)]) -> key = match o with {{\n  | Some(p) if \
                 pred(p.0) => key_from_seed(0i64)\n  | Some(q) => q.0\n  | None => \
                 key_from_seed(1i64)\n}}\n"
            ),
        ),
    ] {
        let errors = rejects(name, &source, CheckErrorKind::KeyReuse);
        assert!(
            errors.iter().any(|error| error
                .message
                .contains("in the guard of an earlier arm, which falls through to this one")),
            "{name}: the diagnostic must name the falling-through guard: {errors:?}"
        );
        rejects_reuse(name, &source);
    }
    for (name, source) in [
        (
            "a guard consuming a key no later arm uses",
            format!(
                "{pred}def good(o: Option[i64], k: key) -> key = match o with {{\n  | Some(n) if \
                 pred(k) => key_from_seed(0i64)\n  | _ => key_from_seed(1i64)\n}}\n"
            ),
        ),
        (
            "a guard that reads no key",
            "def good(o: Option[i64], k: key) -> key = match o with {\n  | Some(n) if gt(n, \
             0i64) => fold_in(k, n)\n  | _ => k\n}\n"
                .to_string(),
        ),
        (
            "a later arm binding another constructor's key",
            format!(
                "{pred}{two}def good(t: Two) -> key = match t with {{\n  | A(j) if pred(j) => \
                 key_from_seed(0i64)\n  | A(_) => key_from_seed(1i64)\n  | B(j2) => j2\n}}\n"
            ),
        ),
        (
            "a later arm taking the other tuple component",
            format!(
                "{pred}def good(p: (key, key)) -> key = match p with {{\n  | (a, b) if pred(a) \
                 => b\n  | (c, d) => d\n}}\n"
            ),
        ),
    ] {
        accepts(name, &source);
    }
}

/// [04-LIN-9] and spec/06 section 3.6: `vmap` broadcasts an argument that is
/// not a tensor to every row, so a key-carrying one would be used by each row.
/// Keys reach the rows only as a mapped `tensor[n, key]`; a broadcast
/// argument with no key stays legal.
///
/// Evidentiary status: REGRESSION TEST. At `b47fdd7d3` the option, record,
/// list and key-tensor-in-a-list negatives checked.
#[test]
fn vmap_never_broadcasts_a_key() {
    let holder = "type Holder =\n  | Holder { k: key, n: i64 }\n";
    for (name, source) in [
        (
            "an option holding a key",
            "def row(o: Option[key], x: tensor[2, f32]) -> tensor[2, f32] = match o with {\n  | \
             Some(k) => dropout(k, x, 0.5f32)\n  | None => x\n}\ndef bad(k: key, xs: tensor[3, 2, \
             f32]) -> tensor[3, 2, f32] = vmap(row)(Some(k), xs)\n"
                .to_string(),
        ),
        (
            "a record with a key field",
            format!(
                "{holder}def row(h: Holder, x: tensor[2, f32]) -> tensor[2, f32] = match h with \
                 {{\n  | Holder {{ k, n }} => dropout(k, x, 0.5f32)\n}}\ndef bad(h: Holder, xs: \
                 tensor[3, 2, f32]) -> tensor[3, 2, f32] = vmap(row)(h, xs)\n"
            ),
        ),
        (
            "a list of keys",
            "def row(ks: List[key], x: tensor[2, f32]) -> tensor[2, f32] = match ks with {\n  | \
             Cons(k, rest) => dropout(k, x, 0.5f32)\n  | Nil => x\n}\ndef bad(k: key, xs: \
             tensor[3, 2, f32]) -> tensor[3, 2, f32] = vmap(row)([k], xs)\n"
                .to_string(),
        ),
        (
            "a list of key tensors",
            "def row(ks: List[tensor[3, key]], x: tensor[2, f32]) -> tensor[2, f32] = x\ndef \
             bad(k: key, xs: tensor[3, 2, f32]) -> tensor[3, 2, f32] = vmap(row)([split_keys(k, \
             3i64)], xs)\n"
                .to_string(),
        ),
    ] {
        let errors = rejects(name, &source, CheckErrorKind::KeyReuse);
        assert!(
            errors.iter().any(
                |error| error.message.contains("`vmap` broadcasts argument 0")
                    && error.message.contains("[04-LIN-9]")
                    && error.suggestions.join(" ").contains("split_keys")
            ),
            "{name}: expected the broadcast diagnostic: {errors:?}"
        );
    }
    let row = "def row(k: key, x: tensor[2, f32]) -> tensor[2, f32] = dropout(k, x, 0.5f32)\n";
    for (name, source) in [
        (
            "split_keys feeding a key formal",
            format!(
                "{row}def good(k: key, xs: tensor[3, 2, f32]) -> tensor[3, 2, f32] = \
                 vmap(row)(split_keys(k, 3i64), xs)\n"
            ),
        ),
        (
            "a broadcast option with no key",
            "def row(o: Option[i64], x: tensor[2, f32]) -> tensor[2, f32] = x\ndef good(xs: \
             tensor[3, 2, f32]) -> tensor[3, 2, f32] = vmap(row)(Some(1i64), xs)\n"
                .to_string(),
        ),
    ] {
        accepts(name, &source);
    }
}

/// [04-LIN-9]: no signature declares a borrowed key-carrying parameter,
/// however the type is spelled: through a type alias, through a `sig`, or as
/// an aliased tuple that holds a key. An alias of a type with no key stays
/// borrowable.
///
/// Evidentiary status: REGRESSION TEST. At `b47fdd7d3` the three aliased
/// negatives checked: the rule read the annotation, which a `sig` leaves
/// untyped and which carries the alias unresolved.
#[test]
fn a_borrowed_key_parameter_is_refused_through_an_alias() {
    for (name, source) in [
        (
            "an aliased key",
            "type K = key\ndef bad(k: &K) -> i64 = 0i64\n",
        ),
        (
            "an aliased key in a sig",
            "type K = key\nsig bad: &K -> i64\ndef bad(k) = 0i64\n",
        ),
        (
            "an aliased tuple holding a key",
            "type KP = (key, tensor[2, f32])\ndef bad(p: &KP) -> i64 = 0i64\n",
        ),
    ] {
        let errors = rejects(name, source, CheckErrorKind::KeyReuse);
        assert!(
            errors
                .iter()
                .any(|error| error.message.contains("borrows a key-carrying type")),
            "{name}: expected the borrowed-parameter diagnostic: {errors:?}"
        );
    }
    accepts(
        "an aliased tensor stays borrowable",
        "type T = tensor[2, f32]\ndef good(t: &T) -> tensor[2, f32] = neg(t)\n",
    );
}

/// #2413: the counter-stream draws took no key. A call at their retired
/// arity is refused with a diagnostic that names the retired spelling and
/// the keyed one, not a bare arity count; the keyed call checks.
///
/// Evidentiary status: REGRESSION TEST. At `b47fdd7d3` both negatives gave
/// only "function arity mismatch: expected N args, got M".
#[test]
fn a_retired_draw_spelling_points_at_keys() {
    for (retired, keyed, source) in [
        (
            "dropout(x, rate)",
            "dropout(k, x, rate)",
            "def bad(x: tensor[4, f32]) -> tensor[4, f32] = dropout(x, 0.5f32)\n",
        ),
        (
            "uniform_like(t, low, high)",
            "uniform_like(k, t, low, high)",
            "def bad(x: tensor[4, f32]) -> tensor[4, f32] = uniform_like(x, 0.0f32, 1.0f32)\n",
        ),
    ] {
        let errors = rejects(retired, source, CheckErrorKind::ArityMismatch);
        assert!(
            errors.iter().any(|error| {
                error.message.contains(&format!(
                    "`{retired}` is the retired counter-stream spelling"
                )) && error.message.contains(keyed)
                    && error.suggestions.join(" ").contains("key_from_seed")
            }),
            "{retired}: expected the retired-spelling diagnostic: {errors:?}"
        );
    }
    accepts(
        "the keyed draws",
        "def good(k: key, x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = {\n  (a, b) = \
         split_key(k)\n  (dropout(a, x, 0.5f32), uniform_like(b, x, 0.0f32, 1.0f32))\n}\n",
    );
}
