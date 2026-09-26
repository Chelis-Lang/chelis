//! chelis#2413 round-1 repairs: the legal twins of the [04-LIN-9] and
//! [04-LIN-10] refusals still check, and run to the same values in the host
//! interpreter (`chelis eval`) and in native C, with the ownership ledger
//! balanced.
//!
//! Every expected draw is a bit pattern computed by
//! `briefs/keys-b-slice2-probes/slice2_ref.py` (`uniform(k, n, 'f32', 0, 1)`)
//! over `key_ref.py`'s independent transcription of [05-RNG-2]; no expected
//! value is computed by compiler code.
mod ownership_support;

use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use chelis_types::types::Prim;
use chelis_types::{ElementRef, format_element};
use std::collections::BTreeMap;

/// `uniform_like(key_from_seed(7), t, 0, 1)` over 2 elements.
const U_KEY7_2: [u32; 2] = [0x3e019516, 0x3f56526f];
/// The first draw of `uniform(split(key(7)).0, 2, f32, 0, 1)` and of `.1`.
const U_LEFT7_FIRST: u32 = 0x3f251727;
const U_RIGHT7_FIRST: u32 = 0x3ef80128;
/// `uniform(fold_in(fold_in(key(7), 3), 5), 2, f32, 0, 1)`.
const U_FOLD7_3_5: [u32; 2] = [0x3f7d19f2, 0x3d937002];
const ONE_TWO: [u32; 2] = [0x3f800000, 0x40000000];
const NEG_ONE_TWO: [u32; 2] = [0xbf800000, 0xc0000000];

fn f32_text(bits: u32) -> String {
    format_element(Prim::F32, ElementRef::F32(f32::from_bits(bits)))
}

fn tensor(shape: &[usize], bits: &[u32]) -> String {
    let data = bits
        .iter()
        .map(|bits| f32_text(*bits))
        .collect::<Vec<_>>()
        .join(", ");
    format!("tensor(shape={shape:?}, data=[{data}])")
}

/// `name = display` for every root `chelis eval` reports, in order.
fn eval_lines(source: &str) -> Vec<String> {
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        },
        &["main".into()],
    )
    .unwrap_or_else(|error| panic!("eval failed: {error:?}\n{source}"));
    result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}",
                root.name.as_deref().expect("a named root"),
                root.display.as_deref().expect("an in-process display")
            )
        })
        .collect()
}

/// The lines the native C program prints for its roots, with the ownership
/// ledger balanced.
fn c_lines(source: &str, label: &str) -> Vec<String> {
    let generated = ownership_support::emit(source, label);
    let (summary, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
    stdout.lines().map(str::to_string).collect()
}

/// Both lanes print exactly `expected`, so they also agree bit for bit.
fn both_lanes(source: &str, label: &str, expected: &[String]) {
    let eval = eval_lines(source);
    assert_eq!(eval, expected, "eval: {label}\n{source}");
    let native = c_lines(source, label);
    assert_eq!(native, expected, "C: {label}\n{source}");
}

fn main_lines(values: &[String]) -> Vec<String> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| format!("main.{index} = {value}"))
        .collect()
}

/// The as-pattern twin: `whole @ Some(_)` binds the key-carrying whole and
/// no component, so the key inside reaches exactly one draw.
#[test]
fn an_as_pattern_binding_no_key_component_draws_once() {
    let source = "def keep(o: Option[key]) -> Option[key] = match o with {\n  | whole @ Some(_) => \
                  whole\n  | None => None\n}\n\
                  def main() = {\n  t = to_tensor([0.0f32, 0.0f32])\n  (match \
                  keep(Some(key_from_seed(7i64))) with {\n    | Some(k) => uniform_like(k, t, \
                  0.0f32, 1.0f32)\n    | None => t\n  }, to_tensor([1.0f32, 2.0f32]))\n}\n";
    both_lanes(
        source,
        "as_pattern_twin",
        &main_lines(&[tensor(&[2], &U_KEY7_2), tensor(&[2], &ONE_TWO)]),
    );
}

/// The guard twin: a guard draws with its key and no later arm uses it.
/// `key_from_seed(7)` draws 0.127 first (below 0.5, so the guarded arm is
/// taken), `key_from_seed(9)` draws 0.852 first (the guard fails and falls
/// through), and `n = 1` never runs the guard.
#[test]
fn a_guard_that_draws_with_a_key_no_later_arm_uses_runs() {
    let source = "def gate(k: key, x: tensor[2, f32]) -> bool = lt(index(to_list(uniform_like(k, \
                  x, 0.0f32, 1.0f32)), 0i64), 0.5f32)\n\
                  def pick(k: key, n: i64, x: tensor[2, f32]) -> tensor[2, f32] = match n with \
                  {\n  | 0 if gate(k, copy(x)) => x\n  | _ => neg(x)\n}\n\
                  def main() = (pick(key_from_seed(7i64), 0i64, to_tensor([1.0f32, 2.0f32])), \
                  pick(key_from_seed(9i64), 0i64, to_tensor([1.0f32, 2.0f32])), \
                  pick(key_from_seed(8i64), 1i64, to_tensor([1.0f32, 2.0f32])))\n";
    both_lanes(
        source,
        "guard_twin",
        &main_lines(&[
            tensor(&[2], &ONE_TWO),
            tensor(&[2], &NEG_ONE_TWO),
            tensor(&[2], &NEG_ONE_TWO),
        ]),
    );
}

/// The `map` twin of the P1-2 refusals: `map` hands each key of a
/// `List[key]` to its callback once and keeps only the callback's results.
#[test]
fn map_over_a_key_list_draws_once_per_key() {
    let source = "def first_draw(j: key) -> f32 = index(to_list(uniform_like(j, to_tensor([0.0f32, \
                  0.0f32]), 0.0f32, 1.0f32)), 0i64)\n\
                  def main() = {\n  (a, b) = split_key(key_from_seed(7i64))\n  map(first_draw, [a, \
                  b])\n}\n";
    both_lanes(
        source,
        "map_key_list",
        &[format!(
            "main = [{}, {}]",
            f32_text(U_LEFT7_FIRST),
            f32_text(U_RIGHT7_FIRST)
        )],
    );
}

/// The `fold` twin of the `scan` refusal: each key accumulator feeds the
/// next call once, and only the last is the result.
#[test]
fn fold_threads_a_key_accumulator_once_per_element() {
    let source = "def main() = {\n  k = fold(fn (acc: key, n: i64) -> fold_in(acc, n), \
                  key_from_seed(7i64), [3i64, 5i64])\n  uniform_like(k, to_tensor([0.0f32, \
                  0.0f32]), 0.0f32, 1.0f32)\n}\n";
    both_lanes(
        source,
        "fold_key_accumulator",
        &[format!("main = {}", tensor(&[2], &U_FOLD7_3_5))],
    );
}
