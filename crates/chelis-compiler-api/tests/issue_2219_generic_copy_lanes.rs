//! chelis#2219: an explicit `copy` of any copyable value runs on both lanes.
//!
//! `spec/04-type-system.md` section 8.2 makes `copy(x)` generic: an owned `T`
//! or a borrowed `&T` yields a fresh owned `T`, and explicit copies lower to
//! the same copy as the compiler's inserted ones. The checker and the
//! evaluator admitted only tensors.
//!
//! Oracle: each program prints the `chelis eval` rendering pinned here, and
//! compiled to C and run against the `ownership-ledger` runtime it prints the
//! same with every allocation finalized. The negative twins are a key, which
//! [04-LIN-9] refuses to copy, and an operand whose type never settles; both
//! lanes refuse each with the same diagnostic.

mod ownership_support;

use chelis_compiler_api::compiler::{compile, eval};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const PRELUDE: &str = "module Demo.Main\n\
type Lin[i] =\n  | Lin { w: tensor[i, f32] }\n\
def weight(p: Lin[2]) -> tensor[2, f32] = match p with {\n  | Lin { w } => w\n}\n\
def loss(p: Lin[2], x: tensor[2, f32]) -> f32 = match p with {\n  | Lin { w } => tensor_to_scalar(sum(mul(w, x), 0i32))\n}\n\
def pair_loss(ws: List[tensor[2, f32]]) -> f32 = tensor_to_scalar(sum(mul(index(ws, 0i64), index(ws, 1i64)), 0i32))\n\
def rows(seed: i64) -> List[tensor[2, f32]] = [to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])]\n\
def grow(xs: List[tensor[2, f32]]) -> List[tensor[2, f32]] = append(xs, to_tensor([5.0f32, 6.0f32]))\n\
def regrow(xs: &List[tensor[2, f32]]) -> List[tensor[2, f32]] = grow(copy(xs))\n\
def bump(ps: List[Lin[2]]) -> List[Lin[2]] = append(ps, Lin { w: to_tensor([9.0f32, 9.0f32]) })\n\
def poke(p: Lin[2]) -> Lin[2] = Lin { w: scatter_replace(weight(p), to_tensor([0i64]), to_tensor([100.0f32]), 0i32) }\n\
def own(p: &Lin[2]) -> Lin[2] = copy(p)\n";

struct Case {
    name: &'static str,
    body: &'static str,
    eval: &'static str,
}

const POSITIVE: &[Case] = &[
    Case {
        name: "list",
        body: "def list_case(seed: i64) -> i64 = {\n  xs = rows(seed)\n  ys = grow(copy(xs))\n  zs = grow(xs)\n  add(len(ys), len(zs))\n}\nn = list_case(0i64)\n",
        eval: "n = 6\n",
    },
    Case {
        name: "borrowed_list_to_owned",
        body: "def borrowed_case(seed: i64) -> i64 = {\n  xs = rows(seed)\n  a = len(regrow(xs))\n  add(a, len(xs))\n}\nn = borrowed_case(0i64)\n",
        eval: "n = 5\n",
    },
    Case {
        name: "adt",
        body: "def adt_case(seed: i64) -> tensor[2, f32] = {\n  p = Lin { w: to_tensor([1.0f32, 2.0f32]) }\n  q = copy(p)\n  add(weight(p), weight(q))\n}\nresult = to_list(adt_case(0i64))\n",
        eval: "result = [2.0, 4.0]\n",
    },
    Case {
        name: "dict",
        body: "def dict_case(seed: i64) -> i64 = {\n  d = dict_of([(\"a\", to_tensor([1.0f32, 2.0f32]))])\n  e = dict_insert(copy(d), \"b\", to_tensor([3.0f32, 4.0f32]))\n  add(len(d), len(e))\n}\nn = dict_case(0i64)\n",
        eval: "n = 3\n",
    },
    Case {
        name: "tuple",
        body: "def tuple_case(seed: i64) -> tensor[1, f32] = {\n  pair = (to_tensor([1.0f32]), to_tensor([2.0f32]))\n  (a, b) = copy(pair)\n  (c, d) = pair\n  add(add(a, b), add(c, d))\n}\nresult = to_list(tuple_case(0i64))\n",
        eval: "result = [6.0]\n",
    },
    Case {
        name: "string",
        body: "def string_case(seed: i64) -> string = {\n  s = \"ab\"\n  string_concat(copy(s), s)\n}\nresult = string_case(0i64)\n",
        eval: "result = abab\n",
    },
    // A List of data values and its copy are independent: growing the copy
    // leaves the original's length, and consuming the original leaves the
    // copy's.
    Case {
        name: "list_of_adts_independent",
        body: "def independence(seed: i64) -> List[i64] = {\n  xs = [Lin { w: to_tensor([1.0f32, 2.0f32]) }, Lin { w: to_tensor([3.0f32, 4.0f32]) }]\n  ys = bump(copy(xs))\n  a = len(xs)\n  zs = bump(bump(xs))\n  [a, len(ys), len(zs)]\n}\nn = independence(0i64)\n",
        eval: "n = [2, 3, 4]\n",
    },
    // Replacing an element of a copied element's tensor, a consuming
    // operation that may reuse a uniquely owned buffer, leaves the original
    // element's tensor as it was.
    Case {
        name: "list_element_independent",
        body: "def element_independence(seed: i64) -> tensor[4, f32] = {\n  xs = [Lin { w: to_tensor([1.0f32, 2.0f32]) }, Lin { w: to_tensor([3.0f32, 4.0f32]) }]\n  ys = copy(xs)\n  poked = poke(index(ys, 0i64))\n  concat([weight(poked), weight(index(xs, 0i64))], 0i32)\n}\nresult = to_list(element_independence(0i64))\n",
        eval: "result = [100.0, 2.0, 1.0, 2.0]\n",
    },
    // The copy of a borrowed data-value parameter is an owned value, and the
    // caller keeps the original it lent.
    Case {
        name: "owned_from_borrowed_adt",
        body: "def owned_from_borrow(seed: i64) -> tensor[2, f32] = {\n  p = Lin { w: to_tensor([1.0f32, 2.0f32]) }\n  q = own(p)\n  add(weight(q), weight(p))\n}\nresult = to_list(owned_from_borrow(0i64))\n",
        eval: "result = [2.0, 4.0]\n",
    },
    // `copy(&x)` copies the referent of an explicit borrow.
    Case {
        name: "copy_of_a_borrowed_list",
        body: "def borrowed_copy(seed: i64) -> i64 = {\n  xs = rows(seed)\n  ys = grow(copy(&xs))\n  add(len(ys), len(xs))\n}\nn = borrowed_copy(0i64)\n",
        eval: "n = 5\n",
    },
    Case {
        name: "copy_of_a_borrowed_adt",
        body: "def borrowed_adt(seed: i64) -> tensor[2, f32] = {\n  p = Lin { w: to_tensor([1.0f32, 2.0f32]) }\n  q = copy(&p)\n  add(weight(q), weight(p))\n}\nresult = to_list(borrowed_adt(0i64))\n",
        eval: "result = [2.0, 4.0]\n",
    },
    // The copy of a function value is that value: a closure capturing a
    // tensor, a declaration, and a copied function as a transform target.
    Case {
        name: "copy_of_a_closure",
        body: "def clo(seed: i64) -> tensor[2, f32] = {\n  w = to_tensor([1.0f32, 2.0f32])\n  g = fn (x: tensor[2, f32]) -> add(x, w)\n  h = copy(g)\n  add(h(to_tensor([1.0f32, 1.0f32])), g(to_tensor([2.0f32, 2.0f32])))\n}\nresult = to_list(clo(0i64))\n",
        eval: "result = [5.0, 7.0]\n",
    },
    Case {
        name: "copy_of_a_declaration",
        body: "def twice_t(x: tensor[2, f32]) -> tensor[2, f32] = add(x, x)\ndef fv(seed: i64) -> tensor[2, f32] = {\n  h = copy(twice_t)\n  h(to_tensor([1.0f32, 1.0f32]))\n}\nresult = to_list(fv(0i64))\n",
        eval: "result = [2.0, 2.0]\n",
    },
    Case {
        name: "copied_function_as_a_grad_target",
        body: "def sq(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0i32))\ndef gq(seed: i64) -> tensor[2, f32] = {\n  scaled = fn (x: tensor[2, f32]) -> tensor_to_scalar(sum(mul(x, to_tensor([3.0f32, 3.0f32])), 0i32))\n  h = copy(scaled)\n  grad(h)(to_tensor([1.0f32, 2.0f32]))\n}\nresult = to_list(gq(0i64))\ndirect = to_list(grad(copy(sq))(to_tensor([1.0f32, 2.0f32])))\n",
        eval: "result = [3.0, 3.0]\ndirect = [2.0, 4.0]\n",
    },
    // School's use: a copied parameter record feeds a gradient, and the
    // original stays usable.
    Case {
        name: "adt_into_grad",
        body: "p = Lin { w: to_tensor([1.0f32, 2.0f32]) }\nx = to_tensor([3.0f32, 4.0f32])\ng = grad(loss, wrt=p)(copy(p), copy(x))\nl = loss(p, x)\n",
        eval: "p.w = tensor(shape=[2], data=[1.0, 2.0])\nx = tensor(shape=[2], data=[3.0, 4.0])\ng = Lin(tensor(shape=[2], data=[3.0, 4.0]))\nl = 11.0\n",
    },
    Case {
        name: "list_into_grad_and_concat",
        body: "ws = [to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])]\ng = grad(pair_loss)(copy(ws))\njoined = concat(copy(ws), 0i32)\nn = len(ws)\n",
        eval: "ws = [tensor(shape=[2], data=[1.0, 2.0]), tensor(shape=[2], data=[3.0, 4.0])]\ng = [tensor(shape=[2], data=[3.0, 4.0]), tensor(shape=[2], data=[1.0, 2.0])]\njoined = tensor(shape=[4], data=[1.0, 2.0, 3.0, 4.0])\nn = 2\n",
    },
];

/// Programs both lanes refuse, with the diagnostic each must name.
const REFUSED: &[(&str, &str)] = &[
    // [04-LIN-9]: a data value that carries a key is key-carrying.
    (
        "type Keyed =\n  | Keyed { k: key, n: i64 }\ndef key_adt(seed: i64) -> i64 = {\n  s = Keyed { k: key_from_seed(seed), n: 1i64 }\n  t = copy(s)\n  1i64\n}\nn = key_adt(0i64)\n",
        "key-carrying variable `s` cannot be copied",
    ),
    (
        "def key_case(seed: i64) -> i64 = {\n  k = key_from_seed(seed)\n  k2 = copy(k)\n  1i64\n}\nn = key_case(0i64)\n",
        "cannot be copied",
    ),
    (
        "def go[t](x: t) -> i32 = {\n  g = copy(x)\n  1i32\n}\nn = go(1i64)\n",
        "copy requires its operand's type to be determined at the call, because whether it \
         is a borrow decides the result (spec/04-type-system.md section 8.2), got `t`",
    ),
];

fn source(body: &str) -> String {
    format!("{PRELUDE}{body}")
}

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

fn evaluated(source: &str) -> String {
    let result = eval(request(source))
        .unwrap_or_else(|error| panic!("evaluator rejected the case: {error:?}"));
    result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}\n",
                root.name.as_deref().expect("named root"),
                root.display.as_deref().expect("rendered root")
            )
        })
        .collect()
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn check_positive(case: &Case) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| {
        let source = source(case.body);
        let expected = evaluated(&source);
        assert_eq!(expected, case.eval, "eval output");
        let generated = ownership_support::emit(&source, case.name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
    }))
    .map_err(|payload| {
        let head: String = panic_message(payload).chars().take(800).collect();
        format!("{}:\n{}\n  -> {head}", case.name, source(case.body))
    })
}

// REGRESSION TEST. On `114a818a3` the checker refused every positive program
// with "copy argument 1: expected tensor, got ...".
#[test]
fn copies_of_every_kind_compile_as_eval_runs_them() {
    let failures: Vec<String> = POSITIVE
        .iter()
        .filter_map(|case| check_positive(case).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn refused_copies_are_refused_on_both_lanes() {
    for &(body, message) in REFUSED {
        let source = source(body);
        let refused = eval(request(&source)).expect_err("the evaluator must refuse");
        let refused = format!("{refused:?}");
        assert!(refused.contains(message), "eval of {body}: {refused}");
        let refused = compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source,
            target: CompileTarget::C,
            entry_name: Some("fixture".into()),
        })
        .expect_err("the C build must refuse");
        let refused = format!("{refused:?}");
        assert!(refused.contains(message), "C build of {body}: {refused}");
    }
}
