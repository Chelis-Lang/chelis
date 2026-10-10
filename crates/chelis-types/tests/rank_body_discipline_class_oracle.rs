//! chelis#3383: the shape-identity and inert classes are checked by behaviour.
//!
//! `spec/04-type-system.md` section 4.5.3 admits a shape-identity builtin and
//! an inert builtin in a `..r` body on the strength of its `BUILTINS` row. A
//! builtin's scheme cannot vouch for that row: `relu`, `permute`, and
//! `reshape` share `&tv -> tv`, and the comparisons, `where`, and `clamp`
//! return a fresh variable whose shape only their checker arms fix. This
//! oracle derives a call for every Identity and Inert row from its scheme and
//! type-checks it on non-square `tensor[2, 3, D]` operands:
//!
//! - an Identity row must check with the declared result `tensor[2, 3, R]`
//!   for some dtypes, and be rejected with `tensor[3, 2, R]`, so its result is
//!   pinned to the operand shape in order;
//! - an Inert row must return unit (a tensor result is rejected), return a
//!   tensor operand's exact type (unit and the transposed shape are rejected),
//!   or never return (unit and both tensor shapes check).
//!
//! A row whose scheme yields no generatable call fails, rather than being
//! skipped.

use std::collections::BTreeSet;

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::types::{Prim, TensorPrec, Type, TypeVar};
use chelis_types::{BUILTINS, ShapeClass, builtin_env, check_ir_program};

const DTYPES: [&str; 3] = ["f32", "i32", "bool"];

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).unwrap_or_else(|error| panic!("parse {source}: {error:?}"));
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("generated probe must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn checks(source: &str) -> bool {
    check_ir_program(&surf_to_deep(source)).is_ok()
}

fn scalar(dtype: &str) -> &'static str {
    match dtype {
        "f32" => "0.5f32",
        "i32" => "1i32",
        _ => "true",
    }
}

fn prim_literal(prim: &Prim) -> Option<&'static str> {
    match prim {
        Prim::Key => Some("key_from_seed(1i64)"),
        Prim::String => Some("\"s\""),
        Prim::Bool => Some("true"),
        Prim::Int64 => Some("1i64"),
        Prim::Int32 => Some("1i32"),
        Prim::F32 => Some("0.5f32"),
        _ => None,
    }
}

/// One argument slot of a builtin's scheme.
enum Slot {
    Literal(&'static str),
    /// A tensor whose precision is the given variable's dtype.
    Tensor(TypeVar),
    /// A scalar of the given precision variable's dtype.
    Scalar(TypeVar),
    /// A whole-value variable: a `tensor[2, 3, D]` or a string.
    Operand(TypeVar),
}

fn tensor_precision_vars(ty: &Type, out: &mut BTreeSet<TypeVar>) {
    match ty {
        Type::Tensor(_, TensorPrec::Var(var)) => {
            out.insert(*var);
        }
        Type::Ref(inner) => tensor_precision_vars(inner, out),
        Type::Fn(params, ret) => {
            params
                .iter()
                .for_each(|param| tensor_precision_vars(param, out));
            tensor_precision_vars(ret, out);
        }
        _ => {}
    }
}

fn slots(name: &str) -> Vec<Slot> {
    let (env, _) = builtin_env();
    let scheme = env
        .lookup(name)
        .unwrap_or_else(|| panic!("`{name}` has a BUILTINS row but no scheme"));
    let Type::Fn(params, _) = &scheme.body else {
        panic!("`{name}`: scheme is not a function, so no call can be generated");
    };
    let mut precision = BTreeSet::new();
    tensor_precision_vars(&scheme.body, &mut precision);
    params
        .iter()
        .map(|param| {
            let param = match param {
                Type::Ref(inner) => inner.as_ref(),
                other => other,
            };
            match param {
                Type::Prim(prim) => Slot::Literal(prim_literal(prim).unwrap_or_else(|| {
                    panic!("`{name}`: no generatable literal for a {prim:?} argument")
                })),
                Type::Tensor(_, TensorPrec::Var(var)) => Slot::Tensor(*var),
                Type::Var(var) if precision.contains(var) => Slot::Scalar(*var),
                Type::Var(var) => Slot::Operand(*var),
                other => panic!("`{name}`: no generatable argument for {other:?}"),
            }
        })
        .collect()
}

/// A generated call: its parameter list, its argument list, and the declared
/// types of its tensor operands.
struct Call {
    params: Vec<String>,
    args: Vec<String>,
    operands: Vec<String>,
}

/// Every call the scheme admits, one per assignment of its variables: a
/// precision variable takes each dtype, an operand variable each
/// `tensor[2, 3, D]` and a string.
fn calls(name: &str) -> Vec<Call> {
    let slots = slots(name);
    let mut vars = Vec::new();
    for slot in &slots {
        let (Slot::Tensor(var) | Slot::Scalar(var) | Slot::Operand(var)) = slot else {
            continue;
        };
        if !vars.iter().any(|(seen, _)| seen == var) {
            let choices = if matches!(slot, Slot::Operand(_)) {
                4
            } else {
                3
            };
            vars.push((*var, choices));
        }
    }
    let total: usize = vars.iter().map(|(_, choices)| choices).product();
    (0..total)
        .map(|mut index| {
            let mut pick = Vec::new();
            for (var, choices) in &vars {
                pick.push((*var, index % choices));
                index /= choices;
            }
            let choice = |var: &TypeVar| pick.iter().find(|(seen, _)| seen == var).unwrap().1;
            let mut call = Call {
                params: Vec::new(),
                args: Vec::new(),
                operands: Vec::new(),
            };
            for slot in &slots {
                let tensor_dtype = match slot {
                    Slot::Literal(text) => {
                        call.args.push((*text).to_string());
                        None
                    }
                    Slot::Scalar(var) => {
                        call.args.push(scalar(DTYPES[choice(var)]).to_string());
                        None
                    }
                    Slot::Tensor(var) => Some(DTYPES[choice(var)]),
                    Slot::Operand(var) => match DTYPES.get(choice(var)) {
                        Some(dtype) => Some(*dtype),
                        None => {
                            call.args.push("\"s\"".to_string());
                            None
                        }
                    },
                };
                if let Some(dtype) = tensor_dtype {
                    let param = format!("a{}", call.params.len());
                    let ty = format!("tensor[2, 3, {dtype}]");
                    call.params.push(format!("{param}: {ty}"));
                    call.args.push(param);
                    call.operands.push(ty);
                }
            }
            call
        })
        .collect()
}

fn probe(name: &str, call: &Call, result: &str) -> String {
    format!(
        "def probe({}) -> {result} = {name}({})\n",
        call.params.join(", "),
        call.args.join(", ")
    )
}

fn identity_holds(name: &str) -> bool {
    calls(name).iter().any(|call| {
        !call.operands.is_empty()
            && DTYPES.iter().any(|result| {
                checks(&probe(name, call, &format!("tensor[2, 3, {result}]")))
                    && !checks(&probe(name, call, &format!("tensor[3, 2, {result}]")))
            })
    })
}

fn inert_holds(name: &str) -> bool {
    calls(name).iter().any(|call| {
        let unit = checks(&probe(name, call, "unit"));
        let tensor = checks(&probe(name, call, "tensor[2, 3, f32]"));
        let transposed = checks(&probe(name, call, "tensor[3, 2, i32]"));
        let returns_unit = unit && !tensor;
        let never_returns = unit && tensor && transposed;
        let returns_operand = call.operands.iter().any(|operand| {
            let transposed = operand.replace("[2, 3,", "[3, 2,");
            checks(&probe(name, call, operand)) && !unit && !checks(&probe(name, call, &transposed))
        });
        returns_unit || never_returns || returns_operand
    })
}

/// Every Identity and Inert row in the table, with no hand list.
#[test]
fn every_identity_and_inert_row_behaves_as_its_class_states() {
    let mut failures = Vec::new();
    let mut checked = 0;
    for decl in BUILTINS {
        let holds = match decl.shape_class {
            ShapeClass::Identity => identity_holds(decl.name),
            ShapeClass::Inert => inert_holds(decl.name),
            ShapeClass::NameTracked | ShapeClass::OrderedPrefix | ShapeClass::Untracked => continue,
        };
        checked += 1;
        if !holds {
            failures.push(format!("{} ({:?})", decl.name, decl.shape_class));
        }
    }
    assert!(checked > 0, "the table declared no Identity or Inert row");
    assert!(
        failures.is_empty(),
        "these rows do not behave as their declared class: {failures:?}"
    );
}
