//! [04-INF-9] (chelis#3149): the admitted set of builtin function values is
//! exactly the set whose scheme reproduces the direct-call rule.
//!
//! For every builtin, a generated witness set calls it once directly and once
//! through a library binding that holds nothing but the builtin's own scheme,
//! which is what a value of the builtin instantiates. The witnesses vary the
//! operand dtype, the declared result (its dtype and extent), and authored
//! generic binders, bounded and unbounded; each operand is also read again
//! after the call and the call is repeated, so consumption and borrowing are
//! compared, and the verdict is the type checker's and the linearity pass's
//! together. Effects are charged from one table for both routes
//! (`builtin_call_effect`; chelis-effects locks that per builtin). Two
//! properties follow:
//!
//! - an admitted builtin agrees with its direct call on every witness, so a
//!   value never checks a program its direct call refuses, or the reverse;
//! - a builtin whose scheme agrees on every witness, with both an accepted and
//!   a refused direct call among them, is admitted, so the predicate cannot
//!   refuse a builtin whose scheme already carries its rule.
//!
//! The admitted set is therefore never listed: removing `fold`'s or `scan`'s
//! result-origin relation, or declaring an incomplete scheme complete, fails
//! one of the two properties.

use std::collections::BTreeMap;

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;

use crate::builtins::{BUILTINS, builtin_env, builtin_value_contract_carried};
use crate::context::TypeEnv;
use crate::types::{Dim, Scheme, TensorPrec, Type};

const ROUTE: &str = "builtin_value_route";

#[derive(Clone, Copy)]
struct Pick {
    text: &'static str,
    binder: Option<&'static str>,
}

const fn pick(text: &'static str) -> Pick {
    Pick { text, binder: None }
}

/// Candidates for a type variable that stands for a whole operand.
const WHOLE: &[Pick] = &[
    pick("tensor[3, f32]"),
    pick("tensor[3, i32]"),
    pick("tensor[3, i64]"),
    pick("tensor[2, f32]"),
    pick("f32"),
    pick("i32"),
    pick("i64"),
    pick("bool"),
    pick("string"),
    pick("key"),
    pick("tensor[3, key]"),
    pick("List[i64]"),
    Pick {
        text: "tensor[3, p]",
        binder: Some("p: Float"),
    },
    Pick {
        text: "tensor[3, q]",
        binder: Some("q"),
    },
];

/// Candidates for a tensor's precision variable.
const PRECISION: &[Pick] = &[
    pick("f32"),
    pick("i32"),
    pick("bool"),
    Pick {
        text: "p",
        binder: Some("p: Float"),
    },
    Pick {
        text: "q",
        binder: Some("q"),
    },
];

fn render(ty: &Type, picks: &BTreeMap<u32, Pick>, nested: bool) -> Option<String> {
    let list = |types: &[Type]| {
        types
            .iter()
            .map(|ty| render(ty, picks, false))
            .collect::<Option<Vec<_>>>()
            .map(|parts| parts.join(", "))
    };
    Some(match ty {
        Type::Prim(prim) => prim.name().to_string(),
        Type::Unit => "unit".to_string(),
        Type::Ref(inner) => format!("&{}", render(inner, picks, true)?),
        Type::Var(var) => picks.get(&var.0)?.text.to_string(),
        Type::Adt(name, args) if args.is_empty() => name.clone(),
        Type::Adt(name, args) => format!("{name}[{}]", list(args)?),
        Type::Tuple(items) => format!("({})", list(items)?),
        Type::Tensor(dims, precision) => {
            let mut parts: Vec<String> = dims
                .iter()
                .map(|dim| match dim {
                    Dim::Lit(extent) => extent.to_string(),
                    Dim::Name(name) => name.clone(),
                    _ => "3".to_string(),
                })
                .collect();
            parts.push(match precision {
                TensorPrec::Concrete(prim) => prim.name().to_string(),
                TensorPrec::Var(var) => picks.get(&var.0)?.text.to_string(),
            });
            format!("tensor[{}]", parts.join(", "))
        }
        Type::Fn(params, result) => {
            let mut parts = params
                .iter()
                .map(|ty| render(ty, picks, true))
                .collect::<Option<Vec<_>>>()?;
            if parts.is_empty() {
                parts.push("unit".to_string());
            }
            parts.push(render(result, picks, true)?);
            let arrow = parts.join(" -> ");
            if nested { format!("({arrow})") } else { arrow }
        }
        _ => return None,
    })
}

fn collect_vars(ty: &Type, whole: &mut Vec<u32>, precision: &mut Vec<u32>) {
    match ty {
        Type::Var(var) if !whole.contains(&var.0) => whole.push(var.0),
        Type::Ref(inner) => collect_vars(inner, whole, precision),
        Type::Adt(_, items) | Type::Tuple(items) => {
            items
                .iter()
                .for_each(|ty| collect_vars(ty, whole, precision));
        }
        Type::Fn(params, result) => {
            params
                .iter()
                .for_each(|ty| collect_vars(ty, whole, precision));
            collect_vars(result, whole, precision);
        }
        Type::Tensor(_, TensorPrec::Var(var)) if !precision.contains(&var.0) => {
            precision.push(var.0);
        }
        _ => {}
    }
}

/// Candidates combined jointly when a relation, not a parameter, fixes the
/// result: a `len` witness needs `List[i64]` and `i64` together.
const JOINT: &[Pick] = &[
    pick("tensor[3, f32]"),
    pick("i64"),
    pick("key"),
    pick("tensor[3, key]"),
    pick("List[i64]"),
    pick("(key, key)"),
];

/// Every type variable takes each candidate in turn while the others hold a
/// fixed first candidate, plus the diagonal in which all take the same one.
/// When the result mentions a variable no parameter does, the variables also
/// range jointly over a small pool, so a relation-carried result can be met.
fn assignments(params: &[Type], result: &Type) -> Vec<BTreeMap<u32, Pick>> {
    let (mut whole, mut precision) = (Vec::new(), Vec::new());
    params
        .iter()
        .for_each(|ty| collect_vars(ty, &mut whole, &mut precision));
    let fixed_by_params = whole.clone();
    collect_vars(result, &mut whole, &mut precision);
    let result_only = whole.iter().any(|var| !fixed_by_params.contains(var));
    precision.retain(|var| !whole.contains(var));
    let slots: Vec<(u32, &[Pick])> = whole
        .iter()
        .map(|var| (*var, WHOLE))
        .chain(precision.iter().map(|var| (*var, PRECISION)))
        .collect();
    let base: BTreeMap<u32, Pick> = slots.iter().map(|(var, pool)| (*var, pool[0])).collect();
    let mut out = vec![base.clone()];
    for index in 0..WHOLE.len() {
        out.push(
            slots
                .iter()
                .map(|(var, pool)| (*var, pool[index % pool.len()]))
                .collect(),
        );
    }
    for (var, pool) in &slots {
        for candidate in *pool {
            let mut picks = base.clone();
            picks.insert(*var, *candidate);
            out.push(picks);
        }
    }
    if result_only && whole.len() <= 3 {
        let mut joint = vec![base.clone()];
        for var in &whole {
            joint = joint
                .into_iter()
                .flat_map(|picks| {
                    JOINT.iter().map(move |candidate| {
                        let mut picks = picks.clone();
                        picks.insert(*var, *candidate);
                        picks
                    })
                })
                .collect();
        }
        out.extend(joint);
    }
    out
}

fn declared_results(result: &str) -> Vec<String> {
    let mut out = vec![result.to_string()];
    for (from, to) in [
        ("f32", "f64"),
        ("i32", "i64"),
        ("i64", "i32"),
        ("bool", "i32"),
        ("tensor[3", "tensor[4"),
        ("string", "bool"),
    ] {
        if result.contains(from) {
            out.push(result.replacen(from, to, 1));
        }
    }
    if result != "bool" {
        out.push("bool".to_string());
    }
    out.sort();
    out.dedup();
    out
}

fn route_env(scheme: &Scheme) -> TypeEnv {
    TypeEnv::empty().with_test_binding(ROUTE, scheme.clone())
}

/// The checker's whole per-program verdict: types, then the linearity pass
/// that owns consumption, borrowing and use after a consuming call.
fn verdict(env: &TypeEnv, source: &str) -> Option<bool> {
    let declarations = parse_str(source).ok()?;
    let deep = desugar_program(&declarations).ok()?;
    Some(
        crate::check_ir_with_context(env, &deep)
            .is_ok_and(|checked| crate::check_linearity(&checked).is_ok()),
    )
}

/// The witness block lines for one instantiation, as `(declared result,
/// lines)` with `CALL` standing for the application. Beside the lone call,
/// each operand is read again after the call, and the call is repeated: a rule
/// about what an operand may do after the call (a consume, a borrow) is
/// invisible to a lone call.
fn witness_bodies(result_type: &str, param_types: &[String]) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = declared_results(result_type)
        .into_iter()
        .map(|declared| (declared, vec!["CALL".to_string()]))
        .collect();
    for (index, ty) in param_types.iter().enumerate() {
        out.push((
            ty.trim_start_matches('&').to_string(),
            vec!["_ = CALL".to_string(), format!("a{index}")],
        ));
    }
    out.push((
        result_type.to_string(),
        vec!["_ = CALL".to_string(), "CALL".to_string()],
    ));
    out
}

/// A function whose block runs `lines`, preceded by `prelude` when given.
fn block(header: &str, prelude: Option<&str>, lines: &[String], call: &str) -> String {
    let mut body: Vec<String> = prelude.map(str::to_string).into_iter().collect();
    body.extend(lines.iter().map(|line| line.replace("CALL", call)));
    match body.as_slice() {
        [only] => format!("{header} = {only}\n"),
        _ => format!("{header} = {{\n  {}\n}}\n", body.join("\n  ")),
    }
}

struct Comparison {
    accepted_directly: usize,
    refused_directly: usize,
    disagreements: Vec<String>,
}

/// An admitted builtin is compared through a real alias, `op = NAME`, which
/// is the route a program takes, key judgement ([04-LIN-9]) included. A
/// refused builtin has no such route, so the comparison binds its bare scheme
/// under a name no builtin route claims; that stops at the first
/// disagreement, which settles it.
fn compare(name: &str, scheme: &Scheme, admitted: bool) -> Comparison {
    let stop_at_disagreement = !admitted;
    let env = route_env(scheme);
    let mut comparison = Comparison {
        accepted_directly: 0,
        refused_directly: 0,
        disagreements: Vec::new(),
    };
    let Type::Fn(params, result) = &scheme.body else {
        return comparison;
    };
    for picks in assignments(params, result) {
        let Some(param_types) = params
            .iter()
            .map(|ty| render(ty, &picks, false))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let Some(result_type) = render(result, &picks, false) else {
            continue;
        };
        let mut binders: Vec<&str> = picks.values().filter_map(|pick| pick.binder).collect();
        binders.sort_unstable();
        binders.dedup();
        let binders = if binders.is_empty() {
            String::new()
        } else {
            format!("[{}]", binders.join(", "))
        };
        let signature: Vec<String> = param_types
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("a{index}: {ty}"))
            .collect();
        let arguments: Vec<String> = (0..param_types.len()).map(|i| format!("a{i}")).collect();
        let call = arguments.join(", ");
        for (declared, lines) in witness_bodies(&result_type, &param_types) {
            let header = format!("def g{binders}({}) -> {declared}", signature.join(", "));
            let direct = block(&header, None, &lines, &format!("{name}({call})"));
            let through_value = if admitted {
                block(
                    &header,
                    Some(&format!("op = {name}")),
                    &lines,
                    &format!("op({call})"),
                )
            } else {
                block(&header, None, &lines, &format!("{ROUTE}({call})"))
            };
            let (Some(direct_ok), Some(value_ok)) =
                (verdict(&env, &direct), verdict(&env, &through_value))
            else {
                continue;
            };
            if direct_ok {
                comparison.accepted_directly += 1;
            } else {
                comparison.refused_directly += 1;
            }
            // [04-LIN-10]: a generic value's type variables are key-free, so
            // a value may refuse a key its direct call admits. Only that
            // refusal is expected; every other difference is a disagreement.
            let key_freedom = direct_ok && !value_ok && direct.contains("key");
            if direct_ok != value_ok && !key_freedom {
                comparison.disagreements.push(format!(
                    "direct={direct_ok} value={value_ok}: {}",
                    direct.trim()
                ));
                if stop_at_disagreement {
                    return comparison;
                }
            }
        }
    }
    comparison
}

#[test]
fn admitted_builtin_values_are_exactly_the_schemes_that_reproduce_the_direct_rule() {
    let (env, _) = builtin_env();
    let mut failures = Vec::new();
    for decl in BUILTINS {
        let scheme = env
            .lookup(decl.name)
            .expect("registry builtin has a scheme");
        let admitted = builtin_value_contract_carried(decl, scheme);
        let comparison = compare(decl.name, scheme, admitted);
        let covered = comparison.accepted_directly > 0 && comparison.refused_directly > 0;
        let reproduces = covered && comparison.disagreements.is_empty();
        if admitted && !comparison.disagreements.is_empty() {
            failures.push(format!(
                "`{}` is admitted as a value but its scheme disagrees with its direct call: {:?}",
                decl.name,
                &comparison.disagreements[..comparison.disagreements.len().min(2)]
            ));
        }
        if admitted && !covered {
            failures.push(format!(
                "`{}` is admitted as a value but no witness both accepts and refuses a direct \
                 call ({} accepted, {} refused)",
                decl.name, comparison.accepted_directly, comparison.refused_directly
            ));
        }
        if !admitted && reproduces {
            failures.push(format!(
                "`{}` is refused as a value although its scheme reproduces its direct call on \
                 every witness",
                decl.name
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
