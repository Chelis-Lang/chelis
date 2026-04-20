use std::collections::{HashMap, HashSet};

use chelis_deep::ast::{Atom, Expr, List};
use chelis_types::{BUILTIN_NAMES, CheckedProgram};

use crate::dag::TensorType;
use crate::lower::{expr_is_dag_lowerable, lower_program, top_level_lowering_map};

#[derive(Debug, Clone)]
pub struct CompiledProgram {
    pub dag: Option<crate::Dag>,
    pub host: Option<HostProgram>,
}

#[derive(Debug, Clone, Default)]
pub struct HostProgram {
    pub globals: Vec<HostBinding>,
    pub global_tensor_helpers: Vec<HostTensorHelper>,
    pub functions: Vec<HostFunction>,
}

#[derive(Debug, Clone)]
pub struct HostBinding {
    pub name: String,
    pub display_name: Option<String>,
    pub ty: HostType,
    pub value: HostExpr,
}

#[derive(Debug, Clone)]
pub struct HostFunction {
    pub name: String,
    pub params: Vec<HostParam>,
    pub ret_ty: HostType,
    pub body: HostExpr,
    pub tensor_helpers: Vec<HostTensorHelper>,
}

#[derive(Debug, Clone)]
pub struct HostParam {
    pub name: String,
    pub ty: HostType,
}

#[derive(Debug, Clone)]
pub struct HostTensorHelper {
    pub name: String,
    pub dag: crate::Dag,
    pub inputs: Vec<HostTensorInput>,
    pub output: TensorType,
}

#[derive(Debug, Clone)]
pub struct HostTensorInput {
    pub name: String,
    pub ty: TensorType,
}

#[derive(Debug, Clone)]
pub struct HostCallback {
    pub kind: HostCallbackKind,
    pub ret_ty: HostType,
}

#[derive(Debug, Clone)]
pub struct HostMatchArm {
    pub ctor: String,
    pub bindings: Vec<HostPatternBinding>,
    pub expr: HostExpr,
}

#[derive(Debug, Clone)]
pub struct HostPatternBinding {
    pub name: String,
    pub ty: HostType,
    pub field_index: usize,
}

#[derive(Debug, Clone)]
pub struct HostAdtField {
    pub name: Option<String>,
    pub ty: HostType,
}

#[derive(Debug, Clone)]
pub enum HostCallbackKind {
    Named {
        function: String,
        params: Vec<HostParam>,
    },
    Inline {
        params: Vec<HostParam>,
        body: Box<HostExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum HostType {
    Int64,
    Float64,
    Bool,
    String,
    Fn(Vec<HostType>, Box<HostType>),
    Adt(String),
    List(Box<HostType>),
    Dict(Box<HostType>, Box<HostType>),
    Tuple(Vec<HostType>),
    Tensor(TensorType),
    Option(Box<HostType>),
    MappedFile,
    Unit,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum HostExpr {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    List(Vec<HostExpr>, HostType),
    Tuple(Vec<HostExpr>, HostType),
    Var(String, HostType),
    Call {
        function: String,
        args: Vec<HostExpr>,
        arg_tys: Vec<HostType>,
        ty: HostType,
    },
    Builtin {
        name: String,
        args: Vec<HostExpr>,
        ty: HostType,
    },
    AdtConstruct {
        ctor: String,
        fields: Vec<HostExpr>,
        ty: HostType,
    },
    AdtFieldAccess {
        base: Box<HostExpr>,
        field_index: usize,
        ty: HostType,
    },
    If {
        cond: Box<HostExpr>,
        then_expr: Box<HostExpr>,
        else_expr: Box<HostExpr>,
        ty: HostType,
    },
    MatchOption {
        scrutinee: Box<HostExpr>,
        bind_name: String,
        some_expr: Box<HostExpr>,
        none_expr: Box<HostExpr>,
        ty: HostType,
    },
    MatchAdt {
        scrutinee: Box<HostExpr>,
        arms: Vec<HostMatchArm>,
        default_expr: Option<Box<HostExpr>>,
        ty: HostType,
    },
    Let {
        bindings: Vec<HostBinding>,
        body: Box<HostExpr>,
        ty: HostType,
    },
    Map {
        callback: HostCallback,
        list: Box<HostExpr>,
        ty: HostType,
    },
    Filter {
        callback: HostCallback,
        list: Box<HostExpr>,
        ty: HostType,
    },
    Fold {
        callback: HostCallback,
        init: Box<HostExpr>,
        list: Box<HostExpr>,
        ty: HostType,
    },
    Scan {
        callback: HostCallback,
        init: Box<HostExpr>,
        list: Box<HostExpr>,
        ty: HostType,
    },
    Partition {
        callback: HostCallback,
        list: Box<HostExpr>,
        ty: HostType,
    },
    FlatMap {
        callback: HostCallback,
        list: Box<HostExpr>,
        ty: HostType,
    },
    TensorCall {
        helper: usize,
        args: Vec<HostExpr>,
        ty: HostType,
    },
    Unit,
}

pub fn lower_compiled_program(program: &CheckedProgram) -> CompiledProgram {
    let lowered_names = top_level_lowering_map(program.exprs(), program.type_env());
    let dag = lower_program(program);
    let host = lower_host_program(program, &lowered_names);

    CompiledProgram {
        dag: (!dag.roots().is_empty()).then_some(dag),
        host: if host.globals.is_empty() && host.functions.is_empty() {
            None
        } else {
            Some(host)
        },
    }
}

pub fn host_program_requires_host_backend(program: &HostProgram) -> bool {
    if !program.globals.is_empty() {
        return true;
    }

    let tensor_only_functions = program
        .functions
        .iter()
        .filter(|function| {
            matches!(function.ret_ty, HostType::Tensor(_))
                && function
                    .params
                    .iter()
                    .all(|param| matches!(param.ty, HostType::Tensor(_)))
        })
        .map(|function| function.name.clone())
        .collect::<HashSet<_>>();

    program.functions.iter().any(|function| {
        !tensor_only_functions.contains(&function.name)
            || !host_expr_stays_on_tensor_path(&function.body, &tensor_only_functions)
    })
}

fn host_expr_stays_on_tensor_path(
    expr: &HostExpr,
    tensor_only_functions: &HashSet<String>,
) -> bool {
    match expr {
        HostExpr::Var(_, HostType::Tensor(_)) => true,
        HostExpr::TensorCall { .. } => true,
        HostExpr::Call {
            function,
            args,
            arg_tys,
            ty,
        } => {
            matches!(ty, HostType::Tensor(_))
                && tensor_only_functions.contains(function)
                && arg_tys.iter().all(|ty| matches!(ty, HostType::Tensor(_)))
                && args
                    .iter()
                    .all(|arg| host_expr_stays_on_tensor_path(arg, tensor_only_functions))
        }
        HostExpr::Let { bindings, body, ty } => {
            matches!(ty, HostType::Tensor(_))
                && bindings.iter().all(|binding| {
                    matches!(binding.ty, HostType::Tensor(_))
                        && host_expr_stays_on_tensor_path(&binding.value, tensor_only_functions)
                })
                && host_expr_stays_on_tensor_path(body, tensor_only_functions)
        }
        _ => false,
    }
}

pub fn preferred_tensor_entry_name(program: &HostProgram) -> Option<&str> {
    fn tensor_signature(function: &HostFunction) -> bool {
        matches!(function.ret_ty, HostType::Tensor(_))
            && function
                .params
                .iter()
                .all(|param| matches!(param.ty, HostType::Tensor(_)))
    }

    if let Some(function) = program
        .functions
        .iter()
        .find(|function| function.name == "main" && tensor_signature(function))
    {
        return Some(function.name.as_str());
    }

    program
        .functions
        .iter()
        .rev()
        .find(|function| tensor_signature(function))
        .map(|function| function.name.as_str())
}

pub fn lower_named_tensor_entry_dag(program: &CheckedProgram, name: &str) -> Option<crate::Dag> {
    let defs = collect_program_defs(program.exprs());
    let body = lookup_program_def(&defs, name)?.clone();
    let Expr::List(list, _) = &body else {
        return None;
    };
    if tag(list) != Some("fn") {
        return None;
    }

    let kids = children(list);
    let params_list = kids.first().and_then(as_list)?;
    if tag(params_list) != Some("params") {
        return None;
    }

    let declared_param_tys = lookup_declared_type_expr(program, name)
        .and_then(parse_fn_type_expr)
        .map(|(params, _)| params)
        .unwrap_or_default();
    let mut scope = HashMap::new();
    for (index, param) in children(params_list).iter().enumerate() {
        let pname = param_name(param)?;
        let pty = param_host_type(param)
            .or_else(|| declared_param_tys.get(index).cloned())
            .filter(|ty| *ty != HostType::Unknown)?;
        let HostType::Tensor(tensor_ty) = pty else {
            return None;
        };
        scope.insert(pname, tensor_ty);
    }

    let body_expr = kids.get(1)?;
    Some(crate::lower::lower_subexpr_program(
        body_expr,
        scope,
        program.type_env().clone(),
        defs,
    ))
}

fn lower_host_program(
    program: &CheckedProgram,
    lowered_names: &HashMap<String, bool>,
) -> HostProgram {
    let mut host = HostProgram::default();
    let mut global_scope = HashMap::new();
    // Count pure-tensor `fn`-body top-level defs in the program. When there
    // is more than one, the legacy DAG-only path would collapse them into a
    // single file-named entry point that drops all but one def's parameters
    // (Nautilus Bug 3c). In that case we emit a host wrapper per def so each
    // gets its own C symbol.
    let lowered_fn_def_count = top_level_items(program.exprs())
        .iter()
        .filter(|expr| {
            let Expr::List(list, _) = expr else {
                return false;
            };
            if tag(list) != Some("def") {
                return false;
            }
            let kids = children(list);
            let Some(def_name) = kids.first().and_then(symbol_name) else {
                return false;
            };
            let is_fn_body = matches!(kids.get(1), Some(Expr::List(body_list, _)) if tag(body_list) == Some("fn"));
            is_fn_body && lowered_names.get(def_name).copied().unwrap_or(false)
        })
        .count();
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("def") {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        let ty_expr = lookup_declared_type_expr(program, name);
        // Pure-tensor top-level function defs are normally lowered to the
        // DAG. But when the program also has host-lane bindings (i.e. some
        // def is NOT DAG-lowerable), downstream host-lane callers still
        // need a real C function symbol for the wrapper. In that case,
        // emit a HostFunction wrapper alongside the DAG lowering.
        //
        // A single pure-tensor function def can still use the legacy
        // DAG-only path (the file-named entry point wraps it 1:1 with the
        // correct signature). But when there are multiple pure-tensor
        // function defs in the same program, the DAG path would collapse
        // them into a single file-named entry that silently drops all but
        // one def's parameters and outputs (Nautilus Bug 3c). Emit a host
        // wrapper per def in that case so each gets its own C symbol.
        let is_fn_body = matches!(body, Expr::List(list, _) if tag(list) == Some("fn"));
        let has_any_host_lane_def = lowered_names.values().any(|lowered| !*lowered);
        let needs_host_wrapper = is_fn_body && (has_any_host_lane_def || lowered_fn_def_count > 1);
        let skip_for_lowered =
            lowered_names.get(name).copied().unwrap_or(false) && !needs_host_wrapper;
        if skip_for_lowered {
            continue;
        }
        if let Some(function) = lower_host_function(name, body, ty_expr, program) {
            host.functions.push(function);
        } else {
            let value = lower_host_expr(
                body,
                program,
                &global_scope,
                &mut host.global_tensor_helpers,
            );
            let ty = host_expr_type(&value);
            host.globals.push(HostBinding {
                name: name.to_string(),
                display_name: None,
                ty,
                value: value.clone(),
            });
            global_scope.insert(name.to_string(), host_expr_type(&value));
        }
    }
    loop {
        let mut changed = false;
        changed |= refine_host_function_signatures(&mut host.functions);
        changed |= refine_host_globals(&mut host.globals, &host.functions);
        changed |= propagate_named_callback_signatures(&mut host.functions, &host.globals);
        if !changed {
            break;
        }
    }
    host
}

fn lower_host_function(
    name: &str,
    body: &Expr,
    ty_expr: Option<&Expr>,
    program: &CheckedProgram,
) -> Option<HostFunction> {
    let declared_fn_type_expr = ty_expr
        .cloned()
        .or_else(|| lookup_declared_type_expr(program, name).cloned());
    let fn_type_parts = declared_fn_type_expr
        .as_ref()
        .and_then(parse_fn_type_expr_parts);
    let (param_tys, ret_ty) = ty_expr
        .and_then(parse_fn_type_expr)
        .or_else(|| expr_fn_type(body))
        .or_else(|| lookup_declared_fn_type(program, name))
        .unwrap_or((Vec::new(), HostType::Unknown));

    let mut scope = HashMap::new();
    let mut params = Vec::new();
    let mut tensor_helpers = Vec::new();
    let body_expr = if let Expr::List(list, _) = body {
        if tag(list) == Some("fn") {
            let kids = children(list);
            let params_list = kids.first().and_then(as_list)?;
            if tag(params_list) != Some("params") {
                return None;
            }
            for (index, param) in children(params_list).iter().enumerate() {
                let Some(pname) = param_name(param) else {
                    continue;
                };
                let pty = param_host_type(param)
                    .filter(|ty| *ty != HostType::Unknown)
                    .or_else(|| {
                        param_tys
                            .get(index)
                            .cloned()
                            .filter(|ty| *ty != HostType::Unknown)
                    })
                    .unwrap_or(HostType::Unknown);
                scope.insert(pname.clone(), pty.clone());
                params.push(HostParam {
                    name: pname,
                    ty: pty,
                });
            }
            kids.get(1)?.clone()
        } else {
            if param_tys.is_empty() && ret_ty == HostType::Unknown {
                return None;
            }
            for (index, param_ty) in param_tys.iter().enumerate() {
                let pname = format!("arg{index}");
                scope.insert(pname.clone(), param_ty.clone());
                params.push(HostParam {
                    name: pname,
                    ty: param_ty.clone(),
                });
            }
            synthesize_callable_application(
                body,
                &params,
                fn_type_parts.as_ref().map(|(params, _)| params.as_slice()),
                fn_type_parts.as_ref().map(|(_, ret)| ret),
            )
        }
    } else {
        return None;
    };
    // If the declared return type is a tensor, the body must produce a
    // tensor even when downstream type-metadata annotations are missing
    // from the reef'd deep AST. Force the body through the tensor-helper
    // path in that case so that pure-tensor wrapper defs like
    // `Std.Tensor.Reduce.min` get a real C function symbol rather than a
    // fallthrough `HostExpr::Builtin` with an "unsupported builtin"
    // placeholder (Phase 3j-pre Batch 5b bug 4).
    let callable_body_is_synthetic =
        !matches!(body, Expr::List(list, _) if tag(list) == Some("fn"));
    let host_body = if let HostType::Tensor(expected) = ret_ty.clone()
        && !should_keep_tensor_expr_in_host_lane(&body_expr)
        && (callable_body_is_synthetic || expr_is_dag_lowerable(&body_expr, program))
    {
        lower_tensor_helper_call(&body_expr, program, &scope, &mut tensor_helpers, expected)
    } else {
        lower_host_expr(&body_expr, program, &scope, &mut tensor_helpers)
    };
    refine_function_params_from_body(&mut params, &host_body);
    let ret_ty = if ret_ty == HostType::Unknown {
        host_expr_type(&host_body)
    } else {
        ret_ty
    };
    Some(HostFunction {
        name: name.to_string(),
        params,
        ret_ty,
        body: host_body,
        tensor_helpers,
    })
}

fn synthesize_callable_application(
    body: &Expr,
    params: &[HostParam],
    param_type_exprs: Option<&[Expr]>,
    ret_type_expr: Option<&Expr>,
) -> Expr {
    let span = body.span();
    let mut elements = vec![
        Expr::Atom(Atom::Symbol("app".to_string()), span),
        Expr::Map(
            chelis_deep::ast::MetaMap {
                entries: ret_type_expr
                    .cloned()
                    .map(|ret| vec![("type".to_string(), ret)])
                    .unwrap_or_default(),
            },
            span,
        ),
        body.clone(),
    ];
    for (index, param) in params.iter().enumerate() {
        let var_meta = chelis_deep::ast::MetaMap {
            entries: param_type_exprs
                .and_then(|tys| tys.get(index))
                .cloned()
                .map(|ty| vec![("type".to_string(), ty)])
                .unwrap_or_default(),
        };
        elements.push(Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Symbol("var".to_string()), span),
                    Expr::Map(var_meta, span),
                    Expr::Atom(Atom::Symbol(param.name.clone()), span),
                ],
            },
            span,
        ));
    }
    Expr::List(List { elements }, span)
}

/// Force-lower an expression through the tensor-helper path using an
/// explicit expected tensor type hint. This is used by
/// `lower_host_function` so that pure-tensor wrapper function bodies get a
/// real C function definition even when downstream type metadata is
/// missing on the reef'd deep AST's `app` nodes.
fn lower_tensor_helper_call(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
    expected: TensorType,
) -> HostExpr {
    if let Expr::List(list, _) = expr
        && tag(list) == Some("var")
        && let Some(name) = children(list).first().and_then(symbol_name)
    {
        return HostExpr::Var(name.to_string(), HostType::Tensor(expected));
    }
    let helper_index = tensor_helpers.len();
    let helper_name = format!("__host_tensor_helper_{helper_index}");
    let dag = crate::lower::lower_subexpr_program(
        expr,
        collect_tensor_scope(scope),
        program.type_env().clone(),
        collect_program_defs(program.exprs()),
    );
    let dag = remap_tensor_helper_dim_symbols(&dag, scope, &expected);
    let inputs = tensor_helper_inputs(&dag);
    let output = dag
        .roots()
        .first()
        .and_then(|id| dag.get(*id))
        .map(|node| node.output_type.clone())
        .unwrap_or_else(|| expected.clone());
    let args = tensor_helper_args(&inputs, scope);
    tensor_helpers.push(HostTensorHelper {
        name: helper_name,
        dag,
        inputs,
        output,
    });
    HostExpr::TensorCall {
        helper: helper_index,
        args,
        ty: HostType::Tensor(expected),
    }
}

fn lower_host_expr(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    if let Some(tensor_ty) = expr_tensor_type(expr, program, scope)
        && expr_is_dag_lowerable(expr, program)
        && !should_keep_tensor_expr_in_host_lane(expr)
    {
        if let Expr::List(list, _) = expr
            && tag(list) == Some("var")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            return HostExpr::Var(name.to_string(), HostType::Tensor(tensor_ty));
        }
        let helper_index = tensor_helpers.len();
        let helper_name = format!("__host_tensor_helper_{helper_index}");
        let dag = crate::lower::lower_subexpr_program(
            expr,
            collect_tensor_scope(scope),
            program.type_env().clone(),
            collect_program_defs(program.exprs()),
        );
        let dag = remap_tensor_helper_dim_symbols(&dag, scope, &tensor_ty);
        let inputs = tensor_helper_inputs(&dag);
        let output = dag
            .roots()
            .first()
            .and_then(|id| dag.get(*id))
            .map(|node| node.output_type.clone())
            .unwrap_or_else(|| tensor_ty.clone());
        let args = tensor_helper_args(&inputs, scope);
        tensor_helpers.push(HostTensorHelper {
            name: helper_name,
            dag,
            inputs,
            output,
        });
        return HostExpr::TensorCall {
            helper: helper_index,
            args,
            ty: HostType::Tensor(tensor_ty),
        };
    }

    match expr {
        Expr::Atom(Atom::Int(value), _) => HostExpr::Int(*value),
        Expr::Atom(Atom::Float(value), _) => HostExpr::Float(*value),
        Expr::Atom(Atom::Bool(value), _) => HostExpr::Bool(*value),
        Expr::Atom(Atom::Str(value), _) => HostExpr::String(value.clone()),
        Expr::List(list, _) if tag(list) == Some("tuple") => {
            let items = children(list)
                .iter()
                .map(|child| lower_host_expr(child, program, scope, tensor_helpers))
                .collect::<Vec<_>>();
            let ty = expr_host_type(expr, program, scope);
            let ty = if ty == HostType::Unknown {
                HostType::Tuple(items.iter().map(host_expr_type).collect())
            } else {
                ty
            };
            HostExpr::Tuple(items, ty)
        }
        Expr::List(list, _) if tag(list) == Some("record") => {
            lower_record_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("lit") => lower_host_expr(
            children(list).first().unwrap_or(expr),
            program,
            scope,
            tensor_helpers,
        ),
        Expr::List(list, _) if tag(list) == Some("var") => {
            let name = children(list)
                .first()
                .and_then(symbol_name)
                .unwrap_or("_")
                .to_string();
            let ty = expr_type(expr)
                .filter(|ty| *ty != HostType::Unknown)
                .or_else(|| scope.get(&name).cloned())
                .or_else(|| lookup_declared_host_type(program, &name))
                .unwrap_or(HostType::Unknown);
            if name == "Nil" {
                return HostExpr::List(
                    Vec::new(),
                    match ty {
                        HostType::List(_) => ty,
                        _ => HostType::List(Box::new(HostType::Unknown)),
                    },
                );
            }
            if let Some((adt_name, fields)) = lookup_adt_ctor(program, &name)
                && fields.is_empty()
            {
                return HostExpr::AdtConstruct {
                    ctor: name,
                    fields: Vec::new(),
                    ty: HostType::Adt(adt_name),
                };
            }
            HostExpr::Var(name, ty)
        }
        Expr::List(list, _) if tag(list) == Some("if") => {
            let kids = children(list);
            let then_expr = lower_host_expr(&kids[1], program, scope, tensor_helpers);
            let else_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
            let explicit_ty = expr_host_type(expr, program, scope);
            let ty = if explicit_ty == HostType::Unknown {
                let then_ty = host_expr_type(&then_expr);
                if then_ty == HostType::Unknown {
                    host_expr_type(&else_expr)
                } else {
                    then_ty
                }
            } else {
                explicit_ty
            };
            HostExpr::If {
                cond: Box::new(lower_host_expr(&kids[0], program, scope, tensor_helpers)),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
                ty,
            }
        }
        Expr::List(list, _) if tag(list) == Some("match") => {
            lower_match_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("let") => {
            let kids = children(list);
            let mut scoped = scope.clone();
            let mut bindings = Vec::new();
            if let Some(bind_list) = kids.first().and_then(as_list)
                && tag(bind_list) == Some("bind")
            {
                let bind_children = children(bind_list);
                let mut index = 0;
                while index + 1 < bind_children.len() {
                    if let Some(name) = symbol_name(&bind_children[index]) {
                        let value = lower_host_expr(
                            &bind_children[index + 1],
                            program,
                            &scoped,
                            tensor_helpers,
                        );
                        let bind_ty = host_expr_type(&value);
                        bindings.push(HostBinding {
                            name: name.to_string(),
                            display_name: None,
                            ty: bind_ty.clone(),
                            value,
                        });
                        scoped.insert(name.to_string(), bind_ty);
                    }
                    index += 2;
                }
            }
            let body = kids
                .get(1)
                .map(|child| lower_host_expr(child, program, &scoped, tensor_helpers))
                .unwrap_or(HostExpr::Unit);
            let explicit_ty = expr_host_type(expr, program, scope);
            HostExpr::Let {
                bindings,
                body: Box::new(body.clone()),
                ty: if explicit_ty == HostType::Unknown {
                    host_expr_type(&body)
                } else {
                    explicit_ty
                },
            }
        }
        Expr::List(list, _) if tag(list) == Some("tuple-get") => {
            lower_tuple_get_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("access") => {
            lower_access_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("cast") => {
            let value = lower_host_expr(
                children(list).first().unwrap_or(expr),
                program,
                scope,
                tensor_helpers,
            );
            let inferred_ty = host_expr_type(&value);
            let ty = expr_host_type(expr, program, scope);
            HostExpr::Builtin {
                name: "cast".to_string(),
                args: vec![value],
                ty: if ty == HostType::Unknown {
                    inferred_ty
                } else {
                    ty
                },
            }
        }
        Expr::List(list, _) if tag(list) == Some("app") => {
            lower_app_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("handle-effect") => {
            // `with seed(...) { body }` and similar effect handlers are
            // pure-result from the host emitter's perspective — the seed
            // flows into random-op lowering at DAG-build time and the
            // visible value is just `body`. Without this arm, the whole
            // form fell through to `HostExpr::Unit`, which is why
            // `kaiming_uniform` showed up as `()` in compiled output.
            let kids = children(list);
            // children(list) skips tag and metadata map, so for
            // `(handle-effect {effect: random, ...} seed body)` kids[0] is
            // the seed expression and kids[1] is the body. Some forms may
            // omit the seed slot.
            let body = kids.get(1).or_else(|| kids.first());
            if let Some(body) = body {
                lower_host_expr(body, program, scope, tensor_helpers)
            } else {
                HostExpr::Unit
            }
        }
        Expr::MetaExpr(meta, _) => lower_host_expr(&meta.expr, program, scope, tensor_helpers),
        _ => HostExpr::Unit,
    }
}

fn refine_function_params_from_body(params: &mut [HostParam], body: &HostExpr) {
    let HostExpr::Call { args, arg_tys, .. } = body else {
        return;
    };
    for (index, param) in params.iter_mut().enumerate() {
        if param.ty != HostType::Unknown {
            continue;
        }
        let Some(HostExpr::Var(name, _)) = args.get(index) else {
            continue;
        };
        if name != &param.name {
            continue;
        }
        let Some(inferred) = arg_tys.get(index) else {
            continue;
        };
        if *inferred != HostType::Unknown {
            param.ty = inferred.clone();
        }
    }
}

fn refine_host_function_signatures(functions: &mut [HostFunction]) -> bool {
    let mut any_changed = false;
    loop {
        let mut changed = false;
        let signatures = functions
            .iter()
            .map(|function| {
                (
                    function.name.clone(),
                    (
                        function
                            .params
                            .iter()
                            .map(|param| param.ty.clone())
                            .collect::<Vec<_>>(),
                        function.ret_ty.clone(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();

        for function in functions.iter_mut() {
            let inferred_param_fns = infer_callable_param_types(&function.params, &function.body);
            for param in function.params.iter_mut() {
                if param.ty == HostType::Unknown
                    && let Some(inferred) = inferred_param_fns.get(&param.name)
                    && !host_type_has_unknown(inferred)
                    && param.ty != *inferred
                {
                    param.ty = inferred.clone();
                    changed = true;
                }
            }

            let mut scope = function
                .params
                .iter()
                .map(|param| (param.name.clone(), param.ty.clone()))
                .collect::<HashMap<_, _>>();
            if refine_host_expr_types(&mut function.body, &mut scope, &signatures) {
                changed = true;
            }
            if function.ret_ty == HostType::Unknown {
                let body_ty = host_expr_type(&function.body);
                if body_ty != HostType::Unknown {
                    function.ret_ty = body_ty;
                    changed = true;
                }
            }

            let HostExpr::Call {
                function: callee,
                args,
                arg_tys,
                ty,
            } = &mut function.body
            else {
                continue;
            };
            let Some((callee_params, callee_ret)) = signatures.get(callee) else {
                continue;
            };

            if function.ret_ty == HostType::Unknown && *callee_ret != HostType::Unknown {
                function.ret_ty = callee_ret.clone();
                if *ty == HostType::Unknown {
                    *ty = callee_ret.clone();
                }
                changed = true;
            }

            for (index, param) in function.params.iter_mut().enumerate() {
                if param.ty != HostType::Unknown {
                    continue;
                }
                let Some(HostExpr::Var(name, arg_ty)) = args.get_mut(index) else {
                    continue;
                };
                if name != &param.name {
                    continue;
                }
                let Some(inferred) = callee_params.get(index) else {
                    continue;
                };
                if *inferred == HostType::Unknown {
                    continue;
                }
                param.ty = inferred.clone();
                *arg_ty = inferred.clone();
                if let Some(call_arg_ty) = arg_tys.get_mut(index) {
                    *call_arg_ty = inferred.clone();
                }
                changed = true;
            }
        }

        if !changed {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn refine_host_globals(globals: &mut [HostBinding], functions: &[HostFunction]) -> bool {
    let mut any_changed = false;
    loop {
        let signatures = functions
            .iter()
            .map(|function| {
                (
                    function.name.clone(),
                    (
                        function
                            .params
                            .iter()
                            .map(|param| param.ty.clone())
                            .collect::<Vec<_>>(),
                        function.ret_ty.clone(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();

        let mut changed = false;
        let mut scope = HashMap::new();
        for binding in globals.iter_mut() {
            if refine_host_expr_types(&mut binding.value, &mut scope, &signatures) {
                changed = true;
            }
            let inferred = host_expr_type(&binding.value);
            if binding.ty != inferred && inferred != HostType::Unknown {
                binding.ty = inferred.clone();
                changed = true;
            }
            scope.insert(binding.name.clone(), binding.ty.clone());
        }

        if !changed {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn propagate_named_callback_signatures(
    functions: &mut [HostFunction],
    globals: &[HostBinding],
) -> bool {
    let mut inferred = HashMap::<String, (Vec<HostType>, HostType)>::new();
    for function in functions.iter() {
        collect_named_callback_signatures(&function.body, &mut inferred);
    }
    for binding in globals {
        collect_named_callback_signatures(&binding.value, &mut inferred);
    }

    let mut changed = false;
    for function in functions.iter_mut() {
        let Some((param_tys, ret_ty)) = inferred.get(&function.name) else {
            continue;
        };
        for (param, inferred_ty) in function.params.iter_mut().zip(param_tys.iter()) {
            if host_type_has_unknown(&param.ty) && !host_type_has_unknown(inferred_ty) {
                param.ty = inferred_ty.clone();
                changed = true;
            }
        }
        if host_type_has_unknown(&function.ret_ty) && !host_type_has_unknown(ret_ty) {
            function.ret_ty = ret_ty.clone();
            changed = true;
        }
    }
    changed
}

fn collect_named_callback_signatures(
    expr: &HostExpr,
    out: &mut HashMap<String, (Vec<HostType>, HostType)>,
) {
    match expr {
        HostExpr::Call { args, .. } | HostExpr::Builtin { args, .. } => {
            for arg in args {
                collect_named_callback_signatures(arg, out);
            }
        }
        HostExpr::List(items, _) | HostExpr::Tuple(items, _) => {
            for item in items {
                collect_named_callback_signatures(item, out);
            }
        }
        HostExpr::AdtConstruct { fields, .. } => {
            for field in fields {
                collect_named_callback_signatures(field, out);
            }
        }
        HostExpr::AdtFieldAccess { base, .. } => {
            collect_named_callback_signatures(base, out);
        }
        HostExpr::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            collect_named_callback_signatures(cond, out);
            collect_named_callback_signatures(then_expr, out);
            collect_named_callback_signatures(else_expr, out);
        }
        HostExpr::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            collect_named_callback_signatures(scrutinee, out);
            collect_named_callback_signatures(some_expr, out);
            collect_named_callback_signatures(none_expr, out);
        }
        HostExpr::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            collect_named_callback_signatures(scrutinee, out);
            for arm in arms {
                collect_named_callback_signatures(&arm.expr, out);
            }
            if let Some(default_expr) = default_expr {
                collect_named_callback_signatures(default_expr, out);
            }
        }
        HostExpr::Let { bindings, body, .. } => {
            for binding in bindings {
                collect_named_callback_signatures(&binding.value, out);
            }
            collect_named_callback_signatures(body, out);
        }
        HostExpr::Map { callback, list, .. }
        | HostExpr::Filter { callback, list, .. }
        | HostExpr::Partition { callback, list, .. }
        | HostExpr::FlatMap { callback, list, .. } => {
            merge_named_callback_signature(callback, out);
            collect_named_callback_signatures_in_callback(callback, out);
            collect_named_callback_signatures(list, out);
        }
        HostExpr::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExpr::Scan {
            callback,
            init,
            list,
            ..
        } => {
            merge_named_callback_signature(callback, out);
            collect_named_callback_signatures_in_callback(callback, out);
            collect_named_callback_signatures(init, out);
            collect_named_callback_signatures(list, out);
        }
        HostExpr::TensorCall { args, .. } => {
            for arg in args {
                collect_named_callback_signatures(arg, out);
            }
        }
        HostExpr::Var(_, _)
        | HostExpr::Int(_)
        | HostExpr::Float(_)
        | HostExpr::Bool(_)
        | HostExpr::String(_)
        | HostExpr::Unit => {}
    }
}

fn collect_named_callback_signatures_in_callback(
    callback: &HostCallback,
    out: &mut HashMap<String, (Vec<HostType>, HostType)>,
) {
    if let HostCallbackKind::Inline { body, .. } = &callback.kind {
        collect_named_callback_signatures(body, out);
    }
}

fn merge_named_callback_signature(
    callback: &HostCallback,
    out: &mut HashMap<String, (Vec<HostType>, HostType)>,
) {
    let HostCallbackKind::Named { function, params } = &callback.kind else {
        return;
    };
    let entry = out
        .entry(function.clone())
        .or_insert_with(|| (vec![HostType::Unknown; params.len()], HostType::Unknown));
    if entry.0.len() < params.len() {
        entry.0.resize(params.len(), HostType::Unknown);
    }
    for (index, param) in params.iter().enumerate() {
        if entry.0[index] == HostType::Unknown && param.ty != HostType::Unknown {
            entry.0[index] = param.ty.clone();
        }
    }
    if entry.1 == HostType::Unknown && callback.ret_ty != HostType::Unknown {
        entry.1 = callback.ret_ty.clone();
    }
}

fn infer_callable_param_types(params: &[HostParam], body: &HostExpr) -> HashMap<String, HostType> {
    let unknown = params
        .iter()
        .filter(|param| param.ty == HostType::Unknown)
        .map(|param| param.name.clone())
        .collect::<HashSet<_>>();
    let mut out = HashMap::new();
    infer_callable_param_types_in_expr(body, &unknown, &mut out);
    out
}

fn infer_callable_param_types_in_expr(
    expr: &HostExpr,
    unknown: &HashSet<String>,
    out: &mut HashMap<String, HostType>,
) {
    match expr {
        HostExpr::Call {
            function,
            args,
            arg_tys,
            ..
        } => {
            if unknown.contains(function) {
                out.entry(function.clone()).or_insert_with(|| {
                    HostType::Fn(arg_tys.clone(), Box::new(HostType::Tuple(Vec::new())))
                });
            }
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExpr::List(items, _) | HostExpr::Tuple(items, _) => {
            for item in items {
                infer_callable_param_types_in_expr(item, unknown, out);
            }
        }
        HostExpr::Builtin { args, .. } => {
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExpr::AdtConstruct { fields, .. } => {
            for field in fields {
                infer_callable_param_types_in_expr(field, unknown, out);
            }
        }
        HostExpr::AdtFieldAccess { base, .. } => {
            infer_callable_param_types_in_expr(base, unknown, out);
        }
        HostExpr::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(cond, unknown, out);
            infer_callable_param_types_in_expr(then_expr, unknown, out);
            infer_callable_param_types_in_expr(else_expr, unknown, out);
        }
        HostExpr::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(scrutinee, unknown, out);
            infer_callable_param_types_in_expr(some_expr, unknown, out);
            infer_callable_param_types_in_expr(none_expr, unknown, out);
        }
        HostExpr::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(scrutinee, unknown, out);
            for arm in arms {
                infer_callable_param_types_in_expr(&arm.expr, unknown, out);
            }
            if let Some(default_expr) = default_expr {
                infer_callable_param_types_in_expr(default_expr, unknown, out);
            }
        }
        HostExpr::Let { bindings, body, .. } => {
            for binding in bindings {
                infer_callable_param_types_in_expr(&binding.value, unknown, out);
            }
            infer_callable_param_types_in_expr(body, unknown, out);
        }
        HostExpr::Map { callback, list, .. }
        | HostExpr::Filter { callback, list, .. }
        | HostExpr::Partition { callback, list, .. }
        | HostExpr::FlatMap { callback, list, .. } => {
            infer_callable_param_types_in_callback(callback, unknown, out);
            infer_callable_param_types_in_expr(list, unknown, out);
        }
        HostExpr::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExpr::Scan {
            callback,
            init,
            list,
            ..
        } => {
            infer_callable_param_types_in_callback(callback, unknown, out);
            infer_callable_param_types_in_expr(init, unknown, out);
            infer_callable_param_types_in_expr(list, unknown, out);
        }
        HostExpr::TensorCall { args, .. } => {
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExpr::Var(_, _)
        | HostExpr::Int(_)
        | HostExpr::Float(_)
        | HostExpr::Bool(_)
        | HostExpr::String(_)
        | HostExpr::Unit => {}
    }
}

fn infer_callable_param_types_in_callback(
    callback: &HostCallback,
    unknown: &HashSet<String>,
    out: &mut HashMap<String, HostType>,
) {
    if let HostCallbackKind::Inline { body, .. } = &callback.kind {
        infer_callable_param_types_in_expr(body, unknown, out);
    }
}

fn refine_host_expr_types(
    expr: &mut HostExpr,
    scope: &mut HashMap<String, HostType>,
    signatures: &HashMap<String, (Vec<HostType>, HostType)>,
) -> bool {
    let mut changed = false;
    match expr {
        HostExpr::Var(name, ty) => {
            if host_type_has_unknown(ty)
                && let Some(inferred) = scope.get(name)
                && !host_type_has_unknown(inferred)
            {
                *ty = inferred.clone();
                changed = true;
            }
        }
        HostExpr::Call {
            function,
            args,
            arg_tys,
            ty,
        } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
            let inferred_sig = scope.get(function).and_then(|ty| match ty {
                HostType::Fn(params, ret) => Some((params.clone(), (**ret).clone())),
                _ => None,
            });
            let declared_sig = signatures.get(function).cloned();
            let sig = inferred_sig.or(declared_sig);
            if let Some((params, ret)) = sig {
                for (index, arg_ty) in arg_tys.iter_mut().enumerate() {
                    if let Some(inferred) = params.get(index)
                        && *inferred != HostType::Unknown
                        && *arg_ty != *inferred
                    {
                        *arg_ty = inferred.clone();
                        changed = true;
                    }
                }
                if *ty == HostType::Unknown && ret != HostType::Unknown {
                    *ty = ret;
                    changed = true;
                }
            }
        }
        HostExpr::Builtin { name, args, ty } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
            if let Some(inferred) = infer_builtin_host_type(name, args)
                && !host_type_has_unknown(&inferred)
                && *ty != inferred
            {
                *ty = inferred;
                changed = true;
            }
        }
        HostExpr::List(items, ty) => {
            for item in items.iter_mut() {
                changed |= refine_host_expr_types(item, scope, signatures);
            }
            let item_ty = items
                .iter()
                .map(host_expr_type)
                .find(|item_ty| !host_type_has_unknown(item_ty))
                .unwrap_or(HostType::Unknown);
            let inferred = HostType::List(Box::new(item_ty));
            if host_type_has_unknown(ty) && !host_type_has_unknown(&inferred) {
                *ty = inferred;
                changed = true;
            }
        }
        HostExpr::Tuple(items, ty) => {
            for item in items.iter_mut() {
                changed |= refine_host_expr_types(item, scope, signatures);
            }
            let inferred = HostType::Tuple(items.iter().map(host_expr_type).collect());
            if host_type_has_unknown(ty) && !host_type_has_unknown(&inferred) {
                *ty = inferred;
                changed = true;
            }
        }
        HostExpr::AdtConstruct { fields, .. } => {
            for field in fields.iter_mut() {
                changed |= refine_host_expr_types(field, scope, signatures);
            }
        }
        HostExpr::AdtFieldAccess { base, .. } => {
            changed |= refine_host_expr_types(base, scope, signatures);
        }
        HostExpr::If {
            cond,
            then_expr,
            else_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(cond, scope, signatures);
            changed |= refine_host_expr_types(then_expr, scope, signatures);
            changed |= refine_host_expr_types(else_expr, scope, signatures);
            if *ty == HostType::Unknown {
                let then_ty = host_expr_type(then_expr);
                let else_ty = host_expr_type(else_expr);
                let inferred = if then_ty != HostType::Unknown {
                    then_ty
                } else {
                    else_ty
                };
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExpr::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(scrutinee, scope, signatures);
            let mut some_scope = scope.clone();
            let inner_ty = option_inner_type(scrutinee);
            if inner_ty != HostType::Unknown {
                some_scope.insert(bind_name.clone(), inner_ty);
            }
            changed |= refine_host_expr_types(some_expr, &mut some_scope, signatures);
            changed |= refine_host_expr_types(none_expr, scope, signatures);
            if *ty == HostType::Unknown {
                let some_ty = host_expr_type(some_expr);
                let none_ty = host_expr_type(none_expr);
                let inferred = if some_ty != HostType::Unknown {
                    some_ty
                } else {
                    none_ty
                };
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExpr::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(scrutinee, scope, signatures);
            for arm in arms.iter_mut() {
                let mut arm_scope = scope.clone();
                for binding in &arm.bindings {
                    arm_scope.insert(binding.name.clone(), binding.ty.clone());
                }
                changed |= refine_host_expr_types(&mut arm.expr, &mut arm_scope, signatures);
            }
            if let Some(default_expr) = default_expr {
                changed |= refine_host_expr_types(default_expr, scope, signatures);
            }
            if *ty == HostType::Unknown {
                let inferred = arms
                    .iter()
                    .map(|arm| host_expr_type(&arm.expr))
                    .find(|ty| *ty != HostType::Unknown)
                    .or_else(|| default_expr.as_ref().map(|expr| host_expr_type(expr)));
                if let Some(inferred) = inferred
                    && inferred != HostType::Unknown
                {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExpr::Let { bindings, body, ty } => {
            let mut local_scope = scope.clone();
            for binding in bindings.iter_mut() {
                changed |= refine_host_expr_types(&mut binding.value, &mut local_scope, signatures);
                if binding.ty == HostType::Unknown {
                    let inferred = host_expr_type(&binding.value);
                    if inferred != HostType::Unknown {
                        binding.ty = inferred.clone();
                        changed = true;
                    }
                }
                local_scope.insert(binding.name.clone(), binding.ty.clone());
            }
            changed |= refine_host_expr_types(body, &mut local_scope, signatures);
            if *ty == HostType::Unknown {
                let inferred = host_expr_type(body);
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExpr::Map { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            let expected_ret = list_item_type(ty);
            changed |= specialize_host_callback_types(callback, &[item_ty], &expected_ret);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if *ty == HostType::Unknown {
                let inferred = host_expr_type(list);
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExpr::Filter { callback, list, ty } | HostExpr::Partition { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            changed |= specialize_host_callback_types(callback, &[item_ty], &HostType::Bool);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if *ty == HostType::Unknown {
                let inferred = host_expr_type(list);
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExpr::FlatMap { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            let expected_ret = match ty {
                HostType::List(inner) => HostType::List(Box::new((**inner).clone())),
                _ => HostType::Unknown,
            };
            changed |= specialize_host_callback_types(callback, &[item_ty], &expected_ret);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if *ty == HostType::Unknown {
                let inferred = host_expr_type(list);
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExpr::Fold {
            callback,
            init,
            list,
            ty,
        }
        | HostExpr::Scan {
            callback,
            init,
            list,
            ty,
        } => {
            changed |= refine_host_expr_types(init, scope, signatures);
            changed |= refine_host_expr_types(list, scope, signatures);
            let init_ty = host_expr_type(init);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            changed |=
                specialize_host_callback_types(callback, &[init_ty.clone(), item_ty], &init_ty);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if *ty == HostType::Unknown {
                let inferred = host_expr_type(init);
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExpr::TensorCall { args, .. } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
        }
        HostExpr::Int(_)
        | HostExpr::Float(_)
        | HostExpr::Bool(_)
        | HostExpr::String(_)
        | HostExpr::Unit => {}
    }
    changed
}

fn list_item_type(ty: &HostType) -> HostType {
    match ty {
        HostType::List(inner) => (**inner).clone(),
        _ => HostType::Unknown,
    }
}

fn callback_params_mut(callback: &mut HostCallback) -> &mut [HostParam] {
    match &mut callback.kind {
        HostCallbackKind::Named { params, .. } | HostCallbackKind::Inline { params, .. } => params,
    }
}

fn specialize_host_callback_types(
    callback: &mut HostCallback,
    param_tys: &[HostType],
    ret_ty: &HostType,
) -> bool {
    let mut changed = false;
    for (param, inferred) in callback_params_mut(callback)
        .iter_mut()
        .zip(param_tys.iter())
    {
        if host_type_has_unknown(&param.ty) && !host_type_has_unknown(inferred) {
            param.ty = inferred.clone();
            changed = true;
        }
    }
    if host_type_has_unknown(&callback.ret_ty) && !host_type_has_unknown(ret_ty) {
        callback.ret_ty = ret_ty.clone();
        changed = true;
    }
    changed
}

fn host_type_has_unknown(ty: &HostType) -> bool {
    match ty {
        HostType::Unknown => true,
        HostType::Fn(params, ret) => {
            params.iter().any(host_type_has_unknown) || host_type_has_unknown(ret)
        }
        HostType::List(inner) | HostType::Option(inner) => host_type_has_unknown(inner),
        HostType::Dict(key, value) => host_type_has_unknown(key) || host_type_has_unknown(value),
        HostType::Tuple(items) => items.iter().any(host_type_has_unknown),
        _ => false,
    }
}

fn refine_host_callback_types(
    callback: &mut HostCallback,
    scope: &mut HashMap<String, HostType>,
    signatures: &HashMap<String, (Vec<HostType>, HostType)>,
) -> bool {
    match &mut callback.kind {
        HostCallbackKind::Inline { params, body } => {
            let mut callback_scope = scope.clone();
            for param in params.iter() {
                callback_scope.insert(param.name.clone(), param.ty.clone());
            }
            let mut changed = refine_host_expr_types(body, &mut callback_scope, signatures);
            let inferred = host_expr_type(body);
            if !host_type_has_unknown(&inferred) && callback.ret_ty != inferred {
                callback.ret_ty = inferred;
                changed = true;
            }
            changed
        }
        HostCallbackKind::Named { .. } => false,
    }
}

fn should_keep_tensor_expr_in_host_lane(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    if matches!(tag(list), Some("tuple-get" | "if" | "let" | "match")) {
        return true;
    }
    if tag(list) != Some("app") {
        return false;
    }
    let Some(callee) = children(list).first().and_then(as_list) else {
        return false;
    };
    if tag(callee) != Some("var") {
        return false;
    }
    let name = children(callee).first().and_then(symbol_name);
    if let Some(name) = name
        && !BUILTIN_NAMES.contains(&name)
        && name != "Some"
        && name != "None"
    {
        return true;
    }
    matches!(
        name,
        Some(
            "copy"
                | "reshape"
                | "to_tensor"
                | "scalar_to_tensor"
                | "pad_sequences"
                | "pad_sequences_to"
                | "concat"
                | "split"
                | "gather"
                | "scatter"
                | "where"
                | "cumsum"
                | "sort"
                | "diagonal"
                | "trace"
                | "clamp"
                | "einsum"
        )
    )
}

fn lower_match_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let scrutinee = lower_host_expr(&kids[0], program, scope, tensor_helpers);
    let scrutinee_ty = host_expr_type(&scrutinee);
    if matches!(
        scrutinee_ty,
        HostType::Int64 | HostType::Float64 | HostType::Bool | HostType::String
    ) {
        return lower_literal_match_host_expr(
            list,
            program,
            scope,
            tensor_helpers,
            scrutinee,
            scrutinee_ty,
        );
    }
    let mut bind_name = "value".to_string();
    let mut some_expr = HostExpr::Unit;
    let mut none_expr = HostExpr::Unit;
    let mut generic_arms = Vec::new();
    let mut generic_default = None;

    for arm in kids.iter().skip(1) {
        let Some(arm_list) = as_list(arm) else {
            continue;
        };
        if tag(arm_list) != Some("arm") {
            continue;
        }
        let arm_kids = children(arm_list);
        let Some(pattern) = arm_kids.first().and_then(as_list) else {
            continue;
        };
        if tag(pattern) == Some("pat-wild") {
            generic_default = Some(Box::new(lower_host_expr(
                &arm_kids[2],
                program,
                scope,
                tensor_helpers,
            )));
            continue;
        }
        if let Some("pat-ctor" | "pat-record") = tag(pattern) {
            let ctor = children(pattern).first().and_then(symbol_name);
            match ctor {
                Some("Some") => {
                    if let Some(bound) = children(pattern).get(1).and_then(as_list)
                        && tag(bound) == Some("pat-var")
                        && let Some(name) = children(bound).first().and_then(symbol_name)
                    {
                        bind_name = name.to_string();
                    }
                    let mut scoped = scope.clone();
                    scoped.insert(bind_name.clone(), option_inner_type(&scrutinee));
                    some_expr = lower_host_expr(&arm_kids[2], program, &scoped, tensor_helpers);
                }
                Some("None") => {
                    none_expr = lower_host_expr(&arm_kids[2], program, scope, tensor_helpers);
                }
                Some(ctor_name) => {
                    let ctor_fields = lookup_adt_ctor_details(program, ctor_name)
                        .map(|(_, fields)| fields)
                        .or_else(|| {
                            program
                                .type_env()
                                .get(ctor_name)
                                .and_then(parse_fn_type_expr)
                                .map(|(args, _)| {
                                    args.into_iter()
                                        .map(|ty| HostAdtField { name: None, ty })
                                        .collect::<Vec<_>>()
                                })
                        })
                        .unwrap_or_default();
                    let mut scoped = scope.clone();
                    let mut bindings = Vec::new();
                    for (field_index, subpat, field_ty) in
                        pattern_field_bindings(pattern, &ctor_fields)
                    {
                        let Some(subpat_list) = as_list(subpat) else {
                            continue;
                        };
                        if tag(subpat_list) != Some("pat-var") {
                            continue;
                        }
                        let Some(name) = children(subpat_list).first().and_then(symbol_name) else {
                            continue;
                        };
                        let ty = field_ty
                            .or_else(|| expr_type(subpat))
                            .unwrap_or(HostType::Unknown);
                        scoped.insert(name.to_string(), ty.clone());
                        bindings.push(HostPatternBinding {
                            name: name.to_string(),
                            ty,
                            field_index,
                        });
                    }
                    generic_arms.push(HostMatchArm {
                        ctor: ctor_name.to_string(),
                        bindings,
                        expr: lower_host_expr(&arm_kids[2], program, &scoped, tensor_helpers),
                    });
                }
                None => {}
            }
        }
    }

    let ty = {
        let explicit = expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        );
        if explicit == HostType::Unknown {
            let some_ty = host_expr_type(&some_expr);
            if some_ty == HostType::Unknown {
                host_expr_type(&none_expr)
            } else {
                some_ty
            }
        } else {
            explicit
        }
    };

    if matches!(scrutinee_ty, HostType::Adt(_)) {
        return HostExpr::MatchAdt {
            scrutinee: Box::new(scrutinee),
            arms: generic_arms,
            default_expr: generic_default,
            ty,
        };
    }

    HostExpr::MatchOption {
        scrutinee: Box::new(scrutinee),
        bind_name,
        some_expr: Box::new(some_expr),
        none_expr: Box::new(none_expr),
        ty,
    }
}

fn lower_literal_match_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
    scrutinee: HostExpr,
    scrutinee_ty: HostType,
) -> HostExpr {
    let kids = children(list);
    let mut literal_arms = Vec::new();
    let mut default_expr = HostExpr::Unit;
    for arm in kids.iter().skip(1) {
        let Some(arm_list) = as_list(arm) else {
            continue;
        };
        if tag(arm_list) != Some("arm") {
            continue;
        }
        let arm_kids = children(arm_list);
        let Some(pattern) = arm_kids.first().and_then(as_list) else {
            continue;
        };
        let body = lower_host_expr(&arm_kids[2], program, scope, tensor_helpers);
        match tag(pattern) {
            Some("pat-wild") => default_expr = body,
            Some("pat-lit") => {
                if let Some(lit) = children(pattern)
                    .first()
                    .and_then(|expr| host_literal_expr(expr, &scrutinee_ty))
                {
                    literal_arms.push((lit, body));
                }
            }
            _ => {}
        }
    }

    let explicit = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    let mut body = default_expr;
    let result_ty = if explicit == HostType::Unknown {
        host_expr_type(&body)
    } else {
        explicit
    };
    for (lit, arm_expr) in literal_arms.into_iter().rev() {
        body = HostExpr::If {
            cond: Box::new(HostExpr::Builtin {
                name: "eq".to_string(),
                args: vec![scrutinee.clone(), lit],
                ty: HostType::Bool,
            }),
            then_expr: Box::new(arm_expr),
            else_expr: Box::new(body),
            ty: result_ty.clone(),
        };
    }
    body
}

fn host_literal_expr(expr: &Expr, expected_ty: &HostType) -> Option<HostExpr> {
    match (expr, expected_ty) {
        (Expr::Atom(Atom::Int(value), _), HostType::Int64) => Some(HostExpr::Int(*value)),
        (Expr::Atom(Atom::Float(value), _), HostType::Float64) => Some(HostExpr::Float(*value)),
        (Expr::Atom(Atom::Bool(value), _), HostType::Bool) => Some(HostExpr::Bool(*value)),
        (Expr::Atom(Atom::Str(value), _), HostType::String) => {
            Some(HostExpr::String(value.clone()))
        }
        (Expr::Atom(Atom::Int(value), _), _) => Some(HostExpr::Int(*value)),
        (Expr::Atom(Atom::Float(value), _), _) => Some(HostExpr::Float(*value)),
        (Expr::Atom(Atom::Bool(value), _), _) => Some(HostExpr::Bool(*value)),
        (Expr::Atom(Atom::Str(value), _), _) => Some(HostExpr::String(value.clone())),
        _ => None,
    }
}

fn lower_record_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let ctor = kids
        .first()
        .and_then(symbol_name)
        .unwrap_or("_")
        .to_string();
    let ctor_info = lookup_adt_ctor_details(program, &ctor);
    let mut supplied = HashMap::new();
    for field in kids.iter().skip(1) {
        let Some(kv_list) = as_list(field) else {
            continue;
        };
        if tag(kv_list) != Some("kv") {
            continue;
        }
        let kv_kids = children(kv_list);
        let Some(name) = kv_kids.first().and_then(symbol_name) else {
            continue;
        };
        let value = kv_kids
            .get(1)
            .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers))
            .unwrap_or(HostExpr::Unit);
        supplied.insert(name.to_string(), value);
    }
    let fields = ctor_info
        .as_ref()
        .map(|(_, declared)| {
            declared
                .iter()
                .map(|field| {
                    let value = field
                        .name
                        .as_ref()
                        .and_then(|name| supplied.remove(name))
                        .unwrap_or(HostExpr::Unit);
                    if host_expr_type(&value) == HostType::Unknown {
                        force_host_expr_type(value, field.ty.clone())
                    } else {
                        value
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    HostExpr::AdtConstruct {
        ctor,
        fields,
        ty: if explicit_ty != HostType::Unknown {
            explicit_ty
        } else {
            ctor_info
                .map(|(adt_name, _)| HostType::Adt(adt_name))
                .unwrap_or(HostType::Unknown)
        },
    }
}

fn lower_access_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let base = kids
        .first()
        .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers))
        .unwrap_or(HostExpr::Unit);
    let field_name = kids.get(1).and_then(symbol_name).unwrap_or("");
    let (field_index, field_ty) =
        lookup_access_field(program, &base, field_name).unwrap_or((0, HostType::Unknown));
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    HostExpr::AdtFieldAccess {
        base: Box::new(base),
        field_index,
        ty: if explicit_ty != HostType::Unknown {
            explicit_ty
        } else {
            field_ty
        },
    }
}

fn pattern_field_bindings<'a>(
    pattern: &'a List,
    ctor_fields: &'a [HostAdtField],
) -> Vec<(usize, &'a Expr, Option<HostType>)> {
    match tag(pattern) {
        Some("pat-record") => children(pattern)
            .iter()
            .skip(1)
            .filter_map(|kv_expr| {
                let kv_list = as_list(kv_expr)?;
                if tag(kv_list) != Some("kv") {
                    return None;
                }
                let kv_kids = children(kv_list);
                let field_name = kv_kids.first().and_then(symbol_name)?;
                let field_index = ctor_fields
                    .iter()
                    .position(|field| field.name.as_deref() == Some(field_name))?;
                Some((
                    field_index,
                    kv_kids.get(1)?,
                    ctor_fields.get(field_index).map(|field| field.ty.clone()),
                ))
            })
            .collect(),
        _ => children(pattern)
            .iter()
            .skip(1)
            .enumerate()
            .map(|(field_index, subpat)| {
                (
                    field_index,
                    subpat,
                    ctor_fields.get(field_index).map(|field| field.ty.clone()),
                )
            })
            .collect(),
    }
}

fn lower_tuple_get_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let tuple_expr = kids
        .first()
        .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers));
    let index_expr = kids
        .get(1)
        .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers));
    let args = tuple_expr.into_iter().chain(index_expr).collect::<Vec<_>>();
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    let ty = if explicit_ty == HostType::Unknown {
        infer_builtin_host_type("tuple-get", &args).unwrap_or(HostType::Unknown)
    } else {
        explicit_ty
    };
    HostExpr::Builtin {
        name: "tuple-get".to_string(),
        args,
        ty,
    }
}

fn lower_app_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let app_expr = Expr::List(list.clone(), chelis_deep::Span::new(0, 0));
    if let Some(tensor_ty) = expr_tensor_type(&app_expr, program, scope)
        && expr_is_dag_lowerable(&app_expr, program)
        && !should_keep_tensor_expr_in_host_lane(&app_expr)
    {
        return lower_tensor_helper_call(&app_expr, program, scope, tensor_helpers, tensor_ty);
    }

    let kids = children(list);
    let name = kids
        .first()
        .and_then(as_list)
        .and_then(|inner| {
            if tag(inner) == Some("var") {
                children(inner).first().and_then(symbol_name)
            } else {
                None
            }
        })
        .unwrap_or("call")
        .to_string();
    let fn_sig = scope
        .get(&name)
        .and_then(host_fn_signature)
        .or_else(|| lookup_declared_fn_type(program, &name))
        .or_else(|| kids.first().and_then(expr_fn_type));
    let ctor_info = lookup_adt_ctor(program, &name);
    let explicit_ty = expr_host_type(&app_expr, program, scope);
    let inferred_ret_ty = fn_sig
        .as_ref()
        .map(|(_, ret_ty)| ret_ty.clone())
        .unwrap_or(HostType::Unknown);
    if name == "Cons" && kids.len() == 3 {
        let expr = Expr::List(list.clone(), chelis_deep::Span::new(0, 0));
        if let Some(items) = lower_list_literal_items(&expr, program, scope, tensor_helpers) {
            let ty = expr_host_type(&expr, program, scope);
            let ty = if ty == HostType::Unknown {
                HostType::List(Box::new(
                    items
                        .first()
                        .map(host_expr_type)
                        .unwrap_or(HostType::Unknown),
                ))
            } else {
                ty
            };
            return HostExpr::List(items, ty);
        }
    }
    if name == "Some" && kids.len() == 2 {
        let arg = lower_host_expr(&kids[1], program, scope, tensor_helpers);
        return HostExpr::Builtin {
            name,
            args: vec![arg.clone()],
            ty: if explicit_ty != HostType::Unknown {
                explicit_ty
            } else {
                HostType::Option(Box::new(host_expr_type(&arg)))
            },
        };
    }
    if name == "map"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            HostType::List(Box::new(callback.ret_ty.clone()))
        } else {
            ty
        };
        return HostExpr::Map {
            callback,
            list: Box::new(list_expr),
            ty,
        };
    }
    if name == "filter"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            host_expr_type(&list_expr)
        } else {
            ty
        };
        return HostExpr::Filter {
            callback,
            list: Box::new(list_expr),
            ty,
        };
    }
    if name == "fold"
        && kids.len() == 4
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let init_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let list_expr = lower_host_expr(&kids[3], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            host_expr_type(&init_expr)
        } else {
            ty
        };
        return HostExpr::Fold {
            callback,
            init: Box::new(init_expr),
            list: Box::new(list_expr),
            ty,
        };
    }
    if name == "scan"
        && kids.len() == 4
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let init_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let list_expr = lower_host_expr(&kids[3], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            HostType::List(Box::new(host_expr_type(&init_expr)))
        } else {
            ty
        };
        return HostExpr::Scan {
            callback,
            init: Box::new(init_expr),
            list: Box::new(list_expr),
            ty,
        };
    }
    if name == "partition"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            let list_ty = host_expr_type(&list_expr);
            HostType::Tuple(vec![list_ty.clone(), list_ty])
        } else {
            ty
        };
        return HostExpr::Partition {
            callback,
            list: Box::new(list_expr),
            ty,
        };
    }
    if name == "flat_map"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            match callback.ret_ty.clone() {
                HostType::List(inner) => HostType::List(inner),
                _ => HostType::Unknown,
            }
        } else {
            ty
        };
        return HostExpr::FlatMap {
            callback,
            list: Box::new(list_expr),
            ty,
        };
    }
    let args = kids[1..]
        .iter()
        .map(|arg| lower_host_expr(arg, program, scope, tensor_helpers))
        .collect::<Vec<_>>();
    let construct_ty = if let Some((adt_name, _)) = &ctor_info {
        if matches!(explicit_ty, HostType::Adt(_)) {
            explicit_ty.clone()
        } else {
            HostType::Adt(adt_name.clone())
        }
    } else {
        inferred_ret_ty.clone()
    };
    if ctor_info.is_some() && !matches!(name.as_str(), "Some" | "None") {
        return HostExpr::AdtConstruct {
            ctor: name,
            fields: args,
            ty: construct_ty,
        };
    }
    if !BUILTIN_NAMES.contains(&name.as_str())
        && name != "Some"
        && name != "None"
        && fn_sig.is_some()
    {
        return HostExpr::Call {
            function: name,
            args,
            arg_tys: fn_sig
                .as_ref()
                .map(|(param_tys, _)| param_tys.clone())
                .unwrap_or_default(),
            ty: if explicit_ty != HostType::Unknown {
                explicit_ty
            } else {
                inferred_ret_ty
            },
        };
    }
    let ty = if explicit_ty != HostType::Unknown {
        explicit_ty
    } else {
        infer_builtin_host_type(&name, &args).unwrap_or(HostType::Unknown)
    };
    HostExpr::Builtin { name, args, ty }
}

fn host_fn_signature(ty: &HostType) -> Option<(Vec<HostType>, HostType)> {
    match ty {
        HostType::Fn(params, ret) => Some((params.clone(), (**ret).clone())),
        _ => None,
    }
}

fn lower_host_callback(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> Option<HostCallback> {
    match expr {
        Expr::MetaExpr(meta, _) => lower_host_callback(&meta.expr, program, scope, tensor_helpers),
        Expr::List(list, _) if tag(list) == Some("fn") => {
            let kids = children(list);
            let params_list = kids.first().and_then(as_list)?;
            if tag(params_list) != Some("params") {
                return None;
            }
            let (param_tys, ret_ty) = expr_fn_type(expr).unwrap_or((Vec::new(), HostType::Unknown));
            let mut callback_scope = scope.clone();
            let mut params = Vec::new();
            for (index, param) in children(params_list).iter().enumerate() {
                let name = param_name(param)?;
                let ty = param_tys
                    .get(index)
                    .cloned()
                    .filter(|ty| *ty != HostType::Unknown)
                    .or_else(|| param_host_type(param))
                    .unwrap_or(HostType::Unknown);
                callback_scope.insert(name.clone(), ty.clone());
                params.push(HostParam { name, ty });
            }
            let body = lower_host_expr(kids.get(1)?, program, &callback_scope, tensor_helpers);
            let ret_ty = if ret_ty == HostType::Unknown {
                host_expr_type(&body)
            } else {
                ret_ty
            };
            Some(HostCallback {
                kind: HostCallbackKind::Inline {
                    params,
                    body: Box::new(body),
                },
                ret_ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some("var") => {
            let name = children(list).first().and_then(symbol_name)?;
            let (param_tys, ret_ty) = lookup_declared_fn_type(program, name).or_else(|| {
                lookup_program_def(&collect_program_defs(program.exprs()), name)
                    .and_then(expr_fn_type)
            })?;
            let params = param_tys
                .into_iter()
                .enumerate()
                .map(|(index, ty)| HostParam {
                    name: format!("arg{index}"),
                    ty,
                })
                .collect();
            Some(HostCallback {
                kind: HostCallbackKind::Named {
                    function: name.to_string(),
                    params,
                },
                ret_ty,
            })
        }
        _ => None,
    }
}

fn lower_list_literal_items(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> Option<Vec<HostExpr>> {
    match expr {
        Expr::MetaExpr(meta, _) => {
            lower_list_literal_items(&meta.expr, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("var") => {
            (children(list).first().and_then(symbol_name) == Some("Nil")).then(Vec::new)
        }
        Expr::List(list, _) if tag(list) == Some("app") => {
            let kids = children(list);
            if kids.len() != 3 {
                return None;
            }
            let func_name = kids
                .first()
                .and_then(as_list)
                .and_then(|inner| (tag(inner) == Some("var")).then_some(inner))
                .and_then(|inner| children(inner).first().and_then(symbol_name));
            if func_name != Some("Cons") {
                return None;
            }
            let head = lower_host_expr(&kids[1], program, scope, tensor_helpers);
            let mut tail = lower_list_literal_items(&kids[2], program, scope, tensor_helpers)?;
            tail.insert(0, head);
            Some(tail)
        }
        _ => None,
    }
}

fn tensor_helper_args(
    inputs: &[HostTensorInput],
    scope: &HashMap<String, HostType>,
) -> Vec<HostExpr> {
    inputs
        .iter()
        .map(|input| {
            HostExpr::Var(
                input.name.clone(),
                scope
                    .get(&input.name)
                    .cloned()
                    .unwrap_or_else(|| host_type_from_tensor_input(&input.ty)),
            )
        })
        .collect()
}

fn tensor_helper_inputs(dag: &crate::Dag) -> Vec<HostTensorInput> {
    let mut seen = HashSet::new();
    dag.nodes()
        .iter()
        .filter_map(|node| match &node.op {
            crate::RiscOp::Load { name } if seen.insert(name.clone()) => Some(HostTensorInput {
                name: name.clone(),
                ty: node.output_type.clone(),
            }),
            _ => None,
        })
        .collect()
}

fn remap_tensor_helper_dim_symbols(
    dag: &crate::Dag,
    scope: &HashMap<String, HostType>,
    expected_output: &TensorType,
) -> crate::Dag {
    fn tensor_type_has_synthetic_dims(tensor_ty: &TensorType) -> bool {
        tensor_ty.dims.iter().any(|dim| {
            matches!(dim, crate::dag::DimInfo::Named(name, None) if {
                let mut chars = name.chars();
                matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
            })
        })
    }

    let formal_inputs = tensor_helper_inputs(dag);
    let mut actual_inputs = formal_inputs
        .iter()
        .map(|input| match scope.get(&input.name) {
            Some(HostType::Tensor(actual)) => actual.clone(),
            _ => input.ty.clone(),
        })
        .collect::<Vec<_>>();
    let mut formal_params = formal_inputs
        .iter()
        .map(|input| input.ty.clone())
        .collect::<Vec<_>>();
    if let Some(root) = dag.roots().first().and_then(|id| dag.get(*id)) {
        formal_params.push(root.output_type.clone());
        let actual_output = if tensor_type_has_synthetic_dims(expected_output) {
            match root.op {
                crate::dag::RiscOp::Permute { ref axes } => root
                    .inputs
                    .first()
                    .and_then(|id| dag.get(*id))
                    .map(|node| {
                        let mut output = node.output_type.clone();
                        output.dims = axes
                            .iter()
                            .filter_map(|axis| node.output_type.dims.get(*axis).cloned())
                            .collect();
                        output
                    })
                    .unwrap_or_else(|| expected_output.clone()),
                crate::dag::RiscOp::UniformLike { .. } | crate::dag::RiscOp::Dropout { .. } => root
                    .inputs
                    .first()
                    .and_then(|id| dag.get(*id))
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| expected_output.clone()),
                _ => expected_output.clone(),
            }
        } else {
            expected_output.clone()
        };
        actual_inputs.push(actual_output);
    }
    let remapped = crate::lower::remap_tensor_dim_symbols(dag, &formal_params, &actual_inputs);
    actualize_tensor_helper_types(&remapped, scope)
}

fn actualize_tensor_helper_types(
    dag: &crate::Dag,
    scope: &HashMap<String, HostType>,
) -> crate::Dag {
    fn inferred_load_type(name: &str, scope: &HashMap<String, HostType>, fallback: &TensorType) -> TensorType {
        match scope.get(name) {
            Some(HostType::Tensor(actual)) => actual.clone(),
            _ => fallback.clone(),
        }
    }

    fn precision_like(input: &TensorType, precision: chelis_types::types::Prim) -> TensorType {
        TensorType {
            dims: input.dims.clone(),
            precision,
        }
    }

    fn synthetic_dims(tensor_ty: &TensorType) -> bool {
        tensor_ty.dims.iter().any(|dim| {
            matches!(dim, crate::dag::DimInfo::Named(name, None) if {
                let mut chars = name.chars();
                matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
            })
        })
    }

    let mut inferred = HashMap::<crate::dag::NodeId, TensorType>::new();
    for node in dag.nodes() {
        let actual = match &node.op {
            crate::dag::RiscOp::Load { name } => Some(inferred_load_type(name, scope, &node.output_type)),
            crate::dag::RiscOp::Add
            | crate::dag::RiscOp::Mul
            | crate::dag::RiscOp::CmpLt
            | crate::dag::RiscOp::MaxElem
            | crate::dag::RiscOp::Neg
            | crate::dag::RiscOp::Exp
            | crate::dag::RiscOp::Log
            | crate::dag::RiscOp::Sin
            | crate::dag::RiscOp::Sqrt
            | crate::dag::RiscOp::UniformLike { .. }
            | crate::dag::RiscOp::Dropout { .. }
            | crate::dag::RiscOp::Realize
            | crate::dag::RiscOp::Cast { .. } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| precision_like(input, node.output_type.precision)),
            crate::dag::RiscOp::Sum { axis }
            | crate::dag::RiscOp::MaxReduce { axis }
            | crate::dag::RiscOp::MinReduce { axis }
            | crate::dag::RiscOp::ProdReduce { axis }
            | crate::dag::RiscOp::Argmax { axis }
            | crate::dag::RiscOp::Argmin { axis } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| TensorType {
                    dims: input
                        .dims
                        .iter()
                        .enumerate()
                        .filter_map(|(index, dim)| (index != *axis).then_some(dim.clone()))
                        .collect(),
                    precision: node.output_type.precision,
                }),
            crate::dag::RiscOp::Permute { axes } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| TensorType {
                    dims: axes
                        .iter()
                        .filter_map(|axis| input.dims.get(*axis).cloned())
                        .collect(),
                    precision: node.output_type.precision,
                }),
            _ => None,
        };
        if let Some(actual) = actual {
            inferred.insert(node.id, actual);
        }
    }

    let mut actualized = dag.clone();
    let node_ids = actualized
        .nodes()
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    for id in node_ids {
        let Some(node) = actualized.get(id).cloned() else {
            continue;
        };
        let Some(actual) = inferred.get(&id) else {
            continue;
        };
        if !synthetic_dims(&node.output_type) || node.output_type.dims.len() != actual.dims.len() {
            continue;
        }
        actualized.replace_node(id, node.op, node.inputs, actual.clone());
        if let Some(reusable_input) = node.reusable_input {
            actualized.set_reusable_input(id, reusable_input);
        }
    }
    actualized
}

fn collect_program_defs(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut defs = HashMap::new();
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) == Some("def")
            && let (Some(name), Some(body)) = (
                children(list).first().and_then(symbol_name),
                children(list).get(1),
            )
        {
            defs.insert(name.to_string(), body.clone());
        }
    }
    defs
}

fn top_level_items(exprs: &[Expr]) -> Vec<&Expr> {
    let mut out = Vec::new();
    for expr in exprs {
        collect_top_level_items(expr, &mut out);
    }
    out
}

fn collect_top_level_items<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    if tag(list) == Some("module") {
        for child in list.elements.iter().skip(3) {
            collect_top_level_items(child, out);
        }
        return;
    }
    out.push(expr);
}

fn collect_tensor_scope(scope: &HashMap<String, HostType>) -> HashMap<String, TensorType> {
    scope
        .iter()
        .filter_map(|(name, ty)| tensor_type_from_host_input(ty).map(|tensor| (name.clone(), tensor)))
        .collect()
}

fn tensor_type_from_host_input(ty: &HostType) -> Option<TensorType> {
    match ty {
        HostType::Tensor(tensor) => Some(tensor.clone()),
        HostType::Float64 => Some(TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::F32,
        }),
        HostType::Int64 => Some(TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Int64,
        }),
        HostType::Bool => Some(TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Bool,
        }),
        _ => None,
    }
}

fn host_type_from_tensor_input(ty: &TensorType) -> HostType {
    if ty.dims.is_empty() {
        match ty.precision {
            chelis_types::types::Prim::Bool => HostType::Bool,
            chelis_types::types::Prim::Int8
            | chelis_types::types::Prim::Int32
            | chelis_types::types::Prim::Int64 => HostType::Int64,
            _ => HostType::Float64,
        }
    } else {
        HostType::Tensor(ty.clone())
    }
}

fn expr_host_type(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
) -> HostType {
    match expr {
        Expr::Atom(Atom::Int(_), _) => HostType::Int64,
        Expr::Atom(Atom::Float(_), _) => HostType::Float64,
        Expr::Atom(Atom::Bool(_), _) => HostType::Bool,
        Expr::Atom(Atom::Str(_), _) => HostType::String,
        Expr::List(list, _) if tag(list) == Some("var") => children(list)
            .first()
            .and_then(symbol_name)
            .and_then(|name| {
                expr_type(expr)
                    .filter(|ty| *ty != HostType::Unknown)
                    .or_else(|| {
                        scope
                            .get(name)
                            .cloned()
                            .or_else(|| lookup_declared_host_type(program, name))
                    })
            })
            .unwrap_or(HostType::Unknown),
        Expr::List(list, _) if tag(list) == Some("app") => {
            let explicit = expr_type(expr).unwrap_or(HostType::Unknown);
            if app_expr_needs_inferred_type(&explicit) {
                let inferred =
                    infer_app_expr_host_type(list, program, scope).unwrap_or(HostType::Unknown);
                if should_prefer_inferred_app_type(&explicit, &inferred) {
                    inferred
                } else if explicit != HostType::Unknown {
                    explicit
                } else {
                    inferred
                }
            } else {
                explicit
            }
        }
        _ => expr_type(expr).unwrap_or(HostType::Unknown),
    }
}

fn app_expr_needs_inferred_type(explicit: &HostType) -> bool {
    explicit == &HostType::Unknown || host_type_has_synthetic_tensor_dims(explicit)
}

fn infer_app_expr_host_type(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
) -> Option<HostType> {
    let kids = children(list);
    let callee = kids.first().and_then(as_list)?;
    if tag(callee) != Some("var") {
        return None;
    }
    let name = children(callee).first().and_then(symbol_name)?;
    if !BUILTIN_NAMES.contains(&name) {
        return lookup_declared_fn_type(program, name).map(|(_, ret)| ret);
    }
    if name == "einsum" {
        let equation = match kids.get(1) {
            Some(Expr::Atom(Atom::Str(value), _)) => value.as_str(),
            _ => return Some(HostType::Unknown),
        };
        let tensors = kids[2..]
            .iter()
            .map(|arg| match expr_host_type(arg, program, scope) {
                HostType::Tensor(tensor_ty) => Some(tensor_ty),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        return infer_einsum_tensor_type(equation, &tensors).map(HostType::Tensor);
    }
    let arg_tys = kids[1..]
        .iter()
        .map(|arg| expr_host_type(arg, program, scope))
        .collect::<Vec<_>>();
    infer_builtin_host_type_from_arg_tys(name, &arg_tys)
}

fn should_prefer_inferred_app_type(explicit: &HostType, inferred: &HostType) -> bool {
    explicit == &HostType::Unknown
        || matches!(
            (explicit, inferred),
            (HostType::Tensor(_), HostType::Tensor(_)) if host_type_has_synthetic_tensor_dims(explicit)
                && !host_type_has_synthetic_tensor_dims(inferred)
        )
}

fn host_type_has_synthetic_tensor_dims(ty: &HostType) -> bool {
    fn synthetic_dim_name(name: &str) -> bool {
        let mut chars = name.chars();
        matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
    }

    match ty {
        HostType::Tensor(tensor_ty) => tensor_ty
            .dims
            .iter()
            .any(|dim| matches!(dim, crate::dag::DimInfo::Named(name, None) if synthetic_dim_name(name))),
        HostType::Tuple(items) => items.iter().any(host_type_has_synthetic_tensor_dims),
        HostType::List(inner) | HostType::Option(inner) => host_type_has_synthetic_tensor_dims(inner),
        _ => false,
    }
}

fn infer_einsum_tensor_type(equation: &str, tensors: &[TensorType]) -> Option<TensorType> {
    let (inputs, output) = equation.split_once("->")?;
    let input_specs = inputs.split(',').map(str::trim).collect::<Vec<_>>();
    if input_specs.len() != tensors.len() {
        return None;
    }

    let mut labels = HashMap::<char, crate::dag::DimInfo>::new();
    for (spec, tensor) in input_specs.iter().zip(tensors.iter()) {
        let axes = spec.chars().filter(|ch| !ch.is_whitespace()).collect::<Vec<_>>();
        if axes.len() != tensor.dims.len() {
            return None;
        }
        for (axis, dim) in axes.into_iter().zip(tensor.dims.iter().cloned()) {
            labels.entry(axis).or_insert(dim);
        }
    }

    let dims = output
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .map(|axis| labels.get(&axis).cloned())
        .collect::<Option<Vec<_>>>()?;
    let precision = tensors
        .first()
        .map(|tensor| tensor.precision)
        .unwrap_or(chelis_types::types::Prim::F32);
    Some(TensorType { dims, precision })
}

fn lookup_type_expr<'a>(type_env: &'a HashMap<String, Expr>, name: &str) -> Option<&'a Expr> {
    type_env.get(name).or_else(|| {
        let mut matches = type_env
            .iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    })
}

fn lookup_declared_type_expr<'a>(program: &'a CheckedProgram, name: &str) -> Option<&'a Expr> {
    find_top_level_sig_expr(program.exprs(), name)
        .or_else(|| lookup_type_expr(program.type_env(), name))
}

fn lookup_declared_host_type(program: &CheckedProgram, name: &str) -> Option<HostType> {
    lookup_declared_type_expr(program, name).map(parse_host_type)
}

fn lookup_declared_fn_type(
    program: &CheckedProgram,
    name: &str,
) -> Option<(Vec<HostType>, HostType)> {
    lookup_declared_type_expr(program, name).and_then(parse_fn_type_expr)
}

fn lookup_program_def<'a>(defs: &'a HashMap<String, Expr>, name: &str) -> Option<&'a Expr> {
    defs.get(name).or_else(|| {
        let mut matches = defs
            .iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    })
}

fn terminal_name_matches(full_name: &str, short_name: &str) -> bool {
    full_name == short_name || terminal_name(full_name) == terminal_name(short_name)
}

fn terminal_name(name: &str) -> &str {
    name.rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(name)
}

fn find_top_level_sig_expr<'a>(exprs: &'a [Expr], name: &str) -> Option<&'a Expr> {
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("defsig") {
            continue;
        }
        let kids = children(list);
        let Some(sig_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if terminal_name_matches(sig_name, name) {
            return kids.get(1);
        }
    }
    None
}

fn expr_tensor_type(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
) -> Option<TensorType> {
    match expr_host_type(expr, program, scope) {
        HostType::Tensor(ty) => Some(ty),
        _ => None,
    }
}

fn expr_type(expr: &Expr) -> Option<HostType> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let meta = match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta,
        _ => return None,
    };
    meta.entries
        .iter()
        .find(|(key, _)| key == "type")
        .map(|(_, value)| parse_host_type(value))
}

fn expr_fn_type(expr: &Expr) -> Option<(Vec<HostType>, HostType)> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let meta = match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta,
        _ => return None,
    };
    meta.entries
        .iter()
        .find(|(key, _)| key == "type")
        .and_then(|(_, value)| parse_fn_type_expr(value))
}

fn parse_fn_type_expr(expr: &Expr) -> Option<(Vec<HostType>, HostType)> {
    let (args, ret) = parse_fn_type_expr_parts(expr)?;
    Some((
        args.iter().map(parse_host_type).collect(),
        parse_host_type(&ret),
    ))
}

fn parse_fn_type_expr_parts(expr: &Expr) -> Option<(Vec<Expr>, Expr)> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if tag(list) != Some("t-fn") {
        return None;
    }
    let kids = children(list);
    let (ret, args) = kids.split_last()?;
    Some((args.to_vec(), ret.clone()))
}

fn parse_host_type(expr: &Expr) -> HostType {
    if let Expr::MetaExpr(meta, _) = expr {
        return parse_host_type(&meta.expr);
    }
    let Expr::List(list, _) = expr else {
        return HostType::Unknown;
    };
    match tag(list) {
        Some("t-prim") => match children(list).first().and_then(symbol_name) {
            Some("int64") | Some("int32") => HostType::Int64,
            Some("f64") | Some("f32") => HostType::Float64,
            Some("bool") => HostType::Bool,
            Some("string") => HostType::String,
            _ => HostType::Unknown,
        },
        Some("t-tensor") => HostType::Tensor(crate::lower::tensor_type_from_deep(expr)),
        Some("t-adt") => {
            let kids = children(list);
            match kids.first().and_then(symbol_name) {
                Some("Option") if kids.len() == 2 => {
                    HostType::Option(Box::new(parse_host_type(&kids[1])))
                }
                Some("List") if kids.len() == 2 => {
                    HostType::List(Box::new(parse_host_type(&kids[1])))
                }
                Some("Dict") if kids.len() == 3 => HostType::Dict(
                    Box::new(parse_host_type(&kids[1])),
                    Box::new(parse_host_type(&kids[2])),
                ),
                Some("MappedFile") if kids.len() == 1 => HostType::MappedFile,
                Some(name) => HostType::Adt(name.to_string()),
                _ => HostType::Unknown,
            }
        }
        Some("t-tuple") => HostType::Tuple(children(list).iter().map(parse_host_type).collect()),
        Some("t-fn") => {
            let kids = children(list);
            match kids.split_last() {
                Some((ret, args)) => HostType::Fn(
                    args.iter().map(parse_host_type).collect(),
                    Box::new(parse_host_type(ret)),
                ),
                None => HostType::Unknown,
            }
        }
        Some("t-unit") => HostType::Unit,
        _ => HostType::Unknown,
    }
}

fn option_inner_type(expr: &HostExpr) -> HostType {
    match host_expr_type(expr) {
        HostType::Option(inner) => (*inner).clone(),
        _ => HostType::Unknown,
    }
}

fn host_expr_type(expr: &HostExpr) -> HostType {
    match expr {
        HostExpr::Int(_) => HostType::Int64,
        HostExpr::Float(_) => HostType::Float64,
        HostExpr::Bool(_) => HostType::Bool,
        HostExpr::String(_) => HostType::String,
        HostExpr::List(_, ty) => ty.clone(),
        HostExpr::Tuple(_, ty) => ty.clone(),
        HostExpr::Var(_, ty)
        | HostExpr::Call { ty, .. }
        | HostExpr::Builtin { ty, .. }
        | HostExpr::AdtConstruct { ty, .. }
        | HostExpr::AdtFieldAccess { ty, .. }
        | HostExpr::If { ty, .. }
        | HostExpr::MatchOption { ty, .. }
        | HostExpr::MatchAdt { ty, .. }
        | HostExpr::Let { ty, .. }
        | HostExpr::Map { ty, .. }
        | HostExpr::Filter { ty, .. }
        | HostExpr::Fold { ty, .. }
        | HostExpr::Scan { ty, .. }
        | HostExpr::Partition { ty, .. }
        | HostExpr::FlatMap { ty, .. }
        | HostExpr::TensorCall { ty, .. } => ty.clone(),
        HostExpr::Unit => HostType::Unit,
    }
}

fn force_host_expr_type(expr: HostExpr, ty: HostType) -> HostExpr {
    match expr {
        HostExpr::Var(name, _) => HostExpr::Var(name, ty),
        HostExpr::Call {
            function,
            args,
            arg_tys,
            ..
        } => HostExpr::Call {
            function,
            args,
            arg_tys,
            ty,
        },
        HostExpr::Builtin { name, args, .. } => HostExpr::Builtin { name, args, ty },
        HostExpr::AdtConstruct { ctor, fields, .. } => HostExpr::AdtConstruct { ctor, fields, ty },
        HostExpr::AdtFieldAccess {
            base, field_index, ..
        } => HostExpr::AdtFieldAccess {
            base,
            field_index,
            ty,
        },
        HostExpr::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => HostExpr::If {
            cond,
            then_expr,
            else_expr,
            ty,
        },
        HostExpr::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ..
        } => HostExpr::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        },
        HostExpr::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => HostExpr::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        },
        HostExpr::Let { bindings, body, .. } => HostExpr::Let { bindings, body, ty },
        HostExpr::Map { callback, list, .. } => HostExpr::Map { callback, list, ty },
        HostExpr::Filter { callback, list, .. } => HostExpr::Filter { callback, list, ty },
        HostExpr::Fold {
            callback,
            init,
            list,
            ..
        } => HostExpr::Fold {
            callback,
            init,
            list,
            ty,
        },
        HostExpr::Scan {
            callback,
            init,
            list,
            ..
        } => HostExpr::Scan {
            callback,
            init,
            list,
            ty,
        },
        HostExpr::Partition { callback, list, .. } => HostExpr::Partition { callback, list, ty },
        HostExpr::FlatMap { callback, list, .. } => HostExpr::FlatMap { callback, list, ty },
        HostExpr::TensorCall { helper, args, .. } => HostExpr::TensorCall { helper, args, ty },
        other => other,
    }
}

fn infer_builtin_host_type(name: &str, args: &[HostExpr]) -> Option<HostType> {
    let arg_tys = args.iter().map(host_expr_type).collect::<Vec<_>>();
    match name {
        "einsum" => {
            let equation = match args.first() {
                Some(HostExpr::String(value)) => value.as_str(),
                _ => return Some(HostType::Unknown),
            };
            let tensors = args[1..]
                .iter()
                .map(|arg| match host_expr_type(arg) {
                    HostType::Tensor(tensor_ty) => Some(tensor_ty),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            infer_einsum_tensor_type(equation, &tensors).map(HostType::Tensor)
        }
        "tuple-get" => match (arg_tys.first(), args.get(1)) {
            (Some(HostType::Tuple(items)), Some(HostExpr::Int(index))) => items
                .get(*index as usize)
                .cloned()
                .or(Some(HostType::Unknown)),
            _ => Some(HostType::Unknown),
        },
        _ => infer_builtin_host_type_from_arg_tys(name, &arg_tys),
    }
}

fn infer_builtin_host_type_from_arg_tys(name: &str, arg_tys: &[HostType]) -> Option<HostType> {
    let tensor_arg = arg_tys.iter().find_map(|ty| match ty {
        HostType::Tensor(tensor_ty) => Some(tensor_ty.clone()),
        _ => None,
    });
    match name {
        "add" | "sub" | "mul" | "div" | "neg" | "exp" | "log" | "sin" | "sqrt" | "relu"
        | "sigmoid" | "max_elem" | "min_elem" | "copy" | "uniform_like" | "dropout" => {
            if let Some(tensor_ty) = tensor_arg {
                Some(HostType::Tensor(tensor_ty))
            } else if arg_tys.iter().any(|ty| matches!(ty, HostType::Float64)) {
                Some(HostType::Float64)
            } else {
                Some(HostType::Int64)
            }
        }
        "mod" | "bitand" | "bitor" | "bitxor" | "shl" | "shr" | "string_len" | "rank" | "shape"
        | "numel" => Some(HostType::Int64),
        "cmplt" => match arg_tys.first() {
            Some(HostType::Tensor(tensor_ty)) => Some(HostType::Tensor(TensorType {
                dims: tensor_ty.dims.clone(),
                precision: chelis_types::types::Prim::Bool,
            })),
            _ => Some(HostType::Bool),
        },
        "reshape" => match arg_tys.first() {
            Some(HostType::Tensor(tensor_ty)) => Some(HostType::Tensor(tensor_ty.clone())),
            _ => Some(HostType::Unknown),
        },
        "lt" | "gt" | "gte" | "lte" | "eq" | "neq" | "and" | "or" | "not" | "string_contains"
        | "string_starts_with" | "string_ends_with" => Some(HostType::Bool),
        "string_concat" | "string_trim" | "string_slice" | "to_string" => Some(HostType::String),
        "to_int" => Some(HostType::Option(Box::new(HostType::Int64))),
        "to_float" => Some(HostType::Option(Box::new(HostType::Float64))),
        "len" => Some(HostType::Int64),
        "index" => match arg_tys.first() {
            Some(HostType::List(inner)) => Some((**inner).clone()),
            _ => Some(HostType::Unknown),
        },
        "append" => arg_tys.first().cloned(),
        "concat" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(HostType::List(inner)), Some(HostType::Int64))
                if matches!(inner.as_ref(), HostType::Tensor(_)) =>
            {
                match inner.as_ref() {
                    HostType::Tensor(tensor_ty) => Some(HostType::Tensor(tensor_ty.clone())),
                    _ => Some(HostType::Unknown),
                }
            }
            (Some(lhs), Some(_)) => Some(lhs.clone()),
            _ => Some(HostType::Unknown),
        },
        "split" => match arg_tys.first() {
            Some(HostType::Tensor(tensor_ty)) => Some(HostType::List(Box::new(HostType::Tensor(
                tensor_ty.clone(),
            )))),
            _ => Some(HostType::Unknown),
        },
        "gather" | "scatter" | "where" | "cumsum" | "diagonal" | "trace" | "clamp" => {
            arg_tys.first().cloned()
        }
        "sort" => match arg_tys.first() {
            Some(HostType::Tensor(tensor_ty)) => Some(HostType::Tuple(vec![
                HostType::Tensor(tensor_ty.clone()),
                HostType::Tensor(TensorType {
                    dims: tensor_ty.dims.clone(),
                    precision: chelis_types::types::Prim::Int64,
                }),
            ])),
            _ => Some(HostType::Unknown),
        },
        "tuple-get" => Some(HostType::Unknown),
        "take" | "drop" => match arg_tys.first() {
            Some(HostType::List(inner)) => Some(HostType::List(Box::new((**inner).clone()))),
            _ => Some(HostType::Unknown),
        },
        "chunk" => match arg_tys.first() {
            Some(HostType::List(inner)) => Some(HostType::List(Box::new(HostType::List(
                Box::new((**inner).clone()),
            )))),
            _ => Some(HostType::Unknown),
        },
        "range" => Some(HostType::List(Box::new(HostType::Int64))),
        "map" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(_), Some(HostType::List(inner))) => {
                Some(HostType::List(Box::new((**inner).clone())))
            }
            _ => Some(HostType::Unknown),
        },
        "filter" => arg_tys.get(1).cloned(),
        "fold" => arg_tys.get(1).cloned(),
        "scan" => match arg_tys.get(1) {
            Some(init_ty) => Some(HostType::List(Box::new(init_ty.clone()))),
            None => Some(HostType::Unknown),
        },
        "partition" => match arg_tys.get(1) {
            Some(list_ty) => Some(HostType::Tuple(vec![list_ty.clone(), list_ty.clone()])),
            None => Some(HostType::Unknown),
        },
        "flat_map" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(_), Some(HostType::List(_))) => match arg_tys.first() {
                Some(HostType::Unknown) => Some(HostType::Unknown),
                Some(_) => None,
                None => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "flatten" => match arg_tys.first() {
            Some(HostType::List(inner)) => match inner.as_ref() {
                HostType::List(nested) => Some(HostType::List(Box::new((**nested).clone()))),
                _ => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "zip" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(HostType::List(lhs)), Some(HostType::List(rhs))) => {
                Some(HostType::List(Box::new(HostType::Tuple(vec![
                    (**lhs).clone(),
                    (**rhs).clone(),
                ]))))
            }
            _ => Some(HostType::Unknown),
        },
        "enumerate" => match arg_tys.first() {
            Some(HostType::List(inner)) => Some(HostType::List(Box::new(HostType::Tuple(vec![
                HostType::Int64,
                (**inner).clone(),
            ])))),
            _ => Some(HostType::Unknown),
        },
        "dict_of" => match arg_tys.first() {
            Some(HostType::List(inner)) => match &**inner {
                HostType::Tuple(parts) if parts.len() == 2 => Some(HostType::Dict(
                    Box::new(parts[0].clone()),
                    Box::new(parts[1].clone()),
                )),
                _ => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "dict_get" => match arg_tys.first() {
            Some(HostType::Dict(_, value)) => Some(HostType::Option(Box::new((**value).clone()))),
            _ => Some(HostType::Unknown),
        },
        "dict_contains" => Some(HostType::Bool),
        "dict_remove" => match arg_tys.first() {
            Some(HostType::Dict(key, value)) => Some(HostType::Dict(
                Box::new((**key).clone()),
                Box::new((**value).clone()),
            )),
            _ => Some(HostType::Unknown),
        },
        "dict_insert" => match arg_tys.first() {
            Some(HostType::Dict(key, value)) => Some(HostType::Dict(
                Box::new((**key).clone()),
                Box::new((**value).clone()),
            )),
            _ => Some(HostType::Unknown),
        },
        "dict_merge" => match (arg_tys.first(), arg_tys.get(1)) {
            (
                Some(HostType::Dict(lhs_key, lhs_value)),
                Some(HostType::Dict(rhs_key, rhs_value)),
            ) if **lhs_key == **rhs_key && **lhs_value == **rhs_value => Some(HostType::Dict(
                Box::new((**lhs_key).clone()),
                Box::new((**lhs_value).clone()),
            )),
            _ => Some(HostType::Unknown),
        },
        "dict_keys" => match arg_tys.first() {
            Some(HostType::Dict(key, _)) => Some(HostType::List(Box::new((**key).clone()))),
            _ => Some(HostType::Unknown),
        },
        "dict_values" => match arg_tys.first() {
            Some(HostType::Dict(_, value)) => Some(HostType::List(Box::new((**value).clone()))),
            _ => Some(HostType::Unknown),
        },
        "dict_entries" => match arg_tys.first() {
            Some(HostType::Dict(key, value)) => {
                Some(HostType::List(Box::new(HostType::Tuple(vec![
                    (**key).clone(),
                    (**value).clone(),
                ]))))
            }
            _ => Some(HostType::Unknown),
        },
        "print" => Some(HostType::Unit),
        "fail" => Some(HostType::Unknown),
        "debug" => arg_tys.first().cloned(),
        "tensor_to_scalar" => match arg_tys.first() {
            Some(HostType::Tensor(tensor)) => Some(match tensor.precision {
                chelis_types::types::Prim::Bool => HostType::Bool,
                chelis_types::types::Prim::Int8
                | chelis_types::types::Prim::Int32
                | chelis_types::types::Prim::Int64 => HostType::Int64,
                _ => HostType::Float64,
            }),
            _ => Some(HostType::Float64),
        },
        "scalar_to_tensor" => match arg_tys.first() {
            Some(HostType::Int64) => Some(HostType::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::Int64,
            })),
            Some(HostType::Bool) => Some(HostType::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::Bool,
            })),
            Some(HostType::Float64) => Some(HostType::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::F64,
            })),
            _ => None,
        },
        "to_tensor" => match arg_tys.first() {
            Some(HostType::List(inner)) => match &**inner {
                HostType::Int64 => Some(HostType::Tensor(TensorType {
                    dims: vec![crate::dag::DimInfo::Named("list".to_string(), None)],
                    precision: chelis_types::types::Prim::Int64,
                })),
                HostType::Float64 => Some(HostType::Tensor(TensorType {
                    dims: vec![crate::dag::DimInfo::Named("list".to_string(), None)],
                    precision: chelis_types::types::Prim::F64,
                })),
                _ => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "to_list" => match arg_tys.first() {
            Some(HostType::Tensor(tensor)) if tensor.dims.len() == 1 => {
                let element_ty = match tensor.precision {
                    chelis_types::types::Prim::Bool => HostType::Bool,
                    chelis_types::types::Prim::Int8
                    | chelis_types::types::Prim::Int32
                    | chelis_types::types::Prim::Int64 => HostType::Int64,
                    chelis_types::types::Prim::F16
                    | chelis_types::types::Prim::Bf16
                    | chelis_types::types::Prim::F32
                    | chelis_types::types::Prim::F64
                    | chelis_types::types::Prim::F8e4m3 => HostType::Float64,
                    _ => HostType::Unknown,
                };
                Some(HostType::List(Box::new(element_ty)))
            }
            Some(HostType::Tensor(_)) => Some(HostType::Unknown),
            _ => Some(HostType::Unknown),
        },
        "pad_sequences" => match arg_tys.first() {
            Some(HostType::List(inner)) => match &**inner {
                HostType::List(nested) => match &**nested {
                    HostType::Int64 => Some(HostType::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::Int64,
                    })),
                    HostType::Float64 => Some(HostType::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::F64,
                    })),
                    _ => Some(HostType::Unknown),
                },
                _ => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "pad_sequences_to" => match arg_tys.first() {
            Some(HostType::List(inner)) => match &**inner {
                HostType::List(nested) => match &**nested {
                    HostType::Int64 => Some(HostType::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::Int64,
                    })),
                    HostType::Float64 => Some(HostType::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::F64,
                    })),
                    _ => Some(HostType::Unknown),
                },
                _ => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "read_file" => Some(HostType::String),
        "write_file" => Some(HostType::Unit),
        "read_lines" => Some(HostType::List(Box::new(HostType::String))),
        "read_bytes" => Some(HostType::List(Box::new(HostType::Int64))),
        "file_exists" => Some(HostType::Bool),
        "list_dir" => Some(HostType::List(Box::new(HostType::String))),
        "mmap_file" => Some(HostType::MappedFile),
        "mmap_read" => Some(HostType::List(Box::new(HostType::Int64))),
        "mmap_len" => Some(HostType::Int64),
        _ => None,
    }
}

fn lookup_adt_ctor(program: &CheckedProgram, ctor_name: &str) -> Option<(String, Vec<HostType>)> {
    lookup_adt_ctor_details(program, ctor_name).map(|(adt_name, fields)| {
        (
            adt_name,
            fields.into_iter().map(|field| field.ty).collect::<Vec<_>>(),
        )
    })
}

fn lookup_adt_ctor_details(
    program: &CheckedProgram,
    ctor_name: &str,
) -> Option<(String, Vec<HostAdtField>)> {
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("deftype") {
            continue;
        }
        let kids = children(list);
        let Some(adt_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        for variant in kids.iter().skip(2) {
            let Some(variant_list) = as_list(variant) else {
                continue;
            };
            if tag(variant_list) != Some("variant") {
                continue;
            }
            let variant_kids = children(variant_list);
            let Some(name) = variant_kids.first().and_then(symbol_name) else {
                continue;
            };
            if !terminal_name_matches(name, ctor_name) {
                continue;
            }
            let mut fields = Vec::new();
            for field in variant_kids.iter().skip(1) {
                if let Some(field_list) = as_list(field)
                    && tag(field_list) == Some("field")
                {
                    let field_kids = children(field_list);
                    if let Some(ty_expr) = field_kids.get(1) {
                        fields.push(HostAdtField {
                            name: field_kids.first().and_then(symbol_name).map(str::to_string),
                            ty: parse_host_type(ty_expr),
                        });
                    }
                } else {
                    fields.push(HostAdtField {
                        name: None,
                        ty: parse_host_type(field),
                    });
                }
            }
            return Some((adt_name.to_string(), fields));
        }
    }
    None
}

fn lookup_access_field(
    program: &CheckedProgram,
    base: &HostExpr,
    field_name: &str,
) -> Option<(usize, HostType)> {
    match base {
        HostExpr::AdtConstruct { ctor, .. } => {
            lookup_adt_ctor_details(program, ctor).and_then(|(_, fields)| {
                fields.iter().enumerate().find_map(|(index, field)| {
                    (field.name.as_deref() == Some(field_name)).then_some((index, field.ty.clone()))
                })
            })
        }
        _ => match host_expr_type(base) {
            HostType::Adt(adt_name) => lookup_adt_field_on_type(program, &adt_name, field_name),
            _ => None,
        },
    }
}

fn lookup_adt_field_on_type(
    program: &CheckedProgram,
    adt_name: &str,
    field_name: &str,
) -> Option<(usize, HostType)> {
    let mut found = None;
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("deftype") {
            continue;
        }
        let kids = children(list);
        if kids.first().and_then(symbol_name) != Some(adt_name) {
            continue;
        }
        for variant in kids.iter().skip(2) {
            let Some(variant_list) = as_list(variant) else {
                continue;
            };
            if tag(variant_list) != Some("variant") {
                continue;
            }
            for (index, field) in children(variant_list).iter().skip(1).enumerate() {
                let Some(field_list) = as_list(field) else {
                    continue;
                };
                if tag(field_list) != Some("field") {
                    continue;
                }
                if children(field_list).first().and_then(symbol_name) == Some(field_name) {
                    let ty = children(field_list)
                        .get(1)
                        .map(parse_host_type)
                        .unwrap_or(HostType::Unknown);
                    if let Some(existing) = &found
                        && existing != &(index, ty.clone())
                    {
                        return None;
                    }
                    found = Some((index, ty));
                }
            }
        }
    }
    found
}

fn tag(list: &List) -> Option<&str> {
    list.elements.first().and_then(|expr| match expr {
        Expr::Atom(Atom::Symbol(tag), _) => Some(tag.as_str()),
        _ => None,
    })
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn as_list(expr: &Expr) -> Option<&List> {
    match expr {
        Expr::List(list, _) => Some(list),
        _ => None,
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn param_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.clone()),
        Expr::MetaExpr(meta, _) => param_name(&meta.expr),
        Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .or_else(|| children(list).first().and_then(symbol_name))
            .map(str::to_string),
        _ => None,
    }
}

fn param_host_type(expr: &Expr) -> Option<HostType> {
    match expr {
        Expr::MetaExpr(meta, _) => expr_type(expr)
            .filter(|ty| *ty != HostType::Unknown)
            .or_else(|| param_host_type(&meta.expr)),
        Expr::List(_, _) => expr_type(expr).filter(|ty| *ty != HostType::Unknown),
        _ => None,
    }
}
