use std::collections::HashMap;

use chelis_deep::ast::{Atom, Expr, List};
use chelis_types::CheckedProgram;

use crate::dag::TensorType;
use crate::lower::{lower_program, top_level_lowering_map};

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
    List(Box<HostType>),
    Dict(Box<HostType>, Box<HostType>),
    Tuple(Vec<HostType>),
    Tensor(TensorType),
    Option(Box<HostType>),
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
    Builtin {
        name: String,
        args: Vec<HostExpr>,
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

fn lower_host_program(
    program: &CheckedProgram,
    lowered_names: &HashMap<String, bool>,
) -> HostProgram {
    let mut host = HostProgram::default();
    let mut global_scope = HashMap::new();
    for expr in program.exprs() {
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
        if lowered_names.get(name).copied().unwrap_or(false) {
            continue;
        }
        let Some(body) = kids.get(1) else {
            continue;
        };
        let ty_expr = program.type_env().get(name);
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
                ty,
                value: value.clone(),
            });
            global_scope.insert(name.to_string(), host_expr_type(&value));
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
    let Expr::List(list, _) = body else {
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
    let (param_tys, ret_ty) = ty_expr
        .and_then(parse_fn_type_expr)
        .unwrap_or((Vec::new(), HostType::Unknown));

    let mut scope = HashMap::new();
    let mut params = Vec::new();
    for (index, param) in children(params_list).iter().enumerate() {
        let Some(pname) = param_name(param) else {
            continue;
        };
        let pty = param_tys.get(index).cloned().unwrap_or(HostType::Unknown);
        scope.insert(pname.clone(), pty.clone());
        params.push(HostParam {
            name: pname,
            ty: pty,
        });
    }

    let mut tensor_helpers = Vec::new();
    let host_body = lower_host_expr(kids.get(1)?, program, &scope, &mut tensor_helpers);
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

fn lower_host_expr(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    if let Some(tensor_ty) = expr_tensor_type(expr, program, scope) {
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
        let inputs = dag
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                crate::RiscOp::Load { name } => Some(HostTensorInput {
                    name: name.clone(),
                    ty: node.output_type.clone(),
                }),
                _ => None,
            })
            .collect::<Vec<_>>();
        let output = dag
            .roots()
            .first()
            .and_then(|id| dag.get(*id))
            .map(|node| node.output_type.clone())
            .unwrap_or_else(|| tensor_ty.clone());
        tensor_helpers.push(HostTensorHelper {
            name: helper_name,
            dag,
            inputs,
            output,
        });
        return HostExpr::TensorCall {
            helper: helper_index,
            args: collect_tensor_args(expr, program, scope, tensor_helpers),
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
                .or_else(|| scope.get(&name).cloned())
                .or_else(|| program.type_env().get(&name).map(parse_host_type))
                .unwrap_or(HostType::Unknown);
            if name == "Nil" && matches!(ty, HostType::List(_)) {
                return HostExpr::List(Vec::new(), ty);
            }
            HostExpr::Var(name, ty)
        }
        Expr::List(list, _) if tag(list) == Some("if") => {
            let kids = children(list);
            let ty = expr_host_type(expr, program, scope);
            HostExpr::If {
                cond: Box::new(lower_host_expr(&kids[0], program, scope, tensor_helpers)),
                then_expr: Box::new(lower_host_expr(&kids[1], program, scope, tensor_helpers)),
                else_expr: Box::new(lower_host_expr(&kids[2], program, scope, tensor_helpers)),
                ty,
            }
        }
        Expr::List(list, _) if tag(list) == Some("match") => {
            lower_match_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("cast") => lower_host_expr(
            children(list).first().unwrap_or(expr),
            program,
            scope,
            tensor_helpers,
        ),
        Expr::List(list, _) if tag(list) == Some("app") => {
            lower_app_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::MetaExpr(meta, _) => lower_host_expr(&meta.expr, program, scope, tensor_helpers),
        _ => HostExpr::Unit,
    }
}

fn lower_match_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let scrutinee = lower_host_expr(&kids[0], program, scope, tensor_helpers);
    let mut bind_name = "value".to_string();
    let mut some_expr = HostExpr::Unit;
    let mut none_expr = HostExpr::Unit;

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
        if let Some("pat-ctor") = tag(pattern) {
            match children(pattern).first().and_then(symbol_name) {
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
                _ => {}
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

    HostExpr::MatchOption {
        scrutinee: Box::new(scrutinee),
        bind_name,
        some_expr: Box::new(some_expr),
        none_expr: Box::new(none_expr),
        ty,
    }
}

fn lower_app_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
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
    if name == "map"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        );
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
        let ty = expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        );
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
        let ty = expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        );
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
        let ty = expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        );
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
        let ty = expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        );
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
        let ty = expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        );
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
    let ty = infer_builtin_host_type(&name, &args).unwrap_or_else(|| {
        expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        )
    });
    HostExpr::Builtin { name, args, ty }
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
                let ty = param_tys.get(index).cloned().unwrap_or(HostType::Unknown);
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
            let (param_tys, ret_ty) = program.type_env().get(name).and_then(parse_fn_type_expr)?;
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

fn collect_tensor_args(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> Vec<HostExpr> {
    let mut args = Vec::new();
    collect_tensor_arg_exprs(expr, program, scope, tensor_helpers, &mut args);
    args
}

fn collect_tensor_arg_exprs(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
    out: &mut Vec<HostExpr>,
) {
    match expr {
        Expr::List(list, _) if tag(list) == Some("var") => {
            if let Some(name) = children(list).first().and_then(symbol_name)
                && scope
                    .get(name)
                    .cloned()
                    .or_else(|| program.type_env().get(name).map(parse_host_type))
                    .is_some_and(|ty| matches!(ty, HostType::Tensor(_)))
            {
                out.push(lower_host_expr(expr, program, scope, tensor_helpers));
            }
        }
        Expr::List(list, _) => {
            for child in children(list) {
                collect_tensor_arg_exprs(child, program, scope, tensor_helpers, out);
            }
        }
        Expr::MetaExpr(meta, _) => {
            collect_tensor_arg_exprs(&meta.expr, program, scope, tensor_helpers, out);
        }
        _ => {}
    }
}

fn collect_program_defs(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut defs = HashMap::new();
    for expr in exprs {
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

fn collect_tensor_scope(scope: &HashMap<String, HostType>) -> HashMap<String, TensorType> {
    scope
        .iter()
        .filter_map(|(name, ty)| match ty {
            HostType::Tensor(tensor) => Some((name.clone(), tensor.clone())),
            _ => None,
        })
        .collect()
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
                expr_type(expr).or_else(|| {
                    scope
                        .get(name)
                        .cloned()
                        .or_else(|| program.type_env().get(name).map(parse_host_type))
                })
            })
            .unwrap_or(HostType::Unknown),
        _ => expr_type(expr).unwrap_or(HostType::Unknown),
    }
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
    let Expr::List(list, _) = expr else {
        return None;
    };
    if tag(list) != Some("t-fn") {
        return None;
    }
    let kids = children(list);
    let (ret, args) = kids.split_last()?;
    Some((
        args.iter().map(parse_host_type).collect(),
        parse_host_type(ret),
    ))
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
                _ => HostType::Unknown,
            }
        }
        Some("t-tuple") => HostType::Tuple(children(list).iter().map(parse_host_type).collect()),
        Some("t-unit") => HostType::Unit,
        _ => HostType::Unknown,
    }
}

fn option_inner_type(expr: &HostExpr) -> HostType {
    match expr {
        HostExpr::Var(_, HostType::Option(inner))
        | HostExpr::Builtin {
            ty: HostType::Option(inner),
            ..
        } => (**inner).clone(),
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
        | HostExpr::Builtin { ty, .. }
        | HostExpr::If { ty, .. }
        | HostExpr::MatchOption { ty, .. }
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

fn infer_builtin_host_type(name: &str, args: &[HostExpr]) -> Option<HostType> {
    let arg_tys = args.iter().map(host_expr_type).collect::<Vec<_>>();
    match name {
        "add" | "sub" | "mul" | "div" | "neg" => {
            if arg_tys.iter().any(|ty| matches!(ty, HostType::Float64)) {
                Some(HostType::Float64)
            } else {
                Some(HostType::Int64)
            }
        }
        "mod" | "bitand" | "bitor" | "bitxor" | "shl" | "shr" | "string_len" | "rank" | "shape"
        | "numel" => Some(HostType::Int64),
        "cmplt" | "gt" | "gte" | "lte" | "eq" | "neq" | "and" | "or" | "not"
        | "string_contains" | "string_starts_with" | "string_ends_with" => Some(HostType::Bool),
        "string_concat" | "string_trim" | "string_slice" | "to_string" => Some(HostType::String),
        "to_int" => Some(HostType::Option(Box::new(HostType::Int64))),
        "to_float" => Some(HostType::Option(Box::new(HostType::Float64))),
        "len" => Some(HostType::Int64),
        "index" => match arg_tys.first() {
            Some(HostType::List(inner)) => Some((**inner).clone()),
            _ => Some(HostType::Unknown),
        },
        "append" | "concat" => arg_tys.first().cloned(),
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
        _ => None,
    }
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
        Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .or_else(|| children(list).first().and_then(symbol_name))
            .map(str::to_string),
        _ => None,
    }
}
