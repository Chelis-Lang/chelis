use std::collections::HashMap;

use chelis_deep::ast::{Atom, Expr, List};
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_ir::lower::top_level_lowering_map;
use chelis_types::{CheckedProgram, types::Prim};

use crate::schema::{ExecutionValue, TensorValue};

#[derive(Debug, Clone)]
pub(crate) struct RuntimeTensorValue {
    pub(crate) value: IrTensorValue,
    pub(crate) precision: Prim,
}

#[derive(Debug, Clone)]
pub(crate) enum RuntimeValue {
    Tensor(RuntimeTensorValue),
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Tuple(Vec<RuntimeValue>),
    Adt {
        ctor: String,
        fields: Vec<RuntimeValue>,
    },
    Closure {
        params: Vec<String>,
        body: Expr,
        env: HashMap<String, RuntimeValue>,
    },
    Unit,
}

#[derive(Debug, Default)]
pub(crate) struct RuntimeOutcome {
    pub(crate) host_bindings: HashMap<String, RuntimeValue>,
    pub(crate) transcript: Vec<String>,
}

pub(crate) fn evaluate_host_program(
    program: &CheckedProgram,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
) -> Result<RuntimeOutcome, String> {
    let lowered_names = top_level_lowering_map(program.exprs(), program.type_env());
    let mut ctx = EvalContext {
        bindings: HashMap::new(),
        tensor_bindings,
        transcript: Vec::new(),
    };

    for expr in program.exprs() {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("def") {
            continue;
        }
        if children(list)
            .first()
            .and_then(symbol_name)
            .and_then(|name| lowered_names.get(name))
            .copied()
            .unwrap_or(false)
        {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        let value = ctx.eval_expr(body)?;
        ctx.bindings.insert(name.to_string(), value);
    }

    Ok(RuntimeOutcome {
        host_bindings: ctx.bindings,
        transcript: ctx.transcript,
    })
}

pub(crate) fn runtime_value_to_schema(value: &RuntimeValue) -> ExecutionValue {
    match value {
        RuntimeValue::Tensor(tensor) => ExecutionValue::Tensor(TensorValue {
            shape: tensor.value.shape.clone(),
            data: tensor.value.data.clone(),
        }),
        RuntimeValue::Int(value) => ExecutionValue::Int(*value),
        RuntimeValue::Float(value) => ExecutionValue::Float(*value),
        RuntimeValue::Bool(value) => ExecutionValue::Bool(*value),
        RuntimeValue::String(value) => ExecutionValue::String(value.clone()),
        RuntimeValue::Tuple(items) => {
            ExecutionValue::Tuple(items.iter().map(runtime_value_to_schema).collect())
        }
        RuntimeValue::Adt { ctor, fields } => ExecutionValue::Adt {
            ctor: ctor.clone(),
            fields: fields.iter().map(runtime_value_to_schema).collect(),
        },
        RuntimeValue::Closure { .. } => ExecutionValue::String("<closure>".to_string()),
        RuntimeValue::Unit => ExecutionValue::Unit {
            kind: "unit".to_string(),
        },
    }
}

pub(crate) fn lookup_runtime_value_for_root(
    name: &str,
    host_bindings: &HashMap<String, RuntimeValue>,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
) -> Option<RuntimeValue> {
    if let Some(value) = tensor_bindings.get(name) {
        return Some(RuntimeValue::Tensor(value.clone()));
    }

    let mut parts = name.split('.');
    let head = parts.next()?;
    let mut value = host_bindings.get(head)?.clone();
    for part in parts {
        let index = part.parse::<usize>().ok()?;
        value = match value {
            RuntimeValue::Tuple(items) => items.get(index)?.clone(),
            _ => return None,
        };
    }
    Some(value)
}

struct EvalContext<'a> {
    bindings: HashMap<String, RuntimeValue>,
    tensor_bindings: &'a HashMap<String, RuntimeTensorValue>,
    transcript: Vec<String>,
}

impl<'a> EvalContext<'a> {
    fn eval_expr(&mut self, expr: &Expr) -> Result<RuntimeValue, String> {
        match expr {
            Expr::Atom(_, _) => Err("bare atom is not a runtime expression".to_string()),
            Expr::Map(_, _) => Ok(RuntimeValue::Unit),
            Expr::MetaExpr(meta, _) => self.eval_expr(&meta.expr),
            Expr::List(list, _) => self.eval_list(list),
        }
    }

    fn eval_list(&mut self, list: &List) -> Result<RuntimeValue, String> {
        match tag(list) {
            Some("lit") => self.eval_lit(list),
            Some("var") => self.eval_var(list),
            Some("app") => self.eval_app(list),
            Some("if") => self.eval_if(list),
            Some("let") => self.eval_let(list),
            Some("tuple") => Ok(RuntimeValue::Tuple(
                children(list)
                    .iter()
                    .map(|child| self.eval_expr(child))
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            Some("tuple-get") => self.eval_tuple_get(list),
            Some("match") => self.eval_match(list),
            Some("fn") => self.eval_fn(list),
            Some("pipe") => self.eval_pipe(list),
            Some("cast") => self.eval_cast(list),
            Some("handle-effect") => {
                let kids = children(list);
                self.eval_expr(
                    kids.get(2)
                        .ok_or_else(|| "handle-effect missing body".to_string())?,
                )
            }
            other => Err(format!(
                "host runtime does not support `{}`",
                other.unwrap_or("?")
            )),
        }
    }

    fn eval_lit(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let value = children(list)
            .first()
            .ok_or_else(|| "lit missing value".to_string())?;
        match value {
            Expr::Atom(Atom::Int(value), _) => Ok(RuntimeValue::Int(*value)),
            Expr::Atom(Atom::Float(value), _) => Ok(RuntimeValue::Float(*value)),
            Expr::Atom(Atom::Bool(value), _) => Ok(RuntimeValue::Bool(*value)),
            Expr::Atom(Atom::Str(value), _) => Ok(RuntimeValue::String(value.clone())),
            _ => Err("unsupported literal form".to_string()),
        }
    }

    fn eval_var(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let name = children(list)
            .first()
            .and_then(symbol_name)
            .ok_or_else(|| "var missing name".to_string())?;
        if let Some(value) = self.bindings.get(name) {
            return Ok(value.clone());
        }
        if let Some(value) = self.tensor_bindings.get(name) {
            return Ok(RuntimeValue::Tensor(value.clone()));
        }
        if name.chars().next().is_some_and(|ch| ch.is_uppercase()) {
            return Ok(RuntimeValue::Adt {
                ctor: name.to_string(),
                fields: Vec::new(),
            });
        }
        Err(format!("unknown runtime name `{name}`"))
    }

    fn eval_fn(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let params_list = kids
            .first()
            .and_then(as_list)
            .ok_or_else(|| "fn missing params".to_string())?;
        if tag(params_list) != Some("params") {
            return Err("fn params malformed".to_string());
        }
        let params = children(params_list)
            .iter()
            .filter_map(symbol_name)
            .map(str::to_string)
            .collect::<Vec<_>>();
        let body = kids
            .get(1)
            .ok_or_else(|| "fn missing body".to_string())?
            .clone();
        Ok(RuntimeValue::Closure {
            params,
            body,
            env: self.bindings.clone(),
        })
    }

    fn eval_app(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let func = kids
            .first()
            .ok_or_else(|| "app missing function".to_string())?;
        let args = kids[1..]
            .iter()
            .map(|arg| self.eval_expr(arg))
            .collect::<Result<Vec<_>, _>>()?;

        if let Some(name) = builtin_name(func) {
            return self.eval_builtin(name, &args);
        }

        if let Some(name) = var_name(func)
            && name.chars().next().is_some_and(|ch| ch.is_uppercase())
        {
            return Ok(RuntimeValue::Adt {
                ctor: name.to_string(),
                fields: args,
            });
        }

        match self.eval_expr(func)? {
            RuntimeValue::Closure { params, body, env } => {
                if params.len() != args.len() {
                    return Err(format!(
                        "closure expected {} args, got {}",
                        params.len(),
                        args.len()
                    ));
                }
                let saved = self.bindings.clone();
                self.bindings = env;
                for (param, arg) in params.into_iter().zip(args) {
                    self.bindings.insert(param, arg);
                }
                let value = self.eval_expr(&body);
                self.bindings = saved;
                value
            }
            other => Err(format!("cannot apply non-callable value {other:?}")),
        }
    }

    fn eval_if(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let cond = self.eval_expr(kids.first().ok_or_else(|| "if missing cond".to_string())?)?;
        match cond {
            RuntimeValue::Bool(true) => self.eval_expr(
                kids.get(1)
                    .ok_or_else(|| "if missing then branch".to_string())?,
            ),
            RuntimeValue::Bool(false) => self.eval_expr(
                kids.get(2)
                    .ok_or_else(|| "if missing else branch".to_string())?,
            ),
            other => Err(format!("if condition must be bool, got {other:?}")),
        }
    }

    fn eval_let(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let bind_list = kids
            .first()
            .and_then(as_list)
            .ok_or_else(|| "let missing bindings".to_string())?;
        if tag(bind_list) != Some("bind") {
            return Err("let bindings malformed".to_string());
        }
        let saved = self.bindings.clone();
        let bind_kids = children(bind_list);
        let mut index = 0;
        while index + 1 < bind_kids.len() {
            let name = symbol_name(&bind_kids[index])
                .ok_or_else(|| "let binding must bind a name in 3c".to_string())?;
            let value = self.eval_expr(&bind_kids[index + 1])?;
            self.bindings.insert(name.to_string(), value);
            index += 2;
        }
        let body = self.eval_expr(kids.get(1).ok_or_else(|| "let missing body".to_string())?)?;
        self.bindings = saved;
        Ok(body)
    }

    fn eval_tuple_get(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let tuple = self.eval_expr(
            kids.first()
                .ok_or_else(|| "tuple-get missing tuple".to_string())?,
        )?;
        let index = children(
            as_list(
                kids.get(1)
                    .ok_or_else(|| "tuple-get missing index".to_string())?,
            )
            .ok_or_else(|| "tuple-get index must be a literal".to_string())?,
        );
        let idx = index
            .first()
            .and_then(int_value)
            .ok_or_else(|| "tuple-get index must be an int literal".to_string())?
            as usize;
        match tuple {
            RuntimeValue::Tuple(items) => items
                .get(idx)
                .cloned()
                .ok_or_else(|| format!("tuple-get index {idx} out of bounds")),
            other => Err(format!("tuple-get expects tuple input, got {other:?}")),
        }
    }

    fn eval_match(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let scrutinee = self.eval_expr(
            kids.first()
                .ok_or_else(|| "match missing scrutinee".to_string())?,
        )?;
        for arm in kids.iter().skip(1) {
            let Some(arm_list) = as_list(arm) else {
                continue;
            };
            if tag(arm_list) != Some("arm") {
                continue;
            }
            let arm_kids = children(arm_list);
            if arm_kids.len() < 3 {
                continue;
            }
            let saved = self.bindings.clone();
            if pattern_matches(&scrutinee, &arm_kids[0], &mut self.bindings)? {
                let value = self.eval_expr(&arm_kids[2]);
                self.bindings = saved;
                return value;
            }
            self.bindings = saved;
        }
        Err("non-exhaustive runtime match".to_string())
    }

    fn eval_pipe(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let mut value = self.eval_expr(
            kids.first()
                .ok_or_else(|| "pipe missing head".to_string())?,
        )?;
        for stage in kids.iter().skip(1) {
            value = self.apply_callable(stage, vec![value])?;
        }
        Ok(value)
    }

    fn apply_callable(
        &mut self,
        stage: &Expr,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, String> {
        if let Some(name) = builtin_name(stage) {
            return self.eval_builtin(name, &args);
        }
        match self.eval_expr(stage)? {
            RuntimeValue::Closure { params, body, env } => {
                if params.len() != args.len() {
                    return Err(format!(
                        "closure expected {} args, got {}",
                        params.len(),
                        args.len()
                    ));
                }
                let saved = self.bindings.clone();
                self.bindings = env;
                for (param, arg) in params.into_iter().zip(args) {
                    self.bindings.insert(param, arg);
                }
                let value = self.eval_expr(&body);
                self.bindings = saved;
                value
            }
            other => Err(format!("pipe stage is not callable: {other:?}")),
        }
    }

    fn eval_cast(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let value = self.eval_expr(
            kids.first()
                .ok_or_else(|| "cast missing value".to_string())?,
        )?;
        let target = kids
            .get(1)
            .and_then(as_list)
            .and_then(|ty| children(ty).first())
            .and_then(symbol_name)
            .ok_or_else(|| "cast missing target type".to_string())?;
        match (value, target) {
            (RuntimeValue::Int(value), "int32" | "int64") => Ok(RuntimeValue::Int(value)),
            (RuntimeValue::Int(value), "f32" | "f64") => Ok(RuntimeValue::Float(value as f64)),
            (RuntimeValue::Float(value), "f32" | "f64") => Ok(RuntimeValue::Float(value)),
            (RuntimeValue::Float(value), "int32" | "int64") => Ok(RuntimeValue::Int(value as i64)),
            (RuntimeValue::Bool(value), "bool") => Ok(RuntimeValue::Bool(value)),
            (RuntimeValue::String(value), "string") => Ok(RuntimeValue::String(value)),
            (other, _) => Err(format!("unsupported cast from {other:?}")),
        }
    }

    fn eval_builtin(&mut self, name: &str, args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
        match name {
            "add" => numeric_binop(args, |lhs, rhs| lhs + rhs),
            "sub" => numeric_binop(args, |lhs, rhs| lhs - rhs),
            "mul" => numeric_binop(args, |lhs, rhs| lhs * rhs),
            "div" => numeric_binop(args, |lhs, rhs| lhs / rhs),
            "mod" => int_binop(args, |lhs, rhs| lhs % rhs),
            "neg" => numeric_unop(args, |value| -value),
            "exp" => float_unop(args, f64::exp),
            "log" => float_unop(args, f64::ln),
            "sin" => float_unop(args, f64::sin),
            "sqrt" => float_unop(args, f64::sqrt),
            "eq" => compare_eq(args),
            "neq" => compare_eq(args).and_then(|value| match value {
                RuntimeValue::Bool(value) => Ok(RuntimeValue::Bool(!value)),
                other => Err(format!("unexpected neq result {other:?}")),
            }),
            "cmplt" => ordered_compare(args, |lhs, rhs| lhs < rhs),
            "gt" => ordered_compare(args, |lhs, rhs| lhs > rhs),
            "gte" => ordered_compare(args, |lhs, rhs| lhs >= rhs),
            "lte" => ordered_compare(args, |lhs, rhs| lhs <= rhs),
            "and" => bool_binop(args, |lhs, rhs| lhs && rhs),
            "or" => bool_binop(args, |lhs, rhs| lhs || rhs),
            "not" => bool_unop(args, |value| !value),
            "bitand" => int_binop(args, |lhs, rhs| lhs & rhs),
            "bitor" => int_binop(args, |lhs, rhs| lhs | rhs),
            "bitxor" => int_binop(args, |lhs, rhs| lhs ^ rhs),
            "shl" => int_shift_binop(args, |lhs, rhs| lhs << rhs),
            "shr" => int_shift_binop(args, |lhs, rhs| lhs >> rhs),
            "string_len" => {
                let value = expect_string_arg(args, 0)?;
                Ok(RuntimeValue::Int(value.chars().count() as i64))
            }
            "string_concat" => Ok(RuntimeValue::String(format!(
                "{}{}",
                expect_string_arg(args, 0)?,
                expect_string_arg(args, 1)?
            ))),
            "string_slice" => {
                let value = expect_string_arg(args, 0)?;
                let start = expect_int_arg(args, 1)?;
                let len = expect_int_arg(args, 2)?;
                if start < 0 || len < 0 {
                    return Err("string_slice requires non-negative start and length".to_string());
                }
                let chars = value.chars().collect::<Vec<_>>();
                let start = start as usize;
                let len = len as usize;
                if start >= chars.len() {
                    return Ok(RuntimeValue::String(String::new()));
                }
                let end = start.saturating_add(len).min(chars.len());
                Ok(RuntimeValue::String(chars[start..end].iter().collect()))
            }
            "string_contains" => Ok(RuntimeValue::Bool(
                expect_string_arg(args, 0)?.contains(&expect_string_arg(args, 1)?),
            )),
            "string_starts_with" => Ok(RuntimeValue::Bool(
                expect_string_arg(args, 0)?.starts_with(&expect_string_arg(args, 1)?),
            )),
            "string_ends_with" => Ok(RuntimeValue::Bool(
                expect_string_arg(args, 0)?.ends_with(&expect_string_arg(args, 1)?),
            )),
            "string_trim" => Ok(RuntimeValue::String(
                expect_string_arg(args, 0)?.trim().to_string(),
            )),
            "to_string" => Ok(RuntimeValue::String(render_value(
                args.first()
                    .ok_or_else(|| "to_string expects 1 argument".to_string())?,
            ))),
            "to_int" => {
                let value = expect_string_arg(args, 0)?;
                Ok(match value.trim().parse::<i64>() {
                    Ok(parsed) => RuntimeValue::Adt {
                        ctor: "Some".to_string(),
                        fields: vec![RuntimeValue::Int(parsed)],
                    },
                    Err(_) => RuntimeValue::Adt {
                        ctor: "None".to_string(),
                        fields: Vec::new(),
                    },
                })
            }
            "to_float" => {
                let value = expect_string_arg(args, 0)?;
                Ok(match value.trim().parse::<f64>() {
                    Ok(parsed) => RuntimeValue::Adt {
                        ctor: "Some".to_string(),
                        fields: vec![RuntimeValue::Float(parsed)],
                    },
                    Err(_) => RuntimeValue::Adt {
                        ctor: "None".to_string(),
                        fields: Vec::new(),
                    },
                })
            }
            "rank" => {
                let tensor = expect_tensor_arg(args, 0)?;
                Ok(RuntimeValue::Int(tensor.value.shape.len() as i64))
            }
            "shape" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                if axis < 0 {
                    return Err(format!("shape requires non-negative axis, got {axis}"));
                }
                let axis = axis as usize;
                let dim = tensor
                    .value
                    .shape
                    .get(axis)
                    .copied()
                    .ok_or_else(|| format!("shape axis {axis} out of bounds"))?;
                Ok(RuntimeValue::Int(dim as i64))
            }
            "numel" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let numel = tensor.value.shape.iter().product::<usize>().max(1);
                Ok(RuntimeValue::Int(numel as i64))
            }
            "tensor_to_scalar" => {
                let tensor = expect_tensor_arg(args, 0)?;
                if !tensor.value.shape.is_empty() {
                    return Err("tensor_to_scalar expects a rank-0 tensor".to_string());
                }
                let value = tensor.value.data.first().copied().unwrap_or(0.0);
                match tensor.precision {
                    Prim::Bool => Ok(RuntimeValue::Bool(value != 0.0)),
                    Prim::Int8 | Prim::Int32 | Prim::Int64 => Ok(RuntimeValue::Int(value as i64)),
                    _ => Ok(RuntimeValue::Float(value)),
                }
            }
            "scalar_to_tensor" => match args.first() {
                Some(RuntimeValue::Int(value)) => Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::scalar(*value as f64),
                    precision: Prim::Int64,
                })),
                Some(RuntimeValue::Float(value)) => Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::scalar(*value),
                    precision: Prim::F64,
                })),
                Some(RuntimeValue::Bool(value)) => Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::scalar(if *value { 1.0 } else { 0.0 }),
                    precision: Prim::Bool,
                })),
                other => Err(format!(
                    "scalar_to_tensor expects scalar input, got {other:?}"
                )),
            },
            "print" => {
                let value = args
                    .first()
                    .ok_or_else(|| "print expects 1 argument".to_string())?;
                self.transcript.push(render_value(value));
                Ok(RuntimeValue::Unit)
            }
            "debug" => {
                let value = args
                    .first()
                    .ok_or_else(|| "debug expects 1 argument".to_string())?;
                self.transcript.push(render_value(value));
                Ok(value.clone())
            }
            other => Err(format!("unsupported builtin `{other}` in host runtime")),
        }
    }
}

fn pattern_matches(
    value: &RuntimeValue,
    pattern: &Expr,
    bindings: &mut HashMap<String, RuntimeValue>,
) -> Result<bool, String> {
    let Some(list) = as_list(pattern) else {
        return Ok(false);
    };
    match tag(list) {
        Some("pat-var") => {
            if let Some(name) = children(list).first().and_then(symbol_name) {
                bindings.insert(name.to_string(), value.clone());
                return Ok(true);
            }
            Ok(false)
        }
        Some("pat-wild") => Ok(true),
        Some("pat-lit") => {
            let lit = children(list)
                .first()
                .ok_or_else(|| "pat-lit missing value".to_string())?;
            Ok(match (value, lit) {
                (RuntimeValue::Int(lhs), Expr::Atom(Atom::Int(rhs), _)) => lhs == rhs,
                (RuntimeValue::Float(lhs), Expr::Atom(Atom::Float(rhs), _)) => lhs == rhs,
                (RuntimeValue::Bool(lhs), Expr::Atom(Atom::Bool(rhs), _)) => lhs == rhs,
                (RuntimeValue::String(lhs), Expr::Atom(Atom::Str(rhs), _)) => lhs == rhs,
                _ => false,
            })
        }
        Some("pat-ctor") => {
            let kids = children(list);
            let Some(ctor) = kids.first().and_then(symbol_name) else {
                return Ok(false);
            };
            let RuntimeValue::Adt { ctor: got, fields } = value else {
                return Ok(false);
            };
            if ctor != got || kids.len().saturating_sub(1) != fields.len() {
                return Ok(false);
            }
            for (subpat, field) in kids.iter().skip(1).zip(fields) {
                if !pattern_matches(field, subpat, bindings)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Some("pat-tuple") => {
            let RuntimeValue::Tuple(items) = value else {
                return Ok(false);
            };
            if items.len() != children(list).len() {
                return Ok(false);
            }
            for (subpat, item) in children(list).iter().zip(items) {
                if !pattern_matches(item, subpat, bindings)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn numeric_binop(
    args: &[RuntimeValue],
    op: impl Fn(f64, f64) -> f64,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Int(lhs)), Some(RuntimeValue::Int(rhs))) => {
            Ok(RuntimeValue::Int(op(*lhs as f64, *rhs as f64) as i64))
        }
        (Some(RuntimeValue::Float(lhs)), Some(RuntimeValue::Float(rhs))) => {
            Ok(RuntimeValue::Float(op(*lhs, *rhs)))
        }
        other => Err(format!(
            "numeric op expects matching int or float args, got {other:?}"
        )),
    }
}

fn numeric_unop(args: &[RuntimeValue], op: impl Fn(f64) -> f64) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Int(value)) => Ok(RuntimeValue::Int(op(*value as f64) as i64)),
        Some(RuntimeValue::Float(value)) => Ok(RuntimeValue::Float(op(*value))),
        other => Err(format!(
            "numeric op expects int or float arg, got {other:?}"
        )),
    }
}

fn int_binop(args: &[RuntimeValue], op: impl Fn(i64, i64) -> i64) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Int(lhs)), Some(RuntimeValue::Int(rhs))) => {
            Ok(RuntimeValue::Int(op(*lhs, *rhs)))
        }
        other => Err(format!("integer op expects int args, got {other:?}")),
    }
}

fn int_shift_binop(
    args: &[RuntimeValue],
    op: impl Fn(i64, u32) -> i64,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Int(lhs)), Some(RuntimeValue::Int(rhs))) if *rhs >= 0 => {
            Ok(RuntimeValue::Int(op(*lhs, *rhs as u32)))
        }
        (Some(RuntimeValue::Int(_)), Some(RuntimeValue::Int(rhs))) => {
            Err(format!("shift amount must be non-negative, got {rhs}"))
        }
        other => Err(format!("shift op expects int args, got {other:?}")),
    }
}

fn float_unop(args: &[RuntimeValue], op: impl Fn(f64) -> f64) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Float(value)) => Ok(RuntimeValue::Float(op(*value))),
        other => Err(format!("float op expects float arg, got {other:?}")),
    }
}

fn compare_eq(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Int(lhs)), Some(RuntimeValue::Int(rhs))) => {
            Ok(RuntimeValue::Bool(lhs == rhs))
        }
        (Some(RuntimeValue::Float(lhs)), Some(RuntimeValue::Float(rhs))) => {
            Ok(RuntimeValue::Bool(lhs == rhs))
        }
        (Some(RuntimeValue::Bool(lhs)), Some(RuntimeValue::Bool(rhs))) => {
            Ok(RuntimeValue::Bool(lhs == rhs))
        }
        (Some(RuntimeValue::String(lhs)), Some(RuntimeValue::String(rhs))) => {
            Ok(RuntimeValue::Bool(lhs == rhs))
        }
        other => Err(format!("eq/neq expect matching scalar args, got {other:?}")),
    }
}

fn ordered_compare(
    args: &[RuntimeValue],
    cmp: impl Fn(f64, f64) -> bool,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Int(lhs)), Some(RuntimeValue::Int(rhs))) => {
            Ok(RuntimeValue::Bool(cmp(*lhs as f64, *rhs as f64)))
        }
        (Some(RuntimeValue::Float(lhs)), Some(RuntimeValue::Float(rhs))) => {
            Ok(RuntimeValue::Bool(cmp(*lhs, *rhs)))
        }
        other => Err(format!(
            "ordered comparison expects matching numeric args, got {other:?}"
        )),
    }
}

fn bool_binop(
    args: &[RuntimeValue],
    op: impl Fn(bool, bool) -> bool,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Bool(lhs)), Some(RuntimeValue::Bool(rhs))) => {
            Ok(RuntimeValue::Bool(op(*lhs, *rhs)))
        }
        other => Err(format!("bool op expects bool args, got {other:?}")),
    }
}

fn bool_unop(args: &[RuntimeValue], op: impl Fn(bool) -> bool) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Bool(value)) => Ok(RuntimeValue::Bool(op(*value))),
        other => Err(format!("bool op expects bool arg, got {other:?}")),
    }
}

fn expect_tensor_arg(args: &[RuntimeValue], index: usize) -> Result<RuntimeTensorValue, String> {
    match args.get(index) {
        Some(RuntimeValue::Tensor(value)) => Ok(value.clone()),
        other => Err(format!(
            "expected tensor arg at index {index}, got {other:?}"
        )),
    }
}

fn expect_string_arg(args: &[RuntimeValue], index: usize) -> Result<String, String> {
    match args.get(index) {
        Some(RuntimeValue::String(value)) => Ok(value.clone()),
        other => Err(format!(
            "expected string arg at index {index}, got {other:?}"
        )),
    }
}

fn expect_int_arg(args: &[RuntimeValue], index: usize) -> Result<i64, String> {
    match args.get(index) {
        Some(RuntimeValue::Int(value)) => Ok(*value),
        other => Err(format!("expected int arg at index {index}, got {other:?}")),
    }
}

fn render_value(value: &RuntimeValue) -> String {
    match value {
        RuntimeValue::Tensor(tensor) => format!(
            "tensor(shape={:?}, data={:?})",
            tensor.value.shape, tensor.value.data
        ),
        RuntimeValue::Int(value) => value.to_string(),
        RuntimeValue::Float(value) => value.to_string(),
        RuntimeValue::Bool(value) => value.to_string(),
        RuntimeValue::String(value) => value.clone(),
        RuntimeValue::Tuple(items) => format!(
            "({})",
            items
                .iter()
                .map(render_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        RuntimeValue::Adt { ctor, fields } if fields.is_empty() => ctor.clone(),
        RuntimeValue::Adt { ctor, fields } => format!(
            "{}({})",
            ctor,
            fields
                .iter()
                .map(render_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        RuntimeValue::Closure { .. } => "<closure>".to_string(),
        RuntimeValue::Unit => "()".to_string(),
    }
}

fn builtin_name(expr: &Expr) -> Option<&str> {
    var_name(expr)
}

fn var_name(expr: &Expr) -> Option<&str> {
    let list = as_list(expr)?;
    if tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn as_list(expr: &Expr) -> Option<&List> {
    match expr {
        Expr::List(list, _) => Some(list),
        _ => None,
    }
}

fn tag(list: &List) -> Option<&str> {
    match list.elements.first() {
        Some(Expr::Atom(Atom::Symbol(tag), _)) => Some(tag.as_str()),
        _ => None,
    }
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn int_value(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Atom(Atom::Int(value), _) => Some(*value),
        _ => None,
    }
}
