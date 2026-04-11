//! Deep AST to RISC DAG lowering.
//!
//! Walks the Deep AST and produces a flat DAG of RISC primitive nodes.

use std::collections::{HashMap, HashSet};

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List};
use chelis_types::{CheckedProgram, LinearityInfo, types::Prim};

use crate::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, TensorType};
use crate::grad::grad_dag;
use crate::tier2;
use crate::vmap;

/// Lower a checked Phase 0e Deep program into a RISC DAG.
pub fn lower_program(program: &CheckedProgram) -> Dag {
    let program_type_env = program.type_env();
    let lowered_names = top_level_lowering_map(program.exprs(), program_type_env);
    for expr in program.exprs() {
        if top_level_expr_is_lowered(expr, program.exprs(), program_type_env) {
            assert_phase0e_lowerable(expr);
            assert_phase0e_typed(expr);
        }
    }
    let program_types = program
        .type_env()
        .iter()
        .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
        .collect();
    let program_defs = program
        .exprs()
        .iter()
        .filter_map(|expr| match expr {
            Expr::List(list, _) if matches!(list.elements.first(), Some(Expr::Atom(Atom::Symbol(tag), _)) if tag == "def") => {
                match (list.elements.get(2), list.elements.get(3)) {
                    (Some(Expr::Atom(Atom::Symbol(name), _)), Some(body)) => {
                        Some((name.clone(), body.clone()))
                    }
                    _ => None,
                }
            }
            _ => None,
        })
        .collect();
    let mut ctx = LowerCtx::new(program_types, program_defs, program.linearity().clone());
    for expr in program.exprs() {
        if top_level_expr_name(expr).and_then(|name| lowered_names.get(name).copied()) == Some(true)
            || (top_level_expr_name(expr).is_none()
                && top_level_expr_is_lowered(expr, program.exprs(), program_type_env))
        {
            ctx.lower_top_level(expr);
        }
    }
    crate::optimize::dead_code_eliminate(&ctx.dag)
}

pub fn tensor_type_from_deep(expr: &Expr) -> TensorType {
    LowerCtx::type_from_type_expr(expr)
}

pub fn lower_subexpr_program(
    expr: &Expr,
    scoped_tensor_types: HashMap<String, TensorType>,
    full_type_env: HashMap<String, Expr>,
    program_defs: HashMap<String, Expr>,
) -> Dag {
    let mut merged_types = full_type_env
        .iter()
        .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
        .collect::<HashMap<_, _>>();
    merged_types.extend(scoped_tensor_types);

    let mut ctx = LowerCtx::new(merged_types, program_defs, LinearityInfo::default());
    let value = ctx.lower_expr(expr);
    for id in value.flatten_nodes() {
        ctx.dag.add_root(id);
    }
    crate::optimize::dead_code_eliminate(&ctx.dag)
}

pub fn top_level_expr_is_lowered(
    expr: &Expr,
    program_exprs: &[Expr],
    type_env: &HashMap<String, Expr>,
) -> bool {
    let lowered_names = top_level_lowering_map(program_exprs, type_env);
    top_level_expr_is_lowered_with_names(expr, type_env, &lowered_names)
}

pub fn top_level_lowering_map(
    exprs: &[Expr],
    type_env: &HashMap<String, Expr>,
) -> HashMap<String, bool> {
    let top_level_defs = collect_top_level_defs(exprs);
    let mut cache = HashMap::new();
    let mut visiting = HashSet::new();
    for name in top_level_defs.keys() {
        let lowered = def_is_lowered(name, &top_level_defs, type_env, &mut cache, &mut visiting);
        cache.insert(name.clone(), lowered);
    }
    cache
}

fn top_level_expr_is_lowered_with_names(
    expr: &Expr,
    type_env: &HashMap<String, Expr>,
    lowered_names: &HashMap<String, bool>,
) -> bool {
    let Expr::List(list, _) = expr else {
        return true;
    };
    if get_tag(list) != Some("def") {
        return true;
    }
    let Some(name) = top_level_expr_name(expr) else {
        return true;
    };
    lowered_names
        .get(name)
        .copied()
        .unwrap_or_else(|| !type_env.get(name).is_some_and(type_is_never_lowerable))
}

fn type_is_never_lowerable(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return true;
    };
    match get_tag(list) {
        Some("t-fn") => list.elements.last().is_some_and(type_is_never_lowerable),
        Some("t-tuple") => children(list).iter().any(type_is_never_lowerable),
        Some("t-adt") | Some("t-unit") => true,
        Some("t-prim") => false,
        _ => false,
    }
}

fn type_is_scalar_primitive(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    get_tag(list) == Some("t-prim")
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn top_level_expr_name(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("def") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn expr_requires_host_runtime(expr: &Expr) -> bool {
    match expr {
        Expr::Atom(Atom::Str(_), _) => true,
        Expr::Atom(_, _) => false,
        Expr::Map(map, _) => map
            .entries
            .iter()
            .any(|(_, value)| expr_requires_host_runtime(value)),
        Expr::MetaExpr(meta, _) => {
            expr_requires_host_runtime(&meta.expr)
                || meta
                    .entries
                    .iter()
                    .any(|(_, value)| expr_requires_host_runtime(value))
        }
        Expr::List(list, _) => {
            if matches!(get_tag(list), Some("if" | "match")) {
                return true;
            }
            if let Some(name) = builtin_name(list) {
                if matches!(
                    name,
                    "print"
                        | "debug"
                        | "string_len"
                        | "string_concat"
                        | "string_slice"
                        | "string_contains"
                        | "string_starts_with"
                        | "string_ends_with"
                        | "string_trim"
                        | "to_string"
                        | "to_int"
                        | "to_float"
                        | "mod"
                        | "bitand"
                        | "bitor"
                        | "bitxor"
                        | "shl"
                        | "shr"
                        | "rank"
                        | "shape"
                        | "numel"
                        | "tensor_to_scalar"
                        | "scalar_to_tensor"
                        | "len"
                        | "index"
                        | "append"
                        | "concat"
                        | "take"
                        | "drop"
                        | "chunk"
                        | "range"
                        | "map"
                        | "filter"
                        | "fold"
                        | "scan"
                        | "partition"
                        | "flat_map"
                        | "flatten"
                        | "zip"
                        | "enumerate"
                        | "dict_of"
                        | "dict_get"
                        | "dict_contains"
                        | "dict_remove"
                        | "dict_insert"
                        | "dict_merge"
                        | "dict_keys"
                        | "dict_values"
                        | "dict_entries"
                        | "to_tensor"
                        | "to_list"
                        | "pad_sequences"
                        | "Cons"
                        | "Nil"
                ) {
                    return true;
                }
                if matches!(
                    name,
                    "add"
                        | "mul"
                        | "sub"
                        | "div"
                        | "max_elem"
                        | "min_elem"
                        | "neg"
                        | "exp"
                        | "log"
                        | "sin"
                        | "sqrt"
                        | "relu"
                        | "sigmoid"
                        | "cmplt"
                        | "gt"
                        | "gte"
                        | "lte"
                        | "eq"
                        | "neq"
                        | "and"
                        | "or"
                        | "not"
                ) && expr_type_metadata(expr).is_some_and(type_is_scalar_primitive)
                {
                    return true;
                }
            }
            list.elements.iter().any(expr_requires_host_runtime)
        }
    }
}

fn collect_top_level_defs(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut defs = HashMap::new();
    for expr in exprs {
        collect_top_level_defs_from_expr(expr, &mut defs);
    }
    defs
}

fn collect_top_level_defs_from_expr(expr: &Expr, defs: &mut HashMap<String, Expr>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("module") => {
            for child in list.elements.iter().skip(3) {
                collect_top_level_defs_from_expr(child, defs);
            }
        }
        Some("def") => {
            let kids = children(list);
            if let (Some(name), Some(body)) =
                (kids.first().and_then(symbol_name), kids.get(1).cloned())
            {
                defs.insert(name.to_string(), body);
            }
        }
        _ => {}
    }
}

fn def_is_lowered(
    name: &str,
    top_level_defs: &HashMap<String, Expr>,
    type_env: &HashMap<String, Expr>,
    cache: &mut HashMap<String, bool>,
    visiting: &mut HashSet<String>,
) -> bool {
    if let Some(lowered) = cache.get(name) {
        return *lowered;
    }
    if !visiting.insert(name.to_string()) {
        return !type_env.get(name).is_some_and(type_is_never_lowerable);
    }

    let lowered = top_level_defs.get(name).is_some_and(|body| {
        !expr_requires_host_runtime(body)
            && !expr_depends_on_nonlowerable_name(
                body,
                top_level_defs,
                type_env,
                cache,
                visiting,
                &HashSet::new(),
            )
            && !type_env.get(name).is_some_and(type_is_never_lowerable)
    });

    visiting.remove(name);
    cache.insert(name.to_string(), lowered);
    lowered
}

fn expr_depends_on_nonlowerable_name(
    expr: &Expr,
    top_level_defs: &HashMap<String, Expr>,
    type_env: &HashMap<String, Expr>,
    cache: &mut HashMap<String, bool>,
    visiting: &mut HashSet<String>,
    bound_names: &HashSet<String>,
) -> bool {
    match expr {
        Expr::Atom(_, _) => false,
        Expr::Map(map, _) => map.entries.iter().any(|(_, value)| {
            expr_depends_on_nonlowerable_name(
                value,
                top_level_defs,
                type_env,
                cache,
                visiting,
                bound_names,
            )
        }),
        Expr::MetaExpr(meta, _) => {
            expr_depends_on_nonlowerable_name(
                &meta.expr,
                top_level_defs,
                type_env,
                cache,
                visiting,
                bound_names,
            ) || meta.entries.iter().any(|(_, value)| {
                expr_depends_on_nonlowerable_name(
                    value,
                    top_level_defs,
                    type_env,
                    cache,
                    visiting,
                    bound_names,
                )
            })
        }
        Expr::List(list, _) => {
            if get_tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
                && !bound_names.contains(name)
                && top_level_defs.contains_key(name)
            {
                return !def_is_lowered(name, top_level_defs, type_env, cache, visiting);
            }
            if get_tag(list) == Some("fn") {
                let kids = children(list);
                let mut scoped = bound_names.clone();
                if let Some(Expr::List(params, _)) = kids.first() {
                    for param in children(params) {
                        collect_param_bound_names(param, &mut scoped);
                    }
                }
                return kids.get(1).is_some_and(|body| {
                    expr_depends_on_nonlowerable_name(
                        body,
                        top_level_defs,
                        type_env,
                        cache,
                        visiting,
                        &scoped,
                    )
                });
            }
            if get_tag(list) == Some("let") {
                let kids = children(list);
                let mut scoped = bound_names.clone();
                if let Some(Expr::List(bindings, _)) = kids.first()
                    && get_tag(bindings) == Some("bind")
                {
                    let binding_children = children(bindings);
                    let mut index = 0;
                    while index + 1 < binding_children.len() {
                        if expr_depends_on_nonlowerable_name(
                            &binding_children[index + 1],
                            top_level_defs,
                            type_env,
                            cache,
                            visiting,
                            &scoped,
                        ) {
                            return true;
                        }
                        if let Some(name) = symbol_name(&binding_children[index]) {
                            scoped.insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                return kids.get(1).is_some_and(|body| {
                    expr_depends_on_nonlowerable_name(
                        body,
                        top_level_defs,
                        type_env,
                        cache,
                        visiting,
                        &scoped,
                    )
                });
            }
            if get_tag(list) == Some("match") {
                let kids = children(list);
                if kids.first().is_some_and(|scrutinee| {
                    expr_depends_on_nonlowerable_name(
                        scrutinee,
                        top_level_defs,
                        type_env,
                        cache,
                        visiting,
                        bound_names,
                    )
                }) {
                    return true;
                }
                for arm in kids.iter().skip(1) {
                    let Expr::List(arm_list, _) = arm else {
                        continue;
                    };
                    if get_tag(arm_list) != Some("arm") {
                        continue;
                    }
                    let arm_children = children(arm_list);
                    let mut scoped = bound_names.clone();
                    if let Some(pattern) = arm_children.first() {
                        collect_pattern_bound_names(pattern, &mut scoped);
                    }
                    if arm_children.get(1).is_some_and(|guard| {
                        expr_depends_on_nonlowerable_name(
                            guard,
                            top_level_defs,
                            type_env,
                            cache,
                            visiting,
                            &scoped,
                        )
                    }) || arm_children.get(2).is_some_and(|body| {
                        expr_depends_on_nonlowerable_name(
                            body,
                            top_level_defs,
                            type_env,
                            cache,
                            visiting,
                            &scoped,
                        )
                    }) {
                        return true;
                    }
                }
                return false;
            }
            list.elements.iter().any(|child| {
                expr_depends_on_nonlowerable_name(
                    child,
                    top_level_defs,
                    type_env,
                    cache,
                    visiting,
                    bound_names,
                )
            })
        }
    }
}

fn collect_param_bound_names(param: &Expr, out: &mut HashSet<String>) {
    match param {
        Expr::Atom(Atom::Symbol(name), _) => {
            out.insert(name.clone());
        }
        Expr::List(list, _) => {
            if let Some(name) = list.elements.first().and_then(symbol_name) {
                out.insert(name.to_string());
            }
        }
        Expr::Map(_, _) | Expr::MetaExpr(_, _) | Expr::Atom(_, _) => {}
    }
}

fn collect_pattern_bound_names(pattern: &Expr, out: &mut HashSet<String>) {
    match pattern {
        Expr::Atom(_, _) | Expr::Map(_, _) => {}
        Expr::MetaExpr(meta, _) => collect_pattern_bound_names(&meta.expr, out),
        Expr::List(list, _) => {
            if get_tag(list) == Some("pat-var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                out.insert(name.to_string());
                return;
            }
            for child in children(list) {
                collect_pattern_bound_names(child, out);
            }
        }
    }
}

fn builtin_name(list: &List) -> Option<&str> {
    if get_tag(list) != Some("app") {
        return None;
    }
    list.elements.get(2).and_then(|expr| match expr {
        Expr::List(var_list, _) if get_tag(var_list) == Some("var") => {
            children(var_list).first().and_then(symbol_name)
        }
        _ => None,
    })
}

fn expr_type_metadata(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        _ => None,
    }
}

fn assert_phase0e_lowerable(expr: &Expr) {
    match expr {
        Expr::List(list, _) => {
            if let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
                && matches!(tag.as_str(), "if" | "match" | "par" | "jit")
            {
                panic!(
                    "`{tag}` is not representable in the Phase 0e RISC DAG; reject it before lowering"
                );
            }
            for elem in &list.elements {
                assert_phase0e_lowerable(elem);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                assert_phase0e_lowerable(value);
            }
        }
        Expr::MetaExpr(inner, _) => {
            for (_, value) in &inner.entries {
                assert_phase0e_lowerable(value);
            }
            assert_phase0e_lowerable(&inner.expr);
        }
        Expr::Atom(_, _) => {}
    }
}

fn assert_phase0e_typed(expr: &Expr) {
    match expr {
        Expr::List(list, _) => {
            if let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
                && tag == "app"
                && is_shape_sensitive_builtin_app(list)
                && !has_type_metadata(list)
            {
                let rendered = chelis_deep::printer::print_canonical(std::slice::from_ref(expr))
                    .replace('\n', " ");
                panic!(
                    "shape-sensitive Phase 0e app nodes must carry explicit type metadata before lowering: {rendered}"
                );
            }
            for elem in &list.elements {
                assert_phase0e_typed(elem);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                assert_phase0e_typed(value);
            }
        }
        Expr::MetaExpr(inner, _) => {
            for (_, value) in &inner.entries {
                assert_phase0e_typed(value);
            }
            assert_phase0e_typed(&inner.expr);
        }
        Expr::Atom(_, _) => {}
    }
}

fn has_type_metadata(list: &List) -> bool {
    matches!(list.elements.get(1), Some(Expr::Map(meta, _)) if meta.entries.iter().any(|(k, _)| k == "type"))
}

fn is_shape_sensitive_builtin_app(list: &List) -> bool {
    let func_name = match list.elements.get(2) {
        Some(Expr::List(func_list, _)) => {
            match (func_list.elements.first(), func_list.elements.get(2)) {
                (
                    Some(Expr::Atom(Atom::Symbol(tag), _)),
                    Some(Expr::Atom(Atom::Symbol(name), _)),
                ) if tag == "var" => Some(name.as_str()),
                _ => None,
            }
        }
        _ => None,
    };

    matches!(
        func_name,
        Some(
            "matmul"
                | "softmax"
                | "mean"
                | "layer_norm"
                | "conv2d"
                | "sum"
                | "max_reduce"
                | "reshape"
                | "permute"
                | "expand"
                | "pad"
                | "shrink"
                | "stride"
        )
    )
}

fn get_tag(list: &List) -> Option<&str> {
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

#[derive(Clone)]
enum CallableExpr {
    Plain(Expr),
    Vmap {
        fn_expr: Expr,
        axis: usize,
    },
    VmapGrad {
        fn_expr: Expr,
        wrt: Option<Vec<usize>>,
        axis: usize,
    },
    Grad {
        fn_expr: Expr,
        wrt: Option<Vec<usize>>,
    },
}

#[derive(Clone)]
enum LoweredValue {
    Node(NodeId),
    Tuple(Vec<LoweredValue>),
}

impl LoweredValue {
    fn expect_node(&self, context: &str) -> NodeId {
        match self {
            Self::Node(id) => *id,
            Self::Tuple(_) => panic!("{context} expected a single tensor value"),
        }
    }

    fn flatten_nodes(&self) -> Vec<NodeId> {
        match self {
            Self::Node(id) => vec![*id],
            Self::Tuple(items) => items.iter().flat_map(Self::flatten_nodes).collect(),
        }
    }

    fn tuple_get(&self, index: usize) -> Option<LoweredValue> {
        match self {
            Self::Tuple(items) => items.get(index).cloned(),
            Self::Node(_) => None,
        }
    }

    fn from_flat(template: &LoweredValue, nodes: &mut dyn Iterator<Item = NodeId>) -> LoweredValue {
        match template {
            Self::Node(_) => Self::Node(nodes.next().expect("flattened lowered value mismatch")),
            Self::Tuple(items) => Self::Tuple(
                items
                    .iter()
                    .map(|item| Self::from_flat(item, nodes))
                    .collect(),
            ),
        }
    }
}

fn extract_param_type(expr: &Expr, index: usize) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("fn") {
        return None;
    }
    let params = children(list).first()?;
    let Expr::List(params_list, _) = params else {
        return None;
    };
    let param = children(params_list).get(index)?;
    let Expr::List(param_list, _) = param else {
        return None;
    };
    match param_list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        _ => None,
    }
}

fn extract_fn_return_type(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("fn") {
        return None;
    }
    let Expr::Map(meta, _) = list.elements.get(1)? else {
        return None;
    };
    let (_, ty_expr) = meta.entries.iter().find(|(key, _)| key == "type")?;
    let Expr::List(fn_ty, _) = ty_expr else {
        return None;
    };
    if get_tag(fn_ty) != Some("t-fn") {
        return None;
    }
    children(fn_ty).last()
}

fn axis_to_front_perm(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(rank);
    perm.push(axis);
    perm.extend((0..rank).filter(|candidate| *candidate != axis));
    perm
}

fn front_to_axis_perm(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (1..rank).collect();
    perm.insert(axis, 0);
    perm
}

fn permuted_tensor_type(ty: &TensorType, axes: &[usize]) -> TensorType {
    TensorType {
        dims: axes.iter().map(|axis| ty.dims[*axis].clone()).collect(),
        precision: ty.precision,
    }
}

struct LowerCtx {
    dag: Dag,
    bindings: HashMap<String, LoweredValue>,
    program_types: HashMap<String, TensorType>,
    program_defs: HashMap<String, Expr>,
    random_seed: Option<u64>,
    linearity: LinearityInfo,
}

impl LowerCtx {
    fn new(
        program_types: HashMap<String, TensorType>,
        program_defs: HashMap<String, Expr>,
        linearity: LinearityInfo,
    ) -> Self {
        Self {
            dag: Dag::new(),
            bindings: HashMap::new(),
            program_types,
            program_defs,
            random_seed: None,
            linearity,
        }
    }

    /// Default tensor type when we don't have richer type info.
    fn default_type() -> TensorType {
        TensorType::scalar_f32()
    }

    fn attach_reuse_hint(
        &mut self,
        node: NodeId,
        app_span: Span,
        candidate_inputs: &[NodeId],
    ) -> NodeId {
        if let Some(input_index) = self.linearity.reusable_input_for_span(app_span)
            && let Some(input) = candidate_inputs.get(input_index)
        {
            self.dag.set_reusable_input(node, *input);
        }
        node
    }

    fn repair_output_type_if_default(&mut self, value: &LoweredValue, desired: &TensorType) {
        let LoweredValue::Node(id) = value else {
            return;
        };
        let Some(node) = self.dag.get(*id) else {
            return;
        };
        if node.output_type != Self::default_type() || desired == &Self::default_type() {
            return;
        }
        self.dag
            .replace_node(*id, node.op.clone(), node.inputs.clone(), desired.clone());
    }

    /// Extract a type from a metadata map if one is present, otherwise return a default.
    fn type_from_meta(meta: &[(String, Expr)]) -> TensorType {
        for (key, val) in meta {
            if key == "type" {
                return Self::type_from_type_expr(val);
            }
        }
        Self::default_type()
    }

    fn type_from_type_expr(expr: &Expr) -> TensorType {
        if let Some(prim) = Self::try_extract_prim(expr) {
            return TensorType {
                dims: vec![],
                precision: prim,
            };
        }
        if let Some(tt) = Self::try_extract_tensor_type(expr) {
            return tt;
        }
        Self::default_type()
    }

    fn try_extract_prim(expr: &Expr) -> Option<Prim> {
        // (t-prim {} f32)
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-prim"
            && let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2]
        {
            return Prim::parse_name(name);
        }
        None
    }

    fn try_extract_tensor_type(expr: &Expr) -> Option<TensorType> {
        // Flat format: (t-tensor {} dim1 dim2 ... (t-prim {} p))
        // Children after tag+meta: dimension nodes followed by a t-prim node as the last child.
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-tensor"
        {
            // elements[0] = tag, elements[1] = meta, elements[2..] = children
            let children = &list.elements[2..];
            if children.is_empty() {
                return None;
            }
            // Last child is the precision (t-prim {} name).
            let prim = Self::try_extract_prim(children.last()?)?;
            // All children before the last are dimension nodes.
            let mut dims = Vec::new();
            for child in &children[..children.len() - 1] {
                if let Some(dim) = Self::try_extract_dim(child) {
                    dims.push(dim);
                }
            }
            return Some(TensorType {
                dims,
                precision: prim,
            });
        }
        None
    }

    /// Extract a single dimension from a dimension node.
    fn try_extract_dim(expr: &Expr) -> Option<DimInfo> {
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
        {
            match tag.as_str() {
                "d-name" => {
                    if let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2] {
                        return Some(DimInfo::Named(name.clone(), None));
                    }
                }
                "d-var" => {
                    if let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2] {
                        return Some(DimInfo::Named(name.clone(), None));
                    }
                }
                "d-lit" => {
                    if let Expr::Atom(Atom::Int(n), _) = &list.elements[2] {
                        return Some(DimInfo::Lit(*n as usize));
                    }
                }
                _ => {}
            }
        }
        // Also handle bare symbols/ints for backward compat.
        match expr {
            Expr::Atom(Atom::Symbol(name), _) => Some(DimInfo::Named(name.clone(), None)),
            Expr::Atom(Atom::Int(n), _) => Some(DimInfo::Lit(*n as usize)),
            _ => None,
        }
    }

    /// Extract a list of dimensions from a `(t-dims {} dim1 dim2 ...)` expression.
    fn try_extract_dims(expr: &Expr) -> Option<Vec<DimInfo>> {
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 2
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-dims"
        {
            // Skip element [0] (tag) and [1] (empty map / metadata), parse remaining as dims.
            let mut dims = Vec::new();
            for elem in list.elements.iter().skip(2) {
                if let Some(d) = Self::try_extract_dim(elem) {
                    dims.push(d);
                }
            }
            if !dims.is_empty() {
                return Some(dims);
            }
        }
        None
    }

    fn dim_info_from_dim_expr(size: &DimExpr) -> Option<DimInfo> {
        match size {
            DimExpr::Concrete(value) => Some(DimInfo::Lit(*value)),
            DimExpr::Sym(name) => Some(DimInfo::Named(name.clone(), None)),
            DimExpr::Mul(_, _) | DimExpr::Div(_, _) => None,
        }
    }

    fn fallback_expand_type(
        &self,
        input: NodeId,
        axis: usize,
        size: &DimExpr,
    ) -> Option<TensorType> {
        let input_ty = self.dag.get(input)?.output_type.clone();
        let inserted_dim = Self::dim_info_from_dim_expr(size)?;
        let mut dims = input_ty.dims;
        if axis > dims.len() {
            return None;
        }
        dims.insert(axis, inserted_dim);
        Some(TensorType {
            dims,
            precision: input_ty.precision,
        })
    }

    fn lower_top_level(&mut self, expr: &Expr) {
        if let Expr::List(list, _) = expr
            && let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
        {
            match tag.as_str() {
                // Skip type-level declarations.
                "defsig" | "deftype" | "typealias" => return,
                _ => {}
            }
        }

        let value = self.lower_expr(expr);
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(|expr| match expr {
                Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
                _ => None,
            })
        {
            self.add_named_roots(name, &value);
        } else {
            for id in value.flatten_nodes() {
                self.dag.add_root(id);
            }
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> LoweredValue {
        match expr {
            Expr::Atom(atom, _) => self.lower_atom(atom),
            Expr::List(list, span) => self.lower_list(list, *span),
            Expr::Map(_, _) => LoweredValue::Node({
                // Bare metadata map -- shouldn't appear as an expression to lower.
                self.dag
                    .add_node(RiscOp::Const { value: 0.0 }, vec![], Self::default_type())
            }),
            Expr::MetaExpr(meta_expr, _) => self.lower_expr(&meta_expr.expr),
        }
    }

    fn add_named_roots(&mut self, prefix: &str, value: &LoweredValue) {
        match value {
            LoweredValue::Node(id) if !prefix.contains('.') => self.dag.add_root(*id),
            LoweredValue::Node(id) => {
                let output_type = self
                    .dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                let stored = self.dag.add_node(
                    RiscOp::Store {
                        name: prefix.to_string(),
                    },
                    vec![*id],
                    output_type,
                );
                self.dag.add_root(stored);
            }
            LoweredValue::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    self.add_named_roots(&format!("{prefix}.{index}"), item);
                }
            }
        }
    }

    fn lower_atom(&mut self, atom: &Atom) -> LoweredValue {
        match atom {
            Atom::Symbol(name) => {
                if let Some(value) = self.bindings.get(name) {
                    value.clone()
                } else {
                    LoweredValue::Node(self.dag.add_node(
                        RiscOp::Load { name: name.clone() },
                        vec![],
                        Self::default_type(),
                    ))
                }
            }
            Atom::Int(n) => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: *n as f64 },
                vec![],
                Self::default_type(),
            )),
            Atom::Float(f) => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: *f },
                vec![],
                Self::default_type(),
            )),
            Atom::Bool(b) => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const {
                    value: if *b { 1.0 } else { 0.0 },
                },
                vec![],
                Self::default_type(),
            )),
            Atom::Str(_) | Atom::Keyword(_) => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            )),
        }
    }

    fn lower_expr_node(&mut self, expr: &Expr, context: &str) -> NodeId {
        self.lower_expr(expr).expect_node(context)
    }

    fn lower_list(&mut self, list: &List, span: Span) -> LoweredValue {
        let elems = &list.elements;
        if elems.is_empty() {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            ));
        }

        let tag = match &elems[0] {
            Expr::Atom(Atom::Symbol(s), _) => s.as_str(),
            _ => {
                return LoweredValue::Node(self.dag.add_node(
                    RiscOp::Const { value: 0.0 },
                    vec![],
                    Self::default_type(),
                ));
            }
        };

        match tag {
            "def" => self.lower_def(elems),
            "let" => self.lower_let(elems),
            "lit" => self.lower_lit(elems),
            "var" => self.lower_var(elems),
            "app" => self.lower_app(elems, span),
            "fn" => self.lower_fn(elems),
            "pipe" => self.lower_pipe(elems),
            "cast" => self.lower_cast(elems),
            "if" => self.lower_if(elems),
            "tuple" => self.lower_tuple(elems),
            "par" => self.lower_par(elems),
            "realize" => self.lower_realize(elems),
            "copy" => self.lower_identity(elems),
            "borrow" => self.lower_identity(elems),
            "tuple-get" => self.lower_tuple_get(elems),
            "match" => self.lower_match(elems),
            "grad" => self.lower_grad(elems),
            "handle-effect" => self.lower_handle_effect(elems),
            "vmap" | "jit" => self.lower_unsupported(tag, elems),
            "defsig" | "deftype" | "typealias" => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            )),
            _ => {
                let mut last = LoweredValue::Node(self.dag.add_node(
                    RiscOp::Const { value: 0.0 },
                    vec![],
                    Self::default_type(),
                ));
                for elem in &elems[2..] {
                    last = self.lower_expr(elem);
                }
                last
            }
        }
    }

    /// `(def {meta...} name body)`
    fn lower_def(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            ));
        }
        let name = match &elems[2] {
            Expr::Atom(Atom::Symbol(s), _) => s.clone(),
            _ => String::new(),
        };
        let body_id = self.lower_expr(&elems[3]);
        if !name.is_empty() {
            self.bindings.insert(name, body_id.clone());
        }
        body_id
    }

    /// `(let {} (bind {} name1 expr1 name2 expr2 ...) body)`
    fn lower_let(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            ));
        }
        let saved = self.bindings.clone();

        // elems[2] = (bind {} name1 expr1 name2 expr2 ...)
        if let Expr::List(bind_list, _) = &elems[2] {
            // Skip tag and meta (elements[0] and [1]).
            let bind_kids = &bind_list.elements[2..];
            let mut i = 0;
            while i + 1 < bind_kids.len() {
                if let Expr::Atom(Atom::Symbol(name), _) = &bind_kids[i] {
                    let val_id = self.lower_expr(&bind_kids[i + 1]);
                    self.bindings.insert(name.clone(), val_id);
                }
                i += 2;
            }
        }

        let result = self.lower_expr(&elems[3]);
        self.bindings = saved; // Restore scope
        result
    }

    /// `(lit {type: T} value)`
    fn lower_lit(&mut self, elems: &[Expr]) -> LoweredValue {
        let ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            Self::type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        let value = if let Some(val_expr) = elems.get(2) {
            match val_expr {
                Expr::Atom(Atom::Int(n), _) => *n as f64,
                Expr::Atom(Atom::Float(f), _) => *f,
                Expr::Atom(Atom::Bool(b), _) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            }
        } else {
            0.0
        };

        LoweredValue::Node(self.dag.add_node(RiscOp::Const { value }, vec![], ty))
    }

    /// `(var {meta...} name)`
    fn lower_var(&mut self, elems: &[Expr]) -> LoweredValue {
        // C6: Extract type from metadata if available, otherwise use checked top-level type info.
        let explicit_ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            Self::type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        if let Some(Expr::Atom(Atom::Symbol(name), _)) = elems.get(2) {
            if let Some(id) = self.bindings.get(name) {
                return id.clone();
            }
            let ty = if explicit_ty == Self::default_type() {
                self.program_types.get(name).cloned().unwrap_or(explicit_ty)
            } else {
                explicit_ty
            };
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Load { name: name.clone() },
                vec![],
                ty,
            ));
        }
        LoweredValue::Node(self.dag.add_node(
            RiscOp::Const { value: 0.0 },
            vec![],
            Self::default_type(),
        ))
    }

    /// `(app {meta...} func arg1 arg2 ...)`
    fn lower_app(&mut self, elems: &[Expr], app_span: Span) -> LoweredValue {
        if elems.len() < 4 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            ));
        }

        let ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            Self::type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        // Check if func is a known built-in: (var {} name).
        if let Expr::List(func_list, _) = &elems[2]
            && let Some(Expr::Atom(Atom::Symbol(func_tag), _)) = func_list.elements.first()
            && func_tag == "var"
            && let Some(Expr::Atom(Atom::Symbol(func_name), _)) = func_list.elements.get(2)
        {
            return LoweredValue::Node(self.lower_builtin_app(
                func_name,
                &elems[3..],
                &ty,
                app_span,
            ));
        }

        if let Some(lowered) = self.try_lower_callable_app(&elems[2], &elems[3..], &ty, app_span) {
            return lowered;
        }

        // Not a recognized built-in -- lower func and args, return last.
        let mut last = self.lower_expr(&elems[2]);
        for arg in &elems[3..] {
            last = self.lower_expr(arg);
        }
        last
    }

    fn try_lower_callable_app(
        &mut self,
        func: &Expr,
        args: &[Expr],
        ty: &TensorType,
        app_span: Span,
    ) -> Option<LoweredValue> {
        match self.resolve_callable_expr(func) {
            Some(CallableExpr::Plain(fn_expr)) => {
                Some(self.lower_plain_callable_app(&fn_expr, args, app_span))
            }
            Some(CallableExpr::Vmap { fn_expr, axis }) => {
                Some(self.lower_vmap_callable_app(&fn_expr, axis, args, ty, app_span))
            }
            Some(CallableExpr::VmapGrad { fn_expr, wrt, axis }) => Some(
                self.lower_vmap_grad_callable_app(&fn_expr, wrt.as_deref(), axis, args, app_span),
            ),
            Some(CallableExpr::Grad { fn_expr, wrt }) => {
                Some(self.lower_grad_callable_app(&fn_expr, wrt.as_deref(), args, app_span))
            }
            None => None,
        }
    }

    fn resolve_callable_expr(&self, expr: &Expr) -> Option<CallableExpr> {
        let Expr::List(list, _) = expr else {
            return None;
        };
        match get_tag(list) {
            Some("fn") => Some(CallableExpr::Plain(expr.clone())),
            Some("var") => children(list)
                .first()
                .and_then(|expr| match expr {
                    Expr::Atom(Atom::Symbol(name), _) => self.program_defs.get(name),
                    _ => None,
                })
                .and_then(|body| self.resolve_callable_expr(body)),
            Some("vmap") => {
                let kids = children(list);
                let axis = kids
                    .get(1)
                    .and_then(|expr| self.extract_usize_value(expr))
                    .unwrap_or(0);
                if let Some(Expr::List(grad_list, _)) = kids.first()
                    && get_tag(grad_list) == Some("grad")
                {
                    let wrt = self.extract_grad_wrt_indices(grad_list);
                    return self
                        .resolve_callable_expr(children(grad_list).first()?)
                        .and_then(|inner| match inner {
                            CallableExpr::Plain(fn_expr) => {
                                Some(CallableExpr::VmapGrad { fn_expr, wrt, axis })
                            }
                            _ => None,
                        });
                }
                self.resolve_callable_expr(kids.first()?)
                    .and_then(|inner| match inner {
                        CallableExpr::Plain(fn_expr) => Some(CallableExpr::Vmap { fn_expr, axis }),
                        CallableExpr::Vmap { .. } => None,
                        CallableExpr::VmapGrad { .. } => None,
                        CallableExpr::Grad { fn_expr, wrt } => {
                            Some(CallableExpr::Grad { fn_expr, wrt })
                        }
                    })
            }
            Some("grad") => self
                .resolve_callable_expr(children(list).first()?)
                .and_then(|inner| match inner {
                    CallableExpr::Plain(fn_expr) => Some(CallableExpr::Grad {
                        fn_expr,
                        wrt: self.extract_grad_wrt_indices(list),
                    }),
                    _ => None,
                }),
            _ => None,
        }
    }

    fn extract_grad_wrt_indices(&self, list: &List) -> Option<Vec<usize>> {
        let wrt_expr = children(list).get(1)?;
        if let Expr::List(tuple, _) = wrt_expr
            && get_tag(tuple) == Some("tuple")
        {
            return Some(
                children(tuple)
                    .iter()
                    .filter_map(|expr| self.extract_usize_value(expr))
                    .collect(),
            );
        }
        self.extract_usize_value(wrt_expr).map(|index| vec![index])
    }

    fn is_selected_wrt(
        &self,
        index: usize,
        ty: &TensorType,
        wrt_indices: Option<&[usize]>,
    ) -> bool {
        let differentiable = ty.precision.is_float();
        match wrt_indices {
            Some(indices) => differentiable && indices.contains(&index),
            None => differentiable,
        }
    }

    fn lower_grad_callable_app(
        &mut self,
        fn_expr: &Expr,
        wrt_indices: Option<&[usize]>,
        args: &[Expr],
        app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self.lower_unrepresentable("grad", std::slice::from_ref(fn_expr));
        };
        let param_types: Vec<TensorType> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| {
                extract_param_type(fn_expr, index)
                    .map(Self::type_from_type_expr)
                    .unwrap_or_else(Self::default_type)
            })
            .collect();
        let actual_args: Vec<NodeId> = args
            .iter()
            .map(|arg| self.lower_expr_node(arg, "grad arguments"))
            .collect();
        let mut subctx = LowerCtx::new(
            self.program_types.clone(),
            self.program_defs.clone(),
            LinearityInfo::default(),
        );
        let mut wrt = Vec::new();
        for (index, (name, param_ty)) in param_names
            .iter()
            .zip(param_types.iter().cloned())
            .enumerate()
        {
            let load = subctx.dag.add_node(
                RiscOp::Load { name: name.clone() },
                vec![],
                param_ty.clone(),
            );
            if self.is_selected_wrt(index, &param_ty, wrt_indices) {
                wrt.push(load);
            }
            subctx
                .bindings
                .insert(name.clone(), LoweredValue::Node(load));
        }
        let output = subctx
            .lower_expr(body)
            .expect_node("grad requires a scalar floating output");
        subctx.dag.add_root(output);
        let grad_result = grad_dag(&subctx.dag, output, &wrt).unwrap_or_else(|| {
            panic!("`grad(...)` lowering requires a scalar floating forward output")
        });

        let arg_map = param_names
            .iter()
            .zip(actual_args.iter().copied())
            .map(|(name, arg)| (name.clone(), arg))
            .collect::<HashMap<_, _>>();
        let remap = self.splice_dag(&grad_result.dag, &arg_map);
        let flattened = wrt
            .iter()
            .filter_map(|wrt_node| grad_result.grad_nodes.get(wrt_node))
            .map(|grad_node| remap[grad_node])
            .collect::<Vec<_>>();
        match flattened.as_slice() {
            [single] => LoweredValue::Node(self.attach_reuse_hint(*single, app_span, &actual_args)),
            _ => LoweredValue::Tuple(flattened.into_iter().map(LoweredValue::Node).collect()),
        }
    }

    fn lower_plain_callable_app(
        &mut self,
        fn_expr: &Expr,
        args: &[Expr],
        _app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self
                .lower_unrepresentable("function application", std::slice::from_ref(fn_expr));
        };
        let arg_ids: Vec<LoweredValue> = args.iter().map(|arg| self.lower_expr(arg)).collect();
        let saved = self.bindings.clone();
        for (name, arg_id) in param_names.iter().zip(arg_ids.iter().cloned()) {
            self.bindings.insert(name.clone(), arg_id);
        }
        let result = self.lower_expr(body);
        self.bindings = saved;
        result
    }

    fn lower_plain_callable_with_values(
        &mut self,
        fn_expr: &Expr,
        args: &[LoweredValue],
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self
                .lower_unrepresentable("function application", std::slice::from_ref(fn_expr));
        };
        let saved = self.bindings.clone();
        for (name, arg_id) in param_names.iter().zip(args.iter().cloned()) {
            self.bindings.insert(name.clone(), arg_id);
        }
        let result = self.lower_expr(body);
        if let Some(ret_ty_expr) = extract_fn_return_type(fn_expr) {
            let ret_ty = Self::type_from_type_expr(ret_ty_expr);
            self.repair_output_type_if_default(&result, &ret_ty);
        }
        self.bindings = saved;
        result
    }

    fn lower_vmap_callable_app(
        &mut self,
        fn_expr: &Expr,
        axis: usize,
        args: &[Expr],
        _ty: &TensorType,
        app_span: Span,
    ) -> LoweredValue {
        let actual_args: Vec<NodeId> = args
            .iter()
            .map(|arg| self.lower_expr_node(arg, "vmap arguments"))
            .collect();
        self.lower_vmap_callable_with_nodes(fn_expr, axis, &actual_args, app_span)
    }

    fn lower_vmap_callable_with_nodes(
        &mut self,
        fn_expr: &Expr,
        axis: usize,
        actual_args: &[NodeId],
        app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self.lower_unrepresentable("vmap", std::slice::from_ref(fn_expr));
        };
        let param_types: Vec<TensorType> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| {
                extract_param_type(fn_expr, index)
                    .map(Self::type_from_type_expr)
                    .unwrap_or_else(Self::default_type)
            })
            .collect();
        let actual_types: Vec<TensorType> = actual_args
            .iter()
            .map(|id| {
                self.dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type)
            })
            .collect();

        let mut canonical_args = Vec::with_capacity(actual_args.len());
        let mut batch_dim = None;
        for (arg_id, arg_ty) in actual_args.iter().copied().zip(actual_types.iter()) {
            if axis < arg_ty.dims.len() {
                let perm = axis_to_front_perm(arg_ty.dims.len(), axis);
                let canon_ty = permuted_tensor_type(arg_ty, &perm);
                let canonical = if axis == 0 {
                    arg_id
                } else {
                    self.dag.add_node(
                        RiscOp::Permute { axes: perm },
                        vec![arg_id],
                        canon_ty.clone(),
                    )
                };
                batch_dim.get_or_insert_with(|| canon_ty.dims[0].clone());
                canonical_args.push(canonical);
            } else {
                canonical_args.push(arg_id);
            }
        }

        let Some(batch_dim) = batch_dim else {
            return self.lower_unrepresentable(
                "vmap with no tensor arguments",
                std::slice::from_ref(fn_expr),
            );
        };

        let mut subctx = LowerCtx::new(
            self.program_types.clone(),
            self.program_defs.clone(),
            LinearityInfo::default(),
        );
        for (name, param_expr) in param_names.iter().zip(param_types.iter().cloned()) {
            let load = subctx
                .dag
                .add_node(RiscOp::Load { name: name.clone() }, vec![], param_expr);
            subctx
                .bindings
                .insert(name.clone(), LoweredValue::Node(load));
        }
        let root_value = subctx.lower_expr(body);
        for root in root_value.flatten_nodes() {
            subctx.dag.add_root(root);
        }

        let vmapped = match vmap::vectorize_axis0(&subctx.dag, batch_dim.clone()) {
            Ok(dag) => dag,
            Err(message) => {
                panic!("`vmap` lowering failed: {message}");
            }
        };

        let mut arg_map = HashMap::new();
        for ((name, param_ty), arg_id) in param_names
            .iter()
            .zip(param_types.iter())
            .zip(canonical_args.iter().copied())
        {
            arg_map.insert(
                name.clone(),
                self.materialize_vmapped_arg(arg_id, param_ty, &batch_dim),
            );
        }

        let remap = self.splice_dag(&vmapped, &arg_map);
        let mut flattened = root_value
            .flatten_nodes()
            .into_iter()
            .map(|node| remap[&node])
            .collect::<Vec<_>>();
        if axis > 0 {
            for result in &mut flattened {
                let result_ty = self
                    .dag
                    .get(*result)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                if axis < result_ty.dims.len() {
                    let perm = front_to_axis_perm(result_ty.dims.len(), axis);
                    let perm_ty = permuted_tensor_type(&result_ty, &perm);
                    *result =
                        self.dag
                            .add_node(RiscOp::Permute { axes: perm }, vec![*result], perm_ty);
                }
            }
        }
        if flattened.len() == 1 {
            let result = self.attach_reuse_hint(flattened[0], app_span, &canonical_args);
            LoweredValue::Node(result)
        } else {
            let mut iter = flattened.into_iter();
            LoweredValue::from_flat(&root_value, &mut iter)
        }
    }

    fn lower_vmap_grad_callable_app(
        &mut self,
        fn_expr: &Expr,
        wrt_indices: Option<&[usize]>,
        axis: usize,
        args: &[Expr],
        app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self.lower_unrepresentable("vmap(grad)", std::slice::from_ref(fn_expr));
        };
        let param_types: Vec<TensorType> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| {
                extract_param_type(fn_expr, index)
                    .map(Self::type_from_type_expr)
                    .unwrap_or_else(Self::default_type)
            })
            .collect();
        let actual_args: Vec<NodeId> = args
            .iter()
            .map(|arg| self.lower_expr_node(arg, "vmap(grad) arguments"))
            .collect();
        let actual_types: Vec<TensorType> = actual_args
            .iter()
            .map(|id| {
                self.dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type)
            })
            .collect();

        let mut canonical_args = Vec::with_capacity(actual_args.len());
        let mut batch_dim = None;
        for (arg_id, arg_ty) in actual_args.iter().copied().zip(actual_types.iter()) {
            if axis < arg_ty.dims.len() {
                let perm = axis_to_front_perm(arg_ty.dims.len(), axis);
                let canon_ty = permuted_tensor_type(arg_ty, &perm);
                let canonical = if axis == 0 {
                    arg_id
                } else {
                    self.dag.add_node(
                        RiscOp::Permute { axes: perm },
                        vec![arg_id],
                        canon_ty.clone(),
                    )
                };
                batch_dim.get_or_insert_with(|| canon_ty.dims[0].clone());
                canonical_args.push(canonical);
            } else {
                canonical_args.push(arg_id);
            }
        }

        let Some(batch_dim) = batch_dim else {
            return self.lower_unrepresentable(
                "vmap(grad) with no tensor arguments",
                std::slice::from_ref(fn_expr),
            );
        };

        let mut subctx = LowerCtx::new(
            self.program_types.clone(),
            self.program_defs.clone(),
            LinearityInfo::default(),
        );
        let mut wrt = Vec::new();
        for (index, (name, param_ty)) in param_names
            .iter()
            .zip(param_types.iter().cloned())
            .enumerate()
        {
            let load = subctx.dag.add_node(
                RiscOp::Load { name: name.clone() },
                vec![],
                param_ty.clone(),
            );
            if self.is_selected_wrt(index, &param_ty, wrt_indices) {
                wrt.push(load);
            }
            subctx
                .bindings
                .insert(name.clone(), LoweredValue::Node(load));
        }

        let output = subctx
            .lower_expr(body)
            .expect_node("vmap(grad(...)) requires a scalar floating output");
        subctx.dag.add_root(output);
        let grad_result = grad_dag(&subctx.dag, output, &wrt).unwrap_or_else(|| {
            panic!("`vmap(grad(...))` lowering requires a scalar floating forward output")
        });
        let vmapped = match vmap::vectorize_axis0(&grad_result.dag, batch_dim.clone()) {
            Ok(dag) => dag,
            Err(message) => {
                panic!("`vmap(grad(...))` lowering failed: {message}");
            }
        };

        let mut arg_map = HashMap::new();
        for ((name, param_ty), arg_id) in param_names
            .iter()
            .zip(param_types.iter())
            .zip(canonical_args.iter().copied())
        {
            arg_map.insert(
                name.clone(),
                self.materialize_vmapped_arg(arg_id, param_ty, &batch_dim),
            );
        }

        let remap = self.splice_dag(&vmapped, &arg_map);
        let mut flattened = wrt
            .iter()
            .filter_map(|wrt_node| grad_result.grad_nodes.get(wrt_node))
            .map(|grad_node| remap[grad_node])
            .collect::<Vec<_>>();
        if axis > 0 {
            for result in &mut flattened {
                let result_ty = self
                    .dag
                    .get(*result)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                if axis < result_ty.dims.len() {
                    let perm = front_to_axis_perm(result_ty.dims.len(), axis);
                    let perm_ty = permuted_tensor_type(&result_ty, &perm);
                    *result =
                        self.dag
                            .add_node(RiscOp::Permute { axes: perm }, vec![*result], perm_ty);
                }
            }
        }
        match flattened.as_slice() {
            [single] => {
                LoweredValue::Node(self.attach_reuse_hint(*single, app_span, &canonical_args))
            }
            _ => LoweredValue::Tuple(flattened.into_iter().map(LoweredValue::Node).collect()),
        }
    }

    fn materialize_vmapped_arg(
        &mut self,
        arg_id: NodeId,
        original_ty: &TensorType,
        batch_dim: &DimInfo,
    ) -> NodeId {
        let actual_ty = self
            .dag
            .get(arg_id)
            .map(|node| node.output_type.clone())
            .unwrap_or_else(Self::default_type);
        if actual_ty.dims.len() > original_ty.dims.len() {
            return arg_id;
        }

        let mut out_ty = actual_ty.clone();
        out_ty.dims.insert(0, batch_dim.clone());
        self.dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::from(batch_dim),
            },
            vec![arg_id],
            out_ty,
        )
    }

    fn splice_dag(
        &mut self,
        dag: &Dag,
        arg_map: &HashMap<String, NodeId>,
    ) -> HashMap<NodeId, NodeId> {
        let mut remap = HashMap::<NodeId, NodeId>::new();
        for node in dag.nodes() {
            let new_id = match &node.op {
                RiscOp::Load { name } => {
                    if let Some(existing) = arg_map.get(name) {
                        *existing
                    } else {
                        self.dag.add_node(
                            RiscOp::Load { name: name.clone() },
                            vec![],
                            node.output_type.clone(),
                        )
                    }
                }
                op => {
                    let inputs = node.inputs.iter().map(|id| remap[id]).collect::<Vec<_>>();
                    let new_id = self
                        .dag
                        .add_node(op.clone(), inputs, node.output_type.clone());
                    if let Some(reusable_input) = node.reusable_input
                        && let Some(mapped_input) = remap.get(&reusable_input)
                    {
                        self.dag.set_reusable_input(new_id, *mapped_input);
                    }
                    new_id
                }
            };
            remap.insert(node.id, new_id);
        }
        remap
    }

    fn extract_fn_parts<'a>(&self, expr: &'a Expr) -> Option<(Vec<String>, &'a Expr)> {
        let Expr::List(list, _) = expr else {
            return None;
        };
        if get_tag(list) != Some("fn") {
            return None;
        }
        let kids = children(list);
        let params = kids.first()?;
        let body = kids.get(1)?;
        let Expr::List(params_list, _) = params else {
            return None;
        };
        if get_tag(params_list) != Some("params") {
            return None;
        }
        let names = children(params_list)
            .iter()
            .filter_map(|param| match param {
                Expr::Atom(Atom::Symbol(name), _) => Some(name.clone()),
                Expr::List(list, _) => list.elements.first().and_then(|expr| match expr {
                    Expr::Atom(Atom::Symbol(name), _) => Some(name.clone()),
                    _ => None,
                }),
                _ => None,
            })
            .collect();
        Some((names, body))
    }

    fn lower_builtin_app(
        &mut self,
        func_name: &str,
        args: &[Expr],
        ty: &TensorType,
        app_span: Span,
    ) -> NodeId {
        match func_name {
            // Tier 1: binary elementwise
            "add" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "add lhs");
                let b = self.lower_expr_node(&args[1], "add rhs");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(a)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let node = self.dag.add_node(RiscOp::Add, vec![a, b], out_ty);
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            "mul" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "mul lhs");
                let b = self.lower_expr_node(&args[1], "mul rhs");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(a)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let node = self.dag.add_node(RiscOp::Mul, vec![a, b], out_ty);
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            "cmplt" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "cmplt lhs");
                let b = self.lower_expr_node(&args[1], "cmplt rhs");
                // C5: CmpLt always produces Bool output regardless of input precision.
                let bool_ty = TensorType {
                    dims: ty.dims.clone(),
                    precision: Prim::Bool,
                };
                self.dag.add_node(RiscOp::CmpLt, vec![a, b], bool_ty)
            }
            "max_elem" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "max_elem lhs");
                let b = self.lower_expr_node(&args[1], "max_elem rhs");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(a)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let node = self.dag.add_node(RiscOp::MaxElem, vec![a, b], out_ty);
                self.attach_reuse_hint(node, app_span, &[a, b])
            }

            // Tier 1: unary elementwise
            "neg" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "neg input");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let node = self.dag.add_node(RiscOp::Neg, vec![x], out_ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "exp" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "exp input");
                let node = self.lower_transcendental(RiscOp::Exp, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "log" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "log input");
                let node = self.lower_transcendental(RiscOp::Log, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "sin" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "sin input");
                let node = self.lower_transcendental(RiscOp::Sin, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "sqrt" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "sqrt input");
                let node = self.lower_transcendental(RiscOp::Sqrt, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "dropout" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "dropout input");
                let rate = self.extract_f64_value(&args[1]).unwrap_or(0.0);
                let seed = self.random_seed.unwrap_or(0);
                let node = self
                    .dag
                    .add_node(RiscOp::Dropout { rate, seed }, vec![x], ty.clone());
                self.attach_reuse_hint(node, app_span, &[x])
            }

            // Tier 2 decompositions
            "sub" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "sub lhs");
                let b = self.lower_expr_node(&args[1], "sub rhs");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(a)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let node = tier2::lower_sub(&mut self.dag, a, b, &out_ty);
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            "relu" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "relu input");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let node = tier2::lower_relu(&mut self.dag, x, &out_ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "sigmoid" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "sigmoid input");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let node = tier2::lower_sigmoid(&mut self.dag, x, &out_ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "div" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "div lhs");
                let b = self.lower_expr_node(&args[1], "div rhs");
                let out_ty = if *ty == Self::default_type() {
                    self.dag
                        .get(a)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                let node = tier2::lower_div(&mut self.dag, a, b, &out_ty);
                self.attach_reuse_hint(node, app_span, &[a, b])
            }

            // Tier 2 higher-level ops (spec §3.4, §4.1–4.2)
            "matmul" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "matmul lhs");
                let b = self.lower_expr_node(&args[1], "matmul rhs");
                let a_ty = self
                    .dag
                    .get(a)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let b_ty = self
                    .dag
                    .get(b)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                tier2::lower_matmul(&mut self.dag, a, b, &a_ty, &b_ty)
            }
            "softmax" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "softmax input");
                let axis = self.extract_axis(&args[1]);
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let node = tier2::lower_softmax(&mut self.dag, x, axis, &x_ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "mean" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "mean input");
                let axis = self.extract_axis(&args[1]);
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let node = tier2::lower_mean(&mut self.dag, x, axis, &x_ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "layer_norm" if args.len() == 3 => {
                let x = self.lower_expr_node(&args[0], "layer_norm input");
                let gamma = self.lower_expr_node(&args[1], "layer_norm gamma");
                let beta = self.lower_expr_node(&args[2], "layer_norm beta");
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let gamma_ty = self
                    .dag
                    .get(gamma)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let beta_ty = self
                    .dag
                    .get(beta)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let node = tier2::lower_layer_norm(
                    &mut self.dag,
                    x,
                    gamma,
                    beta,
                    &x_ty,
                    &gamma_ty,
                    &beta_ty,
                    1e-5,
                );
                self.attach_reuse_hint(node, app_span, &[x, gamma, beta])
            }
            "conv2d" if args.len() >= 2 => {
                let input = self.lower_expr_node(&args[0], "conv2d input");
                let kernel = self.lower_expr_node(&args[1], "conv2d kernel");
                let stride = args
                    .get(2)
                    .and_then(|expr| self.extract_usize_value(expr))
                    .unwrap_or(1);
                let padding = args
                    .get(3)
                    .and_then(|expr| self.extract_usize_value(expr))
                    .unwrap_or(0);
                let input_ty = self
                    .dag
                    .get(input)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let kernel_ty = self
                    .dag
                    .get(kernel)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                tier2::lower_conv2d(
                    &mut self.dag,
                    input,
                    kernel,
                    &input_ty,
                    &kernel_ty,
                    ty,
                    stride,
                    padding,
                )
            }

            // H1: Tier 2 comparison ops
            "gt" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "gt lhs");
                let b = self.lower_expr_node(&args[1], "gt rhs");
                tier2::lower_gt(&mut self.dag, a, b, ty)
            }
            "gte" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "gte lhs");
                let b = self.lower_expr_node(&args[1], "gte rhs");
                tier2::lower_gte(&mut self.dag, a, b, ty)
            }
            "lte" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "lte lhs");
                let b = self.lower_expr_node(&args[1], "lte rhs");
                tier2::lower_lte(&mut self.dag, a, b, ty)
            }
            "eq" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "eq lhs");
                let b = self.lower_expr_node(&args[1], "eq rhs");
                tier2::lower_eq(&mut self.dag, a, b, ty)
            }
            "neq" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "neq lhs");
                let b = self.lower_expr_node(&args[1], "neq rhs");
                tier2::lower_neq(&mut self.dag, a, b, ty)
            }
            "min_elem" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "min_elem lhs");
                let b = self.lower_expr_node(&args[1], "min_elem rhs");
                tier2::lower_min_elem(&mut self.dag, a, b, ty)
            }

            // H2: Boolean operators
            "and" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "and lhs");
                let b = self.lower_expr_node(&args[1], "and rhs");
                tier2::lower_and(&mut self.dag, a, b, ty)
            }
            "or" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "or lhs");
                let b = self.lower_expr_node(&args[1], "or rhs");
                tier2::lower_or(&mut self.dag, a, b, ty)
            }
            "not" if args.len() == 1 => {
                let a = self.lower_expr_node(&args[0], "not input");
                tier2::lower_not(&mut self.dag, a, ty)
            }

            // Tier 1: reductions
            "sum" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "sum input");
                let axis = self.extract_axis(&args[1]);
                let out_ty = if *ty == Self::default_type() {
                    let x_ty = self
                        .dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone());
                    let mut dims = x_ty.dims.clone();
                    if axis < dims.len() {
                        dims.remove(axis);
                    }
                    TensorType {
                        dims,
                        precision: x_ty.precision,
                    }
                } else {
                    ty.clone()
                };
                self.dag.add_node(RiscOp::Sum { axis }, vec![x], out_ty)
            }
            "max_reduce" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "max_reduce input");
                let axis = self.extract_axis(&args[1]);
                let out_ty = if *ty == Self::default_type() {
                    let x_ty = self
                        .dag
                        .get(x)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(|| ty.clone());
                    let mut dims = x_ty.dims.clone();
                    if axis < dims.len() {
                        dims.remove(axis);
                    }
                    TensorType {
                        dims,
                        precision: x_ty.precision,
                    }
                } else {
                    ty.clone()
                };
                self.dag
                    .add_node(RiscOp::MaxReduce { axis }, vec![x], out_ty)
            }

            // H3: Movement ops -- extract parameters from Deep AST args where possible.
            "reshape" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "reshape input");
                // Try to extract new_shape from the second arg; fall back to output type dims.
                let new_shape = if args.len() >= 2 {
                    self.extract_dim_list(&args[1])
                        .unwrap_or_else(|| ty.dims.clone())
                } else {
                    ty.dims.clone()
                };
                let out_ty = TensorType {
                    dims: new_shape.clone(),
                    precision: ty.precision,
                };
                self.dag
                    .add_node(RiscOp::Reshape { new_shape }, vec![x], out_ty)
            }
            "permute" if args.len() >= 2 => {
                let x = self.lower_expr_node(&args[0], "permute input");
                // Extract axes ordering from remaining args.
                let axes = self.extract_usize_list(&args[1..]);
                self.dag
                    .add_node(RiscOp::Permute { axes }, vec![x], ty.clone())
            }
            "expand" if args.len() >= 2 => {
                let x = self.lower_expr_node(&args[0], "expand input");
                let axis = self.extract_usize_value(&args[1]).unwrap_or(0);
                let size = if args.len() >= 3 {
                    self.extract_dim_expr_value(&args[2])
                        .unwrap_or(DimExpr::Concrete(1))
                } else {
                    DimExpr::Concrete(1)
                };
                let out_ty = if *ty == Self::default_type() {
                    self.fallback_expand_type(x, axis, &size)
                        .unwrap_or_else(|| ty.clone())
                } else {
                    ty.clone()
                };
                self.dag
                    .add_node(RiscOp::Expand { axis, size }, vec![x], out_ty)
            }
            "pad" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "pad input");
                let padding = if args.len() >= 2 {
                    self.extract_pair_list(&args[1]).unwrap_or_default()
                } else {
                    vec![]
                };
                let fill = if args.len() >= 3 {
                    self.extract_f64_value(&args[2]).unwrap_or(0.0)
                } else {
                    0.0
                };
                self.dag
                    .add_node(RiscOp::Pad { padding, fill }, vec![x], ty.clone())
            }
            "shrink" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "shrink input");
                let bounds = if args.len() >= 2 {
                    self.extract_pair_list(&args[1]).unwrap_or_default()
                } else {
                    vec![]
                };
                self.dag
                    .add_node(RiscOp::Shrink { bounds }, vec![x], ty.clone())
            }
            "stride" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "stride input");
                let strides = if args.len() >= 2 {
                    self.extract_usize_list(&args[1..])
                } else {
                    vec![]
                };
                self.dag
                    .add_node(RiscOp::Stride { strides }, vec![x], ty.clone())
            }

            // Fallback: unknown function.
            _ => {
                for arg in args {
                    self.lower_expr(arg);
                }
                self.dag.add_node(
                    RiscOp::Load {
                        name: func_name.to_string(),
                    },
                    vec![],
                    Self::default_type(),
                )
            }
        }
    }

    /// Extract an axis value from an expression (for sum/max_reduce).
    fn extract_axis(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Atom(Atom::Int(n), _) => *n as usize,
            // Handle (lit {} n) form.
            Expr::List(list, _) => {
                if let Some(Expr::Atom(Atom::Int(n), _)) = list.elements.get(2) {
                    *n as usize
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    /// Extract a single usize value from an expression.
    fn extract_usize_value(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::Atom(Atom::Int(n), _) => Some(*n as usize),
            Expr::List(list, _) => {
                if let Some(Expr::Atom(Atom::Int(n), _)) = list.elements.get(2) {
                    Some(*n as usize)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn extract_dim_expr_value(&self, expr: &Expr) -> Option<DimExpr> {
        if let Some(value) = self.extract_usize_value(expr) {
            return Some(DimExpr::Concrete(value));
        }

        match expr {
            Expr::Atom(Atom::Symbol(name), _) => Some(DimExpr::Sym(name.clone())),
            Expr::List(list, _) => match (list.elements.first(), list.elements.get(2)) {
                (
                    Some(Expr::Atom(Atom::Symbol(tag), _)),
                    Some(Expr::Atom(Atom::Symbol(name), _)),
                ) if tag == "var" => Some(DimExpr::Sym(name.clone())),
                _ => None,
            },
            _ => None,
        }
    }

    fn lower_handle_effect(&mut self, elems: &[Expr]) -> LoweredValue {
        let effect = match elems.get(1) {
            Some(Expr::Map(meta, _)) => meta
                .entries
                .iter()
                .find(|(key, _)| key == "effect")
                .and_then(|(_, value)| match value {
                    Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
                    _ => None,
                }),
            _ => None,
        };
        match effect {
            Some("random") if elems.len() >= 4 => {
                let saved_seed = self.random_seed;
                self.random_seed = self.extract_u64_value(&elems[2]).or(saved_seed);
                let result = self.lower_expr(&elems[3]);
                self.random_seed = saved_seed;
                result
            }
            Some("resource") if elems.len() >= 4 => self.lower_expr(&elems[3]),
            _ if elems.len() >= 4 => self.lower_expr(&elems[3]),
            _ => LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            )),
        }
    }

    fn extract_u64_value(&self, expr: &Expr) -> Option<u64> {
        self.extract_usize_value(expr).map(|value| value as u64)
    }

    /// Extract an f64 value from an expression.
    fn extract_f64_value(&self, expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Atom(Atom::Float(f), _) => Some(*f),
            Expr::Atom(Atom::Int(n), _) => Some(*n as f64),
            Expr::List(list, _) => {
                if let Some(Expr::Atom(Atom::Float(f), _)) = list.elements.get(2) {
                    Some(*f)
                } else if let Some(Expr::Atom(Atom::Int(n), _)) = list.elements.get(2) {
                    Some(*n as f64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Extract a list of usize values from a slice of expressions.
    fn extract_usize_list(&self, exprs: &[Expr]) -> Vec<usize> {
        let mut result = Vec::new();
        for expr in exprs {
            if let Some(v) = self.extract_usize_value(expr) {
                result.push(v);
            }
        }
        result
    }

    /// Extract dimension info list from an expression (e.g., for reshape).
    fn extract_dim_list(&self, expr: &Expr) -> Option<Vec<DimInfo>> {
        // Handle (t-dims {} dim1 dim2 ...) form.
        if let Expr::List(list, _) = expr {
            if let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
                && tag == "t-dims"
            {
                return Self::try_extract_dims(expr);
            }
            // Try as a plain list of integers.
            let mut dims = Vec::new();
            for elem in &list.elements {
                match elem {
                    Expr::Atom(Atom::Int(n), _) => dims.push(DimInfo::Lit(*n as usize)),
                    Expr::Atom(Atom::Symbol(name), _) => {
                        dims.push(DimInfo::Named(name.clone(), None));
                    }
                    _ => {}
                }
            }
            if !dims.is_empty() {
                return Some(dims);
            }
        }
        None
    }

    /// Extract a list of (usize, usize) pairs from an expression (for pad/shrink bounds).
    fn extract_pair_list(&self, expr: &Expr) -> Option<Vec<(usize, usize)>> {
        if let Expr::List(list, _) = expr {
            let mut pairs = Vec::new();
            for elem in &list.elements {
                if let Expr::List(pair_list, _) = elem {
                    let vals: Vec<usize> = pair_list
                        .elements
                        .iter()
                        .filter_map(|e| {
                            if let Expr::Atom(Atom::Int(n), _) = e {
                                Some(*n as usize)
                            } else {
                                None
                            }
                        })
                        .collect();
                    if vals.len() >= 2 {
                        pairs.push((vals[0], vals[1]));
                    }
                }
            }
            if !pairs.is_empty() {
                return Some(pairs);
            }
        }
        None
    }

    /// C4: Enforce float-only for transcendental ops (exp, log, sin, sqrt).
    /// If the input is not float, produce a Const(0) error placeholder.
    fn lower_transcendental(&mut self, op: RiscOp, x: NodeId, ty: &TensorType) -> NodeId {
        let out_ty = if *ty == Self::default_type() {
            self.dag
                .get(x)
                .map(|n| n.output_type.clone())
                .unwrap_or_else(|| ty.clone())
        } else {
            ty.clone()
        };
        let input_prec = self
            .dag
            .get(x)
            .map(|n| n.output_type.precision)
            .unwrap_or(Prim::F32);
        if input_prec.is_float() {
            self.dag.add_node(op, vec![x], out_ty)
        } else {
            // Non-float input: produce a zero constant as error placeholder.
            self.dag
                .add_node(RiscOp::Const { value: 0.0 }, vec![], out_ty)
        }
    }

    /// `(fn {} (params {} p1 p2 ...) body)`
    fn lower_fn(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            ));
        }
        let saved = self.bindings.clone();

        // Register params as Load nodes.
        if let Expr::List(params_list, _) = &elems[2] {
            for param in &params_list.elements[2..] {
                if let Expr::Atom(Atom::Symbol(name), _) = param {
                    let load_id = self.dag.add_node(
                        RiscOp::Load { name: name.clone() },
                        vec![],
                        Self::default_type(),
                    );
                    self.bindings
                        .insert(name.clone(), LoweredValue::Node(load_id));
                }
                // Handle (param {} name) form.
                if let Expr::List(param_list, _) = param
                    && !param_list.elements.is_empty()
                    && let Expr::Atom(Atom::Symbol(name), _) = &param_list.elements[0]
                {
                    let ty = if let Some(Expr::Map(meta, _)) = param_list.elements.get(1) {
                        Self::type_from_meta(&meta.entries)
                    } else {
                        Self::default_type()
                    };
                    let load_id =
                        self.dag
                            .add_node(RiscOp::Load { name: name.clone() }, vec![], ty);
                    self.bindings
                        .insert(name.clone(), LoweredValue::Node(load_id));
                }
            }
        }

        let result = self.lower_expr(&elems[3]);
        self.bindings = saved; // Restore scope
        result
    }

    /// `(pipe {} x f g ...)` -- chain: lower x, then apply f, then g, etc.
    fn lower_pipe(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 3 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            ));
        }
        let mut current = self.lower_expr(&elems[2]);
        for func_expr in &elems[3..] {
            if let Expr::List(func_list, _) = func_expr
                && let Some(Expr::Atom(Atom::Symbol(tag), _)) = func_list.elements.first()
                && tag == "var"
                && let Some(Expr::Atom(Atom::Symbol(fname), _)) = func_list.elements.get(2)
            {
                let current_node = current.expect_node("pipe stage");
                let ty = self
                    .dag
                    .get(current_node)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                current = match fname.as_str() {
                    "neg" => {
                        LoweredValue::Node(self.dag.add_node(RiscOp::Neg, vec![current_node], ty))
                    }
                    "exp" => {
                        LoweredValue::Node(self.dag.add_node(RiscOp::Exp, vec![current_node], ty))
                    }
                    "log" => {
                        LoweredValue::Node(self.dag.add_node(RiscOp::Log, vec![current_node], ty))
                    }
                    "sin" => {
                        LoweredValue::Node(self.dag.add_node(RiscOp::Sin, vec![current_node], ty))
                    }
                    "sqrt" => {
                        LoweredValue::Node(self.dag.add_node(RiscOp::Sqrt, vec![current_node], ty))
                    }
                    "relu" => {
                        LoweredValue::Node(tier2::lower_relu(&mut self.dag, current_node, &ty))
                    }
                    "sigmoid" => {
                        LoweredValue::Node(tier2::lower_sigmoid(&mut self.dag, current_node, &ty))
                    }
                    _ => current,
                };
                continue;
            }
            if let Some(callable) = self.resolve_callable_expr(func_expr) {
                current = match callable {
                    CallableExpr::Plain(fn_expr) => {
                        self.lower_plain_callable_with_values(&fn_expr, &[current.clone()])
                    }
                    CallableExpr::Vmap { fn_expr, axis } => {
                        let current_node = current.expect_node("pipe stage");
                        self.lower_vmap_callable_with_nodes(
                            &fn_expr,
                            axis,
                            &[current_node],
                            func_expr.span(),
                        )
                    }
                    CallableExpr::VmapGrad { .. } | CallableExpr::Grad { .. } => {
                        self.lower_unrepresentable("pipe stage", std::slice::from_ref(func_expr))
                    }
                };
                continue;
            }
            current = self.lower_unrepresentable("pipe stage", std::slice::from_ref(func_expr));
        }
        current
    }

    /// `(cast {} expr (t-prim {} name))` -- precision cast.
    fn lower_cast(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            ));
        }
        let x = self.lower_expr_node(&elems[2], "cast input");
        let input_ty = self
            .dag
            .get(x)
            .map(|n| n.output_type.clone())
            .unwrap_or_else(Self::default_type);
        let new_precision = if let Some(prim) = Self::try_extract_prim(&elems[3]) {
            // Handle (t-prim {} name) form.
            prim
        } else if let Expr::Atom(Atom::Symbol(pname), _) = &elems[3] {
            // Fallback: bare symbol for backward compat.
            Prim::parse_name(pname).unwrap_or(Prim::F32)
        } else {
            Prim::F32
        };
        let ty = TensorType {
            dims: input_ty.dims,
            precision: new_precision,
        };
        LoweredValue::Node(
            self.dag
                .add_node(RiscOp::Cast { new_precision }, vec![x], ty),
        )
    }

    /// `(grad {} f)` -- rejected before lowering.
    fn lower_grad(&mut self, elems: &[Expr]) -> LoweredValue {
        self.lower_unrepresentable("grad", elems)
    }

    /// `(if {} cond then else)` -- Phase 0: select via arithmetic on bools.
    fn lower_if(&mut self, elems: &[Expr]) -> LoweredValue {
        self.lower_unrepresentable("if", elems)
    }

    /// `(tuple {} elem1 elem2 ...)` -- not representable in the Phase 0 RISC DAG.
    fn lower_tuple(&mut self, elems: &[Expr]) -> LoweredValue {
        LoweredValue::Tuple(
            elems
                .iter()
                .skip(2)
                .map(|expr| self.lower_expr(expr))
                .collect(),
        )
    }

    /// `(par {} expr1 expr2 ...)` -- not representable in the Phase 0 RISC DAG.
    fn lower_par(&mut self, elems: &[Expr]) -> LoweredValue {
        self.lower_unrepresentable("par", elems)
    }

    /// `(realize {} expr)` -- explicit materialization barrier.
    fn lower_realize(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() >= 3 {
            let input = self.lower_expr(&elems[2]);
            if let LoweredValue::Tuple(items) = &input {
                return LoweredValue::Tuple(
                    items
                        .iter()
                        .map(|item| {
                            let id = item.expect_node("realize tuple leaf");
                            let output_type = self
                                .dag
                                .get(id)
                                .map(|node| node.output_type.clone())
                                .unwrap_or_else(Self::default_type);
                            LoweredValue::Node(self.dag.add_node(
                                RiscOp::Realize,
                                vec![id],
                                output_type,
                            ))
                        })
                        .collect(),
                );
            }
            let input = input.expect_node("realize input");
            let output_type = self
                .dag
                .get(input)
                .map(|node| node.output_type.clone())
                .unwrap_or_else(Self::default_type);
            LoweredValue::Node(self.dag.add_node(RiscOp::Realize, vec![input], output_type))
        } else {
            LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            ))
        }
    }

    /// `(copy {} expr)` -- identity in Phase 0/1.
    fn lower_identity(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() >= 3 {
            self.lower_expr(&elems[2])
        } else {
            LoweredValue::Node(self.dag.add_node(
                RiscOp::Const { value: 0.0 },
                vec![],
                Self::default_type(),
            ))
        }
    }

    /// `(tuple-get {} tuple_expr index)` -- not representable in the Phase 0 RISC DAG.
    fn lower_tuple_get(&mut self, elems: &[Expr]) -> LoweredValue {
        let tuple = self.lower_expr(&elems[2]);
        let index = self.extract_usize_value(&elems[3]).unwrap_or(0);
        tuple
            .tuple_get(index)
            .unwrap_or_else(|| panic!("tuple-get index {index} out of bounds during lowering"))
    }

    /// `(match {} scrutinee (arm {} pattern body) ...)` -- not representable in the Phase 0 RISC DAG.
    fn lower_match(&mut self, elems: &[Expr]) -> LoweredValue {
        self.lower_unrepresentable("match", elems)
    }

    fn lower_unrepresentable(&mut self, tag: &str, elems: &[Expr]) -> LoweredValue {
        for expr in elems.iter().skip(2) {
            let _ = self.lower_expr(expr);
        }
        panic!("`{tag}` is not representable in the Phase 0e RISC DAG")
    }

    /// Unsupported Phase 2 constructs (vmap, jit).
    fn lower_unsupported(&mut self, tag: &str, _elems: &[Expr]) -> LoweredValue {
        panic!("`{tag}` is not representable in the Phase 0e RISC DAG")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify;

    fn parse_and_lower(src: &str) -> Dag {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let checked = chelis_types::check_phase0e_program(&exprs)
            .unwrap_or_else(|result| panic!("phase 0e check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        let checked = chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"));
        lower_program(&checked)
    }

    #[test]
    fn lower_single_const() {
        let dag = parse_and_lower("(def {} x (lit {type: (t-prim {} f32)} 1.0))");
        assert_eq!(dag.len(), 1);
        assert_eq!(dag.get(NodeId(0)).unwrap().op, RiscOp::Const { value: 1.0 });
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lowering_marks_reusable_input_from_linearity_hint() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}
                   (var {} relu)
                   (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x)))
        "#;
        let dag = parse_and_lower(src);
        let node = dag
            .roots()
            .last()
            .and_then(|id| dag.get(*id))
            .expect("lowered root node");
        assert_eq!(node.reusable_input, Some(NodeId(0)));
    }

    #[test]
    fn lower_add_two_consts() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
            (def {} c (app {} (var {} add) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 3);
        let add_node = dag.get(NodeId(2)).unwrap();
        assert_eq!(add_node.op, RiscOp::Add);
        assert_eq!(add_node.inputs, vec![NodeId(0), NodeId(1)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_neg() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} y (app {} (var {} neg) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 2);
        let neg_node = dag.get(NodeId(1)).unwrap();
        assert_eq!(neg_node.op, RiscOp::Neg);
        assert_eq!(neg_node.inputs, vec![NodeId(0)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_sub_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
            (def {} c (app {} (var {} sub) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a=Const(3), b=Const(1), Neg(b), Add(a, Neg(b))
        assert_eq!(dag.len(), 4);
        assert!(verify::verify(&dag).is_empty());
        let last = dag.get(NodeId(3)).unwrap();
        assert_eq!(last.op, RiscOp::Add);
    }

    #[test]
    fn lower_relu_decomposes() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (t-prim {} f32))} -2.0))
            (def {} y (app {} (var {} relu) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(-2), Const(0), MaxElem(x, 0)
        assert_eq!(dag.len(), 3);
        assert!(verify::verify(&dag).is_empty());
        let last = dag.get(NodeId(2)).unwrap();
        assert_eq!(last.op, RiscOp::MaxElem);
    }

    #[test]
    fn lower_let_binding() {
        let src = r#"
            (let {} (bind {} x (lit {type: (t-tensor {} (t-prim {} f32))} 10.0))
                (app {} (var {} neg) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(10), Neg(x)
        assert_eq!(dag.len(), 2);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_unknown_var_becomes_load() {
        let src = "(def {} y (var {} weights))";
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 1);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::Load {
                name: "weights".to_string()
            }
        );
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn tuple_return_lowers_to_named_store_roots() {
        let src = r#"
            (def {} grads
              (tuple {}
                (lit {type: (t-prim {} f32)} 1.0)
                (lit {type: (t-prim {} f32)} 2.0)))
        "#;
        let dag = parse_and_lower(src);
        let roots = dag.roots();
        assert_eq!(roots.len(), 2);
        assert!(matches!(
            dag.get(roots[0]).map(|node| &node.op),
            Some(RiscOp::Store { name }) if name == "grads.0"
        ));
        assert!(matches!(
            dag.get(roots[1]).map(|node| &node.op),
            Some(RiscOp::Store { name }) if name == "grads.1"
        ));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn tuple_get_resolves_during_lowering_without_tuple_ir_node() {
        let src = r#"
            (def {} grads
              (tuple {}
                (lit {type: (t-prim {} f32)} 1.0)
                (lit {type: (t-prim {} f32)} 2.0)))
            (def {} answer
              (tuple-get {}
                (var {} grads)
                1))
        "#;
        let dag = parse_and_lower(src);
        assert!(
            dag.nodes()
                .iter()
                .all(|node| { matches!(node.op, RiscOp::Const { .. } | RiscOp::Store { .. }) })
        );
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H1: Tier 2 comparison ops lowering ---

    #[test]
    fn lower_gt_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} gt) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(b, a)
        assert_eq!(dag.len(), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        // Args are swapped: b, a
        assert_eq!(node.inputs, vec![NodeId(1), NodeId(0)]);
    }

    #[test]
    fn lower_gte_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} gte) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(a,b), Const(1), CmpLt(lt, 1)
        assert_eq!(dag.len(), 5);
        let node = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
    }

    #[test]
    fn lower_lte_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} c (app {} (var {} lte) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 5);
    }

    #[test]
    fn lower_eq_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} eq) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem, Const(1), CmpLt(or, 1)
        assert_eq!(dag.len(), 7);
    }

    #[test]
    fn lower_min_elem_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} min_elem) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, neg(a), neg(b), max(neg_a, neg_b), neg(max)
        assert_eq!(dag.len(), 6);
        let node = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert_eq!(node.op, RiscOp::Neg);
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H2: Boolean operators ---

    #[test]
    fn lower_and_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} bool))} true))
            (def {} b (lit {type: (t-tensor {} (t-prim {} bool))} false))
            (def {} c (app {} (var {} and) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, Mul(a, b)
        assert_eq!(dag.len(), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::Mul);
    }

    #[test]
    fn lower_or_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} bool))} false))
            (def {} b (lit {type: (t-tensor {} (t-prim {} bool))} true))
            (def {} c (app {} (var {} or) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, MaxElem(a, b)
        assert_eq!(dag.len(), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
    }

    #[test]
    fn lower_not_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} bool))} true))
            (def {} b (app {} (var {} not) (var {} a)))
        "#;
        let dag = parse_and_lower(src);
        // a, Const(1), CmpLt(a, 1)
        assert_eq!(dag.len(), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
    }

    // --- H3: Movement op stubs ---

    #[test]
    fn lower_reshape_recognized() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (app {type: (t-tensor {} (t-prim {} f32))} (var {} reshape) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Reshape { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_pad_recognized() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (app {type: (t-tensor {} (t-prim {} f32))} (var {} pad) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Pad { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_shrink_recognized() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (app {type: (t-tensor {} (t-prim {} f32))} (var {} shrink) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Shrink { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_stride_recognized() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (app {type: (t-tensor {} (t-prim {} f32))} (var {} stride) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Stride { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H4: sum/max_reduce lowering ---

    #[test]
    fn lower_sum_reduction() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} 1.0))
            (def {} y (app {type: (t-tensor {} (t-prim {} f32))} (var {} sum) (var {} x) (lit {} 0)))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(1), axis_const=Const(0) is lowered inline, Sum{axis:0}
        let found_sum = dag
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::Sum { axis: 0 }));
        assert!(found_sum, "expected a Sum{{axis:0}} node");
    }

    #[test]
    fn lower_max_reduce_reduction() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} 1.0))
            (def {} y (app {type: (t-tensor {} (t-prim {} f32))} (var {} max_reduce) (var {} x) (lit {} 0)))
        "#;
        let dag = parse_and_lower(src);
        let found = dag
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::MaxReduce { axis: 0 }));
        assert!(found, "expected a MaxReduce{{axis:0}} node");
    }

    // --- C5: CmpLt lowering produces Bool ---

    #[test]
    fn lower_cmplt_produces_bool_output() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
            (def {} c (app {} (var {} cmplt) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        let cmplt_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::CmpLt))
            .expect("expected CmpLt node");
        assert_eq!(
            cmplt_node.output_type.precision,
            Prim::Bool,
            "CmpLt output must be Bool"
        );
        assert!(verify::verify(&dag).is_empty());
    }

    // --- C6: type propagation from metadata ---

    #[test]
    fn lower_lit_with_type_metadata() {
        let src = "(def {} x (lit {type: (t-prim {} f64)} 3.14))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.precision, Prim::F64);
    }

    #[test]
    fn lower_var_with_type_metadata() {
        let src = "(def {} y (var {type: (t-prim {} f64)} weights))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.precision, Prim::F64);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::dag::{DimInfo, NodeId, RiscOp};
    use crate::verify;
    use chelis_types::types::Prim;

    fn parse_and_lower(src: &str) -> Dag {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let checked = chelis_types::check_phase0e_program(&exprs)
            .unwrap_or_else(|result| panic!("phase 0e check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        let checked = chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"));
        lower_program(&checked)
    }

    // Fix 1: Tensor type metadata with flat Deep shape format.
    #[test]
    fn fix1_tensor_type_flat_dims() {
        let src = "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(
            node.output_type.dims,
            vec![DimInfo::Named("batch".to_string(), None)]
        );
        assert_eq!(node.output_type.precision, Prim::F32);
    }

    #[test]
    fn fix1_tensor_type_multiple_dims() {
        let src = "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(
            node.output_type.dims,
            vec![
                DimInfo::Named("batch".to_string(), None),
                DimInfo::Named("hidden".to_string(), None),
            ]
        );
        assert_eq!(node.output_type.precision, Prim::F32);
    }

    #[test]
    fn fix1_tensor_type_lit_dim() {
        let src = "(def {} x (lit {type: (t-tensor {} (d-lit {} 512) (t-prim {} f64))} 0))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.dims, vec![DimInfo::Lit(512)]);
        assert_eq!(node.output_type.precision, Prim::F64);
    }

    // Fix 3: Lexical scoping -- let restores bindings.
    #[test]
    fn fix3_let_multiple_bindings() {
        let src = r#"
            (let {} (bind {} x (lit {type: (t-tensor {} (t-prim {} f32))} 1.0)
                           y (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
                (app {} (var {} add) (var {} x) (var {} y)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 3);
        let add_node = dag.get(NodeId(2)).unwrap();
        assert_eq!(add_node.op, RiscOp::Add);
        assert_eq!(add_node.inputs, vec![NodeId(0), NodeId(1)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn fix3_let_scope_does_not_leak() {
        let src = r#"
            (let {} (bind {} x (lit {} 1.0)) (var {} x))
            (var {} x)
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(
            matches!(&last.op, RiscOp::Load { name } if name == "x"),
            "x should not be visible after let scope"
        );
    }

    #[test]
    fn fix3_fn_scope_does_not_leak() {
        let src = r#"
            (fn {} (params {} p) (var {} p))
            (var {} p)
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(
            matches!(&last.op, RiscOp::Load { name } if name == "p"),
            "fn param p should not be visible after fn scope"
        );
    }

    #[test]
    fn typed_fn_params_preserve_tensor_shape_for_lowering() {
        let exprs = chelis_deep::parser::parse_str(
            r#"
                (fn {}
                    (params {}
                        (x {type: (t-tensor {} (d-lit {} 32) (d-lit {} 784) (t-prim {} f32))})
                        (w {type: (t-tensor {} (d-lit {} 784) (d-lit {} 128) (t-prim {} f32))}))
                    (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))}
                        (var {} matmul)
                        (var {} x)
                        (var {} w)))
            "#,
        )
        .expect("parse failed");
        let mut ctx = LowerCtx::new(HashMap::new(), HashMap::new(), LinearityInfo::default());
        let _ = ctx.lower_expr(&exprs[0]);
        let load_x = ctx
            .dag
            .nodes()
            .iter()
            .find(|node| matches!(&node.op, RiscOp::Load { name } if name == "x"))
            .expect("typed x param should lower to a Load");
        assert_eq!(
            load_x.output_type.dims,
            vec![DimInfo::Lit(32), DimInfo::Lit(784)]
        );
        let load_w = ctx
            .dag
            .nodes()
            .iter()
            .find(|node| matches!(&node.op, RiscOp::Load { name } if name == "w"))
            .expect("typed w param should lower to a Load");
        assert_eq!(
            load_w.output_type.dims,
            vec![DimInfo::Lit(784), DimInfo::Lit(128)]
        );
    }

    #[test]
    fn pipe_lambda_stage_preserves_tensor_shape_for_following_matmul() {
        let dag = parse_and_lower(
            r#"
                (def {} x
                  (var {type: (t-tensor {} (d-lit {} 32) (d-lit {} 784) (t-prim {} f32))} x))
                (def {} w1
                  (var {type: (t-tensor {} (d-lit {} 784) (d-lit {} 128) (t-prim {} f32))} w1))
                (def {} bias
                  (var {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))} bias))
                (def {} w2
                  (var {type: (t-tensor {} (d-lit {} 128) (d-lit {} 10) (t-prim {} f32))} w2))
                (def {} h1
                  (pipe {}
                    (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))}
                      (var {} matmul)
                      (var {} x)
                      (var {} w1))
                    (fn {type: (t-fn {} (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32)) (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32)))}
                      (params {} p)
                      (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))}
                        (var {} add)
                        (var {} p)
                        (var {} bias)))
                    (var {} relu)))
                (def {} out
                  (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 10) (t-prim {} f32))}
                    (var {} matmul)
                    (var {} h1)
                    (var {} w2)))
            "#,
        );
        let root = dag
            .roots()
            .last()
            .and_then(|id| dag.get(*id))
            .expect("lowered matmul root");
        assert_eq!(
            root.output_type.dims,
            vec![DimInfo::Lit(32), DimInfo::Lit(10)]
        );
    }

    // Fix 4: Unsupported constructs.
    #[test]
    #[should_panic(expected = "`grad` is not representable in the Phase 0e RISC DAG")]
    fn fix4_grad_is_rejected_before_lowering() {
        let _ = parse_and_lower("(grad {} (var {} f))");
    }

    #[test]
    #[should_panic(expected = "`vmap` is not representable in the Phase 0e RISC DAG")]
    fn fix4_vmap_is_rejected_before_lowering() {
        let _ = parse_and_lower("(vmap {} (var {} f))");
    }

    #[test]
    #[should_panic(expected = "`jit` is not supported by Phase 0e lowering")]
    fn fix4_jit_is_rejected_before_lowering() {
        let _ = parse_and_lower("(jit {} (var {} f))");
    }

    #[test]
    fn fix4_realize_lowers_to_materialization_barrier() {
        let src = "(realize {} (lit {} 42.0))";
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 2);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::Const { value: 42.0 }
        );
        assert_eq!(dag.get(NodeId(1)).unwrap().op, RiscOp::Realize);
    }

    #[test]
    fn fix4_copy_is_identity() {
        let src = "(copy {} (lit {type: (t-tensor {} (t-prim {} f32))} 7.0))";
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 1);
        assert_eq!(dag.get(NodeId(0)).unwrap().op, RiscOp::Const { value: 7.0 });
    }

    // Fix 9: Cast with (t-prim {} bf16) node.
    #[test]
    fn fix9_cast_with_tprim_node() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (cast {} (var {} x) (t-prim {} bf16)))
        "#;
        let dag = parse_and_lower(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.op,
            RiscOp::Cast {
                new_precision: Prim::Bf16
            }
        );
        assert_eq!(cast_node.output_type.precision, Prim::Bf16);
    }

    #[test]
    fn fix9_cast_bare_symbol_still_works() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (cast {} (var {} x) f16))
        "#;
        let dag = parse_and_lower(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.op,
            RiscOp::Cast {
                new_precision: Prim::F16
            }
        );
    }

    #[test]
    fn fix9_cast_preserves_input_dims() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x))
            (def {} y (cast {} (var {} x) (t-prim {} bf16)))
        "#;
        let dag = parse_and_lower(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.output_type.dims,
            vec![DimInfo::Lit(2), DimInfo::Lit(3)]
        );
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    #[should_panic(expected = "`if` is not representable in the Phase 0e RISC DAG")]
    fn unsupported_if_is_rejected_before_lowering() {
        let _ = parse_and_lower("(if {} (lit {} true) (lit {} 1.0) (lit {} 0.0))");
    }

    #[test]
    #[should_panic(expected = "`match` is not representable in the Phase 0e RISC DAG")]
    fn unsupported_match_is_rejected_before_lowering() {
        let _ = parse_and_lower("(match {} (var {} x) (arm {} (pat-var {} y) () (var {} y)))");
    }

    #[test]
    fn shape_sensitive_app_without_type_metadata_is_rejected() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} x (lit {} 1.0))
             (def {} y (app {} (var {} reshape) (var {} x)))",
        )
        .expect("parse failed");
        let err = std::panic::catch_unwind(|| {
            for expr in &exprs {
                assert_phase0e_typed(expr);
            }
        })
        .expect_err("missing type metadata should panic during lowering preflight");
        let message = if let Some(message) = err.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = err.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            String::new()
        };
        assert!(
            message.contains("shape-sensitive Phase 0e app nodes must carry explicit type metadata before lowering"),
            "unexpected panic message: {message}"
        );
    }
}
