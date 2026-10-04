use chelis_deep::DeepTag;
use chelis_unord::UnordMap;
use std::path::Path;

use chelis_deep::ast::{Atom, Expr, ExprCarrier};
use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_ir::host::{EntryPattern, HostDefKernel, host_def_kernel};
use chelis_ir::tier2;
use chelis_types::{
    CompareOp, ElementRef, FloatBinOp, FloatUnOp, IntBinOp, IntUnOp, StorageView, types::Prim,
};
use chelis_vocab::EffectKind;
use chelis_vocab::EffectKindDecodeError;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::host_ops::*;
use super::named_axis::*;
#[cfg(test)]
use super::system_adapter::list_dir_names_to_strings;
use super::transforms::*;
use super::*;

pub(super) fn tensor_result_producer(dag: &Dag, root: NodeId) -> Option<ResultProducer> {
    let sites = chelis_ir::axis_sources::result_extent_sites(dag, root);
    let operation = sites.first()?.operation();
    sites
        .iter()
        .all(|site| site.operation() == operation)
        .then(|| ResultProducer::tensor(operation))
}

thread_local! {
    /// Kernel plannings `def_kernel` performed (chelis#2392). A recursive
    /// program applies its helpers many times; each non-drawing helper is
    /// planned once however often it is applied.
    static DEF_KERNEL_PLANNINGS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn record_def_kernel_planning() {
    DEF_KERNEL_PLANNINGS.with(|plannings| plannings.set(plannings.get() + 1));
}

/// Kernel plannings on this thread since the last reset.
#[cfg(test)]
pub(crate) fn take_def_kernel_plannings() -> u64 {
    DEF_KERNEL_PLANNINGS.with(|plannings| plannings.replace(0))
}

/// Resolve a returned alias to the name it reads outside its local let chain.
/// Bindings are consumed backwards so every initializer sees only earlier
/// bindings; an alias retains its producer even after that name is shadowed.
fn tail_binding_name(body: &Expr) -> Option<&str> {
    let mut current = body;
    let mut visible = Vec::new();
    'value: loop {
        let (current_tag, kids) = tagged_expr_children(current)?;
        match current_tag {
            DeepTag::Var => {
                let name = kids.first().and_then(symbol_name)?;
                while let Some((bound, value)) = visible.pop() {
                    if bound == name {
                        current = value;
                        continue 'value;
                    }
                }
                return Some(name);
            }
            DeepTag::Let => {
                let (DeepTag::Bind, bind_kids) = tagged_expr_children(kids.first()?)? else {
                    return None;
                };
                for pair in bind_kids.as_chunks::<2>().0 {
                    visible.push((symbol_name(&pair[0])?, &pair[1]));
                }
                current = kids.get(1)?;
            }
            DeepTag::Block => current = kids.last()?,
            _ => return None,
        }
    }
}

/// Select the original producing binding, following aliases in sequential
/// scope. The naming walk and the guard-placement walk use this same result.
fn let_result_binding_index(bind_list: &Expr, body: &Expr) -> Option<usize> {
    let mut name = tail_binding_name(body)?;
    let (DeepTag::Bind, bind_kids) = tagged_expr_children(bind_list)? else {
        return None;
    };
    let mut found = None;
    for (index, pair) in bind_kids.as_chunks::<2>().0.iter().enumerate().rev() {
        if symbol_name(&pair[0]) == Some(name) {
            found = Some(index * 2);
            match tail_binding_name(&pair[1]) {
                Some(alias) => name = alias,
                None => break,
            }
        }
    }
    found
}

#[derive(Clone, Copy)]
struct SequentialEvalLetBinding<'expr> {
    binding: &'expr Expr,
    initializer: &'expr Expr,
    layer: usize,
}

struct SequentialEvalLetChain<'expr> {
    bindings: Vec<SequentialEvalLetBinding<'expr>>,
    bind_expr: Expr,
    body: &'expr Expr,
}

/// Flatten only the direct tail chain produced for sequential Surf bindings:
/// `(let b0 (let b1 (... body)))`. Branches and other nested expressions are
/// not traversed. This is the Eval counterpart of C host lowering's
/// `sequential_host_let_chain`: a checked local ascription in a later layer
/// must retain visibility of the producer in an earlier lexical layer.
fn sequential_eval_let_chain(expr: &Expr) -> Option<SequentialEvalLetChain<'_>> {
    let mut bindings = Vec::new();
    let mut current = expr;
    let mut layer = 0;
    let body = loop {
        let (DeepTag::Let, kids) = tagged_expr_children(current)? else {
            return None;
        };
        let bind_expr = kids.first()?;
        let (DeepTag::Bind, bind_kids) = tagged_expr_children(bind_expr)? else {
            return None;
        };
        if !bind_kids.len().is_multiple_of(2) {
            return None;
        }
        for pair in bind_kids.as_chunks::<2>().0 {
            bindings.push(SequentialEvalLetBinding {
                binding: &pair[0],
                initializer: &pair[1],
                layer,
            });
        }
        let next = kids.get(1)?;
        if matches!(tagged_expr_children(next), Some((DeepTag::Let, _))) {
            current = next;
            layer += 1;
        } else {
            break next;
        }
    };

    // Every pair keeps its binder and binding-value positions, so each
    // initializer's binding-origin metadata stays admitted.
    let mut bind_children = Vec::with_capacity(bindings.len() * 2);
    for binding in &bindings {
        bind_children.push(binding.binding.clone());
        bind_children.push(binding.initializer.clone());
    }
    Some(SequentialEvalLetChain {
        bindings,
        bind_expr: Expr::node(
            DeepTag::Bind,
            Metadata::default(),
            bind_children,
            expr.span(),
        ),
        body,
    })
}

/// chelis#1739 and chelis#1771. A host function's declared literal result
/// extents. Attribution belongs to the selected producing expression, not to
/// the declaration: two branches may produce the result with different ops.
///
/// `diagonal` and the rest of `HOST_ONLY_BUILTINS` never become a `RiscOp`, so
/// `expr_is_dag_lowerable` is false for a def that calls one and the DAG lane's
/// literal-result claim (`lower.rs`'s `preserve_literal_result`) cannot reach
/// them. Without this, `def d[n](x: tensor[n, 4, f32]) -> tensor[3, f32] =
/// diagonal(x, 0, 1)` applied at `n = 2` returned a two-element tensor under a
/// three-element declaration, on both lanes and with no diagnostic.
#[derive(Clone)]
struct DeclaredResultClaim {
    /// Declared rank. A rank disagreement is NOT this guard's business: the
    /// checker owns rank, and reporting it here would duplicate a verdict with
    /// a worse message.
    rank: usize,
    /// Every declared dimension this invocation can resolve: a literal, or
    /// a binder a tensor parameter witnesses.
    axes: Vec<ResultAxisClaim>,
    /// Declared dimensions whose binder no parameter declares, so only the
    /// body can bind it (spec/04-type-system.md section 4.4.1). Each is
    /// resolved where the claim is checked, against the first site of its
    /// activation that produced the binder's extent; one the path has not
    /// bound claims nothing, because the result is then its first site.
    /// Grouped by binder in declared order, as the C lane's frames are.
    first_sites: Vec<FirstSiteAxis>,
}

/// One declared result axis named by an output-inferred binder of
/// `activation`.
#[derive(Clone)]
struct FirstSiteAxis {
    axis: usize,
    binder: String,
    activation: u64,
}

/// The dimension-binder extents each executing activation has bound
/// (spec/04-type-system.md section 4.4.1).
///
/// A parameter axis binds its binder when the activation starts. An
/// output-inferred binder is bound by the first site, in evaluation order,
/// that produces its extent, and every later site that names it is a claim
/// against that site. The records are keyed by activation rather than kept
/// in a lexical frame: a binder first bound inside a block stays bound after
/// the block, and a callee's records never mix with its caller's.
#[derive(Default)]
pub(super) struct ActivationExtents {
    current: u64,
    next: u64,
    bound: BTreeMap<(u64, String), (NamedResultSource, i64)>,
}

impl ActivationExtents {
    /// Start a fresh activation; the returned key restores its caller.
    fn enter(&mut self) -> u64 {
        self.next += 1;
        std::mem::replace(&mut self.current, self.next)
    }

    fn exit(&mut self, caller: u64) {
        let ended = self.current;
        self.bound.retain(|(activation, _), _| *activation != ended);
        self.current = caller;
    }

    fn get(&self, activation: u64, binder: &str) -> Option<&(NamedResultSource, i64)> {
        self.bound.get(&(activation, binder.to_string()))
    }

    fn current(&self, binder: &str) -> Option<&(NamedResultSource, i64)> {
        self.get(self.current, binder)
    }

    fn bind(&mut self, binder: String, source: NamedResultSource, extent: i64) {
        self.bound.insert((self.current, binder), (source, extent));
    }
}

/// One declared result axis and the value it requires.
#[derive(Clone)]
struct ResultAxisClaim {
    axis: usize,
    required: i64,
    /// `None` for a literal. A named claim keeps its authored binder and the
    /// declaring parameter axis whose observed extent supplied `required`, so
    /// its trap context names both disagreeing sources (spec/04 section 4.7).
    source: Option<NamedResultSource>,
}

#[derive(Clone)]
struct NamedResultSource {
    claim: String,
    parameter: String,
    axis: usize,
}

impl ResultAxisClaim {
    fn literal(axis: usize, required: i64) -> Self {
        Self {
            axis,
            required,
            source: None,
        }
    }
}

impl DeclaredResultClaim {
    fn verdict(&self, produced: &RuntimeValue, op: &str) -> Result<(), String> {
        let RuntimeValue::Tensor(tensor) = produced else {
            return Ok(());
        };
        self.shape_verdict(&tensor.value.shape, op)
    }

    fn shape_verdict(&self, shape: &[usize], op: &str) -> Result<(), String> {
        if shape.len() != self.rank {
            return Ok(());
        }
        for claim in &self.axes {
            let (axis, required) = (claim.axis, claim.required);
            let observed = shape[axis];
            if required < 0 || required as usize != observed {
                let context = match &claim.source {
                    None => format!("extent `{required}`: claimed = {required}"),
                    Some(source) => format!(
                        "extent `{}`: {} axis {} = {required}",
                        source.claim, source.parameter, source.axis
                    ),
                };
                return Err(format!(
                    "{context}, {op} axis {axis} = {observed}\n\
                     numeric trap: domain in {op} at i64"
                ));
            }
        }
        Ok(())
    }
}

/// The declared rank and literal declared extents of a host function's result.
///
/// `declared` is the CHECKED result type where one exists, not the syntactic
/// annotation, so an aliased declaration carries the same verdict as its
/// expansion and agrees with the C emitter, which reads the resolved ABI type.
fn declared_literal_result_extents(declared: Option<&Expr>) -> Option<(usize, Vec<(usize, i64)>)> {
    let (_, dim_exprs) = tensor_type_dim_exprs(declared?)?;
    let mut axes = Vec::new();
    for (axis, dim_expr) in dim_exprs.iter().enumerate() {
        let Some((DeepTag::DLit, dim_kids)) = tagged_expr_children(dim_expr) else {
            continue;
        };
        if let Some(required) = dim_kids.first().and_then(int_value) {
            axes.push((axis, required));
        }
    }
    (!axes.is_empty()).then_some((dim_exprs.len(), axes))
}

fn tensor_type_dim_exprs(expr: &Expr) -> Option<(&Expr, &[Expr])> {
    let stripped = strip_type_wrappers(expr);
    let (DeepTag::TTensor, kids) = tagged_expr_children(stripped)? else {
        return None;
    };
    kids.split_last()
}

fn close_at_f32_width(actual: f32, expected: f32, tolerance: f32) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return false;
    }
    if actual == expected {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    (actual - expected).abs() <= tolerance
}

fn close_at_f64_width(actual: f64, expected: f64, tolerance: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return false;
    }
    if actual == expected {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    (actual - expected).abs() <= tolerance
}

fn first_f32_mismatch(
    actual: impl Iterator<Item = f32>,
    expected: impl Iterator<Item = f32>,
    tolerance: f32,
) -> Option<usize> {
    actual
        .zip(expected)
        .enumerate()
        .find_map(|(index, (actual, expected))| {
            (!close_at_f32_width(actual, expected, tolerance)).then_some(index)
        })
}

fn first_f64_mismatch(
    actual: impl Iterator<Item = f64>,
    expected: impl Iterator<Item = f64>,
    tolerance: f64,
) -> Option<usize> {
    actual
        .zip(expected)
        .enumerate()
        .find_map(|(index, (actual, expected))| {
            (!close_at_f64_width(actual, expected, tolerance)).then_some(index)
        })
}

fn float_element_is_nan(value: ElementRef) -> bool {
    match value {
        ElementRef::F16(value) => value.is_nan(),
        ElementRef::Bf16(value) => value.is_nan(),
        ElementRef::F32(value) => value.is_nan(),
        ElementRef::F64(value) => value.is_nan(),
        ElementRef::I8(_)
        | ElementRef::I16(_)
        | ElementRef::I32(_)
        | ElementRef::I64(_)
        | ElementRef::Bool(_) => false,
    }
}

fn bind_checked_precision(
    name: &str,
    prim: Prim,
    bindings: &mut UnordMap<String, Prim>,
) -> Result<(), String> {
    match bindings.get(name) {
        Some(existing) if *existing != prim => Err(format!(
            "generic precision `{name}` actualized as both `{}` and `{}`",
            existing.name(),
            prim.name()
        )),
        Some(_) => Ok(()),
        None => {
            bindings.insert(name.to_string(), prim);
            Ok(())
        }
    }
}

fn checked_precision_leaf(actual: &Expr, caller_bindings: &UnordMap<String, Prim>) -> Option<Prim> {
    let (actual_tag, actual_children) = tagged_expr_children(actual)?;
    match actual_tag {
        DeepTag::TPrim | DeepTag::TVar => actual_children
            .first()
            .and_then(symbol_name)
            .and_then(|name| prim_from_name(name).or_else(|| caller_bindings.get(name).copied())),
        DeepTag::TRef => actual_children
            .first()
            .and_then(|inner| checked_precision_leaf(inner, caller_bindings)),
        _ => None,
    }
}

/// Match a declared callee type against the checker's concrete call-site type
/// and collect numeric precision actualizations. This consumes only static
/// type evidence: runtime values cannot recover an empty container's element
/// type and textual binder names are not identities across nested calls.
fn collect_checked_precision_bindings(
    declared: &Expr,
    actual: &Expr,
    caller_bindings: &UnordMap<String, Prim>,
    bindings: &mut UnordMap<String, Prim>,
) -> Result<(), String> {
    let Some((declared_tag, declared_children)) = tagged_expr_children(declared) else {
        return Ok(());
    };
    if declared_tag == DeepTag::TRef {
        let Some(declared_inner) = declared_children.first() else {
            return Ok(());
        };
        let actual_inner = tagged_expr_children(actual)
            .filter(|(tag, _)| *tag == DeepTag::TRef)
            .and_then(|(_, children)| children.first())
            .unwrap_or(actual);
        return collect_checked_precision_bindings(
            declared_inner,
            actual_inner,
            caller_bindings,
            bindings,
        );
    }
    if declared_tag == DeepTag::TVar {
        if let (Some(name), Some(prim)) = (
            declared_children.first().and_then(symbol_name),
            checked_precision_leaf(actual, caller_bindings),
        ) {
            return bind_checked_precision(name, prim, bindings);
        }
        return Ok(());
    }

    let Some((actual_tag, actual_children)) = tagged_expr_children(actual) else {
        return Ok(());
    };
    if declared_tag != actual_tag {
        return Ok(());
    }

    match declared_tag {
        DeepTag::TTensor => {
            let (Some(declared_precision), Some(actual_precision)) =
                (declared_children.last(), actual_children.last())
            else {
                return Ok(());
            };
            collect_checked_precision_bindings(
                declared_precision,
                actual_precision,
                caller_bindings,
                bindings,
            )
        }
        DeepTag::TAdt | DeepTag::TTuple | DeepTag::TFn => {
            for (declared_child, actual_child) in declared_children.iter().zip(actual_children) {
                collect_checked_precision_bindings(
                    declared_child,
                    actual_child,
                    caller_bindings,
                    bindings,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn checked_list_element(actual: &Expr) -> Option<&Expr> {
    let (tag, children) = tagged_expr_children(actual)?;
    match tag {
        DeepTag::TRef => children.first().and_then(checked_list_element),
        DeepTag::TAdt if children.first().and_then(symbol_name) == Some("List") => children.get(1),
        _ => None,
    }
}

/// The compiler and evaluator read one authored-alias normalization rule.
/// The session-less invariant predicate evaluator has no checked program or
/// declaration registry; its already-typed formals are used directly.
fn normalize_authored_entry_type(
    authored: &Expr,
    session: Option<&chelis_ir::host::HostLoweringSession<'_>>,
) -> Result<Expr, String> {
    match session {
        Some(session) => session.normalized_authored_entry_type(authored),
        None => Ok(authored.clone()),
    }
}

fn checked_function_children(actual: &Expr) -> Option<&[Expr]> {
    let (tag, children) = tagged_expr_children(actual)?;
    match tag {
        DeepTag::TRef => children.first().and_then(checked_function_children),
        DeepTag::TFn => Some(children),
        _ => None,
    }
}

fn check_signature_entry_plan(
    entry_plan: &chelis_ir::host::SignatureEntryPlan,
    entry_shapes: &[&[usize]],
) -> Result<(), String> {
    let read =
        |(node, axis): (NodeId, usize)| (entry_plan.label(node), axis, entry_shapes[node.0][axis]);
    for guard in entry_plan.guards() {
        use chelis_ir::axis_sources::EntryExtentGuard;
        let (left, right, context) = match guard {
            EntryExtentGuard::Named {
                claim,
                canonical,
                observed,
            } => {
                let (first, first_axis, left) = read(*canonical);
                let (later, later_axis, right) = read(*observed);
                (
                    left,
                    right,
                    format!(
                        "extent `{claim}`: {first} axis {first_axis} = {left}, \
                         {later} axis {later_axis} = {right}"
                    ),
                )
            }
            EntryExtentGuard::Literal { required, observed } => {
                let (parameter, axis, right) = read(*observed);
                (
                    *required,
                    right,
                    format!(
                        "extent `{required}`: claimed = {required}, \
                         {parameter} axis {axis} = {right}"
                    ),
                )
            }
        };
        if left != right {
            return Err(format!("{context}\nnumeric trap: domain in load at i64"));
        }
    }
    Ok(())
}

fn check_callable_invocation_contract(
    contract: &Expr,
    args: &[RuntimeValue],
    session: Option<&chelis_ir::host::HostLoweringSession<'_>>,
    present: Option<&[bool]>,
) -> Result<Vec<(String, TensorType, Vec<usize>)>, String> {
    let Some((_, params)) = checked_function_children(contract).and_then(<[Expr]>::split_last)
    else {
        return Ok(Vec::new());
    };
    let authored = params.iter().cloned().map(Some).collect::<Vec<_>>();
    let names = (0..args.len())
        .map(|index| format!("arg{index}"))
        .collect::<Vec<_>>();
    let actualized =
        actualize_tensor_entry_parameters(Some(params), &authored, args, &names, session, present)?;
    let mut entry_inputs = Vec::with_capacity(actualized.len());
    let mut entry_shapes: Vec<Vec<usize>> = Vec::with_capacity(actualized.len());
    for (parameter, ty, shape) in &actualized {
        entry_inputs.push(chelis_ir::host::HostTensorInput {
            name: parameter.clone(),
            ty: ty.clone(),
        });
        entry_shapes.push(shape.clone());
    }
    let entry_plan = chelis_ir::host::SignatureEntryPlan::new(entry_inputs);
    check_signature_entry_plan(
        &entry_plan,
        &entry_shapes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )?;
    Ok(actualized)
}

struct EntryActual {
    checked: Expr,
    authored: Expr,
    actual_type: TensorType,
    path: String,
    shape: Vec<usize>,
}

/// Rebuild tensor declarations, including List elements, at one invocation.
///
/// Runtime shapes determine rank-spread widths and dtypes, but their extents
/// are observations rather than declarations: explicit literals and shared
/// authored dimension identities remain the only entry obligations. This is
/// also why two actuals for the same rank spread may disagree in extent and
/// must reach [`check_signature_entry_plan`] instead of being rejected during
/// specialization.
fn actualize_tensor_entry_parameters(
    checked_params: Option<&[Expr]>,
    authored_params: &[Option<Expr>],
    args: &[RuntimeValue],
    names: &[String],
    session: Option<&chelis_ir::host::HostLoweringSession<'_>>,
    present: Option<&[bool]>,
) -> Result<Vec<(String, TensorType, Vec<usize>)>, String> {
    fn collect(
        pattern: &EntryPattern<Expr>,
        checked: Option<&Expr>,
        value: &RuntimeValue,
        path: String,
        out: &mut Vec<EntryActual>,
    ) -> Result<(), String> {
        match pattern {
            EntryPattern::Tensor(authored) => {
                let RuntimeValue::Tensor(tensor) = value else {
                    return Err(format!(
                        "input `{path}` does not match its declared tensor type"
                    ));
                };
                out.push(EntryActual {
                    checked: checked
                        .filter(|checked| tensor_type_dim_exprs(checked).is_some())
                        .unwrap_or(authored)
                        .clone(),
                    authored: authored.clone(),
                    actual_type: TensorType {
                        dims: tensor
                            .value
                            .shape
                            .iter()
                            .copied()
                            .map(DimInfo::Lit)
                            .collect(),
                        precision: tensor.precision,
                    },
                    path,
                    shape: tensor.value.shape.clone(),
                });
            }
            EntryPattern::List(inner) => {
                let RuntimeValue::List(items) = value else {
                    return Err(format!(
                        "input `{path}` does not match its declared tensor type"
                    ));
                };
                let checked_item = checked.and_then(checked_list_element);
                for (index, item) in items.iter().enumerate() {
                    collect(inner, checked_item, item, format!("{path}[{index}]"), out)?;
                }
            }
            EntryPattern::Other => {}
        }
        Ok(())
    }

    let normalized_authored = authored_params
        .iter()
        .map(|formal| {
            formal
                .as_ref()
                .map(|formal| normalize_authored_entry_type(formal, session))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let contract = chelis_ir::host::EntryContract::from_normalized_types(
        &normalized_authored,
        checked_params,
        names,
    )?
    .try_map_tensor(|index, _tensor| {
        fn tensor_leaf(expr: &Expr) -> Option<&Expr> {
            if tensor_type_dim_exprs(expr).is_some() {
                return Some(expr);
            }
            checked_list_element(expr).and_then(tensor_leaf)
        }
        normalized_authored[index]
            .as_ref()
            .and_then(tensor_leaf)
            .or_else(|| checked_params.and_then(|items| items.get(index)))
            .and_then(tensor_leaf)
            .cloned()
            .ok_or_else(|| "host runtime: could not retain entry tensor syntax".to_string())
    })?;
    let mut actuals = Vec::new();
    if args.len() != contract.formals().len() {
        return Err("host runtime: entry contract lost argument alignment".into());
    }
    if present.is_some_and(|present| present.len() != args.len()) {
        return Err("host runtime: selected-root entry lost argument presence".into());
    }
    for (index, (formal, arg)) in contract.formals().iter().zip(args).enumerate() {
        if present.is_some_and(|present| !present[index]) {
            continue;
        }
        collect(
            formal.pattern(),
            checked_params
                .and_then(|params| params.get(index))
                .or_else(|| normalized_authored.get(index).and_then(Option::as_ref)),
            arg,
            formal.name().to_string(),
            &mut actuals,
        )?;
    }
    let types = chelis_ir::lower::actualize_authored_tensor_parameters(
        &actuals
            .iter()
            .map(|actual| Some(actual.checked.clone()))
            .collect::<Vec<_>>(),
        &actuals
            .iter()
            .map(|actual| Some(actual.authored.clone()))
            .collect::<Vec<_>>(),
        &actuals
            .iter()
            .map(|actual| actual.actual_type.clone())
            .collect::<Vec<_>>(),
    )
    .map_err(|error| format!("host runtime: could not actualize entry contract: {error}"))?;
    let actualized = actuals
        .into_iter()
        .zip(types)
        .map(|(actual, ty)| (actual.path, ty, actual.shape))
        .collect::<Vec<_>>();
    for (path, ty, shape) in &actualized {
        if ty.dims.len() != shape.len() {
            return Err(format!(
                "input `{path}` expected rank {}, got {}",
                ty.dims.len(),
                shape.len()
            ));
        }
    }
    Ok(actualized)
}

fn render_shape(shape: &[usize]) -> String {
    let dimensions = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{dimensions}]")
}

fn decode_effect_kind(metadata: &Metadata) -> Result<EffectKind, EffectKindDecodeError<'_>> {
    metadata
        .effect()
        .map(|value| *value.value())
        .ok_or(EffectKindDecodeError::Missing)
}

/// Isolated source-call classification must retain the checked frame types
/// of bare arguments. Attach only missing type evidence: scalar expressions
/// remain source expressions, never their already-evaluated runtime values.
#[derive(Clone, Copy)]
struct EvalNode<'a> {
    expr: &'a Expr,
    tag: DeepTag,
    metadata: &'a Metadata,
    children: &'a [Expr],
}

impl<'a> EvalNode<'a> {
    fn new(expr: &'a Expr, tag: DeepTag, metadata: &'a Metadata, children: &'a [Expr]) -> Self {
        Self {
            expr,
            tag,
            metadata,
            children,
        }
    }
}

impl<'a> EvalContext<'a> {
    /// Resolve a builtin only when ordinary lexical lookup did not select a
    /// runtime binding of the same name (spec/04-type-system.md §8.6,
    /// chelis#1076). Every evaluator builtin fast path goes through this
    /// predicate so a new early dispatch cannot silently bypass scope.
    fn active_builtin_name<'expr>(&self, expr: &'expr Expr) -> Option<&'expr str> {
        let name = builtin_name(expr)?;
        self.active_builtin_symbol(name).then_some(name)
    }

    fn active_builtin_symbol(&self, name: &str) -> bool {
        !self.bindings.contains_key(name) && !self.tensor_bindings.contains_key(name)
    }

    pub(super) fn resolve_top_level(&mut self, name: &str) -> Result<RuntimeValue, String> {
        let Some((resolved_name, expr)) = self.lookup_top_level_def(name) else {
            return Err(format!("unknown runtime name `{name}`"));
        };
        if let Some(value) = self.declaration_values.get(&resolved_name) {
            return Ok(value.clone());
        }
        if self
            .resolving_top_levels
            .iter()
            .any(|existing| existing == &resolved_name)
        {
            return Err(format!("cyclic top-level runtime definition `{name}`"));
        }
        self.resolving_top_levels.push(resolved_name.clone());
        // Declaration scope is independent of its first caller. Nested
        // successful declarations remain cached even if this one fails.
        let saved = std::mem::take(&mut self.bindings);
        let saved_types = std::mem::take(&mut self.binding_types);
        let saved_precisions = std::mem::take(&mut self.precision_bindings);
        // A checked function alias carries the callable, including a nullary
        // one. Evaluating its bare var as a value thunk would replace that
        // callable with its result before the alias is ever invoked.
        let value = if self
            .program
            .type_env()
            .get(&resolved_name)
            .is_some_and(|ty| ty.tag() == Some(DeepTag::TFn))
            && let Some(alias) = var_name(&expr)
        {
            self.resolve_top_level(alias)
        } else {
            self.eval_expr(&expr)
        };
        self.bindings = saved;
        self.binding_types = saved_types;
        self.precision_bindings = saved_precisions;
        self.resolving_top_levels.pop();
        let value = stamp_def_closure(value?, &resolved_name, &expr);
        self.declaration_values.insert(resolved_name, value.clone());
        Ok(value)
    }

    /// chelis#1277 B2h: the kernel the C lane emits for def `name`, or `None`
    /// for the host lane. The decision is `chelis_ir::host::host_def_kernel`,
    /// the function `lower_host_function` itself uses, so the two lanes cannot
    /// disagree about which defs are kernels. A kernel's draws take their keys
    /// from the frame it is evaluated with, so one lowering serves every
    /// application. A kernel decision whose lowering fails is the evaluation's
    /// error, never a fall-through to the interpreter (the C lane's
    /// fall-through is chelis#1515 and is not inherited here).
    pub(super) fn def_kernel(&mut self, name: &str) -> Result<Option<Arc<HostDefKernel>>, String> {
        if let Some(cached) = self.def_kernels.get(name) {
            return Ok(cached.clone());
        }
        let Some(session) = self.session.as_ref() else {
            return Ok(None);
        };
        record_def_kernel_planning();
        // The decision reads `name`'s own body, whatever called it: a body a
        // kernel cannot carry is walked by the host interpreter (chelis#2405).
        let kernel = host_def_kernel(session, name)
            .map_err(|diagnostic| diagnostic.to_string())?
            .map(Arc::new);
        self.def_kernels.insert(name.to_string(), kernel.clone());
        Ok(kernel)
    }

    /// spec/03 §4.4: a binding's initializer is evaluated whether or not the
    /// binding is read. When the host hands a body to a tensor DAG (a kernel,
    /// a transform target, a routed named-axis call) the DAG demands only the
    /// values it reads, so the value declarations the body reaches
    /// ([`super::program_scope::ProgramScope::reached_by_applying`]) are
    /// initialized here, in the order it reaches them, where the caller's
    /// sequential order reaches the body and a live capture's initializer
    /// would run. A value the DAG already produced is not run again.
    pub(super) fn initialize_reached_values(&mut self, reached: &[String]) -> Result<(), String> {
        for key in reached {
            if !self.tensor_bindings.contains_key(key) {
                self.resolve_top_level(key)?;
            }
        }
        Ok(())
    }

    /// Apply def `name` through its kernel: the evaluated arguments become the
    /// kernel's `Load`s by declared parameter name (an unreferenced parameter
    /// is dropped, as the C wrapper drops it), a captured top-level tensor is
    /// served through `resolve_top_level`, the DAG evaluator runs with the
    /// Random stream threaded as `apply_transform` threads it, and the roots
    /// pack back into a runtime value. The evaluator's error text passes
    /// through unchanged: a [04-NUM-9] trap line takes no prefix.
    fn apply_def_kernel(
        &mut self,
        name: &str,
        kernel: &HostDefKernel,
        params: &[String],
        args: Vec<RuntimeValue>,
        inherited_claims: &[DeclaredResultClaim],
    ) -> Result<RuntimeValue, String> {
        self.initialize_reached_values(&self.program.reached_by_call(name))?;
        if let Some(plan) = &kernel.staged {
            return self.apply_staged_host_plan(name, plan, params, args);
        }
        if params.len() != args.len() {
            return Err(format!(
                "closure expected {} args, got {}",
                params.len(),
                args.len()
            ));
        }
        let mut staged: UnordMap<String, IrTensorValue> = UnordMap::new();
        for input in &kernel.inputs {
            let top_level_source =
                chelis_ir::LoadStoreName::top_level_source_for_label(&input.name)?;
            let param_index = top_level_source.as_ref().map_or_else(
                || {
                    kernel
                        .params
                        .iter()
                        .position(|param| param.name == input.name)
                },
                |_| None,
            );
            let value = match param_index {
                Some(index) => {
                    stage_kernel_argument(name, &input.name, &args[index], input.ty.precision)?
                }
                // A captured top-level binding: served through the same
                // resolution a plain reference takes (the transforms'
                // chelis#377 rule) and staged like a parameter, so a captured
                // scalar becomes the rank-0 input the C wrapper passes.
                None => {
                    let source_name = top_level_source.as_deref().unwrap_or(&input.name);
                    let captured = self.resolve_top_level(source_name)?;
                    let staged_value =
                        stage_kernel_argument(name, source_name, &captured, input.ty.precision)?;
                    // A captured kernel input keeps its authored rank. Any
                    // mapped batch lift is an explicit consumer inside the
                    // DAG, never a widened `Load` contract. Keep this guard as
                    // a defensive invariant check before evaluation.
                    if staged_value.shape.len() != input.ty.dims.len() {
                        return Err(format!(
                            "host runtime: kernel `{name}` capture rank invariant failed for \
                             top-level binding `{}`: the authored-rank input expects rank {} \
                             but the binding has rank {}.",
                            source_name,
                            input.ty.dims.len(),
                            staged_value.shape.len(),
                        ));
                    }
                    staged_value
                }
            };
            staged.insert(input.name.clone(), value);
        }
        let roots: Vec<NodeId> = kernel.dag.roots().to_vec();
        if roots.is_empty() {
            return Err(format!(
                "host runtime: kernel `{name}` lowering produced no roots"
            ));
        }
        let result_producer = (roots.len() == 1)
            .then(|| tensor_result_producer(&kernel.dag, roots[0]))
            .flatten();
        let result_claims = inherited_claims
            .iter()
            .map(|claim| {
                let axes = claim
                    .axes
                    .iter()
                    .map(|axis| {
                        Ok(chelis_ir::eval::InheritedResultAxis {
                            axis: axis.axis,
                            required: usize::try_from(axis.required).map_err(|_| {
                                "declared result extent is outside the admitted range".to_string()
                            })?,
                            source: axis.source.as_ref().map(|source| {
                                chelis_ir::axis_sources::ClaimSource {
                                    claim: source.claim.clone(),
                                    parameter: source.parameter.clone(),
                                    axis: source.axis,
                                }
                            }),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(chelis_ir::eval::InheritedResultClaim {
                    rank: claim.rank,
                    axes,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let tensor_bindings = self.tensor_bindings;
        let host_bindings = &self.declaration_values;
        let result = chelis_ir::eval::eval_tensor_roots_exact_with_result_claims(
            &kernel.dag,
            &roots,
            &result_claims,
            |load| {
                staged
                    .get(load)
                    .cloned()
                    .or_else(|| tensor_bindings.get(load).map(|t| t.value.clone()))
                    .or_else(|| match host_bindings.get(load) {
                        Some(RuntimeValue::Tensor(t)) => Some(t.value.clone()),
                        _ => None,
                    })
            },
        );
        let values = self.mark_numeric_trap_from_trusted_result(result)?;
        let value = pack_dag_roots(&kernel.dag, &roots, &values, name)?;
        self.result_producer = result_producer;
        Ok(value)
    }

    fn apply_staged_host_plan(
        &mut self,
        name: &str,
        plan: &chelis_ir::host::staged::HostStagedPlan,
        params: &[String],
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, String> {
        use chelis_ir::host::staged::HostStage;
        use chelis_ir::host_type_state::HostTypeTerm;
        if params.len() != args.len() {
            return Err(format!("staged kernel `{name}` argument arity mismatch"));
        }
        let mut values: UnordMap<String, RuntimeValue> = params.iter().cloned().zip(args).collect();
        let mut result_producers: UnordMap<String, ResultProducer> = UnordMap::new();
        for stage in plan.stages() {
            match stage {
                HostStage::Source {
                    expression,
                    captures,
                    output,
                    ty,
                } => {
                    let saved = std::mem::take(&mut self.bindings);
                    let saved_types = std::mem::take(&mut self.binding_types);
                    for capture in captures {
                        let value = values
                            .get(&capture.value)
                            .expect("validated staged capture")
                            .clone();
                        let value = match (&capture.ty, value) {
                            (HostTypeTerm::Scalar(_), RuntimeValue::Tensor(tensor)) => {
                                RuntimeValue::from_scalar_value(tensor.value.storage().scalar_at(0))
                            }
                            (_, value) => value,
                        };
                        // A staged tensor capture retains the checked type
                        // needed by host primitive/transform routing. An
                        // untyped capture masks any same-named outer type.
                        let declared = match &capture.ty {
                            HostTypeTerm::Tensor(ty) => self.static_type_expr_of(
                                &make_var_with_type(&capture.binding, ty, expression.span()),
                            ),
                            _ => None,
                        };
                        self.binding_types.insert(capture.binding.clone(), declared);
                        let producer = ResultProducer::interface_load(&value);
                        self.bindings.insert_with_result_producer(
                            capture.binding.clone(),
                            value,
                            producer,
                        );
                    }
                    let result = self.eval_expr(expression);
                    self.bindings = saved;
                    self.binding_types = saved_types;
                    let value = result?;
                    let producer = self.result_producer.take();
                    if matches!(
                        ty,
                        HostTypeTerm::Scalar(
                            chelis_ir::host_type_state::HostPrecisionTerm::Concrete(Prim::Int64)
                        )
                    ) && !matches!(&value, RuntimeValue::Scalar(payload) if payload.dtype() == Prim::Int64)
                    {
                        return Err("staged reshape target did not produce exactly i64".into());
                    }
                    values.insert(output.clone(), value);
                    if let Some(producer) = producer {
                        result_producers.insert(output.clone(), producer);
                    }
                }
                HostStage::Kernel { dag, outputs } => {
                    let mut inputs = UnordMap::new();
                    for node in dag.nodes() {
                        if let RiscOp::Load { name: input } = &node.op {
                            let value = match values.get(input.as_str()) {
                                Some(value) => value.clone(),
                                None => self.resolve_top_level(input.as_str())?,
                            };
                            inputs.insert(
                                input.as_str().to_owned(),
                                stage_kernel_argument(
                                    name,
                                    input.as_str(),
                                    &value,
                                    node.output_type.precision,
                                )?,
                            );
                        }
                    }
                    let computed =
                        chelis_ir::eval::eval_tensor_roots_exact(dag, dag.roots(), |input| {
                            inputs.get(input).cloned()
                        });
                    let computed = self.mark_numeric_trap_from_trusted_result(computed)?;
                    for (output, root) in outputs.iter().zip(dag.roots()) {
                        let value = computed
                            .get(root)
                            .ok_or("staged kernel omitted an output")?
                            .clone();
                        values.insert(
                            output.clone(),
                            RuntimeValue::Tensor(RuntimeTensorValue::new(value)),
                        );
                        if let Some(producer) = tensor_result_producer(dag, *root) {
                            result_producers.insert(output.clone(), producer);
                        }
                    }
                }
            }
        }
        let value = values
            .remove(plan.output())
            .ok_or_else(|| "staged kernel omitted its final result".to_string())?;
        self.result_producer = result_producers.remove(plan.output());
        Ok(value)
    }

    pub(super) fn lookup_top_level_def(&self, name: &str) -> Option<(String, Expr)> {
        let key = self.program.resolve_def_key(name)?;
        let body = self.program.defs().get(key)?;
        Some((key.to_owned(), body.clone()))
    }

    fn lookup_declared_signature(&self, name: &str) -> Option<&Expr> {
        self.declared_signatures.get(name).or_else(|| {
            let mut matches =
                self.declared_signatures
                    .to_sorted()
                    .into_iter()
                    .filter_map(|(key, signature)| {
                        terminal_name_matches(key, name).then_some(signature)
                    });
            let signature = matches.next()?;
            matches.next().is_none().then_some(signature)
        })
    }

    pub(super) fn eval_expr(&mut self, expr: &Expr) -> Result<RuntimeValue, String> {
        // Provenance describes this expression's result, never whichever
        // tensor an argument, selector, or preceding effect happened to
        // evaluate last. Recursive calls establish their own result and the
        // outer form either replaces it or transparently returns it.
        self.result_producer = None;
        let value = self.eval_expr_inner(expr)?;
        if !self
            .result_producer
            .as_ref()
            .is_some_and(|producer| producer.matches_value(&value))
        {
            self.result_producer = None;
        }
        Ok(value)
    }

    fn eval_expr_inner(&mut self, expr: &Expr) -> Result<RuntimeValue, String> {
        // chelis#914: cooperative cancellation. Every node visit passes
        // through here — including each element of a fold/map, which reach
        // `eval_expr` via `apply_resolved_callable_with_arg_types` — so this
        // is the single point that bounds how long a cancelled evaluation
        // keeps running. `self.cancel` was captured once at context
        // construction, so the common (no token) case is an `Option`
        // discriminant test and the cancellable case adds one relaxed load.
        if let Some(cancel) = &self.cancel
            && cancel.is_cancelled()
        {
            return Err(chelis_types::EVAL_CANCELLED_MSG.to_string());
        }
        match expr.carrier() {
            ExprCarrier::Atom(_) => Err("bare atom is not a runtime expression".to_string()),
            ExprCarrier::MetadataMap(_) => Ok(RuntimeValue::Unit),
            ExprCarrier::MetadataExpression(meta) => self.eval_expr(&meta.expr),
            ExprCarrier::DecodedNode(tag, metadata, children) => {
                self.eval_decoded(EvalNode::new(expr, tag, metadata, children))
            }
            // chelis#1087: loud rejection, identifying the form the way the
            // resugar boundary describes it rather than by an internal
            // variant name.
            ExprCarrier::StructuralList(_) => {
                Err("a structural bare list is not a runtime expression".to_string())
            }
            ExprCarrier::UndecodableHead(head, _, _) => {
                Err(format!("unknown form `{head}` is not a runtime expression"))
            }
        }
    }

    fn eval_decoded(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        match node.tag {
            DeepTag::Lit => self.eval_lit(node),
            DeepTag::Var => self.eval_var(node),
            DeepTag::App => self.eval_app(node),
            DeepTag::If => self.eval_if(node),
            DeepTag::Let => self.eval_let(node),
            DeepTag::Tuple => {
                let mut values = Vec::with_capacity(node.children.len());
                let mut producers = Vec::with_capacity(node.children.len());
                for child in node.children {
                    values.push(self.eval_expr(child)?);
                    producers.push(self.result_producer.take());
                }
                self.result_producer = ResultProducer::aggregate(producers);
                Ok(RuntimeValue::Tuple(values.into()))
            }
            DeepTag::Copy => {
                let value = self.eval_expr(
                    node.children
                        .first()
                        .ok_or_else(|| "copy missing value".to_string())?,
                )?;
                match value {
                    RuntimeValue::Tensor(tensor) => Ok(RuntimeValue::Tensor(tensor)),
                    other => Err(format!("copy expects tensor input, got {other:?}")),
                }
            }
            DeepTag::Borrow => {
                // The IR lower path treats `borrow` as identity
                // (chelis-ir/src/lower.rs::lower_identity); mirror that
                // here so `&t` syntax type-checks AND evaluates.
                self.eval_expr(
                    node.children
                        .first()
                        .ok_or_else(|| "borrow missing value".to_string())?,
                )
            }
            DeepTag::Block => {
                // chelis#859: sequenced expressions, value is the last
                // child's (spec/03 §2.3). Non-last children evaluate for
                // their effects (e.g. `print` transcript lines).
                let kids = node.children;
                let Some((last, init)) = kids.split_last() else {
                    return Err("a `block` node has no children".to_string());
                };
                for child in init {
                    let _ = self.eval_expr(child)?;
                }
                self.eval_expr(last)
            }
            DeepTag::Record => self.eval_record(node),
            DeepTag::Access => self.eval_access(node),
            DeepTag::TupleGet => self.eval_tuple_get(node),
            DeepTag::Match => self.eval_match(node),
            DeepTag::Fn => self.eval_fn(node),
            // chelis#1923: a pipe cannot reach the evaluator. Every checker
            // entry folds it into the application it denotes
            // (`chelis_deep::pipe::fold_pipe`), and this evaluator reads the
            // checked program. The arm that used to sit here re-derived a
            // static type per stage to recover named axes the stage lambda's
            // unresolved parameter had lost; the folded application carries
            // the operand's own annotation, so there is nothing left to
            // re-derive. Fail closed so a new unfolded ingress is loud.
            DeepTag::Pipe => Err("a pipe reached evaluation unfolded: every checker \
                 entry folds a pipe into the application it denotes \
                 (spec/02-surf-syntax.md section 0.1; chelis#1923)"
                .to_string()),
            DeepTag::Cast => self.eval_cast(node),
            DeepTag::Realize => {
                // Bucket 1: `realize` is identity in the host runtime,
                // matching the C-backend `lower_realize` pass-through
                // (`crates/chelis-ir/src/host.rs::lower_host_expr`).
                self.eval_expr(
                    node.children
                        .first()
                        .ok_or_else(|| "realize missing value".to_string())?,
                )
            }
            DeepTag::Grad => {
                // Bucket 1: capture the `(grad ...)` form so it can be
                // applied later. The application path
                // (`apply_resolved_callable` for a `Transform`) routes
                // through `lower_subexpr_program` + the forward DAG
                // evaluator — the same machinery that `chelis build
                // --target c` uses.
                Ok(RuntimeValue::Transform {
                    kind: TransformKind::Grad,
                    transform_expr: node.expr.clone(),
                    captured_env: self.bindings.capture(),
                    invocation_contracts: Box::default(),
                })
            }
            DeepTag::Vmap => {
                // Bucket 1: same pattern as `grad` above, capture-and-apply.
                Ok(RuntimeValue::Transform {
                    kind: TransformKind::Vmap,
                    transform_expr: node.expr.clone(),
                    captured_env: self.bindings.capture(),
                    invocation_contracts: Box::default(),
                })
            }
            DeepTag::Jit => {
                // `spec/03-deep-syntax.md` §2.7: `jit` is a compilation
                // trigger and a semantic no-op at evaluation. The host
                // runtime evaluates the inner expression and returns its
                // value, mirroring `lower_jit` in
                // `crates/chelis-ir/src/lower.rs` and the IR DAG behavior.
                self.eval_expr(
                    node.children
                        .first()
                        .ok_or_else(|| "jit missing value".to_string())?,
                )
            }
            DeepTag::Par => {
                // Legacy sequential placeholder retained behind the
                // chelis#2503 checker fence. No checked source program
                // reaches this path until scheduler-independent cross-lane
                // effects are complete. Keep the arm fail-closed for a
                // malformed empty node used by internal probes.
                let mut last: Option<RuntimeValue> = None;
                for child in node.children {
                    last = Some(self.eval_expr(child)?);
                }
                last.ok_or_else(|| "par has no children to evaluate".to_string())
            }
            DeepTag::HandleEffect => {
                let kids = node.children;
                match decode_effect_kind(node.metadata)
                    .map_err(|error| format!("{error} in `handle-effect` evaluation"))?
                {
                    EffectKind::Resource => self.eval_expr(
                        kids.get(1)
                            .ok_or_else(|| "handle-effect missing body".to_string())?,
                    ),
                }
            }
            other => Err(format!(
                "host runtime does not support `{}`",
                other.as_str()
            )),
        }
    }

    fn eval_record(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        let kids = node.children;
        let ctor = kids
            .first()
            .and_then(symbol_name)
            .ok_or_else(|| "record missing constructor name".to_string())?;
        let mut fields_by_name = UnordMap::new();
        let mut producers_by_name = UnordMap::new();
        let mut source_order = Vec::new();
        for field in kids.iter().skip(1) {
            let field_kids = match field.carrier() {
                ExprCarrier::DecodedNode(DeepTag::Kv, _, children) => children,
                ExprCarrier::DecodedNode(_, _, _)
                | ExprCarrier::StructuralList(_)
                | ExprCarrier::UndecodableHead(_, _, _)
                | ExprCarrier::Atom(_)
                | ExprCarrier::MetadataMap(_)
                | ExprCarrier::MetadataExpression(_) => {
                    return Err("record field must be a decoded `kv` node".to_string());
                }
            };
            if field_kids.len() != 2 {
                return Err("record `kv` field must contain a name and value".to_string());
            }
            let name = field_kids
                .first()
                .and_then(symbol_name)
                .ok_or_else(|| "record field name must be a symbol".to_string())?;
            let value = self.eval_expr(
                field_kids
                    .get(1)
                    .ok_or_else(|| "record field missing value".to_string())?,
            )?;
            let producer = self.result_producer.take();
            source_order.push(name.to_string());
            fields_by_name.insert(name.to_string(), value);
            if let Some(producer) = producer {
                producers_by_name.insert(name.to_string(), producer);
            }
        }
        let declared = self
            .adt_fields
            .get(ctor)
            .cloned()
            .unwrap_or_else(|| source_order.clone());
        let mut ordered = Vec::with_capacity(declared.len());
        let mut ordered_producers = Vec::with_capacity(declared.len());
        for field_name in &declared {
            let value = fields_by_name.remove(field_name).ok_or_else(|| {
                format!("record `{ctor}` missing field `{field_name}` at runtime")
            })?;
            ordered.push(value);
            ordered_producers.push(producers_by_name.remove(field_name));
        }
        if let Some((extra, _)) = fields_by_name.to_sorted().into_iter().next() {
            return Err(format!(
                "record `{ctor}` has unknown field `{extra}` at runtime"
            ));
        }
        // `field_names` must stay aligned with `fields`: both follow the
        // DECLARED field order used for the reordering above. The pre-#520
        // code stored the kv SOURCE order here (the desugarer sorts record
        // kvs alphabetically), so any record whose alphabetical order
        // differs from its declared order carried misaligned
        // `field_names[i]` metadata against `fields[i]`.
        self.result_producer = ResultProducer::aggregate(ordered_producers);
        Ok(self.adt_value(ctor, ordered.into(), Some(declared)))
    }

    /// An ADT value of the constructor `ctor`. It stores the constructor's
    /// declared source spelling as the name it renders, by the rule the
    /// compiled lane stores (chelis#2889); a constructor with no linker name
    /// keeps the name it has.
    fn adt_value(
        &self,
        ctor: &str,
        fields: Values,
        field_names: Option<Vec<String>>,
    ) -> RuntimeValue {
        RuntimeValue::Adt {
            ctor: ctor.to_string(),
            source_name: self
                .constructor_names
                .get(ctor)
                .cloned()
                .unwrap_or_else(|| ctor.to_string()),
            fields,
            field_names,
        }
    }

    fn eval_access(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        let kids = node.children;
        let target = self.eval_expr(
            kids.first()
                .ok_or_else(|| "access missing target".to_string())?,
        )?;
        let target_producer = self.result_producer.take();
        let field = kids
            .get(1)
            .and_then(symbol_name)
            .ok_or_else(|| "access missing field".to_string())?;
        match target {
            RuntimeValue::Adt {
                ctor,
                fields,
                field_names,
                ..
            } => {
                let declared = self
                    .adt_fields
                    .get(&ctor)
                    .cloned()
                    .or(field_names)
                    .ok_or_else(|| format!("unknown record constructor `{ctor}`"))?;
                let Some(index) = declared.iter().position(|name| name == field) else {
                    return Err(format!("record `{ctor}` has no field `{field}`"));
                };
                let value = fields
                    .get(index)
                    .cloned()
                    .ok_or_else(|| format!("record `{ctor}` missing field `{field}`"))?;
                self.result_producer = target_producer
                    .as_ref()
                    .and_then(|producer| producer.child(index))
                    .filter(|producer| producer.matches_value(&value));
                Ok(value)
            }
            other => Err(format!("field access expects record value, got {other:?}")),
        }
    }

    fn eval_lit(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        let value = node
            .children
            .first()
            .ok_or_else(|| "lit missing value".to_string())?;
        // Per spec/04-type-system.md §5.3, the desugarer narrows
        // unsuffixed integer literals to i32 and unsuffixed float
        // literals to f32. The type checker writes the resolved
        // primitive into the lit's `type` meta as `(t-prim {} <name>)`.
        // Honor that meta where present so a context-typed literal
        // (e.g. `(lit {type: (t-prim {} i64)} 42)`) carries the
        // surrounding-position dtype, not just the bare default.
        // [02-SURF-P10b]: a literal in `cast(<literal>, p)` binds at `p`.
        // Resolve that stamp through the call frame's precision bindings.
        let meta_dtype = match lit_meta_prim(node.metadata) {
            Some(prim) => Some(prim),
            None => match lit_meta_type_var_name(node.metadata) {
                // Internal callee binders may remain unbound until inlining;
                // preserve the default when no call-site binding exists.
                Some(binder) => self.precision_bindings.get(binder).copied(),
                None => None,
            },
        };
        match value {
            Expr::Atom(Atom::Int(value), _) => match meta_dtype {
                Some(dtype) if dtype.is_integer() || dtype.is_float() => {
                    // Keep the exact i64 payload until the sealed dtype
                    // constructor finalizes it at the declared width. An
                    // `as f64` step here double-rounds integer-spelled f32,
                    // f16, and bf16 literals above 2^53 and disagrees with
                    // both IR lowering and generated C ([04-LIT-1]).
                    RuntimeValue::scalar_like_int(dtype, *value)
                }
                _ => Ok(RuntimeValue::int_lit(*value)),
            },
            Expr::Atom(Atom::Float(value), _) => match meta_dtype {
                Some(dtype) if dtype.is_float() => RuntimeValue::scalar_like_float(dtype, *value),
                _ => Ok(RuntimeValue::float_lit(*value)),
            },
            Expr::Atom(Atom::Bool(value), _) => Ok(RuntimeValue::Bool(*value)),
            Expr::Atom(Atom::Str(value), _) => Ok(RuntimeValue::String(value.clone())),
            // Unit literal `()` desugars to `(lit {type: (t-unit {})} ())` where the
            // inner `()` is an empty bare list. Treat that as RuntimeValue::Unit so
            // `def test_noop() -> unit = ()` runs cleanly instead of dying with
            // "unsupported literal form".
            value
                if match value.carrier() {
                    ExprCarrier::StructuralList(elements) => elements.is_empty(),
                    ExprCarrier::DecodedNode(_, _, _)
                    | ExprCarrier::UndecodableHead(_, _, _)
                    | ExprCarrier::Atom(_)
                    | ExprCarrier::MetadataMap(_)
                    | ExprCarrier::MetadataExpression(_) => false,
                } =>
            {
                Ok(RuntimeValue::Unit)
            }
            _ => Err("unsupported literal form".to_string()),
        }
    }

    fn eval_var(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        let name = node
            .children
            .first()
            .and_then(symbol_name)
            .ok_or_else(|| "var missing name".to_string())?;
        if let Some(value) = self.bindings.get(name).cloned() {
            self.result_producer = self.bindings.result_producer(name).cloned();
            return Ok(value);
        }
        // A dimension binder a site of this activation bound, which outlives
        // the block the site sits in (spec/04-type-system.md section 4.4.1).
        if let Some((_, extent)) = self.activation_extents.current(name) {
            self.result_producer = None;
            return Ok(RuntimeValue::int64(*extent));
        }
        if let Some(value) = self.tensor_bindings.get(name) {
            // External tensor bindings enter the host evaluator through the
            // same semantic ingress as a DAG Load.
            self.result_producer = Some(ResultProducer::tensor("load"));
            return Ok(RuntimeValue::Tensor(value.clone()));
        }
        if let Some((_, definition)) = self.lookup_top_level_def(name) {
            let value = self.resolve_top_level(name)?;
            // A zero-parameter top-level declaration is a value thunk when
            // referenced in expression position. Calls still resolve their
            // callee directly in `eval_app`, so `name()` receives the closure
            // and applies it exactly once; a bare `name` consumes its value.
            // This mirrors the checker/lowerer's nullary-def treatment and is
            // required when manifest routing selects the host evaluator.
            // Function-valued aliases carry the callable through argument
            // and local-binding positions; only a declaration is a thunk.
            if definition.tag() == Some(DeepTag::Fn)
                && matches!(&value, RuntimeValue::Closure { params, .. } if params.is_empty())
            {
                return self.apply_resolved_callable(value, Vec::new());
            }
            // A declaration value is read through the top-level binding
            // interface. Its initializer has already run (and may be cached),
            // so a returned tensor has `load` provenance rather than that of
            // whichever operation happened to initialize it.
            self.result_producer = ResultProducer::interface_load(&value);
            return Ok(value);
        }
        // The checked key operations are values as well as direct call
        // heads. Capture the resolved operation now: a later lexical alias
        // or shadow must not change which operation the callable invokes.
        // The wrapper is compiler-owned and has no source-level body or
        // captures. Its parameters and result come from the checked value
        // type, never from a guessed scalar default.
        if matches!(
            name,
            "key_from_seed" | "split_key" | "split_keys" | "fold_in"
        ) && self.active_builtin_symbol(name)
        {
            // An unannotated checked alias can retain the closed operation
            // relation without one selected `TFn` on this value node. Its
            // identity and arity are fixed here; the checked application
            // decides the scalar/tensor alternative from its arguments.
            let signature = node.metadata.ty().map(|ty| ty.expression());
            let types = signature.and_then(checked_function_children);
            let (return_type, param_types) =
                types
                    .and_then(<[Expr]>::split_last)
                    .map_or((None, Vec::new()), |(ret, params)| {
                        (
                            Some(ret.clone()),
                            params.iter().cloned().map(Some).collect(),
                        )
                    });
            let arity = if matches!(name, "key_from_seed" | "split_key") {
                1
            } else {
                2
            };
            let params = (0..arity)
                .map(|index| format!("__chelis_key_arg_{index}"))
                .collect::<Vec<_>>();
            let span = node.expr.span();
            let mut call = Vec::with_capacity(params.len() + 1);
            call.push(Expr::node(
                DeepTag::Var,
                chelis_deep::Metadata::default(),
                vec![Expr::Atom(Atom::Name(name.to_string()), span)],
                span,
            ));
            call.extend(params.iter().map(|param| {
                Expr::node(
                    DeepTag::Var,
                    chelis_deep::Metadata::default(),
                    vec![Expr::Atom(Atom::Name(param.clone()), span)],
                    span,
                )
            }));
            let body = Expr::node(DeepTag::App, chelis_deep::Metadata::default(), call, span);
            return Ok(RuntimeValue::Closure {
                checked_function: Box::new(node.expr.clone()),
                params,
                param_types: if param_types.is_empty() {
                    vec![None; arity]
                } else {
                    param_types
                },
                return_type,
                checked_signature: types.and_then(|_| signature.cloned()),
                invocation_contracts: Box::default(),
                body,
                env: self.bindings.capture(),
                precision_env: self.precision_bindings.clone(),
                def_name: None,
            });
        }
        if name == "Nil" {
            return Ok(RuntimeValue::List(Vec::new().into()));
        }
        if name.chars().next().is_some_and(|ch| ch.is_uppercase()) {
            return Ok(self.adt_value(name, Vec::new().into(), None));
        }
        Err(format!("unknown runtime name `{name}`"))
    }

    fn eval_fn(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        let kids = node.children;
        let params_expr = kids
            .first()
            .ok_or_else(|| "fn missing params".to_string())?;
        let params_children = match params_expr.carrier() {
            ExprCarrier::DecodedNode(DeepTag::Params, _, children) => children,
            ExprCarrier::DecodedNode(_, _, _)
            | ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => {
                return Err("fn params malformed".to_string());
            }
        };
        let decoded_params = params_children
            .iter()
            .map(|param| {
                runtime_param_parts(param).ok_or_else(|| "fn params malformed".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let params = decoded_params
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect::<Vec<_>>();
        let param_types = decoded_params
            .iter()
            .map(|(_, metadata)| {
                metadata.and_then(|metadata| metadata.ty().map(|ty| ty.expression().clone()))
            })
            .collect::<Vec<_>>();
        let body = kids
            .get(1)
            .ok_or_else(|| "fn missing body".to_string())?
            .clone();
        let return_type = self
            .resolving_top_levels
            .last()
            .and_then(|name| self.lookup_declared_signature(name))
            .and_then(|signature| {
                tagged_expr_children(signature)
                    .filter(|(tag, _)| *tag == DeepTag::TFn)
                    .and_then(|(_, children)| children.last())
            })
            .cloned();
        Ok(RuntimeValue::Closure {
            checked_function: Box::new(node.expr.clone()),
            params,
            param_types,
            return_type,
            checked_signature: node.metadata.ty().map(|ty| ty.expression().clone()),
            invocation_contracts: Box::default(),
            body,
            // Named declarations are initialized in an empty lexical frame;
            // an anonymous fn must retain every actual local shadow.
            env: self.bindings.capture(),
            precision_env: self.precision_bindings.clone(),
            def_name: None,
        })
    }

    fn eval_app(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        self.eval_app_under_result_claim(node, &[])
    }

    fn eval_app_under_result_claim(
        &mut self,
        node: EvalNode<'_>,
        claims: &[DeclaredResultClaim],
    ) -> Result<RuntimeValue, String> {
        let kids = node.children;
        let func = kids
            .first()
            .ok_or_else(|| "app missing function".to_string())?;

        // `dropout` is not in the host builtin table: every dropout the host
        // walk reaches draws here from the interpreter's handler.
        if var_name(func) == Some("dropout") && self.active_builtin_symbol("dropout") {
            let mut args = Vec::with_capacity(kids.len().saturating_sub(1));
            for arg in &kids[1..] {
                args.push(self.eval_expr(arg)?);
            }
            let value = self.eval_dropout_builtin(&args)?;
            // Its result is a tensor it produced, as every tensor builtin's is.
            self.result_producer = Some(ResultProducer::tensor("dropout"));
            for claim in claims {
                self.check_declared_result_claim(claim, &value, "dropout")?;
            }
            return Ok(value);
        }

        // chelis#338 site A: a reduction whose axis argument is a bare
        // `(var name)` names a *dimension* of the operand, not a runtime
        // value (spec/04-type-system.md SS4.5.3); the checker admits only
        // int-literal or named axes, so a bare var here is always a named
        // axis. It must be intercepted BEFORE generic argument evaluation
        // (which would fail with `unknown runtime name`) and routed
        // through IR lowering, where name -> index resolution lives.
        if let Some(reduce_name) = self.active_builtin_name(func)
            && REDUCTION_BUILTIN_NAMES.contains(&reduce_name)
            && kids.len() >= 3
            && kids[2..].iter().any(|axis| var_name(axis).is_some())
        {
            return self.eval_named_axis_reduction_app(reduce_name, kids);
        }

        // chelis#339 site A twin: a named-axis EXPAND app — the axis slot
        // is a bare `(var name)` naming the *inserted* axis (and the
        // optional fourth arg names the anchor). Same interception, same
        // routing lane: IR lowering resolves the insertion point against
        // the operand's named dims. The positional form (integer axis,
        // possibly with a symbolic size) keeps the host path.
        if let Some(expand_name) = self.active_builtin_name(func)
            && (expand_name == "expand" || expand_name == "insert")
            && kids.len() >= 4
            && var_name(&kids[2]).is_some()
        {
            return self.eval_named_axis_reduction_app(expand_name, kids);
        }

        let arg_type_exprs = kids[1..]
            .iter()
            .map(|arg| self.static_type_expr_of(arg))
            .collect::<Vec<_>>();
        let result_type_expr = node.metadata.ty().map(|ty| ty.expression().clone());
        let mut args = Vec::with_capacity(kids.len().saturating_sub(1));
        let mut arg_producers = Vec::with_capacity(kids.len().saturating_sub(1));
        for arg in &kids[1..] {
            args.push(self.eval_expr(arg)?);
            arg_producers.push(self.result_producer.take());
        }

        // Std.Io.Json's private serializer intrinsic. Keep generic
        // `dict_entries` insertion-ordered for CSV/tokenizer callers; only
        // the owning JSON boundary canonicalizes string keys. Rust `str` Ord
        // is UTF-8 lexicographic, which preserves Unicode scalar-value order
        // for valid Rust strings.
        if let Some(name) = var_name(func)
            && self
                .lookup_top_level_def(name)
                .is_some_and(|(resolved, _)| {
                    matches!(
                        resolved.as_str(),
                        "Pkg__chelis__std__Std__Io__Json__canonical_object_entries"
                            | "pkg__chelis__std__Std__Io__Json__canonical_object_entries"
                    )
                })
        {
            let mut entries = expect_dict_arg(&args, 0)?;
            entries.sort_by(|(lhs, _), (rhs, _)| match (lhs, rhs) {
                (RuntimeValue::String(lhs), RuntimeValue::String(rhs)) => lhs.cmp(rhs),
                _ => std::cmp::Ordering::Equal,
            });
            return Ok(RuntimeValue::List(
                entries
                    .into_iter()
                    .map(|(key, value)| RuntimeValue::Tuple(vec![key, value].into()))
                    .collect(),
            ));
        }

        if let Some(name) = var_name(func)
            && name.chars().next().is_some_and(|ch| ch.is_uppercase())
        {
            if name == "Cons" {
                if args.len() != 2 {
                    return Err(format!("Cons expects 2 arguments, got {}", args.len()));
                }
                let mut items = match &args[1] {
                    RuntimeValue::List(items) => items.clone(),
                    other => {
                        return Err(format!("Cons tail must be a List, got {other:?}"));
                    }
                };
                items.insert(0, args[0].clone());
                let mut producers = match arg_producers.get(1).and_then(Option::as_ref) {
                    Some(ResultProducer::Aggregate(children)) => children.clone(),
                    Some(uniform @ ResultProducer::Uniform(_)) => {
                        vec![Some(uniform.clone()); items.len().saturating_sub(1)]
                    }
                    _ => vec![None; items.len().saturating_sub(1)],
                };
                producers.insert(0, arg_producers.first().cloned().flatten());
                self.result_producer = ResultProducer::aggregate(producers);
                return Ok(RuntimeValue::List(items));
            }
            self.result_producer = ResultProducer::aggregate(arg_producers);
            return Ok(self.adt_value(name, args.into(), None));
        }

        if let Some(name) = self.active_builtin_name(func) {
            if name == "pad_sequences_to" && !claims.is_empty() {
                let sequences = expect_list_arg(&args, 0)?;
                let width = expect_int_arg(&args, 1)?;
                // Preserve the operation precondition before the local claim.
                if width < 0 {
                    return Err(format!(
                        "pad_sequences_to requires non-negative width, got {width}"
                    ));
                }
                for claim in claims {
                    self.check_declared_shape_claim(
                        claim,
                        &[sequences.len(), width as usize],
                        name,
                    )?;
                }
            }
            let builtin_result =
                self.eval_builtin(name, &args, &arg_type_exprs, result_type_expr.as_ref());
            // The checked builtin catalog owns the numeric operation set.
            // Mixed-domain equality is trusted only for numeric arguments;
            // recursive container comparison can include authored strings.
            let trusted_numeric_source = chelis_types::builtin_decl(name).is_some_and(|decl| {
                decl.capability.domains == [chelis_types::BuiltinSemanticDomain::Numeric]
                    || (decl
                        .capability
                        .domains
                        .contains(&chelis_types::BuiltinSemanticDomain::Numeric)
                        && args.iter().all(|arg| {
                            matches!(
                                arg,
                                RuntimeValue::Scalar(_)
                                    | RuntimeValue::Tensor(_)
                                    | RuntimeValue::Bool(_)
                            )
                        }))
            }) || (name == "concat"
                && matches!(args.first(), Some(RuntimeValue::List(parts)) if parts.iter().all(|part| matches!(part, RuntimeValue::Tensor(_)))));
            let value = if trusted_numeric_source {
                self.mark_numeric_trap_from_trusted_result(builtin_result)?
            } else {
                builtin_result?
            };
            self.result_producer = match name {
                "index" => args
                    .get(1)
                    .and_then(RuntimeValue::as_i64)
                    .and_then(|index| usize::try_from(index).ok())
                    .and_then(|index| {
                        arg_producers
                            .first()
                            .and_then(Option::as_ref)
                            .and_then(|producer| producer.child(index))
                    })
                    .filter(|producer| producer.matches_value(&value)),
                "skip" => {
                    let count = args.get(1).and_then(RuntimeValue::as_i64).ok_or_else(|| {
                        "skip count disappeared after successful builtin evaluation".to_string()
                    })?;
                    if count < 0 {
                        return Err(
                            "negative skip count passed successful builtin evaluation".to_string()
                        );
                    }
                    // On a target whose usize is narrower than i64, a valid
                    // non-negative count that cannot convert is above every
                    // addressable List length and therefore selects no suffix.
                    let producer = match usize::try_from(count) {
                        Ok(count) => arg_producers
                            .first()
                            .and_then(Option::as_ref)
                            .and_then(|producer| producer.aggregate_suffix(count)),
                        Err(_) => None,
                    };
                    producer.filter(|producer| producer.matches_value(&value))
                }
                // `take` is a selection: its prefix keeps each element's
                // producer, as `skip`'s suffix does.
                "take" => {
                    let count = args.get(1).and_then(RuntimeValue::as_i64).ok_or_else(|| {
                        "take count disappeared after successful builtin evaluation".to_string()
                    })?;
                    let producer = match usize::try_from(count) {
                        Ok(count) => arg_producers
                            .first()
                            .and_then(Option::as_ref)
                            .and_then(|producer| producer.aggregate_prefix(count)),
                        // A count above every addressable length keeps the
                        // whole List.
                        Err(_) => arg_producers.first().cloned().flatten(),
                    };
                    producer.filter(|producer| producer.matches_value(&value))
                }
                // spec/04 section 4.7: a builtin produces every tensor held in
                // the aggregate it returns.
                _ if chelis_ir::host::produces_its_result(name)
                    && !matches!(value, RuntimeValue::Tensor(_)) =>
                {
                    Some(ResultProducer::Uniform(name.to_string()))
                }
                _ => matches!(value, RuntimeValue::Tensor(_)).then(|| ResultProducer::tensor(name)),
            };
            let producer = if name == "index" && !self.claims_are_vacuous(claims) {
                self.result_producer
                    .as_ref()
                    .and_then(ResultProducer::operation)
                    .ok_or_else(|| {
                        "host runtime: pending result claim reached a tensor without producer provenance"
                            .to_string()
                    })?
            } else {
                self.result_producer
                    .as_ref()
                    .and_then(ResultProducer::operation)
                    .unwrap_or(name)
            }
            .to_owned();
            if name != "pad_sequences_to" {
                for claim in claims {
                    self.check_declared_result_claim(claim, &value, &producer)?;
                }
            }
            return Ok(value);
        }

        // chelis#338 site B: a call to a top-level def whose body needs
        // named-axis routing (it reduces a named axis directly, or calls
        // a rank-polymorphic def that does). Route the call through IR
        // lowering at this boundary, where the callee's declared (named)
        // param types are available, exactly as `chelis build` calls a
        // signature-typed compiled function. Falls through to ordinary
        // interpretation when no routing strategy applies; the body's own
        // reduction then hits site A with frame-typed bindings.
        if let Some(callee) = var_name(func)
            && !self.bindings.contains_key(callee)
            && !self.tensor_bindings.contains_key(callee)
            && let Some((resolved, def_expr)) = self.lookup_top_level_def(callee)
            && def_expr.tag() == Some(DeepTag::Fn)
            && self.def_requires_named_axis_routing(&resolved)
            // chelis#1277 B2h: a def the C lane lowers as a kernel takes that
            // kernel at application; site B keeps only the defs C also
            // handles per call (rank- and precision-polymorphic ones).
            && self.def_kernel(&resolved)?.is_none()
            && let Some(routed) = self.try_named_axis_def_call(&resolved, &def_expr, kids, &args)?
        {
            return Ok(routed);
        }

        // chelis#721: when the callee names a `(fn …)`-bodied top-level def and
        // is NOT a local binding, resolve it directly to its Closure. A nullary
        // (or otherwise DAG-lowerable) def folds to a constant that lands in
        // `tensor_bindings`; `eval_var`'s precedence returns that Tensor BEFORE
        // `resolve_top_level` (eval.rs eval_var), so `eval_expr(func)` here would
        // hand back the folded Tensor and `apply_*` would reject it as "value is
        // not callable". Going through `resolve_top_level` bypasses only the
        // tensor_bindings shadow — a local binding (checked here) still wins.
        // Checked function aliases need the same direct resolution; applying
        // eval_var's nullary-thunk rule to an alias here would call it twice.
        let callable = if let Some(callee) = var_name(func)
            && !self.bindings.contains_key(callee)
            && let Some((resolved, def_expr)) = self.lookup_top_level_def(callee)
            && (def_expr.tag() == Some(DeepTag::Fn)
                || self
                    .program
                    .type_env()
                    .get(&resolved)
                    .is_some_and(|ty| ty.tag() == Some(DeepTag::TFn)))
        {
            self.resolve_top_level(&resolved)?
        } else {
            self.eval_expr(func)?
        };
        self.apply_resolved_callable_under_result_claim(
            callable,
            args,
            &arg_type_exprs,
            result_type_expr.as_ref(),
            claims,
            None,
        )
    }

    fn eval_if(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        let kids = node.children;
        let cond =
            self.eval_if_condition(kids.first().ok_or_else(|| "if missing cond".to_string())?)?;
        match cond {
            true => self.eval_expr(
                kids.get(1)
                    .ok_or_else(|| "if missing then branch".to_string())?,
            ),
            false => self.eval_expr(
                kids.get(2)
                    .ok_or_else(|| "if missing else branch".to_string())?,
            ),
        }
    }

    fn eval_if_condition(&mut self, condition: &Expr) -> Result<bool, String> {
        match self.eval_expr(condition)? {
            RuntimeValue::Bool(condition) => Ok(condition),
            other => Err(format!("if condition must be bool, got {other:?}")),
        }
    }

    /// One declaration contributes one invocation-local obligation. A producer
    /// need not be known until the returned branch or callee actually executes.
    fn declared_result_claim(declared: Option<&Expr>) -> Option<DeclaredResultClaim> {
        let (rank, axes) = declared_literal_result_extents(declared)?;
        Some(DeclaredResultClaim {
            rank,
            axes: axes
                .into_iter()
                .map(|(axis, required)| ResultAxisClaim::literal(axis, required))
                .collect(),
            first_sites: Vec::new(),
        })
    }

    /// Add a fixed-rank declaration's named axes to its invocation claim.
    ///
    /// A binder is resolved from its first witness among the tensor
    /// parameters, in signature order, which is the canonical side the entry
    /// plan compares every later witness against. With `first_sites`, a
    /// binder in the declaration's [`chelis_ir::lower::DimBinderRoles`]
    /// `return_only` set is bound by the body executing in that activation,
    /// and the claim keeps the axis for [`EvalContext::resolve_first_sites`];
    /// any other binder no tensor parameter declares adds nothing. Axes stay
    /// in declared order, which the C lane's claim frame also follows.
    fn with_named_result_axes(
        claim: Option<DeclaredResultClaim>,
        declared: Option<&Expr>,
        witnesses: &[(String, NamedResultSource, usize)],
        first_sites: Option<(u64, &chelis_ir::lower::DimBinderRoles)>,
    ) -> Result<Option<DeclaredResultClaim>, String> {
        let Some((_, dim_exprs)) = declared.and_then(tensor_type_dim_exprs) else {
            return Ok(claim);
        };
        if dim_exprs
            .iter()
            .any(|dim| dim.tag() == Some(DeepTag::DRank))
        {
            return Ok(claim);
        }
        let mut named = Vec::new();
        let mut first_site_axes = Vec::<FirstSiteAxis>::new();
        for (axis, dim_expr) in dim_exprs.iter().enumerate() {
            let Some((DeepTag::DName | DeepTag::DVar, kids)) = tagged_expr_children(dim_expr)
            else {
                continue;
            };
            let Some(binder) = kids.first().and_then(symbol_name) else {
                continue;
            };
            let Some((_, source, size)) = witnesses.iter().find(|(name, _, _)| name == binder)
            else {
                let binder = chelis_ir::lower::extent_binder_label(binder);
                if let Some((activation, roles)) = first_sites
                    && roles.return_only.contains(&binder)
                {
                    let at = first_site_axes
                        .iter()
                        .rposition(|site| site.binder == binder)
                        .map_or(first_site_axes.len(), |last| last + 1);
                    first_site_axes.insert(
                        at,
                        FirstSiteAxis {
                            axis,
                            binder,
                            activation,
                        },
                    );
                }
                continue;
            };
            let required = i64::try_from(*size)
                .map_err(|_| format!("dimension binder `{binder}` exceeds the exact i64 range"))?;
            named.push(ResultAxisClaim {
                axis,
                required,
                source: Some(source.clone()),
            });
        }
        if named.is_empty() && first_site_axes.is_empty() {
            return Ok(claim);
        }
        let mut claim = claim.unwrap_or(DeclaredResultClaim {
            rank: dim_exprs.len(),
            axes: Vec::new(),
            first_sites: Vec::new(),
        });
        claim.axes.extend(named);
        claim.axes.sort_by_key(|axis| axis.axis);
        claim.first_sites = first_site_axes;
        Ok(Some(claim))
    }

    /// Actualize a rank-polymorphic authored result at the invocation boundary.
    ///
    /// Argument evaluation has already completed, but the values have not yet
    /// moved into the callee frame. Their concrete ranks and dtypes may bind
    /// declaration variables; their literal sizes are evidence only for spread
    /// width and never become result claims. A fixed-rank declaration retains
    /// the established checked-signature path above.
    fn declared_result_claim_at_callsite(
        declared: Option<&Expr>,
        checked_params: Option<&[Expr]>,
        authored_params: &[Option<Expr>],
        args: &[RuntimeValue],
    ) -> Result<Option<DeclaredResultClaim>, String> {
        let Some(declared) = declared else {
            return Ok(None);
        };
        let Some((_, dim_exprs)) = tensor_type_dim_exprs(declared) else {
            return Ok(Self::declared_result_claim(Some(declared)));
        };
        if !dim_exprs
            .iter()
            .any(|dim| dim.tag() == Some(DeepTag::DRank))
        {
            return Ok(Self::declared_result_claim(Some(declared)));
        }
        if !dim_exprs.iter().any(|dim| dim.tag() == Some(DeepTag::DLit)) {
            return Ok(None);
        }

        let mut actualization_formals = Vec::new();
        let mut claim_formals = Vec::new();
        let mut actual_types = Vec::new();
        for (index, arg) in args.iter().enumerate() {
            let RuntimeValue::Tensor(tensor) = arg else {
                continue;
            };
            let checked = checked_params
                .and_then(|params| params.get(index))
                .or_else(|| authored_params.get(index).and_then(Option::as_ref));
            let authored = authored_params
                .get(index)
                .and_then(Option::as_ref)
                .or(checked);
            let (Some(checked), Some(authored)) = (checked, authored) else {
                continue;
            };
            if tensor_type_dim_exprs(checked).is_none() || tensor_type_dim_exprs(authored).is_none()
            {
                continue;
            }
            actualization_formals.push(Some(checked.clone()));
            claim_formals.push(Some(authored.clone()));
            actual_types.push(TensorType {
                dims: tensor
                    .value
                    .shape
                    .iter()
                    .copied()
                    .map(DimInfo::Lit)
                    .collect(),
                precision: tensor.precision,
            });
        }
        let actualized = chelis_ir::lower::actualize_authored_tensor_result_claim(
            &actualization_formals,
            &claim_formals,
            &actual_types,
            declared,
        )
        .map_err(|error| format!("host runtime: could not actualize result claim: {error}"))?;
        let axes = actualized
            .dims
            .iter()
            .enumerate()
            .filter_map(|(axis, dim)| match dim {
                DimInfo::Lit(required) => Some(
                    i64::try_from(*required)
                        .map(|required| ResultAxisClaim::literal(axis, required))
                        .map_err(|_| "declared result extent exceeds the exact i64 range"),
                ),
                DimInfo::Named(_, _) => None,
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(str::to_string)?;
        Ok((!axes.is_empty()).then_some(DeclaredResultClaim {
            rank: actualized.dims.len(),
            axes,
            first_sites: Vec::new(),
        }))
    }

    /// Evaluate a host function's body and discharge its declared-result claim
    /// at the source position of the operation that produces the returned
    /// value, which is what `spec/04-type-system.md` section 4.7's placement
    /// rule requires of a guard over a locally computed value: an independent
    /// effect that precedes that operation is observed first, and one that
    /// follows it is observed only if the guard passes.
    fn eval_under_result_claim(
        &mut self,
        expr: &Expr,
        claims: &[DeclaredResultClaim],
    ) -> Result<RuntimeValue, String> {
        if claims.is_empty() {
            return self.eval_expr(expr);
        }
        match expr.carrier() {
            ExprCarrier::DecodedNode(tag, metadata, kids) => {
                let node = EvalNode::new(expr, tag, metadata, kids);
                match tag {
                    DeepTag::Block => {
                        if let Some((last, init)) = kids.split_last() {
                            for child in init {
                                self.eval_expr(child)?;
                            }
                            return self.eval_under_result_claim(last, claims);
                        }
                    }
                    DeepTag::Let => return self.eval_let_under_result_claim(node, claims),
                    DeepTag::If => {
                        let condition =
                            self.eval_if_condition(kids.first().ok_or("if missing cond")?)?;
                        let selected = match condition {
                            true => kids.get(1).ok_or("if missing then branch")?,
                            false => kids.get(2).ok_or("if missing else branch")?,
                        };
                        return self.eval_under_result_claim(selected, claims);
                    }
                    DeepTag::Match => return self.eval_match_under_result_claim(node, claims),
                    DeepTag::App => return self.eval_app_under_result_claim(node, claims),
                    _ => {}
                }
            }
            ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => {}
        }
        let value = self.eval_expr(expr)?;
        if self.claims_are_vacuous(claims) {
            return Ok(value);
        }
        let producer = self
            .result_producer
            .as_ref()
            .and_then(ResultProducer::operation)
            .ok_or_else(|| {
                "host runtime: pending result claim reached a tensor without producer provenance"
                    .to_string()
            })?
            .to_owned();
        for claim in claims {
            self.check_declared_result_claim(claim, &value, &producer)?;
        }
        Ok(value)
    }

    /// [`Self::eval_let`], forwarding a declared-result claim to the binding
    /// whose value the let returns.
    ///
    /// The lexical selector retains each alias's defining scope. A binding
    /// initialized by a branch passes the obligations into that branch before
    /// subsequent effects execute; other bindings never inherit them.
    fn eval_let_under_result_claim(
        &mut self,
        node: EvalNode<'_>,
        claims: &[DeclaredResultClaim],
    ) -> Result<RuntimeValue, String> {
        self.eval_let_with_claims(node, Some(claims))
    }

    fn eval_let_with_claims(
        &mut self,
        node: EvalNode<'_>,
        claims: Option<&[DeclaredResultClaim]>,
    ) -> Result<RuntimeValue, String> {
        let kids = node.children;
        let bind_list_expr = kids
            .first()
            .ok_or_else(|| "let missing bindings".to_string())?;
        let body = kids.get(1).ok_or_else(|| "let missing body".to_string())?;
        let bind_node = match bind_list_expr.carrier() {
            ExprCarrier::DecodedNode(tag, metadata, children) => {
                EvalNode::new(bind_list_expr, tag, metadata, children)
            }
            ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => {
                return Err("let missing bindings".to_string());
            }
        };
        if bind_node.tag != DeepTag::Bind {
            return Err("let bindings malformed".to_string());
        }
        let saved = self.bindings.clone();
        let saved_types = self.binding_types.clone();
        // Restore both frame maps on every exit path (including bind or body
        // evaluation errors) so a caught-and-continued error can never leak
        // partial binds or stale binding types.
        let result = (|| {
            let declaration_name = self
                .active_declaration_names
                .last()
                .or_else(|| self.resolving_top_levels.last())
                .cloned();
            let lowering = self
                .session
                .as_ref()
                .map(chelis_ir::host::HostLoweringSession::checked_subexpr_lowering_context);

            // Surf's sequential bindings desugar to a direct tail chain of
            // nested Lets. A later local ascription can therefore name a
            // producer in an earlier layer even though neither layer's Bind
            // list alone contains the full alias path. C host lowering
            // already flattens exactly this tail shape. Do the same here only
            // when a checked region actually crosses layers; ordinary Lets
            // retain their established recursive evaluation path.
            let flattened = sequential_eval_let_chain(node.expr);
            let flattened_regions = flattened
                .as_ref()
                .and_then(|chain| {
                    lowering.as_ref().map(|context| {
                        context.local_ascription_binding_regions(
                            &chain.bind_expr,
                            declaration_name.as_deref(),
                        )
                    })
                })
                .unwrap_or_default();
            let crosses_nested_let = flattened.as_ref().is_some_and(|chain| {
                flattened_regions.iter().any(|region| {
                    let producer_layer = chain.bindings[region.producer_binding_index() / 2].layer;
                    region
                        .ascription_binding_indices()
                        .iter()
                        .any(|index| chain.bindings[*index / 2].layer != producer_layer)
                })
            });
            let current_bindings = bind_node
                .children
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| SequentialEvalLetBinding {
                    binding: &pair[0],
                    initializer: &pair[1],
                    layer: 0,
                })
                .collect::<Vec<_>>();
            let (let_bindings, planned_bind_expr, planned_body) = if crosses_nested_let {
                let chain = flattened
                    .as_ref()
                    .expect("cross-layer local region requires a flattened let chain");
                (&chain.bindings, &chain.bind_expr, chain.body)
            } else {
                (&current_bindings, bind_list_expr, body)
            };
            let guarded =
                claims.and_then(|_| let_result_binding_index(planned_bind_expr, planned_body));
            let local_regions = if crosses_nested_let {
                flattened_regions
            } else {
                lowering
                    .as_ref()
                    .map(|context| {
                        context.local_ascription_binding_regions(
                            planned_bind_expr,
                            declaration_name.as_deref(),
                        )
                    })
                    .unwrap_or_default()
            }
            .into_iter()
            .map(|region| (region.producer_binding_index(), region))
            .collect::<BTreeMap<_, _>>();

            for (ordinal, binding) in let_bindings.iter().enumerate() {
                let index = ordinal * 2;
                let name = symbol_name(binding.binding)
                    .ok_or_else(|| "let binding must bind a name".to_string())?;
                // Record the bound expr's static type (checker-annotated
                // `{type: ...}` on the value expr, or the source binding's
                // known type for a bare var) so chelis#338 named-axis routing
                // can recover named dims for let-bound tensors. The explicit
                // `None` insert masks any same-named top-level type.
                let static_ty = self.static_type_expr_of(binding.initializer);
                let local_claims = if guarded == Some(index) {
                    claims.unwrap_or(&[])
                } else {
                    &[]
                };
                let value = if let Some(region) = local_regions.get(&index) {
                    self.eval_checked_local_ascription_region(
                        lowering
                            .as_ref()
                            .expect("a local region requires its lowering context"),
                        region,
                        local_claims,
                    )?
                } else if guarded == Some(index) {
                    self.eval_under_result_claim(
                        binding.initializer,
                        claims.expect("guarded binding requires result claims"),
                    )?
                } else {
                    self.eval_expr(binding.initializer)?
                };
                let result_producer = self.result_producer.take();
                self.binding_types.insert(name.to_string(), static_ty);
                self.bindings
                    .insert_with_result_producer(name.to_string(), value, result_producer);
            }

            match (claims, guarded) {
                (Some(_), Some(_)) | (None, _) => self.eval_expr(planned_body),
                (Some(claims), None) => self.eval_under_result_claim(planned_body, claims),
            }
        })();
        self.bindings = saved;
        self.binding_types = saved_types;
        result
    }

    fn eval_checked_local_ascription_region(
        &mut self,
        lowering: &chelis_ir::lower::SubexprLoweringContext,
        region: &chelis_ir::lower::LocalAscriptionBindingRegion,
        inherited_result_claims: &[DeclaredResultClaim],
    ) -> Result<RuntimeValue, String> {
        if let Some(local_claims) = lowering
            .host_local_ascription_claims(region)
            .map_err(|diagnostic| diagnostic.to_string())?
        {
            let mut claims = inherited_result_claims.to_vec();
            for claim in local_claims {
                let axes = claim
                    .dims
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, dim)| match dim {
                        DimInfo::Lit(required) => {
                            Some(ResultAxisClaim::literal(axis, *required as i64))
                        }
                        DimInfo::Named(_, _) => None,
                    })
                    .collect();
                claims.push(DeclaredResultClaim {
                    rank: claim.dims.len(),
                    axes,
                    first_sites: Vec::new(),
                });
            }
            return self.eval_under_result_claim(region.initializer(), &claims);
        }
        let lowering = lowering
            .for_local_ascription_region(region)
            .claiming_parameter_binders_at_activation();
        let mut scoped_types = UnordMap::new();
        let mut staged_inputs = UnordMap::new();
        for (name, value) in self.bindings.to_sorted() {
            let ty = match value {
                RuntimeValue::Tensor(tensor) => match self.binding_types.get(name) {
                    Some(Some(declared)) => declared_tensor_type_for_value(declared, tensor)
                        .map_err(|error| {
                            format!(
                                "host runtime could not type local-ascription input \
                                 `{name}` for checked lowering: {error}"
                            )
                        })?,
                    Some(None) | None => TensorType {
                        dims: tensor
                            .value
                            .shape
                            .iter()
                            .copied()
                            .map(DimInfo::Lit)
                            .collect(),
                        precision: tensor.precision,
                    },
                },
                RuntimeValue::Scalar(payload) => TensorType {
                    dims: Vec::new(),
                    precision: self
                        .binding_types
                        .get(name)
                        .and_then(Option::as_ref)
                        .and_then(extract_prim_from_type_expr)
                        .unwrap_or(payload.dtype()),
                },
                RuntimeValue::Bool(_) => TensorType {
                    dims: Vec::new(),
                    precision: Prim::Bool,
                },
                _ => continue,
            };
            let input =
                stage_kernel_argument("local tensor ascription", name, value, ty.precision)?;
            scoped_types.insert(name.clone(), ty);
            staged_inputs.insert(name.clone(), input);
        }

        let dag = chelis_ir::lower::try_lower_subexpr_program_with_context(
            region.expression(),
            scoped_types,
            &lowering,
        )
        .map_err(|diagnostic| {
            format!(
                "host runtime could not lower a checked local tensor ascription \
                 for evaluation: {diagnostic}"
            )
        })?;
        let dag = &dag;
        let roots = dag.roots().to_vec();
        if roots.is_empty() {
            return Err(
                "host runtime: checked local tensor-ascription lowering produced no roots"
                    .to_string(),
            );
        }
        let producer_operation = match chelis_ir::axis_sources::local_dim_guard_sites(dag)?.first()
        {
            Some((_, claim)) => claim.op,
            // An ascription at its binder's first producing site owes no
            // comparison and so has no guard site; its value's producer still
            // names the operation a later result claim reports.
            None => chelis_ir::axis_sources::result_extent_sites(dag, roots[0])
                .first()
                .map(|site| site.operation())
                .ok_or_else(|| {
                    "host runtime: checked local tensor ascription produced neither a local \
                     guard site nor a producing operation"
                        .to_string()
                })?,
        };
        let prepared = chelis_ir::eval::prepare_tensor_roots_inputs_with_demand(
            dag,
            &roots,
            |name, demand| self.prepare_named_axis_input(name, demand, &staged_inputs),
        )?;
        let values = chelis_ir::eval::eval_tensor_roots_exact(dag, &roots, |name| {
            prepared.get(name).cloned()
        });
        let values = self.mark_numeric_trap_from_trusted_result(values)?;
        let value = pack_dag_roots(dag, &roots, &values, "local tensor ascription")?;
        for claim in inherited_result_claims {
            self.check_declared_result_claim(claim, &value, producer_operation)?;
        }
        self.observe_local_named_sites(&lowering, region, &value, producer_operation)?;
        self.result_producer = Some(ResultProducer::tensor(producer_operation));
        Ok(value)
    }

    fn eval_let(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        self.eval_let_with_claims(node, None)
    }

    fn eval_tuple_get(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        let kids = node.children;
        let tuple = self.eval_expr(
            kids.first()
                .ok_or_else(|| "tuple-get missing tuple".to_string())?,
        )?;
        let tuple_producer = self.result_producer.take();
        let index_expr = kids
            .get(1)
            .ok_or_else(|| "tuple-get missing index".to_string())?;
        let index = match index_expr.carrier() {
            ExprCarrier::DecodedNode(_, _, children) => children,
            ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => {
                return Err("tuple-get index must be a literal".to_string());
            }
        };
        let idx = index
            .first()
            .and_then(int_value)
            .ok_or_else(|| "tuple-get index must be an int literal".to_string())?
            as usize;
        match tuple {
            RuntimeValue::Tuple(items) => {
                let value = items
                    .get(idx)
                    .cloned()
                    .ok_or_else(|| format!("tuple-get index {idx} out of bounds"))?;
                self.result_producer = tuple_producer
                    .as_ref()
                    .and_then(|producer| producer.child(idx))
                    .filter(|producer| producer.matches_value(&value));
                Ok(value)
            }
            other => Err(format!("tuple-get expects tuple input, got {other:?}")),
        }
    }

    fn eval_match(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        self.eval_match_under_result_claim(node, &[])
    }

    fn eval_match_under_result_claim(
        &mut self,
        node: EvalNode<'_>,
        claims: &[DeclaredResultClaim],
    ) -> Result<RuntimeValue, String> {
        let kids = node.children;
        let scrutinee = self.eval_expr(
            kids.first()
                .ok_or_else(|| "match missing scrutinee".to_string())?,
        )?;
        let scrutinee_producer = self.result_producer.take();
        for arm in kids.iter().skip(1) {
            let arm_kids = match arm.carrier() {
                ExprCarrier::DecodedNode(DeepTag::Arm, _, children) => children,
                ExprCarrier::DecodedNode(_, _, _)
                | ExprCarrier::StructuralList(_)
                | ExprCarrier::UndecodableHead(_, _, _)
                | ExprCarrier::Atom(_)
                | ExprCarrier::MetadataMap(_)
                | ExprCarrier::MetadataExpression(_) => {
                    return Err("match arm must be a decoded `arm` node".to_string());
                }
            };
            if arm_kids.len() != 3 {
                return Err(
                    "match `arm` must contain exactly a pattern, guard, and body".to_string(),
                );
            }
            let saved = self.bindings.clone();
            let saved_types = self.binding_types.clone();
            // Pattern-bound names carry no declared types; insert `None`
            // markers afterwards so they shadow rather than leak an outer
            // same-named binding's type into the arm body (chelis#338).
            if pattern_matches_with_result_producer(
                &scrutinee,
                &arm_kids[0],
                &mut self.bindings,
                &self.adt_fields,
                scrutinee_producer.as_ref(),
            )? {
                for (name, _) in self.bindings.to_sorted() {
                    if !saved.contains_key(name) {
                        self.binding_types.insert(name.clone(), None);
                    }
                }
                // [04-PAT-2]: the guard runs only after the pattern matched,
                // in the arm's scope. `false` discards the arm's bindings and
                // tries the next arm; a failing guard fails the match.
                let selected = self.eval_match_guard(&arm_kids[1]);
                if !matches!(selected, Ok(true)) {
                    self.bindings = saved;
                    self.binding_types = saved_types;
                    selected?;
                    continue;
                }
                let value = self.eval_under_result_claim(&arm_kids[2], claims);
                self.bindings = saved;
                self.binding_types = saved_types;
                return value;
            }
            self.bindings = saved;
            self.binding_types = saved_types;
        }
        Err("non-exhaustive runtime match".to_string())
    }

    /// An arm without a guard carries `()` in its guard slot and is selected
    /// whenever its pattern matches.
    fn eval_match_guard(&mut self, guard: &Expr) -> Result<bool, String> {
        if matches!(guard.carrier(), ExprCarrier::StructuralList([])) {
            return Ok(true);
        }
        match self.eval_expr(guard)? {
            RuntimeValue::Bool(selected) => Ok(selected),
            other => Err(format!(
                "match arm guard must be bool, got {}",
                render_value(&other)
            )),
        }
    }

    pub(super) fn apply_resolved_callable(
        &mut self,
        callable: RuntimeValue,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, String> {
        self.apply_resolved_callable_with_arg_types(callable, args, &[], None)
    }

    /// A selected Host root may have dead authored tensor parameters. Its
    /// admission records which positional actuals exist; the Unit values in
    /// absent slots supply closure arity and must never become tensor claims.
    pub(super) fn apply_selected_root_callable(
        &mut self,
        callable: RuntimeValue,
        args: Vec<RuntimeValue>,
        present: &[bool],
    ) -> Result<RuntimeValue, String> {
        self.apply_resolved_callable_under_result_claim(
            callable,
            args,
            &[],
            None,
            &[],
            Some(present),
        )
    }

    /// Like [`Self::apply_resolved_callable`], but additionally records a
    /// static Deep type expression per argument into the callee frame's
    /// `binding_types` (chelis#338 named-axis routing). The closure's own
    /// declared param type wins; `arg_type_exprs` fills the gap for
    /// synthesized params with no annotation (e.g. `__chelis_pipe`).
    fn apply_resolved_callable_with_arg_types(
        &mut self,
        callable: RuntimeValue,
        args: Vec<RuntimeValue>,
        arg_type_exprs: &[Option<Expr>],
        result_type_expr: Option<&Expr>,
    ) -> Result<RuntimeValue, String> {
        self.apply_resolved_callable_under_result_claim(
            callable,
            args,
            arg_type_exprs,
            result_type_expr,
            &[],
            None,
        )
    }

    fn apply_resolved_callable_under_result_claim(
        &mut self,
        callable: RuntimeValue,
        args: Vec<RuntimeValue>,
        arg_type_exprs: &[Option<Expr>],
        result_type_expr: Option<&Expr>,
        claims: &[DeclaredResultClaim],
        present: Option<&[bool]>,
    ) -> Result<RuntimeValue, String> {
        let mut formal_claims = Vec::new();
        if let Some(contracts) = callable.invocation_contracts() {
            for contract in contracts {
                let actualized = check_callable_invocation_contract(
                    contract,
                    &args,
                    self.session.as_ref(),
                    present,
                )?;
                let Some((result, params)) =
                    checked_function_children(contract).and_then(<[Expr]>::split_last)
                else {
                    continue;
                };
                let mut witnesses = Vec::new();
                for (path, ty, shape) in &actualized {
                    for (axis, (dim, size)) in ty.dims.iter().zip(shape).enumerate() {
                        if let DimInfo::Named(name, _) = dim
                            && name != "*"
                            && !witnesses.iter().any(|(seen, _, _)| seen == name)
                        {
                            witnesses.push((
                                name.clone(),
                                NamedResultSource {
                                    claim: chelis_ir::lower::extent_binder_label(name),
                                    parameter: path.clone(),
                                    axis,
                                },
                                *size,
                            ));
                        }
                    }
                }
                let authored = params.iter().cloned().map(Some).collect::<Vec<_>>();
                let claim = Self::declared_result_claim_at_callsite(
                    Some(result),
                    Some(params),
                    &authored,
                    &args,
                )?;
                let claim = Self::with_named_result_axes(claim, Some(result), &witnesses, None)?;
                formal_claims.extend(claim);
            }
        }
        formal_claims.extend_from_slice(claims);
        self.apply_resolved_callable_with_arg_types_impl(
            callable,
            args,
            arg_type_exprs,
            result_type_expr,
            &formal_claims,
            present,
        )
    }

    fn apply_resolved_callable_with_arg_types_impl(
        &mut self,
        callable: RuntimeValue,
        args: Vec<RuntimeValue>,
        arg_type_exprs: &[Option<Expr>],
        result_type_expr: Option<&Expr>,
        inherited_claims: &[DeclaredResultClaim],
        present: Option<&[bool]>,
    ) -> Result<RuntimeValue, String> {
        // A caller's first-site axes resolve in the caller's activation,
        // whose sites have all run by the time its callee produces the
        // result: the callee only ever sees concrete axes.
        let inherited_claims = &inherited_claims
            .iter()
            .map(|claim| self.resolve_first_sites(claim))
            .collect::<Vec<_>>();
        match callable {
            RuntimeValue::Closure {
                checked_function: _,
                params,
                param_types,
                return_type,
                checked_signature,
                invocation_contracts: _,
                body,
                env,
                precision_env,
                def_name,
            } => {
                if params.len() != args.len() {
                    return Err(format!(
                        "closure expected {} args, got {}",
                        params.len(),
                        args.len()
                    ));
                }
                let active_declaration = def_name.clone();
                // chelis#1277 B2h: a def the C lane lowers as a kernel is
                // applied through that kernel, so eval runs the DAG C emits
                // for it and the runtime-extent classes and guards derived
                // from that DAG fire on both lanes ([05-MOV-1],
                // runtime_extents.md C2.5). The host-lane decision for the
                // same def interprets the body below, exactly as before.
                if let Some(name) = def_name.as_deref()
                    && let Some(kernel) = self.def_kernel(name)?
                {
                    // Arguments already ran in the caller. Kernel/staged
                    // capture providers now see only declaration scope.
                    let saved = std::mem::take(&mut self.bindings);
                    let saved_types = std::mem::take(&mut self.binding_types);
                    let saved_precisions = std::mem::take(&mut self.precision_bindings);
                    let result =
                        self.apply_def_kernel(name, &kernel, &params, args, inherited_claims);
                    self.bindings = saved;
                    self.binding_types = saved_types;
                    self.precision_bindings = saved_precisions;
                    return result;
                }
                let saved = std::mem::take(&mut self.bindings);
                let saved_types = std::mem::take(&mut self.binding_types);
                let saved_precisions = std::mem::take(&mut self.precision_bindings);
                self.bindings = env;
                self.precision_bindings = precision_env;
                if let Some(name) = active_declaration.as_ref() {
                    self.active_declaration_names.push(name.clone());
                }
                let caller_activation = self.activation_extents.enter();
                let value = (|| {
                    // A dimension variable declared by a tensor parameter is
                    // also an exact runtime i64 value in the callee frame.
                    // Recover it from the actual tensor shape before the
                    // arguments are moved into their ordinary bindings. This
                    // makes `expand(b, 0, k)` consume the same witnessed
                    // extent as compiled lowering instead of looking up an
                    // unbound textual runtime name (chelis#1382).
                    let mut dimension_bindings: UnordMap<String, usize> = UnordMap::new();
                    let mut named_result_witnesses: Vec<(String, NamedResultSource, usize)> =
                        Vec::new();
                    let checked_params = checked_signature
                        .as_ref()
                        .and_then(checked_function_children)
                        .and_then(|children| children.split_last())
                        .map(|(_, params)| params);
                    let actualized_entries = actualize_tensor_entry_parameters(
                        checked_params,
                        &param_types,
                        &args,
                        &params,
                        self.session.as_ref(),
                        present,
                    )?;
                    let mut entry_inputs = Vec::with_capacity(actualized_entries.len());
                    let mut entry_shapes: Vec<Vec<usize>> =
                        Vec::with_capacity(actualized_entries.len());
                    for (parameter, ty, shape) in actualized_entries {
                        for (axis, (dim, size)) in ty.dims.iter().zip(&shape).enumerate() {
                            if let DimInfo::Named(name, _) = dim
                                && name != "*"
                            {
                                dimension_bindings.entry(name.clone()).or_insert(*size);
                                if !named_result_witnesses
                                    .iter()
                                    .any(|(seen, _, _)| seen == name)
                                {
                                    named_result_witnesses.push((
                                        name.clone(),
                                        NamedResultSource {
                                            claim: chelis_ir::lower::extent_binder_label(name),
                                            parameter: parameter.clone(),
                                            axis,
                                        },
                                        *size,
                                    ));
                                }
                            }
                        }
                        entry_inputs.push(chelis_ir::host::HostTensorInput {
                            name: parameter,
                            ty,
                        });
                        entry_shapes.push(shape);
                    }
                    let entry_plan = chelis_ir::host::SignatureEntryPlan::new(entry_inputs);
                    check_signature_entry_plan(
                        &entry_plan,
                        &entry_shapes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    )?;
                    let caller_precisions = saved_precisions.clone();
                    let mut call_precisions = UnordMap::new();
                    for (declared, actual) in param_types.iter().zip(arg_type_exprs) {
                        if let (Some(declared), Some(actual)) = (declared, actual) {
                            collect_checked_precision_bindings(
                                declared,
                                actual,
                                &caller_precisions,
                                &mut call_precisions,
                            )?;
                        }
                    }
                    if let (Some(declared), Some(actual)) = (return_type.as_ref(), result_type_expr)
                    {
                        collect_checked_precision_bindings(
                            declared,
                            actual,
                            &caller_precisions,
                            &mut call_precisions,
                        )?;
                    }
                    // A separately declared signature and the checked body can
                    // name the same precision with different binders. Actualize
                    // both from the same checked call-site evidence; body-local
                    // collectors must not infer a dtype from their payload.
                    if let Some((checked_result, checked_params)) = checked_signature
                        .as_ref()
                        .and_then(checked_function_children)
                        .and_then(|children| children.split_last())
                    {
                        for (checked, actual) in checked_params.iter().zip(arg_type_exprs) {
                            if let Some(actual) = actual {
                                collect_checked_precision_bindings(
                                    checked,
                                    actual,
                                    &caller_precisions,
                                    &mut call_precisions,
                                )?;
                            }
                        }
                        if let Some(actual) = result_type_expr {
                            collect_checked_precision_bindings(
                                checked_result,
                                actual,
                                &caller_precisions,
                                &mut call_precisions,
                            )?;
                        }
                    }
                    // The call-site instantiation is fresh (spec/04 §5.8):
                    // callee-owned binders shadow a same-spelled lexical
                    // binding instead of conflicting with it.
                    self.precision_bindings.merge(call_precisions);
                    let checked_result = checked_signature
                        .as_ref()
                        .and_then(checked_function_children)
                        .and_then(<[Expr]>::last);
                    // Alias substitution retains the authored dimension names
                    // through the result, just as through List parameters.
                    let authored_result = return_type
                        .as_ref()
                        .map(|ty| normalize_authored_entry_type(ty, self.session.as_ref()))
                        .transpose()?;
                    let declared_result = authored_result
                        .as_ref()
                        .filter(|ty| tensor_type_dim_exprs(ty).is_some())
                        .or(checked_result);
                    let declaration_claim = Self::declared_result_claim_at_callsite(
                        declared_result,
                        checked_params,
                        &param_types,
                        &args,
                    )?;
                    // The parameters bind their binders before the body runs;
                    // the body's sites bind the rest.
                    for (name, source, size) in &named_result_witnesses {
                        let size = i64::try_from(*size).map_err(|_| {
                            format!("dimension binder `{name}` exceeds the exact i64 range")
                        })?;
                        self.activation_extents.bind(
                            chelis_ir::lower::extent_binder_label(name),
                            source.clone(),
                            size,
                        );
                    }
                    // The body's first sites bind exactly the declaration's
                    // return-only binders, the set its named sites read.
                    let roles = match (active_declaration.as_deref(), self.session.as_ref()) {
                        (Some(name), Some(session)) => session
                            .checked_subexpr_lowering_context()
                            .dim_binder_roles(name),
                        _ => chelis_ir::lower::DimBinderRoles::default(),
                    };
                    let declaration_claim = Self::with_named_result_axes(
                        declaration_claim,
                        declared_result,
                        &named_result_witnesses,
                        Some((self.activation_extents.current, &roles)),
                    )?;
                    // Dimension-name order is canonical for extending the callee frame.
                    for (name, size) in dimension_bindings.into_sorted() {
                        let size = i64::try_from(size).map_err(|_| {
                            format!("dimension binder `{name}` exceeds the exact i64 range")
                        })?;
                        self.bindings.insert(name, RuntimeValue::int64(size));
                    }
                    for (index, (param, arg)) in params.into_iter().zip(args).enumerate() {
                        let declared = param_types
                            .get(index)
                            .cloned()
                            .flatten()
                            .or_else(|| arg_type_exprs.get(index).cloned().flatten());
                        // chelis#729 Phase 1: a tensor argument ingress-finalizes
                        // at the param's DECLARED element dtype (the host-lane
                        // mirror of the DAG evaluator's Load ingress). Without
                        // this, an Int64-tagged `to_tensor` literal flows into an
                        // i8-typed param and the arithmetic runs at the wrong
                        // width (the chelis#718 eval-tensor cell).
                        let mut arg = match (declared.as_ref().and_then(declared_tensor_prim), arg)
                        {
                            (Some(prim), RuntimeValue::Tensor(tensor)) => {
                                RuntimeValue::Tensor(ingress_tensor_to_declared(tensor, prim)?)
                            }
                            (_, arg) => arg,
                        };
                        let authored_callable = declared
                            .as_ref()
                            .map(|ty| normalize_authored_entry_type(ty, self.session.as_ref()))
                            .transpose()?;
                        let callable_contract = authored_callable
                            .as_ref()
                            .filter(|ty| checked_function_children(ty).is_some())
                            .or_else(|| checked_params.and_then(|params| params.get(index)));
                        if callable_contract
                            .is_some_and(|ty| checked_function_children(ty).is_some())
                            && let Some(invocation_contracts) = arg.invocation_contracts_mut()
                        {
                            // This parameter is a new adapter around any
                            // contracts the supplied callable already carries,
                            // so its entry runs first at the eventual call.
                            invocation_contracts
                                .insert(0, callable_contract.expect("checked above").clone());
                        }
                        self.binding_types.insert(param.clone(), declared);
                        let producer = ResultProducer::interface_load(&arg);
                        self.bindings
                            .insert_with_result_producer(param, arg, producer);
                    }
                    // chelis#1739 and chelis#1771: a declared literal result
                    // extent is a claim the host lane owes a runtime verdict
                    // on, and section 4.7 places that verdict at the source
                    // position of the operation that produces the returned
                    // value rather than at the return boundary. That is why the
                    // body is evaluated through `eval_under_result_claim`
                    // instead of being checked after it returns: a block-bodied
                    // return whose producing binding is followed by a `print`
                    // must trap before the print runs.
                    //
                    // Frames retain distinct declarations even when the literal
                    // values agree. Forwarding a frame does not append it again.
                    let late_first_sites = declaration_claim
                        .as_ref()
                        .filter(|claim| !claim.first_sites.is_empty())
                        .map(|claim| DeclaredResultClaim {
                            rank: claim.rank,
                            axes: Vec::new(),
                            first_sites: claim.first_sites.clone(),
                        });
                    let mut claims = declaration_claim.into_iter().collect::<Vec<_>>();
                    claims.extend_from_slice(inherited_claims);
                    let value = self.eval_under_result_claim(&body, &claims)?;
                    // The declared result is a later site of every binder
                    // the body bound, including one whose first site ran
                    // after the returned value's producer, where the claim
                    // above found it unbound.
                    if let Some(claim) = late_first_sites {
                        let producer = self
                            .result_producer
                            .as_ref()
                            .and_then(ResultProducer::operation)
                            .unwrap_or("return")
                            .to_owned();
                        self.check_declared_result_claim(&claim, &value, &producer)?;
                    }
                    Ok(value)
                })();
                self.activation_extents.exit(caller_activation);
                if active_declaration.is_some() {
                    let popped = self.active_declaration_names.pop();
                    debug_assert_eq!(popped.as_ref(), active_declaration.as_ref());
                }
                self.bindings = saved;
                self.binding_types = saved_types;
                self.precision_bindings = saved_precisions;
                value
            }
            RuntimeValue::Transform {
                kind,
                transform_expr,
                captured_env,
                invocation_contracts: _,
            } => self.apply_transform(kind, &transform_expr, captured_env, args),
            other => Err(format!("value is not callable: {other:?}")),
        }
    }

    fn eval_cast(&mut self, node: EvalNode<'_>) -> Result<RuntimeValue, String> {
        let kids = node.children;
        let value = self.eval_expr(
            kids.first()
                .ok_or_else(|| "cast missing value".to_string())?,
        )?;
        let numeric_input = matches!(
            value,
            RuntimeValue::Scalar(_) | RuntimeValue::Tensor(_) | RuntimeValue::Bool(_)
        );
        let target = kids
            .get(1)
            .and_then(|ty| match ty.carrier() {
                ExprCarrier::DecodedNode(_, _, children) => children.first(),
                ExprCarrier::StructuralList(_)
                | ExprCarrier::UndecodableHead(_, _, _)
                | ExprCarrier::Atom(_)
                | ExprCarrier::MetadataMap(_)
                | ExprCarrier::MetadataExpression(_) => None,
            })
            .and_then(symbol_name)
            .ok_or_else(|| "cast missing target type".to_string())?;
        // Resolve the textual target into a Prim using the canonical
        // active dtype map. The type checker has already rejected
        // f8e4m3 (spec/04-type-system.md §1.1.1) at this point so the
        // host eval lane just needs to pick the right re-pack.
        //
        // Binder targets actualize from the call site's concrete precision.
        let target_prim = prim_from_name(target)
            .or_else(|| self.precision_bindings.get(target).copied())
            .ok_or_else(|| format!("cast target `{target}` is not a recognized primitive type"))?;
        // [05-OP-6]: the truncating rung has its own sealed kernel and
        // its own trap brand. The checker has already pinned the pair to
        // float source / integer target.
        if chelis_deep::cast_mode_of(kids)
            .map_err(|selector| format!("`{selector}` is not a recognized cast mode selector"))?
            == chelis_deep::CastMode::Trunc
        {
            let result = match value {
                RuntimeValue::Scalar(payload) => {
                    chelis_types::cast_trunc_scalar("cast_trunc", payload.value(), target_prim)
                        .map(RuntimeValue::from_scalar_value)
                        .map_err(|trap| trap.to_string())
                }
                RuntimeValue::Tensor(tensor) => cast_trunc_tensor_value(tensor, target_prim),
                // The checker pins the source to a float scalar or
                // tensor ([05-OP-6]), so this arm is unreachable for a
                // well-typed program. It names no value: rendering one
                // here would need a third numeric formatter, which
                // `spec/design/faithful_observation.md` B2.4 forbids.
                _ => Err(format!(
                    "unsupported cast_trunc operand for target {}; \
                     [05-OP-6] requires a float scalar or tensor source",
                    target_prim.name()
                )),
            };
            return if numeric_input {
                self.mark_numeric_trap_from_trusted_result(result)
            } else {
                result
            };
        }
        // The CHECKED default ladder (`chelis_types::cast_scalar`; the
        // chelis#759 one-rule-per-direction obligation), identical to
        // the tensor surfaces: out-of-range integer targets trap,
        // fractional-to-integer traps Domain instead of choosing an
        // implicit rounding rule, and a bool target requires exactly 0/1.
        let result = match (value, target_prim) {
            (RuntimeValue::Bool(value), Prim::Bool) => Ok(RuntimeValue::Bool(value)),
            (RuntimeValue::String(value), Prim::String) => Ok(RuntimeValue::String(value)),
            (RuntimeValue::Scalar(payload), dst_dtype)
                if dst_dtype.is_integer() || dst_dtype.is_float() || dst_dtype == Prim::Bool =>
            {
                chelis_types::cast_scalar("cast", payload.value(), dst_dtype)
                    .map(|cast| match cast.as_bool_exact() {
                        Some(flag) => RuntimeValue::Bool(flag),
                        None => RuntimeValue::from_scalar_value(cast),
                    })
                    .map_err(|trap| trap.to_string())
            }
            (RuntimeValue::Bool(value), dst_dtype)
                if dst_dtype.is_integer() || dst_dtype.is_float() =>
            {
                let source =
                    chelis_types::scalar_from_i64("cast", Prim::Bool, if value { 1 } else { 0 })
                        .expect("bool payload is always in the bool value set");
                chelis_types::cast_scalar("cast", source, dst_dtype)
                    .map(RuntimeValue::from_scalar_value)
                    .map_err(|trap| trap.to_string())
            }
            (RuntimeValue::Tensor(tensor), _) => cast_tensor_value(tensor, target_prim),
            (other, _) => Err(format!(
                "unsupported cast from {other:?} to {}",
                target_prim.name()
            )),
        };
        if numeric_input {
            self.mark_numeric_trap_from_trusted_result(result)
        } else {
            result
        }
    }

    fn check_declared_result_claim(
        &mut self,
        claim: &DeclaredResultClaim,
        value: &RuntimeValue,
        producer: &str,
    ) -> Result<(), String> {
        let verdict = self.resolve_first_sites(claim).verdict(value, producer);
        self.mark_numeric_trap_from_trusted_result(verdict)
    }

    fn check_declared_shape_claim(
        &mut self,
        claim: &DeclaredResultClaim,
        shape: &[usize],
        producer: &str,
    ) -> Result<(), String> {
        let verdict = self
            .resolve_first_sites(claim)
            .shape_verdict(shape, producer);
        self.mark_numeric_trap_from_trusted_result(verdict)
    }

    /// Whether no claim checks an axis here: none is declared, and no first
    /// site has bound a binder one names.
    fn claims_are_vacuous(&self, claims: &[DeclaredResultClaim]) -> bool {
        claims
            .iter()
            .all(|claim| self.resolve_first_sites(claim).axes.is_empty())
    }

    /// `claim` with each first-site axis resolved against the site that
    /// bound its binder in the claim's activation, after the declared axes,
    /// as the C lane checks its first-site frames after the declared frame.
    /// An axis whose binder that activation has not bound is dropped.
    fn resolve_first_sites(&self, claim: &DeclaredResultClaim) -> DeclaredResultClaim {
        let mut resolved = DeclaredResultClaim {
            rank: claim.rank,
            axes: claim.axes.clone(),
            first_sites: Vec::new(),
        };
        for site in &claim.first_sites {
            if let Some((source, required)) =
                self.activation_extents.get(site.activation, &site.binder)
            {
                resolved.axes.push(ResultAxisClaim {
                    axis: site.axis,
                    required: *required,
                    source: Some(source.clone()),
                });
            }
        }
        resolved
    }

    /// Relate the named sites of a local tensor ascription region to the
    /// other sites of the executing activation (spec/04-type-system.md
    /// section 4.4.1): a binder the activation has not bound is bound here,
    /// and one it has bound is a section 4.7 claim this site's extent must
    /// meet, trapping `Domain` at `producer` otherwise.
    fn observe_local_named_sites(
        &mut self,
        lowering: &chelis_ir::lower::SubexprLoweringContext,
        region: &chelis_ir::lower::LocalAscriptionBindingRegion,
        value: &RuntimeValue,
        producer: &str,
    ) -> Result<(), String> {
        let RuntimeValue::Tensor(tensor) = value else {
            return Ok(());
        };
        for site in lowering.local_ascription_named_sites(region) {
            let Some(&observed) = tensor.value.shape.get(site.axis) else {
                continue;
            };
            let observed = i64::try_from(observed)
                .map_err(|_| format!("extent `{}` exceeds the exact i64 range", site.binder))?;
            match self.activation_extents.current(&site.binder) {
                Some((_, required)) if *required != observed => {
                    let required = *required;
                    return self.mark_numeric_trap_from_trusted_result(Err(format!(
                        "extent `{}`: claimed = {required}, {producer} axis {} = {observed}\n\
                         numeric trap: domain in {producer} at i64",
                        site.binder, site.axis
                    )));
                }
                Some(_) => {}
                // Only an output-inferred binder may be bound by a site
                // (spec/04-type-system.md section 4.4.1). Any other binder is
                // a parameter's, and an activation without its record, such
                // as a lambda's, cannot claim it: refuse as the lowering does
                // for a binder it cannot resolve.
                None if !site.output_inferred => {
                    return Err(format!(
                        "local tensor ascription `{}` cannot resolve authored extent `{}` in \
                         this activation",
                        site.binding, site.binder
                    ));
                }
                None => self.activation_extents.bind(
                    site.binder.clone(),
                    NamedResultSource {
                        claim: site.binder,
                        parameter: site.binding,
                        axis: site.axis,
                    },
                    observed,
                ),
            }
        }
        Ok(())
    }

    pub(super) fn mark_numeric_trap_from_trusted_result<T>(
        &mut self,
        result: Result<T, String>,
    ) -> Result<T, String> {
        if result.as_ref().err().is_some_and(|message| {
            message
                .lines()
                .any(chelis_types::NumericTrap::is_canonical_line)
        }) {
            self.failure_kind = RuntimeFailureKind::NumericTrap;
        }
        result
    }

    /// [05-OP-37] in the host walk: `dropout(k, x, rate)`. The rate is the
    /// input dtype's tagged scalar, validated before any element is drawn
    /// from the key.
    fn eval_dropout_builtin(&mut self, args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
        let key = expect_key_arg(args, 0, "dropout")?;
        let input = expect_tensor_arg(args, 1)?;
        let rate = expect_float_control(args, 2, "dropout")?;
        let prepared = chelis_types::PreparedDropout::new(input.value.storage(), rate)
            .map_err(|error| error.to_string())?;
        let storage = prepared.apply(key).map_err(|error| error.to_string())?;
        Ok(RuntimeValue::Tensor(RuntimeTensorValue::new(
            IrTensorValue::from_storage(input.value.shape.clone(), storage),
        )))
    }

    fn eval_builtin(
        &mut self,
        name: &str,
        args: &[RuntimeValue],
        arg_type_exprs: &[Option<Expr>],
        result_type_expr: Option<&Expr>,
    ) -> Result<RuntimeValue, String> {
        if matches!(
            name,
            "key_from_seed" | "split_key" | "split_keys" | "fold_in"
        ) && let Some(RuntimeValue::Tensor(input)) = args.first()
            && (!input.value.shape.is_empty()
                || arg_type_exprs
                    .first()
                    .and_then(Option::as_ref)
                    .is_some_and(|ty| {
                        matches!(
                            ty.carrier(),
                            ExprCarrier::DecodedNode(DeepTag::TTensor, _, _)
                        )
                    }))
        {
            // A lowered scalar key can also use a rank-zero tensor carrier.
            // Preserve its scalar operation contract using the checked type.
            use chelis_types::dtype_semantics::{
                KeyHalf, fold_in_storage, key_from_seed_storage, split_key_storage,
                split_keys_storage,
            };
            let shape = &input.value.shape;
            let storage = input.value.storage();
            let tensor = |shape: Vec<usize>, storage| {
                RuntimeValue::Tensor(RuntimeTensorValue::new(IrTensorValue::from_storage(
                    shape, storage,
                )))
            };
            let computed = match name {
                "key_from_seed" => key_from_seed_storage(storage),
                "split_key" => {
                    let left =
                        split_key_storage(storage, KeyHalf::Left).map_err(|e| e.to_string())?;
                    let right =
                        split_key_storage(storage, KeyHalf::Right).map_err(|e| e.to_string())?;
                    return Ok(RuntimeValue::Tuple(
                        vec![tensor(shape.clone(), left), tensor(shape.clone(), right)].into(),
                    ));
                }
                "fold_in" => {
                    let index = expect_tensor_arg(args, 1)?;
                    if shape != &index.value.shape {
                        return Err(format!(
                            "fold_in requires exactly equal key and index shapes ([05-OP-72])\n{}",
                            chelis_types::NumericTrap::Domain {
                                op: "fold_in",
                                prim: Prim::Int64,
                            }
                        ));
                    }
                    fold_in_storage(storage, index.value.storage())
                }
                "split_keys" => {
                    let count = expect_i64_key_operand(args, 1, "split_keys")?
                        .as_i64_exact()
                        .ok_or("split_keys expects i64")?;
                    let count = usize::try_from(count).map_err(|_| {
                        chelis_types::NumericTrap::Domain {
                            op: "split_keys",
                            prim: Prim::Int64,
                        }
                        .to_string()
                    })?;
                    check_split_keys_declared_extent(result_type_expr, count)?;
                    let mut shape = shape.clone();
                    shape.push(count);
                    admit_host_key_result(&shape)?;
                    let storage = split_keys_storage(storage, count).map_err(|e| e.to_string())?;
                    return Ok(tensor(shape, storage));
                }
                _ => unreachable!(),
            }
            .map_err(|e| e.to_string())?;
            return Ok(tensor(shape.clone(), computed));
        }
        match name {
            // [05-OP-69]..[05-OP-72]: the key operations over the scalar key
            // value, with the kernels every lane shares.
            "key_from_seed" => {
                let seed = expect_i64_key_operand(args, 0, "key_from_seed")?;
                chelis_types::RandomKey::from_seed(seed)
                    .map(RuntimeValue::Key)
                    .map_err(|error| error.to_string())
            }
            "split_key" => {
                let (left, right) = expect_key_arg(args, 0, "split_key")?.split();
                Ok(RuntimeValue::Tuple(
                    vec![RuntimeValue::Key(left), RuntimeValue::Key(right)].into(),
                ))
            }
            "fold_in" => {
                let key = expect_key_arg(args, 0, "fold_in")?;
                let n = expect_i64_key_operand(args, 1, "fold_in")?;
                key.fold_in(n)
                    .map(RuntimeValue::Key)
                    .map_err(|error| error.to_string())
            }
            "split_keys" => {
                let key = expect_key_arg(args, 0, "split_keys")?;
                let count = expect_i64_key_operand(args, 1, "split_keys")?
                    .as_i64_exact()
                    .ok_or_else(|| "split_keys expects an i64 count".to_string())?;
                // [05-OP-71]: a negative count traps before allocation, as
                // the DAG evaluator and the C lane trap.
                let count = usize::try_from(count).map_err(|_| {
                    chelis_types::NumericTrap::Domain {
                        op: "split_keys",
                        prim: Prim::Int64,
                    }
                    .to_string()
                })?;
                check_split_keys_declared_extent(result_type_expr, count)?;
                admit_host_key_result(&[count])?;
                Ok(RuntimeValue::Tensor(RuntimeTensorValue::new(
                    IrTensorValue::from_storage(
                        vec![count],
                        chelis_types::TensorStorage::from_keys(key.split_n(count)),
                    ),
                )))
            }
            "add" => numeric_binop(args, Some(IntBinOp::Add), Some(FloatBinOp::Add)),
            "sub" => numeric_binop(args, Some(IntBinOp::Sub), Some(FloatBinOp::Sub)),
            "mul" => numeric_binop(args, Some(IntBinOp::Mul), Some(FloatBinOp::Mul)),
            // #387: integer `div`/`mod` trap on a zero divisor with one
            // shared diagnostic instead of returning a silently-wrong value
            // (`f64` div round-trip yielded `i64::MAX`/`-1`); float `div`
            // keeps IEEE-754 (`1.0 / 0.0 == inf`). The C backend follows the
            // platform SIGFPE for the same integer operands.
            "div" => eval_div(args),
            // chelis#178: integer-division primitives. `floor_div` rounds
            // the quotient toward -inf (ints and floats); `trunc_div`
            // rounds toward zero (integer-only). Both trap on an integer
            // zero divisor with the shared diagnostic.
            "floor_div" => eval_floor_div(args),
            "trunc_div" => eval_trunc_div(args),
            // Tier-1 `max_elem` and `min_elem` are direct element-wise
            // selection identities. The integer path compares at the
            // declared width, and the float path returns the exact operand
            // selected by [05-OP-40], including its stored NaN payload or
            // signed-zero bits. Wired for Chelis-Lang/chelis#185/#1306.
            "max_elem" => numeric_binop(args, Some(IntBinOp::Max), Some(FloatBinOp::Max)),
            "min_elem" => numeric_binop(args, Some(IntBinOp::Min), Some(FloatBinOp::Min)),
            "mod" => eval_mod(args),
            "neg" => numeric_unop(args, Some(IntUnOp::Neg), Some(FloatUnOp::Neg)),
            "recip" => numeric_unop(args, None, Some(FloatUnOp::Recip)),
            "exp" => float_unop_with_tensor(args, FloatUnOp::Exp),
            "log" => float_unop_with_tensor(args, FloatUnOp::Log),
            "sin" => float_unop_with_tensor(args, FloatUnOp::Sin),
            "sqrt" => float_unop_with_tensor(args, FloatUnOp::Sqrt),
            // Tier 1 unary primitives wired for issue Chelis-Lang/chelis#185.
            // Each delegates to the same `float_unop_with_tensor` /
            // `numeric_unop` helper used by the already-wired siblings; the
            // tensor lane runs through `f32` to mirror the C backend's libm
            // emit (`cosf`/`tanf`/`floorf`/`ceilf`/`atanf`), which is the
            // canonical-evaluator equivalent (per
            // `feedback_evaluator_byte_identical_gate`).
            "cos" => float_unop_with_tensor(args, FloatUnOp::Cos),
            "tan" => float_unop_with_tensor(args, FloatUnOp::Tan),
            "atan" => float_unop_with_tensor(args, FloatUnOp::Atan),
            "floor" => numeric_unop(args, Some(IntUnOp::Floor), Some(FloatUnOp::Floor)),
            "ceil" => numeric_unop(args, Some(IntUnOp::Ceil), Some(FloatUnOp::Ceil)),
            // Round-half-to-even (banker's rounding), matching the DAG
            // evaluator and the C backend's `rintf`. NOT `round`, which
            // is ties-away-from-zero.
            "round" => numeric_unop(args, Some(IntUnOp::Round), Some(FloatUnOp::Round)),
            // `abs` accepts ints and floats and is sign-flipping for both;
            // route through `numeric_unop` so scalar Int64/Int32/F32/F64
            // inputs all keep their dtype.
            "abs" => numeric_unop(args, Some(IntUnOp::Abs), Some(FloatUnOp::Abs)),
            // [05-OP-36]: unit and two List, tuple, Dict, Option or ADT values
            // compare structurally to one bool scalar, and `neq` is the
            // complement. Scalars, strings and direct tensor operands keep
            // their own comparison.
            "eq" | "neq" if is_recursive_equality_operand(args) => {
                let equal = runtime_values_equal(&args[0], &args[1])?;
                Ok(RuntimeValue::Bool(equal == (name == "eq")))
            }
            "eq" => compare_eq(args),
            "neq" => compare_runtime(args, CompareOp::Ne),
            "cmplt" => ordered_compare(args, CompareOp::Lt),
            "lt" => ordered_compare(args, CompareOp::Lt),
            "gt" => ordered_compare(args, CompareOp::Gt),
            "gte" => ordered_compare(args, CompareOp::Gte),
            "lte" => ordered_compare(args, CompareOp::Lte),
            "uniform_like" => {
                let key = expect_key_arg(args, 0, "uniform_like")?;
                let template = expect_tensor_arg(args, 1)?;
                let low = expect_float_control(args, 2, "uniform_like")?;
                let high = expect_float_control(args, 3, "uniform_like")?;
                // [05-OP-8]: the bounds validate before any element is drawn
                // from the key.
                let prepared = prepare_uniform_like(&template, low, high)?;
                Ok(RuntimeValue::Tensor(uniform_like_value(
                    &template, &prepared, key,
                )?))
            }
            // Logical ops dispatch on the actual argument shape: scalar
            // bool args (already wired) keep the `bool_binop` /
            // `bool_unop` path; tensor-bool args route through the
            // dedicated `tensor_bool_*` helpers wired for issue
            // Chelis-Lang/chelis#185. Per the brief's pinned decision,
            // the tensor lane is NOT a transparent extension of the
            // scalar lane — it pins input precision to `Bool` and
            // requires matching shapes, which scalar broadcasting
            // would hide.
            "and" => match (args.first(), args.get(1)) {
                (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
                    tensor_bool_binop(lhs, rhs, |a, b| a && b).map(RuntimeValue::Tensor)
                }
                _ => bool_binop(args, |lhs, rhs| lhs && rhs),
            },
            "or" => match (args.first(), args.get(1)) {
                (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
                    tensor_bool_binop(lhs, rhs, |a, b| a || b).map(RuntimeValue::Tensor)
                }
                _ => bool_binop(args, |lhs, rhs| lhs || rhs),
            },
            "not" => match args.first() {
                Some(RuntimeValue::Tensor(tensor)) => {
                    tensor_bool_unop(tensor, |value| !value).map(RuntimeValue::Tensor)
                }
                _ => bool_unop(args, |value| !value),
            },
            "bitand" => bitwise_binop(args, chelis_types::BitwiseKind::And),
            "bitor" => bitwise_binop(args, chelis_types::BitwiseKind::Or),
            "bitxor" => bitwise_binop(args, chelis_types::BitwiseKind::Xor),
            "shl" => bitwise_binop(args, chelis_types::BitwiseKind::ShiftLeft),
            "shr" => bitwise_binop(args, chelis_types::BitwiseKind::ShiftRight),
            "string_len" => {
                let value = expect_string_arg(args, 0)?;
                Ok(RuntimeValue::int64(value.chars().count() as i64))
            }
            "char_code" => {
                let value = expect_string_arg(args, 0)?;
                let mut chars = value.chars();
                let character = chars.next().ok_or_else(|| {
                    "Domain: char_code requires exactly one Unicode scalar value [05-OP-58]"
                        .to_string()
                })?;
                if chars.next().is_some() {
                    return Err(
                        "Domain: char_code requires exactly one Unicode scalar value [05-OP-58]"
                            .to_string(),
                    );
                }
                Ok(RuntimeValue::int64(i64::from(u32::from(character))))
            }
            "char_from_code" => {
                let code = expect_int_arg(args, 0)?;
                let character = u32::try_from(code)
                    .ok()
                    .and_then(char::from_u32)
                    .ok_or_else(|| {
                        "Domain: char_from_code requires a Unicode scalar value [05-OP-58]"
                            .to_string()
                    })?;
                Ok(RuntimeValue::String(character.to_string()))
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
                Ok(match chelis_runtime::text_parse::to_int(&value) {
                    Some(parsed) => RuntimeValue::Adt {
                        ctor: "Some".to_string(),
                        source_name: "Some".to_string(),
                        fields: vec![RuntimeValue::int64(parsed)].into(),
                        field_names: None,
                    },
                    None => RuntimeValue::Adt {
                        ctor: "None".to_string(),
                        source_name: "None".to_string(),
                        fields: Vec::new().into(),
                        field_names: None,
                    },
                })
            }
            "to_float" => {
                let value = expect_string_arg(args, 0)?;
                Ok(match chelis_runtime::text_parse::to_float(&value) {
                    Some(parsed) => RuntimeValue::Adt {
                        ctor: "Some".to_string(),
                        source_name: "Some".to_string(),
                        fields: vec![RuntimeValue::float64(parsed)].into(),
                        field_names: None,
                    },
                    None => RuntimeValue::Adt {
                        ctor: "None".to_string(),
                        source_name: "None".to_string(),
                        fields: Vec::new().into(),
                        field_names: None,
                    },
                })
            }
            "len" => match args.first() {
                Some(RuntimeValue::List(list)) => Ok(RuntimeValue::int64(list.len() as i64)),
                Some(RuntimeValue::Dict(entries)) => Ok(RuntimeValue::int64(entries.len() as i64)),
                other => Err(format!("len expects list or dict arg, got {other:?}")),
            },
            "index" => {
                let list = expect_list_arg(args, 0)?;
                let index = expect_int_arg(args, 1)?;
                if index < 0 {
                    return Err(format!("index requires non-negative index, got {index}"));
                }
                list.get(index as usize).cloned().ok_or_else(|| {
                    format!("index {index} out of bounds for list of len {}", list.len())
                })
            }
            "append" => {
                let mut list = expect_list_arg(args, 0)?;
                list.push(
                    args.get(1)
                        .cloned()
                        .ok_or_else(|| "append expects 2 arguments".to_string())?,
                );
                Ok(RuntimeValue::List(list.into()))
            }
            "concat" => match (args.first(), args.get(1).and_then(RuntimeValue::as_i64)) {
                (Some(RuntimeValue::List(parts)), Some(axis))
                    if parts
                        .iter()
                        .all(|item| matches!(item, RuntimeValue::Tensor(_))) =>
                {
                    tensor_concat_value(parts, axis)
                }
                _ => {
                    let mut lhs = expect_list_arg(args, 0)?;
                    lhs.extend(expect_list_arg(args, 1)?);
                    Ok(RuntimeValue::List(lhs.into()))
                }
            },
            "take" => {
                let list = expect_list_arg(args, 0)?;
                let count = expect_int_arg(args, 1)?;
                if count < 0 {
                    return Err(format!("take requires non-negative count, got {count}"));
                }
                Ok(RuntimeValue::List(
                    list.into_iter().take(count as usize).collect(),
                ))
            }
            "skip" => {
                let list = expect_list_arg(args, 0)?;
                let count = expect_int_arg(args, 1)?;
                if count < 0 {
                    return Err(format!("skip requires non-negative count, got {count}"));
                }
                Ok(RuntimeValue::List(
                    list.into_iter().skip(count as usize).collect(),
                ))
            }
            // [05-OP-67]: the explicit one-argument consume. The checker
            // rejects every other arity for a SURFACE call, but a lowering
            // can synthesize a `Builtin` node below it -- chelis#2310's
            // `Cons` tail did, with the two-argument spelling this operation
            // used to carry -- so the count is checked here rather than
            // assumed. Returning unit for a two-argument call would bind a
            // list tail to unit with no diagnostic.
            "drop" => match args.len() {
                1 => Ok(RuntimeValue::Unit),
                n => Err(format!(
                    "drop expects 1 argument, got {n}: the List slice is `skip` ([05-OP-54])"
                )),
            },
            "chunk" => {
                let list = expect_list_arg(args, 0)?;
                let size = expect_int_arg(args, 1)?;
                if size <= 0 {
                    return Err(format!("chunk requires positive size, got {size}"));
                }
                let mut out = Vec::new();
                let size = size as usize;
                for chunk in list.chunks(size) {
                    out.push(RuntimeValue::List(chunk.to_vec().into()));
                }
                Ok(RuntimeValue::List(out.into()))
            }
            "range" => {
                let start = expect_int_arg(args, 0)?;
                let end = expect_int_arg(args, 1)?;
                Ok(RuntimeValue::List(
                    (start..end).map(RuntimeValue::int64).collect(),
                ))
            }
            "map" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "map expects 2 arguments".to_string())?;
                let items = expect_list_arg(args, 1)?;
                let callback_arg_types = [arg_type_exprs
                    .get(1)
                    .and_then(Option::as_ref)
                    .and_then(checked_list_element)
                    .cloned()];
                let callback_result_type = result_type_expr.and_then(checked_list_element);
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.apply_resolved_callable_with_arg_types(
                        callback.clone(),
                        vec![item],
                        &callback_arg_types,
                        callback_result_type,
                    )?);
                }
                Ok(RuntimeValue::List(out.into()))
            }
            "filter" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "filter expects 2 arguments".to_string())?;
                let items = expect_list_arg(args, 1)?;
                let callback_arg_types = [arg_type_exprs
                    .get(1)
                    .and_then(Option::as_ref)
                    .and_then(checked_list_element)
                    .cloned()];
                let mut out = Vec::new();
                for item in items {
                    let keep = self.apply_resolved_callable_with_arg_types(
                        callback.clone(),
                        vec![item.clone()],
                        &callback_arg_types,
                        None,
                    )?;
                    match keep {
                        RuntimeValue::Bool(true) => out.push(item),
                        RuntimeValue::Bool(false) => {}
                        other => {
                            return Err(format!("filter callback must return bool, got {other:?}"));
                        }
                    }
                }
                Ok(RuntimeValue::List(out.into()))
            }
            "fold" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "fold expects 3 arguments".to_string())?;
                let mut acc = args
                    .get(1)
                    .cloned()
                    .ok_or_else(|| "fold expects 3 arguments".to_string())?;
                let items = expect_list_arg(args, 2)?;
                let callback_arg_types = [
                    arg_type_exprs.get(1).cloned().flatten(),
                    arg_type_exprs
                        .get(2)
                        .and_then(Option::as_ref)
                        .and_then(checked_list_element)
                        .cloned(),
                ];
                for item in items {
                    acc = self.apply_resolved_callable_with_arg_types(
                        callback.clone(),
                        vec![acc, item],
                        &callback_arg_types,
                        result_type_expr,
                    )?;
                }
                Ok(acc)
            }
            "scan" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "scan expects 3 arguments".to_string())?;
                let mut acc = args
                    .get(1)
                    .cloned()
                    .ok_or_else(|| "scan expects 3 arguments".to_string())?;
                let items = expect_list_arg(args, 2)?;
                let callback_arg_types = [
                    arg_type_exprs.get(1).cloned().flatten(),
                    arg_type_exprs
                        .get(2)
                        .and_then(Option::as_ref)
                        .and_then(checked_list_element)
                        .cloned(),
                ];
                let callback_result_type = result_type_expr.and_then(checked_list_element);
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    acc = self.apply_resolved_callable_with_arg_types(
                        callback.clone(),
                        vec![acc, item],
                        &callback_arg_types,
                        callback_result_type,
                    )?;
                    out.push(acc.clone());
                }
                Ok(RuntimeValue::List(out.into()))
            }
            // Issue #257: iterative scan that produces a rank-1 tensor
            // directly, bypassing the right-recursive Surf list build that
            // overflows the host worker stack at ~10k elements. The arg
            // shape is `(initial: T, fn: (T, i64) -> T, n: i64)` and
            // the loop runs `n` times on the host with no Surf-level
            // recursion. The output precision is taken from the initial
            // value's scalar dtype.
            "tensor_scan" => {
                if args.len() != 3 {
                    return Err(format!(
                        "tensor_scan expects 3 arguments (initial, fn, n), got {}",
                        args.len()
                    ));
                }
                let initial = args[0].clone();
                let callback = args[1].clone();
                let n = expect_int_arg(args, 2)?;
                if n < 0 {
                    return Err(format!(
                        "tensor_scan requires a non-negative length, got {n}"
                    ));
                }
                let precision = match &initial {
                    RuntimeValue::Scalar(payload) => payload.dtype(),
                    RuntimeValue::Bool(_) => Prim::Bool,
                    other => {
                        return Err(format!(
                            "tensor_scan expects a scalar initial value (numeric or bool), got {other:?}"
                        ));
                    }
                };
                // Reject non-callable callback up front so the error message
                // points at the second argument instead of failing inside the
                // first apply.
                if !matches!(
                    &callback,
                    RuntimeValue::Closure { .. } | RuntimeValue::Transform { .. }
                ) {
                    return Err(format!(
                        "tensor_scan expects a callable second argument, got {callback:?}"
                    ));
                }
                let n = n as usize;
                // chelis#729 Phase 1: collect per family so integer scans
                // stay exact at full i64 (a scan accumulating above 2^53
                // no longer collapses through f64 storage).
                let mut ints: Vec<i64> = Vec::new();
                let mut floats: Vec<chelis_types::ScalarValue> = Vec::new();
                let mut acc = initial;
                let callback_type_children = arg_type_exprs
                    .get(1)
                    .and_then(Option::as_ref)
                    .and_then(checked_function_children);
                let initial_type = arg_type_exprs.first().cloned().flatten();
                let callback_arg_types = [
                    initial_type.clone(),
                    callback_type_children.and_then(|children| children.get(1).cloned()),
                ];
                let callback_result_type = initial_type.as_ref();
                for i in 0..n {
                    let index = RuntimeValue::int64(i as i64);
                    acc = self.apply_resolved_callable_with_arg_types(
                        callback.clone(),
                        vec![acc, index],
                        &callback_arg_types,
                        callback_result_type,
                    )?;
                    // Validate per-step that the accumulator stayed the same
                    // scalar precision; this catches a misbehaving callback
                    // that returns a different dtype before it corrupts the
                    // output tensor buffer.
                    match &acc {
                        RuntimeValue::Scalar(payload) => {
                            if payload.dtype() != precision {
                                return Err(format!(
                                    "tensor_scan callback returned a {} scalar but the initial \
                                     value's dtype is {}",
                                    payload.dtype().name(),
                                    precision.name()
                                ));
                            }
                            match payload.value().as_i64_exact() {
                                Some(v) => ints.push(v),
                                None => floats.push(payload.value()),
                            }
                        }
                        RuntimeValue::Bool(b) => {
                            if precision != Prim::Bool {
                                return Err(format!(
                                    "tensor_scan callback returned a bool but the initial \
                                     value's dtype is {}",
                                    precision.name()
                                ));
                            }
                            ints.push(if *b { 1 } else { 0 });
                        }
                        other => {
                            return Err(format!(
                                "tensor_scan callback must return a scalar, got {other:?}"
                            ));
                        }
                    }
                }
                let tensor = if precision.is_float() {
                    RuntimeTensorValue::from_scalars(precision, vec![n], &floats)
                } else {
                    RuntimeTensorValue::from_wide_int("tensor_scan", precision, vec![n], ints)?
                };
                Ok(RuntimeValue::Tensor(tensor))
            }
            "partition" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "partition expects 2 arguments".to_string())?;
                let items = expect_list_arg(args, 1)?;
                let callback_arg_types = [arg_type_exprs
                    .get(1)
                    .and_then(Option::as_ref)
                    .and_then(checked_list_element)
                    .cloned()];
                let mut kept = Vec::new();
                let mut rejected = Vec::new();
                for item in items {
                    let keep = self.apply_resolved_callable_with_arg_types(
                        callback.clone(),
                        vec![item.clone()],
                        &callback_arg_types,
                        None,
                    )?;
                    match keep {
                        RuntimeValue::Bool(true) => kept.push(item),
                        RuntimeValue::Bool(false) => rejected.push(item),
                        other => {
                            return Err(format!(
                                "partition callback must return bool, got {other:?}"
                            ));
                        }
                    }
                }
                Ok(RuntimeValue::Tuple(
                    vec![
                        RuntimeValue::List(kept.into()),
                        RuntimeValue::List(rejected.into()),
                    ]
                    .into(),
                ))
            }
            "flat_map" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "flat_map expects 2 arguments".to_string())?;
                let items = expect_list_arg(args, 1)?;
                let callback_arg_types = [arg_type_exprs
                    .get(1)
                    .and_then(Option::as_ref)
                    .and_then(checked_list_element)
                    .cloned()];
                let mut out = Vec::new();
                for item in items {
                    let mapped = self.apply_resolved_callable_with_arg_types(
                        callback.clone(),
                        vec![item.clone()],
                        &callback_arg_types,
                        result_type_expr,
                    )?;
                    let RuntimeValue::List(inner) = mapped else {
                        return Err(format!(
                            "flat_map callback must return List, got {mapped:?}"
                        ));
                    };
                    out.extend(inner);
                }
                Ok(RuntimeValue::List(out.into()))
            }
            "flatten" => {
                let lists = expect_list_arg(args, 0)?;
                let mut out = Vec::new();
                for item in lists {
                    let RuntimeValue::List(inner) = item else {
                        return Err(format!("flatten expects nested List input, got {item:?}"));
                    };
                    out.extend(inner);
                }
                Ok(RuntimeValue::List(out.into()))
            }
            "zip" => {
                let lhs = expect_list_arg(args, 0)?;
                let rhs = expect_list_arg(args, 1)?;
                Ok(RuntimeValue::List(
                    lhs.into_iter()
                        .zip(rhs)
                        .map(|(lhs, rhs)| RuntimeValue::Tuple(vec![lhs, rhs].into()))
                        .collect(),
                ))
            }
            "enumerate" => {
                let items = expect_list_arg(args, 0)?;
                Ok(RuntimeValue::List(
                    items
                        .into_iter()
                        .enumerate()
                        .map(|(index, value)| {
                            RuntimeValue::Tuple(
                                vec![RuntimeValue::int64(index as i64), value].into(),
                            )
                        })
                        .collect(),
                ))
            }
            "dict_of" => {
                let entries = expect_list_arg(args, 0)?;
                let mut dict = Vec::with_capacity(entries.len());
                for entry in entries {
                    let RuntimeValue::Tuple(items) = entry else {
                        return Err("dict_of expects a List of 2-tuples".to_string());
                    };
                    if items.len() != 2 {
                        return Err("dict_of expects a List of 2-tuples".to_string());
                    }
                    ensure_dict_key_supported(&items[0])?;
                    upsert_dict_entry(&mut dict, items[0].clone(), items[1].clone());
                }
                Ok(RuntimeValue::Dict(dict.into()))
            }
            "dict_get" => {
                let dict = expect_dict_arg(args, 0)?;
                let key = args
                    .get(1)
                    .ok_or_else(|| "dict_get expects 2 arguments".to_string())?;
                ensure_dict_key_supported(key)?;
                Ok(match dict_lookup(&dict, key) {
                    Some(value) => RuntimeValue::Adt {
                        ctor: "Some".to_string(),
                        source_name: "Some".to_string(),
                        fields: vec![value.clone()].into(),
                        field_names: None,
                    },
                    None => RuntimeValue::Adt {
                        ctor: "None".to_string(),
                        source_name: "None".to_string(),
                        fields: Vec::new().into(),
                        field_names: None,
                    },
                })
            }
            "dict_contains" => {
                let dict = expect_dict_arg(args, 0)?;
                let key = args
                    .get(1)
                    .ok_or_else(|| "dict_contains expects 2 arguments".to_string())?;
                ensure_dict_key_supported(key)?;
                Ok(RuntimeValue::Bool(dict_lookup(&dict, key).is_some()))
            }
            "dict_remove" => {
                let dict = expect_dict_arg(args, 0)?;
                let key = args
                    .get(1)
                    .ok_or_else(|| "dict_remove expects 2 arguments".to_string())?;
                ensure_dict_key_supported(key)?;
                Ok(RuntimeValue::Dict(
                    dict.into_iter()
                        .filter(|(existing_key, _)| !runtime_value_eq(existing_key, key))
                        .collect(),
                ))
            }
            "dict_insert" => {
                let mut dict = expect_dict_arg(args, 0)?;
                let key = args
                    .get(1)
                    .cloned()
                    .ok_or_else(|| "dict_insert expects 3 arguments".to_string())?;
                ensure_dict_key_supported(&key)?;
                let value = args
                    .get(2)
                    .cloned()
                    .ok_or_else(|| "dict_insert expects 3 arguments".to_string())?;
                upsert_dict_entry(&mut dict, key, value);
                Ok(RuntimeValue::Dict(dict.into()))
            }
            "dict_merge" => {
                let mut lhs = expect_dict_arg(args, 0)?;
                let rhs = expect_dict_arg(args, 1)?;
                for (key, value) in rhs {
                    ensure_dict_key_supported(&key)?;
                    upsert_dict_entry(&mut lhs, key, value);
                }
                Ok(RuntimeValue::Dict(lhs.into()))
            }
            "dict_keys" => {
                let dict = expect_dict_arg(args, 0)?;
                Ok(RuntimeValue::List(
                    dict.into_iter().map(|(key, _)| key).collect(),
                ))
            }
            "dict_values" => {
                let dict = expect_dict_arg(args, 0)?;
                Ok(RuntimeValue::List(
                    dict.into_iter().map(|(_, value)| value).collect(),
                ))
            }
            "dict_entries" => {
                let dict = expect_dict_arg(args, 0)?;
                Ok(RuntimeValue::List(
                    dict.into_iter()
                        .map(|(key, value)| RuntimeValue::Tuple(vec![key, value].into()))
                        .collect(),
                ))
            }
            "copy" => match args.first() {
                Some(RuntimeValue::Tensor(tensor)) => Ok(RuntimeValue::Tensor(tensor.clone())),
                other => Err(format!("copy expects tensor input, got {other:?}")),
            },
            "reshape" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let shape = expect_list_arg(args, 1)?;
                tensor_reshape_value(&tensor, &shape).map(RuntimeValue::Tensor)
            }
            "to_tensor" => {
                let values = expect_list_arg(args, 0)?;
                // [05-OP-57]: recover dtype and rank from checked types, not
                // from payloads (empty Lists cannot carry that evidence).
                let mut list_depth = 0;
                let mut leaf_type = arg_type_exprs.first().and_then(Option::as_ref);
                while let Some(inner) = leaf_type.and_then(checked_list_element) {
                    list_depth += 1;
                    leaf_type = Some(inner);
                }
                let result_children = result_type_expr
                    .and_then(tagged_expr_children)
                    .filter(|(tag, _)| *tag == DeepTag::TTensor)
                    .map(|(_, children)| children);
                let precision = result_children
                    .and_then(|children| children.last())
                    .and_then(|ty| checked_precision_leaf(ty, &self.precision_bindings))
                    .or_else(|| {
                        leaf_type
                            .and_then(|ty| checked_precision_leaf(ty, &self.precision_bindings))
                    })
                    .ok_or("to_tensor requires a resolved checked element dtype [05-OP-57]")?;
                let expected_shape = if let Some((_, dimensions)) =
                    result_children.and_then(|children| children.split_last())
                {
                    dimensions
                        .iter()
                        .map(|dimension| {
                            let Some((tag, children)) = tagged_expr_children(dimension) else {
                                return Err(
                                    "to_tensor has malformed checked dimensions".to_string()
                                );
                            };
                            let extent = match tag {
                                DeepTag::DLit => children.first().and_then(int_value),
                                DeepTag::DName | DeepTag::DVar => children
                                    .first()
                                    .and_then(symbol_name)
                                    .and_then(|name| self.bindings.get(name))
                                    .and_then(|value| match value {
                                        RuntimeValue::Scalar(payload)
                                            if payload.dtype() == Prim::Int64 =>
                                        {
                                            Some(payload.as_i64())
                                        }
                                        _ => None,
                                    }),
                                _ => {
                                    return Err(
                                        "to_tensor requires a resolved checked rank".to_string()
                                    );
                                }
                            };
                            extent
                                .map(|extent| {
                                    usize::try_from(extent).map_err(|_| {
                                        "to_tensor extent is not representable".to_string()
                                    })
                                })
                                .transpose()
                        })
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    vec![None; list_depth]
                };
                let (shape, data) =
                    nested_list_to_tensor_data(&values, precision, &expected_shape)?;
                let storage = data.into_storage("to_tensor", precision)?;
                Ok(RuntimeValue::Tensor(RuntimeTensorValue::new(
                    IrTensorValue::from_storage(shape, storage),
                )))
            }
            "to_list" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let values = tensor_to_list_values(&tensor)?;
                Ok(RuntimeValue::List(values.into()))
            }
            "pad_sequences" => {
                let sequences = expect_list_arg(args, 0)?;
                let pad = args
                    .get(1)
                    .cloned()
                    .ok_or_else(|| "pad_sequences expects 2 arguments".to_string())?;
                let (precision, data, batch, width) = pad_sequences_value(&sequences, &pad)?;
                let storage = data.into_storage("pad_sequences", precision)?;
                Ok(RuntimeValue::Tensor(RuntimeTensorValue::new(
                    IrTensorValue::from_storage(vec![batch, width], storage),
                )))
            }
            "pad_sequences_to" => {
                let sequences = expect_list_arg(args, 0)?;
                let width = expect_int_arg(args, 1)?;
                let pad = args
                    .get(2)
                    .cloned()
                    .ok_or_else(|| "pad_sequences_to expects 3 arguments".to_string())?;
                let (precision, data, batch) = pad_sequences_to_value(&sequences, width, &pad)?;
                let storage = data.into_storage("pad_sequences_to", precision)?;
                Ok(RuntimeValue::Tensor(RuntimeTensorValue::new(
                    IrTensorValue::from_storage(vec![batch, width.max(0) as usize], storage),
                )))
            }
            "read_file" => {
                let path = expect_string_arg(args, 0)?;
                let text = self.system.read_file(Path::new(&path))?;
                Ok(RuntimeValue::String(text))
            }
            "round_to" => {
                // [05-OP-1]: per-dtype at declared widths, f64 and f32
                // only, dispatched STRICTLY on the operand's own dtype --
                // no widen/round/re-narrow lane exists ([04-NUM-8] has no
                // exception vocabulary), and unsupported float widths fail
                // loudly here exactly as they do at check time.
                let places = expect_int_arg(args, 1)?;
                match args.first() {
                    Some(RuntimeValue::Scalar(payload)) if payload.dtype() == Prim::F64 => {
                        let rounded =
                            super::numeric_text::round_to_f64_impl(payload.as_f64_lossy(), places)?;
                        Ok(RuntimeValue::float64(rounded))
                    }
                    Some(RuntimeValue::Scalar(payload)) if payload.dtype() == Prim::F32 => {
                        let rounded = super::numeric_text::round_to_f32_impl(
                            payload.as_f64_lossy() as f32,
                            places,
                        )?;
                        RuntimeValue::scalar_like_float(Prim::F32, f64::from(rounded))
                    }
                    Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_float() => {
                        Err(format!(
                            "round_to: unsupported operand dtype {} ([05-OP-1] authors \
                             decimal rounding for f64 and f32 only; cast the operand \
                             explicitly)",
                            payload.dtype().name()
                        ))
                    }
                    other => Err(format!(
                        "expected float arg at index 0, got {}",
                        describe_argument(other)
                    )),
                }
            }
            // Host-lane CSV I/O (chelis#903) over the canonical
            // List[Dict[string,string]] text-table carrier.
            "parse_csv" => {
                let text = expect_string_arg(args, 0)?;
                super::csv::parse_csv_text(&text)
            }
            "to_csv" => {
                let value = args
                    .first()
                    .ok_or_else(|| "to_csv expects 1 argument".to_string())?;
                super::csv::csv_to_text(value).map(RuntimeValue::String)
            }
            "csv_f64s" => {
                let value = args
                    .first()
                    .ok_or_else(|| "csv_f64s expects 2 arguments".to_string())?;
                let column = expect_string_arg(args, 1)?;
                super::csv::csv_f64s_at(value, &column).map(|values| {
                    RuntimeValue::List(values.into_iter().map(RuntimeValue::float64).collect())
                })
            }
            "csv_ints" => {
                let value = args
                    .first()
                    .ok_or_else(|| "csv_ints expects 2 arguments".to_string())?;
                let column = expect_string_arg(args, 1)?;
                super::csv::csv_ints_at(value, &column).map(|values| {
                    RuntimeValue::List(values.into_iter().map(RuntimeValue::int64).collect())
                })
            }
            "csv_strs" => {
                let value = args
                    .first()
                    .ok_or_else(|| "csv_strs expects 2 arguments".to_string())?;
                let column = expect_string_arg(args, 1)?;
                super::csv::csv_strs_at(value, &column).map(|values| {
                    RuntimeValue::List(values.into_iter().map(RuntimeValue::String).collect())
                })
            }
            "csv_nrows" => {
                let value = args
                    .first()
                    .ok_or_else(|| "csv_nrows expects 1 argument".to_string())?;
                super::csv::csv_nrows_of(value).map(RuntimeValue::int64)
            }
            "csv_cols" => {
                let value = args
                    .first()
                    .ok_or_else(|| "csv_cols expects 1 argument".to_string())?;
                super::csv::csv_cols_of(value).map(|columns| {
                    RuntimeValue::List(columns.into_iter().map(RuntimeValue::String).collect())
                })
            }
            "csv_f64" => {
                let value = args
                    .first()
                    .ok_or_else(|| "csv_f64 expects 3 arguments".to_string())?;
                let row_idx = expect_int_arg(args, 1)?;
                let column = expect_string_arg(args, 2)?;
                super::csv::csv_f64_at(value, row_idx, &column).map(RuntimeValue::float64)
            }
            "csv_int" => {
                let value = args
                    .first()
                    .ok_or_else(|| "csv_int expects 3 arguments".to_string())?;
                let row_idx = expect_int_arg(args, 1)?;
                let column = expect_string_arg(args, 2)?;
                super::csv::csv_int_at(value, row_idx, &column).map(RuntimeValue::int64)
            }
            "csv_str" => {
                let value = args
                    .first()
                    .ok_or_else(|| "csv_str expects 3 arguments".to_string())?;
                let row_idx = expect_int_arg(args, 1)?;
                let column = expect_string_arg(args, 2)?;
                super::csv::csv_str_at(value, row_idx, &column).map(RuntimeValue::String)
            }
            // Hull Phase 0a: `process_run(cmd, args) -> (exit_code, stdout, stderr)`.
            //
            // Current eval/test implementation of subprocess exec. Arguments
            // pass straight to the OS as argv, without implicit shell, glob,
            // `$VAR`, or backtick interpolation. The C and HIP build backends
            // currently reject this builtin; compiled host parity remains
            // required by [05-HOST-2] and tracked by chelis#1297. This
            // evaluator boundary does not implement that separate lane.
            "process_run" => {
                let cmd = expect_string_arg(args, 0)?;
                let raw_args = expect_list_arg(args, 1)?;
                let mut argv = Vec::with_capacity(raw_args.len());
                for (index, value) in raw_args.iter().enumerate() {
                    match value {
                        RuntimeValue::String(text) => argv.push(text.clone()),
                        other => {
                            return Err(format!(
                                "process_run expects List[String] args, got {other:?} at index {index}"
                            ));
                        }
                    }
                }
                let output = self.system.run_process(&cmd, &argv)?;
                // A process killed by a signal has no exit code; report -1 so
                // callers can distinguish it from a clean exit 0.
                let exit_code = output.exit_status.map_or(-1_i64, i64::from);
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                Ok(RuntimeValue::Tuple(
                    vec![
                        RuntimeValue::int64(exit_code),
                        RuntimeValue::String(stdout),
                        RuntimeValue::String(stderr),
                    ]
                    .into(),
                ))
            }
            // [05-OP-75]: `(seconds, nanoseconds)` from one policy-checked
            // host reading. Compiled host execution is chelis#1297.
            "clock_wall_read" => Ok(clock_time_value(self.system.clock_wall_read()?)),
            "clock_monotonic_read" => Ok(clock_time_value(self.system.clock_monotonic_read()?)),
            "write_file" => {
                let path = expect_string_arg(args, 0)?;
                let contents = expect_string_arg(args, 1)?;
                self.system.write_file(Path::new(&path), &contents)?;
                Ok(RuntimeValue::Unit)
            }
            "read_lines" => {
                let path = expect_string_arg(args, 0)?;
                let text = self.system.read_lines_source(Path::new(&path))?;
                Ok(RuntimeValue::List(
                    text.lines()
                        .map(|line| RuntimeValue::String(line.to_string()))
                        .collect(),
                ))
            }
            "read_bytes" => {
                let path = expect_string_arg(args, 0)?;
                let bytes = self.system.read_bytes(Path::new(&path))?;
                Ok(RuntimeValue::List(
                    bytes
                        .into_iter()
                        .map(|byte| RuntimeValue::int64(i64::from(byte)))
                        .collect(),
                ))
            }
            "file_exists" => {
                let path = expect_string_arg(args, 0)?;
                Ok(RuntimeValue::Bool(
                    self.system.file_exists(Path::new(&path))?,
                ))
            }
            "list_dir" => {
                let path = expect_string_arg(args, 0)?;
                let names = self.system.list_dir(Path::new(&path))?;
                Ok(RuntimeValue::List(
                    names.into_iter().map(RuntimeValue::String).collect(),
                ))
            }
            "mmap_file" => {
                let path = expect_string_arg(args, 0)?;
                let bytes = self.system.load_mapped_file_bytes(Path::new(&path))?;
                Ok(RuntimeValue::MappedFile(bytes))
            }
            "mmap_read" => {
                let mapped = args
                    .first()
                    .ok_or_else(|| "mmap_read expects 3 arguments".to_string())?;
                let offset = expect_int_arg(args, 1)?;
                let len = expect_int_arg(args, 2)?;
                let RuntimeValue::MappedFile(bytes) = mapped else {
                    return Err(format!("mmap_read expects MappedFile, got {mapped:?}"));
                };
                if offset < 0 || len < 0 {
                    return Err("mmap_read requires non-negative offset and length".to_string());
                }
                let offset = offset as usize;
                let len = len as usize;
                if offset > bytes.len() {
                    return Err("mmap_read offset out of bounds".to_string());
                }
                let end = offset.saturating_add(len).min(bytes.len());
                Ok(RuntimeValue::List(
                    bytes[offset..end]
                        .iter()
                        .map(|byte| RuntimeValue::int64(i64::from(*byte)))
                        .collect(),
                ))
            }
            "mmap_len" => match args.first() {
                Some(RuntimeValue::MappedFile(bytes)) => {
                    Ok(RuntimeValue::int64(bytes.len() as i64))
                }
                other => Err(format!("mmap_len expects MappedFile, got {other:?}")),
            },
            "split" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                let sizes = expect_list_arg(args, 2)?;
                tensor_split_value(&tensor, axis, &sizes)
            }
            "gather" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let indices = expect_tensor_arg(args, 1)?;
                let axis = expect_int_arg(args, 2)?;
                tensor_gather_value(&tensor, &indices, axis).map(RuntimeValue::Tensor)
            }
            "scatter" => {
                let base = expect_tensor_arg(args, 0)?;
                let indices = expect_tensor_arg(args, 1)?;
                let updates = expect_tensor_arg(args, 2)?;
                let axis = expect_int_arg(args, 3)?;
                let mode = expect_string_arg(args, 4)?;
                tensor_scatter_value(&base, &indices, &updates, axis, &mode)
                    .map(RuntimeValue::Tensor)
            }
            // [05-OP-52]: `scatter_replace` is the replace-mode instance of
            // `scatter`.
            "scatter_replace" => {
                let base = expect_tensor_arg(args, 0)?;
                let indices = expect_tensor_arg(args, 1)?;
                let updates = expect_tensor_arg(args, 2)?;
                let axis = expect_int_arg(args, 3)?;
                tensor_scatter_value(&base, &indices, &updates, axis, "replace")
                    .map(RuntimeValue::Tensor)
            }
            "scatter_elements" => {
                let data = expect_tensor_arg(args, 0)?;
                let indices = expect_tensor_arg(args, 1)?;
                let updates = expect_tensor_arg(args, 2)?;
                let axis = expect_int_arg(args, 3)?;
                tensor_scatter_elements_value(&data, &indices, &updates, axis)
                    .map(RuntimeValue::Tensor)
            }
            "where" => {
                let cond = expect_tensor_arg(args, 0)?;
                let then_tensor = expect_tensor_arg(args, 1)?;
                let else_tensor = expect_tensor_arg(args, 2)?;
                tensor_where_value(&cond, &then_tensor, &else_tensor).map(RuntimeValue::Tensor)
            }
            "cumsum" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_cumsum_value(&tensor, axis).map(RuntimeValue::Tensor)
            }
            "sort" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_sort_value(&tensor, axis)
            }
            "diagonal" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis1 = expect_int_arg(args, 1)?;
                let axis2 = expect_int_arg(args, 2)?;
                tensor_diagonal_value(&tensor, axis1, axis2).map(RuntimeValue::Tensor)
            }
            "trace" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis1 = expect_int_arg(args, 1)?;
                let axis2 = expect_int_arg(args, 2)?;
                tensor_trace_value(&tensor, axis1, axis2).map(RuntimeValue::Tensor)
            }
            "clamp" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let lo = expect_tensor_arg(args, 1)?;
                let hi = expect_tensor_arg(args, 2)?;
                tensor_clamp_value(&tensor, &lo, &hi).map(RuntimeValue::Tensor)
            }
            "einsum" => {
                let equation = expect_string_arg(args, 0)?;
                let lhs = expect_tensor_arg(args, 1)?;
                let rhs = expect_tensor_arg(args, 2)?;
                tensor_einsum_value(&equation, &lhs, &rhs).map(RuntimeValue::Tensor)
            }
            "rank" => {
                let tensor = expect_tensor_arg(args, 0)?;
                Ok(RuntimeValue::int64(tensor.value.shape.len() as i64))
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
                // `shape` returns an extent-domain i64
                // (`spec/05-risc-primitives.md` [05-DIM-2]). The closed
                // kernel boundary rejects mixed widths, so this payload
                // must carry the dtype the checker assigns.
                RuntimeValue::scalar_like_int(Prim::Int64, dim as i64)
            }
            "numel" => {
                let tensor = expect_tensor_arg(args, 0)?;
                // Empty-product identity handles the scalar (shape `[]`) case
                // correctly: `[].iter().product::<usize>() == 1`. For rank-1+
                // tensors with any zero dimension the product is 0, which is
                // the genuine element count and must not be clamped. The
                // historical `.max(1)` clamp here was the root cause of
                // Runtime-EmptyTensorNumel-F1: `numel(to_tensor([]))`
                // returning 1 instead of 0.
                let numel = tensor.value.shape.iter().product::<usize>();
                Ok(RuntimeValue::int64(numel as i64))
            }
            "tensor_to_scalar" => {
                let tensor = expect_tensor_arg(args, 0)?;
                if !tensor.value.shape.is_empty() {
                    return Err("tensor_to_scalar expects a rank-0 tensor".to_string());
                }
                if tensor.value.is_empty() {
                    return Err("tensor_to_scalar expects a non-empty rank-0 tensor".to_string());
                }
                let element = tensor.value.storage().scalar_at(0);
                match element.as_bool_exact() {
                    Some(flag) => Ok(RuntimeValue::Bool(flag)),
                    None => Ok(RuntimeValue::from_scalar_value(element)),
                }
            }
            "scalar_to_tensor" => match args.first() {
                Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_integer() => {
                    Ok(RuntimeValue::Tensor(RuntimeTensorValue::from_wide_int(
                        "scalar_to_tensor",
                        payload.dtype(),
                        vec![],
                        vec![payload.as_i64()],
                    )?))
                }
                Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_float() => {
                    Ok(RuntimeValue::Tensor(RuntimeTensorValue::from_scalars(
                        payload.dtype(),
                        vec![],
                        &[payload.value()],
                    )))
                }
                Some(RuntimeValue::Bool(value)) => {
                    Ok(RuntimeValue::Tensor(RuntimeTensorValue::from_wide_int(
                        "scalar_to_tensor",
                        Prim::Bool,
                        vec![],
                        vec![if *value { 1 } else { 0 }],
                    )?))
                }
                // #381: a top-level scalar binding (e.g. `c = cast(1.1, f64)`)
                // captured by a def body is pre-evaluated through the DAG lane
                // and arrives here as an already rank-0 (0-d) tensor, not a
                // `Scalar`. `scalar_to_tensor` of a scalar produces a rank-0
                // tensor, so applying it to a rank-0 tensor is the identity;
                // accept it and pass the value through (preserving precision).
                // This matches the C backend, where the captured binding stays
                // a scalar C value and `scalar_to_tensor` materializes the same
                // rank-0 tensor, and the DAG lowering, where `scalar_to_tensor`
                // is a pass-through on its input node.
                Some(RuntimeValue::Tensor(tensor)) if tensor.value.shape.is_empty() => {
                    Ok(RuntimeValue::Tensor(tensor.clone()))
                }
                other => Err(format!(
                    "scalar_to_tensor expects a scalar or rank-0 tensor input, got {other:?}"
                )),
            },
            "print" => {
                let value = args
                    .first()
                    .ok_or_else(|| "print expects 1 argument".to_string())?;
                let line = render_value(value);
                if let Some(capture) = &self.transcript_capture {
                    capture.append(line.clone());
                }
                self.transcript.push(line);
                Ok(RuntimeValue::Unit)
            }
            "fail" => {
                let message = expect_string_arg(args, 0)?;
                Err(message)
            }
            "test_assert" => {
                let cond = expect_bool_arg(args, 0)?;
                let label = expect_string_arg(args, 1)?;
                if cond {
                    Ok(RuntimeValue::Unit)
                } else {
                    Err(format!("assert failed: {label}"))
                }
            }
            "test_assert_eq" => {
                let actual = args
                    .first()
                    .ok_or_else(|| "test_assert_eq expects 3 arguments".to_string())?;
                let expected = args
                    .get(1)
                    .ok_or_else(|| "test_assert_eq expects 3 arguments".to_string())?;
                let label = expect_string_arg(args, 2)?;
                if runtime_values_equal(actual, expected)? {
                    Ok(RuntimeValue::Unit)
                } else {
                    Err(format!(
                        "assert_eq ({label}): expected {}, got {}",
                        render_value(expected),
                        render_value(actual)
                    ))
                }
            }
            "test_assert_eq_tensor" => {
                let actual = expect_tensor_arg(args, 0)?;
                let expected = expect_tensor_arg(args, 1)?;
                let label = expect_string_arg(args, 2)?;
                if actual.precision != expected.precision
                    || actual.value.shape != expected.value.shape
                {
                    return Err(format!(
                        "assert_eq_tensor ({label}): expected tensor shape {:?} at {}, got {:?} at {}",
                        expected.value.shape,
                        expected.precision.name(),
                        actual.value.shape,
                        actual.precision.name()
                    ));
                }
                for index in 0..actual.value.storage().len() {
                    if actual.value.storage().scalar_at(index)
                        != expected.value.storage().scalar_at(index)
                    {
                        return Err(format!(
                            "assert_eq_tensor ({label}): first mismatch at row-major index {index}"
                        ));
                    }
                }
                Ok(RuntimeValue::Unit)
            }
            "test_assert_close_tensor" => {
                let actual = expect_tensor_arg(args, 0)?;
                let expected = expect_tensor_arg(args, 1)?;
                let tolerance = match args.get(2) {
                    Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_float() => *payload,
                    _ => {
                        return Err(format!(
                            "assert_close_tensor: expected float tolerance, got {}",
                            describe_argument(args.get(2))
                        ));
                    }
                };
                let label = expect_string_arg(args, 3)?;
                let tensor_prim = actual.value.prim();
                if tensor_prim != expected.value.prim() || tensor_prim != tolerance.dtype() {
                    return Err(format!(
                        "assert_close_tensor ({label}): actual, expected, and tolerance must have one common active float dtype"
                    ));
                }
                if !tensor_prim.is_float() {
                    return Err(format!(
                        "assert_close_tensor ({label}): tensor dtype {} is not an active float dtype",
                        tensor_prim.name()
                    ));
                }
                if actual.value.shape != expected.value.shape {
                    return Err(format!(
                        "assert_close_tensor ({label}): shape mismatch, expected {}, got {}",
                        render_shape(&expected.value.shape),
                        render_shape(&actual.value.shape)
                    ));
                }
                let tolerance_f64 = tolerance.as_f64_lossy();
                if !tolerance_f64.is_finite() || tolerance_f64 < 0.0 {
                    let rendered_tolerance = chelis_types::format_element(
                        tolerance.dtype(),
                        tolerance.value().element_ref(),
                    );
                    return Err(format!(
                        "assert_close_tensor ({label}): invalid tolerance {rendered_tolerance} (must be finite and non-negative)"
                    ));
                }
                if actual.value.len() != expected.value.len() {
                    return Err(format!(
                        "assert_close_tensor ({label}): length mismatch, expected {} elements, got {}",
                        expected.value.len(),
                        actual.value.len()
                    ));
                }

                let mismatch = match (
                    actual.value.storage().view(),
                    expected.value.storage().view(),
                ) {
                    (StorageView::F64(actual), StorageView::F64(expected)) => first_f64_mismatch(
                        actual.iter().copied(),
                        expected.iter().copied(),
                        tolerance_f64,
                    ),
                    (StorageView::F32(actual), StorageView::F32(expected)) => first_f32_mismatch(
                        actual.iter().copied(),
                        expected.iter().copied(),
                        tolerance_f64 as f32,
                    ),
                    (StorageView::F16(actual), StorageView::F16(expected)) => first_f32_mismatch(
                        actual.iter().map(|value| value.to_f32()),
                        expected.iter().map(|value| value.to_f32()),
                        tolerance_f64 as f32,
                    ),
                    (StorageView::Bf16(actual), StorageView::Bf16(expected)) => first_f32_mismatch(
                        actual.iter().map(|value| value.to_f32()),
                        expected.iter().map(|value| value.to_f32()),
                        tolerance_f64 as f32,
                    ),
                    _ => unreachable!("common active-float dtype check makes storage exhaustive"),
                };
                if let Some(index) = mismatch {
                    let actual = actual.value.storage().scalar_at(index);
                    let expected = expected.value.storage().scalar_at(index);
                    let nan_suffix = if float_element_is_nan(actual.element_ref())
                        || float_element_is_nan(expected.element_ref())
                    {
                        " (NaN is never close)"
                    } else {
                        ""
                    };
                    let rendered_actual =
                        chelis_types::format_element(tensor_prim, actual.element_ref());
                    let rendered_expected =
                        chelis_types::format_element(tensor_prim, expected.element_ref());
                    let rendered_tolerance =
                        chelis_types::format_element(tensor_prim, tolerance.value().element_ref());
                    return Err(format!(
                        "assert_close_tensor ({label}): at index {index} expected {rendered_expected}, got {rendered_actual}, tol {rendered_tolerance}{nan_suffix}"
                    ));
                }
                Ok(RuntimeValue::Unit)
            }
            "debug" => {
                let value = args
                    .first()
                    .ok_or_else(|| "debug expects 1 argument".to_string())?;
                let line = render_value(value);
                if let Some(capture) = &self.transcript_capture {
                    capture.append(line.clone());
                }
                self.transcript.push(line);
                Ok(value.clone())
            }
            "min_reduce" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Min).map(RuntimeValue::Tensor)
            }
            // Mirror of `min_reduce` for issue Chelis-Lang/chelis#185. The
            // typer's `tensor_reduce_to_out` signature gives both ops the
            // same shape; the IR evaluator's `RiscOp::MaxReduce` uses
            // `reduce(.., f64::NEG_INFINITY, f64::max)` which the
            // `ReduceOp::Max` variant of `tensor_reduce_host` mirrors.
            "max_reduce" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Max).map(RuntimeValue::Tensor)
            }
            "prod_reduce" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Prod).map(RuntimeValue::Tensor)
            }
            "argmax_reduce" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Argmax).map(RuntimeValue::Tensor)
            }
            "argmin_reduce" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Argmin).map(RuntimeValue::Tensor)
            }
            "sum" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Sum).map(RuntimeValue::Tensor)
            }
            "count" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let rank = tensor.value.shape.len();
                let mut axes = Vec::with_capacity(args.len().saturating_sub(1));
                for index in 1..args.len() {
                    let raw = expect_int_arg(args, index)?;
                    let axis = if raw < 0 {
                        rank.checked_sub(raw.unsigned_abs() as usize)
                    } else {
                        usize::try_from(raw).ok().filter(|&axis| axis < rank)
                    }
                    .ok_or_else(|| {
                        format!("count axis {raw} is out of bounds for rank {rank} tensor")
                    })?;
                    if axes.contains(&axis) {
                        return Err(format!("count has duplicate normalized axis {raw}"));
                    }
                    axes.push(axis);
                }
                axes.sort_unstable_by(|a, b| b.cmp(a));
                tensor_count_host(&tensor, &axes).map(RuntimeValue::Tensor)
            }
            "matmul" => {
                let lhs = expect_tensor_arg(args, 0)?;
                let rhs = expect_tensor_arg(args, 1)?;
                tensor_matmul_host(&lhs, &rhs).map(RuntimeValue::Tensor)
            }
            "permute" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let rank = tensor.value.shape.len();
                if args.len() != rank + 1 {
                    return Err(format!(
                        "permute expects {} axis arguments for rank-{rank} tensor, got {}",
                        rank,
                        args.len().saturating_sub(1)
                    ));
                }
                let mut axes: Vec<usize> = Vec::with_capacity(rank);
                for i in 0..rank {
                    let raw = expect_int_arg(args, i + 1)?;
                    if raw < 0 {
                        return Err(format!("permute requires non-negative axis, got {raw}"));
                    }
                    axes.push(raw as usize);
                }
                tensor_permute_host(&tensor, &axes).map(RuntimeValue::Tensor)
            }
            // The match scrutinee `name` carries the spelling the program
            // actually used, so this shared arm reports it without binding a
            // new alternative. A `callee @ (...)` binder would say the same
            // thing and would stop the host-runtime dispatch invariant, which
            // reads arm names off a leading `"`, from seeing an arm here at
            // all -- for either name.
            "expand" | "insert" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                let count = expect_int_arg(args, 2)?;
                if axis < 0 {
                    return Err(format!("{name} requires non-negative axis, got {axis}"));
                }
                if count < 0 {
                    // A runtime negative size traps `Domain` before allocation
                    // (spec/04-type-system.md section 4.7.2), rendered as the C
                    // runtime renders it: the metadata line, then [04-NUM-9]'s
                    // trap line with the extent dtype.
                    let trap = chelis_types::NumericTrap::Domain {
                        op: if name == "insert" { "insert" } else { "expand" },
                        prim: Prim::Int64,
                    };
                    let error = chelis_abi::metadata::MetadataError::Domain(
                        chelis_abi::metadata::EXPANSION_DOMAIN.into(),
                    );
                    return Err(format!("{error}\n{trap}"));
                }
                // One shape per operation (spec/04-type-system.md section
                // 4.7.2), so the name selects the evaluator rather than a
                // heuristic over the operand's shape.
                if name == "insert" {
                    tensor_insert_host(name, &tensor, axis as usize, count as usize)
                        .map(RuntimeValue::Tensor)
                } else {
                    tensor_expand_host(name, &tensor, axis as usize, count as usize)
                        .map(RuntimeValue::Tensor)
                }
            }
            "softmax" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_softmax_host(&tensor, axis).map(RuntimeValue::Tensor)
            }
            // Composed Tier-2 ops wired for issue Chelis-Lang/chelis#185.
            // Each delegates to the canonical IR decomposition in
            // `crates/chelis-ir/src/tier2.rs` (the same path the C
            // backend takes) and forward-evaluates the resulting small
            // DAG through `chelis_ir::eval` — the canonical numerical
            // oracle per `feedback_evaluator_byte_identical_gate`. We do
            // not reimplement the math here; that's the path that drifts
            // when downstream tier2 updates land.
            "mean" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                let axis = normalize_axis(tensor.value.shape.len(), axis, "mean")?;
                eval_composed_unary(&tensor, |dag, decl, x, ty| {
                    tier2::lower_mean(decl.into(), dag, x, axis, ty, None)
                })
                .map(RuntimeValue::Tensor)
            }
            "layer_norm" => {
                let x = expect_tensor_arg(args, 0)?;
                let gamma = expect_tensor_arg(args, 1)?;
                let beta = expect_tensor_arg(args, 2)?;
                let epsilon = match args.get(3) {
                    Some(RuntimeValue::Scalar(value)) if value.dtype() == x.precision => {
                        value.value()
                    }
                    Some(RuntimeValue::Tensor(value))
                        if value.precision == x.precision && value.value.shape.is_empty() =>
                    {
                        value.value.storage().scalar_at(0)
                    }
                    _ => {
                        return Err(
                            "layer_norm epsilon must be a scalar of the operand dtype".to_string()
                        );
                    }
                };
                eval_composed_triop(
                    &x,
                    &gamma,
                    &beta,
                    |dag, decl, x_id, gamma_id, beta_id, tys| {
                        let epsilon_id = dag.add_node(
                            decl,
                            RiscOp::Const { value: epsilon },
                            vec![],
                            TensorType {
                                dims: vec![],
                                precision: epsilon.prim(),
                            },
                            None,
                        );
                        tier2::lower_layer_norm(
                            decl.into(),
                            dag,
                            x_id,
                            gamma_id,
                            beta_id,
                            tys.0,
                            tys.1,
                            tys.2,
                            epsilon_id,
                            None,
                        )
                    },
                )
                .map(RuntimeValue::Tensor)
            }
            "conv" => {
                let input = expect_tensor_arg(args, 0)?;
                let kernel = expect_tensor_arg(args, 1)?;
                let raw_strides = expect_list_arg(args, 2)?;
                let raw_padding = expect_list_arg(args, 3)?;
                let strides = expect_int_list(&raw_strides, "conv")?;
                let padding = raw_padding
                    .iter()
                    .map(|value| {
                        let RuntimeValue::Tuple(pair) = value else {
                            return Err("conv padding requires (low,high) tuples".to_string());
                        };
                        if pair.len() != 2 {
                            return Err("conv padding requires two entries per pair".to_string());
                        }
                        let low = expect_int_arg(pair, 0)?;
                        let high = expect_int_arg(pair, 1)?;
                        Ok((
                            usize::try_from(low)
                                .map_err(|_| "conv padding must be non-negative")?,
                            usize::try_from(high)
                                .map_err(|_| "conv padding must be non-negative")?,
                        ))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                conv_host(&input, &kernel, &strides, &padding).map(RuntimeValue::Tensor)
            }
            // Movement primitives that take parameterized window args. Both
            // delegate to the same arithmetic the IR evaluator at
            // `crates/chelis-ir/src/eval.rs` uses, so eval-in-context output
            // is byte-identical to a freshly-lowered DAG run -- per the
            // evaluator-vs-backend agreement gate. Issue Chelis-Lang/chelis#187.
            "shrink" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let raw = expect_list_arg(args, 1)?;
                let bounds = extract_bounds_pair_list(&raw, "shrink")?;
                tensor_shrink_host(&tensor, &bounds).map(RuntimeValue::Tensor)
            }
            "reduce_window_max" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let window_raw = expect_list_arg(args, 1)?;
                let strides_raw = expect_list_arg(args, 2)?;
                let window = expect_int_list(&window_raw, "reduce_window_max")?;
                let strides = expect_int_list(&strides_raw, "reduce_window_max")?;
                tensor_reduce_window_host(
                    &tensor,
                    &window,
                    &strides,
                    ReduceWindowOp::Max,
                    "reduce_window_max",
                )
                .map(RuntimeValue::Tensor)
            }
            "reduce_window_min" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let window_raw = expect_list_arg(args, 1)?;
                let strides_raw = expect_list_arg(args, 2)?;
                let window = expect_int_list(&window_raw, "reduce_window_min")?;
                let strides = expect_int_list(&strides_raw, "reduce_window_min")?;
                tensor_reduce_window_host(
                    &tensor,
                    &window,
                    &strides,
                    ReduceWindowOp::Min,
                    "reduce_window_min",
                )
                .map(RuntimeValue::Tensor)
            }
            "reduce_window_sum" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let window_raw = expect_list_arg(args, 1)?;
                let strides_raw = expect_list_arg(args, 2)?;
                let window = expect_int_list(&window_raw, "reduce_window_sum")?;
                let strides = expect_int_list(&strides_raw, "reduce_window_sum")?;
                tensor_reduce_window_host(
                    &tensor,
                    &window,
                    &strides,
                    ReduceWindowOp::Sum,
                    "reduce_window_sum",
                )
                .map(RuntimeValue::Tensor)
            }
            "reduce_window_mean" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let window_raw = expect_list_arg(args, 1)?;
                let strides_raw = expect_list_arg(args, 2)?;
                let window = expect_int_list(&window_raw, "reduce_window_mean")?;
                let strides = expect_int_list(&strides_raw, "reduce_window_mean")?;
                tensor_reduce_window_host(
                    &tensor,
                    &window,
                    &strides,
                    ReduceWindowOp::Mean,
                    "reduce_window_mean",
                )
                .map(RuntimeValue::Tensor)
            }
            "pad" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let raw = expect_list_arg(args, 1)?;
                let padding = extract_bounds_pair_list(&raw, "pad")?;
                let fill = expect_float_arg(args, 2)?;
                tensor_pad_host(&tensor, &padding, fill).map(RuntimeValue::Tensor)
            }
            "stride" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let strides = expect_int_list(&args[1..], "stride")?;
                for (axis, step) in strides.iter().enumerate() {
                    if *step == 0 {
                        return Err(format!(
                            "stride axis {axis} step 0 is not allowed (must be positive)"
                        ));
                    }
                }
                tensor_stride_host(&tensor, &strides).map(RuntimeValue::Tensor)
            }
            // Activation primitives (Bucket 3).
            //
            // Scalar values are rank-0 numeric values under spec/05 §2.2.
            // Route both surfaces through the sealed dtype-keyed Tier-2
            // composition so every constituent primitive finalizes before
            // the next node observes it.
            "relu" => numeric_unop(args, None, Some(FloatUnOp::Relu)),
            "sigmoid" => numeric_unop(args, None, Some(FloatUnOp::Sigmoid)),
            "tanh" => numeric_unop(args, None, Some(FloatUnOp::Tanh)),
            "silu" => numeric_unop(args, None, Some(FloatUnOp::Silu)),
            "gelu" => numeric_unop(args, None, Some(FloatUnOp::Gelu)),
            other => Err(format!("unsupported builtin `{other}` in host runtime")),
        }
    }
}

enum DictStage {
    Find,
    AfterKey,
    AfterValue,
}

enum Step<'a> {
    Pair(&'a RuntimeValue, &'a RuntimeValue),
    Sequence {
        lhs: &'a [RuntimeValue],
        rhs: &'a [RuntimeValue],
        next: usize,
    },
    Dict {
        lhs: &'a [(RuntimeValue, RuntimeValue)],
        rhs: &'a [(RuntimeValue, RuntimeValue)],
        matched: Vec<bool>,
        lhs_index: usize,
        rhs_index: usize,
        stage: DictStage,
    },
}

/// Whether `args` is a pair `eq` and `neq` compare by [05-OP-36]'s recursive
/// rule rather than as scalars, strings or direct tensors.
fn is_recursive_equality_operand(args: &[RuntimeValue]) -> bool {
    let recursive = |value: &RuntimeValue| {
        matches!(
            value,
            RuntimeValue::Unit
                | RuntimeValue::List(_)
                | RuntimeValue::Tuple(_)
                | RuntimeValue::Dict(_)
                | RuntimeValue::Adt { .. }
        )
    };
    matches!(args, [lhs, rhs] if recursive(lhs) && recursive(rhs))
}

/// [05-OP-36]'s structural equality, for `eq`, `neq` and `test_assert_eq`.
/// Every nested container walk, including a dictionary candidate search, runs
/// from an explicit stack so value depth cannot consume the native stack
/// (chelis#2592).
pub(super) fn runtime_values_equal(lhs: &RuntimeValue, rhs: &RuntimeValue) -> Result<bool, String> {
    let mut steps = vec![Step::Pair(lhs, rhs)];
    // The result of the last child comparison is consumed by its parent
    // frame. A false dictionary candidate resumes the search; a false
    // sequence child ends that sequence immediately.
    let mut returned: Option<Result<bool, String>> = None;
    while let Some(step) = steps.pop() {
        match step {
            Step::Pair(lhs, rhs) => {
                returned = Some(match (lhs, rhs) {
                    (RuntimeValue::Scalar(lhs), RuntimeValue::Scalar(rhs)) => Ok(lhs == rhs),
                    (RuntimeValue::Bool(lhs), RuntimeValue::Bool(rhs)) => Ok(lhs == rhs),
                    (RuntimeValue::String(lhs), RuntimeValue::String(rhs)) => Ok(lhs == rhs),
                    (RuntimeValue::Unit, RuntimeValue::Unit) => Ok(true),
                    (RuntimeValue::List(lhs), RuntimeValue::List(rhs))
                    | (RuntimeValue::Tuple(lhs), RuntimeValue::Tuple(rhs)) => {
                        steps.push(Step::Sequence { lhs, rhs, next: 0 });
                        continue;
                    }
                    (RuntimeValue::Dict(lhs), RuntimeValue::Dict(rhs)) => {
                        if lhs.len() != rhs.len() {
                            Ok(false)
                        } else {
                            steps.push(Step::Dict {
                                lhs,
                                rhs,
                                matched: vec![false; rhs.len()],
                                lhs_index: 0,
                                rhs_index: 0,
                                stage: DictStage::Find,
                            });
                            continue;
                        }
                    }
                    (
                        RuntimeValue::Adt {
                            ctor: lhs_ctor,
                            fields: lhs_fields,
                            ..
                        },
                        RuntimeValue::Adt {
                            ctor: rhs_ctor,
                            fields: rhs_fields,
                            ..
                        },
                    ) => {
                        if lhs_ctor != rhs_ctor {
                            Ok(false)
                        } else {
                            steps.push(Step::Sequence {
                                lhs: lhs_fields,
                                rhs: rhs_fields,
                                next: 0,
                            });
                            continue;
                        }
                    }
                    (RuntimeValue::Tensor(lhs), RuntimeValue::Tensor(rhs)) => Ok(lhs.precision
                        == rhs.precision
                        && lhs.value.shape == rhs.value.shape
                        && (0..lhs.value.storage().len()).all(|index| {
                            lhs.value.storage().scalar_at(index)
                                == rhs.value.storage().scalar_at(index)
                        })),
                    (RuntimeValue::MappedFile(_), _)
                    | (_, RuntimeValue::MappedFile(_))
                    | (RuntimeValue::Closure { .. }, _)
                    | (_, RuntimeValue::Closure { .. })
                    | (RuntimeValue::Transform { .. }, _)
                    | (_, RuntimeValue::Transform { .. }) => Err(
                        "[05-OP-36] equality does not admit functions or resource handles"
                            .to_string(),
                    ),
                    _ => Ok(false),
                });
            }
            Step::Sequence { lhs, rhs, next } => {
                if let Some(result) = returned.take() {
                    match result {
                        Ok(true) => {}
                        other => {
                            returned = Some(other);
                            continue;
                        }
                    }
                }
                if lhs.len() != rhs.len() {
                    returned = Some(Ok(false));
                } else if next == lhs.len() {
                    returned = Some(Ok(true));
                } else {
                    steps.push(Step::Sequence {
                        lhs,
                        rhs,
                        next: next + 1,
                    });
                    steps.push(Step::Pair(&lhs[next], &rhs[next]));
                }
            }
            Step::Dict {
                lhs,
                rhs,
                mut matched,
                lhs_index,
                mut rhs_index,
                stage,
            } => match stage {
                DictStage::Find => {
                    if lhs_index == lhs.len() {
                        returned = Some(Ok(true));
                        continue;
                    }
                    while rhs_index < rhs.len() && matched[rhs_index] {
                        rhs_index += 1;
                    }
                    if rhs_index == rhs.len() {
                        returned = Some(Ok(false));
                        continue;
                    }
                    steps.push(Step::Dict {
                        lhs,
                        rhs,
                        matched,
                        lhs_index,
                        rhs_index,
                        stage: DictStage::AfterKey,
                    });
                    steps.push(Step::Pair(&lhs[lhs_index].0, &rhs[rhs_index].0));
                }
                DictStage::AfterKey => match returned.take().expect("key comparison returned") {
                    Ok(true) => {
                        steps.push(Step::Dict {
                            lhs,
                            rhs,
                            matched,
                            lhs_index,
                            rhs_index,
                            stage: DictStage::AfterValue,
                        });
                        steps.push(Step::Pair(&lhs[lhs_index].1, &rhs[rhs_index].1));
                    }
                    Ok(false) => steps.push(Step::Dict {
                        lhs,
                        rhs,
                        matched,
                        lhs_index,
                        rhs_index: rhs_index + 1,
                        stage: DictStage::Find,
                    }),
                    Err(error) => returned = Some(Err(error)),
                },
                DictStage::AfterValue => {
                    match returned.take().expect("value comparison returned") {
                        Ok(true) => {
                            matched[rhs_index] = true;
                            steps.push(Step::Dict {
                                lhs,
                                rhs,
                                matched,
                                lhs_index: lhs_index + 1,
                                rhs_index: 0,
                                stage: DictStage::Find,
                            });
                        }
                        Ok(false) => steps.push(Step::Dict {
                            lhs,
                            rhs,
                            matched,
                            lhs_index,
                            rhs_index: rhs_index + 1,
                            stage: DictStage::Find,
                        }),
                        Err(error) => returned = Some(Err(error)),
                    }
                }
            },
        }
    }
    returned.expect("the root comparison returns a result")
}

/// [05-OP-75]: a checked clock reading as its `(seconds, nanoseconds)` tuple.
fn clock_time_value(time: system::EvalClockTime) -> RuntimeValue {
    RuntimeValue::Tuple(
        vec![
            RuntimeValue::int64(time.seconds),
            RuntimeValue::int64(time.nanoseconds),
        ]
        .into(),
    )
}

/// [05-OP-33]: admit the complete result before deriving any keys.
fn admit_host_key_result(shape: &[usize]) -> Result<(), String> {
    use chelis_abi::metadata::{MetadataError, ShapeMetadata};
    let admitted = shape
        .iter()
        .map(|n| i64::try_from(*n).map_err(|_| MetadataError::Overflow("extent exceeds i64")))
        .collect::<Result<Vec<_>, _>>()
        .and_then(|dims| ShapeMetadata::contiguous(&dims, chelis_vocab::RuntimeDType::Key))
        .and_then(|metadata| metadata.bytes().allocation().map(|_| metadata));
    let metadata = admitted.map_err(|error| {
        let trap = match error {
            MetadataError::Domain(_) => chelis_types::NumericTrap::Domain {
                op: "split_keys",
                prim: Prim::Int64,
            },
            MetadataError::Overflow(_) => chelis_types::NumericTrap::Overflow {
                op: "split_keys",
                prim: Prim::Int64,
            },
        };
        format!("{error}\n{trap}")
    })?;
    metadata
        .elements()
        .scratch_len::<chelis_types::RandomKey>()
        .map_err(|_| "Domain: chelis_alloc tensor allocation failed".to_string())?;
    Ok(())
}

/// The scalar key operand of a key operation or draw: a key value, or the
/// rank-0 key tensor a kernel returns for one.
fn expect_key_arg(
    args: &[RuntimeValue],
    index: usize,
    op: &str,
) -> Result<chelis_types::RandomKey, String> {
    match args.get(index) {
        Some(RuntimeValue::Key(key)) => Ok(*key),
        Some(RuntimeValue::Tensor(tensor)) if tensor.value.shape.is_empty() => tensor
            .value
            .storage()
            .keys()
            .and_then(|keys| keys.first().copied())
            .ok_or_else(|| format!("{op} expects a key at index {index}")),
        other => Err(format!(
            "{op} expects a key at index {index}, got {}",
            describe_argument(other)
        )),
    }
}

/// The i64 seed or index operand of a key operation ([05-OP-69],
/// [05-OP-71], [05-OP-72]).
fn expect_i64_key_operand(
    args: &[RuntimeValue],
    index: usize,
    op: &str,
) -> Result<chelis_types::ScalarValue, String> {
    match args.get(index) {
        Some(RuntimeValue::Scalar(payload)) if payload.dtype() == Prim::Int64 => {
            Ok(payload.value())
        }
        Some(RuntimeValue::Tensor(tensor))
            if tensor.value.shape.is_empty() && tensor.value.prim() == Prim::Int64 =>
        {
            Ok(tensor.value.storage().scalar_at(0))
        }
        other => Err(format!(
            "{op} expects an i64 at index {index}, got {}",
            describe_argument(other)
        )),
    }
}

/// [05-OP-71]: a literal extent the checked result type declares for the
/// count axis is a claim about the runtime count, checked before any key
/// exists, as the DAG evaluator and the C lane check it.
fn check_split_keys_declared_extent(
    result_type_expr: Option<&Expr>,
    count: usize,
) -> Result<(), String> {
    let literal_extent = |dim: &Expr| match dim.carrier() {
        ExprCarrier::Atom(Atom::Int(extent)) => Some(*extent),
        ExprCarrier::DecodedNode(DeepTag::Lit, _, [Expr::Atom(Atom::Int(extent), _), ..]) => {
            Some(*extent)
        }
        _ => None,
    };
    let Some(declared) = result_type_expr.and_then(|ty| match ty.carrier() {
        ExprCarrier::DecodedNode(DeepTag::TTensor, _, parts) => {
            parts.iter().rev().nth(1).and_then(literal_extent)
        }
        _ => None,
    }) else {
        return Ok(());
    };
    if usize::try_from(declared).ok() == Some(count) {
        Ok(())
    } else {
        Err(format!(
            "split_keys: the count {count} does not equal the extent {declared} its type declares"
        ))
    }
}

/// One evaluated argument as the kernel `Load` its declared parameter names.
/// A tensor finalizes at the declared element dtype, the same ingress the
/// interpreter applies to its own frame (chelis#729); a scalar becomes an
/// exact rank-0 tensor at the declared prim, the way `scalar_to_tensor` builds
/// one and the way the C wrapper boxes a scalar parameter.
fn stage_kernel_argument(
    def: &str,
    param: &str,
    value: &RuntimeValue,
    prim: Prim,
) -> Result<IrTensorValue, String> {
    match value {
        // A key tensor enters a kernel as it is: a key has no conversion.
        RuntimeValue::Tensor(tensor) if tensor.value.prim() == Prim::Key && prim == Prim::Key => {
            Ok(tensor.value.clone())
        }
        RuntimeValue::Tensor(tensor) => Ok(ingress_tensor_to_declared(tensor.clone(), prim)?.value),
        // A scalar key enters a kernel as the rank-0 key tensor its key
        // `Load` reads (spec/10 section 3.2).
        RuntimeValue::Key(key) if prim == Prim::Key => Ok(IrTensorValue::from_storage(
            vec![],
            chelis_types::TensorStorage::from_keys(vec![*key]),
        )),
        RuntimeValue::Scalar(payload) if payload.dtype().is_integer() => {
            Ok(RuntimeTensorValue::from_wide_int(
                "kernel argument",
                prim,
                vec![],
                vec![payload.as_i64()],
            )?
            .value)
        }
        RuntimeValue::Scalar(payload) if payload.dtype() == prim => {
            Ok(RuntimeTensorValue::from_scalars(prim, vec![], &[payload.value()]).value)
        }
        RuntimeValue::Scalar(payload) => Ok(RuntimeTensorValue::from_wide(
            "kernel argument",
            prim,
            vec![],
            vec![payload.as_f64_lossy()],
        )?
        .value),
        RuntimeValue::Bool(flag) => Ok(RuntimeTensorValue::from_wide_int(
            "kernel argument",
            prim,
            vec![],
            vec![i64::from(*flag)],
        )?
        .value),
        // Rendered through the runtime's one diagnostic renderer, never a
        // Debug format (faithful_observation.md B2.4).
        other => Err(format!(
            "kernel `{def}` parameter `{param}` expects a tensor or scalar argument, got {}",
            describe_value(other)
        )),
    }
}

#[cfg(test)]
mod tensor_entry_actualization_tests {
    use super::*;

    fn type_expr(source: &str) -> Expr {
        chelis_deep::parser::parse_str(source)
            .expect("type expression parses")
            .into_iter()
            .next()
            .expect("one type expression")
    }

    fn wrapped(expr: Expr) -> Expr {
        let span = expr.span();
        Expr::MetaExpr(
            chelis_deep::MetaExpr {
                metadata: Metadata::default(),
                expr: Box::new(expr),
            },
            span,
        )
    }

    fn tensor(shape: Vec<usize>, values: Vec<f64>) -> RuntimeValue {
        RuntimeValue::Tensor(
            RuntimeTensorValue::from_wide("entry actualization test", Prim::F32, shape, values)
                .expect("test tensor finalizes"),
        )
    }

    #[test]
    fn tensor_entry_actualization_preserves_wrapped_rank_zero_position_between_scalars() {
        let tensor_formal = wrapped(type_expr("(t-tensor {} (t-prim {} f32))"));
        let checked = vec![
            type_expr("(t-prim {} f64)"),
            tensor_formal.clone(),
            type_expr("(t-prim {} i64)"),
        ];
        let authored = checked.iter().cloned().map(Some).collect::<Vec<_>>();
        let names = ["arg0".into(), "arg1".into(), "arg2".into()];
        let actualized = actualize_tensor_entry_parameters(
            Some(&checked),
            &authored,
            &[
                RuntimeValue::float64(1.0),
                tensor(Vec::new(), vec![2.0]),
                RuntimeValue::int64(3),
            ],
            &names,
            None,
            None,
        )
        .expect("scalar neighbors do not contaminate the rank-zero tensor formal");
        assert_eq!(
            actualized,
            vec![(
                "arg1".into(),
                TensorType {
                    dims: Vec::new(),
                    precision: Prim::F32,
                },
                Vec::new(),
            )]
        );

        let error = actualize_tensor_entry_parameters(
            Some(&checked),
            &authored,
            &[
                RuntimeValue::float64(1.0),
                tensor(vec![1], vec![2.0]),
                RuntimeValue::int64(3),
            ],
            &names,
            None,
            None,
        )
        .expect_err("the rank mismatch remains owned by the tensor at position one");
        assert_eq!(error, "input `arg1` expected rank 0, got 1");
    }
}

/// Capture-initialization order in the host interpreter: a captured library
/// declaration initializes lazily, at most once per context, only when an
/// activation that runs reads it, and a failed initialization binds nothing
/// and retries. Before chelis#2413 these cases read the interpreter's random
/// counter to prove which initializers ran. Draws now take explicit keys and
/// leave no counter, so the witness is the ordered transcript: every
/// initializer prints, and the transcript records whether, how often and in
/// which order each ran. Each draw is keyed by `key_from_seed`, and its
/// expected bits are [05-RNG-2]'s reference, transcribed below.
#[cfg(test)]
mod legacy_capture_order_tests {
    use super::*;

    /// The initializer's own draw, keyed by `key_from_seed(17i64)`.
    const DRAW: &str =
        "uniform_like(key_from_seed(17i64), to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32)";
    const LIVE: &str = "add(x, uniform_like(k, weights, 0.0f32, 1.0f32))";
    /// The seeds of the keys the test driver passes to `sample` and to
    /// `next_draw`.
    const SAMPLE_SEED: i64 = 42;
    const NEXT_SEED: i64 = 43;

    fn checked_library(source: &str) -> crate::pipeline::CheckedLibrary {
        let prepared =
            crate::pipeline::prepare_source(crate::schema::SourceKind::Surf, source, None)
                .expect("source fixture parses and expands");
        crate::pipeline::check_prepared_library(prepared)
            .expect("real library fixture passes type, effect and linearity checks")
    }

    /// The captured `weights` initializer is an ascribed block. Unascribed,
    /// a block-bodied declaration that binds `_ = print(...)` is typed rank
    /// zero inside a capturing definition (a defect that predates the key
    /// switch; the retired `with seed` wrapper hid it), and the kernel's
    /// capture rank check then refuses the rank-one value.
    fn library(initializer: &str, params: &str, body: &str) -> crate::pipeline::CheckedLibrary {
        let source = format!(
            "weights: tensor[2, f32] = {{ _ = print(\"initialize\")\n {initializer} }}\n\
             def sample({params}) -> tensor[2, f32] = {body}\n\
             def next_draw(k: key, x: tensor[2, f32]) -> tensor[2, f32] = uniform_like(k, x, 0.0f32, 1.0f32)\n"
        );
        checked_library(&source)
    }

    fn context<'a>(
        library: &'a crate::pipeline::CheckedLibrary,
        tensors: &'a UnordMap<String, RuntimeTensorValue>,
    ) -> EvalContext<'a> {
        let checked = library.program();
        let mut definitions = UnordMap::new();
        let mut eager = Vec::new();
        // The production library registration path omits eager initialization.
        register_top_level_defs(
            checked.exprs(),
            &BTreeMap::new(),
            None,
            &mut definitions,
            &mut eager,
            false,
        );
        assert!(eager.is_empty());
        let mut signatures = UnordMap::new();
        register_declared_signatures(checked.exprs(), &mut signatures);
        EvalContext {
            bindings: Frame::new(),
            result_producer: None,
            binding_types: UnordMap::new(),
            precision_bindings: UnordMap::new(),
            declaration_values: UnordMap::new(),
            named_axis_route_cache: UnordMap::new(),
            named_axis_route_visiting: UnordSet::new(),
            program: ProgramScope::new(
                definitions,
                checked
                    .type_env()
                    .iter()
                    .map(|(name, ty)| (name.clone(), ty.clone()))
                    .collect(),
            ),
            declared_signatures: signatures,
            adt_fields: collect_adt_ctor_fields(checked.exprs()),
            constructor_names: collect_constructor_source_names(checked.exprs()),
            adt_registry: checked.adt_registry().clone(),
            tensor_bindings: tensors,
            session: Some(chelis_ir::host::HostLoweringSession::new(checked)),
            active_declaration_names: Vec::new(),
            def_kernels: UnordMap::new(),
            transcript: Vec::new(),
            transcript_capture: None,
            resolving_top_levels: Vec::new(),
            cancel: None,
            system: system::EvalSystemBoundary::permissive(),
            failure_kind: RuntimeFailureKind::Ordinary,
            activation_extents: Default::default(),
        }
    }

    fn zeros() -> RuntimeValue {
        RuntimeValue::Tensor(RuntimeTensorValue::new(IrTensorValue::from_storage(
            vec![2],
            chelis_types::dtype_semantics::finalize_tensor(
                "test",
                Prim::F32,
                chelis_types::dtype_semantics::RawTensor::Float(vec![0.0, 0.0]),
            )
            .unwrap(),
        )))
    }

    fn bits(value: &RuntimeValue) -> serde_json::Value {
        serde_json::to_value(runtime_value_to_schema(value).unwrap()).unwrap()
    }

    /// `key_from_seed(seed)` as the interpreter's key value.
    fn key(seed: i64) -> RuntimeValue {
        RuntimeValue::Key(
            chelis_types::RandomKey::from_seed(
                chelis_types::scalar_from_i64("test", Prim::Int64, seed).unwrap(),
            )
            .unwrap(),
        )
    }

    /// [05-RNG-2]'s unit value of `word(key_from_seed(seed), index)`,
    /// transcribed from `key_ref.py`; never the evaluator's kernel.
    fn reference_unit(seed: i64, index: u64) -> f64 {
        fn splitmix64(x: u64) -> u64 {
            let x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        }
        let word = splitmix64(seed as u64 ^ splitmix64(index).rotate_left(41));
        (word >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// `uniform_like(key_from_seed(seed), zeros, 0.0f32, 1.0f32)` over two
    /// elements: [05-OP-8] at f32 is the unit rounded to f32.
    fn expected_draw(seed: i64) -> RuntimeValue {
        let values = (0..2)
            .map(|index| f64::from(reference_unit(seed, index) as f32))
            .collect();
        RuntimeValue::Tensor(
            RuntimeTensorValue::from_wide("test", Prim::F32, vec![2], values).unwrap(),
        )
    }

    /// The reference agrees with `key_ref.py`'s `unit` rounded to f32.
    #[test]
    fn capture_order_reference_draws_match_key_ref_py() {
        for (seed, words) in [
            (17, ["3f250d3b", "3f7aee67"]),
            (42, ["3efa06fe", "3e762d86"]),
            (43, ["3f208606", "3f514c88"]),
        ] {
            assert_eq!(
                bits(&expected_draw(seed)),
                serde_json::json!({"type":"tensor","value":{"shape":[2],"data":{"dtype":"f32","bits":words}}})
            );
        }
    }

    /// Call the checked named helper `name`, a kernel with the parameters
    /// `(k: key, x: tensor[2, f32])` whose one draw is keyed by `k`, with
    /// `key_from_seed(seed)` and zeros.
    fn admitted_call(
        ctx: &mut EvalContext<'_>,
        name: &str,
        seed: i64,
    ) -> Result<RuntimeValue, String> {
        let closure = ctx.resolve_top_level(name)?;
        let RuntimeValue::Closure {
            def_name, params, ..
        } = &closure
        else {
            panic!("checked named helper is a closure")
        };
        assert_eq!(def_name.as_deref(), Some(name));
        assert_eq!(params.len(), 2);
        let before = ctx.transcript.clone();
        let kernel = ctx
            .def_kernel(name)?
            .expect("baseline admits real helper kernel");
        let graph = kernel.as_ref();
        assert!(graph.staged.is_none());
        let draws: Vec<_> = graph
            .dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::UniformLike))
            .collect();
        assert_eq!(draws.len(), 1);
        assert_eq!(
            draws[0].inputs.len(),
            4,
            "key-operand UniformLike with no path activation"
        );
        let key_node = graph.dag.get(draws[0].inputs[3]).expect("key operand");
        assert!(
            matches!(&key_node.op, RiscOp::Load { name } if name == "k")
                && key_node.output_type.precision == Prim::Key,
            "the helper draws with the key its caller passes"
        );
        assert_eq!(
            ctx.transcript, before,
            "lowering does not initialize captures"
        );
        // This named closure dispatches through def_kernel then apply_def_kernel.
        ctx.apply_resolved_callable(closure, vec![key(seed), zeros()])
    }

    #[test]
    fn legacy_capture_order_preinitialized_admission_control() {
        let library = library(DRAW, "k: key, x: tensor[2, f32]", LIVE);
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        assert!(!ctx.bindings.contains_key("weights"));
        let initialized = ctx.resolve_top_level("weights").unwrap();
        assert_eq!(bits(&initialized), bits(&expected_draw(17)));
        assert_eq!(ctx.transcript, ["initialize"]);
        let actual = admitted_call(&mut ctx, "sample", SAMPLE_SEED).unwrap();
        let next = admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap();
        assert_eq!(bits(&actual), bits(&expected_draw(SAMPLE_SEED)));
        assert_eq!(bits(&next), bits(&expected_draw(NEXT_SEED)));
        // The preinitialized capture is served, never initialized again.
        assert_eq!(ctx.transcript, ["initialize"]);
    }

    // #1956: unlike a concrete-rank helper, this signature cannot take a
    // standalone def kernel. The ordinary checked call must enter site B.
    #[test]
    fn declaration_spread_formal_uses_actual_named_axis_route() {
        let draws = (0..5)
            .map(|j| {
                format!(
                    "_ = uniform_like(fold_in(key_from_seed(41i64), {j}i64), copy(x), 0.0f32, 1.0f32)\n"
                )
            })
            .collect::<String>();
        let library = checked_library(&format!(
            "weights: tensor[2, f32] = {{ _ = print(\"initialize\")\n to_tensor([3.0f32, 5.0f32]) }}\ndef total[pre, post](weights: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(weights, seq)\ndef main(x: tensor[seq, f32]) = {{ _ = print(\"entry\")\n {draws} result = total(x)\n (result, uniform_like(key_from_seed(42i64), x, 0.0f32, 1.0f32)) }}\ndef next_draw(k: key, x: tensor[2, f32]) -> tensor[2, f32] = uniform_like(k, x, 0.0f32, 1.0f32)\n"
        ));
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        assert!(ctx.def_kernel("total").unwrap().is_none());
        assert!(ctx.def_kernel("main").unwrap().is_none());
        assert!(!ctx.named_axis_route_cache.contains_key("total"));
        let main = ctx.resolve_top_level("main").unwrap();
        let input = RuntimeValue::Tensor(
            RuntimeTensorValue::from_wide("test", Prim::F32, vec![2], vec![7.0, 11.0]).unwrap(),
        );
        let value = ctx.apply_resolved_callable(main, vec![input]).unwrap();
        assert_eq!(ctx.named_axis_route_cache.get("total"), Some(&true));
        let RuntimeValue::Tuple(values) = value else {
            panic!("checked main returns two tensors")
        };
        assert_eq!(
            bits(&values[0]),
            serde_json::json!({"type":"tensor","value":{"shape":[],"data":{"dtype":"f32","bits":["41900000"]}}})
        );
        assert_eq!(bits(&values[1]), bits(&expected_draw(42)));
        assert_eq!(
            bits(&admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap()),
            bits(&expected_draw(NEXT_SEED))
        );
        // The formal shadows the declaration, which never initializes.
        assert_eq!(ctx.transcript, ["entry"]);
    }

    /// The named-axis route reads a block-bodied declaration as rank zero even
    /// when it is ascribed (the defect `library` describes), so these cases
    /// initialize `weights` through a printing nullary helper instead.
    fn named_axis_capture_case(
        initializer: &str,
        body: &str,
        expected: Result<&str, &str>,
        transcript: &[&str],
    ) {
        let draws = (0..5)
            .map(|j| {
                format!(
                    "_ = uniform_like(fold_in(key_from_seed(41i64), {j}i64), copy(x), 0.0f32, 1.0f32)\n"
                )
            })
            .collect::<String>();
        let library = checked_library(&format!(
            "baseline = to_tensor([7.0f32, 11.0f32])\ndef initialize() -> tensor[2, f32] = {{ _ = print(\"initialize\")\n {initializer} }}\nweights = initialize()\n\
             def total[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> (tensor[..pre, ..post, f32], tensor[f32]) = {body}\n\
             def main(x: tensor[seq, f32]) = {{ _ = print(\"entry\")\n {draws} result = total(x)\n (result, uniform_like(key_from_seed(42i64), x, 0.0f32, 1.0f32)) }}\n\
             def next_draw(k: key, x: tensor[2, f32]) -> tensor[2, f32] = uniform_like(k, x, 0.0f32, 1.0f32)\n"
        ));
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        assert!(ctx.def_kernel("total").unwrap().is_none());
        assert!(ctx.def_kernel("main").unwrap().is_none());
        assert!(!ctx.named_axis_route_cache.contains_key("total"));
        let main = ctx.resolve_top_level("main").unwrap();
        let input = RuntimeValue::Tensor(
            RuntimeTensorValue::from_wide("test", Prim::F32, vec![2], vec![7.0, 11.0]).unwrap(),
        );
        let result = ctx.apply_resolved_callable(main, vec![input]);
        assert_eq!(ctx.named_axis_route_cache.get("total"), Some(&true));
        let next = admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap();
        assert_eq!(bits(&next), bits(&expected_draw(NEXT_SEED)));
        match expected {
            Ok(expected_bits) => {
                let RuntimeValue::Tuple(values) = result.unwrap() else {
                    panic!("checked main returns two tensors")
                };
                let RuntimeValue::Tuple(captures) = &values[0] else {
                    panic!("routed total returns its input and capture reductions")
                };
                assert_eq!(
                    bits(&captures[0]),
                    serde_json::json!({"type":"tensor","value":{"shape":[],"data":{"dtype":"f32","bits":["41900000"]}}})
                );
                assert_eq!(
                    bits(&captures[1]),
                    serde_json::json!({"type":"tensor","value":{"shape":[],"data":{"dtype":"f32","bits":[expected_bits]}}})
                );
                assert_eq!(bits(&values[1]), bits(&expected_draw(42)));
            }
            Err(error) => {
                assert_eq!(result.unwrap_err(), error);
                assert!(!ctx.bindings.contains_key("weights"));
            }
        }
        // The later helper call never initializes the capture again.
        assert_eq!(ctx.transcript, transcript);
    }

    #[test]
    fn named_axis_selected_capture_keeps_initializer_and_next_draw() {
        named_axis_capture_case(
            "to_tensor([3.0f32, 5.0f32])",
            "(sum(x, seq), sum(weights, 0i32))",
            Ok("41000000"),
            &["entry", "initialize"],
        );
    }

    #[test]
    fn named_axis_caller_shadow_does_not_replace_a_declaration_capture() {
        let library = checked_library(
            "def initialize() -> tensor[2, f32] = { _ = print(\"initialize\")\n to_tensor([3.0f32, 5.0f32]) }\nweights = initialize()\ndef total[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> (tensor[..pre, ..post, f32], tensor[f32]) = (sum(x, seq), sum(weights, 0i32))\ndef main(weights: tensor[seq, f32]) = total(weights)\n",
        );
        let tensors = UnordMap::new();
        for warm in [false, true] {
            let mut ctx = context(&library, &tensors);
            assert!(ctx.def_kernel("total").unwrap().is_none());
            assert!(ctx.def_kernel("main").unwrap().is_none());
            if warm {
                ctx.resolve_top_level("total").unwrap();
                ctx.resolve_top_level("weights").unwrap();
            }
            let main = ctx.resolve_top_level("main").unwrap();
            let input = RuntimeValue::Tensor(
                RuntimeTensorValue::from_wide("test", Prim::F32, vec![2], vec![7.0, 11.0]).unwrap(),
            );
            let RuntimeValue::Tuple(values) =
                ctx.apply_resolved_callable(main, vec![input]).unwrap()
            else {
                panic!("checked call returns two reductions")
            };
            assert_eq!(ctx.named_axis_route_cache.get("total"), Some(&true));
            assert_eq!(values.len(), 2);
            for (value, expected) in values.iter().zip(["41900000", "41000000"]) {
                assert_eq!(
                    bits(value),
                    serde_json::json!({"type":"tensor", "value":{"shape":[], "data":{"dtype":"f32", "bits":[expected]}}})
                );
            }
            assert_eq!(ctx.transcript, ["initialize"]);
            assert!(ctx.bindings.is_empty());
        }
    }

    #[test]
    fn named_axis_dead_capture_does_not_initialize() {
        named_axis_capture_case(
            "to_tensor([3.0f32, 5.0f32])",
            "(sum(x, seq), if true then sum(baseline, 0i32) else sum(weights, 0i32))",
            Ok("41900000"),
            &["entry"],
        );
    }

    #[test]
    fn named_axis_capture_error_preserves_original_trap() {
        named_axis_capture_case(
            &format!("_ = {DRAW}\n to_tensor([cast(floor_div(1i32, 0i32), f32), 0.0f32])"),
            "(sum(x, seq), sum(weights, 0i32))",
            Err("numeric trap: division by zero in floor_div at i32"),
            &["entry", "initialize"],
        );
    }

    #[test]
    fn named_axis_available_shape_never_enters_failing_initializer() {
        use chelis_ir::eval::TensorInputDemand::{AvailableShape, RequiredShape, Selected};
        let initializer =
            format!("_ = {DRAW}\n to_tensor([cast(floor_div(1i32, 0i32), f32), 0.0f32])");
        let library = library(&initializer, "k: key, x: tensor[2, f32]", LIVE);
        let tensors = UnordMap::new();
        let staged = UnordMap::new();
        for required in [RequiredShape, Selected] {
            let mut ctx = context(&library, &tensors);
            assert!(
                ctx.prepare_named_axis_input("weights", AvailableShape, &staged)
                    .unwrap()
                    .is_none()
            );
            assert!(ctx.transcript.is_empty());
            assert!(!ctx.bindings.contains_key("weights"));
            assert_eq!(
                ctx.prepare_named_axis_input("weights", required, &staged)
                    .unwrap_err(),
                "numeric trap: division by zero in floor_div at i32"
            );
            assert_eq!(ctx.transcript, ["initialize"]);
            assert!(!ctx.bindings.contains_key("weights"));
            assert!(
                ctx.prepare_named_axis_input("weights", AvailableShape, &staged)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(
                bits(&admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap()),
                bits(&expected_draw(NEXT_SEED))
            );
            assert_eq!(ctx.transcript, ["initialize"]);
        }
    }

    #[test]
    fn named_axis_input_roles_preserve_available_tensor_precedence() {
        use chelis_ir::eval::TensorInputDemand::{AvailableShape, RequiredShape, Selected};
        let library = library(DRAW, "k: key, x: tensor[2, f32]", LIVE);
        // Three distinct tensors, one per source the roles choose between.
        let tensor = |level: usize| {
            let RuntimeValue::Tensor(value) = expected_draw(level as i64) else {
                panic!("draw is a tensor")
            };
            value
        };
        for role in [AvailableShape, RequiredShape, Selected] {
            for level in 0..3 {
                let mut tensors = UnordMap::new();
                let mut staged = UnordMap::new();
                if level >= 1 {
                    tensors.insert("weights".to_owned(), tensor(1));
                }
                if level == 2 {
                    staged.insert("weights".to_owned(), tensor(2).value);
                }
                let mut ctx = context(&library, &tensors);
                ctx.bindings
                    .insert("weights".to_owned(), RuntimeValue::Tensor(tensor(0)));
                let actual = ctx
                    .prepare_named_axis_input("weights", role, &staged)
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    bits(&RuntimeValue::Tensor(RuntimeTensorValue::new(actual))),
                    bits(&expected_draw(level as i64))
                );
                // A supplied value serves the role; the initializer never runs.
                assert!(ctx.transcript.is_empty());
            }
        }
    }

    #[test]
    fn named_axis_input_roles_do_not_observe_callables_or_unknown_names() {
        use chelis_ir::eval::TensorInputDemand::{AvailableShape, RequiredShape, Selected};
        let library = checked_library(&format!(
            "def sample() -> tensor[2, f32] = {{ _ = print(\"observe\")\n {DRAW} }}\nalias = sample\n"
        ));
        let tensors = UnordMap::new();
        let staged = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        for role in [AvailableShape, RequiredShape, Selected] {
            for name in ["sample", "alias", "unknown"] {
                assert!(
                    ctx.prepare_named_axis_input(name, role, &staged)
                        .unwrap()
                        .is_none()
                );
                assert!(!ctx.bindings.contains_key(name));
            }
        }
        assert!(ctx.transcript.is_empty());
    }

    #[test]
    fn named_axis_required_shape_initializes_once_then_available_can_serve() {
        use chelis_ir::eval::TensorInputDemand::{AvailableShape, RequiredShape};
        let library = library(DRAW, "k: key, x: tensor[2, f32]", LIVE);
        let tensors = UnordMap::new();
        let staged = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        for role in [RequiredShape, AvailableShape] {
            let actual = ctx
                .prepare_named_axis_input("weights", role, &staged)
                .unwrap()
                .unwrap();
            assert_eq!(
                bits(&RuntimeValue::Tensor(RuntimeTensorValue::new(actual))),
                bits(&expected_draw(17))
            );
            assert_eq!(ctx.transcript, ["initialize"]);
        }
    }

    #[test]
    fn declaration_cold_warm_builtin_and_local_shadow_control() {
        // Top-level builtin-name declarations are rejected by the checker;
        // warm a legal declaration, never manufacture a cached builtin.
        let library = checked_library(
            "marker = 7\nplain = relu(-2.0f64)\nshadowed = { relu = fn (v: f64) -> add(v, 100.0f64)\n relu(-2.0f64) }\n",
        );
        let tensors = UnordMap::new();
        for warm in [false, true] {
            let mut ctx = context(&library, &tensors);
            if warm {
                ctx.resolve_top_level("marker").unwrap();
            }
            let plain = ctx.lookup_top_level_def("plain").unwrap().1;
            let shadowed = ctx.lookup_top_level_def("shadowed").unwrap().1;
            assert!(ctx.active_builtin_symbol("relu"));
            let zero = bits(&ctx.eval_expr(&plain).unwrap());
            let local = bits(&ctx.eval_expr(&shadowed).unwrap());
            assert_eq!(zero, bits(&RuntimeValue::float64(0.0)));
            assert_eq!(local, bits(&RuntimeValue::float64(98.0)));
            assert!(ctx.active_builtin_symbol("relu"));
            assert_eq!(bits(&ctx.eval_expr(&plain).unwrap()), zero);
            assert!(ctx.transcript.is_empty());
        }
    }

    #[test]
    fn declaration_cold_warm_dropout_specialization_and_local_shadow_control() {
        // Reuse #1764's admitted generic/static-rate and local-shadow shapes.
        // Evaluating the checked value body directly reaches eval_app rather
        // than lowering a whole main wrapper. Warming resolves the same keep.
        // The local shadow takes the key too, so both rows spell one call.
        let tensors = UnordMap::new();
        for local in [false, true] {
            let shadow = if local {
                "keep = fn (j: key, v: tensor[4, f32]) -> v\n"
            } else {
                ""
            };
            let library = checked_library(&format!(
                "def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))\nout = {{ {shadow}x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n (k1, k2) = split_key(key_from_seed(42i64))\n first = keep(k1, copy(x))\n next = dropout(k2, x, 0.5f32)\n (first, next) }}\n"
            ));
            for warm in [false, true] {
                let mut ctx = context(&library, &tensors);
                if warm {
                    let closure = ctx.resolve_top_level("keep").unwrap();
                    assert!(
                        matches!(closure, RuntimeValue::Closure { def_name: Some(ref name), .. } if name == "keep")
                    );
                }
                let body = ctx.lookup_top_level_def("out").unwrap().1;
                let RuntimeValue::Tuple(values) = ctx.eval_expr(&body).unwrap() else {
                    panic!("checked output is a pair")
                };
                assert_eq!(values.len(), 2);
                // `key_ref.py`'s rate-0.5 masks over ones for the halves of
                // `split_key(key_from_seed(42i64))`: [2,0,0,2] and [2,2,0,0].
                let expected = if local {
                    [
                        ["3f800000"; 4],
                        ["40000000", "40000000", "00000000", "00000000"],
                    ]
                } else {
                    [
                        ["40000000", "00000000", "00000000", "40000000"],
                        ["40000000", "40000000", "00000000", "00000000"],
                    ]
                };
                for (value, expected_bits) in values.iter().zip(expected) {
                    assert_eq!(
                        bits(value),
                        serde_json::json!({"type":"tensor", "value":{"shape":[4], "data":{"dtype":"f32", "bits":expected_bits}}})
                    );
                }
                assert!(ctx.transcript.is_empty());
            }
        }
    }

    // #1956: these are real checked library contexts, with the production
    // session and resolver. Caller-only entries model an already-entered
    // lexical frame; none is inserted into the declaration environment.
    fn declaration_caller_frame(ctx: &mut EvalContext<'_>) -> Option<Expr> {
        let ty = ctx.program.type_env().get("next_draw").cloned();
        assert!(ty.is_some());
        ctx.bindings
            .insert("caller_value".into(), RuntimeValue::int_lit(71));
        ctx.binding_types.insert("caller_type".into(), ty.clone());
        ctx.precision_bindings.insert("p".into(), Prim::F64);
        ty
    }

    fn assert_declaration_caller_frame(ctx: &EvalContext<'_>, ty: &Option<Expr>) {
        assert_eq!(ctx.binding_types.len(), 1);
        assert_eq!(ctx.binding_types.get("caller_type"), Some(ty));
        assert_eq!(ctx.precision_bindings.len(), 1);
        assert_eq!(ctx.precision_bindings.get("p"), Some(&Prim::F64));
        assert_eq!(
            bits(
                ctx.bindings
                    .get("caller_value")
                    .expect("caller_value bound")
            ),
            bits(&RuntimeValue::int_lit(71))
        );
        assert_eq!(
            ctx.bindings.len(),
            1,
            "declaration successes are not lexical entries"
        );
    }

    #[test]
    fn declaration_success_restores_all_three_caller_maps_without_losing_the_value() {
        let library = library(DRAW, "k: key, x: tensor[2, f32]", LIVE);
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        let ty = declaration_caller_frame(&mut ctx);
        let first = ctx.resolve_top_level("weights").unwrap();
        let second = ctx.resolve_top_level("weights").unwrap();
        assert_eq!(bits(&first), bits(&expected_draw(17)));
        assert_eq!(bits(&second), bits(&first));
        assert_eq!(ctx.transcript, ["initialize"]);
        assert!(ctx.resolving_top_levels.is_empty());
        let next = admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap();
        assert_eq!(bits(&next), bits(&expected_draw(NEXT_SEED)));
        assert_eq!(ctx.transcript, ["initialize"]);
        assert_declaration_caller_frame(&ctx, &ty);
    }

    #[test]
    fn declaration_closure_does_not_capture_caller_values_or_precision_parameters() {
        let library = library(DRAW, "k: key, x: tensor[2, f32]", LIVE);
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        let ty = declaration_caller_frame(&mut ctx);
        let callable = ctx.resolve_top_level("sample").unwrap();
        let RuntimeValue::Closure {
            env,
            precision_env,
            def_name,
            ..
        } = callable
        else {
            panic!("checked named declaration resolves to a closure")
        };
        assert_eq!(def_name.as_deref(), Some("sample"));
        assert!(ctx.transcript.is_empty());
        assert_eq!(
            (env.is_empty(), precision_env.is_empty()),
            (true, true),
            "caller values and precision parameters must both be absent"
        );
        assert_declaration_caller_frame(&ctx, &ty);
    }

    #[test]
    fn declaration_failure_retries_and_restores_three_maps() {
        let initializer =
            format!("_ = {DRAW}\n to_tensor([cast(floor_div(1i32, 0i32), f32), 0.0f32])");
        let library = library(&initializer, "k: key, x: tensor[2, f32]", LIVE);
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        let ty = declaration_caller_frame(&mut ctx);
        for _ in 0..2 {
            let error = ctx.resolve_top_level("weights").unwrap_err();
            assert_eq!(error, "numeric trap: division by zero in floor_div at i32");
            assert_declaration_caller_frame(&ctx, &ty);
            assert!(ctx.resolving_top_levels.is_empty());
        }
        // A failed initialization caches nothing, so each resolution runs it.
        assert_eq!(ctx.transcript, ["initialize", "initialize"]);
        let next = admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap();
        assert_eq!(bits(&next), bits(&expected_draw(NEXT_SEED)));
        assert_eq!(ctx.transcript, ["initialize", "initialize"]);
    }

    #[test]
    fn declaration_inner_success_survives_outer_failure_and_retry() {
        let source = format!(
            "inner = {{ _ = print(\"inner\")\n 9 }}\nouter = {{ _ = print(\"outer\")\n saved = inner\n _ = {DRAW}\n add(saved, floor_div(1i32, 0i32)) }}\ndef next_draw(k: key, x: tensor[2, f32]) -> tensor[2, f32] = uniform_like(k, x, 0.0f32, 1.0f32)\n"
        );
        let library = checked_library(&source);
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        let ty = declaration_caller_frame(&mut ctx);
        for _ in 0..2 {
            assert_eq!(
                ctx.resolve_top_level("outer").unwrap_err(),
                "numeric trap: division by zero in floor_div at i32"
            );
            assert_declaration_caller_frame(&ctx, &ty);
            assert!(ctx.resolving_top_levels.is_empty());
        }
        let inner = ctx.resolve_top_level("inner").unwrap();
        assert_eq!(bits(&inner), bits(&RuntimeValue::int_lit(9)));
        let next = admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap();
        assert_eq!(bits(&next), bits(&expected_draw(NEXT_SEED)));
        // `inner` initialized once, inside the first `outer` attempt, and
        // survived both failures; the retry reran only `outer`.
        assert_eq!(ctx.transcript, ["outer", "inner", "outer"]);
    }

    #[test]
    fn declaration_canonical_resolution_ignores_same_named_caller_value() {
        let library = library(DRAW, "k: key, x: tensor[2, f32]", LIVE);
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        ctx.bindings.insert("weights".into(), zeros());
        let declared = ctx.resolve_top_level("weights").unwrap();
        assert_eq!(bits(&declared), bits(&expected_draw(17)));
        assert_eq!(
            bits(ctx.bindings.get("weights").expect("weights bound")),
            bits(&zeros())
        );
        assert_eq!(ctx.transcript, ["initialize"]);
    }

    #[test]
    fn declaration_supplied_value_does_not_replace_callable_or_its_observation() {
        let library = checked_library(
            "def value() -> tensor[2, f32] = to_tensor([3.0f32, 5.0f32])\nalias = value\nseen = value\ncalled = value()\naliased = alias()\n",
        );
        let RuntimeValue::Tensor(supplied) = zeros() else {
            unreachable!()
        };
        let tensors = [("value".into(), supplied)].into_iter().collect();
        let mut ctx = context(&library, &tensors);
        let seen = ctx.lookup_top_level_def("seen").unwrap().1;
        let called = ctx.lookup_top_level_def("called").unwrap().1;
        let aliased = ctx.lookup_top_level_def("aliased").unwrap().1;
        let expected = serde_json::json!({"type":"tensor","value":{"shape":[2],"data":{"dtype":"f32","bits":["40400000", "40a00000"]}}});
        for warm in [false, true] {
            if warm {
                let closure = ctx.resolve_top_level("value").unwrap();
                assert!(matches!(closure, RuntimeValue::Closure { .. }));
            }
            let observation = ctx.eval_expr(&seen).unwrap();
            assert_eq!(bits(&ctx.eval_expr(&called).unwrap()), expected);
            assert_eq!(bits(&ctx.eval_expr(&aliased).unwrap()), expected);
            assert_eq!(bits(&observation), bits(&zeros()));
        }
        assert!(ctx.transcript.is_empty());
    }

    #[test]
    fn declaration_named_kernel_route_and_draws_are_independent_of_warm_cache() {
        let library = library(DRAW, "k: key, x: tensor[2, f32]", LIVE);
        let tensors = UnordMap::new();
        for warm in [false, true] {
            let mut ctx = context(&library, &tensors);
            if warm {
                ctx.resolve_top_level("sample").unwrap();
            }
            // admitted_call separately proves the kernel route, no staged
            // plan, and exactly one UniformLike keyed by the caller's key.
            let actual = admitted_call(&mut ctx, "sample", SAMPLE_SEED).unwrap();
            let next = admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap();
            assert_eq!(bits(&actual), bits(&expected_draw(SAMPLE_SEED)));
            assert_eq!(bits(&next), bits(&expected_draw(NEXT_SEED)));
            assert_eq!(ctx.transcript, ["initialize"]);
        }
    }

    #[test]
    fn declaration_anonymous_closure_keeps_lexical_shadow_and_no_named_dispatch() {
        let library =
            checked_library("a = 3.0f32\nmaker = { a = 7.0f32\n fn (x: f32) -> add(x, a) }\n");
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        let closure = ctx.resolve_top_level("maker").unwrap();
        let RuntimeValue::Closure { def_name, env, .. } = &closure else {
            panic!("checked anonymous function remains a closure")
        };
        assert!(
            def_name.is_none(),
            "anonymous application must bypass named-kernel dispatch"
        );
        let captured = env.get("a").map(bits);
        let actual = ctx
            .apply_resolved_callable(closure, vec![RuntimeValue::float_lit(1.0)])
            .unwrap();
        assert_eq!(
            (bits(&actual), captured),
            (
                bits(&RuntimeValue::float_lit(8.0)),
                Some(bits(&RuntimeValue::float_lit(7.0)))
            )
        );
        assert!(ctx.transcript.is_empty());
    }

    #[test]
    fn declaration_exact_and_unique_resolution_never_populates_an_ambiguous_alias() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::create_dir(directory.path().join("src")).unwrap();
        std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"identity-frames\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Probe\"\n", crate::COMPILER_VERSION)).unwrap();
        for (module, value) in [("Left", 3), ("Right", 5)] {
            std::fs::write(directory.path().join("src").join(format!("{}.ch", module.to_lowercase())), format!("module Probe.{module}\nexport (value, unique_{value})\nvalue = {{ _ = print(\"{module}\")\n {value} }}\nunique_{value} = {value}\n")).unwrap();
        }
        let compiled = crate::compile_reef_context(
            directory.path(),
            directory.path(),
            &chelis_std_bundle::EMBEDDED_RUNTIME,
        )
        .unwrap();
        let library = compiled.checked_library();
        let tensors = UnordMap::new();
        let mut ctx = context(library, &tensors);
        let identities = ctx
            .program
            .defs()
            .to_sorted()
            .into_iter()
            .filter(|(name, _)| terminal_name_matches(name, "value"))
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            identities.len(),
            2,
            "actual linked declarations supply the collision"
        );
        assert!(ctx.lookup_top_level_def("value").is_none());
        for name in identities {
            let canonical = ctx.lookup_top_level_def(&name).unwrap().0;
            assert_eq!(canonical, name);
            let expected = if name.contains("Left") { 3 } else { 5 };
            assert_eq!(
                bits(&ctx.resolve_top_level(&name).unwrap()),
                bits(&RuntimeValue::int_lit(expected))
            );
            assert!(ctx.lookup_top_level_def("value").is_none());
            assert_eq!(
                ctx.resolve_top_level("value").unwrap_err(),
                "unknown runtime name `value`"
            );
        }
        let canonical = ctx.lookup_top_level_def("unique_3").unwrap().0;
        let first = ctx.resolve_top_level("unique_3").unwrap();
        let second = ctx.resolve_top_level(&canonical).unwrap();
        assert_eq!(bits(&first), bits(&RuntimeValue::int_lit(3)));
        assert_eq!(bits(&second), bits(&first));
        let mut events = ctx.transcript.clone();
        events.sort();
        assert_eq!(events, ["Left", "Right"]);
        assert!(
            !ctx.bindings.contains_key("unique_3"),
            "canonical successes do not create lexical short aliases"
        );
    }

    #[test]
    fn legacy_capture_order_valid_lazy_initializer_runs_once_in_the_first_call() {
        let library = library(DRAW, "k: key, x: tensor[2, f32]", LIVE);
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        assert!(!ctx.declaration_values.contains_key("weights"));
        assert!(ctx.transcript.is_empty());
        let actual = admitted_call(&mut ctx, "sample", SAMPLE_SEED).unwrap();
        // The first call initialized the capture, with its own key.
        assert_eq!(ctx.transcript, ["initialize"]);
        let initialized = ctx
            .declaration_values
            .get("weights")
            .expect("capture initialized lazily");
        assert_eq!(bits(initialized), bits(&expected_draw(17)));
        assert!(!ctx.bindings.contains_key("weights"));
        let next = admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap();
        let again = admitted_call(&mut ctx, "sample", SAMPLE_SEED).unwrap();
        assert_eq!(
            (bits(&actual), bits(&next), bits(&again), &ctx.transcript),
            (
                bits(&expected_draw(SAMPLE_SEED)),
                bits(&expected_draw(NEXT_SEED)),
                bits(&expected_draw(SAMPLE_SEED)),
                &vec!["initialize".to_owned()]
            )
        );
    }

    #[test]
    fn legacy_capture_order_initializer_failure_retains_only_entered_prefix() {
        let initializer =
            format!("_ = {DRAW}\n to_tensor([cast(floor_div(1i32, 0i32), f32), 0.0f32])");
        let library = library(&initializer, "k: key, x: tensor[2, f32]", LIVE);
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        let error = admitted_call(&mut ctx, "sample", SAMPLE_SEED).unwrap_err();
        assert_eq!(error, "numeric trap: division by zero in floor_div at i32");
        // The entered prefix is the initializer's print; nothing is bound.
        assert_eq!(ctx.transcript, ["initialize"]);
        assert!(!ctx.bindings.contains_key("weights"));
        assert!(!ctx.declaration_values.contains_key("weights"));
        let next = admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap();
        assert_eq!(bits(&next), bits(&expected_draw(NEXT_SEED)));
        assert_eq!(ctx.transcript, ["initialize"]);
    }

    #[test]
    fn legacy_capture_order_dead_and_formal_shadow_do_not_initialize_library_value() {
        for (params, body) in [
            (
                "k: key, x: tensor[2, f32]",
                "if true then uniform_like(k, x, 0.0f32, 1.0f32) else uniform_like(k, weights, 0.0f32, 1.0f32)",
            ),
            (
                "k: key, weights: tensor[2, f32]",
                "uniform_like(k, weights, 0.0f32, 1.0f32)",
            ),
        ] {
            let library = library(DRAW, params, body);
            let tensors = UnordMap::new();
            let mut ctx = context(&library, &tensors);
            let actual = admitted_call(&mut ctx, "sample", SAMPLE_SEED).unwrap();
            let next = admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap();
            assert_eq!(bits(&actual), bits(&expected_draw(SAMPLE_SEED)));
            assert_eq!(bits(&next), bits(&expected_draw(NEXT_SEED)));
            assert!(ctx.transcript.is_empty() && !ctx.bindings.contains_key("weights"));
        }
    }

    /// The checker's messages, for an assertion's failure text.
    fn messages(errors: &[chelis_types::errors::CheckError]) -> String {
        errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    }

    #[test]
    fn legacy_capture_order_nullary_tensor_observation_boundary() {
        let source = format!(
            "def weights() -> tensor[2, f32] = {{ _ = print(\"initialize\")\n {DRAW} }}\n\
             def sample(k: key, x: tensor[2, f32]) -> tensor[2, f32] = {LIVE}\n"
        );
        let prepared =
            crate::pipeline::prepare_source(crate::schema::SourceKind::Surf, &source, None)
                .unwrap();
        let error = crate::pipeline::check_prepared_library(prepared).unwrap_err();
        let crate::pipeline::LibraryRejection::Type { report } = error else {
            panic!("a nullary callable template must fail type checking");
        };
        assert!(
            report.errors.iter().any(|error| {
                matches!(
                    error.kind,
                    chelis_types::errors::CheckErrorKind::TypeMismatch
                ) && error.message
                    == "uniform_like expects tensor template input, got () -> tensor[2, f32]"
            }),
            "{}",
            messages(&report.errors)
        );
        let declaration = source.split("def sample").next().unwrap();
        let prepared =
            crate::pipeline::prepare_source(crate::schema::SourceKind::Surf, declaration, None)
                .unwrap();
        let library = crate::pipeline::check_prepared_library(prepared).unwrap();
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        let closure = ctx.resolve_top_level("weights").unwrap();
        assert!(matches!(closure, RuntimeValue::Closure { .. }));
        let error = stage_kernel_argument("sample", "weights", &closure, Prim::F32).unwrap_err();
        assert!(error.contains("expects a tensor or scalar argument"));
        // Neither resolving nor staging the callable ran its body.
        assert!(ctx.transcript.is_empty());
    }

    #[test]
    fn legacy_capture_order_transform_failure_stops_at_the_initializer() {
        let source = format!(
            "weights: tensor[2, f32] = {{ _ = print(\"initialize\")\n _ = {DRAW}\n to_tensor([cast(floor_div(1i32, 0i32), f32), 0.0f32]) }}\n\
             def loss(k: key, x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, uniform_like(k, weights, 0.0f32, 1.0f32)), 0)\n\
             def derivative() = grad(loss, wrt=x)\n\
             def next_draw(k: key, x: tensor[2, f32]) -> tensor[2, f32] = uniform_like(k, x, 0.0f32, 1.0f32)\n"
        );
        let prepared =
            crate::pipeline::prepare_source(crate::schema::SourceKind::Surf, &source, None)
                .unwrap();
        let library = crate::pipeline::check_prepared_library(prepared)
            .expect("transform fixture passes real type, effect and linearity admission");
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        let factory = ctx.resolve_top_level("derivative").unwrap();
        let callable = ctx.apply_resolved_callable(factory, vec![]).unwrap();
        assert!(matches!(callable, RuntimeValue::Transform { .. }));
        // Building the transform initializes nothing.
        assert!(ctx.transcript.is_empty() && !ctx.bindings.contains_key("weights"));
        let error = ctx
            .apply_resolved_callable(callable, vec![key(SAMPLE_SEED), zeros()])
            .unwrap_err();
        assert_eq!(error, "numeric trap: division by zero in floor_div at i32");
        assert_eq!(ctx.transcript, ["initialize"]);
        assert!(!ctx.bindings.contains_key("weights"));
        let next = admitted_call(&mut ctx, "next_draw", NEXT_SEED).unwrap();
        assert_eq!(bits(&next), bits(&expected_draw(NEXT_SEED)));
        assert_eq!(ctx.transcript, ["initialize"]);
    }

    #[test]
    fn transform_preparation_supports_vmap_capture_at_authored_rank() {
        let library = checked_library(
            "weights = { _ = print(\"initialize\")\n scalar_to_tensor(3.0f32) }\n\
             def weighted(x: tensor[f32]) -> tensor[f32] = mul(x, weights)\n\
             def mapped() = vmap(weighted)\n",
        );
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        let factory = ctx.resolve_top_level("mapped").unwrap();
        let callable = ctx.apply_resolved_callable(factory, vec![]).unwrap();
        assert!(ctx.transcript.is_empty());
        let input = RuntimeValue::Tensor(
            RuntimeTensorValue::from_wide("test", Prim::F32, vec![2], vec![2.0, 4.0]).unwrap(),
        );
        let actual = ctx.apply_resolved_callable(callable, vec![input]).unwrap();
        let expected = RuntimeValue::Tensor(
            RuntimeTensorValue::from_wide("test", Prim::F32, vec![2], vec![6.0, 12.0]).unwrap(),
        );
        assert_eq!(bits(&actual), bits(&expected));
        // One initialization serves every mapped row.
        assert_eq!(ctx.transcript, ["initialize"]);
    }

    #[test]
    fn transform_preparation_vmap_capture_preserves_initializer_failure() {
        let library = checked_library(
            "weights = { _ = print(\"initialize\")\n scalar_to_tensor(cast(floor_div(1i32, 0i32), f32)) }\n\
             def weighted(x: tensor[f32]) -> tensor[f32] = mul(x, weights)\n\
             def mapped() = vmap(weighted)\n",
        );
        let tensors = UnordMap::new();
        let mut ctx = context(&library, &tensors);
        let factory = ctx.resolve_top_level("mapped").unwrap();
        let callable = ctx.apply_resolved_callable(factory, vec![]).unwrap();
        let error = ctx
            .apply_resolved_callable(callable, vec![zeros()])
            .unwrap_err();
        assert_eq!(error, "numeric trap: division by zero in floor_div at i32");
        assert_eq!(ctx.transcript, ["initialize"]);
        assert!(!ctx.bindings.contains_key("weights"));
        assert!(!ctx.declaration_values.contains_key("weights"));
    }

    /// The key-form analogue of the retired unhandled-`Random` rejection: a
    /// library initializer whose draw omits its key, directly or through a
    /// helper, is rejected before any initializer can run.
    #[test]
    fn legacy_capture_order_keyless_library_initializer_is_not_admitted() {
        let keyless = "uniform_like(to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32)";
        for source in [
            format!("weights = {{ _ = print(\"initialize\")\n {keyless} }}"),
            "def draw(x: tensor[2, f32]) -> tensor[2, f32] = uniform_like(x, 0.0f32, 1.0f32)\nweights = draw(to_tensor([0.0f32, 0.0f32]))".to_owned(),
            format!("def draw() -> tensor[2, f32] = {keyless}\nweights = draw()"),
        ] {
            let prepared =
                crate::pipeline::prepare_source(crate::schema::SourceKind::Surf, &source, None)
                    .unwrap();
            let error = crate::pipeline::check_prepared_library(prepared).unwrap_err();
            let crate::pipeline::LibraryRejection::Type { report } = error else {
                panic!("a keyless draw must fail type checking: {source}");
            };
            assert!(
                report
                    .errors
                    .iter()
                    .any(|error| error.message.contains("is the retired counter-stream spelling")),
                "{source}: {}",
                messages(&report.errors)
            );
        }
    }
}

#[cfg(test)]
mod list_dir_conversion_tests {
    use super::list_dir_names_to_strings;
    use std::ffi::OsString;
    use std::path::Path;

    #[test]
    fn list_dir_conversion_preserves_unicode_and_empty_lists() {
        let names = ["替", "\u{fffd}", "é", "e\u{301}", "a\n\"\\z"];
        let mut expected = names.to_vec();
        expected.sort();
        assert_eq!(
            list_dir_names_to_strings(
                names.into_iter().map(OsString::from).collect(),
                Path::new("/dir")
            )
            .map_err(|error| error.to_string()),
            Ok(expected.into_iter().map(str::to_owned).collect())
        );
        assert_eq!(
            list_dir_names_to_strings(vec![], Path::new("/dir")).map_err(|error| error.to_string()),
            Ok(vec![])
        );
    }

    #[cfg(unix)]
    #[test]
    fn list_dir_conversion_rejects_collisions_and_selects_first_raw_name() {
        use std::os::unix::ffi::OsStringExt;
        // In-memory host names exercise the production conversion on macOS,
        // including filesystems that cannot create an invalid-name fixture.
        for names in [
            vec![b"a\xff".to_vec(), b"a\xfe".to_vec()],
            vec![b"a\xfe".to_vec(), b"a\xff".to_vec()],
        ] {
            let mut names: Vec<_> = names.into_iter().map(OsString::from_vec).collect();
            names.push(OsString::from("0-valid"));
            assert_eq!(
                list_dir_names_to_strings(names, Path::new("/dir")).map_err(|error| error.to_string()),
                Err("IO trap in list_dir: directory b\"/dir\", entry b\"a\\xfe\": name is not valid UTF-8".to_owned())
            );
        }
        // Raw order and replacement-string order disagree for these names.
        let names = vec![
            OsString::from_vec(b"\x81a".to_vec()),
            OsString::from_vec(b"\x80z".to_vec()),
        ];
        assert_eq!(
            list_dir_names_to_strings(names, Path::new("/dir")).map_err(|error| error.to_string()),
            Err("IO trap in list_dir: directory b\"/dir\", entry b\"\\x80z\": name is not valid UTF-8".to_owned())
        );
    }

    #[cfg(unix)]
    #[test]
    fn list_dir_conversion_escapes_directory_and_offending_entry_reversibly() {
        use std::os::unix::ffi::OsStringExt;
        let names = vec![OsString::from_vec(b"bad\n\r\t\\\"'\xff".to_vec())];
        assert_eq!(
            list_dir_names_to_strings(names, Path::new("/d\n\r\t\\\"'é")).map_err(|error| error.to_string()),
            Err("IO trap in list_dir: directory b\"/d\\n\\r\\t\\\\\\\"\\'\\xc3\\xa9\", entry b\"bad\\n\\r\\t\\\\\\\"\\'\\xff\": name is not valid UTF-8".to_owned())
        );
    }
}
