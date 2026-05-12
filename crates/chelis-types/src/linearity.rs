use std::collections::{HashMap, HashSet};

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use serde::{Deserialize, Serialize};

use crate::CheckedProgram;
use crate::errors::{CheckError, CheckErrorKind};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearityInfo {
    reusable_inputs_by_offset: HashMap<usize, usize>,
}

impl LinearityInfo {
    pub fn reusable_input_for_span(&self, span: Span) -> Option<usize> {
        self.reusable_inputs_by_offset.get(&span.offset).copied()
    }

    fn mark_reusable_input(&mut self, span: Span, input_index: usize) {
        self.reusable_inputs_by_offset
            .entry(span.offset)
            .or_insert(input_index);
    }
}

#[derive(Debug, Clone)]
enum BindingState {
    Live { borrow_sites: Vec<String> },
    Consumed(ConsumeSite),
}

#[derive(Debug, Clone)]
struct ConsumeSite {
    description: String,
}

#[derive(Debug, Clone, Default)]
struct LinearScope {
    bindings: HashMap<String, Vec<BindingState>>,
    types: HashMap<String, Vec<Option<Expr>>>,
}

impl LinearScope {
    fn declare<S: Into<String>>(&mut self, name: S, ty: Option<Expr>) {
        let name = name.into();
        self.bindings
            .entry(name.clone())
            .or_default()
            .push(BindingState::Live {
                borrow_sites: Vec::new(),
            });
        self.types.entry(name).or_default().push(ty);
    }

    fn pop(&mut self, name: &str) -> Option<(Option<Expr>, BindingState)> {
        let ty = if let Some(stack) = self.types.get_mut(name) {
            let ty = stack.pop();
            if stack.is_empty() {
                self.types.remove(name);
            }
            ty
        } else {
            None
        };

        let state = if let Some(stack) = self.bindings.get_mut(name) {
            let state = stack.pop();
            if stack.is_empty() {
                self.bindings.remove(name);
            }
            state
        } else {
            None
        };

        state.map(|state| (ty.flatten(), state))
    }

    fn top(&self, name: &str) -> Option<&BindingState> {
        self.bindings.get(name).and_then(|stack| stack.last())
    }

    fn ty(&self, name: &str) -> Option<&Expr> {
        self.types
            .get(name)
            .and_then(|stack| stack.last())
            .and_then(|ty| ty.as_ref())
    }

    fn consume(&mut self, name: &str, site: ConsumeSite) {
        if let Some(stack) = self.bindings.get_mut(name)
            && let Some(top) = stack.last_mut()
        {
            *top = BindingState::Consumed(site);
        }
    }

    fn borrow(&mut self, name: &str, site: String) {
        if let Some(stack) = self.bindings.get_mut(name)
            && let Some(BindingState::Live { borrow_sites }) = stack.last_mut()
        {
            borrow_sites.push(site);
        }
    }

    fn visible_names(&self) -> Vec<String> {
        self.bindings.keys().cloned().collect()
    }
}

struct Checker {
    errors: Vec<CheckError>,
    info: LinearityInfo,
    top_level_types: HashMap<String, Expr>,
}

pub fn check_linearity(program: &CheckedProgram) -> Result<CheckedProgram, Vec<CheckError>> {
    let mut checker = Checker {
        errors: Vec::new(),
        info: LinearityInfo::default(),
        top_level_types: program.type_env().clone(),
    };
    let mut scope = LinearScope::default();

    for expr in program.annotated_exprs() {
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            scope.declare(name, program.type_env().get(name).cloned());
        }
    }

    for expr in program.annotated_exprs() {
        checker.check_top_level(expr, &mut scope);
    }

    if checker.errors.is_empty() {
        Ok(program.clone().with_linearity(checker.info))
    } else {
        Err(checker.errors)
    }
}

/// Phase E: check linearity of `new_program` against an outer-scope
/// `library_program` whose linearity was already validated when its
/// context was built.
///
/// Library bindings are treated as always-available references —
/// library tensor parameters are scoped to their owning library `def`,
/// not to new-code defs, and so MUST NOT be added to new-code's
/// "consumed" set when called. To enforce that, this entry point walks
/// ONLY `new_program.annotated_exprs()` for body checks; library
/// bodies are never re-walked. Library def names are pre-declared in
/// the top-level scope (with their unioned types from
/// `new_program.type_env()`) so that `(var libname)` references in
/// new code resolve correctly. Library defs all have function types,
/// which the linearity checker treats as non-linear, so referencing
/// one never triggers consumption of new-code free variables.
///
/// Per Phase E acceptance: for `(library, snippet)`, this function's
/// Result on `snippet` must match `check_linearity(library + snippet)`'s
/// Result on the snippet portion. The function is pure — repeated calls
/// with the same `library_program` see no leaked state from prior new
/// programs.
///
/// `library_program` is consumed only by reference; it is left
/// unchanged.
pub fn check_linearity_with_context(
    library_program: &CheckedProgram,
    new_program: &CheckedProgram,
) -> Result<CheckedProgram, Vec<CheckError>> {
    // Pre-compute library callable signatures: a name set keyed on each
    // library `def`, used to ensure new-code call sites referring to a
    // library def don't accidentally walk the library body. The set is
    // currently informational — the checker never walks call targets, so
    // simply not feeding library exprs to `check_top_level` is what
    // enforces the "don't re-walk library bodies" invariant. The
    // pre-computation here is the documented contract surface.
    let _library_callables: HashSet<String> = library_program
        .annotated_exprs()
        .iter()
        .filter_map(|expr| {
            if let Expr::List(list, _) = expr
                && get_tag(list) == Some("def")
            {
                children(list)
                    .first()
                    .and_then(symbol_name)
                    .map(str::to_string)
            } else {
                None
            }
        })
        .collect();

    // Top-level types come from new_program.type_env(), which Phase C
    // already unioned (library + new-code). New-code types win on shadow.
    let mut checker = Checker {
        errors: Vec::new(),
        info: LinearityInfo::default(),
        top_level_types: new_program.type_env().clone(),
    };

    let mut scope = LinearScope::default();

    // Pre-declare library def names so `(var libname)` references in
    // new code resolve to a Live binding with the library's function
    // type. Function types are non-linear (no tensor content), so a
    // pure reference never triggers consumption of new-code locals.
    for expr in library_program.annotated_exprs() {
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            scope.declare(name, new_program.type_env().get(name).cloned());
        }
    }

    // Pre-declare new-code def names. New-code shadows library on
    // collision (declare last → top of stack wins).
    for expr in new_program.annotated_exprs() {
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            scope.declare(name, new_program.type_env().get(name).cloned());
        }
    }

    // Walk ONLY new-code bodies. Library bodies are never re-walked,
    // so library tensor parameters never enter the new-code scope.
    for expr in new_program.annotated_exprs() {
        checker.check_top_level(expr, &mut scope);
    }

    if checker.errors.is_empty() {
        Ok(new_program.clone().with_linearity(checker.info))
    } else {
        Err(checker.errors)
    }
}

impl Checker {
    fn check_top_level(&mut self, expr: &Expr, scope: &mut LinearScope) {
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
        {
            let kids = children(list);
            if let (Some(name), Some(body)) = (kids.first().and_then(symbol_name), kids.get(1))
                && !(is_var_expr(body) && var_name(body) == Some(name))
            {
                if matches!(get_tag_expr(body), Some("borrow")) {
                    self.invalid_borrow(body, "borrow cannot be returned from a function");
                    return;
                }
                // V2-F4: top-level `def name = x` where the body is a
                // bare `(var x)` of an owned-linear type is an
                // aliasing binding consume; at the IR level
                // `lower_var` returns the cached `bindings["x"]` node
                // for both `(var x)` and the new top-level `name`, so
                // the value is structurally shared, not destroyed.
                // Mirror the `check_let` path
                // (`linearity.rs:397-407`) by tagging the consume
                // with a `"binding `name` at offset N"` description.
                // That feeds the PR #29 `read_or_error` tolerance
                // (`linearity.rs:644`), letting subsequent borrow
                // reads of the original variable succeed. Without
                // this branch the body would fall through to
                // `check_expr -> consume_var_expr(generic_site)` and
                // tag the consume with `"use at offset N"`, which
                // the tolerance does not match.
                if is_var_expr(body) && self.expr_is_owned_linear(body, scope) {
                    self.consume_var_expr(
                        body,
                        scope,
                        ConsumeSite {
                            description: format!(
                                "binding `{name}` at offset {}",
                                body.span().offset
                            ),
                        },
                    );
                } else {
                    self.check_expr(body, scope);
                }
            }
            return;
        }
        self.check_expr(expr, scope);
    }

    fn check_expr(&mut self, expr: &Expr, scope: &mut LinearScope) {
        match expr {
            Expr::Atom(_, _) => {}
            Expr::Map(map, _) => {
                for (_, value) in &map.entries {
                    self.check_expr(value, scope);
                }
            }
            Expr::MetaExpr(meta, _) => self.check_expr(&meta.expr, scope),
            Expr::List(list, _) => match get_tag(list) {
                Some("var") => self.consume_var_expr(expr, scope, generic_site(expr)),
                Some("copy") => self.check_copy(list, scope),
                Some("realize") => self.check_realize(expr, list, scope),
                Some("borrow") => {
                    self.invalid_borrow(expr, "borrow is only valid as a direct call argument")
                }
                Some("app") => self.check_app(expr, list, scope),
                Some("pipe") => self.check_pipe(list, scope),
                Some("let") => self.check_let(list, scope),
                Some("fn") => self.check_fn(expr, list, scope),
                Some("if") => self.check_if(list, scope),
                Some("match") => self.check_match(list, scope),
                _ => {
                    for child in children(list) {
                        self.check_expr(child, scope);
                    }
                }
            },
        }
    }

    fn check_copy(&mut self, list: &List, scope: &mut LinearScope) {
        if let Some(child) = children(list).first() {
            if let Some(borrowed) = borrow_inner(child) {
                self.check_borrow_arg(child, borrowed, scope);
            } else if is_var_expr(child) && self.expr_is_owned_linear(child, scope) {
                self.read_var_expr(child, scope);
            } else {
                self.check_expr(child, scope);
            }
        }
    }

    fn check_realize(&mut self, expr: &Expr, list: &List, scope: &mut LinearScope) {
        if let Some(child) = children(list).first() {
            if is_var_expr(child) && self.expr_is_owned_linear(child, scope) {
                self.consume_var_expr(child, scope, realize_site(expr));
            } else {
                self.check_expr(child, scope);
            }
        }
    }

    fn check_app(&mut self, expr: &Expr, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        let builtin = kids.first().and_then(var_name);
        if let Some(func) = kids.first() {
            self.check_expr(func, scope);
        }
        for (index, arg) in kids.iter().enumerate().skip(1) {
            if let Some(borrowed) = borrow_inner(arg) {
                self.check_borrow_arg(arg, borrowed, scope);
            } else if self.arg_is_borrowed(kids.first(), builtin, index - 1, scope)
                && is_var_expr(arg)
                && self.expr_is_owned_linear(arg, scope)
            {
                self.read_var_expr(arg, scope);
            } else if is_var_expr(arg) && self.expr_is_owned_linear(arg, scope) {
                self.consume_var_expr(arg, scope, app_site(expr, list));
            } else {
                self.check_expr(arg, scope);
            }
        }
        self.maybe_mark_reusable_app_input(expr, kids, scope);
    }

    fn check_pipe(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if kids.is_empty() {
            return;
        }
        let mut current = &kids[0];
        for stage in &kids[1..] {
            let stage_builtin = var_name(stage);
            if self.arg_is_borrowed(Some(stage), stage_builtin, 0, scope) {
                if is_var_expr(current) && self.expr_is_owned_linear(current, scope) {
                    self.read_var_expr(current, scope);
                } else {
                    self.check_expr(current, scope);
                }
            } else if is_var_expr(current) && self.expr_is_owned_linear(current, scope) {
                self.consume_var_expr(current, scope, pipe_site(current, stage));
            } else {
                self.check_expr(current, scope);
            }
            current = stage;
        }
    }

    fn check_borrow_arg(&mut self, borrow_expr: &Expr, inner: &Expr, scope: &mut LinearScope) {
        if !is_var_expr(inner) {
            self.invalid_borrow(
                borrow_expr,
                "borrowed arguments must be direct variable references",
            );
            return;
        }
        if !self.expr_is_owned_or_borrow_linear(inner, scope) {
            self.invalid_borrow(
                borrow_expr,
                "borrowed arguments must be tensor or tensor-carrying values",
            );
            return;
        }
        self.read_var_expr(inner, scope);
    }

    fn check_let(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if kids.len() < 2 {
            return;
        }
        let mut pushed = Vec::new();
        if let Expr::List(bind_list, _) = &kids[0] {
            let bind_kids = children(bind_list);
            let mut index = 0;
            while index + 1 < bind_kids.len() {
                let Some(name) = symbol_name(&bind_kids[index]) else {
                    index += 2;
                    continue;
                };
                let value = &bind_kids[index + 1];
                if is_var_expr(value) && self.expr_is_owned_linear(value, scope) {
                    self.consume_var_expr(
                        value,
                        scope,
                        ConsumeSite {
                            description: format!(
                                "binding `{name}` at offset {}",
                                value.span().offset
                            ),
                        },
                    );
                } else if matches!(get_tag_expr(value), Some("borrow")) {
                    self.invalid_borrow(value, "borrow cannot be stored in a binding");
                } else {
                    self.check_expr(value, scope);
                }
                scope.declare(name, self.expr_type(value, scope).cloned());
                pushed.push(name.to_string());
                index += 2;
            }
        }
        self.check_expr(&kids[1], scope);
        for name in pushed.into_iter().rev() {
            self.pop_and_check(scope, &name, expr_scope_end(&kids[1]));
        }
    }

    fn check_fn(&mut self, expr: &Expr, list: &List, outer_scope: &mut LinearScope) {
        let kids = children(list);
        if kids.len() < 2 {
            return;
        }

        let params = param_names(&kids[0]);
        let captured = free_vars(&kids[1], &params);
        let mut inner_scope = outer_scope.clone();
        for name in captured {
            if outer_scope.ty(&name).is_some_and(type_expr_contains_tensor) {
                self.read_or_error(name.as_str(), expr, outer_scope);
                outer_scope.consume(
                    &name,
                    ConsumeSite {
                        description: format!("closure capture at offset {}", expr.span().offset),
                    },
                );
                inner_scope.declare(name.clone(), outer_scope.ty(&name).cloned());
            }
        }

        let mut pushed = Vec::new();
        if let Some(params_list) = as_list(&kids[0]) {
            for param in children(params_list) {
                if let Some((name, ty)) = param_name_and_type(param) {
                    inner_scope.declare(name, ty.cloned());
                    pushed.push(name.to_string());
                }
            }
        }
        for param in params {
            if !pushed.iter().any(|p| p == &param) {
                inner_scope.declare(param.clone(), None);
                pushed.push(param);
            }
        }
        if matches!(get_tag_expr(&kids[1]), Some("borrow")) {
            self.invalid_borrow(&kids[1], "borrow cannot be returned from a function");
        } else {
            self.check_expr(&kids[1], &mut inner_scope);
        }
        for name in pushed.into_iter().rev() {
            self.pop_and_check_param(&mut inner_scope, &name, expr_scope_end(&kids[1]));
        }
    }

    fn check_if(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if kids.len() < 3 {
            return;
        }
        self.check_expr(&kids[0], scope);
        let visible_names = scope.visible_names();
        let mut then_scope = scope.clone();
        let mut else_scope = scope.clone();
        self.check_expr(&kids[1], &mut then_scope);
        self.check_expr(&kids[2], &mut else_scope);
        self.join_branch_states(scope, &visible_names, &[then_scope, else_scope]);
    }

    fn check_match(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if kids.is_empty() {
            return;
        }
        if is_var_expr(&kids[0]) && self.expr_is_owned_linear(&kids[0], scope) {
            self.consume_var_expr(
                &kids[0],
                scope,
                ConsumeSite {
                    description: format!("match scrutinee at offset {}", kids[0].span().offset),
                },
            );
        } else {
            self.check_expr(&kids[0], scope);
        }

        let visible_names = scope.visible_names();
        let mut arm_scopes = Vec::new();
        for arm in kids.iter().skip(1) {
            let Expr::List(arm_list, _) = arm else {
                continue;
            };
            if get_tag(arm_list) != Some("arm") {
                continue;
            }
            let arm_kids = children(arm_list);
            if arm_kids.len() < 3 {
                continue;
            }
            let mut arm_scope = scope.clone();
            let pattern_names = pattern_names(&arm_kids[0]);
            for name in &pattern_names {
                arm_scope.declare(name.clone(), None);
            }
            self.check_expr(&arm_kids[1], &mut arm_scope);
            self.check_expr(&arm_kids[2], &mut arm_scope);
            for name in pattern_names.into_iter().rev() {
                self.pop_and_check(&mut arm_scope, &name, expr_scope_end(&arm_kids[2]));
            }
            arm_scopes.push(arm_scope);
        }
        self.join_branch_states(scope, &visible_names, &arm_scopes);
    }

    fn join_branch_states(
        &mut self,
        scope: &mut LinearScope,
        visible_names: &[String],
        branches: &[LinearScope],
    ) {
        for name in visible_names {
            let consumed_site = branches.iter().find_map(|branch| match branch.top(name) {
                Some(BindingState::Consumed(site)) => Some(site.clone()),
                _ => None,
            });
            if let Some(site) = consumed_site
                && matches!(scope.top(name), Some(BindingState::Live { .. }))
            {
                scope.consume(name, site);
            }
        }
    }

    fn maybe_mark_reusable_app_input(&mut self, expr: &Expr, kids: &[Expr], scope: &LinearScope) {
        if !self.expr_is_owned_linear(expr, scope) {
            return;
        }
        let Some(output_ty) = self.expr_type(expr, scope) else {
            return;
        };
        for (index, arg) in kids.iter().skip(1).enumerate() {
            if borrow_inner(arg).is_some()
                || !is_var_expr(arg)
                || !self.expr_is_owned_linear(arg, scope)
            {
                continue;
            }
            if self
                .expr_type(arg, scope)
                .is_some_and(|arg_ty| type_expr_eq(arg_ty, output_ty))
            {
                self.info.mark_reusable_input(expr.span(), index);
                break;
            }
        }
    }

    fn consume_var_expr(&mut self, expr: &Expr, scope: &mut LinearScope, site: ConsumeSite) {
        let Some(name) = var_name(expr) else {
            return;
        };
        if !self.expr_is_owned_linear(expr, scope) {
            return;
        }
        match scope.top(name) {
            Some(BindingState::Live { .. }) => scope.consume(name, site),
            Some(BindingState::Consumed(consumed_at))
                if consumed_at.description.contains("closure capture")
                    || consumed_at.description.contains("match scrutinee") =>
            {
                self.errors.push(CheckError::new(
                    CheckErrorKind::UseAfterConsume,
                    with_macro_provenance(
                        expr,
                        format!(
                            "variable `{name}` was already consumed by {}; later use at offset {} is invalid",
                            consumed_at.description,
                            expr.span().offset
                        ),
                    ),
                    vec![format!(
                        "Structural ownership consumes cannot be auto-copied; move the later use before the consume or copy before the structural consume"
                    )],
                ));
            }
            Some(BindingState::Consumed(_)) => {
                // The implicit-linearity pass will insert a Copy for consuming
                // fan-out. Borrow-after-consume remains an error through
                // `read_or_error`.
            }
            None => {}
        }
    }

    fn read_var_expr(&mut self, expr: &Expr, scope: &mut LinearScope) {
        let Some(name) = var_name(expr) else {
            return;
        };
        if !self.expr_is_owned_linear(expr, scope) {
            return;
        }
        self.read_or_error(name, expr, scope);
        scope.borrow(name, borrow_site(expr));
    }

    fn read_or_error(&mut self, name: &str, expr: &Expr, scope: &LinearScope) {
        let Some(BindingState::Consumed(site)) = scope.top(name) else {
            return;
        };
        // Var-RHS let-bindings (`alias = x`) are aliasing consumes: at the
        // IR level `lower_let` maps `alias` to the same NodeId as `x`
        // (the `Load { name: "x" }` node), so the value is structurally
        // shared, not destroyed. Per `spec/design/implicit_linearity.md`
        // §"Copy Insertion" + "Borrows do not count as fan-out", later
        // borrow-reads (mul, add, matmul, ...) of `x` must succeed: the
        // DAG keeps `x` and `alias` pointing to the same source and the
        // Copy-insertion pass at `crates/chelis-ir/src/lower.rs:364` only
        // forks values reached by multiple `Realize | Drop | Store`
        // consumers. Real consumes (realize/drop/store, app-arg,
        // pipe-stage, closure capture, match scrutinee) remain hard
        // errors here — once a value is truly gone, borrow-reads of it
        // would alias freed storage at runtime.
        //
        // Discrimination is by site description, mirroring the pattern
        // already used in `consume_var_expr` (lines 580-607). A typed
        // `ConsumeKind { Aliasing, Structural }` refactor is a candidate
        // §5 follow-up; see
        // `docs/investigations/var_rhs_let_fanout_diagnosis.md`.
        if site.description.starts_with("binding ") {
            return;
        }
        self.errors.push(CheckError::new(
            CheckErrorKind::UseAfterConsume,
            with_macro_provenance(
                expr,
                format!(
                    "variable `{name}` was already consumed by {}; later use at offset {} is invalid",
                    site.description,
                    expr.span().offset
                ),
            ),
            vec![format!(
                "Insert `copy({name})` before the first consuming use if you need to reuse it"
            )],
        ));
    }

    fn invalid_borrow(&mut self, expr: &Expr, message: &str) {
        self.errors.push(CheckError::new(
            CheckErrorKind::InvalidBorrow,
            with_macro_provenance(expr, format!("{message} (offset {})", expr.span().offset)),
            vec!["Use `&x` only as a direct function-call argument".to_string()],
        ));
    }

    fn pop_and_check(&mut self, scope: &mut LinearScope, name: &str, end_offset: usize) {
        self.pop_and_check_inner(scope, name, end_offset, false);
    }

    fn pop_and_check_param(&mut self, scope: &mut LinearScope, name: &str, end_offset: usize) {
        self.pop_and_check_inner(scope, name, end_offset, true);
    }

    fn pop_and_check_inner(
        &mut self,
        scope: &mut LinearScope,
        name: &str,
        end_offset: usize,
        allow_param_boundary_drop: bool,
    ) {
        let Some((ty, state)) = scope.pop(name) else {
            return;
        };
        if !ty.as_ref().is_some_and(type_expr_is_owned_linear) {
            return;
        }
        let BindingState::Live { borrow_sites } = state else {
            return;
        };
        if allow_param_boundary_drop {
            return;
        }
        let _ = (borrow_sites, end_offset);
        // Unconsumed locals are handled by inserted Drop nodes in lowered IR.
    }

    fn expr_type<'a>(&'a self, expr: &'a Expr, scope: &'a LinearScope) -> Option<&'a Expr> {
        type_metadata(expr).or_else(|| {
            var_name(expr)
                .and_then(|name| scope.ty(name).or_else(|| self.top_level_types.get(name)))
        })
    }

    fn arg_is_borrowed(
        &self,
        func: Option<&Expr>,
        builtin: Option<&str>,
        arg_index: usize,
        scope: &LinearScope,
    ) -> bool {
        if builtin_arg_is_borrowed(builtin, arg_index) {
            return true;
        }
        let Some(func_ty) = func.and_then(|expr| self.expr_type(expr, scope)) else {
            return false;
        };
        type_expr_fn_arg(func_ty, arg_index).is_some_and(type_expr_is_ref)
    }

    fn expr_is_owned_linear(&self, expr: &Expr, scope: &LinearScope) -> bool {
        self.expr_type(expr, scope)
            .is_some_and(type_expr_is_owned_linear)
    }

    fn expr_is_owned_or_borrow_linear(&self, expr: &Expr, scope: &LinearScope) -> bool {
        self.expr_type(expr, scope)
            .is_some_and(type_expr_contains_tensor)
    }
}

fn get_tag(list: &List) -> Option<&str> {
    match list.elements.first() {
        Some(Expr::Atom(Atom::Symbol(tag), _)) => Some(tag.as_str()),
        _ => None,
    }
}

fn with_macro_provenance(expr: &Expr, message: String) -> String {
    let Some(source) = macro_source(expr) else {
        return message;
    };
    format!("{message} (in expansion of {source})")
}

fn macro_source(expr: &Expr) -> Option<String> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let Expr::Map(meta, _) = list.elements.get(1)? else {
        return None;
    };
    let source = meta
        .entries
        .iter()
        .find(|(key, _)| key == "source")
        .map(|(_, value)| value)?;
    let rendered = chelis_deep::printer::print_canonical(std::slice::from_ref(source));
    Some(rendered.replace('\n', " ").trim().to_string())
}

fn get_tag_expr(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::List(list, _) => get_tag(list),
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

fn as_list(expr: &Expr) -> Option<&List> {
    match expr {
        Expr::List(list, _) => Some(list),
        _ => None,
    }
}

fn is_var_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::List(list, _) if get_tag(list) == Some("var"))
}

fn var_name(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn borrow_inner(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("borrow") {
        return None;
    }
    children(list).first()
}

fn param_names(expr: &Expr) -> Vec<String> {
    let Expr::List(list, _) = expr else {
        return Vec::new();
    };
    if get_tag(list) != Some("params") {
        return Vec::new();
    }
    children(list)
        .iter()
        .filter_map(|param| match param {
            Expr::Atom(Atom::Symbol(name), _) => Some(name.clone()),
            Expr::List(param_list, _) => param_list
                .elements
                .first()
                .and_then(symbol_name)
                .map(str::to_string),
            _ => None,
        })
        .collect()
}

fn pattern_names(expr: &Expr) -> Vec<String> {
    let mut names = Vec::new();
    collect_pattern_names(expr, &mut names);
    names
}

fn collect_pattern_names(expr: &Expr, names: &mut Vec<String>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("pat-var") => {
            if let Some(name) = children(list).first().and_then(symbol_name) {
                names.push(name.to_string());
            }
        }
        Some("pat-as") => {
            let kids = children(list);
            if let Some(name) = kids.first().and_then(symbol_name) {
                names.push(name.to_string());
            }
            if let Some(inner) = kids.get(1) {
                collect_pattern_names(inner, names);
            }
        }
        _ => {
            for child in children(list) {
                collect_pattern_names(child, names);
            }
        }
    }
}

fn free_vars(expr: &Expr, params: &[String]) -> Vec<String> {
    let mut bound = vec![params.iter().cloned().collect::<HashSet<_>>()];
    let mut free = HashSet::new();
    collect_free_vars(expr, &mut bound, &mut free);
    free.into_iter().collect()
}

fn collect_free_vars(expr: &Expr, bound: &mut Vec<HashSet<String>>, free: &mut HashSet<String>) {
    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => {}
        Expr::MetaExpr(meta, _) => collect_free_vars(&meta.expr, bound, free),
        Expr::List(list, _) => match get_tag(list) {
            Some("var") => {
                if let Some(name) = children(list).first().and_then(symbol_name)
                    && !bound.iter().rev().any(|scope| scope.contains(name))
                {
                    free.insert(name.to_string());
                }
            }
            Some("fn") => {
                let kids = children(list);
                if kids.len() >= 2 {
                    bound.push(param_names(&kids[0]).into_iter().collect());
                    collect_free_vars(&kids[1], bound, free);
                    bound.pop();
                }
            }
            Some("let") => {
                let kids = children(list);
                if kids.len() < 2 {
                    return;
                }
                let mut let_scope = HashSet::new();
                if let Expr::List(bind_list, _) = &kids[0] {
                    let bind_kids = children(bind_list);
                    let mut index = 0;
                    while index + 1 < bind_kids.len() {
                        collect_free_vars(&bind_kids[index + 1], bound, free);
                        if let Some(name) = symbol_name(&bind_kids[index]) {
                            let_scope.insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                bound.push(let_scope);
                collect_free_vars(&kids[1], bound, free);
                bound.pop();
            }
            Some("match") => {
                let kids = children(list);
                if kids.is_empty() {
                    return;
                }
                collect_free_vars(&kids[0], bound, free);
                for arm in kids.iter().skip(1) {
                    let Expr::List(arm_list, _) = arm else {
                        continue;
                    };
                    if get_tag(arm_list) != Some("arm") {
                        continue;
                    }
                    let arm_kids = children(arm_list);
                    if arm_kids.len() < 3 {
                        continue;
                    }
                    bound.push(pattern_names(&arm_kids[0]).into_iter().collect());
                    collect_free_vars(&arm_kids[1], bound, free);
                    collect_free_vars(&arm_kids[2], bound, free);
                    bound.pop();
                }
            }
            _ => {
                for child in children(list) {
                    collect_free_vars(child, bound, free);
                }
            }
        },
    }
}

fn builtin_arg_is_borrowed(name: Option<&str>, arg_index: usize) -> bool {
    // Tensor→host conversions read the tensor without taking ownership — the
    // runtime implementations (`chelis_list_from_tensor`, `chelis_tensor_to_f64`,
    // `chelis_print_f32`, `chelis_tensor_rank`, `chelis_tensor_shape`,
    // `chelis_tensor_numel`, `tensor_to_string`) all read via the pointer and
    // never call `chelis_free`, so the caller still owns the input afterwards.
    // Keeping these observational avoids forcing callers to sprinkle
    // `copy(x)` before every query or host-lane conversion.
    let Some(name) = name else {
        return false;
    };
    matches!(
        name,
        "add"
            | "mul"
            | "max_elem"
            | "sub"
            | "div"
            | "eq"
            | "neq"
            | "lt"
            | "gt"
            | "lte"
            | "gte"
            | "and"
            | "or"
            | "matmul"
            | "layer_norm"
            | "where"
            | "clamp"
            | "test_assert_close_tensor"
            | "test_assert_eq_tensor_int64"
    ) || matches!(
        (name, arg_index),
        (
            "neg"
                | "exp"
                | "log"
                | "sin"
                | "sqrt"
                | "cos"
                | "tan"
                | "atan"
                | "abs"
                | "floor"
                | "ceil"
                | "uniform_like"
                | "cmplt"
                | "not"
                | "relu"
                | "sigmoid"
                | "softmax"
                | "normalize"
                | "mean"
                | "min_elem"
                | "sum"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
                | "reshape"
                | "permute"
                | "expand"
                | "pad"
                | "shrink"
                | "stride"
                | "dropout"
                | "print"
                | "debug"
                | "to_string"
                | "rank"
                | "shape"
                | "numel"
                | "to_list"
                | "tensor_to_scalar",
            0
        ) | ("conv2d", 0 | 1)
            | ("einsum", 1 | 2)
            | ("split", 0)
            | ("gather", 0 | 1)
            | ("cumsum", 0)
            | ("sort", 0)
            | ("diagonal", 0)
            | ("trace", 0)
    )
}

fn type_metadata(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    match list.elements.get(1) {
        Some(Expr::Map(MetaMap { entries }, _)) => entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        _ => None,
    }
}

fn param_name_and_type(param: &Expr) -> Option<(&str, Option<&Expr>)> {
    match param {
        Expr::Atom(Atom::Symbol(name), _) => Some((name.as_str(), None)),
        Expr::List(param_list, _) => Some((
            param_list.elements.first().and_then(symbol_name)?,
            get_meta(param_list)
                .and_then(|meta| meta.entries.iter().find(|(k, _)| k == "type"))
                .map(|(_, value)| value),
        )),
        _ => None,
    }
}

fn get_meta(list: &List) -> Option<&MetaMap> {
    match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => Some(meta),
        _ => None,
    }
}

fn type_expr_contains_tensor(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    match get_tag(list) {
        Some("t-tensor") => true,
        Some("t-ref") => children(list).iter().any(type_expr_contains_tensor),
        Some("t-tuple") | Some("t-adt") => children(list).iter().any(type_expr_contains_tensor),
        Some("t-fn") => false,
        _ => false,
    }
}

fn type_expr_is_ref(expr: &Expr) -> bool {
    matches!(get_tag_expr(expr), Some("t-ref"))
}

fn type_expr_is_owned_linear(expr: &Expr) -> bool {
    type_expr_contains_tensor(expr) && !type_expr_is_ref(expr)
}

fn type_expr_fn_arg(expr: &Expr, index: usize) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("t-fn") {
        return None;
    }
    let kids = children(list);
    if index >= kids.len().saturating_sub(1) {
        return None;
    }
    kids.get(index)
}

fn type_expr_eq(lhs: &Expr, rhs: &Expr) -> bool {
    chelis_deep::printer::print_canonical(std::slice::from_ref(lhs))
        == chelis_deep::printer::print_canonical(std::slice::from_ref(rhs))
}

fn app_site(expr: &Expr, list: &List) -> ConsumeSite {
    let name = children(list)
        .first()
        .and_then(var_name)
        .map(|name| format!("call to `{name}`"))
        .unwrap_or_else(|| "call".to_string());
    ConsumeSite {
        description: format!("{name} at offset {}", expr.span().offset),
    }
}

fn generic_site(expr: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!("use at offset {}", expr.span().offset),
    }
}

fn realize_site(expr: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!("realize at offset {}", expr.span().offset),
    }
}

fn pipe_site(current: &Expr, stage: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!(
            "pipe into stage at offset {} from offset {}",
            stage.span().offset,
            current.span().offset
        ),
    }
}

fn borrow_site(expr: &Expr) -> String {
    format!("borrow at offset {}", expr.span().offset)
}

fn expr_scope_end(expr: &Expr) -> usize {
    expr.span().offset + expr.span().len
}
