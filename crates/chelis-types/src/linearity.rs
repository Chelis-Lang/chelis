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

    /// Merge two linearity infos into one. Used by
    /// [`crate::CheckedProgram::compose`] to reconstitute a whole-program
    /// linearity map from a cached library half plus a freshly-checked
    /// new-code half. `self` (the library half) wins on an offset clash;
    /// in practice the two halves carry disjoint span offsets because
    /// they come from separately-parsed source regions.
    pub fn merged_with(&self, other: &LinearityInfo) -> LinearityInfo {
        let mut reusable_inputs_by_offset = self.reusable_inputs_by_offset.clone();
        for (offset, input_index) in &other.reusable_inputs_by_offset {
            reusable_inputs_by_offset
                .entry(*offset)
                .or_insert(*input_index);
        }
        LinearityInfo {
            reusable_inputs_by_offset,
        }
    }
}

#[derive(Debug, Clone)]
enum BindingState {
    Live { borrow_sites: Vec<String> },
    Consumed(ConsumeSite),
}

/// Discrimination axis on a consume site (Linearity-F1).
///
/// Replaces the string-prefix check on `ConsumeSite::description`
/// (formerly at `read_or_error`) with a typed field. Phase 0 spec
/// lock (`docs/design/compiler_cleanup_0_7_8_spec_lock.md` Contract 1)
/// pins this as two variants; tuple-destructure tmp bindings are
/// handled as `Aliasing` (for the `let __chelis_tmp = (var ...)`
/// shape) or `Structural` (for the `(tuple-get ...)` reads) by the
/// same rules as any other consume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConsumeKind {
    /// `let alias = x` and similar var-RHS bindings.  At the IR
    /// level `lower_let` maps `alias` to the same NodeId as `x`,
    /// so the value is structurally shared rather than destroyed.
    /// Later borrow-reads of `x` must succeed.  See
    /// `spec/design/implicit_linearity.md` "Copy Insertion" and
    /// "Borrows do not count as fan-out".
    Aliasing,
    /// Every other consume site: realize / drop / store, app-arg,
    /// pipe-stage, closure capture, match scrutinee, etc.  The
    /// value is gone after this point and later borrow-reads are
    /// `UseAfterConsume`.
    Structural,
}

#[derive(Debug, Clone)]
struct ConsumeSite {
    description: String,
    kind: ConsumeKind,
}

#[derive(Debug, Clone, Default)]
struct LinearScope {
    bindings: HashMap<String, Vec<BindingState>>,
    types: HashMap<String, Vec<Option<Expr>>>,
    /// Alias chain map (Linearity-AliasedConsume-F1).  When
    /// `check_let` records a `let y = (var x)` binding as an
    /// `Aliasing` consume, it also pushes `x` onto the alias stack
    /// at `aliases["y"]`.  When `consume_var_expr` resolves a
    /// `Structural` consume on `y` it walks the alias chain via
    /// `resolve_alias_chain` and forwards the consume to the
    /// underlying source name's scope entry.  Multi-level chains
    /// (`let z = y; let y = x`) are walked iteratively until the
    /// resolved name has no alias.  Stacks parallel `bindings` so
    /// scoping (`pop`) keeps the alias map consistent with the
    /// shadowing semantics already in place for names and types.
    aliases: HashMap<String, Vec<Option<String>>>,
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
        self.types.entry(name.clone()).or_default().push(ty);
        // Push a `None` onto the alias stack so a re-let of `name`
        // with a non-var RHS shadows any prior alias.  Callers that
        // record an alias must follow with `record_alias` to flip
        // the top of the stack from `None` to `Some(source)`.
        self.aliases.entry(name).or_default().push(None);
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

        if let Some(stack) = self.aliases.get_mut(name) {
            stack.pop();
            if stack.is_empty() {
                self.aliases.remove(name);
            }
        }

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

    /// Record that the top-of-stack binding for `alias` is an
    /// aliasing copy of `source`.  Must be called after `declare`
    /// for `alias` (the alias stack carries a `None` at the top
    /// after `declare`; this flips it to `Some(source)`).  Used by
    /// the `Aliasing` consume producers in `check_let` and
    /// `check_def_body` per Linearity-AliasedConsume-F1.
    fn record_alias(&mut self, alias: &str, source: &str) {
        if let Some(stack) = self.aliases.get_mut(alias)
            && let Some(top) = stack.last_mut()
        {
            *top = Some(source.to_string());
        }
    }

    /// Walk the alias chain for `name` to the underlying non-alias
    /// source.  Returns `None` if `name` is not an alias today;
    /// returns `Some(source)` if `name` aliases `source` (possibly
    /// through one or more intermediate names).  Bounded by chain
    /// length, which is bounded by source-program nesting depth.
    ///
    /// Cycles are guarded against by a visited set; an alias chain
    /// that closes a cycle is treated as terminating at the first
    /// re-visited node (defensive guard; the desugarer should never
    /// produce a cycle in practice).
    fn resolve_alias_chain(&self, name: &str) -> Option<String> {
        let mut current = name.to_string();
        let mut visited: HashSet<String> = HashSet::new();
        let mut walked = false;
        loop {
            if !visited.insert(current.clone()) {
                return None;
            }
            match self
                .aliases
                .get(&current)
                .and_then(|stack| stack.last())
                .and_then(|entry| entry.as_ref())
            {
                Some(source) => {
                    current = source.clone();
                    walked = true;
                }
                None => return if walked { Some(current) } else { None },
            }
        }
    }
}

struct Checker {
    errors: Vec<CheckError>,
    info: LinearityInfo,
    top_level_types: HashMap<String, Expr>,
    /// Names of ADTs whose definitions (transitively) carry a tensor
    /// field. Computed once per `check_linearity` call by walking
    /// `deftype` declarations in `annotated_exprs`. Used by
    /// `expr_is_owned_or_borrow_linear` so `&adt_value` is accepted as
    /// a borrow whenever the ADT's definition contains a tensor, not
    /// only when the ADT's type *arguments* contain one. Resolves the
    /// downstream blocker for `School` P1.5 (BatchNorm) and P2.5
    /// (optimizer `_step_tree`) where `&BatchNormParams` /
    /// `&AdamState[tensor[..]]` (with the tensor in a record field,
    /// not the ADT-arg position) was rejected with `InvalidBorrow`.
    ///
    /// Keyed on bare ADT name. Two-name collisions are rejected up
    /// front by `collect_declarations` in `infer.rs` with
    /// `CheckErrorKind::DuplicateDefinition`, so by the time the
    /// linearity checker runs, every name in this set corresponds to
    /// exactly one `deftype`. That guarantee is what makes a bare
    /// `String` key safe here; without it the carrier set would be
    /// order-dependent (last-write-wins via `HashMap::insert` in
    /// `compute_tensor_carrying_adts`). Once Chelis gains qualified
    /// ADT names, this set should migrate to a `Set<AdtId>` queried
    /// off the shared `AdtRegistry` instead of reparsing `deftype`
    /// exprs here. See the function-level note on
    /// [`compute_tensor_carrying_adts`].
    tensor_carrying_adts: HashSet<String>,
    /// Depth counter for desugarer-synthesized destructure scopes
    /// (`__chelis_tmp_N` bind chains tagged with `destructure: true`
    /// in their meta-map). Incremented by `check_let` when entering
    /// a destructure-marked bind and decremented on return. The
    /// `consume_var_expr` already-consumed arm uses this to gate
    /// the Linearity-F2 use-after-consume diagnostic: implicit
    /// Copy insertion does not apply to destructured components
    /// (tuple-get produces a fresh owned value, not an aliased
    /// borrow), so double-consume on a destructured name is a
    /// hard error rather than the silent fallthrough used by
    /// regular bindings.
    destructure_scope_depth: usize,
}

impl Checker {
    fn push_diagnostic(&mut self, error: CheckError) {
        // Linearity-F2 / F3 W2 cascade close: every linearity
        // violation is an error.  W1 PR 1's destructure-cascade
        // warning channel closed once the corpus survey confirmed
        // zero surfaced warnings (see
        // `docs/investigations/linearity_destructure_cleanup_survey.md`).
        self.errors.push(error);
    }
}

/// Pre-declare top-level def names into `scope`, descending through
/// `(module {} name children...)` wrappers. Mirrors the
/// `top_level_decl_items` pattern in `infer.rs:1056-1074` so module-
/// wrapped defs participate in cross-statement linearity tracking.
fn pre_declare_top_level_defs(
    exprs: &[Expr],
    type_env: &HashMap<String, Expr>,
    scope: &mut LinearScope,
) {
    for expr in exprs {
        pre_declare_one(expr, type_env, scope);
    }
}

fn pre_declare_one(expr: &Expr, type_env: &HashMap<String, Expr>, scope: &mut LinearScope) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("module") => {
            // `(module {} name children...)` — skip tag, meta, name.
            for child in list.elements.iter().skip(3) {
                pre_declare_one(child, type_env, scope);
            }
        }
        Some("def") => {
            if let Some(name) = children(list).first().and_then(symbol_name) {
                scope.declare(name, type_env.get(name).cloned());
            }
        }
        _ => {}
    }
}

pub fn check_linearity(program: &CheckedProgram) -> Result<CheckedProgram, Vec<CheckError>> {
    let tensor_carrying_adts = compute_tensor_carrying_adts(program.annotated_exprs());
    let mut checker = Checker {
        errors: Vec::new(),
        info: LinearityInfo::default(),
        top_level_types: program.type_env().clone(),
        tensor_carrying_adts,
        destructure_scope_depth: 0,
    };
    let mut scope = LinearScope::default();

    pre_declare_top_level_defs(program.annotated_exprs(), program.type_env(), &mut scope);

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
    //
    // The carrier set comes from BOTH library and new code, chained
    // into one `compute_tensor_carrying_adts` call so the fixed-point
    // sees every ADT at once. Library decls can introduce tensor-
    // carrying ADTs that new-code borrows; new-code can introduce
    // additional ones whose fields reference library ADTs (the
    // cross-package transitive case). Two independent calls — one
    // per half, each with its own local set — would miss any new-
    // code ADT whose carrying status depends on a library ADT, even
    // though both halves are eventually unioned.
    //
    // ALWAYS-RECOMPUTE INVARIANT: this helper is recomputed from the
    // chained iterator on every call, with no per-call state held by
    // `Checker`, the library `CheckedProgram`, or any global cache.
    // The fixed-point bound (`O(adt_count)` passes, each `O(adt_count
    // * field_count)`) is small relative to the per-expression
    // linearity walk that follows. Future change risk: if a later
    // refactor caches `tensor_carrying_adts` per `CheckedProgram` and
    // composes the library's cached set with a fresh new-code pass,
    // the fixed-point will not re-resolve new-code ADTs whose
    // carrying status depends on library ADTs and borrow semantics
    // will silently desync. The locking test for this contract lives
    // at `tests/linearity_with_context.rs::
    // check_linearity_with_context_is_pure_across_repeated_calls`.
    // Once `CheckedProgram` exposes a shared `AdtRegistry`, the
    // registry query replaces this helper entirely (and the new
    // call site is responsible for re-establishing the same union-
    // and-recompute discipline).
    let tensor_carrying_adts = compute_tensor_carrying_adts(
        library_program
            .annotated_exprs()
            .iter()
            .chain(new_program.annotated_exprs().iter()),
    );
    let mut checker = Checker {
        errors: Vec::new(),
        info: LinearityInfo::default(),
        top_level_types: new_program.type_env().clone(),
        tensor_carrying_adts,
        destructure_scope_depth: 0,
    };

    let mut scope = LinearScope::default();

    // Pre-declare library def names so `(var libname)` references in
    // new code resolve to a Live binding with the library's function
    // type. Function types are non-linear (no tensor content), so a
    // pure reference never triggers consumption of new-code locals.
    // Recurse through module wrappers so library .ch sources written
    // with `module Foo` participate in pre-declaration the same as
    // bare-top-level library sources (Linearity-F3 PR 1).
    pre_declare_top_level_defs(
        library_program.annotated_exprs(),
        new_program.type_env(),
        &mut scope,
    );

    // Pre-declare new-code def names. New-code shadows library on
    // collision (declare last → top of stack wins).
    pre_declare_top_level_defs(
        new_program.annotated_exprs(),
        new_program.type_env(),
        &mut scope,
    );

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
        // Linearity-F3 PR 1 + PR 2: recurse through `(module {} name
        // children...)` wrappers so module-wrapped top-level defs
        // participate in cross-statement linearity tracking. PR 1
        // routed diagnostics raised inside this recursion to a
        // warning channel for a deprecation window; PR 2 removed that
        // channel and unified the severity with bare-top-level
        // violations, so all `push_diagnostic` calls route to
        // `Checker::errors`.
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some("module")
        {
            // Skip tag, meta, name — walk every remaining child as a
            // top-level expression.
            for child in list.elements.iter().skip(3) {
                self.check_top_level(child, scope);
            }
            return;
        }
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
                            kind: ConsumeKind::Aliasing,
                        },
                    );
                    if let Some(source) = var_name(body) {
                        scope.record_alias(name, source);
                    }
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
                // `tuple-get(t, i)` is a read of `t`, not a
                // consume.  The implicit-linearity IR pass inserts a
                // Copy where needed.  Pre Linearity-F2 the
                // distinction was invisible because destructure tmp
                // bindings were untyped and the consume was skipped
                // via `expr_is_owned_linear`; now that the type
                // metadata threads onto the tmps, `(var __chelis_tmpN)`
                // would otherwise consume through `generic_site` and
                // forward (via the alias chain) to the underlying
                // tuple source.  Treat the var argument as a borrow.
                Some("tuple-get") => self.check_tuple_get(list, scope),
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

    /// `tuple-get(t, i)` reads the i-th element of `t` without
    /// consuming `t`.  Treat the var argument as a borrow read so
    /// the surrounding destructure desugar of `let (a, b) = pair`
    /// (which produces two tuple-gets on the same source tmp) does
    /// not double-consume.  Linearity-F2: this method is added here
    /// because the destructure type-metadata threading made the
    /// previously-untyped tmp consumes visible.
    fn check_tuple_get(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if let Some(target) = kids.first() {
            if is_var_expr(target) && self.expr_is_owned_linear(target, scope) {
                self.read_var_expr(target, scope);
            } else {
                self.check_expr(target, scope);
            }
        }
        // Walk remaining children (the index lit) so any nested
        // expressions inside the index are still checked.
        for child in kids.iter().skip(1) {
            self.check_expr(child, scope);
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

    // Issue #226 diagnosis (regression introduced indirectly by #183
    // `fix(types): substitute ADT type params into record-pattern
    // bindings`). Before #183 the destructured record-field bindings
    // (`PosEmbedParams { table: table }` -> a new local `table`) carried
    // a fresh type variable in the annotated Deep, so
    // `expr_is_owned_linear` returned `false` and `check_pipe`
    // accidentally accepted `table |> shape(0)` followed by a later
    // `gather(table, ...)`. #183 stamps the resolved field type onto
    // pattern bindings (the right fix in isolation); that exposed a
    // pre-existing gap in `check_pipe`. The branch below classifies the
    // piped value with `var_name(stage)`, which only matches the bare-
    // var stage shape `(var f)`. For any non-bare-var stage
    // `chelis_surf::desugar::desugar_pipe_stage` emits a synthesized
    // `(fn (params __chelis_pipe) (app callee ... (var __chelis_pipe)
    // ...))` lambda, so `var_name(stage)` returns `None` and
    // `arg_is_borrowed(stage, None, 0, scope)` falls through to a
    // function-type lookup on the LAMBDA itself, not on the inner
    // `callee`. That means borrow-arg builtins called with explicit
    // arguments (`shape(0)`, `add(y)`, `mul(k)`, `matmul(w)`, ...) get
    // mis-tagged as structural consumes of the piped variable, tripping
    // `UseAfterConsume` on any later read with a malformed "pipe into
    // stage at offset 0 from offset 0" message (both offsets zero
    // because the synthesized lambda has no source span). The upcoming
    // fix introduces `resolve_pipe_stage_callee` and peers through the
    // synthesized lambda to recover the inner callee and the piped
    // value's arg position before consulting `arg_is_borrowed`.
    fn check_pipe(&mut self, list: &List, scope: &mut LinearScope) {
        let kids = children(list);
        if kids.is_empty() {
            return;
        }
        let mut current = &kids[0];
        for stage in &kids[1..] {
            // Pipe stages with explicit args (e.g. `x |> shape(0)`)
            // are desugared by `chelis_surf::desugar::desugar_pipe_stage`
            // into a synthesized one-arg lambda
            // `(fn (params __chelis_pipe) (app callee ... (var __chelis_pipe) ...))`
            // where the piped value lands at the `__chelis_pipe`
            // position in the inner app's args. To classify whether the
            // piped value is borrowed (read) or consumed by the stage,
            // look through that lambda and ask the inner callee at its
            // actual arg position. Without this peering, every
            // non-bare-var stage falls through to the "stage callee
            // unknown" branch and the piped value is treated as a
            // structural consume — which mis-fires whenever a borrow-
            // arg builtin (`shape`, `add`, `mul`, ...) is invoked with
            // explicit non-piped args. Closes issue #226.
            let (callee_expr, callee_builtin, piped_arg_index) = resolve_pipe_stage_callee(stage);
            if self.arg_is_borrowed(callee_expr, callee_builtin, piped_arg_index, scope) {
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
        // Linearity-F2 destructure-scope tracking.  When the bind
        // is one of the desugarer-synthesized destructure
        // intermediates (`__chelis_tmp_N` or a user-visible
        // destructure component, tagged with `destructure: true`),
        // bump the destructure-scope depth so the
        // `consume_var_expr` already-consumed arm fires as an
        // error rather than the silent fallthrough used by regular
        // bindings.
        let bind_introduces_destructure = match &kids[0] {
            Expr::List(bind_list, _) => bind_introduces_destructure_tmp(bind_list),
            _ => false,
        };
        if bind_introduces_destructure {
            self.destructure_scope_depth += 1;
        }
        if let Expr::List(bind_list, _) = &kids[0] {
            let bind_kids = children(bind_list);
            let mut index = 0;
            while index + 1 < bind_kids.len() {
                let Some(name) = symbol_name(&bind_kids[index]) else {
                    index += 2;
                    continue;
                };
                let value = &bind_kids[index + 1];
                let mut alias_source: Option<String> = None;
                if is_var_expr(value) && self.expr_is_owned_linear(value, scope) {
                    alias_source = var_name(value).map(str::to_string);
                    self.consume_var_expr(
                        value,
                        scope,
                        ConsumeSite {
                            description: format!(
                                "binding `{name}` at offset {}",
                                value.span().offset
                            ),
                            kind: ConsumeKind::Aliasing,
                        },
                    );
                } else if matches!(get_tag_expr(value), Some("borrow")) {
                    self.invalid_borrow(value, "borrow cannot be stored in a binding");
                } else {
                    self.check_expr(value, scope);
                }
                scope.declare(name, self.expr_type(value, scope).cloned());
                if let Some(source) = alias_source {
                    scope.record_alias(name, &source);
                }
                pushed.push(name.to_string());
                index += 2;
            }
        }
        self.check_expr(&kids[1], scope);
        for name in pushed.into_iter().rev() {
            self.pop_and_check(scope, &name, expr_scope_end(&kids[1]));
        }
        if bind_introduces_destructure {
            self.destructure_scope_depth -= 1;
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
            if outer_scope
                .ty(&name)
                .is_some_and(|ty| type_expr_contains_tensor(ty, &self.tensor_carrying_adts))
            {
                self.read_or_error(name.as_str(), expr, outer_scope);
                outer_scope.consume(
                    &name,
                    ConsumeSite {
                        description: format!("closure capture at offset {}", expr.span().offset),
                        kind: ConsumeKind::Structural,
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
                    kind: ConsumeKind::Structural,
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
            let pattern_bindings = pattern_named_types(&arm_kids[0]);
            let pattern_names: Vec<String> = pattern_bindings
                .iter()
                .map(|(name, _)| name.clone())
                .collect();
            for (name, ty) in &pattern_bindings {
                arm_scope.declare(name.clone(), ty.clone());
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
        // Linearity-AliasedConsume-F1: a `Structural` consume on an
        // aliased name forwards to the underlying source name's
        // scope entry, so a later borrow of the source trips
        // `read_or_error` correctly.  `Aliasing` consumes do not
        // forward; they stay pinned to the alias's own entry because
        // the alias bind itself is what introduces the aliasing
        // relationship in the IR.
        let target: String = match site.kind {
            ConsumeKind::Structural => scope
                .resolve_alias_chain(name)
                .unwrap_or_else(|| name.to_string()),
            ConsumeKind::Aliasing => name.to_string(),
        };
        match scope.top(&target) {
            Some(BindingState::Live { .. }) => scope.consume(&target, site),
            Some(BindingState::Consumed(consumed_at))
                if matches!(consumed_at.kind, ConsumeKind::Structural)
                    && (consumed_at.description.contains("closure capture")
                        || consumed_at.description.contains("match scrutinee")) =>
            {
                let description = consumed_at.description.clone();
                self.push_diagnostic(CheckError::new(
                    CheckErrorKind::UseAfterConsume,
                    with_macro_provenance(
                        expr,
                        format!(
                            "variable `{target}` was already consumed by {description}; later use at offset {} is invalid",
                            expr.span().offset
                        ),
                    ),
                    vec![format!(
                        "Structural ownership consumes cannot be auto-copied; move the later use before the consume or copy before the structural consume"
                    )],
                ));
            }
            Some(BindingState::Consumed(consumed_at))
                if matches!(consumed_at.kind, ConsumeKind::Aliasing)
                    && matches!(site.kind, ConsumeKind::Structural) =>
            {
                // The target was previously aliased (e.g. `let y = x`
                // recorded an `Aliasing` consume on `x`); a later
                // `Structural` consume forwarded through the alias
                // chain replaces the aliasing record so subsequent
                // borrows of the target trip `read_or_error`.
                scope.consume(&target, site);
            }
            Some(BindingState::Consumed(consumed_at)) if self.destructure_scope_depth > 0 => {
                // Linearity-F2: inside a destructure-let scope a
                // consume-after-consume on a destructured component
                // is an error.  Implicit Copy insertion does not
                // apply because tuple-get produces a fresh owned
                // value rather than an aliased borrow, so reuse of
                // a destructured tensor name must be made explicit
                // via `copy()`.  Outside the destructure scope the
                // implicit-linearity pass inserts a Copy for
                // consuming fan-out, matching the spec's
                // "Copy Insertion" semantics; per the existing
                // baseline we do not flag that shape.
                let description = consumed_at.description.clone();
                self.push_diagnostic(CheckError::new(
                    CheckErrorKind::UseAfterConsume,
                    with_macro_provenance(
                        expr,
                        format!(
                            "variable `{name}` (from a destructured binding) was already consumed by {description}; later use at offset {} is invalid",
                            expr.span().offset
                        ),
                    ),
                    vec![format!(
                        "Insert `copy({name})` before the first consuming use if you need to reuse it"
                    )],
                ));
            }
            Some(BindingState::Consumed(_)) => {
                // The implicit-linearity pass will insert a Copy for
                // consuming fan-out.  Borrow-after-consume remains an
                // error through `read_or_error`.
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
        // Linearity-AliasedConsume-F1: when `name` is an alias, the
        // structural consume on it would have forwarded to the
        // underlying source (see `consume_var_expr`).  Check the
        // alias chain's terminal source first so borrows of either
        // the alias or the source surface the violation
        // symmetrically.
        let resolved = scope
            .resolve_alias_chain(name)
            .unwrap_or_else(|| name.to_string());
        let Some(BindingState::Consumed(site)) = scope.top(&resolved) else {
            return;
        };
        // Var-RHS let-bindings (`alias = x`) are `ConsumeKind::Aliasing`
        // consumes: at the IR level `lower_let` maps `alias` to the
        // same NodeId as `x` (the `Load { name: "x" }` node), so the
        // value is structurally shared, not destroyed.  Per
        // `spec/design/implicit_linearity.md` §"Copy Insertion" and
        // "Borrows do not count as fan-out", later borrow-reads
        // (`mul`, `add`, `matmul`, ...) of `x` must succeed: the DAG
        // keeps `x` and `alias` pointing to the same source and the
        // Copy-insertion pass at `crates/chelis-ir/src/lower.rs:364`
        // only forks values reached by multiple `Realize | Drop |
        // Store` consumers.  Real consumes (realize / drop / store,
        // app-arg, pipe-stage, closure capture, match scrutinee)
        // remain hard errors here: once a value is truly gone,
        // borrow-reads of it would alias freed storage at runtime.
        //
        // Linearity-F1 (`docs/gap_synthesis.md`) replaced the prior
        // string-prefix check on the description with this typed
        // discrimination via `ConsumeKind`.  The description text
        // stays for diagnostic rendering only.
        if matches!(site.kind, ConsumeKind::Aliasing) {
            return;
        }
        let description = site.description.clone();
        let message = with_macro_provenance(
            expr,
            format!(
                "variable `{name}` was already consumed by {}; later use at offset {} is invalid",
                description,
                expr.span().offset
            ),
        );
        let suggestion =
            format!("Insert `copy({name})` before the first consuming use if you need to reuse it");
        self.push_diagnostic(CheckError::new(
            CheckErrorKind::UseAfterConsume,
            message,
            vec![suggestion],
        ));
    }

    fn invalid_borrow(&mut self, expr: &Expr, message: &str) {
        let formatted =
            with_macro_provenance(expr, format!("{message} (offset {})", expr.span().offset));
        self.push_diagnostic(CheckError::new(
            CheckErrorKind::InvalidBorrow,
            formatted,
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
        if !ty
            .as_ref()
            .is_some_and(|ty| type_expr_is_owned_linear(ty, &self.tensor_carrying_adts))
        {
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
                .or_else(|| {
                    // Linearity-F2: `(tuple-get (var t) i)` does not
                    // carry `:type` metadata because the `tuple-get`
                    // tag is in `should_attach_type_metadata`'s deny
                    // list.  Recover the element type by inspecting
                    // the underlying tuple var's `:type` and indexing
                    // into its `t-tuple` children.  Used by the
                    // destructure-let desugar path where the
                    // synthesized tmp bind's value is the unannotated
                    // tuple-get.
                    tuple_get_element_type(expr, scope, self)
                })
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
        if func.is_some_and(callee_is_observational_higher_order) {
            // Implicit-copy fan-out v3 Shape B: `grad(f, ...)(args)` and
            // `vmap(f, ...)(args)` call sites are observational at the
            // linearity level.  Lowering at
            // `chelis-ir::lower::lower_grad_callable_with_nodes` synthesizes
            // a fresh closure body around Load nodes for every arg and
            // splices the caller's NodeId in without destroying it; the
            // forward and backward (or batched) DAGs both read from the
            // same Load.  Classifying every arg position as a borrow lets
            // the linearity checker permit fan-out across multiple grad or
            // vmap calls of the same arg followed by later borrow-reads,
            // matching the semantics already implemented in the IR.
            return true;
        }
        let Some(func_ty) = func.and_then(|expr| self.expr_type(expr, scope)) else {
            return false;
        };
        type_expr_fn_arg(func_ty, arg_index).is_some_and(type_expr_is_ref)
    }

    fn expr_is_owned_linear(&self, expr: &Expr, scope: &LinearScope) -> bool {
        self.expr_type(expr, scope)
            .is_some_and(|ty| type_expr_is_owned_linear(ty, &self.tensor_carrying_adts))
    }

    fn expr_is_owned_or_borrow_linear(&self, expr: &Expr, scope: &LinearScope) -> bool {
        self.expr_type(expr, scope)
            .is_some_and(|ty| type_expr_contains_tensor(ty, &self.tensor_carrying_adts))
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

/// Resolve the effective callee of a pipe stage and the arg position
/// the piped value occupies in that callee.
///
/// Two stage shapes are produced by `chelis_surf::desugar::desugar_pipe_stage`:
///
/// 1. Bare var stage (e.g. `x |> f`): `(var f)`. The piped value is the
///    only arg, at position 0. Returns `(stage, Some("f"), 0)`.
///
/// 2. Synthesized lambda stage (e.g. `x |> f(y)`): a one-arg lambda
///    `(fn (params __chelis_pipe) (app callee ... (var __chelis_pipe) ...))`
///    where the piped value lands at whatever position the desugarer
///    placed `__chelis_pipe` in the inner app's args. Today
///    `desugar_pipe_stage` always inserts the pipe param at position 0,
///    but we scan the arg list so this stays robust against future
///    desugarer shapes (e.g. `cast(type)` post-stage forms where the
///    piped value lands at a non-zero arg index). Returns
///    `(inner_callee_expr, builtin_name_or_none, piped_arg_index)`.
///
/// Any other stage shape (parser-level oddity, macro-expanded result we
/// have not classified) falls back to the historical contract of
/// `var_name(stage)` so we never regress the bare-var case.
fn resolve_pipe_stage_callee(stage: &Expr) -> (Option<&Expr>, Option<&str>, usize) {
    if let Some(name) = var_name(stage) {
        return (Some(stage), Some(name), 0);
    }
    if let Some((callee, idx)) = pipe_lambda_callee_and_pipe_arg_index(stage) {
        return (Some(callee), var_name(callee), idx);
    }
    (Some(stage), None, 0)
}

/// If `stage` is `(fn (params __chelis_pipe) (app callee args...))` with
/// exactly one of `args` being `(var __chelis_pipe)`, return the inner
/// callee and the index of the pipe arg within `args`.
fn pipe_lambda_callee_and_pipe_arg_index(stage: &Expr) -> Option<(&Expr, usize)> {
    let Expr::List(list, _) = stage else {
        return None;
    };
    if get_tag(list) != Some("fn") {
        return None;
    }
    let kids = children(list);
    let params = kids.first()?;
    let body = kids.get(1)?;
    let pipe_param = sole_pipe_param_name(params)?;
    let Expr::List(body_list, _) = body else {
        return None;
    };
    if get_tag(body_list) != Some("app") {
        return None;
    }
    let app_kids = children(body_list);
    let callee = app_kids.first()?;
    for (offset, arg) in app_kids.iter().skip(1).enumerate() {
        if var_name(arg) == Some(pipe_param) {
            return Some((callee, offset));
        }
    }
    None
}

/// If `params` is `(params __chelis_pipe[N])` with exactly one
/// pipe-synthesized parameter, return its name. The desugarer prefixes
/// every synthetic pipe parameter with `__chelis_pipe`; a user-written
/// lambda that happens to take a single param does not match.
fn sole_pipe_param_name(params: &Expr) -> Option<&str> {
    let Expr::List(list, _) = params else {
        return None;
    };
    if get_tag(list) != Some("params") {
        return None;
    }
    let kids = children(list);
    if kids.len() != 1 {
        return None;
    }
    let name = symbol_name(&kids[0])?;
    if name.starts_with("__chelis_pipe") {
        Some(name)
    } else {
        None
    }
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

/// True when `expr` is a `(grad ...)`, `(vmap ...)`, or nested
/// composition thereof.  Used by `arg_is_borrowed` to mark grad-app and
/// vmap-app call sites as observational at the linearity level.
fn callee_is_observational_higher_order(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    matches!(get_tag(list), Some("grad") | Some("vmap"))
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

/// Like [`pattern_names`], but also returns each binding's resolved
/// type expression when the inferencer stamped one onto the pattern
/// node's metadata. Used by `check_match` to populate arm `LinearScope`
/// entries with their concrete types — required for destructured
/// fields whose type comes from the scrutinee's ADT instantiation
/// rather than a `let`-style RHS. (closes #181)
fn pattern_named_types(expr: &Expr) -> Vec<(String, Option<Expr>)> {
    let mut bindings = Vec::new();
    collect_pattern_named_types(expr, &mut bindings);
    bindings
}

fn collect_pattern_named_types(expr: &Expr, bindings: &mut Vec<(String, Option<Expr>)>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("pat-var") => {
            if let Some(name) = children(list).first().and_then(symbol_name) {
                bindings.push((name.to_string(), type_metadata(expr).cloned()));
            }
        }
        Some("pat-as") => {
            let kids = children(list);
            if let Some(name) = kids.first().and_then(symbol_name) {
                bindings.push((name.to_string(), type_metadata(expr).cloned()));
            }
            if let Some(inner) = kids.get(1) {
                collect_pattern_named_types(inner, bindings);
            }
        }
        _ => {
            for child in children(list) {
                collect_pattern_named_types(child, bindings);
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
                | "recip"
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

/// Linearity-F2 helper: resolve the element type of a
/// `(tuple-get (var t) i)` expression by looking up the tuple var
/// `t`'s scope-type and indexing into its `t-tuple` children.
/// Returns `None` if any part of the lookup fails.  Used by
/// `Checker::expr_type` when the synthesized destructure-tmp bind
/// value is an unannotated tuple-get.
fn tuple_get_element_type<'a>(
    expr: &'a Expr,
    scope: &'a LinearScope,
    checker: &'a Checker,
) -> Option<&'a Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("tuple-get") {
        return None;
    }
    let kids = children(list);
    let tuple_expr = kids.first()?;
    let tuple_ty = type_metadata(tuple_expr).or_else(|| {
        var_name(tuple_expr)
            .and_then(|name| scope.ty(name).or_else(|| checker.top_level_types.get(name)))
    })?;
    let index_expr = kids.get(1)?;
    let index = match index_expr {
        Expr::List(idx_list, _) if get_tag(idx_list) == Some("lit") => {
            children(idx_list).first().and_then(|child| match child {
                Expr::Atom(Atom::Int(n), _) => Some(*n as usize),
                _ => None,
            })
        }
        Expr::Atom(Atom::Int(n), _) => Some(*n as usize),
        _ => None,
    }?;
    let Expr::List(tuple_ty_list, _) = tuple_ty else {
        return None;
    };
    if get_tag(tuple_ty_list) != Some("t-tuple") {
        return None;
    }
    let tys = children(tuple_ty_list);
    tys.get(index)
}

/// Linearity-F2 destructure-scope gate.  Returns `true` if `bind_list`
/// is a `(bind {meta} name value ...)` whose meta-map contains the
/// `destructure: true` marker injected by `chelis_surf::desugar`
/// when synthesizing the `__chelis_tmp_N` intermediates for
/// `let (a, b) = ...` patterns.  Used by `Checker::check_let` to
/// bump `destructure_scope_depth`, which gates the
/// `consume_var_expr` already-consumed arm so use-after-consume on
/// a destructured component surfaces as an error rather than the
/// silent fallthrough used by regular bindings (where implicit
/// Copy insertion covers consuming fan-out).
fn bind_introduces_destructure_tmp(bind_list: &List) -> bool {
    let Some(meta) = get_meta(bind_list) else {
        return false;
    };
    meta.entries.iter().any(|(key, value)| {
        key == "destructure" && matches!(value, Expr::Atom(Atom::Bool(true), _))
    })
}

fn type_expr_contains_tensor(expr: &Expr, tensor_carrying_adts: &HashSet<String>) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    match get_tag(list) {
        Some("t-tensor") => true,
        Some("t-ref") => children(list)
            .iter()
            .any(|c| type_expr_contains_tensor(c, tensor_carrying_adts)),
        Some("t-tuple") => children(list)
            .iter()
            .any(|c| type_expr_contains_tensor(c, tensor_carrying_adts)),
        Some("t-adt") => {
            // An ADT is tensor-carrying if EITHER one of its type
            // arguments is (the original behavior — e.g. `Wrapper[a]`
            // where `a` is `tensor[..]`), OR the ADT's own definition
            // has a variant with a tensor-carrying field (the
            // chelis#153 fix for `&BatchNormParams { weight: tensor[..],
            // ... }`). The pre-computed set in `tensor_carrying_adts`
            // already accounts for transitive ADT-field tensor-carry.
            let name_carries = children(list)
                .first()
                .and_then(symbol_name)
                .is_some_and(|n| tensor_carrying_adts.contains(n));
            name_carries
                || children(list)
                    .iter()
                    .skip(1) // skip the name; only check type args
                    .any(|c| type_expr_contains_tensor(c, tensor_carrying_adts))
        }
        Some("t-fn") => false,
        _ => false,
    }
}

fn type_expr_is_ref(expr: &Expr) -> bool {
    matches!(get_tag_expr(expr), Some("t-ref"))
}

fn type_expr_is_owned_linear(expr: &Expr, tensor_carrying_adts: &HashSet<String>) -> bool {
    type_expr_contains_tensor(expr, tensor_carrying_adts) && !type_expr_is_ref(expr)
}

/// Walk top-level declarations and return the set of ADT names whose
/// definitions (transitively) carry a tensor field. Used by the
/// linearity checker to recognize `&MyParams` as a valid borrow when
/// `MyParams` is a record with a `tensor[...]` field — previously the
/// `t-adt` arm of `type_expr_contains_tensor` only inspected the ADT's
/// type *arguments*, missing tensor fields declared in the variant.
///
/// # Caller contract
///
/// The caller MUST pass every `deftype` whose name might appear
/// (transitively) in the field type of any other `deftype` in the
/// same call. The fixed point converges only over the ADTs visible
/// in `exprs`: an unrelated-library ADT whose carrier status would
/// flip a new-code ADT into the result is invisible if the library
/// half is not chained in. `check_linearity_with_context` enforces
/// this by chaining `library_program.annotated_exprs()` with
/// `new_program.annotated_exprs()` in one call; any future caller
/// (e.g. an incremental `AdtRegistry`-backed query) must preserve
/// the same union-and-recompute discipline or the carrier set will
/// silently desync from the cross-package borrow rules. The locking
/// regression test is at `tests/linearity_with_context.rs::
/// check_linearity_with_context_is_pure_across_repeated_calls`.
///
/// # Complexity
///
/// Fixed-point iteration handles ADTs whose fields reference other
/// ADTs (the standard "is this type transitively tensor-carrying?"
/// graph walk). Each pass scans every (adt, field) pair; the loop
/// terminates after at most `O(adt_count)` passes (one per
/// transitive layer), giving a worst-case bound of
/// `O(adt_count^2 * fields_per_adt)`. In practice 2-3 passes suffice
/// on the existing corpus, so the cost is small relative to the
/// per-expression linearity walk. The function is recomputed on
/// every `check_linearity` / `check_linearity_with_context` call —
/// notably including REPL-driven re-evaluations (`chelis surf`,
/// `chelis eval`). When that cost becomes load-bearing, the
/// migration trigger is exposure of a shared `AdtRegistry` query on
/// `CheckedProgram`: replace this helper with a registry lookup of
/// variant-field types, keeping the same union-and-recompute
/// invariant on the lookup side.
///
/// Typealiases (`typealias`) are NOT walked here. The inference layer
/// owns alias resolution — see `resolve_type_aliases` in `infer.rs`
/// (exercised by `typealias_zero_param_resolves_in_defsig` and
/// siblings) — and any case where a `typealias` name reaches the
/// linearity checker still wearing a `(t-adt {} Alias ...)` shape is
/// a bug in inference, not in this carrier set. If/when the linearity
/// checker gains direct access to a shared `AdtRegistry`, this helper
/// retires in favor of querying that registry's variant-field types
/// (which already know about aliases too).
fn compute_tensor_carrying_adts<'a, I>(exprs: I) -> HashSet<String>
where
    I: IntoIterator<Item = &'a Expr>,
{
    // Step 1: collect every (adt_name, field_type_exprs) pair from
    // `(deftype {} Name (params?) (variant {} VariantName [field_or_tyarg]...)...)`
    // declarations, descending through `(module {} name ...)` wrappers.
    //
    // The caller decides what to include: a single program passes its
    // own `annotated_exprs()`; the with-context entry chains library
    // and new-code so cross-package field references (a new-code ADT
    // wrapping a library tensor-carrying ADT) are resolved by the same
    // fixed-point pass instead of two independent ones.
    let mut adt_field_types: HashMap<String, Vec<Expr>> = HashMap::new();
    fn collect(expr: &Expr, out: &mut HashMap<String, Vec<Expr>>) {
        let Expr::List(list, _) = expr else {
            return;
        };
        let tag = get_tag(list);
        match tag {
            Some("module") => {
                // `(module {} name body...)` — `children()` skips tag
                // and meta, leaving `[name, body...]`; skip the name
                // for the same shape the `deftype` branch below uses.
                for child in children(list).iter().skip(1) {
                    collect(child, out);
                }
            }
            Some("deftype") => {
                let kids = children(list);
                let Some(name) = kids.first().and_then(symbol_name) else {
                    return;
                };
                let mut field_tys: Vec<Expr> = Vec::new();
                for child in kids.iter().skip(1) {
                    let Expr::List(inner, _) = child else {
                        continue;
                    };
                    if get_tag(inner) != Some("variant") {
                        continue;
                    }
                    // variant children: name, then either `(field name ty)`
                    // entries (record-style) or bare type exprs (positional).
                    for v in children(inner).iter().skip(1) {
                        match v {
                            Expr::List(vlist, _) if get_tag(vlist) == Some("field") => {
                                if let Some(ty) = children(vlist).get(1) {
                                    field_tys.push(ty.clone());
                                }
                            }
                            other => field_tys.push(other.clone()),
                        }
                    }
                }
                out.insert(name.to_string(), field_tys);
            }
            _ => {}
        }
    }
    for expr in exprs {
        collect(expr, &mut adt_field_types);
    }

    // Step 2: fixed-point iteration. An ADT is tensor-carrying iff any
    // of its field types contains a tensor (looking up other ADTs in
    // the current set). Reuses `type_expr_contains_tensor` against
    // the in-progress carrier set, so the recursive `t-adt` lookup
    // walks the same code path used at check time. Stop when a pass
    // adds no new names; bounded by the ADT count.
    let mut carriers: HashSet<String> = HashSet::new();
    loop {
        let mut grew = false;
        for (name, field_tys) in &adt_field_types {
            if carriers.contains(name) {
                continue;
            }
            if field_tys
                .iter()
                .any(|ty| type_expr_contains_tensor(ty, &carriers))
            {
                carriers.insert(name.clone());
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    carriers
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
        kind: ConsumeKind::Structural,
    }
}

fn generic_site(expr: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!("use at offset {}", expr.span().offset),
        kind: ConsumeKind::Structural,
    }
}

fn realize_site(expr: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!("realize at offset {}", expr.span().offset),
        kind: ConsumeKind::Structural,
    }
}

fn pipe_site(current: &Expr, stage: &Expr) -> ConsumeSite {
    ConsumeSite {
        description: format!(
            "pipe into stage at offset {} from offset {}",
            stage.span().offset,
            current.span().offset
        ),
        kind: ConsumeKind::Structural,
    }
}

fn borrow_site(expr: &Expr) -> String {
    format!("borrow at offset {}", expr.span().offset)
}

fn expr_scope_end(expr: &Expr) -> usize {
    expr.span().offset + expr.span().len
}

#[cfg(test)]
mod tests {
    //! Unit tests for the private linearity helpers. Locks the contract
    //! `pattern_named_types` must uphold for `check_match` to populate
    //! arm scopes with the correct binding types after the issue #181
    //! substitution fix.

    use super::*;
    use chelis_deep::Span;
    use chelis_deep::ast::{Atom, Expr, List, MetaMap};

    fn span() -> Span {
        Span::new(0, 0)
    }

    fn sym(name: &str) -> Expr {
        Expr::Atom(Atom::Symbol(name.to_string()), span())
    }

    fn meta(entries: Vec<(&str, Expr)>) -> Expr {
        Expr::Map(
            MetaMap {
                entries: entries
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            },
            span(),
        )
    }

    /// Build `(tag {meta} children...)`.
    fn node(tag: &str, meta_entries: Vec<(&str, Expr)>, children: Vec<Expr>) -> Expr {
        let mut elements = vec![sym(tag), meta(meta_entries)];
        elements.extend(children);
        Expr::List(List { elements }, span())
    }

    /// Build a synthetic `(t-tensor {} (d-lit 4) (t-prim f32))` so the
    /// tests can assert metadata is the exact `Expr` we stamped.
    fn tensor_4_f32() -> Expr {
        node(
            "t-tensor",
            vec![],
            vec![
                node("d-lit", vec![], vec![Expr::Atom(Atom::Int(4), span())]),
                node("t-prim", vec![], vec![sym("f32")]),
            ],
        )
    }

    #[test]
    fn pat_var_with_type_metadata_returns_some_ty() {
        // (pat-var {type: <ty>} x)
        let pat = node("pat-var", vec![("type", tensor_4_f32())], vec![sym("x")]);
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, "x");
        assert_eq!(bindings[0].1, Some(tensor_4_f32()));
    }

    #[test]
    fn pat_var_without_type_metadata_returns_none() {
        // (pat-var {} y)
        let pat = node("pat-var", vec![], vec![sym("y")]);
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings, vec![("y".to_string(), None)]);
    }

    #[test]
    fn pat_tuple_walks_into_children() {
        // (pat-tuple {} (pat-var {type:<ty>} a) (pat-var {} b))
        let pat = node(
            "pat-tuple",
            vec![],
            vec![
                node("pat-var", vec![("type", tensor_4_f32())], vec![sym("a")]),
                node("pat-var", vec![], vec![sym("b")]),
            ],
        );
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0], ("a".to_string(), Some(tensor_4_f32())));
        assert_eq!(bindings[1], ("b".to_string(), None));
    }

    #[test]
    fn pat_record_walks_into_kv_children() {
        // (pat-record {} FooState (kv {} x (pat-var {type:<ty>} x)))
        let pat = node(
            "pat-record",
            vec![],
            vec![
                sym("FooState"),
                node(
                    "kv",
                    vec![],
                    vec![
                        sym("x"),
                        node("pat-var", vec![("type", tensor_4_f32())], vec![sym("x")]),
                    ],
                ),
            ],
        );
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings, vec![("x".to_string(), Some(tensor_4_f32()))]);
    }

    #[test]
    fn pat_ctor_walks_into_positional_subpatterns() {
        // (pat-ctor {} Some (pat-var {type:<ty>} v))
        let pat = node(
            "pat-ctor",
            vec![],
            vec![
                sym("Some"),
                node("pat-var", vec![("type", tensor_4_f32())], vec![sym("v")]),
            ],
        );
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings, vec![("v".to_string(), Some(tensor_4_f32()))]);
    }

    #[test]
    fn pat_as_returns_outer_and_inner_bindings() {
        // (pat-as {type:<ty>} whole (pat-var {} x))
        let pat = node(
            "pat-as",
            vec![("type", tensor_4_f32())],
            vec![sym("whole"), node("pat-var", vec![], vec![sym("x")])],
        );
        let bindings = pattern_named_types(&pat);
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0], ("whole".to_string(), Some(tensor_4_f32())));
        assert_eq!(bindings[1], ("x".to_string(), None));
    }

    #[test]
    fn pat_wild_returns_no_bindings() {
        // (pat-wild {})
        let pat = node("pat-wild", vec![], vec![]);
        let bindings = pattern_named_types(&pat);
        assert!(bindings.is_empty());
    }
}
