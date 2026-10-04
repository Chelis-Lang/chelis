//! The program tables an evaluation context evaluates against, and the facts
//! it derives from them once.
//!
//! This is its own module for one reason, and the reason is the whole point of
//! the type. A derived fact describes the exact tables it was built from, so a
//! table that can still be written after a fact is cached is how a stale fact
//! happens; chelis#1835 and chelis#1921 are that defect on the host side.
//! `chelis_ir::host::HostLoweringSession` closes it by binding its facts to a
//! borrowed program they cannot outlive. These tables are owned rather than
//! borrowed, so the guarantee has to come from visibility instead: the
//! constructor is the only writer and no accessor hands out a mutable
//! reference.
//!
//! Privacy alone does not deliver that. A private field is visible to the
//! defining module **and every descendant of it**, so while this type lived in
//! `runtime/mod.rs` each of `runtime::eval`, `runtime::named_axis`,
//! `runtime::transforms` and `runtime::invariant` could reach `self.program.defs`
//! through `&mut self` and compile clean. The fields are private to this file,
//! which contains no evaluator code, so those modules are siblings rather than
//! descendants and the write does not compile. That is the part a comment could
//! not supply.
//!
//! To re-verify rather than trust this paragraph, add
//! `self.program.defs.insert(name, body)` to an `impl EvalContext` block in any
//! of those four modules and build the crate. It must fail with E0616, "field
//! `defs` of struct `ProgramScope` is private". A repair that moves this type
//! back into `runtime/mod.rs`, or that adds a `&mut` accessor here, silently
//! restores the defect and that build stops failing.

use std::cell::{OnceCell, RefCell};
use std::rc::Rc;

use chelis_deep::DeepTag;
use chelis_deep::ast::Expr;
use chelis_ir::lower::SubexprLoweringContext;
use chelis_unord::{UnordMap, UnordSet};

thread_local! {
    /// Terminal-name index builds on this thread (chelis#2393): one per
    /// scope that resolves a short name, never one per ask.
    static TERMINAL_INDEX_BUILDS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn record_terminal_index_build() {
    TERMINAL_INDEX_BUILDS.with(|builds| builds.set(builds.get() + 1));
}

/// Terminal-name index builds on this thread since the last call.
#[cfg(test)]
pub(super) fn take_terminal_index_builds() -> u64 {
    TERMINAL_INDEX_BUILDS.with(|builds| builds.replace(0))
}

/// The lazily built index of [`ProgramScope::resolve_def_key`].
pub(super) type TerminalIndexCell = std::sync::OnceLock<UnordMap<String, Vec<String>>>;

pub(super) struct ProgramScope {
    /// Every top-level definition in scope, library and new code, registered
    /// once by `register_top_level_defs` while the context is built.
    /// Shared with every other evaluation of the same prepared program
    /// (chelis#3144).
    defs: std::sync::Arc<UnordMap<String, Expr>>,
    /// Combined library + new-code Deep type-env. Threaded into
    /// subexpression lowering when the host runtime hits a `grad` / `vmap`
    /// form or routes a named-axis call, so the lowerer resolves free names
    /// the same way the C backend does. Empty when no library context is
    /// present (e.g. unit tests that don't need transform support).
    type_env: std::sync::Arc<UnordMap<String, Expr>>,
    /// The subexpression lowering context named-axis routing lowers through
    /// (chelis#2207). Preparing one folds the pipes in every definition, so
    /// before this a routed reduction re-folded the whole program, all of
    /// `chelis-std` included, on every call.
    routing_lowering_context: OnceCell<SubexprLoweringContext>,
    /// The lowering context `grad` and `vmap` applications lower through
    /// (chelis#2439). It also reads the checked program and the declared
    /// signatures, which are fixed for the evaluation context that owns this
    /// scope, so its first application builds it and the rest reuse it.
    transform_lowering_context: OnceCell<SubexprLoweringContext>,
    /// Every definition key grouped by its terminal segment, each group
    /// sorted (chelis#2393). A short or import-qualified reference resolves
    /// through one group instead of sorting and scanning the whole linked
    /// program on every ask, which the evaluator made up to four times per
    /// application.
    /// Shared with every other evaluation of the same prepared program, like
    /// the definitions it indexes (chelis#3144).
    terminal_index: std::sync::Arc<TerminalIndexCell>,
    /// [`Self::reached_by_call`] per spelling, derived once: a kernel applied
    /// in a loop walks its body once, not once per application.
    reached_by_call: RefCell<UnordMap<String, Rc<[String]>>>,
}

impl ProgramScope {
    pub(super) fn new(defs: UnordMap<String, Expr>, type_env: UnordMap<String, Expr>) -> Self {
        Self::shared(
            std::sync::Arc::new(defs),
            std::sync::Arc::new(type_env),
            std::sync::Arc::default(),
        )
    }

    /// A scope over definitions and a type environment that a prepared
    /// program derived once for all of its evaluations (chelis#3144).
    pub(super) fn shared(
        defs: std::sync::Arc<UnordMap<String, Expr>>,
        type_env: std::sync::Arc<UnordMap<String, Expr>>,
        terminal_index: std::sync::Arc<TerminalIndexCell>,
    ) -> Self {
        Self {
            defs,
            type_env,
            routing_lowering_context: OnceCell::new(),
            transform_lowering_context: OnceCell::new(),
            terminal_index,
            reached_by_call: RefCell::new(UnordMap::new()),
        }
    }

    pub(super) fn defs(&self) -> &UnordMap<String, Expr> {
        &self.defs
    }

    pub(super) fn type_env(&self) -> &UnordMap<String, Expr> {
        &self.type_env
    }

    /// The definition key a reference spelled `name` resolves to: the exact
    /// key when one exists, otherwise the one key whose terminal segment is
    /// `name`'s (`host_ops::terminal_name_matches`), and `None` when there is
    /// no such key or more than one.
    pub(super) fn resolve_def_key<'s>(&'s self, name: &'s str) -> Option<&'s str> {
        if let Some((key, _)) = self.defs.get_key_value(name) {
            return Some(key);
        }
        let index = self.terminal_index.get_or_init(|| {
            record_terminal_index_build();
            let mut index = UnordMap::<String, Vec<String>>::new();
            for (key, _) in self.defs.to_sorted() {
                index
                    .entry(super::host_ops::terminal_name(key).to_owned())
                    .or_default()
                    .push(key.clone());
            }
            index
        });
        match index.get(super::host_ops::terminal_name(name))?.as_slice() {
            [key] => Some(key),
            _ => None,
        }
    }

    /// The value declarations a function's body reaches when it runs, in the
    /// order it reaches them (spec/03 §4.4: a binding's initializer is
    /// evaluated whether or not the binding is read, at the point evaluation
    /// reaches the binding). The host hands a body to a tensor DAG when it
    /// runs it as a kernel, a transform target or a routed named-axis call,
    /// and that DAG never demands a dead binding's value, so the caller
    /// initializes these first, where its sequential order reaches the body.
    ///
    /// Reached means on every path through the body: argument, binding and
    /// block order, into a called top-level function's body at its call (as
    /// lowering inlines it), whether it is called by its own name or through
    /// a local that aliases it (`g = f` then `g(x)`), into an applied pipe
    /// stage and into an applied `grad` or `vmap` target. A lambda that is
    /// not applied, an `if` or `match` arm, and a name `masked` reports as
    /// lexically bound by the caller reach nothing. A name resolves as a
    /// read of it resolves ([`Self::resolve_def_key`]); an input declaration
    /// (`x: T = x`) has no initializer, so it is never listed.
    ///
    /// This form applies an inline `fn` whose free names `masked` may bind.
    pub(super) fn reached_by_applying(
        &self,
        function: &Expr,
        masked: &dyn Fn(&str) -> bool,
    ) -> Vec<String> {
        let mut walk = ReachWalk::new(self, masked);
        walk.function(function);
        walk.reached
    }

    /// [`Self::reached_by_applying`] for a call of the top-level function
    /// `name`, whose free names are all declaration scope.
    pub(super) fn reached_by_call(&self, name: &str) -> Rc<[String]> {
        if let Some(reached) = self.reached_by_call.borrow().get(name) {
            return reached.clone();
        }
        let unmasked = |_: &str| false;
        let mut walk = ReachWalk::new(self, &unmasked);
        walk.call(name);
        let reached: Rc<[String]> = walk.reached.into();
        self.reached_by_call
            .borrow_mut()
            .insert(name.to_owned(), reached.clone());
        reached
    }

    /// Whether `body` is a bare read of the declaration `key` itself, the
    /// shape of an input declaration the checker admitted.
    fn reads_itself(&self, key: &str, body: &Expr) -> bool {
        let mut current = body;
        loop {
            match current {
                Expr::MetaExpr(meta, _) => current = &meta.expr,
                Expr::Node(node, _) => {
                    return node.tag() == DeepTag::Var
                        && node
                            .children_slice()
                            .first()
                            .and_then(super::symbol_name)
                            .and_then(|name| self.resolve_def_key(name))
                            == Some(key);
                }
                _ => return false,
            }
        }
    }

    /// The transform lowering context, built by `build` on first use and
    /// reused for the scope's lifetime (chelis#2439).
    pub(super) fn transform_lowering_context(
        &self,
        build: impl FnOnce() -> SubexprLoweringContext,
    ) -> SubexprLoweringContext {
        self.transform_lowering_context.get_or_init(build).clone()
    }

    /// The lowering context named-axis routing uses, built on first use and
    /// reused for the scope's lifetime (chelis#2207).
    ///
    /// This is the context `chelis_ir::lower::try_lower_subexpr_program`
    /// builds for itself from the same two tables, so routing through it
    /// lowers exactly as that entry does.
    pub(super) fn routing_lowering_context(&self) -> SubexprLoweringContext {
        self.routing_lowering_context
            .get_or_init(|| {
                SubexprLoweringContext::over_program(
                    self.type_env.as_ref().clone(),
                    self.defs.as_ref().clone(),
                )
            })
            .clone()
    }
}

/// One [`ProgramScope::reached_by_applying`] walk: the lexical scopes
/// mirror `chelis_types::linearity::free_runtime_variables`, so a name the
/// walk treats as free is free there too.
struct ReachWalk<'s> {
    scope: &'s ProgramScope,
    masked: &'s dyn Fn(&str) -> bool,
    /// The lexical scopes, innermost last. Each bound name maps to the
    /// top-level definition its binding aliases when the binding's value is
    /// a bare name of one (`g = f`), so applying the local runs it.
    bound: Vec<UnordMap<String, Option<String>>>,
    /// Top-level functions whose bodies this walk entered; a recursive call
    /// reaches nothing its first entry did not.
    entered_functions: UnordSet<String>,
    listed: UnordSet<String>,
    reached: Vec<String>,
}

impl<'s> ReachWalk<'s> {
    fn new(scope: &'s ProgramScope, masked: &'s dyn Fn(&str) -> bool) -> Self {
        Self {
            scope,
            masked,
            bound: Vec::new(),
            entered_functions: UnordSet::new(),
            listed: UnordSet::new(),
            reached: Vec::new(),
        }
    }

    /// Apply a `fn`: its parameters are bound over the current scope (empty
    /// for a top-level function, see [`Self::call`]) and its body runs.
    fn function(&mut self, function: &Expr) {
        let Some((DeepTag::Fn, children)) = super::tagged_expr_children(peel(function)) else {
            return;
        };
        let [params, body, ..] = children else {
            return;
        };
        let names = super::tagged_expr_children(params)
            .map(|(_, params)| {
                params
                    .iter()
                    .filter_map(|param| super::transforms::runtime_param_parts(param))
                    .map(|(name, _)| (name.to_owned(), None))
                    .collect::<UnordMap<_, _>>()
            })
            .unwrap_or_default();
        self.bound.push(names);
        self.expr(body);
        self.bound.pop();
    }

    /// The top-level declaration a free `name` reads, if any.
    fn declaration(&self, name: &str) -> Option<(String, &'s Expr)> {
        if self.bound.iter().any(|scope| scope.contains_key(name)) || (self.masked)(name) {
            return None;
        }
        let scope: &'s ProgramScope = self.scope;
        let key = scope.resolve_def_key(name)?;
        Some((key.to_owned(), scope.defs.get(key)?))
    }

    /// The top-level definition `name` names here: the one a local binding
    /// aliases when `name` is lexically bound (none when the binding is not
    /// a bare name of one), nothing when `masked` binds it, and otherwise the
    /// declaration it resolves to.
    fn definition_named(&self, name: &str) -> Option<String> {
        match self.bound.iter().rev().find_map(|scope| scope.get(name)) {
            Some(alias) => alias.clone(),
            None if (self.masked)(name) => None,
            None => self.scope.resolve_def_key(name).map(str::to_owned),
        }
    }

    /// Run the top-level function `name` names, by its own name or through
    /// a local alias of it, as a call of it does.
    fn call(&mut self, name: &str) {
        if let Some(key) = self.definition_named(name) {
            self.enter(key);
        }
    }

    /// Run the top-level definition `key`, following bare aliases
    /// (`alias = entry`), whose names resolve in declaration scope.
    fn enter(&mut self, mut key: String) {
        let scope: &'s ProgramScope = self.scope;
        for _ in 0..scope.defs.len() {
            let Some(body) = scope.defs.get(&key) else {
                return;
            };
            if let Some(alias) = var_name(body) {
                match scope.resolve_def_key(alias) {
                    Some(next) if next != key => {
                        key = next.to_owned();
                        continue;
                    }
                    _ => return,
                }
            }
            if !self.entered_functions.insert(key) {
                return;
            }
            let outer = std::mem::take(&mut self.bound);
            self.function(body);
            self.bound = outer;
            return;
        }
    }

    /// Run a function-valued expression that is being applied: a top-level
    /// function by name, or an inline `fn`.
    fn applied(&mut self, function: &Expr) {
        let function = peel(function);
        if let Some(name) = var_name(function) {
            self.call(name);
        } else if let Some((DeepTag::Fn, _)) = super::tagged_expr_children(function) {
            self.function(function);
        }
    }

    fn read(&mut self, name: &str) {
        let Some((key, body)) = self.declaration(name) else {
            return;
        };
        match super::tagged_expr_children(peel(body)) {
            // A bare reference to a nullary definition applies it.
            Some((DeepTag::Fn, [params, ..]))
                if super::tagged_expr_children(params)
                    .is_some_and(|(_, params)| params.is_empty()) =>
            {
                self.call(name);
            }
            Some((DeepTag::Fn, _)) => {}
            _ => {
                if !self.scope.reads_itself(&key, body) && self.listed.insert(key.clone()) {
                    self.reached.push(key);
                }
            }
        }
    }

    fn expr(&mut self, expr: &Expr) {
        let Some((tag, children)) = super::tagged_expr_children(peel(expr)) else {
            return;
        };
        match tag {
            DeepTag::Var => {
                if let Some(name) = children.first().and_then(super::symbol_name) {
                    self.read(name);
                }
            }
            // A lambda runs only where it is applied.
            DeepTag::Fn => {}
            // Only the condition or scrutinee runs on every path.
            DeepTag::If | DeepTag::Match => {
                if let Some(first) = children.first() {
                    self.expr(first);
                }
            }
            DeepTag::Let => {
                let [binds, body, ..] = children else {
                    return;
                };
                let mut names = UnordMap::new();
                if let Some((DeepTag::Bind, pairs)) = super::tagged_expr_children(binds) {
                    for pair in pairs.chunks(2) {
                        if let [name, value] = pair {
                            self.expr(value);
                            if let Some(name) = super::symbol_name(name) {
                                let alias =
                                    var_name(value).and_then(|value| self.definition_named(value));
                                names.insert(name.to_owned(), alias);
                            }
                        }
                    }
                }
                self.bound.push(names);
                self.expr(body);
                self.bound.pop();
            }
            DeepTag::App => {
                for child in children {
                    self.expr(child);
                }
                let Some(callee) = children.first().map(peel) else {
                    return;
                };
                if let Some(name) = var_name(callee) {
                    self.call(name);
                } else if let Some((DeepTag::Grad | DeepTag::Vmap, transform)) =
                    super::tagged_expr_children(callee)
                    && let Some(target) = transform.first()
                {
                    // An applied `grad(f)` or `vmap(f)` runs its target.
                    self.applied(target);
                }
            }
            // Each stage after the first is applied to the running value.
            DeepTag::Pipe => {
                for (index, child) in children.iter().enumerate() {
                    self.expr(child);
                    if index == 0 {
                        continue;
                    }
                    self.applied(child);
                }
            }
            _ => {
                for child in children {
                    self.expr(child);
                }
            }
        }
    }
}

fn peel(expr: &Expr) -> &Expr {
    let mut current = expr;
    while let Expr::MetaExpr(meta, _) = current {
        current = &meta.expr;
    }
    current
}

fn var_name(expr: &Expr) -> Option<&str> {
    match super::tagged_expr_children(peel(expr)) {
        Some((DeepTag::Var, children)) => children.first().and_then(super::symbol_name),
        _ => None,
    }
}
