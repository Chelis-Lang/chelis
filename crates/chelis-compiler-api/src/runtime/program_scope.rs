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

use std::cell::OnceCell;

use chelis_deep::DeepTag;
use chelis_deep::ast::Expr;
use chelis_ir::lower::SubexprLoweringContext;
use chelis_types::linearity::free_runtime_variables;
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

pub(super) struct ProgramScope {
    /// Every top-level definition in scope, library and new code, registered
    /// once by `register_top_level_defs` while the context is built.
    defs: UnordMap<String, Expr>,
    /// Combined library + new-code Deep type-env. Threaded into
    /// subexpression lowering when the host runtime hits a `grad` / `vmap`
    /// form or routes a named-axis call, so the lowerer resolves free names
    /// the same way the C backend does. Empty when no library context is
    /// present (e.g. unit tests that don't need transform support).
    type_env: UnordMap<String, Expr>,
    /// The subexpression lowering context named-axis routing lowers through
    /// (chelis#2207). Preparing one folds the pipes in every definition, so
    /// before this a routed reduction re-folded the whole program, all of
    /// `chelis-std` included, on every call.
    routing_lowering_context: OnceCell<SubexprLoweringContext>,
    /// Every definition key grouped by its terminal segment, each group
    /// sorted (chelis#2393). A short or import-qualified reference resolves
    /// through one group instead of sorting and scanning the whole linked
    /// program on every ask, which the evaluator made up to four times per
    /// application.
    terminal_index: OnceCell<UnordMap<String, Vec<String>>>,
}

impl ProgramScope {
    pub(super) fn new(defs: UnordMap<String, Expr>, type_env: UnordMap<String, Expr>) -> Self {
        Self {
            defs,
            type_env,
            routing_lowering_context: OnceCell::new(),
            terminal_index: OnceCell::new(),
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

    /// The value declarations an evaluation that runs the declarations
    /// `roots` initializes (spec/03 §4.4): every value declaration a root's
    /// body names, or that a declaration so named names in turn, whether or
    /// not anything reads it. A function contributes what its own body names
    /// whether or not it is applied, as `chelis_ir::Dag::entered_declarations`
    /// traverses a lowered program. A name resolves as a read of it resolves
    /// ([`Self::resolve_def_key`]), so a name that is no declaration names
    /// nothing. An input declaration (`x: T = x`) has no initializer to run:
    /// its value is the caller's, so it is never initialized here.
    pub(super) fn entered_value_declarations<'r>(
        &self,
        roots: impl IntoIterator<Item = &'r str>,
    ) -> UnordSet<String> {
        let mut entered = UnordSet::new();
        let mut visited = UnordSet::new();
        let mut pending = Vec::new();
        for root in roots {
            if let Some(key) = self.resolve_def_key(root).map(str::to_owned)
                && visited.insert(key.clone())
            {
                pending.push(key);
            }
        }
        while let Some(declaration) = pending.pop() {
            let Some(body) = self.defs.get(&declaration) else {
                continue;
            };
            for name in free_runtime_variables(body) {
                let Some(key) = self.resolve_def_key(&name).map(str::to_owned) else {
                    continue;
                };
                if !visited.insert(key.clone()) {
                    continue;
                }
                let Some(referenced) = self.defs.get(&key) else {
                    continue;
                };
                let is_function = super::tagged_expr_children(referenced)
                    .is_some_and(|(tag, _)| tag == DeepTag::Fn);
                if !is_function && !self.reads_itself(&key, referenced) {
                    entered.insert(key.clone());
                }
                pending.push(key);
            }
        }
        entered
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

    /// The lowering context named-axis routing uses, built on first use and
    /// reused for the scope's lifetime (chelis#2207).
    ///
    /// This is the context `chelis_ir::lower::try_lower_subexpr_program`
    /// builds for itself from the same two tables, so routing through it
    /// lowers exactly as that entry does.
    pub(super) fn routing_lowering_context(&self) -> SubexprLoweringContext {
        self.routing_lowering_context
            .get_or_init(|| {
                SubexprLoweringContext::over_program(self.type_env.clone(), self.defs.clone())
            })
            .clone()
    }
}
