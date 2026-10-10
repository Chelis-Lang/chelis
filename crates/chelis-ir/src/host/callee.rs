//! The identity a host call's callee resolves to (chelis#3484).
//!
//! Host lowering resolves each callee spelling exactly once, here, against
//! the lexical scope at the call. Every later stage reads the resulting
//! [`HostCallee`] and never resolves the spelling again. Its representation
//! is private to this module, so other code obtains one only from
//! [`resolve`], [`resolve_value`] or a named constructor whose documentation
//! says why the name it is given cannot be a lexical binding at the call.

use super::{HostTypeTerm, UnordMap, host_lexically_binds, is_host_unresolved_marker};
use std::fmt;

/// The callee a call or named callback resolved to during host lowering.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct HostCallee(Identity);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Identity {
    Function(String),
    Local(String),
    NativeProvider(String),
    Unresolved(String),
}

/// A read-only view of a [`HostCallee`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostCalleeView<'c> {
    /// The host program's function of this name.
    Function(&'c str),
    /// The lexical binding of this name visible at the call: a parameter,
    /// let, pattern or callback binder that holds a function value.
    Local(&'c str),
    /// A native provider entry point, by its generated symbol.
    NativeProvider(&'c str),
    /// A callee host lowering could not resolve, spelled by one of the
    /// unspellable unresolved markers. The C ABI projection rejects it.
    Unresolved(&'c str),
}

impl HostCallee {
    pub fn view(&self) -> HostCalleeView<'_> {
        match &self.0 {
            Identity::Function(name) => HostCalleeView::Function(name),
            Identity::Local(name) => HostCalleeView::Local(name),
            Identity::NativeProvider(symbol) => HostCalleeView::NativeProvider(symbol),
            Identity::Unresolved(marker) => HostCalleeView::Unresolved(marker),
        }
    }

    /// The resolved callee's spelling within its own namespace.
    pub fn name(&self) -> &str {
        match &self.0 {
            Identity::Function(name)
            | Identity::Local(name)
            | Identity::NativeProvider(name)
            | Identity::Unresolved(name) => name,
        }
    }

    /// The call a manifest root's driver makes. The driver is a generated
    /// global binding with no lexical scope of its own, and `function` is
    /// the def the root names, so no lexical binding can hold the name.
    pub fn root_driver(function: String) -> Self {
        Self(Identity::Function(function))
    }

    /// A callee host lowering could not resolve. The marker is unspellable,
    /// so it names nothing, and the C ABI projection rejects the call.
    pub fn unresolved_callable() -> Self {
        Self(Identity::Unresolved(
            super::HOST_UNRESOLVED_CALLABLE_MARKER.to_string(),
        ))
    }

    /// The function a staged callable alias names. The staged plan resolved
    /// the alias at its binding position, through its captures, then the
    /// lexical scope, then the def table, so `function` is a def's name, not
    /// a binding that the rewritten use site may see.
    pub(super) fn staged_alias_target(function: String) -> Self {
        Self(Identity::Function(function))
    }
}

#[cfg(test)]
impl HostCallee {
    /// A unit-test fixture's call of the program function `name`.
    pub(super) fn test_function(name: &str) -> Self {
        Self(Identity::Function(name.to_string()))
    }

    /// A unit-test fixture's call through the lexical binding `name`.
    pub(super) fn test_local(name: &str) -> Self {
        Self(Identity::Local(name.to_string()))
    }
}

impl fmt::Debug for HostCallee {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

pub(super) enum HostAppCallee<'s> {
    Lexical(LexicalCallee<'s>),
    Program(ProgramCallee),
}

/// A lexical binding: it shadows every definition and builtin of its
/// spelling.
pub(super) struct LexicalCallee<'s> {
    name: String,
    binding: &'s HostTypeTerm,
}

impl<'s> LexicalCallee<'s> {
    pub(super) fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn binding(&self) -> &'s HostTypeTerm {
        self.binding
    }

    /// The call through this binding.
    pub(super) fn callee(&self) -> HostCallee {
        HostCallee(Identity::Local(self.name.clone()))
    }
}

/// A callee spelling that no lexical binding shadows. Only [`resolve`] and
/// [`resolve_value`] make one, so a branch that takes it cannot consult a
/// definition for a lexically bound name.
pub(super) struct ProgramCallee(String);

impl ProgramCallee {
    pub(super) fn name(&self) -> &str {
        &self.0
    }

    /// The call of the program function this spelling names, or the call of
    /// an unresolved marker.
    pub(super) fn callee(&self) -> HostCallee {
        if is_host_unresolved_marker(&self.0) {
            HostCallee(Identity::Unresolved(self.0.clone()))
        } else {
            HostCallee(Identity::Function(self.0.clone()))
        }
    }

    /// The call of this function's monomorphized specialization `symbol`.
    pub(super) fn monomorphized(&self, symbol: String) -> HostCallee {
        HostCallee(Identity::Function(symbol))
    }

    /// The native provider entry point `symbol` that implements this
    /// function.
    pub(super) fn native_provider(&self, symbol: String) -> HostCallee {
        HostCallee(Identity::NativeProvider(symbol))
    }
}

/// Resolve an applied callee head.
pub(super) fn resolve(name: String, scope: &UnordMap<String, HostTypeTerm>) -> HostAppCallee<'_> {
    // spec/01-nomenclature.md section 3.2: an applied uppercase head is a
    // constructor head, even where a single-letter value binding shares its
    // spelling.
    let constructor_head = name.starts_with(|first: char| first.is_ascii_uppercase());
    if constructor_head {
        return HostAppCallee::Program(ProgramCallee(name));
    }
    resolve_value(name, scope)
}

/// Resolve a callee named in value position, such as a loop callback. A
/// value reference to a single-letter uppercase binding is that binding
/// (spec/01-nomenclature.md section 3.2).
pub(super) fn resolve_value(
    name: String,
    scope: &UnordMap<String, HostTypeTerm>,
) -> HostAppCallee<'_> {
    match scope.get(&name) {
        Some(binding) if host_lexically_binds(scope, &name) => {
            HostAppCallee::Lexical(LexicalCallee { name, binding })
        }
        _ => HostAppCallee::Program(ProgramCallee(name)),
    }
}
