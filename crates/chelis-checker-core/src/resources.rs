//! Explicit resources for one checker operation.

use crate::prelude::*;
use chelis_stack_services::StackServices;
use core::cell::{Cell, RefCell};

/// Borrowed cancellation access. Each poll reads the caller token.
pub trait Cancellation {
    fn is_cancelled(&self) -> bool;
}

/// Policy selected by the caller for one operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct CheckPolicy {
    pub linked_program: bool,
}

/// Observations at the existing detailed profile boundaries.
pub trait CheckObserver {
    fn phase(&self, label: &str);
    fn declaration_begin(&self, _name: &str) {}
    fn declaration_end(&self, _name: &str) {}
}

/// Caller resources. The checker does not allocate native stacks or get native services.
pub struct Resources<'a, S: StackServices> {
    pub stack: &'a S,
    pub cancellation: Option<&'a dyn Cancellation>,
    pub policy: CheckPolicy,
    pub observer: Option<&'a dyn CheckObserver>,
    pub stack_bytes: usize,
    pub generalize_sweep_oracle: bool,
    pub generalize_sweep_visits: Option<&'a Cell<usize>>,
}

impl<'a, S: StackServices> Resources<'a, S> {
    pub const fn new(stack: &'a S) -> Self {
        Self {
            stack,
            cancellation: None,
            policy: CheckPolicy {
                linked_program: false,
            },
            observer: None,
            stack_bytes: 512 * 1024 * 1024,
            generalize_sweep_oracle: true,
            generalize_sweep_visits: None,
        }
    }

    pub(crate) fn run<R>(&self, run: impl FnOnce(&Operation<'_>) -> R) -> R {
        self.stack.grow(self.stack_bytes, || self.run_borrowed(run))
    }

    pub(crate) fn run_borrowed<R>(&self, run: impl FnOnce(&Operation<'_>) -> R) -> R {
        let remaining = || self.stack.remaining_stack();
        #[cfg(any(test, feature = "generalize-sweep-oracle"))]
        let sweep_visits = Cell::new(0);
        let operation = Operation {
            remaining: &remaining,
            cancellation: self.cancellation,
            policy: self.policy,
            observer: self.observer,
            depth: Cell::new(0),
            #[cfg(any(test, feature = "generalize-sweep-oracle"))]
            sweep_enabled: Cell::new(self.generalize_sweep_oracle),
            #[cfg(any(test, feature = "generalize-sweep-oracle"))]
            sweep_visits: self.generalize_sweep_visits.unwrap_or(&sweep_visits),
            stack_failure: RefCell::new(None),
            stack_scope_depth: Cell::new(0),

            recursion: RefCell::new(None),
            opacity: RefCell::new(None),
        };
        run(&operation)
    }
}

pub(crate) struct Operation<'a> {
    pub(crate) remaining: &'a dyn Fn() -> Option<usize>,
    pub(crate) cancellation: Option<&'a dyn Cancellation>,
    pub(crate) policy: CheckPolicy,
    pub(crate) observer: Option<&'a dyn CheckObserver>,
    pub(crate) depth: Cell<usize>,
    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub(crate) sweep_enabled: Cell<bool>,
    #[cfg(any(test, feature = "generalize-sweep-oracle"))]
    pub(crate) sweep_visits: &'a Cell<usize>,
    pub(crate) stack_failure: RefCell<Option<(String, Option<String>)>>,
    pub(crate) stack_scope_depth: Cell<usize>,
    pub(crate) recursion: RefCell<Option<crate::infer::recursion::GroupCtx>>,
    pub(crate) opacity: RefCell<Option<crate::opacity::OpacityContextData>>,
}

impl Operation<'_> {
    pub(crate) fn declaration_begin(&self, name: &str) {
        if let Some(observer) = self.observer {
            observer.declaration_begin(name);
        }
    }
    pub(crate) fn declaration_end(&self, name: &str) {
        if let Some(observer) = self.observer {
            observer.declaration_end(name);
        }
    }

    pub(crate) fn cancelled(&self) -> bool {
        self.cancellation.is_some_and(|token| token.is_cancelled())
    }

    pub(crate) fn phase(&self, label: &str) {
        if let Some(observer) = self.observer {
            observer.phase(label);
        }
    }
}
