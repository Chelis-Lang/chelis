//! `core::panic::AssertUnwindSafe` (Aeneas.Std Core/UnwindSafe.lean) as chelis-ir
//! `HostCx::catch_lowering` uses it: the lowering body is wrapped, moved whole into the closure
//! that the catch calls with the runtime and the explicit state, and called through the field
//! (`(body.0)(..)`); what the body writes to the runtime and the state is given back to the
//! caller. Then the wrapper's `Deref` and `DerefMut` (the inner value) and its `FnOnce<()>` (the
//! inner closure's call, through a generic `impl FnOnce() -> R`).
use std::panic::AssertUnwindSafe;

pub struct Runtime {
    pub entered: u32,
    pub steps: u32,
    pub budget: u32,
}

pub struct Session {
    pub log: Vec<u32>,
    pub depth: u32,
}

/// As `LoweringRuntime::catch_lowering_with` without an unwind: counts the entry, then runs the
/// body on the runtime and the explicit state; the result is rejected (with the entry count)
/// when the body exceeded the runtime's step budget.
pub fn catch_with<S, R>(
    runtime: &mut Runtime,
    state: &mut S,
    body: impl FnOnce(&mut Runtime, &mut S) -> R,
) -> Result<R, u32> {
    runtime.entered += 1;
    let value = body(runtime, state);
    if runtime.steps > runtime.budget { Err(runtime.entered) } else { Ok(value) }
}

/// As `HostCx::catch_lowering`: the body in `AssertUnwindSafe`, moved whole into the catch's
/// closure and called through the field.
pub fn catch_lowering<R>(
    runtime: &mut Runtime,
    state: &mut Session,
    body: impl FnOnce(&mut Runtime, &mut Session) -> R,
) -> Result<R, u32> {
    let body = AssertUnwindSafe(body);
    catch_with(runtime, state, move |runtime, state| {
        // Move the whole wrapper; a field access would capture `body`.
        let body = body;
        (body.0)(runtime, state)
    })
}

/// A lowering step: logs `a`, nests one level and counts `b` runtime steps.
pub fn lower_step(runtime: &mut Runtime, state: &mut Session, a: u32, b: u32) -> Result<u32, u32> {
    catch_lowering(runtime, state, move |runtime, state| {
        state.log.push(a);
        state.depth += 1;
        runtime.steps += b;
        a + b
    })
}

/// Two nested catches, the inner one inside the outer body.
pub fn lower_nested(runtime: &mut Runtime, state: &mut Session, a: u32) -> Result<Result<u32, u32>, u32> {
    catch_lowering(runtime, state, move |runtime, state| {
        state.depth += 1;
        let inner = lower_step(runtime, state, a, a * 2);
        state.log.push(state.depth);
        inner
    })
}

/// `Deref`: the inner value, read through the wrapper (auto-deref of a method call).
pub fn read_through(wrapper: &AssertUnwindSafe<Vec<u32>>) -> (usize, u32) {
    let mut total = 0;
    for x in wrapper.iter() {
        total += *x;
    }
    (wrapper.len(), total)
}

/// `DerefMut`: the inner value, updated through the wrapper (given back).
pub fn bump_through(wrapper: &mut AssertUnwindSafe<u32>, by: u32) -> u32 {
    **wrapper += by;
    let inner: &mut u32 = &mut *wrapper;
    *inner *= 2;
    wrapper.0
}

/// `DerefMut` of a wrapped vector: pushes through the wrapper.
pub fn push_through(values: Vec<u32>, extra: u32) -> Vec<u32> {
    let mut wrapper = AssertUnwindSafe(values);
    wrapper.push(extra);
    (*wrapper).push(extra + 1);
    wrapper.0
}

fn run_once<R>(f: impl FnOnce() -> R) -> R {
    f()
}

/// `FnOnce<()>`: the wrapper called as a closure, directly and through a generic caller.
pub fn call_wrapped(x: u32, log: Vec<u32>) -> (u32, Vec<u32>) {
    let doubled = AssertUnwindSafe(move || x * 2);
    let direct = doubled();
    let wrapped = AssertUnwindSafe(move || {
        let mut log = log;
        log.push(direct);
        log
    });
    (direct, run_once(wrapped))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catches() {
        let mut runtime = Runtime { entered: 0, steps: 5, budget: 20 };
        let mut state = Session { log: vec![9], depth: 0 };
        assert_eq!(lower_step(&mut runtime, &mut state, 3, 4), Ok(7));
        assert_eq!((runtime.entered, runtime.steps), (1, 9));
        assert_eq!((state.log.clone(), state.depth), (vec![9, 3], 1));
        // Rejected after the body ran: its writes stay.
        assert_eq!(lower_step(&mut runtime, &mut state, 90, 20), Err(2));
        assert_eq!((runtime.entered, runtime.steps), (2, 29));
        assert_eq!((state.log.clone(), state.depth), (vec![9, 3, 90], 2));
    }

    #[test]
    fn nested() {
        let mut runtime = Runtime { entered: 0, steps: 0, budget: 50 };
        let mut state = Session { log: vec![], depth: 0 };
        assert_eq!(lower_nested(&mut runtime, &mut state, 5), Ok(Ok(15)));
        assert_eq!((runtime.entered, runtime.steps), (2, 10));
        assert_eq!((state.log.clone(), state.depth), (vec![5, 2], 2));
        // The inner catch rejects, then the outer one (with the inner entry count).
        assert_eq!(lower_nested(&mut runtime, &mut state, 25), Err(4));
        assert_eq!((runtime.entered, runtime.steps), (4, 60));
        assert_eq!((state.log.clone(), state.depth), (vec![5, 2, 25, 4], 4));
    }

    #[test]
    fn derefs() {
        assert_eq!(read_through(&AssertUnwindSafe(vec![1, 2, 3])), (3, 6));
        assert_eq!(read_through(&AssertUnwindSafe(vec![])), (0, 0));
        let mut wrapper = AssertUnwindSafe(5);
        assert_eq!(bump_through(&mut wrapper, 2), 14);
        assert_eq!(wrapper.0, 14);
        assert_eq!(push_through(vec![1], 7), vec![1, 7, 8]);
    }

    #[test]
    #[should_panic]
    fn deref_mut_overflows() {
        bump_through(&mut AssertUnwindSafe(u32::MAX), 1);
    }

    #[test]
    fn calls() {
        assert_eq!(call_wrapped(21, vec![1]), (42, vec![1, 42]));
        assert_eq!(call_wrapped(0, vec![]), (0, vec![0]));
    }
}
