//! Native resource policy for full-host measurement windows.
use std::cell::Cell;
thread_local! { static SWEEP_ENABLED: Cell<bool> = const { Cell::new(true) }; }
pub fn without_generalize_sweep_oracle<R>(run: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SWEEP_ENABLED.with(|value| value.set(self.0));
        }
    }
    let previous = SWEEP_ENABLED.with(|value| value.replace(false));
    let _restore = Restore(previous);
    run()
}
pub(crate) fn enabled() -> bool {
    SWEEP_ENABLED.with(Cell::get)
}
