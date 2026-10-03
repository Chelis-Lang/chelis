//! The floating-point environment compiled entry points run under
//! (chelis#2964).
//!
//! [04-NUM-2] fixes float finalization as round-to-nearest-ties-to-even with
//! subnormals preserved. A static library runs in its host's environment: a
//! host that set a rounding mode, or loaded a `-ffast-math` library whose
//! `crtfastmath` startup code turned on flush-to-zero, would otherwise change
//! every result without a diagnostic. Every exported entry point the C
//! backend emits calls [`chelis_fp_env_enter`] first and
//! [`chelis_fp_env_leave`] before it returns. The outermost entry on a
//! thread saves the caller's control register and installs the IEEE
//! default; nested entries (an exported kernel called from an exported
//! wrapper) only count, and the matching outermost leave restores the
//! caller's register exactly.
//!
//! ISO C `<fenv.h>` has no flush-to-zero control, so the register is read
//! and written directly:
//!
//! - arm64 FPCR: zero is the IEEE default (round to nearest, FZ, FZ16, DN,
//!   AH and FIZ clear, every exception trap disabled).
//! - x86_64 MXCSR: `0x1f80` masks every exception and clears the rounding
//!   field, FTZ and DAZ. Float arithmetic in emitted code and in this runtime
//!   is SSE; neither uses `long double`, so the x87 control word never
//!   governs a result.
//!
//! Exception flags raised inside the call are discarded with the restore.

use std::cell::Cell;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
compile_error!("the Chelis runtime pins the floating-point environment only on aarch64 and x86_64");

thread_local! {
    static DEPTH: Cell<u64> = const { Cell::new(0) };
    static SAVED: Cell<u64> = const { Cell::new(0) };
}

#[cfg(target_arch = "aarch64")]
const IEEE_DEFAULT: u64 = 0;
#[cfg(target_arch = "x86_64")]
const IEEE_DEFAULT: u64 = 0x1f80;

#[cfg(target_arch = "aarch64")]
fn read_control() -> u64 {
    let fpcr: u64;
    // SAFETY: reading FPCR has no side effect.
    unsafe { std::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack)) };
    fpcr
}

#[cfg(target_arch = "aarch64")]
fn write_control(fpcr: u64) {
    // SAFETY: FPCR holds only floating-point control bits; every value
    // written is either the IEEE default or a value read from FPCR.
    unsafe { std::arch::asm!("msr fpcr, {}", in(reg) fpcr, options(nostack)) };
}

#[cfg(target_arch = "x86_64")]
fn read_control() -> u64 {
    let mut mxcsr: u32 = 0;
    // SAFETY: `stmxcsr` stores four bytes to the given live local.
    unsafe {
        std::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack));
    }
    u64::from(mxcsr)
}

#[cfg(target_arch = "x86_64")]
fn write_control(mxcsr: u64) {
    let mxcsr = mxcsr as u32;
    // SAFETY: `ldmxcsr` loads four bytes from the given live local; every
    // value written is the IEEE default or a value read from MXCSR, so no
    // reserved bit is set.
    unsafe {
        std::arch::asm!("ldmxcsr [{}]", in(reg) &mxcsr, options(nostack, readonly));
    }
}

/// Enter compiled code: on the thread's outermost entry, save the caller's
/// floating-point control state and install the IEEE default.
#[no_mangle]
pub extern "C" fn chelis_fp_env_enter() {
    DEPTH.with(|depth| {
        if depth.get() == 0 {
            SAVED.with(|saved| saved.set(read_control()));
            write_control(IEEE_DEFAULT);
        }
        depth.set(depth.get() + 1);
    });
}

/// Leave compiled code: on the thread's outermost exit, restore the state
/// [`chelis_fp_env_enter`] saved.
#[no_mangle]
pub extern "C" fn chelis_fp_env_leave() {
    DEPTH.with(|depth| {
        let entered = depth.get();
        if entered == 0 {
            runtime_fail!("chelis_fp_env_leave without a matching chelis_fp_env_enter");
        }
        depth.set(entered - 1);
        if entered == 1 {
            write_control(SAVED.with(Cell::get));
        }
    });
}

/// The same entry and exit for Rust hosts of Chelis numerics: the evaluator
/// runs inside its caller's process (the CLI, and the Python and
/// compiler-api bindings), whose thread may carry any control state. Holding
/// the guard pins the IEEE default; dropping it, on return or unwind,
/// restores the caller's state.
#[must_use = "the IEEE default holds only while the guard is alive"]
pub struct FpEnvGuard {
    // Entry and exit are per thread, so the guard must not cross threads.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl FpEnvGuard {
    pub fn enter() -> Self {
        chelis_fp_env_enter();
        Self {
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for FpEnvGuard {
    fn drop(&mut self) {
        chelis_fp_env_leave();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outermost_entry_installs_the_ieee_default_and_leave_restores_the_caller() {
        let caller = read_control();
        #[cfg(target_arch = "aarch64")]
        let custom = caller | (1 << 24) | (0b10 << 22);
        #[cfg(target_arch = "x86_64")]
        let custom = (caller & !0x6000) | 0x8040 | 0x4000;
        write_control(custom);
        chelis_fp_env_enter();
        assert_eq!(read_control(), IEEE_DEFAULT);
        chelis_fp_env_enter();
        chelis_fp_env_leave();
        assert_eq!(
            read_control(),
            IEEE_DEFAULT,
            "a nested leave keeps the default"
        );
        chelis_fp_env_leave();
        assert_eq!(read_control(), custom);
        write_control(caller);
    }

    #[test]
    fn guard_installs_the_ieee_default_and_restores_the_caller_on_unwind() {
        let caller = read_control();
        #[cfg(target_arch = "aarch64")]
        let custom = caller | (1 << 24) | (0b10 << 22);
        #[cfg(target_arch = "x86_64")]
        let custom = (caller & !0x6000) | 0x8040 | 0x4000;
        write_control(custom);
        {
            let _guard = FpEnvGuard::enter();
            assert_eq!(read_control(), IEEE_DEFAULT);
        }
        assert_eq!(read_control(), custom);
        let unwound = std::panic::catch_unwind(|| {
            let _guard = FpEnvGuard::enter();
            panic!("evaluation failed");
        });
        assert!(unwound.is_err());
        assert_eq!(read_control(), custom, "an unwind restores the caller");
        write_control(caller);
    }
}
