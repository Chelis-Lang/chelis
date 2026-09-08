//! Test-only deterministic allocation ledger for chelis#1286.
//!
//! The public runtime ABI does not expose this module. With the
//! `ownership-ledger` feature disabled every call is an inline no-op, and
//! release artifacts contain no ledger state or environment dependency.

use std::ffi::c_void;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Tensor,
    TensorStorage,
    String,
    List,
    Tuple,
    Dict,
    Adt,
    Option,
    MappedFile,
}

impl Kind {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Tensor => "Tensor",
            Self::TensorStorage => "TensorStorage",
            Self::String => "String",
            Self::List => "List",
            Self::Tuple => "Tuple",
            Self::Dict => "Dict",
            Self::Adt => "Adt",
            Self::Option => "Option",
            Self::MappedFile => "MappedFile",
        }
    }
}

#[cfg(feature = "ownership-ledger")]
mod enabled {
    use super::{c_void, Kind};
    use chelis_unord::UnordMap;
    use std::env;
    use std::fs::{File, OpenOptions};
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    const PATH_ENV: &str = "CHELIS_OWNERSHIP_LEDGER_PATH";
    const SCHEMA: &str = "compiled-value-ownership-ledger-v1";

    #[derive(Clone, Copy)]
    struct Allocation {
        id: u64,
        kind: Kind,
        bytes: u64,
        owners: u64,
        finalized: bool,
    }

    struct State {
        file: File,
        by_pointer: UnordMap<usize, Allocation>,
        next_id: u64,
        allocations: u64,
        finalized: u64,
        live_owners: u64,
        live_bytes: u64,
        peak_live_bytes: u64,
        invalid_operations: u64,
        summary_written: bool,
    }

    impl State {
        fn new(mut file: File) -> std::io::Result<Self> {
            writeln!(file, "{{\"event\":\"header\",\"schema\":\"{SCHEMA}\"}}")?;
            file.flush()?;
            Ok(Self {
                file,
                by_pointer: UnordMap::new(),
                next_id: 1,
                allocations: 0,
                finalized: 0,
                live_owners: 0,
                live_bytes: 0,
                peak_live_bytes: 0,
                invalid_operations: 0,
                summary_written: false,
            })
        }

        fn write(&mut self, record: &str) {
            if let Err(error) = writeln!(self.file, "{record}").and_then(|_| self.file.flush()) {
                eprintln!("compiled ownership ledger write failed: {error}");
                // `write` runs while the ledger mutex is held. Running exit
                // handlers here would re-enter `write_summary_at_exit` and
                // deadlock on that mutex. Abort instead: the missing final
                // summary makes the oracle fail closed on the I/O error.
                std::process::abort();
            }
        }

        fn write_transition(
            &mut self,
            event: &str,
            allocation: Allocation,
            before: u64,
            after: u64,
            site: &str,
        ) {
            self.write(&format!(
                "{{\"event\":\"{event}\",\"id\":{},\"kind\":\"{}\",\"bytes\":{},\"owners_before\":{before},\"owners_after\":{after},\"site\":\"{}\"}}",
                allocation.id,
                allocation.kind.name(),
                allocation.bytes,
                escape(site),
            ));
        }

        fn write_summary(&mut self) {
            if self.summary_written {
                return;
            }
            self.summary_written = true;
            self.write(&format!(
                "{{\"event\":\"summary\",\"allocations\":{},\"finalized\":{},\"live_owners\":{},\"live_bytes\":{},\"peak_live_bytes\":{},\"invalid_operations\":{}}}",
                self.allocations,
                self.finalized,
                self.live_owners,
                self.live_bytes,
                self.peak_live_bytes,
                self.invalid_operations,
            ));
        }

        fn invalid(&mut self, event: &str, pointer: usize, site: &str) {
            self.invalid_operations = self.invalid_operations.saturating_add(1);
            let identity = self
                .by_pointer
                .get(&pointer)
                .map(|allocation| allocation.id.to_string())
                .unwrap_or_else(|| "null".to_owned());
            self.write(&format!(
                "{{\"event\":\"{event}\",\"id\":{identity},\"site\":\"{}\"}}",
                escape(site),
            ));
        }
    }

    fn escape(value: &str) -> String {
        value.replace('\\', "\\\\").replace('"', "\\\"")
    }

    static LEDGER: OnceLock<Option<Mutex<State>>> = OnceLock::new();

    fn initialize() -> Option<Mutex<State>> {
        let path = env::var_os(PATH_ENV)?;
        if path.is_empty() {
            eprintln!("{PATH_ENV} must not be empty when the ownership ledger is enabled");
            std::process::exit(1);
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .unwrap_or_else(|error| {
                eprintln!("open ownership ledger {}: {error}", path.to_string_lossy());
                std::process::exit(1);
            });
        let state = State::new(file).unwrap_or_else(|error| {
            eprintln!("initialize ownership ledger: {error}");
            std::process::exit(1);
        });
        // The callback writes the one final summary after generated `main`
        // has completed. Every event is flushed independently so a process
        // that exits early still leaves its precise invalid transition.
        let status = unsafe { libc::atexit(write_summary_at_exit) };
        if status != 0 {
            eprintln!("register ownership ledger exit summary failed");
            std::process::exit(1);
        }
        Some(Mutex::new(state))
    }

    fn with_state<T>(action: impl FnOnce(&mut State) -> T) -> Option<T> {
        let ledger = LEDGER.get_or_init(initialize).as_ref()?;
        let mut state = ledger.lock().unwrap_or_else(|_| {
            eprintln!("ownership ledger lock poisoned");
            std::process::exit(1);
        });
        Some(action(&mut state))
    }

    extern "C" fn write_summary_at_exit() {
        let _ = with_state(State::write_summary);
    }

    pub(crate) fn allocate(pointer: *const c_void, kind: Kind, bytes: u64, site: &str) -> bool {
        if pointer.is_null() {
            return false;
        }
        with_state(|state| {
            let key = pointer as usize;
            if state
                .by_pointer
                .get(&key)
                .is_some_and(|allocation| !allocation.finalized)
            {
                state.invalid("invalid_allocate", key, site);
                return false;
            }
            let allocation = Allocation {
                id: state.next_id,
                kind,
                bytes,
                owners: 1,
                finalized: false,
            };
            state.next_id = state.next_id.saturating_add(1);
            state.allocations = state.allocations.saturating_add(1);
            state.live_owners = state.live_owners.saturating_add(1);
            state.live_bytes = state.live_bytes.saturating_add(bytes);
            state.peak_live_bytes = state.peak_live_bytes.max(state.live_bytes);
            state.by_pointer.insert(key, allocation);
            state.write_transition("allocate", allocation, 0, 1, site);
            true
        })
        .unwrap_or(true)
    }

    pub(crate) fn retain(pointer: *const c_void, site: &str) -> bool {
        if pointer.is_null() {
            return true;
        }
        with_state(|state| {
            let key = pointer as usize;
            let Some(mut allocation) = state.by_pointer.get(&key).copied() else {
                state.invalid("invalid_retain", key, site);
                return false;
            };
            if allocation.finalized || allocation.owners == 0 {
                state.invalid("invalid_retain", key, site);
                return false;
            }
            let before = allocation.owners;
            allocation.owners = allocation.owners.saturating_add(1);
            state.live_owners = state.live_owners.saturating_add(1);
            state.by_pointer.insert(key, allocation);
            state.write_transition("retain", allocation, before, allocation.owners, site);
            true
        })
        .unwrap_or(true)
    }

    pub(crate) fn release(pointer: *const c_void, site: &str) -> bool {
        if pointer.is_null() {
            return true;
        }
        with_state(|state| {
            let key = pointer as usize;
            let Some(mut allocation) = state.by_pointer.get(&key).copied() else {
                state.invalid("invalid_release", key, site);
                return false;
            };
            if allocation.finalized || allocation.owners == 0 {
                state.invalid("invalid_release", key, site);
                return false;
            }
            let before = allocation.owners;
            allocation.owners -= 1;
            state.live_owners -= 1;
            if allocation.owners == 0 {
                state.live_bytes -= allocation.bytes;
            }
            state.by_pointer.insert(key, allocation);
            state.write_transition("release", allocation, before, allocation.owners, site);
            true
        })
        .unwrap_or(true)
    }

    pub(crate) fn finalize(pointer: *const c_void, site: &str) -> bool {
        if pointer.is_null() {
            return true;
        }
        with_state(|state| {
            let key = pointer as usize;
            let Some(mut allocation) = state.by_pointer.get(&key).copied() else {
                state.invalid("invalid_finalize", key, site);
                return false;
            };
            if allocation.finalized || allocation.owners != 0 {
                state.invalid("invalid_finalize", key, site);
                return false;
            }
            allocation.finalized = true;
            state.finalized = state.finalized.saturating_add(1);
            state.by_pointer.insert(key, allocation);
            state.write_transition("finalize", allocation, 0, 0, site);
            true
        })
        .unwrap_or(true)
    }

    pub(crate) fn resize(pointer: *const c_void, bytes: u64, site: &str) -> bool {
        if pointer.is_null() {
            return true;
        }
        with_state(|state| {
            let key = pointer as usize;
            let Some(mut allocation) = state.by_pointer.get(&key).copied() else {
                state.invalid("invalid_resize", key, site);
                return false;
            };
            if allocation.finalized || allocation.owners == 0 {
                state.invalid("invalid_resize", key, site);
                return false;
            }
            let before = allocation.bytes;
            state.live_bytes = state.live_bytes.saturating_sub(before).saturating_add(bytes);
            state.peak_live_bytes = state.peak_live_bytes.max(state.live_bytes);
            allocation.bytes = bytes;
            state.by_pointer.insert(key, allocation);
            state.write(&format!(
                "{{\"event\":\"resize\",\"id\":{},\"kind\":\"{}\",\"bytes_before\":{before},\"bytes_after\":{bytes},\"site\":\"{}\"}}",
                allocation.id,
                allocation.kind.name(),
                escape(site),
            ));
            true
        })
        .unwrap_or(true)
    }

    pub(crate) fn borrow(pointer: *const c_void, site: &str) {
        if pointer.is_null() {
            return;
        }
        let _ = with_state(|state| {
            let key = pointer as usize;
            let Some(allocation) = state.by_pointer.get(&key).copied() else {
                state.invalid("invalid_borrow", key, site);
                return;
            };
            if allocation.finalized || allocation.owners == 0 {
                state.invalid("invalid_borrow", key, site);
                return;
            }
            state.write(&format!(
                "{{\"event\":\"borrow\",\"id\":{},\"kind\":\"{}\",\"site\":\"{}\"}}",
                allocation.id,
                allocation.kind.name(),
                escape(site),
            ));
        });
    }
}

#[cfg(feature = "ownership-ledger")]
pub(crate) use enabled::{allocate, borrow, finalize, release, resize, retain};

#[cfg(not(feature = "ownership-ledger"))]
#[inline(always)]
pub(crate) fn allocate(_pointer: *const c_void, _kind: Kind, _bytes: u64, _site: &str) -> bool {
    true
}

#[cfg(not(feature = "ownership-ledger"))]
#[inline(always)]
pub(crate) fn retain(_pointer: *const c_void, _site: &str) -> bool {
    true
}

#[cfg(not(feature = "ownership-ledger"))]
#[inline(always)]
pub(crate) fn release(_pointer: *const c_void, _site: &str) -> bool {
    true
}

#[cfg(not(feature = "ownership-ledger"))]
#[inline(always)]
pub(crate) fn finalize(_pointer: *const c_void, _site: &str) -> bool {
    true
}

#[cfg(not(feature = "ownership-ledger"))]
#[inline(always)]
pub(crate) fn resize(_pointer: *const c_void, _bytes: u64, _site: &str) -> bool {
    true
}

#[cfg(not(feature = "ownership-ledger"))]
#[inline(always)]
pub(crate) fn borrow(_pointer: *const c_void, _site: &str) {}
