"""Execute the version core portability and host-consumer contract."""

from __future__ import annotations

import sys

from no_std_core_support import ROOT, OracleFailure, consumer, portable, run


PORTABLE_SOURCE = """#![no_std]
pub fn classify(value: &str) -> (bool, bool) {
    (
        chelis_version::is_strict_semver(value),
        chelis_version::is_safe_path_component(value),
    )
}
"""

VERSION_TESTS = (
    "tests::strict_semver_accepts_xyz",
    "tests::strict_semver_rejects_non_xyz",
    "tests::path_component_rejects_traversal_and_separators",
    "tests::path_component_accepts_versions_and_channels",
    "tests::strict_semver_has_no_integer_component_limit",
)
SHIM_TESTS = (
    "level5_default_resolves_outside_any_package",
    "level1_plus_arg_wins_and_is_stripped",
    "level2_env_overrides_default",
    "level3_chelis_toolchain_file_overrides_default",
    "level4_reef_pin_overrides_default",
    "level3_beats_level4_toolchain_file_over_reef_pin",
    "reef_pin_walks_up_from_subdirectory",
    "resolved_but_not_installed_is_a_loud_error",
    "nothing_resolved_is_a_loud_error",
    "plus_arg_not_installed_does_not_fall_back",
)

HOST_SOURCE = r'''
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use chelis_version::{is_safe_path_component, is_strict_semver};

struct CountedAllocator;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountedAllocator = CountedAllocator;

fn main() {
    let component = "9".repeat(4096);
    let long = format!("{component}.{component}.{component}");
    let releases: [&str; 4] = ["0.12.0", "10.0.255", "00.001.0002", &long];
    let rejected = ["", "0.13", "1.2.3.4", "v0.1.0", "1.2.3-rc1", "1.2.3+build", "0..0", " 1.2.3", "１.2.3"];
    let safe_names = ["nightly-2026-06-30", "0.13.0-rc1", "0.13", "a..b", "a b", "版本"];
    let hostile_names = ["", ".", "..", "a/b", "a\\b", "../etc", "a\0b", "a\nb", "a\tb", "a\u{7f}b", "a\u{85}b"];

    ALLOCATIONS.store(0, Ordering::SeqCst);
    for _ in 0..100 {
        for value in releases {
            assert!(black_box(is_strict_semver(black_box(value))));
        }
        for value in rejected {
            assert!(!black_box(is_strict_semver(black_box(value))));
        }
        for value in safe_names {
            assert!(black_box(is_safe_path_component(black_box(value))));
            assert!(!black_box(is_strict_semver(black_box(value))));
        }
        for value in hostile_names {
            assert!(!black_box(is_safe_path_component(black_box(value))));
        }
    }
    let allocations = ALLOCATIONS.load(Ordering::SeqCst);
    assert_eq!(allocations, 0);

    for value in releases {
        assert!(chelisup::version::validate_install_version(value).is_ok());
        assert!(chelis_conformance::audit::is_installable_version(value));
    }
    for value in rejected.into_iter().chain(safe_names) {
        assert!(!chelis_conformance::audit::is_installable_version(value));
        assert_eq!(
            chelisup::version::validate_install_version(value).unwrap_err(),
            format!("malformed version {value:?}; expected X.Y.Z (for example 0.12.0)")
        );
    }
    println!("version consumer: exact predicates, zero allocations, and host adapters pass");
}
'''


def validate() -> None:
    run(("python3", "--version"))
    for default_features in (False, True):
        packages = portable(
            "chelis-version", PORTABLE_SOURCE, default_features=default_features
        )
        if packages != {"chelis-version", "chelis-no-std-consumer"}:
            raise OracleFailure("version core has an unexpected package dependency")
    dependencies = (
        f'chelisup = {{ path = "{ROOT / "crates" / "chelisup"}" }}\n'
        f'chelis-conformance = {{ path = "{ROOT / "crates" / "chelis-conformance"}" }}\n'
    )
    consumer("chelis-version", HOST_SOURCE, dependencies=dependencies)
    run(
        ("cargo", "nextest", "run", "-p", "chelis-version"),
        required_tests=VERSION_TESTS,
    )
    run(
        ("cargo", "nextest", "run", "-p", "chelisup", "--test", "shim_resolution"),
        required_tests=SHIM_TESTS,
    )
    run((sys.executable, "scripts/test_no_std_core_support.py"))
    print("version core oracle: PASS", flush=True)


def main() -> int:
    try:
        validate()
    except OracleFailure as error:
        print(f"version core oracle: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
