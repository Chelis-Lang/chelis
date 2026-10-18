//! Verifying and projecting a body's host sites pays for each site action
//! once, not once per operation of its block (chelis#3825).
//!
//! Every host site carries actions that name a block, an operation, an
//! edge-local terminal or a control edge. `verify_host_actions` and the
//! verified emission's `actions()` (what the C host-ABI projection reads)
//! found each by scanning the unit's blocks and the block's operations, and
//! each control edge re-derived the whole program's control-edge table. A
//! `let` chain of `k` bindings puts its `k` operations in one block, so both
//! did about `k^2` work. They now read per-unit position tables built in one
//! pass.
//!
//! The verified emission's per-function views found each function's sites
//! by filtering every site record of the program, so projecting `n`
//! definitions compared `n` times the program's sites (chelis#3847). They now
//! read the records indexed by unit.
//!
//! The oracle is the lookups' own count of the entries they visit (table
//! entries built plus lookups answered), kept per thread while the test
//! counts, which is deterministic for a fixed program. Doubling the chain must
//! grow the count by at most 2.5 times: linear growth gives about 2, quadratic
//! about 4.

use std::fmt::Write as _;

use chelis_effects::realizability::{compute_root_manifest, infer_realizability};
use chelis_ir::host::try_lower_compiled_program_with_manifest;
use chelis_ir::ownership::{counting_site_lookup_entries, lower_host_ownership, verify_ownership};
use chelis_types::manifest::ManifestedProgram;
use chelis_types::types::{Prim, Target};

const SIZES: [usize; 4] = [125, 250, 500, 1000];
/// Smaller for the branch chain, whose lowering costs more per binding.
const BRANCH_SIZES: [usize; 4] = [32, 64, 128, 256];
const C_TENSOR_PRIMS: &[Prim] = &[Prim::F32, Prim::F64, Prim::Int32, Prim::Int64];

fn grows_linearly(small: u64, large: u64) -> bool {
    large * 2 <= small * 5
}

/// Lower, verify and walk every site's actions, as the C projection does,
/// counting the entries the site lookups visit.
fn site_lookup_entries(source: &str) -> u64 {
    let declarations = chelis_surf::parser::parse_str(source).expect("the probe parses");
    let deep = chelis_surf::desugar::desugar_program(&declarations).expect("the probe desugars");
    let checked = chelis_types::check_typed_program(&deep).expect("the probe checks");
    let checked = chelis_effects::check_program(&checked).expect("the probe has no effect errors");
    let checked = chelis_types::check_linearity(&checked).expect("the probe is linear");
    let realizability = infer_realizability(&checked, C_TENSOR_PRIMS);
    let manifest = compute_root_manifest(&checked, &realizability);
    let compiled =
        try_lower_compiled_program_with_manifest(&checked, &manifest).expect("the probe lowers");
    let host = compiled.host.expect("the probe has a host lane");
    let manifested = ManifestedProgram::new(checked, manifest, Target::C);
    let ((), entries) = counting_site_lookup_entries(|| {
        let lowered = lower_host_ownership(&manifested, host).expect("ownership lowers");
        let verified = verify_ownership(lowered).expect("ownership verifies");
        let emission = verified.emission();
        let actions = emission
            .sites()
            .map(|site| site.actions().count())
            .sum::<usize>();
        assert!(actions > 0, "the probe has site actions");
        // What the C projection reads per function, and for the roots.
        let function_sites = (0..emission.function_count())
            .map(|index| {
                emission
                    .function(index)
                    .expect("function census")
                    .sites()
                    .count()
            })
            .sum::<usize>();
        let root_sites = emission.root_sites().count();
        assert_eq!(
            function_sites + root_sites,
            emission.sites().count(),
            "every site belongs to the roots or to one function"
        );
    });
    entries
}

/// chelis#3793's probe: one host definition whose body chains `k` bindings
/// and returns them as a list, and one eager use.
fn let_chain(k: usize) -> String {
    let mut source = String::from("def f(x: i64) -> List[i64] = {\n");
    for index in 0..k {
        writeln!(source, "  t{index} = add(x, {index}i64)").unwrap();
    }
    let items = (0..k).map(|index| format!("t{index}")).collect::<Vec<_>>();
    writeln!(source, "  [{}]\n}}", items.join(", ")).unwrap();
    source.push_str("r = index(f(1i64), 0i64)\n");
    source
}

/// A body of `k` conditionals in sequence, so it has about `2k` control
/// edges and `k` blocks.
fn branch_chain(k: usize) -> String {
    let mut source = String::from("def g(x: i64) -> List[i64] = {\n");
    for index in 0..k {
        writeln!(
            source,
            "  t{index} = if gt(x, {index}i64) then add(x, {index}i64) else sub(x, {index}i64)"
        )
        .unwrap();
    }
    let items = (0..k).map(|index| format!("t{index}")).collect::<Vec<_>>();
    writeln!(source, "  [{}]\n}}", items.join(", ")).unwrap();
    source.push_str("r = index(g(1i64), 0i64)\n");
    source
}

/// `k` one-line definitions and one root that calls each of them.
fn definitions(k: usize) -> String {
    let mut source = String::new();
    for index in 0..k {
        writeln!(
            source,
            "def f{index}(x: i64) -> i64 = add(mul(x, x), {index}i64)"
        )
        .unwrap();
    }
    let calls = (0..k)
        .map(|index| format!("f{index}(1i64)"))
        .collect::<Vec<_>>();
    writeln!(source, "r = [{}]", calls.join(", ")).unwrap();
    source
}

fn sweep(probe: fn(usize) -> String, sizes: [usize; 4]) -> [u64; 4] {
    std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || sizes.map(|k| site_lookup_entries(&probe(k))))
        .expect("spawn the measuring thread")
        .join()
        .expect("the measuring thread finishes")
}

fn assert_linear(what: &str, sizes: [usize; 4], counts: [u64; 4]) {
    eprintln!("{what} at k = {sizes:?}: {counts:?}");
    assert!(counts[0] > 0, "{what}: the lookups ran on the probe");
    for pair in counts.windows(2) {
        assert!(
            grows_linearly(pair[0], pair[1]),
            "{what} grew superlinearly in the chain: {counts:?}"
        );
    }
}

#[test]
fn site_lookups_over_a_let_chain_visit_linearly_many_entries() {
    assert_linear(
        "site lookup entries, let chain",
        SIZES,
        sweep(let_chain, SIZES),
    );
}

#[test]
fn site_lookups_over_a_branch_chain_visit_linearly_many_entries() {
    assert_linear(
        "site lookup entries, branch chain",
        BRANCH_SIZES,
        sweep(branch_chain, BRANCH_SIZES),
    );
}

#[test]
fn per_function_site_views_visit_linearly_many_entries_over_many_definitions() {
    assert_linear(
        "site lookup entries, definitions",
        BRANCH_SIZES,
        sweep(definitions, BRANCH_SIZES),
    );
}

#[test]
fn the_detector_rejects_a_scan_of_the_block_per_action() {
    // Negative control: what the removed lookups did, a scan of the block's
    // operations up to each action's own.
    fn scan_per_action(k: u64) -> u64 {
        (0..k).map(|position| position + 1).sum()
    }
    assert!(
        !grows_linearly(scan_per_action(500), scan_per_action(1000)),
        "the detector accepted a scan of the block per action"
    );
}
