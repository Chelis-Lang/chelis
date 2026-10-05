//! Release inventory: host-runtime operations carry no whole-program or
//! live-only refusal scope; reachable and unreachable calls build alike
//! (chelis#1297).
#[path = "common/mod.rs"]
mod common;

#[test]
fn unreachable_host_runtime_calls_build() {
    let source = "def dead(seed: f32) -> tensor[3, f32] = tensor_scan(seed, fn (previous: f32, i: i64) -> add(previous, 1.0f32), 3i64)\n\
                  def dead_clock() -> (i64, i64) = clock_wall_read()\n\
                  out = print(7i32)\n";
    assert_eq!(common::build_and_run(source, "scope"), "7\nout = ()\n");
}

#[test]
fn live_and_unreachable_assertions_both_compile() {
    assert_eq!(
        common::build_and_run("out = test_assert(true, \"live\")\n", "scope"),
        "out = ()\n"
    );
    let source = "def dead() -> unit = test_assert(false, \"unreachable\")\nout = print(7i32)\n";
    assert_eq!(common::build_and_run(source, "scope"), "7\nout = ()\n");
}
