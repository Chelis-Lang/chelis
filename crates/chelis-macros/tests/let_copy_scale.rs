//! Macro-free scalar blocks should not copy the unvisited let tail at each binding.
use chelis_macros::{ExpansionOptions, expand_program};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

#[test]
fn expanding_scalar_blocks_does_not_copy_each_remaining_tail() {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(|| {
            let mut source = String::new();
            for function in 0..32 {
                writeln!(source, "def helper{function}(x: f64) -> f64 = {{").unwrap();
                for index in 0..512 {
                    writeln!(source, "v{index} = (x + {index}.0f64)").unwrap();
                }
                source.push_str("v511\n}\n");
            }
            let decls = chelis_surf::parser::parse_str(&source).unwrap();
            let deep = chelis_surf::desugar::desugar_program(&decls);
            let started = Instant::now();
            let expanded = expand_program(
                &deep,
                &ExpansionOptions {
                    max_iterations: 100,
                    load_std_prelude: false,
                },
            )
            .unwrap();
            let elapsed = started.elapsed();
            assert_eq!(expanded.exprs(), deep.as_slice());
            assert_eq!(expanded.expansions(), 0);
            assert!(
                elapsed < Duration::from_secs(5),
                "expansion took {elapsed:?}"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}
