//! Generated modules must not repeatedly clone every descendant while normalizing Nodes.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

#[test]
fn scalar_helpers_normalize_without_quadratic_subtree_copying() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let mut source = String::new();
            for function in 0..64 {
                writeln!(source, "def helper{function}(x: f64) -> f64 = {{").unwrap();
                for index in 0..256 {
                    writeln!(source, "v{index} = (x + {index}.0f64)").unwrap();
                }
                source.push_str("v255\n}\n");
            }
            let declarations = parse_str(&source).unwrap();
            let started = Instant::now();
            let deep = desugar_program(&declarations).expect("Surf fixture must desugar");
            let elapsed = started.elapsed();
            assert_eq!(chelis_deep::validate::find_raw_vocabulary_tag(&deep), None);
            assert_eq!(deep.len(), 128, "one definition and signature per helper");
            assert!(
                elapsed < Duration::from_secs(10),
                "desugar took {elapsed:?}"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}
