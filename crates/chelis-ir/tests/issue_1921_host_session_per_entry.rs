//! chelis#1921 and chelis#3728: a public host-lowering entry opens one
//! session per program, never one per ask. A fresh session derives every
//! program-wide fact again, and no type can bound how long a caller holds
//! one, so these tests count the sessions an entry opens with one ask and
//! with many. The count is process-wide, and nextest runs each test in its
//! own process.

use chelis_ir::host::{
    host_lowering_sessions, lower_named_tensor_entry_dag, lower_named_tensor_entry_dags,
    try_lower_compiled_program,
};

fn checked_surf(src: &str) -> chelis_types::CheckedProgram {
    let decls =
        chelis_surf::parser::parse_str(src, &chelis_syntax_host::NativeStack).expect("surf parse");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expansion")
    .into_exprs();
    let checked = chelis_types::check_ir_program(&exprs)
        .unwrap_or_else(|report| panic!("type check failed: {:?}", report.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects");
    chelis_types::check_linearity(&checked).expect("linearity")
}

/// `definitions` tensor entries, each calling the one before it.
fn chain(definitions: usize) -> (chelis_types::CheckedProgram, Vec<String>) {
    let mut source = String::from("def f0(x: tensor[3, f32]) -> tensor[3, f32] = add(x, x)\n");
    for index in 1..definitions {
        let below = index - 1;
        source.push_str(&format!(
            "def f{index}(x: tensor[3, f32]) -> tensor[3, f32] = add(f{below}(x), x)\n"
        ));
    }
    let names = (0..definitions).map(|index| format!("f{index}")).collect();
    (checked_surf(&source), names)
}

fn sessions_during<T>(entry: impl FnOnce() -> T) -> (T, u64) {
    let before = host_lowering_sessions();
    let result = entry();
    (result, host_lowering_sessions() - before)
}

/// Evidentiary status: REGRESSION TEST for the batch entry `chelis cost`
/// uses; the negative control is the per-entry loop it replaced, which the
/// counter catches opening one session per entry.
#[test]
fn named_entries_of_one_program_share_one_session() {
    for definitions in [1, 8] {
        let (program, names) = chain(definitions);
        let (batched, sessions) =
            sessions_during(|| lower_named_tensor_entry_dags(&program, &names).collect::<Vec<_>>());
        assert_eq!(
            sessions, 1,
            "{definitions} entries opened {sessions} sessions"
        );

        let (one_by_one, per_entry_sessions) = sessions_during(|| {
            names
                .iter()
                .map(|name| lower_named_tensor_entry_dag(&program, name))
                .collect::<Vec<_>>()
        });
        assert_eq!(
            per_entry_sessions, definitions as u64,
            "negative control: a session per entry is counted"
        );
        assert!(batched.iter().all(Option::is_some), "every entry lowers");
        assert_eq!(batched, one_by_one, "sharing the session changes no DAG");
    }
}

/// Disposition lock: the whole-program lowering opens one session however
/// many definitions the program holds.
#[test]
fn whole_program_lowering_opens_one_session() {
    for definitions in [1, 8] {
        let (program, _) = chain(definitions);
        let (lowered, sessions) = sessions_during(|| try_lower_compiled_program(&program));
        lowered.expect("the chain lowers");
        assert_eq!(
            sessions, 1,
            "{definitions} definitions opened {sessions} sessions"
        );
    }
}
