Illustrative Chelis examples: programs that show a language form or a design
pattern but sit outside the executable corpus.

Every `.ch` file directly under `examples/` belongs to the executable corpus,
and the parity test (`crates/chelis-cli/tests/parity.rs`) checks each one. For
most, it runs the program in the evaluator, builds and runs it through C, and
requires the two outputs to agree. A program with nothing to print, such as
`linreg.ch`, has its C built as an object and must print nothing in the
evaluator. `annotated_concat_softmax.ch` is the exception: the test pins its
evaluator output and the C build's rejection instead.

Every `.ch` file in this directory passes `chelis fmt --check` and
`chelis check`, but is not held to that parity run. Each is here for one of
these reasons:

- It only declares functions and has no top-level expression to evaluate, so
  it shows signatures and shapes rather than output, as the attention-block
  files and `mlp.ch` do.
- It uses a form that `chelis eval` accepts but `chelis build --target c`
  rejects, such as host subprocesses in `process_run_chelis_version.ch`, a
  non-`cpu` device region in `effects_handlers.ch`, or differentiation through
  functions stored in constructor or record payloads in
  `grad_selector_provenance.ch`.
- A compiler test reads it as a fixture, as its header comment says.

Subdirectories are complete Reef packages with their own `reef.toml` and data.
`io_pipeline/` reads its CSV and JSON inputs with `chelis eval --file src/main.ch`
run from the package directory.
