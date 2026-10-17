`chelis test` and `chelis prove` stop redoing per-worker and per-property work.
`chelis test --batch-mode auto` no longer moves a test file out of its batch
because another file declares or imports the same name: each batched file
already compiles as its own module, so suites whose files reuse a test or helper
name (for example a generated `test_probe` in every file) now batch, where they
used to run one worker per file. A warm `chelis test` hands its workers the
verified cache bytes instead of encoding the context again, and the workers no
longer decode the lowered library they re-derive. `chelis prove` checks and
lowers a file's declarations once per file rather than once per property. Test
rows and machine-readable output keep their format. See
[#3744](https://github.com/Chelis-Lang/chelis/issues/3744),
[#2256](https://github.com/Chelis-Lang/chelis/issues/2256),
[#3746](https://github.com/Chelis-Lang/chelis/issues/3746) and
[#3745](https://github.com/Chelis-Lang/chelis/issues/3745).
