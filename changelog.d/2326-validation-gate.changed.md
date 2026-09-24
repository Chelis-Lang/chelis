The repository gate's optional extra-validation run is now
`python3 scripts/gate.py --validation`; the `--local` spelling is gone, and the
inherited agent contract names the new flag. `--fast` remains the pre-push
gate. On macOS that run's runtime-representation stage no longer fails on the
capacity census's preprocessor attribution, which now drops the blank lines
Apple clang emits and GCC does not. See
[#2326](https://github.com/Chelis-Lang/chelis/issues/2326).
