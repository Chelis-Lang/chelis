# Local Z3 Environment

The `z3` cargo feature on `chelis-prove` (WI-12 / WS-5) adds a second SMT
discharge engine backed by Z3's nonlinear-real-arithmetic decision procedure. It
links a **prebuilt** libz3 rather than building Z3 from source, so a usable
libz3 must be discoverable at build and run time.

## Quick reference (run z3-feature tests this way)

Use `scripts/z3_test.py` — it finds a prebuilt `libz3.so`, sets the link +
loader env vars, and execs the cargo command (defaulting to
`cargo nextest run -p chelis-prove --features z3`):

```sh
scripts/z3_test.py
scripts/z3_test.py -p chelis-prove --features z3 --test cross_engine_oracle
scripts/z3_test.py --features "smt z3" --test cross_engine_oracle   # cross-engine oracle
# Inside an active Devenv shell, replace scripts/z3_test.py with chelis-z3-test.
```

If you need the env in your own shell:

```sh
eval $(scripts/z3_test.py --print-env)
```

## How the link works

The feature pins `z3 = 0.20` / `z3-sys = 0.11` with
`default-features = false`. That matters:

- With default features OFF, z3-sys does **not** vendor/compile Z3 from source
  (no cmake). Its build.rs takes the default branch, which calls
  `pkg-config probe("z3")` (non-fatal) and adds a `-L` link-search path from the
  `Z3_LIBRARY_PATH_OVERRIDE` env var. The actual link directive is
  `#[link(name = "z3")]` in z3-sys's `lib.rs`, so the linker resolves `-lz3`.
- z3-sys uses its **committed** bindings (`src/generated/functions.rs`) when the
  `bindgen` feature is off, so no `z3.h` parsing happens at build time.

So the two requirements are:

1. **Build time:** the linker must find `libz3.so` for `-lz3`. Set
   `Z3_LIBRARY_PATH_OVERRIDE=<dir containing libz3.so>` (build.rs turns it into a
   `-L` search path), or have libz3 on the default linker path.
2. **Run time:** the loader must find the shared object. Put the same dir on
   `LD_LIBRARY_PATH`, or have libz3 on the default loader path.

`-lz3` needs the **unversioned** `libz3.so` symlink, not only a versioned
`libz3.so.4.15`.

## This workstation

The prebuilt libz3 ships inside the python `z3` package's site-packages:

```
~/.local/lib/python3.12/site-packages/z3/lib/libz3.so        # 4.15, unversioned symlink
~/.local/lib/python3.12/site-packages/z3/lib/libz3.so.4.15
```

This is **not** on the default linker/loader paths, so the env vars are
required. `scripts/z3_test.py` discovers this dir automatically (it probes `import z3`
and globs `~/.local/lib/python3.*/site-packages/z3/lib`), so prefer the wrapper
over setting the vars by hand. If the python z3 package is relocated or its
python minor version changes, the glob still finds it; only an entirely
different install location needs `Z3_LIBRARY_PATH_OVERRIDE` set explicitly.

## CI

In CI the prebuilt libz3 is the apt `libz3-dev` package, which installs to the
default linker/loader paths. There `Z3_LIBRARY_PATH_OVERRIDE` and the
`LD_LIBRARY_PATH` addition are no-ops (and `scripts/z3_test.py` prints
"using system libz3"). The exact CI-lane recipe (apt deps + the test command)
is owned by the workflow files, not this doc.

## Default and `smt` builds are unaffected

The `z3` feature is opt-in. The default build and the `smt` (cvc5) build pull no
`z3` dependency and link no z3 symbols, so the solver-free corpus gate
(`chelis check` links zero solver symbols) stays green. The Z3 engine is
`#[cfg(feature = "z3")]`-gated end to end.
