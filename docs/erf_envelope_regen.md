# erf Envelope Regeneration and CI Re-validation (WI-13)

The `chelis-prove` crate ships a committed, Arb-certified `erf` envelope —
`crates/chelis-prove/data/erf_envelope.json` — that the runtime discharge and
Beacon's `erf` relaxation (WI-B6) consume to bound `erf` soundly without any
runtime Arb dependency. This document covers how the data is generated, how CI
re-validates it, and the environment each step needs.

## What the committed data is

A piecewise envelope of contiguous boxes covering `[-300, 300]`:

| Box          | Arm                          | Certified `eps` |
|--------------|------------------------------|-----------------|
| `[-300, -3]` | saturation `-1`              | `~2.2e-5`       |
| `[-3, 3]`    | central degree-21 polynomial | `~6.5e-7`       |
| `[3, 300]`   | saturation `+1`              | `~2.2e-5`       |

Each box's `eps` is a rigorous sup-norm bound `sup_{x in box} |approx(x) -
erf(x)| <= eps`, certified by the WI-14 Arb oracle (whole-box ball arithmetic,
not sampling). The runtime evaluates `approx(x) +- eps` in pure `f64`; it never
calls Arb. A release binary that consumes the envelope gains **no** FLINT/Arb
link (the `arb` feature is off by default and absent from the `default` and
`smt` dependency trees).

## Trust model: float proposer, Arb certifier

The Python generator is only a **proposer** — it fits the central polynomial in
floating point. The Arb certifier is the **trust anchor** — it stamps each
`eps` by rigorous ball arithmetic. A poor proposed polynomial cannot make the
envelope unsound; it only yields a larger (still sound) `eps`. This mirrors the
Clarabel SoS engine's float-proposer / rational-exact-verifier split.

## Regeneration pipeline

```sh
# 1. Propose coefficients (float; emits a draft with eps placeholders).
.venv/bin/python scripts/generate_erf_envelope.py > /tmp/erf_envelope_draft.json

# 2. Certify eps with Arb and write the committed data.
cargo run -p chelis-prove --features arb --bin certify_erf_envelope \
  -- /tmp/erf_envelope_draft.json crates/chelis-prove/data/erf_envelope.json

# 3. Inspect the diff and commit the regenerated data.
git diff crates/chelis-prove/data/erf_envelope.json
```

Step 2 takes roughly ten seconds (the central box is certified with 524288
sub-boxes at 128-bit precision; the count is recorded in the data's
`provenance.certify_subdivisions`).

## Environment

### Proposer (step 1) — Python

- the uv-managed Python at `.venv/bin/python` (`>= 3.11`)
- `numpy` (least-squares polynomial solve)
- `mpmath` (arbitrary-precision `erf` truth for the fit targets)

```sh
uv venv --python 3.11    # if not already present
uv pip install numpy mpmath
```

### Certifier (step 2) and CI re-validation — Rust `arb` feature

The `arb` feature links `arb-sys`, which vendors FLINT + Arb C source and
compiles it (and pulls `gmp-mpfr-sys`, which compiles GMP/MPFR). It does **not**
use a system FLINT.

| Tool         | Purpose                          | Install (Debian/Ubuntu) |
|--------------|----------------------------------|-------------------------|
| C compiler   | compile vendored FLINT/Arb/GMP/MPFR | `apt install gcc`    |
| make         | vendored build driver            | `apt install make`      |
| m4           | GMP build prerequisite           | `apt install m4`        |

Two build hazards on a PIE-default toolchain (e.g. Fedora), both fixed durably
in `.cargo/config.toml` so a clean `--features arb` build needs no per-developer
step:

1. **`-fPIC`** — the vendored C must be position-independent or the link fails
   with `relocation R_X86_64_32 ... recompile with -fPIC`. `CFLAGS`/`CXXFLAGS`
   are set to `-fPIC`.
2. **Isolated build caches** — `gmp-mpfr-sys` / `flint-sys` / `arb-sys` cache
   their compiled archives in a per-user dir keyed by version + compiler but
   **not** by CFLAGS, so a stale non-PIC archive would be silently reused. The
   `GMP_MPFR_SYS_CACHE` / `FLINT_SYS_CACHE` / `ARB_SYS_CACHE` vars point at a
   workspace-local `target/arb-cache` dir.

A cold `--features arb` build compiles the C stack once (~1-2 minutes); it is
cached thereafter.

### License

`arb-sys` / `flint-sys` are MIT/Apache, but the vendored FLINT/Arb C and
`gmp-mpfr-sys` are LGPL. The obligation attaches only to a binary built with the
`arb` feature, which is offline/CI tooling that never ships in a deployed build.
Recorded and deferred per master-plan WI-19.

## CI re-validation (the soundness gate)

The committed `eps` is **not** trusted on faith. The `arb`-gated test
`committed_envelope_eps_still_bounds_the_truth` (in `arb_oracle.rs`) re-runs the
Arb certification for every committed box and asserts the committed `eps` still
bounds the recertified sup-norm error. Its negative partner
`a_too_small_eps_fails_revalidation` confirms a shrunk `eps` would be caught, so
the harness is a real gate rather than a rubber stamp. Run:

```sh
cargo test -p chelis-prove --features arb committed_envelope
```

This runs in the `arb` CI lane only; the `default` and `smt` lanes neither link
Arb nor re-validate (they consume the committed data as-is).
